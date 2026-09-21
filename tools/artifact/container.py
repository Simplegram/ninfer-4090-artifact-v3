"""Minimal reader and streaming writer for the NInfer v3 object directory.

The on-disk format is the NInfer v3 entry framing (32-byte header: ``NINFER\x00\x03``,
little-endian JSON byte count, 16-byte artifact id).  The Python reference API keeps the
fork's field names (``name``, v2-style spellings) and translates them to the v3 on-disk
spellings at the JSON boundary.  Legacy v1/v2 entries remain readable so pre-existing
artifacts can be upgraded in place.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import mmap
import struct
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeAlias
from uuid import uuid4

from .layouts import align_up, encoded_size, get_layout

MAGIC = b"NINFER\x00\x03"
PART_MAGIC = b"NINPRT\x00\x03"
_V1_MAGIC = b"NINFER\x00\x01"
_V2_MAGIC = b"NINFER\x00\x02"
HEADER = struct.Struct("<8sQ16s")
PREFIX = struct.Struct("<8sQ")
HEADER_BYTES = HEADER.size
PREFIX_BYTES = PREFIX.size
ARTIFACT_ID_BYTES = 16
PAYLOAD_ALIGNMENT = 4096
RAW_BYTES_V1 = "raw-bytes-v1"

_V3_FORMATS = {
    "BF16": "bf16",
    "FP32": "fp32",
    "I32": "int32",
    "Q4G64_F16S": "q4_g64_fp16",
    "Q5G64_F16S": "q5_g64_fp16",
    "Q6G64_F16S": "q6_g64_fp16",
    "W8G32_F16S": "q8_g32_fp16",
}
_V3_LAYOUTS = {
    "contiguous-le-v1": "contiguous_le_v1",
    "row-split-k128-v1": "row_split_k128_v1",
}
_V3_ENCODINGS = {
    RAW_BYTES_V1: "raw_bytes_v1",
}
_V2_FORMATS = {value: key for key, value in _V3_FORMATS.items()}
_V2_LAYOUTS = {value: key for key, value in _V3_LAYOUTS.items()}
_V2_ENCODINGS = {value: key for key, value in _V3_ENCODINGS.items()}

_DIRECTORY_MEMBERS = frozenset(
    {"components", "objects", "bindings", "uses", "files"}
)
_DIRECTORY_OPTIONAL = frozenset({"metadata", "provenance"})
_TENSOR_MEMBERS = frozenset(
    {"id", "kind", "shape", "format", "layout", "offset", "bytes"}
)
_RESOURCE_MEMBERS = frozenset(
    {"id", "kind", "encoding", "offset", "bytes"}
)
_FILE_MEMBERS = frozenset({"path", "payload_bytes"})


class ArtifactError(ValueError):
    """The file does not satisfy the NInfer v3 directory contract."""


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    model_id: str
    weights_id: str


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    name: str
    encoding: str
    bytes: int


ObjectSpec: TypeAlias = TensorSpec | ResourceSpec


@dataclass(frozen=True, slots=True)
class TensorObject:
    name: str
    kind: str
    shape: tuple[int, ...]
    format: str
    layout: str
    offset: int
    bytes: int

    @property
    def elements(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@dataclass(frozen=True, slots=True)
class ResourceObject:
    name: str
    kind: str
    encoding: str
    offset: int
    bytes: int


ArtifactObject: TypeAlias = TensorObject | ResourceObject
PayloadChunk: TypeAlias = bytes | bytearray | memoryview
Payload: TypeAlias = PayloadChunk | Iterable[PayloadChunk]


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ArtifactError(f"{field} must be a nonempty string without NUL")
    return value


def _require_integer(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ArtifactError(f"{field} must be an integer")
    if value < 0 or (positive and value == 0):
        raise ArtifactError(f"{field} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _require_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ArtifactError("tensor shape must be an array")
    return tuple(_require_integer(dim, "shape dimension", positive=True) for dim in value)


def _v3_tensor_json(obj: TensorObject) -> dict[str, object]:
    return {
        "id": obj.name,
        "kind": "tensor",
        "shape": list(obj.shape),
        "format": _V3_FORMATS[obj.format],
        "layout": _V3_LAYOUTS[obj.layout],
        "offset": obj.offset,
        "bytes": obj.bytes,
    }


def _v3_resource_json(obj: ResourceObject) -> dict[str, object]:
    return {
        "id": obj.name,
        "kind": "resource",
        "encoding": _V3_ENCODINGS[obj.encoding],
        "offset": obj.offset,
        "bytes": obj.bytes,
    }


def _v3_object_json(obj: ArtifactObject) -> dict[str, object]:
    if isinstance(obj, TensorObject):
        return _v3_tensor_json(obj)
    return _v3_resource_json(obj)


def _parse_v3_tensor(value: object) -> TensorObject:
    if not isinstance(value, dict):
        raise ArtifactError("tensor entry must be an object")
    missing = _TENSOR_MEMBERS - value.keys()
    extra = set(value.keys()) - _TENSOR_MEMBERS
    if missing or extra:
        raise ArtifactError(f"tensor entry has missing or extra members: {sorted(missing | extra)}")
    if value["kind"] != "tensor":
        raise ArtifactError("tensor entry kind must be 'tensor'")
    return TensorObject(
        name=_require_string(value["id"], "tensor id"),
        kind="tensor",
        shape=_require_shape(value["shape"]),
        format=_V2_FORMATS.get(value["format"], str(value["format"])),
        layout=_V2_LAYOUTS.get(value["layout"], str(value["layout"])),
        offset=_require_integer(value["offset"], "tensor offset"),
        bytes=_require_integer(value["bytes"], "tensor bytes", positive=True),
    )


def _parse_v3_resource(value: object) -> ResourceObject:
    if not isinstance(value, dict):
        raise ArtifactError("resource entry must be an object")
    missing = _RESOURCE_MEMBERS - value.keys()
    extra = set(value.keys()) - _RESOURCE_MEMBERS
    if missing or extra:
        raise ArtifactError(
            f"resource entry has missing or extra members: {sorted(missing | extra)}"
        )
    if value["kind"] != "resource":
        raise ArtifactError("resource entry kind must be 'resource'")
    return ResourceObject(
        name=_require_string(value["id"], "resource id"),
        kind="resource",
        encoding=_V2_ENCODINGS.get(value["encoding"], str(value["encoding"])),
        offset=_require_integer(value["offset"], "resource offset"),
        bytes=_require_integer(value["bytes"], "resource bytes", positive=True),
    )


def _parse_legacy_object(value: object) -> ArtifactObject:
    if not isinstance(value, dict):
        raise ArtifactError("each object entry must be a JSON object")
    kind = value.get("kind")
    if kind == "tensor":
        members = {"name", "kind", "shape", "format", "layout", "offset", "bytes"}
        if set(value.keys()) != members:
            raise ArtifactError("legacy tensor entry has missing or extra members")
        return TensorObject(
            name=_require_string(value["name"], "tensor name"),
            kind="tensor",
            shape=_require_shape(value["shape"]),
            format=str(value["format"]),
            layout=str(value["layout"]),
            offset=_require_integer(value["offset"], "tensor offset"),
            bytes=_require_integer(value["bytes"], "tensor bytes", positive=True),
        )
    if kind == "resource":
        members = {"name", "kind", "encoding", "offset", "bytes"}
        if set(value.keys()) != members:
            raise ArtifactError("legacy resource entry has missing or extra members")
        return ResourceObject(
            name=_require_string(value["name"], "resource name"),
            kind="resource",
            encoding=str(value["encoding"]),
            offset=_require_integer(value["offset"], "resource offset"),
            bytes=_require_integer(value["bytes"], "resource bytes", positive=True),
        )
    raise ArtifactError("object kind must be 'tensor' or 'resource'")


def object_alignment(obj: ArtifactObject) -> int:
    if isinstance(obj, TensorObject):
        return get_layout(obj.layout).alignment
    if obj.encoding != RAW_BYTES_V1:
        raise ArtifactError(f"unknown resource encoding: {obj.encoding}")
    return 1


def plan_objects(specs: Sequence[ObjectSpec]) -> tuple[ArtifactObject, ...]:
    """Assign aligned payload-relative offsets to an ordered complete inventory."""
    result: list[ArtifactObject] = []
    cursor = 0
    for spec in specs:
        if isinstance(spec, TensorSpec):
            if spec.format not in _V3_FORMATS:
                raise ArtifactError(f"unknown numeric format: {spec.format!r}")
            alignment = get_layout(spec.layout).alignment
            bytes_ = encoded_size(spec.layout, spec.format, spec.shape)
            cursor = align_up(cursor, alignment)
            result.append(
                TensorObject(
                    name=spec.name,
                    kind="tensor",
                    shape=tuple(spec.shape),
                    format=spec.format,
                    layout=spec.layout,
                    offset=cursor,
                    bytes=bytes_,
                )
            )
        elif isinstance(spec, ResourceSpec):
            alignment = 1
            bytes_ = _require_integer(spec.bytes, "resource bytes", positive=True)
            cursor = align_up(cursor, alignment)
            result.append(
                ResourceObject(
                    name=spec.name,
                    kind="resource",
                    encoding=spec.encoding,
                    offset=cursor,
                    bytes=bytes_,
                )
            )
        else:
            raise ArtifactError(f"unknown object spec: {spec!r}")
        cursor += bytes_
    return tuple(result)


def _v3_directory(
    identity: ArtifactIdentity, objects: Sequence[ArtifactObject],
    text_config: dict[str, object] | None,
    provenance: dict[str, object] | None,
) -> dict[str, object]:
    payload_bytes = objects[-1].offset + objects[-1].bytes if objects else 0
    return {
        "components": {
            "text": {
                "config": text_config or {},
                "target": "text",
            },
        },
        "objects": [_v3_object_json(obj) for obj in objects],
        "bindings": {},
        "uses": [],
        "metadata": {
            "name": identity.model_id,
            "weights_id": identity.weights_id,
        },
        "provenance": provenance or {},
        "files": [{"path": None, "payload_bytes": payload_bytes}],
    }


def encode_directory(
    identity: ArtifactIdentity,
    objects: Sequence[ArtifactObject],
    text_config: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
) -> bytes:
    if not isinstance(identity, ArtifactIdentity):
        raise ArtifactError("identity must be an ArtifactIdentity")
    if not identity.model_id or not identity.weights_id:
        raise ArtifactError("artifact identity must name the model and weights")
    directory = _v3_directory(identity, objects, text_config, provenance)
    return json.dumps(directory, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _parse_v3_directory(data: bytes) -> tuple[ArtifactIdentity, tuple[ArtifactObject, ...]]:
    try:
        root = json.loads(data)
    except json.JSONDecodeError as error:
        raise ArtifactError(f"artifact JSON is invalid: {error}") from error
    if not isinstance(root, dict):
        raise ArtifactError("artifact directory must be a JSON object")
    missing = _DIRECTORY_MEMBERS - root.keys()
    extra = set(root.keys()) - _DIRECTORY_MEMBERS - _DIRECTORY_OPTIONAL
    if missing or extra:
        raise ArtifactError(f"artifact directory has missing or extra members: {sorted(missing | extra)}")

    components = root["components"]
    if not isinstance(components, dict) or "text" not in components:
        raise ArtifactError("artifact components must contain text")
    text = components["text"]
    if not isinstance(text, dict) or not isinstance(text.get("config"), dict):
        raise ArtifactError("text component must carry a config object")

    files = root["files"]
    if not isinstance(files, list) or not files:
        raise ArtifactError("artifact files must be a nonempty array")
    if files[0].get("path") is not None:
        raise ArtifactError("files[0].path must be null for the entry file")
    for file in files:
        if not isinstance(file, dict) or set(file.keys()) != _FILE_MEMBERS:
            raise ArtifactError("file record has missing or extra members")
        _require_integer(file["payload_bytes"], "file payload bytes", positive=True)
    if len(files) > 1:
        raise ArtifactError("continuation files are not supported by the Python reference reader")

    raw_objects = root["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ArtifactError("objects must be a nonempty array")
    objects: list[ArtifactObject] = []
    names: set[str] = set()
    previous_end = 0
    for value in raw_objects:
        if not isinstance(value, dict):
            raise ArtifactError("each object entry must be a JSON object")
        kind = value.get("kind")
        if kind == "tensor":
            obj = _parse_v3_tensor(value)
        elif kind == "resource":
            obj = _parse_v3_resource(value)
        else:
            raise ArtifactError(f"unknown object kind: {kind!r}")
        if obj.name in names:
            raise ArtifactError(f"duplicate object id: {obj.name}")
        names.add(obj.name)
        objects.append(obj)

    metadata = root.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ArtifactError("metadata must be an object")
    model_id = metadata.get("name", "")
    if not isinstance(model_id, str) or not model_id:
        raise ArtifactError("artifact identity is missing a model name")
    weights_id = metadata.get("weights_id", "")
    if not isinstance(weights_id, str) or not weights_id:
        raise ArtifactError("artifact identity is missing a weights id")

    return ArtifactIdentity(model_id, weights_id), tuple(objects)


def _parse_legacy_directory(
    data: bytes, v1: bool
) -> tuple[ArtifactIdentity, tuple[ArtifactObject, ...]]:
    try:
        root = json.loads(data)
    except json.JSONDecodeError as error:
        raise ArtifactError(f"artifact JSON is invalid: {error}") from error
    if not isinstance(root, dict):
        raise ArtifactError("artifact directory must be a JSON object")
    if v1:
        if set(root.keys()) != {"model_id", "objects"}:
            raise ArtifactError("legacy v1 directory has missing or extra members")
        model_id = _require_string(root["model_id"], "model_id")
        if model_id not in ("qwen3.6-27b", "qwen3.6-35b-a3b"):
            raise ArtifactError("legacy artifact model_id is not a registered groupwise target")
        identity = ArtifactIdentity(model_id, "groupwise-int")
    else:
        if set(root.keys()) != {"identity", "objects"}:
            raise ArtifactError("artifact directory has missing or extra members")
        raw_identity = root["identity"]
        if not isinstance(raw_identity, dict) or set(raw_identity.keys()) != {
            "model_id",
            "weights_id",
        }:
            raise ArtifactError("artifact identity has missing or extra members")
        identity = ArtifactIdentity(
            _require_string(raw_identity["model_id"], "model_id"),
            _require_string(raw_identity["weights_id"], "weights_id"),
        )
    raw_objects = root["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ArtifactError("objects must be a nonempty array")
    return identity, tuple(_parse_legacy_object(value) for value in raw_objects)


def parse_directory(
    data: bytes, v1: bool = False, v3: bool = False
) -> tuple[ArtifactIdentity, tuple[ArtifactObject, ...]]:
    if v3:
        return _parse_v3_directory(data)
    return _parse_legacy_directory(data, v1)


def _validate_ranges(
    objects: Sequence[ArtifactObject], payload_bytes: int
) -> dict[str, ArtifactObject]:
    cursor = 0
    index: dict[str, ArtifactObject] = {}
    for obj in objects:
        if obj.name in index:
            raise ArtifactError(f"duplicate object name: {obj.name}")
        if obj.offset < cursor:
            raise ArtifactError(f"object {obj.name} overlaps or is out of order")
        if obj.offset % object_alignment(obj) != 0:
            raise ArtifactError(f"object {obj.name} is not {object_alignment(obj)}-byte aligned")
        if isinstance(obj, TensorObject):
            expected = encoded_size(obj.layout, obj.format, obj.shape)
            if obj.bytes != expected:
                raise ArtifactError(f"tensor {obj.name} byte count does not match its layout")
        end = obj.offset + obj.bytes
        if end > payload_bytes:
            raise ArtifactError(f"object {obj.name} extends beyond the payload")
        index[obj.name] = obj
        cursor = end
    return index


class Artifact:
    """Mmap-backed, structurally validated `.ninfer` artifact (v3 entries; v1/v2 legacy)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        self.artifact_id: bytes | None = None
        self.v3 = False
        try:
            self._file.seek(0, 2)
            self.file_bytes = self._file.tell()
            self._file.seek(0)
            head = self._file.read(HEADER_BYTES)
            if head[:8] == MAGIC:
                self._init_v3(head[:HEADER_BYTES])
            else:
                self._init_legacy()
        except BaseException:
            if self._mapping is not None:
                self._mapping.close()
            self._file.close()
            raise

    def _init_v3(self, head: bytes) -> None:
        if self.file_bytes < HEADER_BYTES:
            raise ArtifactError("artifact is shorter than the v3 header")
        magic, json_bytes, artifact_id = HEADER.unpack(head)
        self.artifact_id = artifact_id
        self.v3 = True
        metadata_end = HEADER_BYTES + json_bytes
        self.payload_offset = align_up(metadata_end, PAYLOAD_ALIGNMENT)
        if metadata_end > self.file_bytes or self.payload_offset > self.file_bytes:
            raise ArtifactError("declared JSON or payload start extends beyond the file")
        directory = self._file.read(json_bytes)
        if len(directory) != json_bytes:
            raise ArtifactError("artifact JSON is truncated")
        self.identity, self.objects = _parse_v3_directory(directory)
        payload_bytes = self.file_bytes - self.payload_offset
        self._index = _validate_ranges(self.objects, payload_bytes)
        files = json.loads(directory, object_pairs_hook=_unique_pairs)["files"]
        if files[0]["payload_bytes"] != payload_bytes:
            raise ArtifactError("entry length differs from the artifact directory")
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def _init_legacy(self) -> None:
        if self.file_bytes < PREFIX_BYTES:
            raise ArtifactError("artifact is shorter than the v2 prefix")
        self._file.seek(0)
        head = self._file.read(PREFIX_BYTES)
        magic, json_bytes = PREFIX.unpack(head)
        if magic == _V1_MAGIC:
            v1 = True
        elif magic == _V2_MAGIC:
            v1 = False
        else:
            raise ArtifactError("artifact magic is not NInfer v1, v2, or v3")
        metadata_end = PREFIX_BYTES + json_bytes
        self.payload_offset = align_up(metadata_end, PAYLOAD_ALIGNMENT)
        if metadata_end > self.file_bytes or self.payload_offset > self.file_bytes:
            raise ArtifactError("declared JSON or payload start extends beyond the file")
        directory = self._file.read(json_bytes)
        if len(directory) != json_bytes:
            raise ArtifactError("artifact JSON is truncated")
        self.identity, self.objects = _parse_legacy_directory(directory, v1)
        payload_bytes = self.file_bytes - self.payload_offset
        self._index = _validate_ranges(self.objects, payload_bytes)
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    @classmethod
    def open(cls, path: str | Path) -> "Artifact":
        return cls(path)

    def find(self, name: str) -> ArtifactObject:
        return self._index[name]

    def payload(self, obj: ArtifactObject | str) -> memoryview:
        if isinstance(obj, str):
            obj = self.find(obj)
        if self._mapping is None:
            raise RuntimeError("artifact is closed")
        begin = self.payload_offset + obj.offset
        return memoryview(self._mapping)[begin : begin + obj.bytes]

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "Artifact":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _payload_chunks(payload: Payload) -> Iterator[memoryview]:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        yield memoryview(payload).cast("B")
        return
    for chunk in payload:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("payload chunks must support the buffer protocol")
        yield memoryview(chunk).cast("B")


class ArtifactWriter:
    """Write one preplanned v3 artifact payload at a time in directory order."""

    def __init__(
        self,
        path: str | Path,
        identity: ArtifactIdentity,
        specs: Sequence[ObjectSpec],
        *,
        text_config: dict[str, object] | None = None,
        provenance: dict[str, object] | None = None,
    ):
        self.path = Path(path)
        self.identity = _require_identity(identity)
        self.objects = plan_objects(specs)
        self.artifact_id = uuid4().bytes
        directory = encode_directory(
            self.identity, self.objects, text_config=text_config, provenance=provenance
        )
        self.payload_offset = align_up(HEADER_BYTES + len(directory), PAYLOAD_ALIGNMENT)
        if self.path.exists():
            raise FileExistsError(self.path)
        self._file = self.path.open("wb")
        self._file.write(HEADER.pack(MAGIC, len(directory), self.artifact_id))
        self._file.write(directory)
        self._file.write(b"\x00" * (self.payload_offset - HEADER_BYTES - len(directory)))
        self._next = 0
        self._cursor = 0
        self._finished = False

    def write(self, name: str, payload: Payload) -> None:
        if self._finished:
            raise RuntimeError("artifact writer is already finished")
        if self._next >= len(self.objects):
            raise ArtifactError("artifact already has every planned payload")
        obj = self.objects[self._next]
        if name != obj.name:
            raise ArtifactError(f"expected payload {obj.name}, got {name}")
        if obj.offset > self._cursor:
            self._file.write(b"\x00" * (obj.offset - self._cursor))
            self._cursor = obj.offset
        written = 0
        for chunk in _payload_chunks(payload):
            if written + len(chunk) > obj.bytes:
                raise ArtifactError(f"payload {name} exceeds its planned byte length")
            self._file.write(chunk)
            written += len(chunk)
        if written != obj.bytes:
            raise ArtifactError(f"payload {name} has {written} bytes; expected {obj.bytes}")
        self._cursor = obj.offset + obj.bytes
        self._next += 1

    def finish(self) -> None:
        if self._finished:
            return
        if self._next != len(self.objects):
            missing = self.objects[self._next].name
            raise ArtifactError(f"artifact is missing payload {missing}")
        self._file.truncate(self.payload_offset + self._cursor)
        self._file.flush()
        self._file.close()
        self._finished = True

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "ArtifactWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            try:
                self.finish()
            finally:
                if not self._finished:
                    self.close()
        else:
            self.close()


def _require_identity(identity: ArtifactIdentity) -> ArtifactIdentity:
    if not isinstance(identity, ArtifactIdentity):
        raise ArtifactError("identity must be an ArtifactIdentity")
    if not identity.model_id or not identity.weights_id:
        raise ArtifactError("artifact identity must name the model and weights")
    return identity


def write_artifact(
    path: str | Path,
    identity: ArtifactIdentity,
    entries: Sequence[tuple[ObjectSpec, Payload]],
    *,
    text_config: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
) -> tuple[ArtifactObject, ...]:
    specs = [spec for spec, _ in entries]
    with ArtifactWriter(
        path, identity, specs, text_config=text_config, provenance=provenance
    ) as writer:
        for spec, payload in entries:
            writer.write(spec.name, payload)
    return writer.objects


__all__ = [
    "ARTIFACT_ID_BYTES",
    "Artifact",
    "ArtifactError",
    "ArtifactIdentity",
    "ArtifactObject",
    "ArtifactWriter",
    "HEADER",
    "HEADER_BYTES",
    "MAGIC",
    "PART_MAGIC",
    "PAYLOAD_ALIGNMENT",
    "PREFIX",
    "PREFIX_BYTES",
    "RAW_BYTES_V1",
    "ResourceObject",
    "ResourceSpec",
    "TensorObject",
    "TensorSpec",
    "encode_directory",
    "object_alignment",
    "parse_directory",
    "plan_objects",
    "write_artifact",
]