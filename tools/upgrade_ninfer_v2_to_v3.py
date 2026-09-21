"""Reframe an existing NInfer v1/v2 artifact as a v3 entry file.

This is the fork-local upgrade path: it validates the legacy directory, translates the
object spellings to the v3 registry, preserves the fork's chat template / config metadata
as-is (the upstream upgrade script replaces them with upstream-maintained ones), and
copies the payload bytes unchanged and pads the v3 payload to a 4096-byte sector boundary.

Usage:
    python tools/upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact.container import (  # noqa: E402
    _parse_legacy_directory,
    _require_integer,
    _V3_ENCODINGS,
    _V3_FORMATS,
    _V3_LAYOUTS,
    _V1_MAGIC,
    _V2_MAGIC,
    HEADER,
    HEADER_BYTES,
    MAGIC,
    PAYLOAD_ALIGNMENT,
    PREFIX,
    PREFIX_BYTES,
)
from artifact.layouts import align_up, encoded_size, get_layout  # noqa: E402

_CHUNK = 8 * 1024 * 1024


def _validate_and_translate(objects, payload_bytes):
    """Translate legacy objects to v3 spellings and validate ranges/geometry."""
    names = set()
    cursor = 0
    translated = []
    for obj in objects:
        name = obj.name
        if name in names:
            raise ValueError(f"duplicate object name: {name}")
        names.add(name)
        kind = obj.kind
        offset = _require_integer(obj.offset, "object offset")
        nbytes = _require_integer(obj.bytes, "object bytes", positive=True)
        if offset < cursor:
            raise ValueError(f"object {name} overlaps or is out of order")
        if kind == "tensor":
            v3_format = _V3_FORMATS.get(obj.format)
            v3_layout = _V3_LAYOUTS.get(obj.layout)
            if v3_format is None:
                raise ValueError(f"tensor {name} has unknown v2 format {obj.format!r}")
            if v3_layout is None:
                raise ValueError(f"tensor {name} has unknown v2 layout {obj.layout!r}")
            alignment = get_layout(obj.layout).alignment
            expected = encoded_size(obj.layout, obj.format, obj.shape)
            if nbytes != expected:
                raise ValueError(
                    f"tensor {name} byte count {nbytes} does not match its layout ({expected})"
                )
            value = v3_format
        elif kind == "resource":
            v3_encoding = _V3_ENCODINGS.get(obj.encoding)
            if v3_encoding is None:
                raise ValueError(f"resource {name} has unknown v2 encoding {obj.encoding!r}")
            alignment = 1
            value = v3_encoding
        else:
            raise ValueError(f"object {name} has unknown kind {kind!r}")
        if offset % alignment != 0:
            raise ValueError(f"object {name} is not {alignment}-byte aligned")
        end = offset + nbytes
        if end > payload_bytes:
            raise ValueError(f"object {name} extends beyond the payload")
        cursor = end
        translated.append(
            {
                "id": name,
                "kind": kind,
                "shape": list(obj.shape) if kind == "tensor" else None,
                "value": value,
                "layout": v3_layout if kind == "tensor" else None,
                "offset": offset,
                "bytes": nbytes,
            }
        )
    return translated


def upgrade(input_path: Path, output_path: Path) -> Path:
    file_bytes = input_path.stat().st_size
    with input_path.open("rb") as handle:
        head = handle.read(PREFIX_BYTES)
        if len(head) < PREFIX_BYTES:
            raise ValueError("artifact is shorter than the v2 prefix")
        magic, json_bytes = PREFIX.unpack(head)
        if magic == _V1_MAGIC:
            label = "v1"
        elif magic == _V2_MAGIC:
            label = "v2"
        else:
            raise ValueError("artifact magic is not NInfer v1 or v2")
        metadata_end = PREFIX_BYTES + json_bytes
        payload_offset = align_up(metadata_end, PAYLOAD_ALIGNMENT)
        if file_bytes < payload_offset:
            raise ValueError("declared JSON or payload start extends beyond the file")
        directory = handle.read(json_bytes)
        if len(directory) != json_bytes:
            raise ValueError("artifact JSON is truncated")
        payload_bytes = file_bytes - payload_offset
        identity, objects = _parse_legacy_directory(directory, label == "v1")
        if payload_bytes > 0 and not objects:
            raise ValueError("payload is present but the directory has no objects")

        translated = _validate_and_translate(objects, payload_bytes)
        v3_objects = []
        for item in translated:
            if item["kind"] == "tensor":
                v3_objects.append(
                    {
                        "id": item["id"],
                        "kind": "tensor",
                        "shape": item["shape"],
                        "format": item["value"],
                        "layout": item["layout"],
                        "offset": item["offset"],
                        "bytes": item["bytes"],
                    }
                )
            else:
                v3_objects.append(
                    {
                        "id": item["id"],
                        "kind": "resource",
                        "encoding": item["value"],
                        "offset": item["offset"],
                        "bytes": item["bytes"],
                    }
                )
        v3_payload_bytes = align_up(payload_bytes, PAYLOAD_ALIGNMENT)
        v3_directory = {
            "components": {"text": {"config": {}, "target": "text"}},
            "objects": v3_objects,
            "bindings": {},
            "uses": [],
            "metadata": {
                "name": identity.model_id,
                "weights_id": identity.weights_id,
            },
            "provenance": {
                "source": f"{label}-artifact:{input_path.name}",
                "upgraded": "tools/upgrade_ninfer_v2_to_v3.py",
            },
            "files": [{"path": None, "payload_bytes": v3_payload_bytes}],
        }
        encoded = json.dumps(v3_directory, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        v3_payload_offset = align_up(HEADER_BYTES + len(encoded), PAYLOAD_ALIGNMENT)

        payload_handle = input_path.open("rb")
        payload_handle.seek(payload_offset)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(HEADER.pack(MAGIC, len(encoded), uuid4().bytes))
                out.write(encoded)
                out.write(b"\x00" * (v3_payload_offset - HEADER_BYTES - len(encoded)))
                copied = 0
                remaining = payload_bytes
                while remaining > 0:
                    chunk = payload_handle.read(min(_CHUNK, remaining))
                    if not chunk:
                        raise ValueError("artifact payload is truncated")
                    out.write(chunk)
                    copied += len(chunk)
                    remaining -= len(chunk)
                out.write(b"\x00" * (v3_payload_bytes - payload_bytes))
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp_path, output_path)
        except BaseException:
            payload_handle.close()
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        payload_handle.close()
        print(f"upgraded {label} artifact {input_path.name} -> {output_path.name}")
        print(f"  model: {identity.model_id} ({identity.weights_id})")
        print(f"  objects: {len(v3_objects)}  payload: {payload_bytes} bytes")
        return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        upgrade(args.input, args.output)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())