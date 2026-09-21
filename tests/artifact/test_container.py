from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from tools.artifact.container import (
    ARTIFACT_ID_BYTES,
    HEADER,
    HEADER_BYTES,
    MAGIC,
    PAYLOAD_ALIGNMENT,
    PREFIX,
    Artifact,
    ArtifactError,
    ArtifactIdentity,
    ResourceSpec,
    TensorSpec,
    write_artifact,
)
from tools.artifact.inspect import artifact_summary
from tools.artifact.layouts import align_up, encoded_size

REPO_ROOT = Path(__file__).resolve().parents[2]


def _small_specs():
    return [
        ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 2),
        TensorSpec("direct/bf16", (2,), "BF16", "contiguous-le-v1"),
        TensorSpec("direct/fp32", (1,), "FP32", "contiguous-le-v1"),
        TensorSpec("direct/i32", (1,), "I32", "contiguous-le-v1"),
        TensorSpec("quant/q4", (1, 64), "Q4G64_F16S", "row-split-k128-v1"),
        TensorSpec("quant/q5", (1, 64), "Q5G64_F16S", "row-split-k128-v1"),
        TensorSpec("quant/q6", (1, 64), "Q6G64_F16S", "row-split-k128-v1"),
        TensorSpec("quant/w8", (1, 32), "W8G32_F16S", "row-split-k128-v1"),
    ]


def _payload(spec):
    if isinstance(spec, ResourceSpec):
        return b"{}"
    size = encoded_size(spec.layout, spec.format, spec.shape)
    return bytes(size)


def _v3_directory(objects, model_id="test-model", weights_id="test-weights", **extra):
    payload_bytes = objects[-1]["offset"] + objects[-1]["bytes"] if objects else 0
    directory = {
        "components": {"text": {"config": {}, "target": model_id}},
        "objects": objects,
        "bindings": {},
        "uses": [],
        "metadata": {"name": model_id, "weights_id": weights_id},
        "provenance": {},
        "files": [{"path": None, "payload_bytes": payload_bytes}],
    }
    directory.update(extra)
    return directory


def _write_raw_v3(path, directory, payload: bytes = b"", artifact_id=b"\x00" * ARTIFACT_ID_BYTES):
    encoded = json.dumps(directory, separators=(",", ":")).encode("utf-8")
    payload_offset = align_up(HEADER_BYTES + len(encoded), PAYLOAD_ALIGNMENT)
    path.write_bytes(
        HEADER.pack(MAGIC, len(encoded), artifact_id)
        + encoded
        + bytes(payload_offset - HEADER_BYTES - len(encoded))
        + payload
    )


def _write_raw_v2(path, metadata, payload: bytes = b"", magic=b"NINFER\x00\x02"):
    encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    payload_offset = align_up(PREFIX.size + len(encoded), PAYLOAD_ALIGNMENT)
    path.write_bytes(
        PREFIX.pack(magic, len(encoded))
        + encoded
        + bytes(payload_offset - PREFIX.size - len(encoded))
        + payload
    )


def test_v3_round_trip_covers_every_registered_storage(tmp_path):
    path = tmp_path / "small.ninfer"
    specs = _small_specs()
    entries = [(spec, _payload(spec)) for spec in specs]
    identity = ArtifactIdentity("test-model", "test-weights")
    planned = write_artifact(path, identity, entries)

    head = path.read_bytes()[:HEADER_BYTES]
    magic, json_bytes, artifact_id = HEADER.unpack(head)
    assert magic == MAGIC
    assert len(artifact_id) == ARTIFACT_ID_BYTES
    assert head[8:16] == struct.pack("<Q", json_bytes)

    with Artifact.open(path) as artifact:
        assert artifact.v3 is True
        assert artifact.identity == identity
        assert artifact.payload_offset == align_up(HEADER_BYTES + json_bytes, PAYLOAD_ALIGNMENT)
        assert artifact.objects == planned
        for spec, expected in entries:
            assert bytes(artifact.payload(spec.name)) == expected
        summary = artifact_summary(artifact)
        assert summary["model_id"] == "test-model"
        assert summary["weights_id"] == "test-weights"
        assert summary["objects"] == 8
        assert summary["formats"] == {
            "BF16": 1,
            "FP32": 1,
            "I32": 1,
            "Q4G64_F16S": 1,
            "Q5G64_F16S": 1,
            "Q6G64_F16S": 1,
            "W8G32_F16S": 1,
        }
    # The v3 directory on disk carries the closed-registry spellings.
    directory = json.loads(path.read_bytes()[HEADER_BYTES : HEADER_BYTES + json_bytes])
    tensor = directory["objects"][1]
    assert tensor == {
        "id": "direct/bf16",
        "kind": "tensor",
        "shape": [2],
        "format": "bf16",
        "layout": "contiguous_le_v1",
        "offset": planned[1].offset,
        "bytes": planned[1].bytes,
    }
    declared_payload = path.stat().st_size - align_up(
        HEADER_BYTES + json_bytes, PAYLOAD_ALIGNMENT
    )
    assert directory["files"] == [{"path": None, "payload_bytes": declared_payload}]


def test_v3_reader_rejects_invalid_framing_schema_and_geometry(tmp_path):
    path = tmp_path / "invalid.ninfer"
    objects = [
        {
            "id": "a",
            "kind": "tensor",
            "shape": [1],
            "format": "int32",
            "layout": "contiguous_le_v1",
            "offset": 0,
            "bytes": 4,
        },
        {
            "id": "b",
            "kind": "resource",
            "encoding": "raw_bytes_v1",
            "offset": 2,
            "bytes": 2,
        },
    ]

    # Declared JSON extends beyond the file.
    path.write_bytes(HEADER.pack(MAGIC, 100, b"\x00" * ARTIFACT_ID_BYTES) + b"{}")
    with pytest.raises(ArtifactError, match="beyond the file"):
        Artifact.open(path)

    # Overlapping objects are a geometry violation.
    _write_raw_v3(path, _v3_directory(objects), b"\x00" * 4)
    with pytest.raises(ArtifactError, match="overlaps"):
        Artifact.open(path)

    # Extra top-level members never enter the container.
    _write_raw_v3(path, _v3_directory(objects, source_recipe="nope"), b"\x00" * 4)
    with pytest.raises(ArtifactError, match="missing or extra members"):
        Artifact.open(path)

    # The entry file record must name no sidecar path.
    directory = _v3_directory(objects)
    directory["files"] = [{"path": "part-1.ninfer", "payload_bytes": 4}]
    _write_raw_v3(path, directory, b"\x00" * 4)
    with pytest.raises(ArtifactError, match="files\\[0\\].path"):
        Artifact.open(path)

    # Entry length must agree with the declared file record (valid geometry here).
    clean_objects = [
        {
            "id": "a",
            "kind": "tensor",
            "shape": [1],
            "format": "int32",
            "layout": "contiguous_le_v1",
            "offset": 0,
            "bytes": 4,
        },
        {
            "id": "b",
            "kind": "resource",
            "encoding": "raw_bytes_v1",
            "offset": 4,
            "bytes": 2,
        },
    ]
    directory = _v3_directory(clean_objects)
    directory["files"] = [{"path": None, "payload_bytes": 5}]
    _write_raw_v3(path, directory, b"\x00" * 6)
    with pytest.raises(ArtifactError, match="entry length"):
        Artifact.open(path)

    # Tensor byte count must match its layout geometry.
    bad_size = [
        {
            "id": "bad-size",
            "kind": "tensor",
            "shape": [2],
            "format": "bf16",
            "layout": "contiguous_le_v1",
            "offset": 0,
            "bytes": 2,
        }
    ]
    _write_raw_v3(path, _v3_directory(bad_size), b"\x00" * 2)
    with pytest.raises(ArtifactError, match="does not match its layout"):
        Artifact.open(path)


def test_v3_reader_rejects_unknown_magic_and_zero_length_json(tmp_path):
    path = tmp_path / "invalid.ninfer"
    path.write_bytes(HEADER.pack(b"NINFER\x00\x04", 2, b"\x00" * ARTIFACT_ID_BYTES) + b"{}")
    with pytest.raises(ArtifactError, match="magic"):
        Artifact.open(path)
    path.write_bytes(HEADER.pack(MAGIC, 0, b"\x00" * ARTIFACT_ID_BYTES))
    with pytest.raises(ArtifactError, match="beyond the file"):
        Artifact.open(path)


def test_legacy_v2_artifact_stays_readable(tmp_path):
    path = tmp_path / "legacy.ninfer"
    _write_raw_v2(
        path,
        {
            "identity": {
                "model_id": "qwen3.6-27b",
                "weights_id": "groupwise-int",
            },
            "objects": [
                {
                    "name": "direct/i32",
                    "kind": "tensor",
                    "shape": [1],
                    "format": "I32",
                    "layout": "contiguous-le-v1",
                    "offset": 0,
                    "bytes": 4,
                }
            ],
        },
        b"\x01\x02\x03\x04",
    )
    with Artifact.open(path) as artifact:
        assert artifact.v3 is False
        assert artifact.identity.model_id == "qwen3.6-27b"
        assert artifact.objects[0].name == "direct/i32"
        assert bytes(artifact.payload("direct/i32")) == b"\x01\x02\x03\x04"


def test_legacy_v1_artifact_reads_for_registered_targets_only(tmp_path):
    registered = tmp_path / "registered.ninfer"
    _write_raw_v2(
        registered,
        {
            "model_id": "qwen3.6-27b",
            "objects": [
                {
                    "name": "direct/i32",
                    "kind": "tensor",
                    "shape": [1],
                    "format": "I32",
                    "layout": "contiguous-le-v1",
                    "offset": 0,
                    "bytes": 4,
                }
            ],
        },
        b"\x01\x02\x03\x04",
        magic=b"NINFER\x00\x01",
    )
    with Artifact.open(registered) as artifact:
        assert artifact.identity.weights_id == "groupwise-int"

    unregistered = tmp_path / "unregistered.ninfer"
    _write_raw_v2(
        unregistered,
        {"model_id": "other-model", "objects": []},
        magic=b"NINFER\x00\x01",
    )
    with pytest.raises(ArtifactError, match="registered groupwise target"):
        Artifact.open(unregistered)


def test_upgrade_script_reframes_v2_as_v3(tmp_path):
    legacy = tmp_path / "legacy.ninfer"
    _write_raw_v2(
        legacy,
        {
            "identity": {
                "model_id": "qwen3.6-27b",
                "weights_id": "groupwise-int",
            },
            "objects": [
                {
                    "name": "direct/i32",
                    "kind": "tensor",
                    "shape": [1],
                    "format": "I32",
                    "layout": "contiguous-le-v1",
                    "offset": 0,
                    "bytes": 4,
                },
                {
                    "name": "frontend/tokenizer.json",
                    "kind": "resource",
                    "encoding": "raw-bytes-v1",
                    "offset": 4,
                    "bytes": 2,
                },
            ],
        },
        b"\x01\x02\x03\x04{}",
    )
    upgraded = tmp_path / "upgraded.ninfer"
    # The reframe path never calls torch; shadow it so the subprocess survives
    # environments where the installed torch cannot be imported.
    stub_dir = tmp_path / "torch_stub"
    stub_dir.mkdir()
    (stub_dir / "torch.py").write_text(
        "class Tensor: pass\n"
        "bfloat16 = 'bfloat16'\n"
        "float16 = 'float16'\n"
        "float32 = 'float32'\n"
        "int8 = 'int8'\n"
        "int16 = 'int16'\n"
        "int32 = 'int32'\n"
        "int64 = 'int64'\n"
        "long = 'int64'\n"
        "uint8 = 'uint8'\n"
        "device = object\n"
        "bool = 'bool'\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(stub_dir))
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "upgrade_ninfer_v2_to_v3.py"),
         str(legacy), str(upgraded)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr

    with Artifact.open(upgraded) as artifact:
        assert artifact.v3 is True
        assert artifact.identity == ArtifactIdentity("qwen3.6-27b", "groupwise-int")
        assert [obj.name for obj in artifact.objects] == [
            "direct/i32",
            "frontend/tokenizer.json",
        ]
        assert bytes(artifact.payload("direct/i32")) == b"\x01\x02\x03\x04"
        assert bytes(artifact.payload("frontend/tokenizer.json")) == b"{}"
        payload_offset = artifact.payload_offset
    # Payload bytes survive the reframe untouched. The v3 file pads its payload to a
    # 4096-byte sector boundary, so anchor the comparison on the payload offset.
    legacy_bytes = legacy.read_bytes()
    upgraded_bytes = upgraded.read_bytes()
    assert upgraded_bytes[payload_offset : payload_offset + 6] == legacy_bytes[-6:]
    assert upgraded_bytes[:8] == MAGIC