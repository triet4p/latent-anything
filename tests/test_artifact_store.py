"""Focused Sprint 74 Task 03 tests for safe artifact storage."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from latent_anything.artifact_store import ArtifactStore, ArtifactStoreError
from latent_anything.portable import PortableLimits, PortableNodeError, decode_portable, encode_portable


def test_artifact_store_round_trip_and_stable_identity(tmp_path: Path) -> None:
    root = os.fspath(tmp_path)
    store = ArtifactStore(root)

    written = store.write("nested/value.la", b"portable-bytes", artifact_type="latent-value", metadata={"v": 1})
    restored = store.read("nested/value.la")

    assert restored.payload == b"portable-bytes"
    assert restored.identity == written.identity
    assert restored.artifact_type == "latent-value"
    assert restored.metadata == {"v": 1}


def test_artifact_store_rejects_traversal_symlink_and_checksum_tampering(tmp_path: Path) -> None:
    root = os.fspath(tmp_path)
    store = ArtifactStore(root)
    store.write("value.la", b"payload", artifact_type="value")

    with pytest.raises(ArtifactStoreError, match="traversal"):
        store.read("../value.la")
    with pytest.raises(ArtifactStoreError, match="checksum"):
        path = os.path.join(root, "value.la")
        with open(path, "r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            stream.write(b"!")
        store.read("value.la")

    outside = os.path.join(root, "outside")
    os.mkdir(outside)
    link = os.path.join(root, "link")
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable in this test environment")
    with pytest.raises(ArtifactStoreError, match="symlink|reparse"):
        store.write("link/escape.la", b"nope", artifact_type="value")

    if os.name == "nt":
        junction = os.path.join(root, "junction")
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", junction, outside], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            with pytest.raises(ArtifactStoreError, match="junction|reparse"):
                store.write("junction/escape.la", b"nope", artifact_type="value")


def test_artifact_store_bounds_truncated_and_version_mismatch(tmp_path: Path) -> None:
    root = os.fspath(tmp_path)
    store = ArtifactStore(root, max_artifact_bytes=64)
    with pytest.raises(ArtifactStoreError, match="maximum"):
        store.write("too-large.la", b"x" * 65, artifact_type="value")

    path = os.path.join(root, "broken.la")
    with open(path, "wb") as stream:
        stream.write(b"LATENT-ARTIFACT\x01\x00\x00\x00\x08{}")
    with pytest.raises(ArtifactStoreError, match="truncated"):
        store.read("broken.la")

    store.write("version.la", b"payload", artifact_type="value")
    version_path = os.path.join(root, "version.la")
    with open(version_path, "rb") as stream:
        version_bytes = stream.read().replace(b"artifact-envelope-v1", b"artifact-envelope-v0", 1)
    with open(version_path, "wb") as stream:
        stream.write(version_bytes)
    with pytest.raises(ArtifactStoreError, match="schema version"):
        store.read("version.la")

    payload = encode_portable({"array": [1, 2, 3]})
    with pytest.raises(PortableNodeError, match="cannot be read"):
        decode_portable(payload[:-12])


def test_artifact_metadata_is_recursively_immutable(tmp_path: Path) -> None:
    store = ArtifactStore(os.fspath(tmp_path))
    restored = store.write(
        "nested.la",
        b"payload",
        artifact_type="value",
        metadata={"plugin": {"name": "demo", "tags": ["one"]}},
    )

    with pytest.raises(TypeError):
        restored.metadata["plugin"]["name"] = "changed"  # type: ignore[index]
    with pytest.raises(AttributeError):
        restored.metadata["plugin"]["tags"].append("two")  # type: ignore[union-attr]


def test_portable_decode_allocation_guard_is_enforced() -> None:
    payload = encode_portable({"array": np.zeros(4, dtype=np.float64)})
    with pytest.raises(PortableNodeError, match="allocation guard"):
        decode_portable(payload, limits=PortableLimits(max_total_array_bytes=1))
