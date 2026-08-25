"""Versioned, checksummed, atomic storage for trusted portable payloads."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

_MAGIC = b"LATENT-ARTIFACT\x01"
_SCHEMA_VERSION = "artifact-envelope-v1"
_DEFAULT_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_HEADER_BYTES = 1 * 1024 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class ArtifactStoreError(ValueError):
    """Raised when an artifact is unsafe, corrupt, or exceeds configured limits."""


@dataclass(frozen=True)
class StoredArtifact:
    """A validated artifact envelope and its opaque portable payload."""

    artifact_type: str
    identity: str
    payload: bytes
    metadata: Mapping[str, object]
    schema_version: str = _SCHEMA_VERSION


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactStoreError(f"artifact metadata is not canonical JSON: {exc}") from exc


def _validate_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactStoreError("artifact metadata must be a JSON object")

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ArtifactStoreError("artifact metadata keys must be strings")
                visit(nested)
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            for nested in item:
                visit(nested)

    visit(value)
    _canonical_json(value)
    return cast(dict[str, object], value)


def _freeze_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        stat_result = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return bool(getattr(stat_result, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _reject_link_like_ancestors(path: Path) -> None:
    for component in (path, *path.parents):
        if _is_link_like(component):
            raise ArtifactStoreError("artifact path must not contain a symlink, junction, or reparse point")


class ArtifactStore:
    """Store versioned payloads below a non-symlink root directory."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
        max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES,
    ) -> None:
        if max_artifact_bytes < len(_MAGIC) + 4 or max_header_bytes < 128:
            raise ValueError("artifact limits are too small for a valid envelope")
        self.root = Path(root)
        self.max_artifact_bytes = max_artifact_bytes
        self.max_header_bytes = max_header_bytes
        _reject_link_like_ancestors(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        _reject_link_like_ancestors(self.root)

    def _safe_path(self, relative_path: str | os.PathLike[str]) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ArtifactStoreError("artifact path must be a non-empty relative path without traversal")
        target = self.root.joinpath(*candidate.parts)
        current = self.root
        for part in candidate.parts[:-1]:
            current = current / part
            if current.exists() and _is_link_like(current):
                raise ArtifactStoreError("artifact path contains a symlink, junction, or reparse component")
        if _is_link_like(target):
            raise ArtifactStoreError("artifact target must not be a symlink, junction, or reparse point")
        return target

    @staticmethod
    def _identity(artifact_type: str, payload_digest: str, metadata: dict[str, object]) -> str:
        canonical = _canonical_json(
            {"artifact_type": artifact_type, "payload_sha256": payload_digest, "metadata": metadata}
        )
        return hashlib.sha256(canonical).hexdigest()

    def write(
        self,
        relative_path: str | os.PathLike[str],
        payload: object,
        *,
        artifact_type: object,
        metadata: dict[str, object] | None = None,
        identity: str | None = None,
    ) -> StoredArtifact:
        """Atomically write a checksummed payload and return its envelope."""

        if not isinstance(payload, bytes):
            raise TypeError("artifact payload must be bytes")
        if not artifact_type or not isinstance(artifact_type, str):
            raise ArtifactStoreError("artifact_type must be a non-empty string")
        safe_metadata = _validate_metadata(dict(metadata or {}))
        if len(payload) > self.max_artifact_bytes:
            raise ArtifactStoreError("artifact payload exceeds maximum configured size")
        digest = hashlib.sha256(payload).hexdigest()
        computed_identity = self._identity(artifact_type, digest, safe_metadata)
        if identity is not None and identity != computed_identity:
            raise ArtifactStoreError("provided artifact identity does not match payload and metadata")
        envelope = {
            "schema_version": _SCHEMA_VERSION,
            "artifact_type": artifact_type,
            "identity": computed_identity,
            "payload_sha256": digest,
            "payload_size": len(payload),
            "metadata": safe_metadata,
        }
        header = _canonical_json(envelope)
        if len(header) > self.max_header_bytes:
            raise ArtifactStoreError("artifact header exceeds maximum configured size")
        encoded = _MAGIC + len(header).to_bytes(4, "big") + header + payload
        target = self._safe_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Recheck parent components after mkdir; a pre-existing symlink is never
        # followed by this writer and os.replace replaces the target itself.
        self._safe_path(relative_path)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as stream:
                temporary_name = stream.name
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
            temporary_name = None
            try:
                directory_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Windows does not permit opening directories this way. The
                # file itself was still fsynced before atomic publication.
                pass
        finally:
            if temporary_name is not None:
                with suppress(OSError):
                    os.unlink(temporary_name)
        return StoredArtifact(
            artifact_type,
            computed_identity,
            payload,
            cast(Mapping[str, object], _freeze_metadata(safe_metadata)),
        )

    def read(self, relative_path: str | os.PathLike[str]) -> StoredArtifact:
        """Read, bound-check, and checksum-validate one stored artifact."""

        target = self._safe_path(relative_path)
        try:
            size = target.stat().st_size
        except OSError as exc:
            raise ArtifactStoreError(f"artifact cannot be stat'ed: {exc}") from exc
        if size > self.max_artifact_bytes + self.max_header_bytes + len(_MAGIC) + 4:
            raise ArtifactStoreError("artifact exceeds maximum configured size")
        try:
            raw = target.read_bytes()
        except OSError as exc:
            raise ArtifactStoreError(f"artifact cannot be read: {exc}") from exc
        if (
            len(raw) != size
            or len(raw) > self.max_artifact_bytes + self.max_header_bytes + len(_MAGIC) + 4
            or not raw.startswith(_MAGIC)
            or len(raw) < len(_MAGIC) + 4
        ):
            raise ArtifactStoreError("artifact envelope is truncated or has an invalid magic header")
        header_length_start = len(_MAGIC)
        header_length = int.from_bytes(raw[header_length_start : header_length_start + 4], "big")
        if header_length > self.max_header_bytes or header_length < 2:
            raise ArtifactStoreError("artifact header length exceeds configured bounds")
        header_start = header_length_start + 4
        header_end = header_start + header_length
        if header_end > len(raw):
            raise ArtifactStoreError("artifact envelope header is truncated")
        try:
            header = json.loads(raw[header_start:header_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactStoreError("artifact header is not valid JSON") from exc
        if not isinstance(header, dict):
            raise ArtifactStoreError("artifact header must be a JSON object")
        if header.get("schema_version") != _SCHEMA_VERSION:
            raise ArtifactStoreError("unsupported artifact envelope schema version")
        artifact_type = header.get("artifact_type")
        identity = header.get("identity")
        digest = header.get("payload_sha256")
        payload_size = header.get("payload_size")
        metadata = header.get("metadata")
        if (
            not isinstance(artifact_type, str)
            or not isinstance(identity, str)
            or not isinstance(digest, str)
            or isinstance(payload_size, bool)
            or not isinstance(payload_size, int)
            or not isinstance(metadata, dict)
        ):
            raise ArtifactStoreError("artifact header fields are malformed")
        safe_metadata = _validate_metadata(metadata)
        payload = raw[header_end:]
        if payload_size != len(payload) or payload_size > self.max_artifact_bytes:
            raise ArtifactStoreError("artifact payload size is invalid or exceeds configured bounds")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ArtifactStoreError("artifact payload checksum mismatch")
        if self._identity(artifact_type, digest, safe_metadata) != identity:
            raise ArtifactStoreError("artifact identity mismatch")
        return StoredArtifact(
            artifact_type,
            identity,
            payload,
            cast(Mapping[str, object], _freeze_metadata(safe_metadata)),
        )


__all__ = ["ArtifactStore", "ArtifactStoreError", "StoredArtifact"]
