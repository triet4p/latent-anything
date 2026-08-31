"""Atomic payload/audit transaction primitives for L04 retention."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts._m14_l04_retention_parser import RetentionError, sha256, validate_triplet


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_atomic(path: Path, data: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if not replace and path.exists():
            if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
                return
            raise RetentionError(f"audit collision at {path.name}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_payloads(
    final_dir: Path, files: dict[str, bytes], *, name_map: Mapping[str, str] | None = None
) -> list[Path]:
    """Install only missing exact files; rollback files created by this call."""
    final_dir.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(prefix=".l04-retain-", dir=final_dir))
    created: list[Path] = []
    try:
        staged: dict[Path, Path] = {}
        destinations: dict[Path, bytes] = {}
        for relative, data in files.items():
            destination = final_dir / (name_map.get(relative, Path(relative).name) if name_map else Path(relative).name)
            destinations[destination] = data
            if destination.exists() or destination.is_symlink():
                if destination.is_file() and not destination.is_symlink() and destination.read_bytes() == data:
                    continue
                raise RetentionError(f"final payload collision at {destination.name}")
            temporary = staged_dir / destination.name
            temporary.write_bytes(data)
            staged[destination] = temporary
        for destination, temporary in staged.items():
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                expected = destinations[destination]
                if destination.is_file() and not destination.is_symlink() and destination.read_bytes() == expected:
                    temporary.unlink(missing_ok=True)
                    continue
                raise RetentionError(f"final payload collision at {destination.name}") from exc
            temporary.unlink(missing_ok=True)
            created.append(destination)
        return created
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for child in staged_dir.iterdir():
            child.unlink(missing_ok=True)
        staged_dir.rmdir()


def reopen_payloads(
    final_dir: Path,
    files: dict[str, bytes],
    plan: dict[str, Any],
    source_sha: str,
    use_case: str,
    attempt: str,
    *,
    name_map: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    reopened: dict[str, bytes] = {}
    final_hashes: dict[str, dict[str, Any]] = {}
    for relative, expected in files.items():
        path = final_dir / (name_map.get(relative, Path(relative).name) if name_map else Path(relative).name)
        if not path.is_file() or path.is_symlink():
            raise RetentionError(f"final payload is missing or not regular: {path.name}")
        data = path.read_bytes()
        if data != expected:
            raise RetentionError(f"final payload changed after retention: {path.name}")
        reopened[relative] = data
        final_hashes[path.name] = {
            "bytes": len(data),
            "sha256": sha256(data),
            "path": f"artifacts/m14/{path.name}",
        }
    validate_triplet(reopened, plan, source_sha, use_case, attempt)
    return final_hashes


def quarantine_raw(raw_path: Path) -> Path:
    """Atomically move a regular raw capture to a same-directory quarantine."""
    if not raw_path.is_file() or raw_path.is_symlink():
        raise RetentionError("raw capture must be a regular file before quarantine")
    quarantine = raw_path.with_name(f".{raw_path.name}.{uuid.uuid4().hex}.quarantine")
    os.replace(raw_path, quarantine)
    if raw_path.exists() or not quarantine.is_file() or quarantine.is_symlink():
        raise RetentionError("raw quarantine publication was not verified")
    return quarantine


def restore_quarantine(quarantine: Path, raw_path: Path, expected: bytes) -> None:
    """Atomically restore a quarantine to its original path and verify bytes."""
    if not quarantine.is_file() or quarantine.is_symlink():
        raise RetentionError("raw quarantine is missing or not regular")
    if raw_path.exists():
        if raw_path.is_file() and not raw_path.is_symlink() and raw_path.read_bytes() == expected:
            quarantine.unlink()
            return
        raise RetentionError("cannot restore raw capture over a conflicting path")
    os.replace(quarantine, raw_path)
    if not raw_path.is_file() or raw_path.is_symlink() or raw_path.read_bytes() != expected or quarantine.exists():
        raise RetentionError("raw quarantine restoration was not verified")


def delete_quarantine(quarantine: Path) -> None:
    """Delete a quarantined raw capture and verify absence."""
    quarantine.unlink()
    if quarantine.exists():
        raise RetentionError("raw quarantine deletion was not verified")


def restore_raw_snapshot(raw_path: Path, expected: bytes) -> None:
    """Atomically recreate a deleted raw capture from its in-memory snapshot."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{raw_path.name}.", suffix=".restore", dir=raw_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, raw_path)
    except Exception as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise RetentionError(f"raw snapshot restoration failed; temporary snapshot retained at {temporary}") from exc
    restored = raw_path.read_bytes() if raw_path.is_file() and not raw_path.is_symlink() else b""
    if (
        not raw_path.is_file()
        or raw_path.is_symlink()
        or raw_path.stat().st_size != len(expected)
        or sha256(restored) != sha256(expected)
        or restored != expected
    ):
        raise RetentionError("raw snapshot restoration was not verified")


__all__ = [
    "delete_quarantine",
    "install_payloads",
    "json_bytes",
    "quarantine_raw",
    "reopen_payloads",
    "restore_quarantine",
    "restore_raw_snapshot",
    "write_atomic",
]
