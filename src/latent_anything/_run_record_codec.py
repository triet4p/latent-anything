"""Pure codec, identity, and schema-migration helpers for run records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from types import MappingProxyType
from typing import cast

from latent_anything.runtime.profiling import RuntimeProfile

RUN_RECORD_SCHEMA_VERSION = 1
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def now() -> str:
    """Return the current UTC timestamp in the historical format."""
    return datetime.now(UTC).isoformat()


def freeze_json_value(value: object, *, active: set[int] | None = None) -> object:
    """Return a deeply immutable, JSON-compatible snapshot."""
    active_ids = set() if active is None else active
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in active_ids:
            raise ValueError("run-record inputs must not contain cycles")
        active_ids.add(value_id)
        try:
            frozen = {
                key: freeze_json_value(item, active=active_ids) for key, item in value.items() if isinstance(key, str)
            }
        finally:
            active_ids.remove(value_id)
        if len(frozen) != len(value):
            raise TypeError("run-record mappings must use string keys")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        value_id = id(value)
        if value_id in active_ids:
            raise ValueError("run-record inputs must not contain cycles")
        active_ids.add(value_id)
        try:
            return tuple(freeze_json_value(item, active=active_ids) for item in value)
        finally:
            active_ids.remove(value_id)
    if hasattr(value, "__fspath__"):
        from pathlib import Path

        if isinstance(value, Path):
            return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError(f"unsupported run-record value: {type(value).__name__}") from error
        if scalar is value:
            raise TypeError(f"unsupported run-record value: {type(value).__name__}")
        return freeze_json_value(scalar, active=active_ids)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("run-record inputs must contain only finite floats")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported run-record value: {type(value).__name__}")


def json_value(value: object) -> object:
    """Return a deterministic JSON-compatible representation."""
    frozen = freeze_json_value(value)
    if isinstance(frozen, Mapping):
        return {key: json_value(item) for key, item in frozen.items()}
    if isinstance(frozen, tuple):
        return [json_value(item) for item in frozen]
    return frozen


def canonical_json(value: object) -> bytes:
    """Encode a value using the frozen canonical run-record JSON contract."""
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def mapping(value: object) -> dict[str, object]:
    converted = json_value(value)
    return cast(dict[str, object], converted) if isinstance(converted, dict) else {}


def sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


def runtime_profile_metadata(profile: RuntimeProfile | None) -> dict[str, object]:
    """Serialize an existing runtime profile into run metadata."""
    if profile is None:
        return {}
    events = [
        {
            "stage": event.stage,
            "duration_seconds": event.duration_seconds,
            "metadata": dict(event.metadata),
        }
        for event in profile.events
    ]
    return {
        "total_seconds": profile.total_seconds,
        "stage_totals": profile.stage_totals(),
        "events": events,
    }


def compute_run_identity(
    *,
    name: str,
    config: Mapping[str, object],
    code_version: str,
    framework_version: str,
    model_revisions: Mapping[str, str],
    dataset_revisions: Mapping[str, str],
    seeds: Sequence[int],
    environment: Mapping[str, object],
    parent_run_ids: Sequence[str],
    metadata: Mapping[str, object],
) -> str:
    """Hash reproducible inputs while excluding lifecycle timestamps/status."""
    payload = {
        "name": name,
        "config": config,
        "code_version": code_version,
        "framework_version": framework_version,
        "model_revisions": model_revisions,
        "dataset_revisions": dataset_revisions,
        "seeds": list(seeds),
        "environment": environment,
        "parent_run_ids": list(parent_run_ids),
        "metadata": metadata,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def migrate_run_record(payload: Mapping[str, object]) -> dict[str, object]:
    """Migrate a legacy unversioned record into the current schema."""
    result = dict(payload)
    version = int(str(result.get("schema_version", 0)))
    if version > RUN_RECORD_SCHEMA_VERSION:
        raise ValueError(f"run record schema {version} is newer than supported schema {RUN_RECORD_SCHEMA_VERSION}")
    artifacts = result.get("artifacts")
    if version <= RUN_RECORD_SCHEMA_VERSION and isinstance(artifacts, list):
        migrated_artifacts: list[object] = []
        for item in artifacts:
            if isinstance(item, Mapping):
                migrated_item = dict(item)
                digest = migrated_item.get("digest")
                relative_path = migrated_item.get("relative_path")
                if (
                    isinstance(digest, str)
                    and _SHA256_DIGEST.fullmatch(digest) is not None
                    and relative_path == f"artifacts\\{digest}"
                ):
                    migrated_item["relative_path"] = f"artifacts/{digest}"
                migrated_artifacts.append(migrated_item)
            else:
                migrated_artifacts.append(item)
        result["artifacts"] = migrated_artifacts
    result.setdefault("name", result.get("run_name", "legacy-run"))
    result.setdefault("run_id", result.get("id", "legacy"))
    result.setdefault("status", "completed")
    result.setdefault("created_at", now())
    result.setdefault("updated_at", result["created_at"])
    result.setdefault("model_revisions", {})
    result.setdefault("dataset_revisions", {})
    result.setdefault("seeds", [])
    result.setdefault("environment", {})
    result.setdefault("metrics", {})
    result.setdefault("artifacts", [])
    result.setdefault("parent_run_ids", result.get("parents", []))
    result.setdefault("child_run_ids", result.get("children", []))
    result.setdefault("runtime_profile", {})
    result.setdefault("theory_evidence_ids", [])
    result.setdefault("metadata", {})
    result["schema_version"] = RUN_RECORD_SCHEMA_VERSION
    result.pop("run_name", None)
    result.pop("id", None)
    result.pop("parents", None)
    result.pop("children", None)
    return result
