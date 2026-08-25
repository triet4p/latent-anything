"""Allowlisted, versioned envelopes for typed results and configuration.

The registry below is intentionally explicit.  Decoding never imports a
module named by an artifact and never evaluates a constructor outside this
allowlist.  This module builds on :mod:`latent_anything.portable`, so arrays
remain NumPy values at the public boundary and no pickle fallback exists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass as dataclass_type
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from latent_anything.portable import PortableNodeError, decode_portable, encode_portable

_SCHEMA_VERSION = "result-envelope-v1"
_T = TypeVar("_T")


class PortableResultError(ValueError):
    """Raised when an envelope is unsupported, malformed, or unsafe."""


@dataclass_type(frozen=True)
class PortableEnvelope:
    """Decoded value plus the identity metadata needed for coherent reuse."""

    value: object
    type_id: str
    identity: str
    provenance: Mapping[str, object]
    behavior_state: Mapping[str, object]


def _registry() -> dict[str, type[object]]:
    """Return the explicit Sprint 74 result/config allowlist."""

    # Imports are lazy to preserve base import isolation and avoid loading
    # optional model integrations merely to use the portable node layer.
    from latent_anything.cem import CEMConfig, CEMIteration, CEMPlanResult
    from latent_anything.clustering import KMeansConfig, KMeansResult
    from latent_anything.config import ObjectSpec
    from latent_anything.covariance import CovarianceConfig
    from latent_anything.dtw import DTWConfig, DTWCostSummary, DTWResult
    from latent_anything.geodesic import GeodesicConfig, GeodesicPath
    from latent_anything.integrated_gradients import IntegratedGradientsConfig, IntegratedGradientsResult
    from latent_anything.mlp_probe import MLPProbeConfig, MLPProbeResult
    from latent_anything.mppi import MPPIConfig, MPPIIteration, MPPIPlanResult, MPPIRecedingHorizonResult
    from latent_anything.pipeline_config import (
        CEMPlannerSpec,
        MPPIPlannerSpec,
        PipelineSpec,
        RewardValueEvaluationSpec,
    )
    from latent_anything.pipeline_models import PipelineResult, RolloutResult
    from latent_anything.pose import PoseConfig
    from latent_anything.probes import LinearProbeConfig, LinearProbeResult
    from latent_anything.rssm import RSSMOneStepMetrics, RSSMPrediction, RSSMRollout, RSSMTransitionConfig
    from latent_anything.runtime.profiling import ProfileEvent, RuntimeProfile
    from latent_anything.tokenized_world_model import TokenizedWorldModelConfig

    classes: tuple[type[object], ...] = (
        ObjectSpec,
        CEMConfig,
        CEMIteration,
        CEMPlanResult,
        KMeansConfig,
        KMeansResult,
        CovarianceConfig,
        DTWConfig,
        DTWCostSummary,
        DTWResult,
        GeodesicConfig,
        GeodesicPath,
        IntegratedGradientsConfig,
        IntegratedGradientsResult,
        MLPProbeConfig,
        MLPProbeResult,
        MPPIConfig,
        MPPIIteration,
        MPPIPlanResult,
        MPPIRecedingHorizonResult,
        CEMPlannerSpec,
        MPPIPlannerSpec,
        PipelineSpec,
        RewardValueEvaluationSpec,
        PipelineResult,
        RolloutResult,
        LinearProbeConfig,
        LinearProbeResult,
        PoseConfig,
        RSSMTransitionConfig,
        RSSMOneStepMetrics,
        RSSMPrediction,
        RSSMRollout,
        ProfileEvent,
        RuntimeProfile,
        TokenizedWorldModelConfig,
    )
    return {f"{item.__module__}:{item.__qualname__}": item for item in classes}


def _type_id(value: object, registry: Mapping[str, type[object]]) -> str:
    for type_id, candidate in registry.items():
        if type(value) is candidate:
            return type_id
    raise PortableResultError(f"type is not in the portable result/config allowlist: {type(value).__name__}")


def _tree(value: object, registry: Mapping[str, type[object]], depth: int = 0) -> object:
    if depth > 32:
        raise PortableResultError("typed result exceeds maximum nesting depth")
    type_id = _type_id(value, registry) if (is_dataclass(value) or isinstance(value, BaseModel)) else None
    if type_id is not None:
        if isinstance(value, BaseModel):
            fields_value = value.model_dump(mode="python")
            kind = "pydantic"
        else:
            fields_value = {field.name: getattr(value, field.name) for field in fields(cast(Any, value))}
            kind = "dataclass"
        return {
            "kind": kind,
            "type_id": type_id,
            "fields": {key: _tree(item, registry, depth + 1) for key, item in fields_value.items()},
        }
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PortableResultError("typed envelope mappings must use string keys")
            result[key] = _tree(item, registry, depth + 1)
        return result
    if isinstance(value, tuple):
        return {
            "__latent_anything_sequence__": "tuple",
            "items": [_tree(item, registry, depth + 1) for item in value],
        }
    if isinstance(value, list):
        return [_tree(item, registry, depth + 1) for item in value]
    return value


def _restore(value: object, registry: Mapping[str, type[object]], depth: int = 0) -> object:
    if depth > 32:
        raise PortableResultError("typed result exceeds maximum nesting depth")
    if isinstance(value, list):
        return [_restore(item, registry, depth + 1) for item in value]
    if not isinstance(value, dict):
        return value
    sequence_marker = value.get("__latent_anything_sequence__")
    if sequence_marker is not None:
        items = value.get("items")
        if sequence_marker != "tuple" or not isinstance(items, list):
            raise PortableResultError("typed envelope sequence marker is malformed")
        return tuple(_restore(item, registry, depth + 1) for item in items)
    kind = value.get("kind")
    if kind not in {"dataclass", "pydantic"}:
        return {key: _restore(item, registry, depth + 1) for key, item in value.items()}
    type_id = value.get("type_id")
    fields_value = value.get("fields")
    if not isinstance(type_id, str) or type_id not in registry or not isinstance(fields_value, dict):
        raise PortableResultError("typed result type marker is not allowlisted or is malformed")
    target = registry[type_id]
    restored_fields = {key: _restore(item, registry, depth + 1) for key, item in fields_value.items()}
    try:
        if kind == "pydantic":
            if not issubclass(target, BaseModel):
                raise PortableResultError("pydantic marker names a non-pydantic type")
            return target.model_validate(restored_fields)
        if not is_dataclass(target):
            raise PortableResultError("dataclass marker names a non-dataclass type")
        return target(**cast(Any, restored_fields))
    except (TypeError, ValueError) as exc:
        raise PortableResultError(f"allowlisted type {type_id!r} failed validation: {exc}") from exc


def _canonical_identity(type_id: str, provenance: Mapping[str, object], behavior_state: Mapping[str, object]) -> str:
    try:
        canonical = json.dumps(
            {"type_id": type_id, "provenance": provenance, "behavior_state": behavior_state},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PortableResultError(f"envelope identity metadata is not canonical JSON: {exc}") from exc
    return hashlib.sha256(canonical).hexdigest()


def _freeze_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _migrate_envelope(decoded: dict[str, object]) -> dict[str, object]:
    """Apply only explicit, local envelope migrations before validation."""

    version = decoded.get("schema_version")
    if version == _SCHEMA_VERSION:
        return decoded
    if version != "result-envelope-v0":
        raise PortableResultError("unsupported typed result envelope schema version")
    migrated = dict(decoded)
    migrated["schema_version"] = _SCHEMA_VERSION
    type_id = migrated.get("type_id")
    provenance = migrated.get("provenance", {})
    behavior_state = migrated.get("behavior_state", {})
    if (
        "identity" not in migrated
        and isinstance(type_id, str)
        and isinstance(provenance, dict)
        and isinstance(behavior_state, dict)
    ):
        migrated["identity"] = _canonical_identity(type_id, provenance, behavior_state)
    return migrated


def encode_result_envelope(
    value: object,
    *,
    provenance: Mapping[str, object] | None = None,
    behavior_state: Mapping[str, object] | None = None,
) -> bytes:
    """Encode one allowlisted typed result/config with state metadata."""

    registry = _registry()
    type_id = _type_id(value, registry)
    safe_provenance = dict(provenance or {})
    safe_behavior_state = dict(behavior_state or {})
    identity = _canonical_identity(type_id, safe_provenance, safe_behavior_state)
    tree = _tree(value, registry)
    envelope = {
        "schema_version": _SCHEMA_VERSION,
        "type_id": type_id,
        "identity": identity,
        "provenance": safe_provenance,
        "behavior_state": safe_behavior_state,
        "value": tree,
    }
    try:
        return encode_portable(envelope)
    except (PortableNodeError, TypeError, ValueError) as exc:
        raise PortableResultError(f"typed result envelope cannot be encoded: {exc}") from exc


def decode_result_envelope(payload: object) -> PortableEnvelope:
    """Decode and validate an allowlisted typed result/config envelope."""

    registry = _registry()
    try:
        decoded = decode_portable(payload)
    except (PortableNodeError, TypeError, ValueError) as exc:
        raise PortableResultError(f"typed result envelope cannot be decoded: {exc}") from exc
    if not isinstance(decoded, dict):
        raise PortableResultError("typed result envelope root must be a mapping")
    decoded = _migrate_envelope(decoded)
    type_id = decoded.get("type_id")
    identity = decoded.get("identity")
    provenance = decoded.get("provenance")
    behavior_state = decoded.get("behavior_state")
    if (
        not isinstance(type_id, str)
        or type_id not in registry
        or not isinstance(identity, str)
        or not isinstance(provenance, dict)
        or not isinstance(behavior_state, dict)
    ):
        raise PortableResultError("typed result envelope metadata is malformed or not allowlisted")
    expected_identity = _canonical_identity(type_id, provenance, behavior_state)
    if identity != expected_identity:
        raise PortableResultError("typed result envelope identity does not match provenance or behavior state")
    if "value" not in decoded:
        raise PortableResultError("typed result envelope value is missing")
    try:
        value = _restore(decoded["value"], registry)
    except (PortableResultError, TypeError, ValueError) as exc:
        if isinstance(exc, PortableResultError):
            raise
        raise PortableResultError(f"typed result value is invalid: {exc}") from exc
    if type(value) is not registry[type_id]:
        raise PortableResultError("typed result value type does not match its allowlisted marker")
    return PortableEnvelope(
        value,
        type_id,
        identity,
        cast(Mapping[str, object], _freeze_metadata(provenance)),
        cast(Mapping[str, object], _freeze_metadata(behavior_state)),
    )


__all__ = ["PortableEnvelope", "PortableResultError", "decode_result_envelope", "encode_result_envelope"]
