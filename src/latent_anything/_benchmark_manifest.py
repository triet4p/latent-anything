"""Fail-closed validation for the frozen benchmark-manifest schema.

This contract describes benchmark declarations only. It does not run models,
select datasets, execute controls, or construct concrete Sprint 80 manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from latent_anything._artifact_path import resolve_artifact_path

SCHEMA_VERSION = "benchmark-manifest-schema-v1"
SCHEMA_PATH = resolve_artifact_path("benchmark_manifest_schema_v1.json")
_CANONICALIZATION = "json-sort-keys-no-whitespace-utf8"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class BenchmarkManifestValidationError(ValueError):
    """Raised when a benchmark manifest is malformed, incomplete, or mutable."""


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestValidationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkManifestValidationError(f"{name} must be a non-empty string")
    return value


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], *, name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise BenchmarkManifestValidationError(f"{name} fields are invalid ({'; '.join(details)})")


def _string_list(value: object, *, name: str, minimum: int = 0) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BenchmarkManifestValidationError(f"{name} must be a list of strings")
    result = list(value)
    if len(result) < minimum or not all(isinstance(item, str) and item for item in result):
        raise BenchmarkManifestValidationError(f"{name} must contain at least {minimum} non-empty strings")
    if len(set(result)) != len(result):
        raise BenchmarkManifestValidationError(f"{name} contains duplicate identifiers")
    return result


def _int_list(value: object, *, name: str, minimum: int = 1) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BenchmarkManifestValidationError(f"{name} must be a list of integers")
    result = list(value)
    if len(result) < minimum or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in result):
        raise BenchmarkManifestValidationError(f"{name} must contain at least {minimum} non-negative integers")
    if len(set(result)) != len(result):
        raise BenchmarkManifestValidationError(f"{name} contains duplicate seeds")
    return result


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BenchmarkManifestValidationError(f"{name} must be a finite number")
    return float(value)


def _digest(value: object, *, name: str) -> str:
    text = _string(value, name=name)
    if _DIGEST_RE.fullmatch(text) is None:
        raise BenchmarkManifestValidationError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _validate_schema_shape(document: Mapping[str, object]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION or document.get("schema_id") != "benchmark-manifest":
        raise BenchmarkManifestValidationError("unsupported benchmark-manifest schema version or identity")
    if document.get("status") != "frozen" or document.get("canonicalization") != _CANONICALIZATION:
        raise BenchmarkManifestValidationError("benchmark-manifest schema is not frozen")
    required = frozenset(_string_list(document.get("required_fields"), name="required_fields", minimum=1))
    properties = _mapping(document.get("properties"), name="properties")
    expected = frozenset(properties)
    if required != expected | {"schema_version", "manifest_id", "status"}:
        raise BenchmarkManifestValidationError("schema required fields do not cover declared properties")
    for field in expected:
        property_value = _mapping(properties[field], name=f"properties.{field}")
        if property_value.get("type") not in {"object", "array", "string"}:
            raise BenchmarkManifestValidationError(f"properties.{field}.type is unsupported")
    fail_closed = _mapping(document.get("fail_closed"), name="fail_closed")
    for case_name in ("malformed", "incomplete", "post_hoc", "unsupported"):
        case = _mapping(fail_closed.get(case_name), name=f"fail_closed.{case_name}")
        if case.get("claim_allowed") is not False or case.get("outcome") not in {"rejected", "unsupported"}:
            raise BenchmarkManifestValidationError(f"fail_closed.{case_name} must reject claims")


def load_schema(path: Path = SCHEMA_PATH) -> Mapping[str, object]:
    """Load and validate the canonical machine-readable schema document."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkManifestValidationError(f"benchmark-manifest schema cannot be loaded from {path}") from exc
    document = _mapping(raw, name="schema")
    _validate_schema_shape(document)
    return document


def _validate_model(model: Mapping[str, object]) -> None:
    _exact_keys(
        model,
        frozenset({"id", "revision", "revision_kind", "revision_immutable", "artifact_digest"}),
        name="model",
    )
    _string(model.get("id"), name="model.id")
    _string(model.get("revision"), name="model.revision")
    if model.get("revision_kind") not in {"commit", "digest"} or model.get("revision_immutable") is not True:
        raise BenchmarkManifestValidationError("model revision must be an immutable commit or digest")
    _digest(model.get("artifact_digest"), name="model.artifact_digest")


def _validate_dataset(dataset: Mapping[str, object]) -> None:
    _exact_keys(dataset, frozenset({"id", "revision", "split", "split_identity", "data_digest"}), name="dataset")
    for field in ("id", "revision", "split", "split_identity"):
        _string(dataset.get(field), name=f"dataset.{field}")
    _digest(dataset.get("data_digest"), name="dataset.data_digest")


def _validate_representation(representation: Mapping[str, object]) -> None:
    _exact_keys(representation, frozenset({"identity", "axes"}), name="representation")
    _string(representation.get("identity"), name="representation.identity")
    axes = representation.get("axes")
    if isinstance(axes, (str, bytes)) or not isinstance(axes, Sequence) or not axes:
        raise BenchmarkManifestValidationError("representation.axes must be a non-empty list")
    identities: list[str] = []
    for index, raw_axis in enumerate(axes):
        axis = _mapping(raw_axis, name=f"representation.axes[{index}]")
        _exact_keys(axis, frozenset({"name", "selection", "identity"}), name=f"representation.axes[{index}]")
        _string(axis.get("name"), name=f"representation.axes[{index}].name")
        _string(axis.get("selection"), name=f"representation.axes[{index}].selection")
        identities.append(_string(axis.get("identity"), name=f"representation.axes[{index}].identity"))
    if len(set(identities)) != len(identities):
        raise BenchmarkManifestValidationError("representation axis identities must be unique")


def _validate_defect(defect: Mapping[str, object]) -> None:
    _exact_keys(defect, frozenset({"classification", "declaration", "predeclared", "source"}), name="defect")
    classification = defect.get("classification")
    if classification not in {"injected", "known", "counterexample", "none"}:
        raise BenchmarkManifestValidationError("defect.classification is unsupported")
    _string(defect.get("declaration"), name="defect.declaration")
    if defect.get("predeclared") is not True:
        raise BenchmarkManifestValidationError("defect must be predeclared")
    source = _string(defect.get("source"), name="defect.source")
    if classification != "none" and source == "not_applicable":
        raise BenchmarkManifestValidationError("declared defect requires a source")
    if classification == "none" and source != "not_applicable":
        raise BenchmarkManifestValidationError("defect source must be not_applicable when no defect is declared")


def _validate_metrics(metrics: object) -> set[str]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence) or not metrics:
        raise BenchmarkManifestValidationError("metrics must be a non-empty list")
    ids: set[str] = set()
    for index, raw_metric in enumerate(metrics):
        metric = _mapping(raw_metric, name=f"metrics[{index}]")
        _exact_keys(
            metric,
            frozenset({"id", "taxonomy_family_id", "direction", "unit", "estimator", "aggregation"}),
            name=f"metrics[{index}]",
        )
        metric_id = _string(metric.get("id"), name=f"metrics[{index}].id")
        if metric_id in ids:
            raise BenchmarkManifestValidationError("metric identifiers must be unique")
        ids.add(metric_id)
        _string(metric.get("taxonomy_family_id"), name=f"metrics[{index}].taxonomy_family_id")
        if metric.get("direction") not in {"higher_is_better", "lower_is_better", "target_value"}:
            raise BenchmarkManifestValidationError(f"metrics[{index}].direction is unsupported")
        for field in ("unit", "estimator", "aggregation"):
            _string(metric.get(field), name=f"metrics[{index}].{field}")
    return ids


def _validate_seeds(seeds: Mapping[str, object]) -> None:
    _exact_keys(seeds, frozenset({"training", "evaluation", "controls", "independent"}), name="seeds")
    for field in ("training", "evaluation", "controls"):
        _int_list(seeds.get(field), name=f"seeds.{field}")
    if seeds.get("independent") is not True:
        raise BenchmarkManifestValidationError("seeds.independent must be true")


def _validate_controls(controls: object, metric_ids: set[str]) -> None:
    if isinstance(controls, (str, bytes)) or not isinstance(controls, Sequence) or not controls:
        raise BenchmarkManifestValidationError("controls must be a non-empty list")
    ids: set[str] = set()
    required_count = 0
    for index, raw_control in enumerate(controls):
        control = _mapping(raw_control, name=f"controls[{index}]")
        _exact_keys(
            control,
            frozenset({"id", "kind", "metric_ids", "required", "expected_behavior"}),
            name=f"controls[{index}]",
        )
        control_id = _string(control.get("id"), name=f"controls[{index}].id")
        if control_id in ids:
            raise BenchmarkManifestValidationError("control identifiers must be unique")
        ids.add(control_id)
        _string(control.get("kind"), name=f"controls[{index}].kind")
        linked = set(_string_list(control.get("metric_ids"), name=f"controls[{index}].metric_ids", minimum=1))
        if not linked.issubset(metric_ids):
            raise BenchmarkManifestValidationError(f"controls[{index}] references an undeclared metric")
        if not isinstance(control.get("required"), bool):
            raise BenchmarkManifestValidationError(f"controls[{index}].required must be boolean")
        required_count += int(control["required"] is True)
        _string(control.get("expected_behavior"), name=f"controls[{index}].expected_behavior")
    if required_count == 0:
        raise BenchmarkManifestValidationError("at least one control must be required")


def _validate_uncertainty(uncertainty: Mapping[str, object]) -> None:
    _exact_keys(
        uncertainty,
        frozenset({"method", "confidence_level", "repetitions", "unit_of_analysis", "predeclared"}),
        name="uncertainty",
    )
    _string(uncertainty.get("method"), name="uncertainty.method")
    confidence = _finite_number(uncertainty.get("confidence_level"), name="uncertainty.confidence_level")
    if not 0.0 < confidence < 1.0:
        raise BenchmarkManifestValidationError("uncertainty.confidence_level must be between zero and one")
    repetitions = uncertainty.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise BenchmarkManifestValidationError("uncertainty.repetitions must be at least two")
    _string(uncertainty.get("unit_of_analysis"), name="uncertainty.unit_of_analysis")
    if uncertainty.get("predeclared") is not True:
        raise BenchmarkManifestValidationError("uncertainty must be predeclared")


def _validate_causal_expectation(causal: Mapping[str, object], metric_ids: set[str]) -> None:
    _exact_keys(
        causal,
        frozenset({"applicable", "expectation", "target_metric_ids", "falsification_rule", "non_applicable_reason"}),
        name="causal_expectation",
    )
    applicable = causal.get("applicable")
    if not isinstance(applicable, bool):
        raise BenchmarkManifestValidationError("causal_expectation.applicable must be boolean")
    expectation = _string(causal.get("expectation"), name="causal_expectation.expectation")
    linked = set(_string_list(causal.get("target_metric_ids"), name="causal_expectation.target_metric_ids"))
    if not linked.issubset(metric_ids):
        raise BenchmarkManifestValidationError("causal_expectation references an undeclared metric")
    _string(causal.get("falsification_rule"), name="causal_expectation.falsification_rule")
    reason = _string(causal.get("non_applicable_reason"), name="causal_expectation.non_applicable_reason")
    if applicable and (not linked or expectation == "not_applicable" or reason != "not_applicable"):
        raise BenchmarkManifestValidationError("applicable causal expectation must name metrics and have no non-applicable reason")
    if not applicable and (linked or expectation != "not_applicable" or reason == "not_applicable"):
        raise BenchmarkManifestValidationError("non-applicable causal expectation must fail closed explicitly")


def _validate_thresholds(thresholds: object, metric_ids: set[str]) -> None:
    if isinstance(thresholds, (str, bytes)) or not isinstance(thresholds, Sequence) or not thresholds:
        raise BenchmarkManifestValidationError("thresholds must be a non-empty list")
    seen: set[str] = set()
    for index, raw_threshold in enumerate(thresholds):
        threshold = _mapping(raw_threshold, name=f"thresholds[{index}]")
        _exact_keys(
            threshold,
            frozenset({"metric_id", "comparator", "value", "tolerance", "predeclared"}),
            name=f"thresholds[{index}]",
        )
        metric_id = _string(threshold.get("metric_id"), name=f"thresholds[{index}].metric_id")
        if metric_id not in metric_ids or metric_id in seen:
            raise BenchmarkManifestValidationError(f"thresholds[{index}] metric binding is invalid")
        seen.add(metric_id)
        if threshold.get("comparator") not in {">", ">=", "<", "<=", "=="}:
            raise BenchmarkManifestValidationError(f"thresholds[{index}].comparator is unsupported")
        _finite_number(threshold.get("value"), name=f"thresholds[{index}].value")
        tolerance = _finite_number(threshold.get("tolerance"), name=f"thresholds[{index}].tolerance")
        if tolerance < 0.0 or threshold.get("predeclared") is not True:
            raise BenchmarkManifestValidationError(f"thresholds[{index}] must be non-negative and predeclared")
    if seen != metric_ids:
        raise BenchmarkManifestValidationError("every metric must have exactly one predeclared threshold")


def manifest_digest(manifest: Mapping[str, object]) -> str:
    """Compute the canonical digest with the self-referential digest omitted."""

    unsigned = deepcopy(dict(manifest))
    commitment = unsigned.get("commitment")
    if not isinstance(commitment, Mapping):
        raise BenchmarkManifestValidationError("manifest.commitment must be an object")
    unsigned_commitment = dict(commitment)
    unsigned_commitment.pop("manifest_sha256", None)
    unsigned["commitment"] = unsigned_commitment
    encoded = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest(manifest: Mapping[str, object], *, schema: Mapping[str, object] | None = None) -> None:
    """Validate one predeclared manifest against the frozen schema."""

    if schema is not None:
        _validate_schema_shape(schema)
    else:
        load_schema()
    _exact_keys(
        manifest,
        frozenset({"schema_version", "manifest_id", "status", "model", "dataset", "representation", "defect", "metrics", "seeds", "controls", "uncertainty", "causal_expectation", "thresholds", "commitment"}),
        name="manifest",
    )
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "predeclared":
        raise BenchmarkManifestValidationError("manifest must use the frozen schema and predeclared status")
    _string(manifest.get("manifest_id"), name="manifest.manifest_id")
    _validate_model(_mapping(manifest.get("model"), name="model"))
    _validate_dataset(_mapping(manifest.get("dataset"), name="dataset"))
    _validate_representation(_mapping(manifest.get("representation"), name="representation"))
    _validate_defect(_mapping(manifest.get("defect"), name="defect"))
    metric_ids = _validate_metrics(manifest.get("metrics"))
    _validate_seeds(_mapping(manifest.get("seeds"), name="seeds"))
    _validate_controls(manifest.get("controls"), metric_ids)
    _validate_uncertainty(_mapping(manifest.get("uncertainty"), name="uncertainty"))
    _validate_causal_expectation(_mapping(manifest.get("causal_expectation"), name="causal_expectation"), metric_ids)
    _validate_thresholds(manifest.get("thresholds"), metric_ids)
    commitment = _mapping(manifest.get("commitment"), name="commitment")
    _exact_keys(commitment, frozenset({"locked", "declared_at", "manifest_sha256", "canonicalization"}), name="commitment")
    if commitment.get("locked") is not True:
        raise BenchmarkManifestValidationError("manifest commitment must be locked")
    _string(commitment.get("declared_at"), name="commitment.declared_at")
    if commitment.get("canonicalization") != _CANONICALIZATION:
        raise BenchmarkManifestValidationError("manifest canonicalization is unsupported")
    expected = manifest_digest(manifest)
    if commitment.get("manifest_sha256") != expected:
        raise BenchmarkManifestValidationError("manifest digest does not match canonical content")


__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "BenchmarkManifestValidationError",
    "load_schema",
    "manifest_digest",
    "validate_manifest",
]
