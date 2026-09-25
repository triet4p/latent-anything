"""Layer, slice, checkpoint, token, time, and feature localization (Sprint 80.13-80.14).

Scope: consume machine-readable, control-qualified detection evidence (already
computed metric values plus predeclared threshold/direction/control wiring) and
localize findings across explicit ordered layer identities and explicit sample
identities with predeclared dataset-slice membership/criteria (80.13 layer/slice
seam below), plus checkpoint, token, time, and feature axes with explicit
manifest/binding-declared order and aligned coordinate observations (80.14 axial
seam below). No detector algorithm is recomputed here; no centralized
statistical-control execution (80.15); no explanations, interventions,
comparisons, or benchmark proof.

Non-goals: parallel estimators, central control engines, public API growth, or
frozen-contract changes. Each seam handles only its own axes; checkpoint/token/
time/feature behavior lives in the axial seam, never in the layer/slice seam.

Report confidence semantics: every localization row emits ``confidence=1.0``
meaning deterministic selection certainty under the already-qualified
predeclared rule (the affected-criterion decision was unambiguous), not a
probability, effect strength, or statistical confidence interval. Evidence
strength travels separately in the qualified detection evidence and control
outcomes that later report assembly (80.22) must carry alongside the location.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, cast

from latent_anything._benchmark_manifest import (
    BenchmarkManifestValidationError,
    validate_manifest,
)
from latent_anything._portable_contract import PortableNodeError, canonical_json
from latent_anything.diagnostics import ControlSelection, DiagnosticRequest, DiagnosticSelection

SUPPORTED_COMPARATORS = (">=", "<=", ">", "<", "==")
"""Threshold comparators accepted from frozen manifests."""

SUPPORTED_DIRECTIONS = ("higher_is_better", "lower_is_better", "target_value")
"""Metric directions accepted from frozen manifests."""

_DIRECTION_COMPARATORS: Mapping[str, frozenset[str]] = {
    "higher_is_better": frozenset({">=", ">"}),
    "lower_is_better": frozenset({"<=", "<"}),
    "target_value": frozenset({"=="}),
}
"""Frozen direction/comparator wiring. Anything else is a mismatch, not a guess."""

_CONTROL_OUTCOMES = frozenset({"passed", "failed"})
"""Control-status convention shared with the detector adapters."""

_LOCALIZER_VERSION = "layer-slice-localizer-v1"

Verdict = Literal["localized", "negative", "unsupported"]
AffectedWhen = Literal["threshold_fail", "threshold_pass"]


class LocalizationError(ValueError):
    """Raised when localization input, evidence, or alignment is fail-closed invalid."""


def _require_axis_results(value: object) -> tuple[AxialAxisResult, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LocalizationError("axes must be a sequence of AxialAxisResult items")
    items = tuple(value)
    for item in items:
        if not isinstance(item, AxialAxisResult):
            raise LocalizationError("axes must hold AxialAxisResult items")
    return items


def _require_members(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)) or not value:
        raise LocalizationError(f"slice {name!r} members must be a non-empty tuple")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise LocalizationError(f"slice {name!r} members must be strings")
    return items


def _require_result(value: object) -> LocalizationResult:
    if not isinstance(value, LocalizationResult):
        raise LocalizationError("result must be a LocalizationResult")
    return value


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LocalizationError(f"{name} must be a mapping")
    return value


def _require_str_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LocalizationError(f"{name} must be a tuple of strings")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise LocalizationError(f"{name} must be a tuple of strings")
    return items


def _require_report_rows(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LocalizationError("report_localization must be a sequence")
    return tuple(value)


def _require_diagnostics(value: object) -> DiagnosticSelection:
    if not isinstance(value, DiagnosticSelection):
        raise LocalizationError("request must declare diagnostics")
    return value


def _require_controls(value: object) -> ControlSelection:
    if not isinstance(value, ControlSelection):
        raise LocalizationError("request must declare controls")
    return value


def _require_public_request(value: object) -> DiagnosticRequest:
    if not isinstance(value, DiagnosticRequest):
        raise LocalizationError("request must be a DiagnosticRequest")
    return value


def _require_request(value: object) -> DiagnosticRequest:
    from latent_anything._diagnostic_workflow import StageContractError as _ContractError

    if not isinstance(value, DiagnosticRequest):
        raise _ContractError("localize executor requires a DiagnosticRequest")
    return value


def _require_input(value: object) -> LocalizationInput:
    if not isinstance(value, LocalizationInput):
        raise LocalizationError("source must be a LocalizationInput")
    return value


def _require_evidence(value: object) -> DetectionEvidence:
    if not isinstance(value, DetectionEvidence):
        raise LocalizationError("evidence must be a DetectionEvidence")
    return value


def _require_manifest(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LocalizationError("manifest must be a mapping")
    return value


def _require_version(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LocalizationError("version must be a non-empty string")
    return value


def _require_order_tuple(value: object, *, name: str) -> tuple[str, ...]:
    items = _string_tuple(value, name=name, minimum=1)
    if len(set(items)) != len(items):
        raise LocalizationError(f"{name} must not contain duplicates")
    return items


def _require_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LocalizationError(f"{name} must be a non-negative integer")
    return int(value)


def _require_index_map(value: object, *, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise LocalizationError(f"{name} must be a mapping")
    declared: dict[str, int] = {}
    for key in value:
        position = _require_non_negative_int(value[key], name=f"{name}[{key!r}]")
        declared[str(key)] = position
    return declared


def _require_controls_tuple(value: object, *, name: str) -> tuple[str, ...]:
    items = _string_tuple(value, name=name, minimum=1)
    if len(set(items)) != len(items):
        raise LocalizationError(f"{name} must not contain duplicates")
    return items


def _require_outcomes(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise LocalizationError("control_outcomes must be a mapping")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise LocalizationError("control_outcomes must map strings to strings")
    return value


def _require_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise LocalizationError("claim_allowed must be boolean")
    return value


def _require_layer_cells(value: object) -> tuple[LayerCell, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LocalizationError("layer_cells must be a list of LayerCell items")
    items = tuple(value)
    if not items:
        raise LocalizationError("layer_cells must not be empty: a global score alone is insufficient")
    for cell in items:
        if not isinstance(cell, LayerCell):
            raise LocalizationError("layer_cells must hold LayerCell items")
    return items


def _require_sample_cells(value: object) -> tuple[SampleCell, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LocalizationError("sample_cells must be a list of SampleCell items")
    items = tuple(value)
    if not items:
        raise LocalizationError("sample_cells must not be empty: a global score alone is insufficient")
    for cell in items:
        if not isinstance(cell, SampleCell):
            raise LocalizationError("sample_cells must hold SampleCell items")
    return items


def _require_slice_defs(value: object) -> tuple[SliceDefinition, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LocalizationError("slices must be a list of SliceDefinition items")
    items = tuple(value)
    if not items:
        raise LocalizationError("slices must not be empty")
    for item in items:
        if not isinstance(item, SliceDefinition):
            raise LocalizationError("slices must hold SliceDefinition items")
    return items


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LocalizationError(f"{name} must be a non-empty string")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise LocalizationError(f"{name} must be a finite number")
    return float(value)


def _passes(value: float, comparator: str, target: float) -> bool:
    if comparator == ">=":
        return bool(value >= target)
    if comparator == "<=":
        return bool(value <= target)
    if comparator == ">":
        return bool(value > target)
    if comparator == "<":
        return bool(value < target)
    if comparator == "==":
        return bool(value == target)
    raise LocalizationError(f"unsupported threshold comparator: {comparator!r}")


def _string_tuple(value: object, *, name: str, minimum: int = 1) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LocalizationError(f"{name} must be a list of strings")
    items = tuple(value)
    for position, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise LocalizationError(f"{name}[{position}] must be a non-empty string")
    if len(items) < minimum:
        raise LocalizationError(f"{name} must hold at least {minimum} item(s)")
    return items


@dataclass(frozen=True)
class DetectionEvidence:
    """Control-qualified detection evidence. Already computed; never re-estimated here."""

    manifest_id: str
    family_id: str
    metric_id: str
    comparator: str
    threshold_value: float
    tolerance: float
    direction: str
    affected_when: AffectedWhen
    required_controls: object
    control_outcomes: object
    outcome: str
    claim_allowed: object
    representation_identity: str

    def __post_init__(self) -> None:
        _non_empty_string(self.manifest_id, name="manifest_id")
        _non_empty_string(self.family_id, name="family_id")
        _non_empty_string(self.metric_id, name="metric_id")
        if self.comparator not in SUPPORTED_COMPARATORS:
            raise LocalizationError(f"unsupported threshold comparator: {self.comparator!r}")
        if self.direction not in SUPPORTED_DIRECTIONS:
            raise LocalizationError(f"unsupported metric direction: {self.direction!r}")
        if self.comparator not in _DIRECTION_COMPARATORS[self.direction]:
            raise LocalizationError(
                f"threshold-direction mismatch: comparator {self.comparator!r} "
                f"does not match direction {self.direction!r}"
            )
        if self.affected_when not in ("threshold_fail", "threshold_pass"):
            raise LocalizationError("affected_when must be 'threshold_fail' or 'threshold_pass'")
        _finite_number(self.threshold_value, name="threshold_value")
        tolerance = _finite_number(self.tolerance, name="tolerance")
        if tolerance < 0.0:
            raise LocalizationError("tolerance must be non-negative")
        required = _require_controls_tuple(self.required_controls, name="required_controls")
        object.__setattr__(self, "required_controls", required)
        outcomes = dict(_require_outcomes(self.control_outcomes))
        for control_id in required:
            status = outcomes.get(control_id)
            if status not in _CONTROL_OUTCOMES:
                raise LocalizationError(
                    f"required control {control_id!r} is unqualified: control outcome must be 'passed' or 'failed'"
                )
        for control_id, status in outcomes.items():
            if status not in _CONTROL_OUTCOMES:
                raise LocalizationError(f"control outcome for {control_id!r} must be passed or failed")
        object.__setattr__(self, "control_outcomes", dict(outcomes))
        if self.outcome not in ("supported", "inconclusive", "unsupported"):
            raise LocalizationError(f"unsupported detection outcome: {self.outcome!r}")
        object.__setattr__(self, "claim_allowed", _require_bool(self.claim_allowed))
        _non_empty_string(self.representation_identity, name="representation_identity")

    def passes(self, value: float) -> bool:
        """Apply the predeclared comparator to one already-computed metric value."""
        return _passes(float(value), self.comparator, float(self.threshold_value))

    def is_affected(self, value: float) -> bool:
        """Apply the declared correctness rule: affected means failing (or passing) the threshold."""
        healthy = self.passes(float(value))
        return (not healthy) if self.affected_when == "threshold_fail" else healthy

    def is_ambiguous(self, value: float) -> bool:
        """Boundary values within tolerance of the threshold cannot carry a location."""
        return abs(float(value) - float(self.threshold_value)) <= float(self.tolerance)

    def to_dict(self) -> dict[str, object]:
        return {
            "affected_when": self.affected_when,
            "claim_allowed": self.claim_allowed,
            "comparator": self.comparator,
            "control_outcomes": dict(_require_outcomes(self.control_outcomes)),
            "direction": self.direction,
            "family_id": self.family_id,
            "manifest_id": self.manifest_id,
            "metric_id": self.metric_id,
            "outcome": self.outcome,
            "representation_identity": self.representation_identity,
            "required_controls": list(_require_controls_tuple(self.required_controls, name="required_controls")),
            "threshold_value": float(self.threshold_value),
            "tolerance": float(self.tolerance),
        }


@dataclass(frozen=True)
class LayerCell:
    """One already-computed per-layer metric observation with preserved provenance."""

    layer_id: str
    metric_value: float
    representation_identity: str

    def __post_init__(self) -> None:
        _non_empty_string(self.layer_id, name="layer_id")
        _finite_number(self.metric_value, name=f"layer {self.layer_id!r} metric_value")
        _non_empty_string(self.representation_identity, name="representation_identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "metric_value": float(self.metric_value),
            "representation_identity": self.representation_identity,
        }


@dataclass(frozen=True)
class SampleCell:
    """One already-computed per-sample metric observation with preserved provenance."""

    sample_id: str
    metric_value: float
    representation_identity: str

    def __post_init__(self) -> None:
        _non_empty_string(self.sample_id, name="sample_id")
        _finite_number(self.metric_value, name=f"sample {self.sample_id!r} metric_value")
        _non_empty_string(self.representation_identity, name="representation_identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_value": float(self.metric_value),
            "representation_identity": self.representation_identity,
            "sample_id": self.sample_id,
        }


@dataclass(frozen=True)
class SliceDefinition:
    """One predeclared dataset slice: explicit criteria plus explicit member identities."""

    slice_id: str
    criteria: str
    member_sample_ids: object

    def __post_init__(self) -> None:
        _non_empty_string(self.slice_id, name="slice_id")
        _non_empty_string(self.criteria, name=f"slice {self.slice_id!r} criteria")
        members = _require_members(self.member_sample_ids, name=self.slice_id)
        if len(set(members)) != len(members):
            raise LocalizationError(f"slice {self.slice_id!r} members must not contain duplicates")
        object.__setattr__(self, "member_sample_ids", members)

    def to_dict(self) -> dict[str, object]:
        return {
            "criteria": self.criteria,
            "member_sample_ids": list(_require_members(self.member_sample_ids, name=self.slice_id)),
            "slice_id": self.slice_id,
        }


@dataclass(frozen=True)
class LocalizationInput:
    """Complete declared localization scope: ordered layers, samples, predeclared slices."""

    manifest_id: str
    evidence: DetectionEvidence
    layer_order: object
    layer_cells: object
    sample_cells: object
    declared_slice_ids: object
    slices: object
    global_score: float | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.manifest_id, name="manifest_id")
        evidence = _require_evidence(self.evidence)
        object.__setattr__(self, "evidence", evidence)
        if evidence.manifest_id != self.manifest_id:
            raise LocalizationError("evidence manifest identity does not match the localization input")
        order = _require_controls_tuple(self.layer_order, name="layer_order")
        lowered = [item.casefold() for item in order]
        if len(set(lowered)) != len(lowered):
            raise LocalizationError("layer_order identities are ambiguous (casefold collision)")
        cells = _require_layer_cells(self.layer_cells)
        for cell in cells:
            if cell.representation_identity != self.evidence.representation_identity:
                raise LocalizationError(f"layer {cell.layer_id!r} representation identity is misaligned")
        cell_ids = [cell.layer_id for cell in cells]
        if len(set(cell_ids)) != len(cell_ids):
            raise LocalizationError("layer_cells must not contain duplicate layer identities")
        if set(cell_ids) != set(order):
            raise LocalizationError("layer_cells must cover exactly the declared layer_order")
        object.__setattr__(self, "layer_cells", cells)
        samples = _require_sample_cells(self.sample_cells)
        for cell in samples:
            if cell.representation_identity != self.evidence.representation_identity:
                raise LocalizationError(f"sample {cell.sample_id!r} representation identity is misaligned")
                raise LocalizationError(f"sample {cell.sample_id!r} representation identity is misaligned")
        sample_ids = [cell.sample_id for cell in samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise LocalizationError("sample_cells must not contain duplicate sample identities")
        object.__setattr__(self, "sample_cells", samples)
        declared = _require_controls_tuple(self.declared_slice_ids, name="declared_slice_ids")
        object.__setattr__(self, "declared_slice_ids", declared)
        slice_defs = _require_slice_defs(self.slices)
        defined_ids = [item.slice_id for item in slice_defs]
        if len(set(defined_ids)) != len(defined_ids):
            raise LocalizationError("slices must not contain duplicate slice identities")
        if set(defined_ids) != set(declared):
            undeclared = sorted(set(defined_ids) - set(declared))
            missing = sorted(set(declared) - set(defined_ids))
            if undeclared:
                raise LocalizationError(f"undeclared slices are not localizable: {', '.join(undeclared)}")
            raise LocalizationError(f"declared slices are missing evidence: {', '.join(missing)}")
        sample_set = set(sample_ids)
        for item in slice_defs:
            unknown = sorted(set(_require_members(item.member_sample_ids, name=item.slice_id)) - sample_set)
            if unknown:
                raise LocalizationError(
                    f"slice {item.slice_id!r} members are misaligned with sample identities: {', '.join(unknown)}"
                )
        covered: set[str] = set()
        for item in slice_defs:
            covered.update(_require_members(item.member_sample_ids, name=item.slice_id))
        orphaned = sorted(sample_set - covered)
        if orphaned:
            raise LocalizationError(f"samples lack predeclared slice membership: {', '.join(orphaned)}")
        object.__setattr__(self, "slices", slice_defs)
        if self.global_score is not None:
            _finite_number(self.global_score, name="global_score")

    def to_dict(self) -> dict[str, object]:
        return {
            "declared_slice_ids": list(_require_controls_tuple(self.declared_slice_ids, name="declared_slice_ids")),
            "evidence": self.evidence.to_dict(),
            "global_score": None if self.global_score is None else float(self.global_score),
            "layer_cells": [cell.to_dict() for cell in _require_layer_cells(self.layer_cells)],
            "layer_order": list(_require_controls_tuple(self.layer_order, name="layer_order")),
            "manifest_id": self.manifest_id,
            "sample_cells": [cell.to_dict() for cell in _require_sample_cells(self.sample_cells)],
            "slices": [item.to_dict() for item in _require_slice_defs(self.slices)],
        }


@dataclass(frozen=True)
class LocalizationResult:
    """Deterministic localization decision. Benign negatives carry no location."""

    verdict: Verdict
    earliest_layer: str | None
    affected_layers: object
    affected_samples: object
    affected_slices: object
    representation_identity: str
    family_id: str
    metric_id: str
    reason: str
    report_localization: object
    control_outcomes: object

    def __post_init__(self) -> None:
        if self.verdict not in ("localized", "negative", "unsupported"):
            raise LocalizationError(f"unsupported localization verdict: {self.verdict!r}")
        object.__setattr__(self, "affected_layers", _require_str_tuple(self.affected_layers, name="affected_layers"))
        object.__setattr__(self, "affected_samples", _require_str_tuple(self.affected_samples, name="affected_samples"))
        object.__setattr__(self, "affected_slices", _require_str_tuple(self.affected_slices, name="affected_slices"))
        object.__setattr__(self, "report_localization", _require_report_rows(self.report_localization))
        object.__setattr__(self, "control_outcomes", dict(_require_outcomes(self.control_outcomes)))
        _non_empty_string(self.representation_identity, name="representation_identity")
        _non_empty_string(self.family_id, name="family_id")
        _non_empty_string(self.metric_id, name="metric_id")
        _non_empty_string(self.reason, name="reason")
        if self.verdict == "localized" and self.earliest_layer is None:
            raise LocalizationError("localized verdicts must name the earliest affected layer")
        if self.verdict in ("negative", "unsupported") and self.earliest_layer is not None:
            raise LocalizationError(f"{self.verdict} verdicts must not name a layer")
        if self.verdict in ("negative", "unsupported") and self.report_localization:
            raise LocalizationError(f"{self.verdict} verdicts must not produce report locations")

    def to_dict(self) -> dict[str, object]:
        return {
            "affected_layers": list(_require_str_tuple(self.affected_layers, name="affected_layers")),
            "affected_samples": list(_require_str_tuple(self.affected_samples, name="affected_samples")),
            "affected_slices": list(_require_str_tuple(self.affected_slices, name="affected_slices")),
            "control_outcomes": dict(_require_outcomes(self.control_outcomes)),
            "earliest_layer": self.earliest_layer,
            "family_id": self.family_id,
            "metric_id": self.metric_id,
            "reason": self.reason,
            "report_localization": [
                dict(_require_mapping(item, name="report row"))
                for item in _require_report_rows(self.report_localization)
            ],
            "verdict": self.verdict,
        }


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LocalizationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def evidence_from_manifest(
    request: object,
    manifest: Mapping[str, object],
    *,
    family_id: str,
    metric_id: str,
    affected_when: AffectedWhen,
    control_outcomes: Mapping[str, str],
    outcome: str,
    claim_allowed: bool,
    representation_identity: str,
) -> DetectionEvidence:
    """Build control-qualified localization evidence from frozen manifest wiring.

    Threshold value/comparator/tolerance, metric direction, and required controls
    come from the predeclared manifest only; observed per-layer/per-sample values
    are never consulted here. Fails closed on any manifest, wiring, or
    threshold-direction gap.
    """
    request = _require_public_request(request)
    manifest = _require_manifest(manifest)
    try:
        validate_manifest(manifest)
    except BenchmarkManifestValidationError as exc:
        raise LocalizationError(f"invalid benchmark manifest: {exc}") from exc
    if manifest.get("manifest_id") != request.manifest_id:
        raise LocalizationError("request manifest_id does not match the manifest")
    _non_empty_string(family_id, name="family_id")
    _non_empty_string(metric_id, name="metric_id")
    if affected_when not in ("threshold_fail", "threshold_pass"):
        raise LocalizationError("affected_when must be 'threshold_fail' or 'threshold_pass'")
    diagnostics = _require_diagnostics(request.diagnostics)
    controls = _require_controls(request.controls)
    if family_id not in diagnostics.family_ids:
        raise LocalizationError(f"family {family_id!r} is not selected by the request")
    if metric_id not in controls.metric_ids:
        raise LocalizationError(f"metric {metric_id!r} is not selected by the request")

    raw_metrics = manifest.get("metrics")
    if isinstance(raw_metrics, (str, bytes)) or not isinstance(raw_metrics, Sequence):
        raise LocalizationError("manifest metrics must be a list")
    direction: str | None = None
    metric_family: str | None = None
    for raw in raw_metrics:
        metric = _mapping(raw, name="metric")
        if metric.get("id") == metric_id:
            direction = cast(str, metric.get("direction"))
            metric_family = cast(str, metric.get("taxonomy_family_id"))
    if direction is None or metric_family is None:
        raise LocalizationError(f"metric {metric_id!r} is not declared by the manifest")
    if metric_family != family_id:
        raise LocalizationError(f"metric {metric_id!r} belongs to family {metric_family!r}, not {family_id!r}")
    raw_thresholds = manifest.get("thresholds")
    if isinstance(raw_thresholds, (str, bytes)) or not isinstance(raw_thresholds, Sequence):
        raise LocalizationError("manifest thresholds must be a list")
    comparator: str | None = None
    threshold_value: float | None = None
    tolerance: float | None = None
    for raw in raw_thresholds:
        threshold = _mapping(raw, name="threshold")
        if threshold.get("metric_id") == metric_id:
            comparator = cast(str, threshold.get("comparator"))
            candidate = threshold.get("value")
            slack = threshold.get("tolerance")
            if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
                raise LocalizationError("threshold value must be a finite number")
            if isinstance(slack, bool) or not isinstance(slack, (int, float)):
                raise LocalizationError("threshold tolerance must be a finite number")
            threshold_value = float(candidate)
            tolerance = float(slack)
    if comparator is None or threshold_value is None or tolerance is None:
        raise LocalizationError(f"metric {metric_id!r} has no predeclared threshold")
    raw_controls = manifest.get("controls")
    if isinstance(raw_controls, (str, bytes)) or not isinstance(raw_controls, Sequence):
        raise LocalizationError("manifest controls must be a list")
    required: list[str] = []
    for raw in raw_controls:
        control = _mapping(raw, name="control")
        if control.get("required") is True:
            control_id = control.get("id")
            if not isinstance(control_id, str) or not control_id:
                raise LocalizationError("required control identity is malformed")
            required.append(control_id)
    if not required:
        raise LocalizationError("manifest declares no required controls")
    missing_request = [control_id for control_id in required if control_id not in request.controls.control_ids]
    if missing_request:
        raise LocalizationError(f"required controls are missing from the request: {', '.join(missing_request)}")
    return DetectionEvidence(
        manifest_id=cast(str, manifest.get("manifest_id")),
        family_id=family_id,
        metric_id=metric_id,
        comparator=comparator,
        threshold_value=threshold_value,
        tolerance=tolerance,
        direction=direction,
        affected_when=affected_when,
        required_controls=tuple(required),
        control_outcomes=dict(control_outcomes),
        outcome=outcome,
        claim_allowed=bool(claim_allowed),
        representation_identity=representation_identity,
    )


def _slice_means(source: LocalizationInput) -> dict[str, float]:
    by_sample = {cell.sample_id: float(cell.metric_value) for cell in _require_sample_cells(source.sample_cells)}
    means: dict[str, float] = {}
    for item in _require_slice_defs(source.slices):
        members = _require_members(item.member_sample_ids, name=item.slice_id)
        total = 0.0
        for sample_id in members:
            total += by_sample[sample_id]
        means[item.slice_id] = total / len(members)
    return means


def _report_item(*, item_id: str, axis: str, selection: str, evidence: DetectionEvidence) -> dict[str, object]:
    # confidence=1.0 is deterministic selection certainty under the
    # already-qualified predeclared affected-criterion decision (unambiguous
    # by construction: ambiguous boundary values reject before a location is
    # emitted). It is not a probability, effect strength, or statistical
    # confidence interval; evidence strength travels in the qualified
    # detection evidence and control outcomes that 80.22 report assembly must
    # carry alongside the location.
    return {
        "axis": axis,
        "confidence": 1.0,
        "evidence_refs": [f"detect:{evidence.family_id}:{evidence.metric_id}"],
        "id": item_id,
        "selection": selection,
        "status": "supported",
    }


def localize_findings(source: LocalizationInput) -> LocalizationResult:
    """Localize already-detected findings across layers and predeclared slices.

    Consumes per-layer and per-sample metric values plus control qualification;
    never recomputes a detector algorithm and never consults ``global_score``.
    The earliest affected layer follows the declared ``layer_order`` only.
    """
    source = _require_input(source)
    evidence = _require_evidence(source.evidence)
    if evidence.outcome == "unsupported":
        return LocalizationResult(
            verdict="unsupported",
            earliest_layer=None,
            affected_layers=(),
            affected_samples=(),
            affected_slices=(),
            representation_identity=evidence.representation_identity,
            family_id=evidence.family_id,
            metric_id=evidence.metric_id,
            reason="detection evidence is unsupported; localization is non-applicable, not success",
            report_localization=(),
            control_outcomes=dict(_require_outcomes(evidence.control_outcomes)),
        )
    if evidence.outcome != "supported" or not evidence.claim_allowed:
        raise LocalizationError("unqualified detection evidence cannot localize")
    required_controls = _require_controls_tuple(evidence.required_controls, name="required_controls")
    control_outcomes = _require_outcomes(evidence.control_outcomes)
    failed = sorted(control_id for control_id in required_controls if control_outcomes.get(control_id) == "failed")
    if failed:
        raise LocalizationError(f"failed required controls block localization: {', '.join(failed)}")
    by_layer = {cell.layer_id: float(cell.metric_value) for cell in _require_layer_cells(source.layer_cells)}
    affected_in_order = [
        layer_id
        for layer_id in _require_controls_tuple(source.layer_order, name="layer_order")
        if evidence.is_affected(by_layer[layer_id])
    ]
    by_sample = {cell.sample_id: float(cell.metric_value) for cell in _require_sample_cells(source.sample_cells)}
    affected_sample_ids = sorted(sample_id for sample_id, value in by_sample.items() if evidence.is_affected(value))
    if not affected_in_order and not affected_sample_ids:
        return LocalizationResult(
            verdict="negative",
            earliest_layer=None,
            affected_layers=(),
            affected_samples=(),
            affected_slices=(),
            representation_identity=evidence.representation_identity,
            family_id=evidence.family_id,
            metric_id=evidence.metric_id,
            reason="no layer or sample meets the declared affected criterion; benign negative produces no location",
            report_localization=(),
            control_outcomes=dict(_require_outcomes(evidence.control_outcomes)),
        )
    if affected_in_order:
        earliest_value = by_layer[affected_in_order[0]]
        if evidence.is_ambiguous(earliest_value):
            raise LocalizationError(
                f"earliest layer {affected_in_order[0]!r} sits within tolerance of the "
                "predeclared threshold; the location is ambiguous"
            )
    means = _slice_means(source)
    affected_slice_ids = [
        item.slice_id for item in _require_slice_defs(source.slices) if evidence.is_affected(means[item.slice_id])
    ]
    affected_slice_ids.sort(
        key=lambda slice_id: list(_require_controls_tuple(source.declared_slice_ids, name="declared_slice_ids")).index(
            slice_id
        )
    )
    for slice_id in affected_slice_ids:
        if evidence.is_ambiguous(means[slice_id]):
            raise LocalizationError(
                f"slice {slice_id!r} sits within tolerance of the predeclared threshold; the location is ambiguous"
            )
    if not affected_in_order:
        return LocalizationResult(
            verdict="negative",
            earliest_layer=None,
            affected_layers=(),
            affected_samples=(),
            affected_slices=(),
            representation_identity=evidence.representation_identity,
            family_id=evidence.family_id,
            metric_id=evidence.metric_id,
            reason="no layer meets the declared affected criterion; sample-only movement produces no layer location",
            report_localization=(),
            control_outcomes=dict(_require_outcomes(evidence.control_outcomes)),
        )
    earliest = affected_in_order[0]
    items: list[dict[str, object]] = [
        _report_item(item_id=f"location-layer-{earliest}", axis="layer", selection=earliest, evidence=evidence)
    ]
    for sample_id in affected_sample_ids:
        items.append(
            _report_item(
                item_id=f"location-sample-{sample_id}",
                axis="sample",
                selection=sample_id,
                evidence=evidence,
            )
        )
    for slice_id in affected_slice_ids:
        items.append(
            _report_item(item_id=f"location-slice-{slice_id}", axis="slice", selection=slice_id, evidence=evidence)
        )
    return LocalizationResult(
        verdict="localized",
        earliest_layer=earliest,
        affected_layers=tuple(affected_in_order),
        affected_samples=tuple(affected_sample_ids),
        affected_slices=tuple(affected_slice_ids),
        representation_identity=evidence.representation_identity,
        family_id=evidence.family_id,
        metric_id=evidence.metric_id,
        reason=(
            f"earliest affected layer {earliest!r} under the declared order with "
            f"{len(affected_sample_ids)} affected sample(s) and {len(affected_slice_ids)} "
            "affected declared slice(s); global score ignored"
        ),
        report_localization=tuple(items),
        control_outcomes=dict(_require_outcomes(evidence.control_outcomes)),
    )


def localization_payload(result: object, source: object) -> dict[str, object]:
    """Assemble the canonical machine-readable localize-stage payload."""
    result = _require_result(result)
    source = _require_input(source)
    source_evidence = _require_evidence(source.evidence)
    if result.representation_identity != source_evidence.representation_identity:
        raise LocalizationError("result provenance does not match the localization input")
    try:
        canonical_json([result.to_dict(), source.to_dict()])
    except PortableNodeError as exc:
        raise LocalizationError(f"localization payload is not canonical JSON: {exc}") from exc
    by_layer = {cell.layer_id: float(cell.metric_value) for cell in _require_layer_cells(source.layer_cells)}
    ordered_cells = [by_layer[layer_id] for layer_id in _require_controls_tuple(source.layer_order, name="layer_order")]
    payload: dict[str, object] = {
        "affected_layers": list(_require_str_tuple(result.affected_layers, name="affected_layers")),
        "affected_samples": list(_require_str_tuple(result.affected_samples, name="affected_samples")),
        "affected_slices": list(_require_str_tuple(result.affected_slices, name="affected_slices")),
        "config": source.evidence.to_dict(),
        "control_outcomes": dict(_require_outcomes(result.control_outcomes)),
        "declared_slice_ids": list(_require_controls_tuple(source.declared_slice_ids, name="declared_slice_ids")),
        "earliest_layer": result.earliest_layer,
        "family_id": result.family_id,
        "global_score": None if source.global_score is None else float(source.global_score),
        "global_score_ignored": True,
        "layer_cells_in_declared_order": [
            {"layer_id": layer_id, "metric_value": value}
            for layer_id, value in zip(
                _require_controls_tuple(source.layer_order, name="layer_order"), ordered_cells, strict=True
            )
        ],
        "layer_order": list(_require_controls_tuple(source.layer_order, name="layer_order")),
        "manifest_id": source.manifest_id,
        "metric_id": result.metric_id,
        "reason": result.reason,
        "report_localization": [
            dict(_require_mapping(item, name="report row")) for item in _require_report_rows(result.report_localization)
        ],
        "representation_identity": result.representation_identity,
        "sample_cells_by_sample_id": [
            {"metric_value": float(by_sample), "sample_id": sample_id}
            for sample_id, by_sample in sorted(
                {
                    cell.sample_id: float(cell.metric_value) for cell in _require_sample_cells(source.sample_cells)
                }.items()
            )
        ],
        "slice_definitions": [item.to_dict() for item in _require_slice_defs(source.slices)],
        "slice_means": {slice_id: float(mean) for slice_id, mean in sorted(_slice_means(source).items())},
        "verdict": result.verdict,
    }
    try:
        canonical_json(payload)
    except PortableNodeError as exc:
        raise LocalizationError(f"localization payload is not canonical JSON: {exc}") from exc
    return payload


def evaluate_localization(source: object) -> tuple[LocalizationResult, dict[str, object]]:
    """Decide one localization input once and return the decision plus its payload."""
    source = _require_input(source)
    result = localize_findings(source)
    return result, localization_payload(result, source)


def make_localize_executor(source: object, *, version: object = _LOCALIZER_VERSION) -> Any:
    """Build a supplied ``localize``-stage executor bound to one localization input.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable verifies the invocation targets ``localize`` with the expected
    request/manifest identity, evaluates :func:`evaluate_localization` once, and
    returns a ``completed`` (or honest ``unsupported``) :class:`StageOutput`.
    No localization algorithm enters ``DiagnosticWorkflow`` itself.
    """
    version = _require_version(version)
    source = _require_input(source)

    def _execute(invocation: Any) -> Any:
        from latent_anything._diagnostic_workflow import StageContractError as _ContractError
        from latent_anything._diagnostic_workflow import StageOutput as _StageOutput

        if invocation.stage != "localize":
            raise _ContractError(f"localize executor received stage {invocation.stage!r}")
        request = _require_request(invocation.request)
        if request.manifest_id != source.manifest_id:
            raise _ContractError("localize executor manifest identity mismatch")
        result, payload = evaluate_localization(source)
        outcome = "completed" if result.verdict in ("localized", "negative") else "unsupported"
        return _StageOutput(stage="localize", outcome=outcome, payload=payload, artifact_refs=())

    _execute.localizer_version = version  # type: ignore[attr-defined]
    return _execute


AXIAL_AXES: tuple[str, ...] = ("checkpoint", "token", "time", "feature")

_AXIAL_VERSION = "axial-localizer-v2"


def manifest_axis_lookup(manifest: object) -> dict[str, dict[str, str]]:
    """Return the manifest's declared representation axes keyed by axis name.

    Read-only binding coverage for 80.14: axis applicability requires an
    explicit manifest declaration plus an explicit axis declaration in the
    localization input. Never infers order from names or input order.
    """
    manifest = _require_manifest(manifest)
    try:
        validate_manifest(manifest)
    except BenchmarkManifestValidationError as exc:
        raise LocalizationError(f"invalid benchmark manifest: {exc}") from exc
    representation = _mapping(manifest.get("representation"), name="representation")
    raw_axes = representation.get("axes")
    if isinstance(raw_axes, (str, bytes)) or not isinstance(raw_axes, Sequence) or not raw_axes:
        raise LocalizationError("manifest representation.axes must be a non-empty list")
    table: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(raw_axes):
        axis = _mapping(raw, name=f"representation.axes[{index}]")
        name = axis.get("name")
        if not isinstance(name, str) or not name:
            raise LocalizationError(f"representation.axes[{index}].name must be a non-empty string")
        if name in table:
            raise LocalizationError(f"duplicate manifest axis: {name!r}")
        table[name] = {
            "identity": _non_empty_string(axis.get("identity"), name=f"representation.axes[{index}].identity"),
            "selection": _non_empty_string(axis.get("selection"), name=f"representation.axes[{index}].selection"),
        }
    return table


@dataclass(frozen=True)
class CheckpointAxisDeclaration:
    """Declared checkpoint axis: explicit monotonic order plus aligned dataset identity."""

    checkpoint_order: object
    checkpoint_index: object
    dataset_slice_id: str
    dataset_configuration: str
    model_identity: str
    representation_identity: str

    def __post_init__(self) -> None:
        order = _require_order_tuple(self.checkpoint_order, name="checkpoint_order")
        lowered = [item.casefold() for item in order]
        if len(set(lowered)) != len(lowered):
            raise LocalizationError("checkpoint_order identities are ambiguous (casefold collision)")
        object.__setattr__(self, "checkpoint_order", order)
        declared = _require_index_map(self.checkpoint_index, name="checkpoint_index")
        if set(declared) != set(order):
            raise LocalizationError("checkpoint_index must cover exactly the declared checkpoint_order")
        ordered = [declared[layer] for layer in order]
        if ordered != sorted(ordered):
            raise LocalizationError("checkpoint_index must be monotonic with the declared checkpoint_order")
        if len(set(ordered)) != len(ordered):
            raise LocalizationError("checkpoint_index positions must be distinct (ambiguous order)")
        _non_empty_string(self.dataset_slice_id, name="dataset_slice_id")
        _non_empty_string(self.dataset_configuration, name="dataset_configuration")
        _non_empty_string(self.model_identity, name="model_identity")
        _non_empty_string(self.representation_identity, name="representation_identity")
        object.__setattr__(self, "checkpoint_index", dict(declared))

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_index": dict(_require_index_map(self.checkpoint_index, name="checkpoint_index")),
            "checkpoint_order": list(_require_order_tuple(self.checkpoint_order, name="checkpoint_order")),
            "dataset_configuration": self.dataset_configuration,
            "dataset_slice_id": self.dataset_slice_id,
            "model_identity": self.model_identity,
            "representation_identity": self.representation_identity,
        }


@dataclass(frozen=True)
class CheckpointCell:
    """One already-computed per-checkpoint metric observation with aligned provenance."""

    checkpoint_id: str
    metric_value: float
    representation_identity: str
    model_identity: str
    dataset_slice_id: str
    dataset_configuration: str
    layer_id: str | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.checkpoint_id, name="checkpoint_id")
        _finite_number(self.metric_value, name=f"checkpoint {self.checkpoint_id!r} metric_value")
        _non_empty_string(self.representation_identity, name="representation_identity")
        _non_empty_string(self.model_identity, name="model_identity")
        _non_empty_string(self.dataset_slice_id, name="dataset_slice_id")
        _non_empty_string(self.dataset_configuration, name="dataset_configuration")
        if self.layer_id is not None:
            _non_empty_string(self.layer_id, name="layer_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "dataset_configuration": self.dataset_configuration,
            "dataset_slice_id": self.dataset_slice_id,
            "layer_id": self.layer_id,
            "metric_value": float(self.metric_value),
            "model_identity": self.model_identity,
            "representation_identity": self.representation_identity,
        }


@dataclass(frozen=True)
class TokenAxisDeclaration:
    """Declared token axis: explicit sample/sequence binding plus tokenization identity."""

    sample_id: str
    sequence_id: str
    token_order: object
    token_positions: object
    tokenization_identity: str
    preprocessing_identity: str
    representation_identity: str

    def __post_init__(self) -> None:
        _non_empty_string(self.sample_id, name="sample_id")
        _non_empty_string(self.sequence_id, name="sequence_id")
        order = _require_order_tuple(self.token_order, name="token_order")
        object.__setattr__(self, "token_order", order)
        declared = _require_index_map(self.token_positions, name="token_positions")
        if set(declared) != set(order):
            raise LocalizationError("token_positions must cover exactly the declared token_order")
        ordered = [declared[token] for token in order]
        if ordered != sorted(ordered):
            raise LocalizationError("token_positions must be monotonic with the declared token_order")
        if len(set(ordered)) != len(ordered):
            raise LocalizationError("token_positions must be distinct (ambiguous order)")
        _non_empty_string(self.tokenization_identity, name="tokenization_identity")
        _non_empty_string(self.preprocessing_identity, name="preprocessing_identity")
        _non_empty_string(self.representation_identity, name="representation_identity")
        object.__setattr__(self, "token_positions", dict(declared))

    def to_dict(self) -> dict[str, object]:
        return {
            "preprocessing_identity": self.preprocessing_identity,
            "representation_identity": self.representation_identity,
            "sample_id": self.sample_id,
            "sequence_id": self.sequence_id,
            "token_order": list(_require_order_tuple(self.token_order, name="token_order")),
            "token_positions": dict(_require_index_map(self.token_positions, name="token_positions")),
            "tokenization_identity": self.tokenization_identity,
        }


@dataclass(frozen=True)
class TokenCell:
    """One already-computed per-token metric observation with bound coordinates."""

    token_id: str
    metric_value: float
    representation_identity: str
    sample_id: str
    sequence_id: str
    tokenization_identity: str
    preprocessing_identity: str

    def __post_init__(self) -> None:
        _non_empty_string(self.token_id, name="token_id")
        _finite_number(self.metric_value, name=f"token {self.token_id!r} metric_value")
        _non_empty_string(self.representation_identity, name="representation_identity")
        _non_empty_string(self.sample_id, name="sample_id")
        _non_empty_string(self.sequence_id, name="sequence_id")
        _non_empty_string(self.tokenization_identity, name="tokenization_identity")
        _non_empty_string(self.preprocessing_identity, name="preprocessing_identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_value": float(self.metric_value),
            "preprocessing_identity": self.preprocessing_identity,
            "representation_identity": self.representation_identity,
            "sample_id": self.sample_id,
            "sequence_id": self.sequence_id,
            "token_id": self.token_id,
            "tokenization_identity": self.tokenization_identity,
        }


@dataclass(frozen=True)
class TimeAxisDeclaration:
    """Declared time axis: explicit ordered trajectory steps with declared ordering."""

    trajectory_id: str
    step_order: object
    step_index: object
    ordering: str
    representation_identity: str

    def __post_init__(self) -> None:
        _non_empty_string(self.trajectory_id, name="trajectory_id")
        order = _require_order_tuple(self.step_order, name="step_order")
        object.__setattr__(self, "step_order", order)
        declared = _require_index_map(self.step_index, name="step_index")
        if set(declared) != set(order):
            raise LocalizationError("step_index must cover exactly the declared step_order")
        ordered = [declared[step] for step in order]
        if ordered != sorted(ordered):
            raise LocalizationError("step_index must be monotonic with the declared step_order")
        if len(set(ordered)) != len(ordered):
            raise LocalizationError("step_index positions must be distinct (ambiguous order)")
        _non_empty_string(self.ordering, name="ordering")
        _non_empty_string(self.representation_identity, name="representation_identity")
        object.__setattr__(self, "step_index", dict(declared))

    def to_dict(self) -> dict[str, object]:
        return {
            "ordering": self.ordering,
            "representation_identity": self.representation_identity,
            "step_index": dict(_require_index_map(self.step_index, name="step_index")),
            "step_order": list(_require_order_tuple(self.step_order, name="step_order")),
            "trajectory_id": self.trajectory_id,
        }


@dataclass(frozen=True)
class TimeCell:
    """One already-computed per-step metric observation with trajectory binding."""

    step_id: str
    metric_value: float
    representation_identity: str
    trajectory_id: str

    def __post_init__(self) -> None:
        _non_empty_string(self.step_id, name="step_id")
        _finite_number(self.metric_value, name=f"step {self.step_id!r} metric_value")
        _non_empty_string(self.representation_identity, name="representation_identity")
        _non_empty_string(self.trajectory_id, name="trajectory_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_value": float(self.metric_value),
            "representation_identity": self.representation_identity,
            "step_id": self.step_id,
            "trajectory_id": self.trajectory_id,
        }


@dataclass(frozen=True)
class FeatureAxisDeclaration:
    """One manifest-bound ordered feature coordinate system."""

    feature_order: object
    feature_indices: object
    axis_identity: str
    selection: str
    representation_identity: str

    def __post_init__(self) -> None:
        order = _require_order_tuple(self.feature_order, name="feature_order")
        indices = _require_index_map(self.feature_indices, name="feature_indices")
        if set(indices) != set(order):
            raise LocalizationError("feature_indices must cover exactly the declared feature_order")
        if [indices[feature] for feature in order] != list(range(len(order))):
            raise LocalizationError("feature_indices must be contiguous and monotonic with feature_order")
        object.__setattr__(self, "feature_order", order)
        object.__setattr__(self, "feature_indices", indices)
        _non_empty_string(self.axis_identity, name="axis_identity")
        _non_empty_string(self.selection, name="selection")
        _non_empty_string(self.representation_identity, name="representation_identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "axis_identity": self.axis_identity,
            "feature_indices": dict(_require_index_map(self.feature_indices, name="feature_indices")),
            "feature_order": list(_require_order_tuple(self.feature_order, name="feature_order")),
            "representation_identity": self.representation_identity,
            "selection": self.selection,
        }


@dataclass(frozen=True)
class FeatureCell:
    """One precomputed metric value bound to a declared feature coordinate."""

    feature_id: str
    feature_index: int
    metric_value: float
    representation_identity: str
    axis_identity: str

    def __post_init__(self) -> None:
        _non_empty_string(self.feature_id, name="feature_id")
        _require_non_negative_int(self.feature_index, name="feature_index")
        _finite_number(self.metric_value, name=f"feature {self.feature_id!r} metric_value")
        _non_empty_string(self.representation_identity, name="representation_identity")
        _non_empty_string(self.axis_identity, name="axis_identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "axis_identity": self.axis_identity,
            "feature_id": self.feature_id,
            "feature_index": int(self.feature_index),
            "metric_value": float(self.metric_value),
            "representation_identity": self.representation_identity,
        }


@dataclass(frozen=True)
class AxialAxisResult:
    """Deterministic per-axis decision: supported location, negative, or explicit absence."""

    axis: str
    status: str
    earliest: str | None
    affected: object
    reason: str
    report_rows: object = ()

    def __post_init__(self) -> None:
        if self.axis not in AXIAL_AXES:
            raise LocalizationError(f"unsupported axial axis: {self.axis!r}")
        if self.status not in ("localized", "negative", "not_applicable", "unsupported"):
            raise LocalizationError(f"unsupported axial status: {self.status!r}")
        object.__setattr__(self, "affected", _require_str_tuple(self.affected, name="affected"))
        object.__setattr__(self, "report_rows", _require_report_rows(self.report_rows))
        _non_empty_string(self.reason, name=f"axis {self.axis} reason")
        if self.status == "localized" and self.earliest is None:
            raise LocalizationError(f"axis {self.axis} localized verdicts must name the earliest coordinate")
        if self.status in ("negative", "not_applicable", "unsupported") and self.earliest is not None:
            raise LocalizationError(f"axis {self.axis} {self.status} verdicts must not name a coordinate")
        if self.status in ("negative", "not_applicable", "unsupported") and self.report_rows:
            raise LocalizationError(f"axis {self.axis} {self.status} verdicts must not produce report rows")
        if self.status in ("negative", "not_applicable", "unsupported") and self.affected:
            raise LocalizationError(f"axis {self.axis} {self.status} verdicts must not list affected coordinates")

    def to_dict(self) -> dict[str, object]:
        return {
            "affected": list(_require_str_tuple(self.affected, name="affected")),
            "axis": self.axis,
            "status": self.status,
            "earliest": self.earliest,
            "reason": self.reason,
            "report_rows": [
                dict(_require_mapping(item, name="report row")) for item in _require_report_rows(self.report_rows)
            ],
        }


@dataclass(frozen=True)
class AxialLocalizationInput:
    """Requested axes with manifest bindings and fully aligned coordinate cells.

    An axis is applicable only when requested, declared in the manifest and
    supplied with a declaration plus aligned cells. Requested-but-absent axes
    are explicit non-applicability. Feature coordinates use the same contract
    as checkpoint, token, and time axes without architecture-specific rules.
    """

    manifest_id: str
    manifest_axes: object
    evidence: object
    requested_axes: object
    checkpoint_declaration: object = None
    checkpoint_cells: object = ()
    token_declaration: object = None
    token_cells: object = ()
    time_declaration: object = None
    time_cells: object = ()
    global_score: float | None = None
    feature_declaration: object = None
    feature_cells: object = ()

    def __post_init__(self) -> None:
        _non_empty_string(self.manifest_id, name="manifest_id")
        manifest_axes = _require_manifest_axes(self.manifest_axes)
        object.__setattr__(self, "manifest_axes", manifest_axes)
        evidence = _require_evidence(self.evidence)
        object.__setattr__(self, "evidence", evidence)
        if evidence.manifest_id != self.manifest_id:
            raise LocalizationError("evidence manifest identity does not match the axial input")
        requested = _require_requested_axes(self.requested_axes)
        if len(set(requested)) != len(requested):
            raise LocalizationError("requested_axes must not contain duplicates")
        for axis in requested:
            if axis not in AXIAL_AXES:
                raise LocalizationError(f"undeclared axis {axis!r} is not localizable here")
        object.__setattr__(self, "requested_axes", requested)
        object.__setattr__(self, "checkpoint_cells", _require_checkpoint_cells(self.checkpoint_cells))
        object.__setattr__(self, "token_cells", _require_token_cells(self.token_cells))
        object.__setattr__(self, "time_cells", _require_time_cells(self.time_cells))
        feature_cells = _require_feature_cells(self.feature_cells)
        object.__setattr__(self, "feature_cells", feature_cells)
        if self.feature_declaration is None and feature_cells:
            raise LocalizationError("feature_cells require a feature declaration")
        if self.feature_declaration is not None:
            object.__setattr__(self, "feature_declaration", _require_feature_declaration(self.feature_declaration))
        for axis in requested:
            if axis not in manifest_axes:
                continue
            if axis == "checkpoint" and self.checkpoint_declaration is None:
                raise LocalizationError("declared checkpoint axis requires a checkpoint declaration")
            if axis == "token" and self.token_declaration is None:
                raise LocalizationError("declared token axis requires a token declaration")
            if axis == "time" and self.time_declaration is None:
                raise LocalizationError("declared time axis requires a time declaration")
            if axis == "feature" and self.feature_declaration is None:
                raise LocalizationError("declared feature axis requires a feature declaration")
        if self.checkpoint_declaration is not None and "checkpoint" not in requested:
            raise LocalizationError("checkpoint declaration without a requested checkpoint axis")
        if self.token_declaration is not None and "token" not in requested:
            raise LocalizationError("token declaration without a requested token axis")
        if self.time_declaration is not None and "time" not in requested:
            raise LocalizationError("time declaration without a requested time axis")
        if self.feature_declaration is not None and "feature" not in requested:
            raise LocalizationError("feature declaration without a requested feature axis")
        if self.checkpoint_declaration is not None:
            _check_checkpoint_alignment(self)
        if self.token_declaration is not None:
            _check_token_alignment(self)
        if self.time_declaration is not None:
            _check_time_alignment(self)
        if self.feature_declaration is not None:
            _check_feature_alignment(self)
        if self.global_score is not None:
            _finite_number(self.global_score, name="global_score")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "checkpoint_cells": [cell.to_dict() for cell in _require_checkpoint_cells(self.checkpoint_cells)],
            "checkpoint_declaration": (
                None
                if self.checkpoint_declaration is None
                else _require_checkpoint_declaration(self.checkpoint_declaration).to_dict()
            ),
            "evidence": _require_evidence(self.evidence).to_dict(),
            "global_score": None if self.global_score is None else float(self.global_score),
            "manifest_axes": {
                axis: dict(value) for axis, value in sorted(_require_manifest_axes(self.manifest_axes).items())
            },
            "manifest_id": self.manifest_id,
            "requested_axes": list(_require_requested_axes(self.requested_axes)),
            "time_cells": [cell.to_dict() for cell in _require_time_cells(self.time_cells)],
            "time_declaration": (
                None if self.time_declaration is None else _require_time_declaration(self.time_declaration).to_dict()
            ),
            "token_cells": [cell.to_dict() for cell in _require_token_cells(self.token_cells)],
            "token_declaration": (
                None if self.token_declaration is None else _require_token_declaration(self.token_declaration).to_dict()
            ),
        }
        if self.feature_declaration is not None:
            payload["feature_cells"] = [cell.to_dict() for cell in _require_feature_cells(self.feature_cells)]
            payload["feature_declaration"] = _require_feature_declaration(self.feature_declaration).to_dict()
        return payload


@dataclass(frozen=True)
class AxialLocalizationResult:
    """Aggregate 80.14 decision across requested axes with honest mixed semantics."""

    verdict: str
    axes: object
    representation_identity: str
    family_id: str
    metric_id: str
    reason: str
    report_localization: object
    control_outcomes: object = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verdict not in ("localized", "negative", "not_applicable", "unsupported"):
            raise LocalizationError(f"unsupported axial verdict: {self.verdict!r}")
        object.__setattr__(self, "report_localization", _require_report_rows(self.report_localization))
        object.__setattr__(self, "control_outcomes", dict(_require_outcomes(self.control_outcomes)))
        _non_empty_string(self.representation_identity, name="representation_identity")
        _non_empty_string(self.family_id, name="family_id")
        _non_empty_string(self.metric_id, name="metric_id")
        _non_empty_string(self.reason, name="reason")
        axes = _require_axis_results(self.axes)
        names = [item.axis for item in axes]
        if len(set(names)) != len(names):
            raise LocalizationError("axial result must not repeat an axis")
        supported = [item for item in axes if item.status == "localized"]
        if self.verdict == "localized" and not supported:
            raise LocalizationError("localized axial verdicts require one supported axis")
        if self.verdict in ("negative", "not_applicable", "unsupported") and supported:
            raise LocalizationError(f"{self.verdict} axial verdicts must not carry supported axes")
        report_rows = _require_report_rows(self.report_localization)
        if self.verdict in ("negative", "not_applicable", "unsupported") and report_rows:
            raise LocalizationError(f"{self.verdict} axial verdicts must not produce report locations")

    def to_dict(self) -> dict[str, object]:
        return {
            "axes": [item.to_dict() for item in _require_axis_results(self.axes)],
            "control_outcomes": dict(_require_outcomes(self.control_outcomes)),
            "family_id": self.family_id,
            "metric_id": self.metric_id,
            "reason": self.reason,
            "report_localization": [
                dict(_require_mapping(item, name="report row"))
                for item in _require_report_rows(self.report_localization)
            ],
            "representation_identity": self.representation_identity,
            "verdict": self.verdict,
        }


def _check_checkpoint_alignment(source: AxialLocalizationInput) -> None:
    evidence = _require_evidence(source.evidence)
    declaration = _require_checkpoint_declaration(source.checkpoint_declaration)
    if declaration.representation_identity != evidence.representation_identity:
        raise LocalizationError("checkpoint representation identity is misaligned")
    cells = _require_checkpoint_cells(source.checkpoint_cells)
    if not cells:
        raise LocalizationError("checkpoint_cells must not be empty: a global score alone is insufficient")
    for cell in cells:
        if cell.representation_identity != evidence.representation_identity:
            raise LocalizationError(f"checkpoint {cell.checkpoint_id!r} representation identity is misaligned")
        if cell.model_identity != declaration.model_identity:
            raise LocalizationError(f"checkpoint {cell.checkpoint_id!r} model identity is misaligned")
        if cell.dataset_slice_id != declaration.dataset_slice_id:
            raise LocalizationError(
                f"checkpoint {cell.checkpoint_id!r} dataset slice is misaligned: "
                "checkpoint comparisons require the same declared dataset slice"
            )
        if cell.dataset_configuration != declaration.dataset_configuration:
            raise LocalizationError(f"checkpoint {cell.checkpoint_id!r} dataset configuration is misaligned")
    identities = [cell.checkpoint_id for cell in cells]
    if len(set(identities)) != len(identities):
        raise LocalizationError("checkpoint_cells must not contain duplicate checkpoint identities")
    if set(identities) != set(_require_order_tuple(declaration.checkpoint_order, name="checkpoint_order")):
        raise LocalizationError("checkpoint_cells must cover exactly the declared checkpoint_order")


def _check_token_alignment(source: AxialLocalizationInput) -> None:
    evidence = _require_evidence(source.evidence)
    declaration = _require_token_declaration(source.token_declaration)
    if declaration.representation_identity != evidence.representation_identity:
        raise LocalizationError("token representation identity is misaligned")
    cells = _require_token_cells(source.token_cells)
    if not cells:
        raise LocalizationError("token_cells must not be empty: a global score alone is insufficient")
    for cell in cells:
        if cell.representation_identity != evidence.representation_identity:
            raise LocalizationError(f"token {cell.token_id!r} representation identity is misaligned")
        if cell.sample_id != declaration.sample_id or cell.sequence_id != declaration.sequence_id:
            raise LocalizationError(f"token {cell.token_id!r} sample/sequence binding is misaligned")
        if cell.tokenization_identity != declaration.tokenization_identity:
            raise LocalizationError(f"token {cell.token_id!r} tokenization identity drifted: refusing to localize")
        if cell.preprocessing_identity != declaration.preprocessing_identity:
            raise LocalizationError(f"token {cell.token_id!r} preprocessing identity drifted: refusing to localize")
    identities = [cell.token_id for cell in cells]
    if len(set(identities)) != len(identities):
        raise LocalizationError("token_cells must not contain duplicate token identities")
    if set(identities) != set(_require_order_tuple(declaration.token_order, name="token_order")):
        raise LocalizationError("token_cells must cover exactly the declared token_order")


def _check_time_alignment(source: AxialLocalizationInput) -> None:
    evidence = _require_evidence(source.evidence)
    declaration = _require_time_declaration(source.time_declaration)
    if declaration.representation_identity != evidence.representation_identity:
        raise LocalizationError("time representation identity is misaligned")
    cells = _require_time_cells(source.time_cells)
    if not cells:
        raise LocalizationError("time_cells must not be empty: a global score alone is insufficient")
    for cell in cells:
        if cell.representation_identity != evidence.representation_identity:
            raise LocalizationError(f"step {cell.step_id!r} representation identity is misaligned")
        if cell.trajectory_id != declaration.trajectory_id:
            raise LocalizationError(f"step {cell.step_id!r} trajectory identity is misaligned")
    identities = [cell.step_id for cell in cells]
    if len(set(identities)) != len(identities):
        raise LocalizationError("time_cells must not contain duplicate step identities")
    if set(identities) != set(_require_order_tuple(declaration.step_order, name="step_order")):
        raise LocalizationError("time_cells must cover exactly the declared step_order")


def _check_feature_alignment(source: AxialLocalizationInput) -> None:
    evidence = _require_evidence(source.evidence)
    declaration = _require_feature_declaration(source.feature_declaration)
    if declaration.representation_identity != evidence.representation_identity:
        raise LocalizationError("feature representation identity is misaligned")
    manifest_axis = _require_manifest_axes(source.manifest_axes).get("feature")
    if manifest_axis is None:
        raise LocalizationError("feature declaration is not bound to a manifest feature axis")
    if manifest_axis.get("identity") != declaration.axis_identity:
        raise LocalizationError("feature axis identity does not match the manifest")
    if manifest_axis.get("selection") != declaration.selection:
        raise LocalizationError("feature selection does not match the manifest")
    cells = _require_feature_cells(source.feature_cells)
    if not cells:
        raise LocalizationError("feature_cells must not be empty: a global score alone is insufficient")
    order = _require_order_tuple(declaration.feature_order, name="feature_order")
    indices = _require_index_map(declaration.feature_indices, name="feature_indices")
    by_feature = {cell.feature_id: cell for cell in cells}
    if len(by_feature) != len(cells):
        raise LocalizationError("feature_cells must not contain duplicate feature identities")
    if set(by_feature) != set(order):
        raise LocalizationError("feature_cells must cover exactly the declared feature_order")
    for feature_id in order:
        cell = by_feature[feature_id]
        if cell.representation_identity != evidence.representation_identity:
            raise LocalizationError(f"feature {feature_id!r} representation identity is misaligned")
        if cell.axis_identity != declaration.axis_identity:
            raise LocalizationError(f"feature {feature_id!r} axis identity is misaligned")
        if cell.feature_index != indices[feature_id]:
            raise LocalizationError(f"feature {feature_id!r} index is misaligned")


def _require_axial_result(value: object) -> AxialLocalizationResult:
    if not isinstance(value, AxialLocalizationResult):
        raise LocalizationError("result must be an AxialLocalizationResult")
    return value


def _require_axial_input(value: object) -> AxialLocalizationInput:
    if not isinstance(value, AxialLocalizationInput):
        raise LocalizationError("source must be an AxialLocalizationInput")
    return value


def _require_checkpoint_declaration(value: object) -> CheckpointAxisDeclaration:
    if not isinstance(value, CheckpointAxisDeclaration):
        raise LocalizationError("checkpoint declaration must be a CheckpointAxisDeclaration")
    return value


def _require_token_declaration(value: object) -> TokenAxisDeclaration:
    if not isinstance(value, TokenAxisDeclaration):
        raise LocalizationError("token declaration must be a TokenAxisDeclaration")
    return value


def _require_time_declaration(value: object) -> TimeAxisDeclaration:
    if not isinstance(value, TimeAxisDeclaration):
        raise LocalizationError("time declaration must be a TimeAxisDeclaration")
    return value


def _require_feature_declaration(value: object) -> FeatureAxisDeclaration:
    if not isinstance(value, FeatureAxisDeclaration):
        raise LocalizationError("feature declaration must be a FeatureAxisDeclaration")
    return value


def _require_checkpoint_cells(value: object) -> tuple[CheckpointCell, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LocalizationError("checkpoint_cells must be a sequence")
    items = tuple(value)
    for item in items:
        if not isinstance(item, CheckpointCell):
            raise LocalizationError("checkpoint_cells must hold CheckpointCell items")
    return items


def _require_token_cells(value: object) -> tuple[TokenCell, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LocalizationError("token_cells must be a sequence")
    items = tuple(value)
    for item in items:
        if not isinstance(item, TokenCell):
            raise LocalizationError("token_cells must hold TokenCell items")
    return items


def _require_time_cells(value: object) -> tuple[TimeCell, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LocalizationError("time_cells must be a sequence")
    items = tuple(value)
    for item in items:
        if not isinstance(item, TimeCell):
            raise LocalizationError("time_cells must hold TimeCell items")
    return items


def _require_feature_cells(value: object) -> tuple[FeatureCell, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LocalizationError("feature_cells must be a sequence")
    items = tuple(value)
    for item in items:
        if not isinstance(item, FeatureCell):
            raise LocalizationError("feature_cells must hold FeatureCell items")
    return items


def _require_manifest_axes(value: object) -> Mapping[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise LocalizationError("manifest_axes must be a mapping")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, Mapping):
            raise LocalizationError("manifest_axes must map axis names to axis mappings")
    return value


def _require_requested_axes(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LocalizationError("requested_axes must be a sequence of axis names")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise LocalizationError("requested_axes must be a sequence of axis names")
    return items


def _qualified_evidence(source: AxialLocalizationInput) -> DetectionEvidence:
    evidence = _require_evidence(source.evidence)
    if evidence.outcome == "unsupported":
        raise LocalizationError("unsupported")
    if evidence.outcome != "supported" or not evidence.claim_allowed:
        raise LocalizationError("unqualified detection evidence cannot localize")
    required_controls = _require_controls_tuple(evidence.required_controls, name="required_controls")
    control_outcomes = _require_outcomes(evidence.control_outcomes)
    failed = sorted(control_id for control_id in required_controls if control_outcomes.get(control_id) == "failed")
    if failed:
        raise LocalizationError(f"failed required controls block localization: {', '.join(failed)}")
    return evidence


def _axial_item(*, item_id: str, axis: str, selection: str, evidence: DetectionEvidence) -> dict[str, object]:
    # Same confidence contract as _report_item: deterministic selection
    # certainty under the qualified predeclared rule, not a probability or
    # statistical interval. Evidence strength travels separately.
    return {
        "axis": axis,
        "confidence": 1.0,
        "evidence_refs": [f"detect:{evidence.family_id}:{evidence.metric_id}"],
        "id": item_id,
        "selection": selection,
        "status": "supported",
    }


def _localize_checkpoint(source: AxialLocalizationInput, evidence: DetectionEvidence) -> AxialAxisResult:
    declaration = _require_checkpoint_declaration(source.checkpoint_declaration)
    by_checkpoint = {
        cell.checkpoint_id: float(cell.metric_value) for cell in _require_checkpoint_cells(source.checkpoint_cells)
    }
    checkpoint_order = _require_order_tuple(declaration.checkpoint_order, name="checkpoint_order")
    affected = [name for name in checkpoint_order if evidence.is_affected(by_checkpoint[name])]
    if not affected:
        return AxialAxisResult(
            axis="checkpoint",
            status="negative",
            earliest=None,
            affected=(),
            reason="no checkpoint meets the declared affected criterion; benign negative produces no location",
        )
    earliest = affected[0]
    if evidence.is_ambiguous(by_checkpoint[earliest]):
        raise LocalizationError(f"earliest checkpoint {earliest!r} sits within tolerance of the predeclared threshold")
    layer_suffix = ""
    ordered_layers = sorted(
        {cell.layer_id for cell in _require_checkpoint_cells(source.checkpoint_cells) if cell.layer_id is not None}
    )
    if ordered_layers:
        layer_suffix = f" at layer {', '.join(ordered_layers)}"
    return AxialAxisResult(
        axis="checkpoint",
        status="localized",
        earliest=earliest,
        affected=tuple(affected),
        reason=f"earliest affected checkpoint {earliest!r} under the declared order{layer_suffix}",
        report_rows=tuple(
            _axial_item(item_id=f"location-checkpoint-{name}", axis="checkpoint", selection=name, evidence=evidence)
            for name in affected
        ),
    )


def _localize_token(source: AxialLocalizationInput, evidence: DetectionEvidence) -> AxialAxisResult:
    declaration = _require_token_declaration(source.token_declaration)
    by_token = {cell.token_id: float(cell.metric_value) for cell in _require_token_cells(source.token_cells)}
    token_order = _require_order_tuple(declaration.token_order, name="token_order")
    affected = [name for name in token_order if evidence.is_affected(by_token[name])]
    if not affected:
        return AxialAxisResult(
            axis="token",
            status="negative",
            earliest=None,
            affected=(),
            reason="no token meets the declared affected criterion; benign negative produces no location",
        )
    earliest = affected[0]
    if evidence.is_ambiguous(by_token[earliest]):
        raise LocalizationError(f"earliest token {earliest!r} sits within tolerance of the predeclared threshold")
    return AxialAxisResult(
        axis="token",
        status="localized",
        earliest=earliest,
        affected=tuple(affected),
        reason=(
            f"affected token coordinate(s) {', '.join(affected)} bound to sample "
            f"{declaration.sample_id!r} sequence {declaration.sequence_id!r} under "
            f"tokenization {declaration.tokenization_identity!r}"
        ),
        report_rows=tuple(
            _axial_item(item_id=f"location-token-{name}", axis="token", selection=name, evidence=evidence)
            for name in affected
        ),
    )


def _localize_time(source: AxialLocalizationInput, evidence: DetectionEvidence) -> AxialAxisResult:
    declaration = _require_time_declaration(source.time_declaration)
    by_step = {cell.step_id: float(cell.metric_value) for cell in _require_time_cells(source.time_cells)}
    step_order = _require_order_tuple(declaration.step_order, name="step_order")
    affected = [name for name in step_order if evidence.is_affected(by_step[name])]
    if not affected:
        return AxialAxisResult(
            axis="time",
            status="negative",
            earliest=None,
            affected=(),
            reason="no step meets the declared affected criterion; benign negative produces no location",
        )
    earliest = affected[0]
    if evidence.is_ambiguous(by_step[earliest]):
        raise LocalizationError(f"earliest step {earliest!r} sits within tolerance of the predeclared threshold")
    return AxialAxisResult(
        axis="time",
        status="localized",
        earliest=earliest,
        affected=tuple(affected),
        reason=(
            f"earliest affected step {earliest!r} under declared ordering "
            f"{declaration.ordering!r} on trajectory {declaration.trajectory_id!r}"
        ),
        report_rows=tuple(
            _axial_item(item_id=f"location-time-{name}", axis="time", selection=name, evidence=evidence)
            for name in affected
        ),
    )


def _localize_feature(source: AxialLocalizationInput, evidence: DetectionEvidence) -> AxialAxisResult:
    declaration = _require_feature_declaration(source.feature_declaration)
    by_feature = {cell.feature_id: float(cell.metric_value) for cell in _require_feature_cells(source.feature_cells)}
    feature_order = _require_order_tuple(declaration.feature_order, name="feature_order")
    affected = [feature_id for feature_id in feature_order if evidence.is_affected(by_feature[feature_id])]
    if not affected:
        return AxialAxisResult(
            axis="feature",
            status="negative",
            earliest=None,
            affected=(),
            reason="no feature meets the declared affected criterion; benign negative produces no location",
        )
    ambiguous = [feature_id for feature_id in affected if evidence.is_ambiguous(by_feature[feature_id])]
    if ambiguous:
        raise LocalizationError(f"feature {ambiguous[0]!r} sits within tolerance of the predeclared threshold")
    return AxialAxisResult(
        axis="feature",
        status="localized",
        earliest=affected[0],
        affected=tuple(affected),
        reason=(
            f"affected feature coordinate(s) {', '.join(affected)} under declared feature-axis "
            f"identity {declaration.axis_identity!r} and order"
        ),
        report_rows=tuple(
            _axial_item(
                item_id=f"location-feature-{feature_id}", axis="feature", selection=feature_id, evidence=evidence
            )
            for feature_id in affected
        ),
    )


def axial_localize(source: AxialLocalizationInput) -> AxialLocalizationResult:
    """Localize detected findings across checkpoint, token, time, and feature axes.

    Consumes aligned per-coordinate metric values plus control qualification;
    never recomputes a detector algorithm and never consults ``global_score``.
    Requested-but-absent axes yield explicit per-axis non-applicability while
    supported present axes are preserved; the aggregate verdict is fail-closed
    honest about the mix.
    """
    source = _require_axial_input(source)
    try:
        evidence = _qualified_evidence(source)
    except LocalizationError as exc:
        if str(exc) == "unsupported":
            absent = [
                AxialAxisResult(
                    axis=axis,
                    status="unsupported",
                    earliest=None,
                    affected=(),
                    reason="detection evidence is unsupported; axial localization is non-applicable, not success",
                )
                for axis in _require_requested_axes(source.requested_axes)
            ]
            return AxialLocalizationResult(
                verdict="unsupported",
                axes=tuple(absent),
                representation_identity=_require_evidence(source.evidence).representation_identity,
                family_id=_require_evidence(source.evidence).family_id,
                metric_id=_require_evidence(source.evidence).metric_id,
                reason="detection evidence is unsupported; axial localization is non-applicable, not success",
                report_localization=(),
                control_outcomes=dict(_require_outcomes(_require_evidence(source.evidence).control_outcomes)),
            )
        raise
    per_axis: list[AxialAxisResult] = []
    for axis in _require_requested_axes(source.requested_axes):
        if axis not in _require_manifest_axes(source.manifest_axes):
            per_axis.append(
                AxialAxisResult(
                    axis=axis,
                    status="not_applicable",
                    earliest=None,
                    affected=(),
                    reason=(
                        f"axis {axis!r} is absent from the manifest representation axes; "
                        "explicit non-applicability, not success"
                    ),
                )
            )
            continue
        if axis == "checkpoint" and source.checkpoint_declaration is None:
            per_axis.append(
                AxialAxisResult(
                    axis=axis,
                    status="not_applicable",
                    earliest=None,
                    affected=(),
                    reason=(
                        "checkpoint axis is manifest-declared but supplies no aligned declaration/cells; "
                        "explicit non-applicability, not success"
                    ),
                )
            )
            continue
        if axis == "token" and source.token_declaration is None:
            per_axis.append(
                AxialAxisResult(
                    axis=axis,
                    status="not_applicable",
                    earliest=None,
                    affected=(),
                    reason=(
                        "token axis is manifest-declared but supplies no aligned declaration/cells; "
                        "explicit non-applicability, not success"
                    ),
                )
            )
            continue
        if axis == "time" and source.time_declaration is None:
            per_axis.append(
                AxialAxisResult(
                    axis=axis,
                    status="not_applicable",
                    earliest=None,
                    affected=(),
                    reason=(
                        "time axis is manifest-declared but supplies no aligned declaration/cells; "
                        "explicit non-applicability, not success"
                    ),
                )
            )
            continue
        if axis == "checkpoint":
            per_axis.append(_localize_checkpoint(source, evidence))
        elif axis == "token":
            per_axis.append(_localize_token(source, evidence))
        elif axis == "time":
            per_axis.append(_localize_time(source, evidence))
        else:
            per_axis.append(_localize_feature(source, evidence))
    supported = [item for item in per_axis if item.status == "localized"]
    rows: list[dict[str, object]] = [
        dict(_require_mapping(row, name="report row"))
        for item in supported
        for row in _require_report_rows(item.report_rows)
    ]
    if supported:
        absent = sorted(item.axis for item in per_axis if item.status in ("not_applicable", "unsupported"))
        negatives = sorted(item.axis for item in per_axis if item.status == "negative")
        suffix = ""
        if absent:
            suffix += f"; absent axe(s) {', '.join(absent)} explicitly non-applicable"
        if negatives:
            suffix += f"; negative axe(s) {', '.join(negatives)} carry no location"
        return AxialLocalizationResult(
            verdict="localized",
            axes=tuple(per_axis),
            representation_identity=evidence.representation_identity,
            family_id=evidence.family_id,
            metric_id=evidence.metric_id,
            reason=(
                f"{len(supported)} axe(s) localized ({', '.join(item.axis for item in supported)}); "
                f"global score ignored{suffix}"
            ),
            report_localization=tuple(rows),
            control_outcomes=dict(_require_outcomes(evidence.control_outcomes)),
        )
    if all(item.status in ("not_applicable", "unsupported") for item in per_axis):
        return AxialLocalizationResult(
            verdict="not_applicable",
            axes=tuple(per_axis),
            representation_identity=evidence.representation_identity,
            family_id=evidence.family_id,
            metric_id=evidence.metric_id,
            reason="every requested axis is absent; explicit non-applicability, not success",
            report_localization=(),
            control_outcomes=dict(_require_outcomes(evidence.control_outcomes)),
        )
    return AxialLocalizationResult(
        verdict="negative",
        axes=tuple(per_axis),
        representation_identity=evidence.representation_identity,
        family_id=evidence.family_id,
        metric_id=evidence.metric_id,
        reason="no requested axis meets the declared affected criterion; benign negative produces no location",
        report_localization=(),
        control_outcomes=dict(_require_outcomes(evidence.control_outcomes)),
    )


def axial_localization_payload(result: object, source: object) -> dict[str, object]:
    """Assemble the canonical machine-readable axial localize-stage payload."""
    result = _require_axial_result(result)
    source = _require_axial_input(source)
    source_evidence = _require_evidence(source.evidence)
    if result.representation_identity != source_evidence.representation_identity:
        raise LocalizationError("result provenance does not match the axial input")
    try:
        canonical_json([result.to_dict(), source.to_dict()])
    except PortableNodeError as exc:
        raise LocalizationError(f"axial payload is not canonical JSON: {exc}") from exc
    payload: dict[str, object] = {
        "axes": [item.to_dict() for item in _require_axis_results(result.axes)],
        "control_outcomes": dict(_require_outcomes(result.control_outcomes)),
        "family_id": result.family_id,
        "global_score": None if source.global_score is None else float(source.global_score),
        "global_score_ignored": True,
        "manifest_axes": {
            axis: dict(value) for axis, value in sorted(_require_manifest_axes(source.manifest_axes).items())
        },
        "manifest_id": source.manifest_id,
        "metric_id": result.metric_id,
        "reason": result.reason,
        "report_localization": [
            dict(_require_mapping(item, name="report row")) for item in _require_report_rows(result.report_localization)
        ],
        "representation_identity": result.representation_identity,
        "requested_axes": list(_require_requested_axes(source.requested_axes)),
        "verdict": result.verdict,
    }

    if source.feature_declaration is not None:
        payload["feature_declaration"] = _require_feature_declaration(source.feature_declaration).to_dict()
        payload["feature_cells"] = [cell.to_dict() for cell in _require_feature_cells(source.feature_cells)]
    try:
        canonical_json(payload)
    except PortableNodeError as exc:
        raise LocalizationError(f"axial payload is not canonical JSON: {exc}") from exc
    return payload


def evaluate_axial_localization(
    source: object,
) -> tuple[AxialLocalizationResult, dict[str, object]]:
    source = _require_axial_input(source)
    """Decide one axial input once and return the decision plus its payload."""
    result = axial_localize(source)
    return result, axial_localization_payload(result, source)


def make_axial_localize_executor(source: object, *, version: object = _AXIAL_VERSION) -> Any:
    """Build a supplied ``localize``-stage executor bound to one axial input.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable verifies the invocation targets ``localize`` with the expected
    request/manifest identity, evaluates :func:`evaluate_axial_localization`
    once, and returns a ``completed`` (or honest ``not_applicable`` /
    ``unsupported``) :class:`StageOutput`. No axis algorithm enters
    ``DiagnosticWorkflow`` itself.
    """
    version = _require_version(version)
    if not isinstance(source, AxialLocalizationInput):
        raise LocalizationError("source must be an AxialLocalizationInput")

    def _execute(invocation: Any) -> Any:
        from latent_anything._diagnostic_workflow import StageContractError as _ContractError
        from latent_anything._diagnostic_workflow import StageOutput as _StageOutput

        if invocation.stage != "localize":
            raise _ContractError(f"localize executor received stage {invocation.stage!r}")
        request = _require_request(invocation.request)
        if request.manifest_id != source.manifest_id:
            raise _ContractError("localize executor manifest identity mismatch")
        result, payload = evaluate_axial_localization(source)
        if result.verdict == "localized" or result.verdict == "negative":
            outcome = "completed"
        elif result.verdict == "not_applicable":
            outcome = "not_applicable"
        else:
            outcome = "unsupported"
        return _StageOutput(stage="localize", outcome=outcome, payload=payload, artifact_refs=())

    _execute.localizer_version = version  # type: ignore[attr-defined]
    return _execute


__all__ = [
    "AXIAL_AXES",
    "SUPPORTED_COMPARATORS",
    "SUPPORTED_DIRECTIONS",
    "AxialAxisResult",
    "AxialLocalizationInput",
    "AxialLocalizationResult",
    "CheckpointAxisDeclaration",
    "FeatureAxisDeclaration",
    "FeatureCell",
    "LayerCell",
    "LocalizationError",
    "LocalizationInput",
    "LocalizationResult",
    "SampleCell",
    "SliceDefinition",
    "TimeAxisDeclaration",
    "TimeCell",
    "TokenAxisDeclaration",
    "TokenCell",
    "axial_localization_payload",
    "axial_localize",
    "evaluate_axial_localization",
    "evaluate_localization",
    "evidence_from_manifest",
    "localization_payload",
    "localize_findings",
    "make_axial_localize_executor",
    "make_localize_executor",
    "manifest_axis_lookup",
]
