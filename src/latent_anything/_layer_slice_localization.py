"""Layer, slice, checkpoint, token, and time localization (Sprint 80.13-80.14).

Scope: consume machine-readable, control-qualified detection evidence (already
computed metric values plus predeclared threshold/direction/control wiring) and
localize findings across explicit ordered layer identities and explicit sample
identities with predeclared dataset-slice membership/criteria (80.13 layer/slice
seam below), plus checkpoint, token, and time axes with explicit
capture/binding-declared order and aligned observation coordinates (80.14 axial
seam below). No detector algorithm is recomputed here; no centralized
statistical-control execution (80.15); no explanations, interventions,
comparisons, or benchmark proof.

Non-goals: parallel estimators, central control engines, public API growth, or
frozen-contract changes. Each seam handles only its own axes; checkpoint/token/
time behavior lives in the axial seam, never in the layer/slice seam.

Report confidence semantics: every localization row emits ``confidence=1.0``
meaning deterministic selection certainty under the already-qualified
predeclared rule (the affected-criterion decision was unambiguous), not a
probability, effect strength, or statistical confidence interval. Evidence
strength travels separately in the qualified detection evidence and control
outcomes that later report assembly (80.22) must carry alongside the location.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, Mapping, Sequence, cast

from latent_anything._benchmark_manifest import (
    BenchmarkManifestValidationError,
    validate_manifest,
)
from latent_anything._portable_contract import PortableNodeError, canonical_json
from latent_anything.diagnostics import DiagnosticRequest

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
    required_controls: tuple[str, ...]
    control_outcomes: Mapping[str, str]
    outcome: str
    claim_allowed: bool
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
            raise LocalizationError(f"affected_when must be 'threshold_fail' or 'threshold_pass'")
        _finite_number(self.threshold_value, name="threshold_value")
        tolerance = _finite_number(self.tolerance, name="tolerance")
        if tolerance < 0.0:
            raise LocalizationError("tolerance must be non-negative")
        required = _string_tuple(self.required_controls, name="required_controls", minimum=1)
        if len(set(required)) != len(required):
            raise LocalizationError("required_controls must not contain duplicates")
        object.__setattr__(self, "required_controls", required)
        if not isinstance(self.control_outcomes, Mapping):
            raise LocalizationError("control_outcomes must be a mapping")
        outcomes = dict(self.control_outcomes)
        for control_id in required:
            status = outcomes.get(control_id)
            if status not in _CONTROL_OUTCOMES:
                raise LocalizationError(
                    f"required control {control_id!r} is unqualified: "
                    "control outcome must be 'passed' or 'failed'"
                )
        for control_id, status in outcomes.items():
            if status not in _CONTROL_OUTCOMES:
                raise LocalizationError(f"control outcome for {control_id!r} must be passed or failed")
        object.__setattr__(self, "control_outcomes", dict(outcomes))
        if self.outcome not in ("supported", "inconclusive", "unsupported"):
            raise LocalizationError(f"unsupported detection outcome: {self.outcome!r}")
        if not isinstance(self.claim_allowed, bool):
            raise LocalizationError("claim_allowed must be boolean")
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
            "control_outcomes": dict(self.control_outcomes),
            "direction": self.direction,
            "family_id": self.family_id,
            "manifest_id": self.manifest_id,
            "metric_id": self.metric_id,
            "outcome": self.outcome,
            "representation_identity": self.representation_identity,
            "required_controls": list(self.required_controls),
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
    member_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty_string(self.slice_id, name="slice_id")
        _non_empty_string(self.criteria, name=f"slice {self.slice_id!r} criteria")
        members = _string_tuple(self.member_sample_ids, name=f"slice {self.slice_id!r} members", minimum=1)
        if len(set(members)) != len(members):
            raise LocalizationError(f"slice {self.slice_id!r} members must not contain duplicates")
        object.__setattr__(self, "member_sample_ids", members)

    def to_dict(self) -> dict[str, object]:
        return {
            "criteria": self.criteria,
            "member_sample_ids": list(self.member_sample_ids),
            "slice_id": self.slice_id,
        }


@dataclass(frozen=True)
class LocalizationInput:
    """Complete declared localization scope: ordered layers, samples, predeclared slices."""

    manifest_id: str
    evidence: DetectionEvidence
    layer_order: tuple[str, ...]
    layer_cells: tuple[LayerCell, ...]
    sample_cells: tuple[SampleCell, ...]
    declared_slice_ids: tuple[str, ...]
    slices: tuple[SliceDefinition, ...]
    global_score: float | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.manifest_id, name="manifest_id")
        if not isinstance(self.evidence, DetectionEvidence):
            raise LocalizationError("evidence must be a DetectionEvidence")
        if self.evidence.manifest_id != self.manifest_id:
            raise LocalizationError("evidence manifest identity does not match the localization input")
        order = _string_tuple(self.layer_order, name="layer_order", minimum=1)
        if len(set(order)) != len(order):
            raise LocalizationError("layer_order must not contain duplicates")
        lowered = [item.casefold() for item in order]
        if len(set(lowered)) != len(lowered):
            raise LocalizationError("layer_order identities are ambiguous (casefold collision)")
        object.__setattr__(self, "layer_order", order)
        if not isinstance(self.layer_cells, Sequence) or isinstance(self.layer_cells, (str, bytes)):
            raise LocalizationError("layer_cells must be a list of LayerCell items")
        cells = tuple(self.layer_cells)
        if not cells:
            raise LocalizationError("layer_cells must not be empty: a global score alone is insufficient")
        for cell in cells:
            if not isinstance(cell, LayerCell):
                raise LocalizationError("layer_cells must hold LayerCell items")
            if cell.representation_identity != self.evidence.representation_identity:
                raise LocalizationError(
                    f"layer {cell.layer_id!r} representation identity is misaligned"
                )
        cell_ids = [cell.layer_id for cell in cells]
        if len(set(cell_ids)) != len(cell_ids):
            raise LocalizationError("layer_cells must not contain duplicate layer identities")
        if set(cell_ids) != set(order):
            raise LocalizationError("layer_cells must cover exactly the declared layer_order")
        object.__setattr__(self, "layer_cells", cells)
        if not isinstance(self.sample_cells, Sequence) or isinstance(self.sample_cells, (str, bytes)):
            raise LocalizationError("sample_cells must be a list of SampleCell items")
        samples = tuple(self.sample_cells)
        if not samples:
            raise LocalizationError("sample_cells must not be empty: a global score alone is insufficient")
        for cell in samples:
            if not isinstance(cell, SampleCell):
                raise LocalizationError("sample_cells must hold SampleCell items")
            if cell.representation_identity != self.evidence.representation_identity:
                raise LocalizationError(
                    f"sample {cell.sample_id!r} representation identity is misaligned"
                )
        sample_ids = [cell.sample_id for cell in samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise LocalizationError("sample_cells must not contain duplicate sample identities")
        object.__setattr__(self, "sample_cells", samples)
        declared = _string_tuple(self.declared_slice_ids, name="declared_slice_ids", minimum=1)
        if len(set(declared)) != len(declared):
            raise LocalizationError("declared_slice_ids must not contain duplicates")
        object.__setattr__(self, "declared_slice_ids", declared)
        if not isinstance(self.slices, Sequence) or isinstance(self.slices, (str, bytes)):
            raise LocalizationError("slices must be a list of SliceDefinition items")
        slice_defs = tuple(self.slices)
        if not slice_defs:
            raise LocalizationError("slices must not be empty")
        for item in slice_defs:
            if not isinstance(item, SliceDefinition):
                raise LocalizationError("slices must hold SliceDefinition items")
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
            unknown = sorted(set(item.member_sample_ids) - sample_set)
            if unknown:
                raise LocalizationError(
                    f"slice {item.slice_id!r} members are misaligned with sample identities: "
                    f"{', '.join(unknown)}"
                )
        covered: set[str] = set()
        for item in slice_defs:
            covered.update(item.member_sample_ids)
        orphaned = sorted(sample_set - covered)
        if orphaned:
            raise LocalizationError(
                f"samples lack predeclared slice membership: {', '.join(orphaned)}"
            )
        object.__setattr__(self, "slices", slice_defs)
        if self.global_score is not None:
            _finite_number(self.global_score, name="global_score")

    def to_dict(self) -> dict[str, object]:
        return {
            "declared_slice_ids": list(self.declared_slice_ids),
            "evidence": self.evidence.to_dict(),
            "global_score": None if self.global_score is None else float(self.global_score),
            "layer_cells": [cell.to_dict() for cell in self.layer_cells],
            "layer_order": list(self.layer_order),
            "manifest_id": self.manifest_id,
            "sample_cells": [cell.to_dict() for cell in self.sample_cells],
            "slices": [item.to_dict() for item in self.slices],
        }


@dataclass(frozen=True)
class LocalizationResult:
    """Deterministic localization decision. Benign negatives carry no location."""

    verdict: Verdict
    earliest_layer: str | None
    affected_layers: tuple[str, ...]
    affected_samples: tuple[str, ...]
    affected_slices: tuple[str, ...]
    representation_identity: str
    family_id: str
    metric_id: str
    reason: str
    report_localization: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    control_outcomes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verdict not in ("localized", "negative", "unsupported"):
            raise LocalizationError(f"unsupported localization verdict: {self.verdict!r}")
        object.__setattr__(self, "affected_layers", tuple(self.affected_layers))
        object.__setattr__(self, "affected_samples", tuple(self.affected_samples))
        object.__setattr__(self, "affected_slices", tuple(self.affected_slices))
        object.__setattr__(self, "report_localization", tuple(self.report_localization))
        object.__setattr__(self, "control_outcomes", dict(self.control_outcomes))
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
            "affected_layers": list(self.affected_layers),
            "affected_samples": list(self.affected_samples),
            "affected_slices": list(self.affected_slices),
            "control_outcomes": dict(self.control_outcomes),
            "earliest_layer": self.earliest_layer,
            "family_id": self.family_id,
            "metric_id": self.metric_id,
            "reason": self.reason,
            "report_localization": [dict(item) for item in self.report_localization],
            "representation_identity": self.representation_identity,
            "verdict": self.verdict,
        }


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LocalizationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def evidence_from_manifest(
    request: DiagnosticRequest,
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
    if not isinstance(request, DiagnosticRequest):
        raise LocalizationError("request must be a DiagnosticRequest")
    if not isinstance(manifest, Mapping):
        raise LocalizationError("manifest must be a mapping")
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
    if family_id not in request.diagnostics.family_ids:
        raise LocalizationError(f"family {family_id!r} is not selected by the request")
    if metric_id not in request.controls.metric_ids:
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
        raise LocalizationError(
            f"metric {metric_id!r} belongs to family {metric_family!r}, not {family_id!r}"
        )
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
        raise LocalizationError(
            f"required controls are missing from the request: {', '.join(missing_request)}"
        )
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
    by_sample = {cell.sample_id: float(cell.metric_value) for cell in source.sample_cells}
    means: dict[str, float] = {}
    for item in source.slices:
        total = 0.0
        for sample_id in item.member_sample_ids:
            total += by_sample[sample_id]
        means[item.slice_id] = total / len(item.member_sample_ids)
    return means


def _report_item(
    *, item_id: str, axis: str, selection: str, evidence: DetectionEvidence
) -> dict[str, object]:
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
    if not isinstance(source, LocalizationInput):
        raise LocalizationError("source must be a LocalizationInput")
    evidence = source.evidence
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
            control_outcomes=dict(evidence.control_outcomes),
        )
    if evidence.outcome != "supported" or not evidence.claim_allowed:
        raise LocalizationError("unqualified detection evidence cannot localize")
    failed = sorted(
        control_id
        for control_id in evidence.required_controls
        if evidence.control_outcomes.get(control_id) == "failed"
    )
    if failed:
        raise LocalizationError(
            f"failed required controls block localization: {', '.join(failed)}"
        )
    by_layer = {cell.layer_id: float(cell.metric_value) for cell in source.layer_cells}
    affected_in_order = [layer_id for layer_id in source.layer_order if evidence.is_affected(by_layer[layer_id])]
    by_sample = {cell.sample_id: float(cell.metric_value) for cell in source.sample_cells}
    affected_sample_ids = sorted(
        sample_id for sample_id, value in by_sample.items() if evidence.is_affected(value)
    )
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
            control_outcomes=dict(evidence.control_outcomes),
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
        item.slice_id for item in source.slices if evidence.is_affected(means[item.slice_id])
    ]
    affected_slice_ids.sort(key=lambda slice_id: list(source.declared_slice_ids).index(slice_id))
    for slice_id in affected_slice_ids:
        if evidence.is_ambiguous(means[slice_id]):
            raise LocalizationError(
                f"slice {slice_id!r} sits within tolerance of the predeclared "
                "threshold; the location is ambiguous"
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
            control_outcomes=dict(evidence.control_outcomes),
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
            _report_item(
                item_id=f"location-slice-{slice_id}", axis="slice", selection=slice_id, evidence=evidence
            )
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
        control_outcomes=dict(evidence.control_outcomes),
    )


def localization_payload(result: LocalizationResult, source: LocalizationInput) -> dict[str, object]:
    """Assemble the canonical machine-readable localize-stage payload."""
    if not isinstance(result, LocalizationResult):
        raise LocalizationError("result must be a LocalizationResult")
    if not isinstance(source, LocalizationInput):
        raise LocalizationError("source must be a LocalizationInput")
    if result.representation_identity != source.evidence.representation_identity:
        raise LocalizationError("result provenance does not match the localization input")
    try:
        canonical_json([result.to_dict(), source.to_dict()])
    except PortableNodeError as exc:
        raise LocalizationError(f"localization payload is not canonical JSON: {exc}") from exc
    by_layer = {cell.layer_id: float(cell.metric_value) for cell in source.layer_cells}
    ordered_cells = [by_layer[layer_id] for layer_id in source.layer_order]
    payload: dict[str, object] = {
        "affected_layers": list(result.affected_layers),
        "affected_samples": list(result.affected_samples),
        "affected_slices": list(result.affected_slices),
        "config": source.evidence.to_dict(),
        "control_outcomes": dict(result.control_outcomes),
        "declared_slice_ids": list(source.declared_slice_ids),
        "earliest_layer": result.earliest_layer,
        "family_id": result.family_id,
        "global_score": None if source.global_score is None else float(source.global_score),
        "global_score_ignored": True,
        "layer_cells_in_declared_order": [
            {"layer_id": layer_id, "metric_value": value}
            for layer_id, value in zip(source.layer_order, ordered_cells, strict=True)
        ],
        "layer_order": list(source.layer_order),
        "manifest_id": source.manifest_id,
        "metric_id": result.metric_id,
        "reason": result.reason,
        "report_localization": [dict(item) for item in result.report_localization],
        "representation_identity": result.representation_identity,
        "sample_cells_by_sample_id": [
            {"metric_value": float(by_sample), "sample_id": sample_id}
            for sample_id, by_sample in sorted(
                {cell.sample_id: float(cell.metric_value) for cell in source.sample_cells}.items()
            )
        ],
        "slice_definitions": [item.to_dict() for item in source.slices],
        "slice_means": {slice_id: float(mean) for slice_id, mean in sorted(_slice_means(source).items())},
        "verdict": result.verdict,
    }
    try:
        canonical_json(payload)
    except PortableNodeError as exc:
        raise LocalizationError(f"localization payload is not canonical JSON: {exc}") from exc
    return payload


def evaluate_localization(source: LocalizationInput) -> tuple[LocalizationResult, dict[str, object]]:
    """Decide one localization input once and return the decision plus its payload."""
    result = localize_findings(source)
    return result, localization_payload(result, source)


def make_localize_executor(source: LocalizationInput, *, version: str = _LOCALIZER_VERSION) -> Any:
    """Build a supplied ``localize``-stage executor bound to one localization input.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable verifies the invocation targets ``localize`` with the expected
    request/manifest identity, evaluates :func:`evaluate_localization` once, and
    returns a ``completed`` (or honest ``unsupported``) :class:`StageOutput`.
    No localization algorithm enters ``DiagnosticWorkflow`` itself.
    """
    if not isinstance(version, str) or not version.strip():
        raise LocalizationError("version must be a non-empty string")
    if not isinstance(source, LocalizationInput):
        raise LocalizationError("source must be a LocalizationInput")

    def _execute(invocation: Any) -> Any:
        from latent_anything._diagnostic_workflow import StageContractError as _ContractError
        from latent_anything._diagnostic_workflow import StageOutput as _StageOutput

        if invocation.stage != "localize":
            raise _ContractError(f"localize executor received stage {invocation.stage!r}")
        if invocation.request.manifest_id != source.manifest_id:
            raise _ContractError("localize executor manifest identity mismatch")
        result, payload = evaluate_localization(source)
        outcome = "completed" if result.verdict in ("localized", "negative") else "unsupported"
        return _StageOutput(stage="localize", outcome=outcome, payload=payload, artifact_refs=())

    _execute.localizer_version = version  # type: ignore[attr-defined]
    return _execute


AXIAL_AXES: tuple[str, ...] = ("checkpoint", "token", "time")

_AXIAL_VERSION = "axial-localizer-v1"


def manifest_axis_lookup(manifest: Mapping[str, object]) -> dict[str, dict[str, str]]:
    """Return the manifest's declared representation axes keyed by axis name.

    Read-only binding coverage for 80.14: axis applicability requires an
    explicit manifest declaration plus an explicit axis declaration in the
    localization input. Never infers order from names or input order.
    """
    if not isinstance(manifest, Mapping):
        raise LocalizationError("manifest must be a mapping")
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

    checkpoint_order: tuple[str, ...]
    checkpoint_index: Mapping[str, int]
    dataset_slice_id: str
    dataset_configuration: str
    model_identity: str
    representation_identity: str

    def __post_init__(self) -> None:
        order = _string_tuple(self.checkpoint_order, name="checkpoint_order", minimum=1)
        if len(set(order)) != len(order):
            raise LocalizationError("checkpoint_order must not contain duplicates")
        lowered = [item.casefold() for item in order]
        if len(set(lowered)) != len(lowered):
            raise LocalizationError("checkpoint_order identities are ambiguous (casefold collision)")
        object.__setattr__(self, "checkpoint_order", order)
        if not isinstance(self.checkpoint_index, Mapping):
            raise LocalizationError("checkpoint_index must be a mapping")
        declared = {str(key): self.checkpoint_index[key] for key in self.checkpoint_index}
        if set(declared) != set(order):
            raise LocalizationError("checkpoint_index must cover exactly the declared checkpoint_order")
        for key, position in declared.items():
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise LocalizationError(f"checkpoint_index[{key!r}] must be a non-negative integer")
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
            "checkpoint_index": dict(self.checkpoint_index),
            "checkpoint_order": list(self.checkpoint_order),
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
    token_order: tuple[str, ...]
    token_positions: Mapping[str, int]
    tokenization_identity: str
    preprocessing_identity: str
    representation_identity: str

    def __post_init__(self) -> None:
        _non_empty_string(self.sample_id, name="sample_id")
        _non_empty_string(self.sequence_id, name="sequence_id")
        order = _string_tuple(self.token_order, name="token_order", minimum=1)
        if len(set(order)) != len(order):
            raise LocalizationError("token_order must not contain duplicates")
        object.__setattr__(self, "token_order", order)
        if not isinstance(self.token_positions, Mapping):
            raise LocalizationError("token_positions must be a mapping")
        declared = {str(key): self.token_positions[key] for key in self.token_positions}
        if set(declared) != set(order):
            raise LocalizationError("token_positions must cover exactly the declared token_order")
        for key, position in declared.items():
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise LocalizationError(f"token_positions[{key!r}] must be a non-negative integer")
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
            "token_order": list(self.token_order),
            "token_positions": dict(self.token_positions),
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
    step_order: tuple[str, ...]
    step_index: Mapping[str, int]
    ordering: str
    representation_identity: str

    def __post_init__(self) -> None:
        _non_empty_string(self.trajectory_id, name="trajectory_id")
        order = _string_tuple(self.step_order, name="step_order", minimum=1)
        if len(set(order)) != len(order):
            raise LocalizationError("step_order must not contain duplicates")
        object.__setattr__(self, "step_order", order)
        if not isinstance(self.step_index, Mapping):
            raise LocalizationError("step_index must be a mapping")
        declared = {str(key): self.step_index[key] for key in self.step_index}
        if set(declared) != set(order):
            raise LocalizationError("step_index must cover exactly the declared step_order")
        for key, position in declared.items():
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise LocalizationError(f"step_index[{key!r}] must be a non-negative integer")
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
            "step_index": dict(self.step_index),
            "step_order": list(self.step_order),
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
class AxialAxisResult:
    """Deterministic per-axis decision: supported location, negative, or explicit absence."""

    axis: str
    status: str
    earliest: str | None
    affected: tuple[str, ...]
    reason: str
    report_rows: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.axis not in AXIAL_AXES:
            raise LocalizationError(f"unsupported axial axis: {self.axis!r}")
        if self.status not in ("localized", "negative", "not_applicable", "unsupported"):
            raise LocalizationError(f"unsupported axial status: {self.status!r}")
        object.__setattr__(self, "affected", tuple(self.affected))
        object.__setattr__(self, "report_rows", tuple(self.report_rows))
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
            "affected": list(self.affected),
            "axis": self.axis,
            "earliest": self.earliest,
            "reason": self.reason,
            "report_rows": [dict(item) for item in self.report_rows],
            "status": self.status,
        }


@dataclass(frozen=True)
class AxialLocalizationInput:
    """Declared 80.14 scope: requested axes with manifest binding and aligned cells.

    An axis is applicable only when it is requested, declared in the manifest
    representation axes, and supplied with a declaration plus full-coverage
    aligned cells. Requested-but-absent axes are explicit non-applicability,
    never empty success. Undeclared axes (outside checkpoint/token/time) reject.
    """

    manifest_id: str
    manifest_axes: Mapping[str, dict[str, str]]
    evidence: DetectionEvidence
    requested_axes: tuple[str, ...]
    checkpoint_declaration: CheckpointAxisDeclaration | None = None
    checkpoint_cells: tuple[CheckpointCell, ...] = ()
    token_declaration: TokenAxisDeclaration | None = None
    token_cells: tuple[TokenCell, ...] = ()
    time_declaration: TimeAxisDeclaration | None = None
    time_cells: tuple[TimeCell, ...] = ()
    global_score: float | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.manifest_id, name="manifest_id")
        if not isinstance(self.manifest_axes, Mapping):
            raise LocalizationError("manifest_axes must be a mapping")
        if not isinstance(self.evidence, DetectionEvidence):
            raise LocalizationError("evidence must be a DetectionEvidence")
        if self.evidence.manifest_id != self.manifest_id:
            raise LocalizationError("evidence manifest identity does not match the axial input")
        requested = _string_tuple(self.requested_axes, name="requested_axes", minimum=1)
        if len(set(requested)) != len(requested):
            raise LocalizationError("requested_axes must not contain duplicates")
        for axis in requested:
            if axis not in AXIAL_AXES:
                raise LocalizationError(f"undeclared axis {axis!r} is not localizable here")
        object.__setattr__(self, "requested_axes", requested)
        object.__setattr__(self, "checkpoint_cells", tuple(self.checkpoint_cells))
        object.__setattr__(self, "token_cells", tuple(self.token_cells))
        object.__setattr__(self, "time_cells", tuple(self.time_cells))
        for axis in requested:
            if axis not in self.manifest_axes:
                continue
            if axis == "checkpoint" and self.checkpoint_declaration is None:
                raise LocalizationError("declared checkpoint axis requires a checkpoint declaration")
            if axis == "token" and self.token_declaration is None:
                raise LocalizationError("declared token axis requires a token declaration")
            if axis == "time" and self.time_declaration is None:
                raise LocalizationError("declared time axis requires a time declaration")
        if self.checkpoint_declaration is not None and "checkpoint" not in requested:
            raise LocalizationError("checkpoint declaration without a requested checkpoint axis")
        if self.token_declaration is not None and "token" not in requested:
            raise LocalizationError("token declaration without a requested token axis")
        if self.time_declaration is not None and "time" not in requested:
            raise LocalizationError("time declaration without a requested time axis")
        if self.checkpoint_declaration is not None:
            _check_checkpoint_alignment(self)
        if self.token_declaration is not None:
            _check_token_alignment(self)
        if self.time_declaration is not None:
            _check_time_alignment(self)
        if self.global_score is not None:
            _finite_number(self.global_score, name="global_score")

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_cells": [cell.to_dict() for cell in self.checkpoint_cells],
            "checkpoint_declaration": None if self.checkpoint_declaration is None else self.checkpoint_declaration.to_dict(),
            "evidence": self.evidence.to_dict(),
            "global_score": None if self.global_score is None else float(self.global_score),
            "manifest_axes": {axis: dict(value) for axis, value in sorted(self.manifest_axes.items())},
            "manifest_id": self.manifest_id,
            "requested_axes": list(self.requested_axes),
            "time_cells": [cell.to_dict() for cell in self.time_cells],
            "time_declaration": None if self.time_declaration is None else self.time_declaration.to_dict(),
            "token_cells": [cell.to_dict() for cell in self.token_cells],
            "token_declaration": None if self.token_declaration is None else self.token_declaration.to_dict(),
        }


@dataclass(frozen=True)
class AxialLocalizationResult:
    """Aggregate 80.14 decision across requested axes with honest mixed semantics."""

    verdict: str
    axes: tuple[AxialAxisResult, ...]
    representation_identity: str
    family_id: str
    metric_id: str
    reason: str
    report_localization: tuple[Mapping[str, object], ...] = ()
    control_outcomes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verdict not in ("localized", "negative", "not_applicable", "unsupported"):
            raise LocalizationError(f"unsupported axial verdict: {self.verdict!r}")
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "report_localization", tuple(self.report_localization))
        object.__setattr__(self, "control_outcomes", dict(self.control_outcomes))
        _non_empty_string(self.representation_identity, name="representation_identity")
        _non_empty_string(self.family_id, name="family_id")
        _non_empty_string(self.metric_id, name="metric_id")
        _non_empty_string(self.reason, name="reason")
        names = [item.axis for item in self.axes]
        if len(set(names)) != len(names):
            raise LocalizationError("axial result must not repeat an axis")
        supported = [item for item in self.axes if item.status == "localized"]
        if self.verdict == "localized" and not supported:
            raise LocalizationError("localized axial verdicts require one supported axis")
        if self.verdict in ("negative", "not_applicable", "unsupported") and supported:
            raise LocalizationError(f"{self.verdict} axial verdicts must not carry supported axes")
        if self.verdict in ("negative", "not_applicable", "unsupported") and self.report_localization:
            raise LocalizationError(f"{self.verdict} axial verdicts must not produce report locations")

    def to_dict(self) -> dict[str, object]:
        return {
            "axes": [item.to_dict() for item in self.axes],
            "control_outcomes": dict(self.control_outcomes),
            "family_id": self.family_id,
            "metric_id": self.metric_id,
            "reason": self.reason,
            "report_localization": [dict(item) for item in self.report_localization],
            "representation_identity": self.representation_identity,
            "verdict": self.verdict,
        }


def _check_checkpoint_alignment(source: AxialLocalizationInput) -> None:
    declaration = cast(CheckpointAxisDeclaration, source.checkpoint_declaration)
    if declaration.representation_identity != source.evidence.representation_identity:
        raise LocalizationError("checkpoint representation identity is misaligned")
    cells = tuple(source.checkpoint_cells)
    if not cells:
        raise LocalizationError("checkpoint_cells must not be empty: a global score alone is insufficient")
    for cell in cells:
        if not isinstance(cell, CheckpointCell):
            raise LocalizationError("checkpoint_cells must hold CheckpointCell items")
        if cell.representation_identity != source.evidence.representation_identity:
            raise LocalizationError(f"checkpoint {cell.checkpoint_id!r} representation identity is misaligned")
        if cell.model_identity != declaration.model_identity:
            raise LocalizationError(f"checkpoint {cell.checkpoint_id!r} model identity is misaligned")
        if cell.dataset_slice_id != declaration.dataset_slice_id:
            raise LocalizationError(
                f"checkpoint {cell.checkpoint_id!r} dataset slice is misaligned: "
                "checkpoint comparisons require the same declared dataset slice"
            )
        if cell.dataset_configuration != declaration.dataset_configuration:
            raise LocalizationError(
                f"checkpoint {cell.checkpoint_id!r} dataset configuration is misaligned"
            )
    identities = [cell.checkpoint_id for cell in cells]
    if len(set(identities)) != len(identities):
        raise LocalizationError("checkpoint_cells must not contain duplicate checkpoint identities")
    if set(identities) != set(declaration.checkpoint_order):
        raise LocalizationError("checkpoint_cells must cover exactly the declared checkpoint_order")


def _check_token_alignment(source: AxialLocalizationInput) -> None:
    declaration = cast(TokenAxisDeclaration, source.token_declaration)
    if declaration.representation_identity != source.evidence.representation_identity:
        raise LocalizationError("token representation identity is misaligned")
    cells = tuple(source.token_cells)
    if not cells:
        raise LocalizationError("token_cells must not be empty: a global score alone is insufficient")
    for cell in cells:
        if not isinstance(cell, TokenCell):
            raise LocalizationError("token_cells must hold TokenCell items")
        if cell.representation_identity != source.evidence.representation_identity:
            raise LocalizationError(f"token {cell.token_id!r} representation identity is misaligned")
        if cell.sample_id != declaration.sample_id or cell.sequence_id != declaration.sequence_id:
            raise LocalizationError(
                f"token {cell.token_id!r} sample/sequence binding is misaligned"
            )
        if cell.tokenization_identity != declaration.tokenization_identity:
            raise LocalizationError(
                f"token {cell.token_id!r} tokenization identity drifted: refusing to localize"
            )
        if cell.preprocessing_identity != declaration.preprocessing_identity:
            raise LocalizationError(
                f"token {cell.token_id!r} preprocessing identity drifted: refusing to localize"
            )
    identities = [cell.token_id for cell in cells]
    if len(set(identities)) != len(identities):
        raise LocalizationError("token_cells must not contain duplicate token identities")
    if set(identities) != set(declaration.token_order):
        raise LocalizationError("token_cells must cover exactly the declared token_order")


def _check_time_alignment(source: AxialLocalizationInput) -> None:
    declaration = cast(TimeAxisDeclaration, source.time_declaration)
    if declaration.representation_identity != source.evidence.representation_identity:
        raise LocalizationError("time representation identity is misaligned")
    cells = tuple(source.time_cells)
    if not cells:
        raise LocalizationError("time_cells must not be empty: a global score alone is insufficient")
    for cell in cells:
        if not isinstance(cell, TimeCell):
            raise LocalizationError("time_cells must hold TimeCell items")
        if cell.representation_identity != source.evidence.representation_identity:
            raise LocalizationError(f"step {cell.step_id!r} representation identity is misaligned")
        if cell.trajectory_id != declaration.trajectory_id:
            raise LocalizationError(
                f"step {cell.step_id!r} trajectory identity is misaligned"
            )
    identities = [cell.step_id for cell in cells]
    if len(set(identities)) != len(identities):
        raise LocalizationError("time_cells must not contain duplicate step identities")
    if set(identities) != set(declaration.step_order):
        raise LocalizationError("time_cells must cover exactly the declared step_order")


def _qualified_evidence(source: AxialLocalizationInput) -> DetectionEvidence:
    evidence = source.evidence
    if evidence.outcome == "unsupported":
        raise LocalizationError("unsupported")
    if evidence.outcome != "supported" or not evidence.claim_allowed:
        raise LocalizationError("unqualified detection evidence cannot localize")
    failed = sorted(
        control_id
        for control_id in evidence.required_controls
        if evidence.control_outcomes.get(control_id) == "failed"
    )
    if failed:
        raise LocalizationError(
            f"failed required controls block localization: {', '.join(failed)}"
        )
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
    declaration = cast(CheckpointAxisDeclaration, source.checkpoint_declaration)
    by_checkpoint = {cell.checkpoint_id: float(cell.metric_value) for cell in source.checkpoint_cells}
    affected = [name for name in declaration.checkpoint_order if evidence.is_affected(by_checkpoint[name])]
    if not affected:
        return AxialAxisResult(
            axis="checkpoint", status="negative", earliest=None, affected=(),
            reason="no checkpoint meets the declared affected criterion; benign negative produces no location",
        )
    earliest = affected[0]
    if evidence.is_ambiguous(by_checkpoint[earliest]):
        raise LocalizationError(
            f"earliest checkpoint {earliest!r} sits within tolerance of the predeclared threshold"
        )
    layer_suffix = ""
    ordered_layers = sorted({cell.layer_id for cell in source.checkpoint_cells if cell.layer_id is not None})
    if ordered_layers:
        layer_suffix = f" at layer {', '.join(ordered_layers)}"
    return AxialAxisResult(
        axis="checkpoint", status="localized", earliest=earliest, affected=tuple(affected),
        reason=f"earliest affected checkpoint {earliest!r} under the declared order{layer_suffix}",
        report_rows=tuple(
            _axial_item(item_id=f"location-checkpoint-{name}", axis="checkpoint", selection=name, evidence=evidence)
            for name in affected
        ),
    )


def _localize_token(source: AxialLocalizationInput, evidence: DetectionEvidence) -> AxialAxisResult:
    declaration = cast(TokenAxisDeclaration, source.token_declaration)
    by_token = {cell.token_id: float(cell.metric_value) for cell in source.token_cells}
    affected = [name for name in declaration.token_order if evidence.is_affected(by_token[name])]
    if not affected:
        return AxialAxisResult(
            axis="token", status="negative", earliest=None, affected=(),
            reason="no token meets the declared affected criterion; benign negative produces no location",
        )
    earliest = affected[0]
    if evidence.is_ambiguous(by_token[earliest]):
        raise LocalizationError(
            f"earliest token {earliest!r} sits within tolerance of the predeclared threshold"
        )
    return AxialAxisResult(
        axis="token", status="localized", earliest=earliest, affected=tuple(affected),
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
    declaration = cast(TimeAxisDeclaration, source.time_declaration)
    by_step = {cell.step_id: float(cell.metric_value) for cell in source.time_cells}
    affected = [name for name in declaration.step_order if evidence.is_affected(by_step[name])]
    if not affected:
        return AxialAxisResult(
            axis="time", status="negative", earliest=None, affected=(),
            reason="no step meets the declared affected criterion; benign negative produces no location",
        )
    earliest = affected[0]
    if evidence.is_ambiguous(by_step[earliest]):
        raise LocalizationError(
            f"earliest step {earliest!r} sits within tolerance of the predeclared threshold"
        )
    return AxialAxisResult(
        axis="time", status="localized", earliest=earliest, affected=tuple(affected),
        reason=(
            f"earliest affected step {earliest!r} under declared ordering "
            f"{declaration.ordering!r} on trajectory {declaration.trajectory_id!r}"
        ),
        report_rows=tuple(
            _axial_item(item_id=f"location-time-{name}", axis="time", selection=name, evidence=evidence)
            for name in affected
        ),
    )


def axial_localize(source: AxialLocalizationInput) -> AxialLocalizationResult:
    """Localize already-detected findings across checkpoint, token, and time axes.

    Consumes aligned per-coordinate metric values plus control qualification;
    never recomputes a detector algorithm and never consults ``global_score``.
    Requested-but-absent axes yield explicit per-axis non-applicability while
    supported present axes are preserved; the aggregate verdict is fail-closed
    honest about the mix.
    """
    if not isinstance(source, AxialLocalizationInput):
        raise LocalizationError("source must be an AxialLocalizationInput")
    try:
        evidence = _qualified_evidence(source)
    except LocalizationError as exc:
        if str(exc) == "unsupported":
            absent = [
                AxialAxisResult(
                    axis=axis, status="unsupported", earliest=None, affected=(),
                    reason="detection evidence is unsupported; axial localization is non-applicable, not success",
                )
                for axis in source.requested_axes
            ]
            return AxialLocalizationResult(
                verdict="unsupported",
                axes=tuple(absent),
                representation_identity=source.evidence.representation_identity,
                family_id=source.evidence.family_id,
                metric_id=source.evidence.metric_id,
                reason="detection evidence is unsupported; axial localization is non-applicable, not success",
                report_localization=(),
                control_outcomes=dict(source.evidence.control_outcomes),
            )
        raise
    per_axis: list[AxialAxisResult] = []
    for axis in source.requested_axes:
        if axis not in source.manifest_axes:
            per_axis.append(
                AxialAxisResult(
                    axis=axis, status="not_applicable", earliest=None, affected=(),
                    reason=f"axis {axis!r} is absent from the manifest representation axes; explicit non-applicability, not success",
                )
            )
            continue
        if axis == "checkpoint" and source.checkpoint_declaration is None:
            per_axis.append(
                AxialAxisResult(
                    axis=axis, status="not_applicable", earliest=None, affected=(),
                    reason="checkpoint axis is manifest-declared but supplies no aligned declaration/cells; explicit non-applicability, not success",
                )
            )
            continue
        if axis == "token" and source.token_declaration is None:
            per_axis.append(
                AxialAxisResult(
                    axis=axis, status="not_applicable", earliest=None, affected=(),
                    reason="token axis is manifest-declared but supplies no aligned declaration/cells; explicit non-applicability, not success",
                )
            )
            continue
        if axis == "time" and source.time_declaration is None:
            per_axis.append(
                AxialAxisResult(
                    axis=axis, status="not_applicable", earliest=None, affected=(),
                    reason="time axis is manifest-declared but supplies no aligned declaration/cells; explicit non-applicability, not success",
                )
            )
            continue
        if axis == "checkpoint":
            per_axis.append(_localize_checkpoint(source, evidence))
        elif axis == "token":
            per_axis.append(_localize_token(source, evidence))
        else:
            per_axis.append(_localize_time(source, evidence))
    supported = [item for item in per_axis if item.status == "localized"]
    rows: list[dict[str, object]] = [dict(row) for item in supported for row in item.report_rows]
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
            reason=f"{len(supported)} axe(s) localized ({', '.join(item.axis for item in supported)}); global score ignored{suffix}",
            report_localization=tuple(rows),
            control_outcomes=dict(evidence.control_outcomes),
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
            control_outcomes=dict(evidence.control_outcomes),
        )
    return AxialLocalizationResult(
        verdict="negative",
        axes=tuple(per_axis),
        representation_identity=evidence.representation_identity,
        family_id=evidence.family_id,
        metric_id=evidence.metric_id,
        reason="no requested axis meets the declared affected criterion; benign negative produces no location",
        report_localization=(),
        control_outcomes=dict(evidence.control_outcomes),
    )


def axial_localization_payload(result: AxialLocalizationResult, source: AxialLocalizationInput) -> dict[str, object]:
    """Assemble the canonical machine-readable axial localize-stage payload."""
    if not isinstance(result, AxialLocalizationResult):
        raise LocalizationError("result must be an AxialLocalizationResult")
    if not isinstance(source, AxialLocalizationInput):
        raise LocalizationError("source must be an AxialLocalizationInput")
    if result.representation_identity != source.evidence.representation_identity:
        raise LocalizationError("result provenance does not match the axial input")
    try:
        canonical_json([result.to_dict(), source.to_dict()])
    except PortableNodeError as exc:
        raise LocalizationError(f"axial payload is not canonical JSON: {exc}") from exc
    payload: dict[str, object] = {
        "axes": [item.to_dict() for item in result.axes],
        "config": source.evidence.to_dict(),
        "control_outcomes": dict(result.control_outcomes),
        "family_id": result.family_id,
        "global_score": None if source.global_score is None else float(source.global_score),
        "global_score_ignored": True,
        "manifest_axes": {axis: dict(value) for axis, value in sorted(source.manifest_axes.items())},
        "manifest_id": source.manifest_id,
        "metric_id": result.metric_id,
        "reason": result.reason,
        "report_localization": [dict(item) for item in result.report_localization],
        "representation_identity": result.representation_identity,
        "requested_axes": list(source.requested_axes),
        "verdict": result.verdict,
    }
    try:
        canonical_json(payload)
    except PortableNodeError as exc:
        raise LocalizationError(f"axial payload is not canonical JSON: {exc}") from exc
    return payload


def evaluate_axial_localization(
    source: AxialLocalizationInput,
) -> tuple[AxialLocalizationResult, dict[str, object]]:
    """Decide one axial input once and return the decision plus its payload."""
    result = axial_localize(source)
    return result, axial_localization_payload(result, source)


def make_axial_localize_executor(source: AxialLocalizationInput, *, version: str = _AXIAL_VERSION) -> Any:
    """Build a supplied ``localize``-stage executor bound to one axial input.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable verifies the invocation targets ``localize`` with the expected
    request/manifest identity, evaluates :func:`evaluate_axial_localization`
    once, and returns a ``completed`` (or honest ``not_applicable`` /
    ``unsupported``) :class:`StageOutput`. No axis algorithm enters
    ``DiagnosticWorkflow`` itself.
    """
    if not isinstance(version, str) or not version.strip():
        raise LocalizationError("version must be a non-empty string")
    if not isinstance(source, AxialLocalizationInput):
        raise LocalizationError("source must be an AxialLocalizationInput")

    def _execute(invocation: Any) -> Any:
        from latent_anything._diagnostic_workflow import StageContractError as _ContractError
        from latent_anything._diagnostic_workflow import StageOutput as _StageOutput

        if invocation.stage != "localize":
            raise _ContractError(f"localize executor received stage {invocation.stage!r}")
        if invocation.request.manifest_id != source.manifest_id:
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
    "CheckpointCell",
    "DetectionEvidence",
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
