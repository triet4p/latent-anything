"""Density, OOD, and distribution-drift detection (Sprint 80.11).

Private detector-family adapter behind the frozen taxonomy/workflow contracts
for ``density_ood_distribution_drift``. It consumes bound reference and test
representations plus predeclared metric/control/threshold configuration and
emits machine-readable observations, taxonomy family-evidence statuses
(``evaluate_claim``), control outcomes, seeded uncertainty metadata, and honest
``supported``/``inconclusive``/``unsupported`` outcomes.

A supported claim requires all of: explicitly bound reference and test
representation identities in one shared space, declared dataset/split/revision/
preprocessing/axis provenance, calibration on held-out reference rows only, a
predeclared drift statistic clearing its threshold, the no-shift negative
control staying below threshold, and every required control passed. Any
reference/test identity mismatch or overlap, missing calibration or control, or
failed required control blocks promotion (``inconclusive``,
``claim_allowed=False``). Representations whose geometry the density convention
does not support, and any caller-declared distribution-free request, are
recorded as ``unsupported`` with reasons — never silently treated as
in-distribution. Where the manifest predeclares no promotable declaration for
this family, the detector returns honest ``unsupported`` rather than borrowing
another family's metrics.

Reused existing primitives (no new estimators):

- ``GaussianMixtureDensity`` (``GMMConfig``) is the single shared density
  convention: one fit on reference/train rows only, one ``calibrate`` on
  held-out reference rows only, and ``score`` for every evaluation batch. No
  parallel estimator exists.
- The drift statistic is the mean calibrated OOD-score shift
  (``mean(test) - mean(reference-held-out)``) under that single fitted model.
  Gap uncertainty resamples fitted test and held-out reference scores jointly
  (exactly ``repetitions`` draws under the evaluation seed).
- The shuffled null independently permutes each test feature column under the
  control seed, destroying joint feature structure; it is an optional recorded
  control that demonstrates estimator response, not a threshold gate.
- ``ReferenceTestPair.geometry`` propagates into ``GaussianMixtureDensity.fit``
  and the recorded fit/calibration provenance (euclidean and unit_norm).
- Taxonomy evidence gating reuses ``evaluate_claim``; manifest validity reuses
  ``validate_manifest``; canonical JSON reuses the shared ``canonical_json``
  contract.

Single-evaluation seam: :func:`evaluate_detection` fits the density once and
calibrates once per executor call, then scores every batch (target, negative,
shuffled null) from that one fitted model. Flag-rate and gap uncertainties
resample fitted calibrated scores without refitting through the central
``_statistical_controls`` executor (identity-derived ``evaluation`` streams,
exactly ``repetitions`` draws); the column-shuffle null runs as one central
``shuffled`` control. One shared context feeds family decisions and payload
assembly.
:func:`detect_families` and :func:`detection_payload` remain available for
focused unit checks but must not be composed naively in production paths,
since that composition would evaluate everything twice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np

from latent_anything._benchmark_manifest import (
    BenchmarkManifestValidationError,
    validate_manifest,
)
from latent_anything._diagnostic_workflow import StageInvocation, StageOutput
from latent_anything._portable_contract import PortableNodeError, canonical_json
from latent_anything._statistical_controls import run_bootstrap as _central_bootstrap
from latent_anything._statistical_controls import ControlPlan as _ControlPlan
from latent_anything._statistical_controls import ControlSpec as _ControlSpec
from latent_anything._statistical_controls import execute_plan as _execute_plan
from latent_anything._statistical_controls import failed_required as _failed_required
from latent_anything._statistical_controls import run_permutation_control as _central_control
from latent_anything._representation_taxonomy import evaluate_claim
from latent_anything.density import GaussianMixtureDensity, GMMConfig
from latent_anything.diagnostics import DiagnosticRequest
from latent_anything.latent_value import LatentValue

SUPPORTED_FAMILIES: tuple[str, ...] = ("density_ood_distribution_drift",)
"""Detector scope for this task. Exact taxonomy identifier; nothing else is evaluated."""

METRIC_FAMILY: Mapping[str, str] = MappingProxyType(
    {
        "ood-flag-rate": "density_ood_distribution_drift",
        "density-drift-gap": "density_ood_distribution_drift",
    }
)
"""Predeclared metric-to-family wiring. Unknown metrics reject."""

CALIBRATION_RULE = "heldout-reference-quantile-0.9"
"""The only calibration rule this detector may promote.

Scores are calibrated OOD quantiles from held-out reference rows only; the
flag threshold is the 0.9 quantile of those calibration scores.
"""

_DENSITY_GEOMETRIES = frozenset({"euclidean", "unit_norm"})
"""Geometries the shared density convention supports."""

_SUPPLIED_KINDS = frozenset({"counterexample", "negative"})
"""Manifest control kinds that require caller-supplied batch data.

``shuffled``/``null`` controls are derived by the detector from the test batch
under manifest seeds, so supplying batch data for them is a fail-closed error
rather than silently ignored input.
"""

ClaimOutcome = Literal["supported", "inconclusive", "unsupported"]


class DetectionError(ValueError):
    """Raised when detection input, config, or evidence is fail-closed invalid."""


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DetectionError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class FamilyThreshold:
    """One predeclared pass/fail rule, owned by the manifest, never tuned here."""

    metric_id: str
    comparator: str
    value: float
    tolerance: float

    def __post_init__(self) -> None:
        _non_empty_string(self.metric_id, name="threshold metric_id")
        if self.comparator not in (">=", "<="):
            raise DetectionError(f"unsupported threshold comparator: {self.comparator!r}")
        for name in ("value", "tolerance"):
            candidate = getattr(self, name)
            if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
                raise DetectionError(f"threshold {name} must be a finite number")
            if not np.isfinite(float(candidate)):
                raise DetectionError(f"threshold {name} must be a finite number")
        if float(self.tolerance) < 0.0:
            raise DetectionError("threshold tolerance must be non-negative")

    def passes(self, observed: float) -> bool:
        """Apply the predeclared comparator to one observed metric value."""
        if not np.isfinite(observed):
            raise DetectionError(f"metric {self.metric_id!r} produced a non-finite value")
        target = float(self.value)
        if self.comparator == ">=":
            return bool(observed >= target)
        return bool(observed <= target)


@dataclass(frozen=True)
class Provenance:
    """Declared reference/test binding: identity, dataset, and axis provenance."""

    representation_identity: str
    dataset_id: str
    dataset_revision: str
    split: str
    split_identity: str
    preprocessing: str
    axes: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty_string(self.representation_identity, name="representation_identity")
        _non_empty_string(self.dataset_id, name="dataset_id")
        _non_empty_string(self.dataset_revision, name="dataset_revision")
        _non_empty_string(self.split, name="split")
        _non_empty_string(self.split_identity, name="split_identity")
        _non_empty_string(self.preprocessing, name="preprocessing")
        if not self.axes:
            raise DetectionError("axes must not be empty")
        for index, axis in enumerate(self.axes):
            if not isinstance(axis, str) or not axis.strip():
                raise DetectionError(f"axes[{index}] must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this provenance."""
        return {
            "axes": list(self.axes),
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "preprocessing": self.preprocessing,
            "representation_identity": self.representation_identity,
            "split": self.split,
            "split_identity": self.split_identity,
        }


@dataclass(frozen=True)
class DetectionConfig:
    """Predeclared detector configuration parsed from manifest plus request."""

    manifest_id: str
    family_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    thresholds: tuple[FamilyThreshold, ...]
    required_controls: tuple[str, ...]
    optional_controls: tuple[str, ...]
    control_metrics: Mapping[str, tuple[str, ...]]
    control_kinds: Mapping[str, str]
    calibration_rule: str
    repetitions: int
    confidence_level: float
    training_seed: int
    evaluation_seed: int
    control_seed: int
    manifest_metric_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _non_empty_string(self.manifest_id, name="manifest_id")
        if not self.family_ids or set(self.family_ids) - set(SUPPORTED_FAMILIES):
            raise DetectionError(f"family_ids must be a non-empty subset of {list(SUPPORTED_FAMILIES)!r}")
        if len(set(self.family_ids)) != len(self.family_ids):
            raise DetectionError("family_ids must not contain duplicates")
        if not self.metric_ids:
            raise DetectionError("metric_ids must not be empty")
        wired = {threshold.metric_id for threshold in self.thresholds}
        if set(self.metric_ids) != wired:
            raise DetectionError("every metric must have exactly one predeclared threshold")
        for metric_id in self.metric_ids:
            if metric_id not in METRIC_FAMILY:
                raise DetectionError(f"metric {metric_id!r} is not wired to a supported family")
        if not self.required_controls:
            raise DetectionError("at least one required control must be declared")
        if len(set((*self.required_controls, *self.optional_controls))) != len(self.required_controls) + len(
            self.optional_controls
        ):
            raise DetectionError("control identifiers must be unique")
        for control_id, metric_ids in self.control_metrics.items():
            if control_id not in (*self.required_controls, *self.optional_controls):
                raise DetectionError(f"control {control_id!r} is not a declared control")
            if not metric_ids or set(metric_ids) - set(self.metric_ids):
                raise DetectionError(f"control {control_id!r} must link declared metrics")
        for control_id in (*self.required_controls, *self.optional_controls):
            if control_id not in self.control_kinds:
                raise DetectionError(f"control {control_id!r} is missing its manifest kind")
        object.__setattr__(self, "manifest_metric_families", tuple(self.manifest_metric_families))
        _non_empty_string(self.calibration_rule, name="calibration_rule")
        if self.calibration_rule != CALIBRATION_RULE:
            raise DetectionError(f"unsupported calibration rule: {self.calibration_rule!r}")
        if self.repetitions < 2:
            raise DetectionError("repetitions must be at least two")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise DetectionError("confidence_level must be between zero and one")
        for name in ("training_seed", "evaluation_seed", "control_seed"):
            candidate = getattr(self, name)
            if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 0:
                raise DetectionError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this configuration."""
        return {
            "calibration_rule": self.calibration_rule,
            "confidence_level": float(self.confidence_level),
            "control_metrics": {key: list(value) for key, value in self.control_metrics.items()},
            "control_kinds": dict(self.control_kinds),
            "manifest_metric_families": list(self.manifest_metric_families),
            "control_seed": int(self.control_seed),
            "evaluation_seed": int(self.evaluation_seed),
            "family_ids": list(self.family_ids),
            "manifest_id": self.manifest_id,
            "metric_ids": list(self.metric_ids),
            "optional_controls": list(self.optional_controls),
            "repetitions": int(self.repetitions),
            "required_controls": list(self.required_controls),
            "thresholds": [
                {
                    "comparator": item.comparator,
                    "metric_id": item.metric_id,
                    "tolerance": float(item.tolerance),
                    "value": float(item.value),
                }
                for item in self.thresholds
            ],
            "training_seed": int(self.training_seed),
        }


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DetectionError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise DetectionError(f"{name} must be a finite number")
    return float(value)


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DetectionError(f"{name} must be a non-negative integer")
    return value


def detection_config_from_manifest(
    request: DiagnosticRequest,
    manifest: Mapping[str, object],
    *,
    calibration_rule: str = CALIBRATION_RULE,
) -> DetectionConfig:
    """Parse the predeclared detector configuration; fail closed on any gap."""
    if not isinstance(request, DiagnosticRequest):
        raise DetectionError("request must be a DiagnosticRequest")
    if not isinstance(manifest, Mapping):
        raise DetectionError("manifest must be a mapping")
    try:
        validate_manifest(manifest)
    except BenchmarkManifestValidationError as exc:
        raise DetectionError(f"invalid benchmark manifest: {exc}") from exc
    if manifest.get("manifest_id") != request.manifest_id:
        raise DetectionError("request manifest_id does not match the manifest")
    manifest_id = cast(str, manifest.get("manifest_id"))

    family_ids = tuple(request.diagnostics.family_ids)
    for family_id in family_ids:
        if family_id not in SUPPORTED_FAMILIES:
            raise DetectionError(f"family {family_id!r} is not supported by this detector")

    raw_metrics = manifest.get("metrics")
    if isinstance(raw_metrics, (str, bytes)) or not isinstance(raw_metrics, Sequence):
        raise DetectionError("manifest metrics must be a list")
    manifest_metric_families: list[str] = []
    for index, raw in enumerate(raw_metrics):
        item = _mapping(raw, name=f"metrics[{index}]")
        manifest_metric_families.append(_non_empty_string(item.get("taxonomy_family_id"), name=f"metrics[{index}].taxonomy_family_id"))
    raw_thresholds = manifest.get("thresholds")
    if isinstance(raw_thresholds, (str, bytes)) or not isinstance(raw_thresholds, Sequence):
        raise DetectionError("manifest thresholds must be a list")
    thresholds: list[FamilyThreshold] = []
    for index, raw in enumerate(raw_thresholds):
        item = _mapping(raw, name=f"thresholds[{index}]")
        thresholds.append(
            FamilyThreshold(
                metric_id=_non_empty_string(item.get("metric_id"), name=f"thresholds[{index}].metric_id"),
                comparator=_non_empty_string(item.get("comparator"), name=f"thresholds[{index}].comparator"),
                value=_finite_number(item.get("value"), name=f"thresholds[{index}].value"),
                tolerance=_finite_number(item.get("tolerance"), name=f"thresholds[{index}].tolerance"),
            )
        )
    if len({item.metric_id for item in thresholds}) != len(thresholds):
        raise DetectionError("every metric must have exactly one predeclared threshold")

    raw_controls = manifest.get("controls")
    if isinstance(raw_controls, (str, bytes)) or not isinstance(raw_controls, Sequence) or not raw_controls:
        raise DetectionError("manifest controls must be a non-empty list")
    manifest_controls: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(raw_controls):
        item = _mapping(raw, name=f"controls[{index}]")
        control_id = _non_empty_string(item.get("id"), name=f"controls[{index}].id")
        if control_id in manifest_controls:
            raise DetectionError(f"duplicate manifest control: {control_id!r}")
        metric_ids = item.get("metric_ids")
        if isinstance(metric_ids, (str, bytes)) or not isinstance(metric_ids, Sequence) or not metric_ids:
            raise DetectionError(f"controls[{index}].metric_ids must be a non-empty list")
        manifest_controls[control_id] = {
            "metric_ids": tuple(metric_ids),
            "required": item.get("required") is True,
            "kind": _non_empty_string(item.get("kind"), name=f"controls[{index}].kind"),
        }
    for control_id in request.controls.control_ids:
        if control_id not in manifest_controls:
            raise DetectionError(f"request control {control_id!r} is not declared by the manifest")
    required = tuple(control_id for control_id, item in manifest_controls.items() if item["required"] is True)
    missing_required = [control_id for control_id in required if control_id not in request.controls.control_ids]
    if missing_required:
        raise DetectionError(f"required controls are missing from the request: {', '.join(missing_required)}")
    for control_id in required:
        linked = cast(tuple[str, ...], manifest_controls[control_id]["metric_ids"])
        if set(linked) - set(request.controls.metric_ids):
            raise DetectionError(f"required control {control_id!r} links undeclared metrics")

    uncertainty = _mapping(manifest.get("uncertainty"), name="uncertainty")
    repetitions = uncertainty.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise DetectionError("uncertainty.repetitions must be at least two")
    seeds = _mapping(manifest.get("seeds"), name="seeds")
    training = seeds.get("training")
    evaluation = seeds.get("evaluation")
    controls_seed = seeds.get("controls")
    for field_name, field_value in (("training", training), ("evaluation", evaluation), ("controls", controls_seed)):
        if isinstance(field_value, (str, bytes)) or not isinstance(field_value, Sequence) or not field_value:
            raise DetectionError(f"seeds.{field_name} must be a non-empty list")
    return DetectionConfig(
        manifest_id=manifest_id,
        family_ids=family_ids,
        metric_ids=tuple(request.controls.metric_ids),
        thresholds=tuple(thresholds),
        required_controls=tuple(control_id for control_id in manifest_controls if manifest_controls[control_id]["required"] is True),
        optional_controls=tuple(
            control_id for control_id in manifest_controls if manifest_controls[control_id]["required"] is not True
        ),
        control_metrics={
            control_id: tuple(str(item) for item in cast(Sequence[object], manifest_controls[control_id]["metric_ids"]))
            for control_id in manifest_controls
        },
        control_kinds={control_id: str(manifest_controls[control_id]["kind"]) for control_id in manifest_controls},
        manifest_metric_families=tuple(manifest_metric_families),
        calibration_rule=calibration_rule,
        repetitions=int(repetitions),
        confidence_level=_finite_number(uncertainty.get("confidence_level"), name="uncertainty.confidence_level"),
        training_seed=_non_negative_int(training[0], name="seeds.training[0]"),
        evaluation_seed=_non_negative_int(evaluation[0], name="seeds.evaluation[0]"),
        control_seed=_non_negative_int(controls_seed[0], name="seeds.controls[0]"),
    )


@dataclass(frozen=True)
class FamilyDetection:
    """Evidence-bound outcome for one taxonomy family on one reference/test pair."""

    family_id: str
    outcome: ClaimOutcome
    claim_allowed: bool
    observed_metrics: Mapping[str, float]
    threshold_pass: Mapping[str, bool]
    control_outcomes: Mapping[str, str]
    evidence_status: Mapping[str, str]
    missing_evidence: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.family_id not in SUPPORTED_FAMILIES:
            raise DetectionError(f"unsupported family: {self.family_id!r}")
        if self.outcome not in ("supported", "inconclusive", "unsupported"):
            raise DetectionError(f"unsupported claim outcome: {self.outcome!r}")
        if not isinstance(self.claim_allowed, bool):
            raise DetectionError("claim_allowed must be boolean")
        object.__setattr__(self, "observed_metrics", MappingProxyType(dict(self.observed_metrics)))
        object.__setattr__(self, "threshold_pass", MappingProxyType(dict(self.threshold_pass)))
        object.__setattr__(self, "control_outcomes", MappingProxyType(dict(self.control_outcomes)))
        object.__setattr__(self, "evidence_status", MappingProxyType(dict(self.evidence_status)))
        object.__setattr__(self, "missing_evidence", tuple(self.missing_evidence))
        _non_empty_string(self.reason, name="reason")
        if self.outcome == "supported" and (not self.claim_allowed or self.missing_evidence):
            raise DetectionError("supported outcomes must allow the claim and admit no missing evidence")
        if self.outcome == "unsupported" and self.claim_allowed:
            raise DetectionError("unsupported outcomes must not allow the claim")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this family outcome."""
        return {
            "claim_allowed": self.claim_allowed,
            "control_outcomes": dict(self.control_outcomes),
            "evidence_status": dict(self.evidence_status),
            "family_id": self.family_id,
            "missing_evidence": list(self.missing_evidence),
            "observed_metrics": dict(self.observed_metrics),
            "outcome": self.outcome,
            "reason": self.reason,
            "threshold_pass": dict(self.threshold_pass),
        }


@dataclass(frozen=True)
class ReferenceTestPair:
    """Bound reference/test representations with explicit identity and provenance.

    Reference rows fit the density; calibration rows (held-out reference rows)
    calibrate it; test rows are evaluated without leakage. Reference and test
    share one representation identity, geometry, feature width, dataset
    revision, and preprocessing, with distinct split identities and no shared
    sample identity. ``distribution_free=True`` declares a nonparametric
    request the density convention cannot support: construction succeeds but
    evaluation returns honest ``unsupported``.
    """

    reference: LatentValue
    calibration: LatentValue
    test: LatentValue
    reference_sample_ids: tuple[str, ...]
    calibration_sample_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]
    reference_provenance: Provenance
    test_provenance: Provenance
    distribution_free: bool = False

    def __post_init__(self) -> None:
        for name in ("reference", "calibration", "test"):
            if not isinstance(getattr(self, name), LatentValue):
                raise DetectionError(f"{name} must be a LatentValue")
        for name in ("reference_provenance", "test_provenance"):
            if not isinstance(getattr(self, name), Provenance):
                raise DetectionError(f"{name} must be a Provenance")
        if not isinstance(self.distribution_free, bool):
            raise DetectionError("distribution_free must be boolean")
        reference_matrix = _batch_matrix(self.reference, name="reference")
        calibration_matrix = _batch_matrix(self.calibration, name="calibration")
        test_matrix = _batch_matrix(self.test, name="test")
        widths = {int(reference_matrix.shape[1]), int(calibration_matrix.shape[1]), int(test_matrix.shape[1])}
        if len(widths) != 1:
            raise DetectionError(
                f"reference/calibration/test feature widths diverge: {sorted(widths)}"
            )
        reference_space = self.reference.space
        for name, value in (("calibration", self.calibration), ("test", self.test)):
            other = value.space
            if other.geometry != reference_space.geometry:
                raise DetectionError(
                    f"{name} geometry {other.geometry!r} does not match reference geometry {reference_space.geometry!r}"
                )
        geometries = {reference_space.geometry}
        if len(geometries) != 1:
            raise DetectionError("reference/test geometries diverge")
        geometry = reference_space.geometry
        if geometry not in _DENSITY_GEOMETRIES:
            raise DetectionError(f"unsupported geometry for density estimation: {geometry!r}")
        for name, value in (
            ("reference_sample_ids", self.reference_sample_ids),
            ("calibration_sample_ids", self.calibration_sample_ids),
            ("test_sample_ids", self.test_sample_ids),
        ):
            items = tuple(value)
            object.__setattr__(self, name, items)
            expected = int({"reference_sample_ids": reference_matrix.shape[0], "calibration_sample_ids": calibration_matrix.shape[0], "test_sample_ids": test_matrix.shape[0]}[name])
            if len(items) != expected:
                raise DetectionError(f"{name} cover {len(items)} samples but the batch holds {expected}")
            for position, sample_id in enumerate(items):
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise DetectionError(f"{name}[{position}] must be a non-empty string")
            if len(set(items)) != len(items):
                raise DetectionError(f"{name} must be unique within one batch")
        if set(self.reference_sample_ids) & set(self.calibration_sample_ids):
            raise DetectionError("reference/calibration overlap: a sample identity appears on both sides")
        if set(self.reference_sample_ids) & set(self.test_sample_ids):
            raise DetectionError("reference/test overlap: a sample identity appears on both sides")
        if set(self.calibration_sample_ids) & set(self.test_sample_ids):
            raise DetectionError("calibration/test overlap: a sample identity appears on both sides")
        reference_identity = self.reference.identity
        for name, value in (("calibration", self.calibration), ("test", self.test)):
            if value.identity != reference_identity:
                raise DetectionError(
                    f"{name} representation identity does not match the reference identity"
                )
        if not reference_identity:
            raise DetectionError("bound values must declare a representation identity")
        if self.reference_provenance.representation_identity != self.test_provenance.representation_identity:
            raise DetectionError("reference/test representation identities must match")
        if self.reference_provenance.split_identity == self.test_provenance.split_identity:
            raise DetectionError("reference/test split identities must be distinct")
        for field in ("dataset_id", "dataset_revision", "preprocessing"):
            if getattr(self.reference_provenance, field) != getattr(self.test_provenance, field):
                raise DetectionError(f"reference/test {field} must match")
        if tuple(self.reference_provenance.axes) != tuple(self.test_provenance.axes):
            raise DetectionError("reference/test axes must match")


def _batch_matrix(value: LatentValue, *, name: str = "batch") -> np.ndarray:
    data = np.asarray(value.to_numpy(), dtype=np.float64)
    if data.ndim != 2:
        raise DetectionError(f"incompatible rank: {name} requires one 2D (n_samples, dim) batch, got {data.ndim}D")
    if data.shape[0] < 2 or data.shape[1] < 1:
        raise DetectionError(f"undersampled batch: {name} requires at least two samples, got shape {data.shape}")
    if not np.isfinite(data).all():
        raise DetectionError(f"{name} contains non-finite values")
    return data


def _summarize(
    samples: Sequence[float], *, repetitions: int, seed: int, confidence_level: float
) -> dict[str, object]:
    """Summarize draws through the central statistical-control executor.

    Strict seam: the percentile math lives in
    ``_statistical_controls.summarize_interval``; this local alias keeps the
    detector's one-fit/one-calibration call sites unchanged while the central
    executor owns the interval contract.
    """
    from latent_anything._statistical_controls import summarize_interval as _summarize_central

    return dict(
        _summarize_central(
            samples, repetitions=repetitions, seed=seed, confidence_level=confidence_level
        )
    )


def _flag_rate(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(scores, dtype=np.float64) >= threshold))


def _bootstrap_scores(
    scores: np.ndarray, threshold: float, *, repetitions: int, seed: int, confidence_level: float
) -> dict[str, object]:
    """Resample fitted calibrated scores without refitting.

    Central schedule: draws run under the executor's identity-derived stream
    (``bootstrap:ood-flag-rate``); the fitted scores are never refit.
    """
    flat = np.asarray(scores, dtype=np.float64).ravel()
    n = int(flat.shape[0])

    def _draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, n, size=n)
        return float(np.mean(flat[positions] >= threshold))

    _, interval = _central_bootstrap(
        control_id="bootstrap:ood-flag-rate",
        base_seed=seed,
        seed_role="evaluation",
        repetitions=repetitions,
        confidence_level=confidence_level,
        required=False,
        kind="bootstrap",
        metric_ids=("ood-flag-rate",),
        expected_behavior="seeded resampling of fitted calibrated scores",
        draw=_draw,
    )
    return dict(interval)


@dataclass(frozen=True)
class _DensityEvaluation:
    """One reference/test pair evaluated once: one density fit, one calibration."""

    n_reference: int
    n_calibration: int
    n_test: int
    dim: int
    geometry: str
    flag_threshold: float
    test_flag_rate: float
    negative_flag_rate: float
    shuffled_flag_rate: float
    drift_gap: float
    negative_drift_gap: float
    shuffled_drift_gap: float
    test_scores: tuple[float, ...]
    reference_scores: tuple[float, ...]
    negative_scores: tuple[float, ...]
    shuffled_scores: tuple[float, ...]
    estimator_digest: str
    uncertainty: Mapping[str, dict[str, object]]


@dataclass(frozen=True)
class _DetectionContext:
    """One executor call evaluated exactly once and shared by decisions and payload."""

    representation_identity: str
    geometry: str
    evaluation: _DensityEvaluation | None
    null_evaluation: _DensityEvaluation | None
    counterexample_evaluation: _DensityEvaluation | None
    control_outcomes: Mapping[str, str]
    central_outcomes: Mapping[str, object]
    distribution_free: bool
    unsupported_geometry: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_outcomes", dict(self.control_outcomes))
        object.__setattr__(self, "central_outcomes", dict(self.central_outcomes))


def _fit_once(
    reference_matrix: np.ndarray,
    calibration_matrix: np.ndarray,
    identity: str,
    config: DetectionConfig,
    *,
    geometry: str,
    reference_provenance: Provenance,
    test_provenance: Provenance,
) -> tuple[GaussianMixtureDensity, float]:
    """Fit one density on reference rows and calibrate on held-out reference rows."""
    if geometry not in _DENSITY_GEOMETRIES:
        raise DetectionError(f"unsupported geometry for density estimation: {geometry!r}")
    estimator = GaussianMixtureDensity(
        GMMConfig(n_components=2, random_state=config.training_seed)
    )
    try:
        estimator.fit(
            reference_matrix,
            source_representation_identity=identity,
            geometry=geometry,
            provenance={**reference_provenance.to_dict(), "role": "reference-fit"},
        )
    except ValueError as exc:
        raise DetectionError(f"density fit failed: {exc}") from exc
    try:
        estimator.calibrate(
            calibration_matrix,
            provenance={**reference_provenance.to_dict(), "role": "heldout-calibration"},
        )
    except ValueError as exc:
        raise DetectionError(f"density calibration failed: {exc}") from exc
    calibration_scores = np.asarray(estimator.score(calibration_matrix).calibrated_ood_score, dtype=np.float64)
    threshold = float(np.quantile(calibration_scores, 0.9))
    if not np.isfinite(threshold):
        raise DetectionError("calibration produced a non-finite flag threshold")
    return estimator, threshold


def _bootstrap_gap(
    test_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    """Resample fitted calibrated scores and summarize the resampled mean gaps.

    Central schedule: exactly ``repetitions`` draws under the executor's
    identity-derived stream (``bootstrap:density-drift-gap``); each draw
    resamples the fitted test scores and the fitted held-out reference scores
    with replacement and records ``mean(test draw) - mean(reference draw)``.
    """
    flat_test = np.asarray(test_scores, dtype=np.float64).ravel()
    flat_reference = np.asarray(reference_scores, dtype=np.float64).ravel()
    n_test = int(flat_test.shape[0])
    n_reference = int(flat_reference.shape[0])

    def _draw(rng: np.random.Generator) -> float:
        test_positions = rng.integers(0, n_test, size=n_test)
        reference_positions = rng.integers(0, n_reference, size=n_reference)
        return float(np.mean(flat_test[test_positions]) - np.mean(flat_reference[reference_positions]))

    _, interval = _central_bootstrap(
        control_id="bootstrap:density-drift-gap",
        base_seed=seed,
        seed_role="evaluation",
        repetitions=repetitions,
        confidence_level=confidence_level,
        required=False,
        kind="bootstrap",
        metric_ids=("density-drift-gap",),
        expected_behavior="seeded resampling of fitted calibrated scores",
        draw=_draw,
    )
    return dict(interval)


def _score_with(
    estimator: GaussianMixtureDensity,
    matrix: np.ndarray,
    threshold: float,
    reference_scores: np.ndarray,
    config: DetectionConfig,
) -> tuple[float, float, tuple[float, ...], dict[str, dict[str, object]]]:
    """Score one batch with the shared fitted model; uncertainty resamples scores only."""
    scores = np.asarray(estimator.score(matrix).calibrated_ood_score, dtype=np.float64)
    reference_mean = float(np.mean(np.asarray(reference_scores, dtype=np.float64)))
    flag = _flag_rate(scores, threshold)
    gap = float(float(np.mean(scores)) - reference_mean)
    uncertainty = {
        "ood-flag-rate": _bootstrap_scores(
            scores, threshold, repetitions=config.repetitions, seed=config.evaluation_seed, confidence_level=config.confidence_level
        ),
        "density-drift-gap": _bootstrap_gap(
            scores,
            reference_scores,
            repetitions=config.repetitions,
            seed=config.evaluation_seed,
            confidence_level=config.confidence_level,
        ),
    }
    return flag, gap, tuple(float(item) for item in scores.ravel()), uncertainty


def _evaluate_context(
    pair: ReferenceTestPair | LatentValue,
    config: DetectionConfig,
    supplied: Mapping[str, ReferenceTestPair | LatentValue],
) -> _DetectionContext:
    """Validate and evaluate the pair plus every control batch exactly once."""
    for control_id in supplied:
        if control_id not in (*config.required_controls, *config.optional_controls):
            raise DetectionError(f"unknown control reference: {control_id!r}")
        kind = config.control_kinds[control_id]
        if kind not in _SUPPLIED_KINDS:
            raise DetectionError(f"control {control_id!r} is derived by the detector; no batch data is accepted")
    for control_id in config.required_controls:
        if config.control_kinds[control_id] in _SUPPLIED_KINDS and control_id not in supplied:
            raise DetectionError(f"required controls are missing batch data: {control_id}")

    wants_family = "density_ood_distribution_drift" in config.family_ids
    declared_metrics = [metric_id for metric_id in config.metric_ids if METRIC_FAMILY.get(metric_id) == "density_ood_distribution_drift"]
    manifest_declared = [family for family in getattr(config, "manifest_metric_families", ()) if isinstance(family, str)]
    manifest_covers = (
        "density_ood_distribution_drift" in manifest_declared
        and set(declared_metrics) == set(config.metric_ids)
    )
    if wants_family and (not declared_metrics or (manifest_declared and not manifest_covers)):
        # Honest missing-declaration path: no promotable metrics predeclared.
        identity = "" if isinstance(pair, LatentValue) else pair.reference.identity
        geometry_name = "" if isinstance(pair, LatentValue) else pair.reference.space.geometry
        return _DetectionContext(
            representation_identity=identity,
            geometry=geometry_name,
            evaluation=None,
            null_evaluation=None,
            counterexample_evaluation=None,
            control_outcomes={},
            central_outcomes={},
            distribution_free=isinstance(pair, ReferenceTestPair) and pair.distribution_free,
            unsupported_geometry=None,
        )
    if isinstance(pair, LatentValue):
        raise DetectionError("density evaluation requires a bound reference/test pair with declared provenance")
    if not isinstance(pair, ReferenceTestPair):
        raise DetectionError("pair must be a ReferenceTestPair")
    geometry = pair.reference.space.geometry
    if geometry not in _DENSITY_GEOMETRIES:
        return _DetectionContext(
            representation_identity=pair.reference.identity,
            geometry=geometry,
            evaluation=None,
            null_evaluation=None,
            counterexample_evaluation=None,
            control_outcomes={},
            central_outcomes={},
            distribution_free=pair.distribution_free,
            unsupported_geometry=geometry,
        )
    if pair.distribution_free:
        return _DetectionContext(
            representation_identity=pair.reference.identity,
            geometry=geometry,
            evaluation=None,
            null_evaluation=None,
            counterexample_evaluation=None,
            control_outcomes={},
            central_outcomes={},
            distribution_free=True,
            unsupported_geometry=None,
        )

    reference_matrix = _batch_matrix(pair.reference, name="reference")
    calibration_matrix = _batch_matrix(pair.calibration, name="calibration")
    test_matrix = _batch_matrix(pair.test, name="test")
    identity = pair.reference.identity
    estimator, threshold = _fit_once(
        reference_matrix,
        calibration_matrix,
        identity,
        config,
        geometry=geometry,
        reference_provenance=pair.reference_provenance,
        test_provenance=pair.test_provenance,
    )
    reference_scores = np.asarray(estimator.score(calibration_matrix).calibrated_ood_score, dtype=np.float64)
    reference_mean = float(np.mean(reference_scores))

    test_scores_raw = np.asarray(estimator.score(test_matrix).calibrated_ood_score, dtype=np.float64)
    test_flag = _flag_rate(test_scores_raw, threshold)
    gap = float(float(np.mean(test_scores_raw)) - reference_mean)
    evaluation = _DensityEvaluation(
        n_reference=int(reference_matrix.shape[0]),
        n_calibration=int(calibration_matrix.shape[0]),
        n_test=int(test_matrix.shape[0]),
        dim=int(test_matrix.shape[1]),
        geometry=geometry,
        flag_threshold=threshold,
        test_flag_rate=test_flag,
        negative_flag_rate=test_flag,
        shuffled_flag_rate=test_flag,
        drift_gap=gap,
        negative_drift_gap=gap,
        shuffled_drift_gap=gap,
        test_scores=tuple(float(item) for item in test_scores_raw.ravel()),
        reference_scores=tuple(float(item) for item in reference_scores.ravel()),
        negative_scores=tuple(float(item) for item in test_scores_raw.ravel()),
        shuffled_scores=tuple(float(item) for item in test_scores_raw.ravel()),
        estimator_digest=estimator.state_digest(),
        uncertainty={
            "ood-flag-rate": _bootstrap_scores(
                test_scores_raw, threshold, repetitions=config.repetitions, seed=config.evaluation_seed, confidence_level=config.confidence_level
            ),
            "density-drift-gap": _bootstrap_gap(
                test_scores_raw,
                reference_scores,
                repetitions=config.repetitions,
                seed=config.evaluation_seed,
                confidence_level=config.confidence_level,
            ),
        },
    )

    # Shuffled null: the actual column permutation plus scoring through the
    # already-fitted/calibrated shared model runs inside the
    # executor-supplied stream. Observed flag/gap bind from the outcome.
    def _null_statistic(rng: np.random.Generator) -> Mapping[str, object]:
        shuffled = np.column_stack(
            [rng.permutation(test_matrix[:, j]) for j in range(test_matrix.shape[1])]
        )
        scores = np.asarray(estimator.score(shuffled).calibrated_ood_score, dtype=np.float64)
        reference_mean = float(np.mean(np.asarray(reference_scores, dtype=np.float64)))
        return {
            "ood-flag-rate": float(np.mean(scores >= threshold)),
            "density-drift-gap": float(float(np.mean(scores)) - reference_mean),
            "shuffled_scores": [float(item) for item in scores.ravel()],
        }

    # The shuffled null executes inside ``execute_plan`` below (single
    # central schedule); placeholder scores are replaced once the plan
    # returns the real ControlOutcome. Null uncertainty resamples those
    # fitted scores without refitting.
    null_evaluation = _DensityEvaluation(
        n_reference=evaluation.n_reference,
        n_calibration=evaluation.n_calibration,
        n_test=evaluation.n_test,
        dim=evaluation.dim,
        geometry=geometry,
        flag_threshold=threshold,
        test_flag_rate=evaluation.test_flag_rate,
        negative_flag_rate=evaluation.negative_flag_rate,
        shuffled_flag_rate=evaluation.test_flag_rate,
        drift_gap=evaluation.drift_gap,
        negative_drift_gap=evaluation.negative_flag_rate,
        shuffled_drift_gap=evaluation.drift_gap,
        test_scores=evaluation.test_scores,
        reference_scores=evaluation.reference_scores,
        negative_scores=evaluation.negative_scores,
        shuffled_scores=evaluation.test_scores,
        estimator_digest=evaluation.estimator_digest,
        uncertainty=evaluation.uncertainty,
    )

    # No-shift negative: caller-supplied pair evaluated with the shared model.
    negative_flag = evaluation.test_flag_rate
    negative_gap = evaluation.drift_gap
    negative_scores = evaluation.test_scores
    negative_uncertainty = evaluation.uncertainty
    counterexample_evaluation: _DensityEvaluation | None = None
    negative_id: str | None = None
    for control_id in (*config.required_controls, *config.optional_controls):
        if config.control_kinds[control_id] in _SUPPLIED_KINDS and set(config.control_metrics[control_id]) & set(declared_metrics):
            negative_id = control_id
            break
    if negative_id is not None:
        negative_pair = supplied[negative_id]
        if not isinstance(negative_pair, ReferenceTestPair):
            raise DetectionError(f"control {negative_id!r} must carry a bound reference/test pair")
        if negative_pair.distribution_free:
            raise DetectionError(f"control {negative_id!r} must not declare distribution-free estimation")
        negative_test = _batch_matrix(negative_pair.test, name=f"control {negative_id!r} test")
        if negative_test.shape[1] != test_matrix.shape[1]:
            raise DetectionError(
                f"control {negative_id!r} has {negative_test.shape[1]} features but the target has {test_matrix.shape[1]}"
            )
        if negative_pair.test.identity != identity:
            raise DetectionError(f"control {negative_id!r} representation identity does not match the reference identity")
        negative_flag, negative_gap, negative_scores, negative_uncertainty = _score_with(
            estimator, negative_test, threshold, reference_scores, config
        )
        counterexample_evaluation = _DensityEvaluation(
            n_reference=evaluation.n_reference,
            n_calibration=evaluation.n_calibration,
            n_test=int(negative_test.shape[0]),
            dim=evaluation.dim,
            geometry=geometry,
            flag_threshold=threshold,
            test_flag_rate=evaluation.test_flag_rate,
            negative_flag_rate=negative_flag,
            shuffled_flag_rate=evaluation.shuffled_flag_rate,
            drift_gap=evaluation.drift_gap,
            negative_drift_gap=negative_gap,
            shuffled_drift_gap=evaluation.shuffled_drift_gap,
            test_scores=evaluation.test_scores,
            reference_scores=evaluation.reference_scores,
            negative_scores=negative_scores,
            shuffled_scores=evaluation.shuffled_scores,
            estimator_digest=evaluation.estimator_digest,
            uncertainty=negative_uncertainty,
        )
        evaluation = _DensityEvaluation(
            n_reference=evaluation.n_reference,
            n_calibration=evaluation.n_calibration,
            n_test=evaluation.n_test,
            dim=evaluation.dim,
            geometry=evaluation.geometry,
            flag_threshold=evaluation.flag_threshold,
            test_flag_rate=evaluation.test_flag_rate,
            negative_flag_rate=negative_flag,
            shuffled_flag_rate=evaluation.shuffled_flag_rate,
            drift_gap=evaluation.drift_gap,
            negative_drift_gap=negative_gap,
            shuffled_drift_gap=evaluation.shuffled_drift_gap,
            test_scores=evaluation.test_scores,
            reference_scores=evaluation.reference_scores,
            negative_scores=negative_scores,
            shuffled_scores=evaluation.shuffled_scores,
            estimator_digest=evaluation.estimator_digest,
            uncertainty=evaluation.uncertainty,
        )
    # Central status/gating: every declared control is evaluated by
    # ``execute_plan``. Supplied no-shift negatives gate per linked metric
    # (must NOT meet headline thresholds, one central part per metric);
    # shuffled/null controls record without gating. Parts fold back to
    # declared identities (all parts must pass/record).
    _dens_thresholds = {item.metric_id: item for item in config.thresholds}
    _dens_specs: list[_ControlSpec] = []
    _dens_supplied: dict[str, Mapping[str, float] | None] = {}
    _dens_derived: dict[str, str] = {}
    for control_id in (*config.required_controls, *config.optional_controls):
        linked = tuple(config.control_metrics[control_id])
        if not set(linked) & set(declared_metrics):
            continue
        required = control_id in config.required_controls
        kind = config.control_kinds[control_id]
        if kind in _SUPPLIED_KINDS:
            for metric_id in linked:
                if metric_id not in declared_metrics:
                    continue
                gate_threshold = _dens_thresholds[metric_id]
                gate = "below_threshold" if gate_threshold.comparator == ">=" else "meets_threshold"
                observed_negative = negative_flag if metric_id == "ood-flag-rate" else negative_gap
                part_id = f"{control_id}::{metric_id}"
                _dens_derived[part_id] = control_id
                _dens_specs.append(
                    _ControlSpec(
                        control_id=part_id, kind="counterexample", required=required,
                        metric_ids=(metric_id,),
                        expected_behavior="no-shift negative must not meet headline thresholds",
                        comparator=gate,  # type: ignore[arg-type]
                        threshold_value=float(gate_threshold.value),
                    )
                )
                _dens_supplied[part_id] = {metric_id: float(observed_negative)}
            continue
        _dens_specs.append(
            _ControlSpec(
                control_id=control_id, kind="null", required=required,
                metric_ids=linked,
                expected_behavior="shuffled null recorded without gating",
            )
        )
        _dens_supplied[control_id] = {}
    _dens_gating = _execute_plan(
        _ControlPlan(
            plan_id=f"{config.manifest_id}:density",
            repetitions=config.repetitions,
            confidence_level=config.confidence_level,
            evaluation_seed=config.evaluation_seed,
            control_seed=config.control_seed,
            training_seed=config.training_seed,
            controls=tuple(_dens_specs),
        ),
        _dens_supplied,
        statistics={"control-shuffled-null": _null_statistic}
        if any(spec.control_id == "control-shuffled-null" for spec in _dens_specs) else None,
    )
    # Bind the executed null outcome: real flag/gap/scores from the central
    # callback replace the placeholder null evaluation (no replay, no refit).
    _executed_null = _dens_gating.get("control-shuffled-null")
    if _executed_null is not None:
        if _executed_null.status == "failed":
            raise DetectionError(f"central null control failed: {_executed_null.reason}")
        _null_detail = dict(_executed_null.detail or {})
        shuffled_scores = tuple(float(item) for item in _null_detail.get("shuffled_scores", ()))  # type: ignore[union-attr]
        shuffled_flag = float(_executed_null.observed["ood-flag-rate"])
        shuffled_gap = float(_executed_null.observed["density-drift-gap"])
        shuffled_uncertainty = {
            "ood-flag-rate": _bootstrap_scores(
                np.asarray(shuffled_scores, dtype=np.float64), threshold,
                repetitions=config.repetitions, seed=config.evaluation_seed,
                confidence_level=config.confidence_level,
            ),
            "density-drift-gap": _bootstrap_gap(
                np.asarray(shuffled_scores, dtype=np.float64), reference_scores,
                repetitions=config.repetitions, seed=config.evaluation_seed,
                confidence_level=config.confidence_level,
            ),
        }
        null_evaluation = _DensityEvaluation(
            n_reference=evaluation.n_reference,
            n_calibration=evaluation.n_calibration,
            n_test=evaluation.n_test,
            dim=evaluation.dim,
            geometry=geometry,
            flag_threshold=threshold,
            test_flag_rate=evaluation.test_flag_rate,
            negative_flag_rate=evaluation.negative_flag_rate,
            shuffled_flag_rate=shuffled_flag,
            drift_gap=evaluation.drift_gap,
            negative_drift_gap=evaluation.negative_drift_gap,
            shuffled_drift_gap=shuffled_gap,
            test_scores=evaluation.test_scores,
            reference_scores=evaluation.reference_scores,
            negative_scores=evaluation.negative_scores,
            shuffled_scores=shuffled_scores,
            estimator_digest=evaluation.estimator_digest,
            uncertainty=shuffled_uncertainty,
        )
    control_outcomes: dict[str, str] = {}
    for control_id in (*config.required_controls, *config.optional_controls):
        if control_id in _dens_derived.values() or any(parent == control_id for parent in _dens_derived.values()):
            parts = [part for part, parent in _dens_derived.items() if parent == control_id]
            statuses = [_dens_gating[part].status for part in parts]
            control_outcomes[control_id] = "passed" if all(status in ("passed", "recorded") for status in statuses) else "failed"
            continue
        outcome = _dens_gating.get(control_id)
        if outcome is None:
            continue
        control_outcomes[control_id] = "passed" if outcome.status in ("passed", "recorded") else "failed"
    _dens_central: dict[str, object] = {control_id: outcome.to_dict() for control_id, outcome in _dens_gating.items()}
    _dens_blocked = sorted({_dens_derived.get(part, part) for part in _failed_required(_dens_gating)} & set(config.required_controls))
    if _dens_blocked:
        control_outcomes.update({control_id: "failed" for control_id in _dens_blocked})

    return _DetectionContext(
        representation_identity=identity,
        geometry=geometry,
        evaluation=evaluation,
        null_evaluation=null_evaluation,
        counterexample_evaluation=counterexample_evaluation,
        control_outcomes=control_outcomes,
        central_outcomes=_dens_central,
        distribution_free=False,
        unsupported_geometry=None,
    )


def _decide(context: _DetectionContext, config: DetectionConfig) -> tuple[FamilyDetection, ...]:
    """Decide every configured family from one shared evaluation context."""
    thresholds = {item.metric_id: item for item in config.thresholds}
    detections: list[FamilyDetection] = []
    for family_id in config.family_ids:
        family_metrics = [metric_id for metric_id in config.metric_ids if METRIC_FAMILY.get(metric_id) == family_id]
        if context.unsupported_geometry is not None:
            detections.append(
                FamilyDetection(
                    family_id=family_id,
                    outcome="unsupported",
                    claim_allowed=False,
                    observed_metrics={},
                    threshold_pass={},
                    control_outcomes=dict(context.control_outcomes),
                    evidence_status={
                        "density_reference_test_identity": "not_evaluated",
                        "density_or_distance_score": "not_evaluated",
                        "density_calibration_uncertainty": "not_evaluated",
                        "density_shuffled_null_control": "not_evaluated",
                        "density_no_shift_negative_control": "not_evaluated",
                    },
                    missing_evidence=("unsupported-geometry:density_ood_distribution_drift",),
                    reason=f"density convention does not support geometry {context.unsupported_geometry!r}; refusing a distribution-free verdict",
                )
            )
            continue
        if context.distribution_free or context.evaluation is None:
            reason = (
                "caller-declared distribution-free request; the parametric density convention cannot support it"
                if context.distribution_free
                else "manifest predeclares no density/OOD/drift metrics; refusing to borrow another family's label"
            )
            missing = (
                "distribution-free:density_ood_distribution_drift"
                if context.distribution_free
                else "missing-declaration:density_ood_distribution_drift"
            )
            detections.append(
                FamilyDetection(
                    family_id=family_id,
                    outcome="unsupported",
                    claim_allowed=False,
                    observed_metrics={},
                    threshold_pass={},
                    control_outcomes=dict(context.control_outcomes),
                    evidence_status={
                        "density_reference_test_identity": "not_evaluated",
                        "density_or_distance_score": "not_evaluated",
                        "density_calibration_uncertainty": "not_evaluated",
                        "density_shuffled_null_control": "not_evaluated",
                        "density_no_shift_negative_control": "not_evaluated",
                    },
                    missing_evidence=(missing,),
                    reason=reason,
                )
            )
            continue
        evaluation = context.evaluation
        observed = {
            "ood-flag-rate": evaluation.test_flag_rate,
            "density-drift-gap": evaluation.drift_gap,
        }
        observed = {metric_id: observed[metric_id] for metric_id in family_metrics if metric_id in observed}
        verdicts = {metric_id: thresholds[metric_id].passes(observed[metric_id]) for metric_id in observed}
        evidence = {
            "density_reference_test_identity": "observed",
            "density_or_distance_score": "observed",
            "density_calibration_uncertainty": "observed",
            "density_shuffled_null_control": "observed",
            "density_no_shift_negative_control": "observed",
        }
        decision = evaluate_claim(family_id, applicability="applicable", evidence_status=evidence)
        failed = sorted(
            control_id
            for control_id in config.required_controls
            if control_id in context.control_outcomes
            and context.control_outcomes[control_id] == "failed"
            and bool(set(config.control_metrics[control_id]) & set(family_metrics))
        )
        unmet = sorted(metric_id for metric_id in family_metrics if metric_id in verdicts and not verdicts[metric_id])
        if unmet:
            # A drift score alone never promotes: report the unmet predeclared
            # threshold even when a control also fails.
            detections.append(
                FamilyDetection(
                    family_id=family_id,
                    outcome="inconclusive",
                    claim_allowed=False,
                    observed_metrics=dict(observed),
                    threshold_pass=dict(verdicts),
                    control_outcomes=dict(context.control_outcomes),
                    evidence_status=dict(evidence),
                    missing_evidence=tuple(f"threshold:{metric_id}" for metric_id in unmet),
                    reason=f"predeclared thresholds unmet; a drift score alone is not a diagnosis: {', '.join(unmet)}",
                )
            )
            continue
        if failed:
            detections.append(
                FamilyDetection(
                    family_id=family_id,
                    outcome="inconclusive",
                    claim_allowed=False,
                    observed_metrics=dict(observed),
                    threshold_pass=dict(verdicts),
                    control_outcomes=dict(context.control_outcomes),
                    evidence_status=dict(evidence),
                    missing_evidence=tuple(f"failed-control:{control_id}" for control_id in failed),
                    reason=f"required control failure blocks the conclusion: {', '.join(failed)}",
                )
            )
            continue
        if decision.outcome != "supported" or not decision.claim_allowed:
            detections.append(
                FamilyDetection(
                    family_id=family_id,
                    outcome="inconclusive",
                    claim_allowed=False,
                    observed_metrics=dict(observed),
                    threshold_pass=dict(verdicts),
                    control_outcomes=dict(context.control_outcomes),
                    evidence_status=dict(evidence),
                    missing_evidence=tuple(decision.missing),
                    reason=decision.reason,
                )
            )
            continue
        detections.append(
            FamilyDetection(
                family_id=family_id,
                outcome="supported",
                claim_allowed=True,
                observed_metrics=dict(observed),
                threshold_pass=dict(verdicts),
                control_outcomes=dict(context.control_outcomes),
                evidence_status=dict(evidence),
                missing_evidence=(),
                reason="predeclared thresholds met with required controls passed",
            )
        )
    return tuple(detections)


def detect_families(
    pair: ReferenceTestPair | LatentValue,
    config: DetectionConfig,
    *,
    controls: Mapping[str, ReferenceTestPair | LatentValue] | None = None,
) -> tuple[FamilyDetection, ...]:
    """Evaluate the configured family on one bound pair plus named controls.

    Focused unit-check seam: evaluates its own context, so composing it with a
    separate :func:`detection_payload` call evaluates twice. Production paths
    (including :func:`make_detect_executor`) must use :func:`evaluate_detection`.
    """
    if not isinstance(config, DetectionConfig):
        raise DetectionError("config must be a DetectionConfig")
    if not isinstance(pair, (ReferenceTestPair, LatentValue)):
        raise DetectionError("pair must be a ReferenceTestPair or LatentValue")
    supplied = dict(controls or {})
    return _decide(_evaluate_context(pair, config, supplied), config)


def detection_payload(
    detections: Sequence[FamilyDetection],
    config: DetectionConfig,
    pair: ReferenceTestPair | LatentValue,
    *,
    control_metrics: Mapping[str, Mapping[str, float]] | None = None,
    context: _DetectionContext | None = None,
) -> dict[str, object]:
    """Assemble the canonical machine-readable detect-stage payload."""
    if not detections:
        raise DetectionError("detections must not be empty")
    resolved = context if context is not None else _evaluate_context(pair, config, {})
    try:
        canonical_json([detection.to_dict() for detection in detections])
    except PortableNodeError as exc:
        raise DetectionError(f"detections are not canonical JSON: {exc}") from exc
    supplied_metrics = {control_id: dict(metrics) for control_id, metrics in (control_metrics or {}).items()}
    for control_id in supplied_metrics:
        if control_id not in config.control_metrics:
            raise DetectionError(f"unknown control reference: {control_id!r}")
    measurements: dict[str, object] = {"representation_identity": resolved.representation_identity}
    evaluation = resolved.evaluation
    if evaluation is not None:
        measurements["density"] = {
            "calibration_rule": CALIBRATION_RULE,
            "shuffled_null_method": "independent-per-column-permutation",
            "shuffled_null_seed": int(config.control_seed),
            "shuffled_null_stream_seed": int(resolved.central_outcomes.get("control-shuffled-null", {}).get("stream_seed", config.control_seed)),  # type: ignore[union-attr]
            "fit_provenance": dict(pair.reference_provenance.to_dict(), role="reference-fit", geometry=evaluation.geometry) if isinstance(pair, ReferenceTestPair) else {},
            "calibration_provenance": dict(pair.reference_provenance.to_dict(), role="heldout-calibration", geometry=evaluation.geometry) if isinstance(pair, ReferenceTestPair) else {},
            "estimator_digest": evaluation.estimator_digest,
            "flag_threshold": float(evaluation.flag_threshold),
            "geometry": evaluation.geometry,
            "n_calibration": int(evaluation.n_calibration),
            "n_reference": int(evaluation.n_reference),
            "n_test": int(evaluation.n_test),
            "negative_drift_gap": float(evaluation.negative_drift_gap),
            "negative_flag_rate": float(evaluation.negative_flag_rate),
            "ood_flag_rate": float(evaluation.test_flag_rate),
            "drift_gap": float(evaluation.drift_gap),
            "shuffled_drift_gap": float(evaluation.shuffled_drift_gap),
            "shuffled_flag_rate": float(evaluation.shuffled_flag_rate),
            "uncertainty": {key: dict(item) for key, item in evaluation.uncertainty.items()},
        }
        if isinstance(pair, ReferenceTestPair):
            measurements["provenance"] = {
                "reference": pair.reference_provenance.to_dict(),
                "test": pair.test_provenance.to_dict(),
            }
    return {
        "config": config.to_dict(),
        "control_metrics": supplied_metrics,
        "controls": {control_id: dict(outcome) for control_id, outcome in resolved.central_outcomes.items()},
        "families": [detection.to_dict() for detection in detections],
        "family_evidence": {detection.family_id: dict(detection.evidence_status) for detection in detections},
        "measurements": measurements,
    }


def _control_table(
    context: _DetectionContext, config: DetectionConfig
) -> dict[str, dict[str, float]]:
    """Record per-control metric observations without refitting anything."""
    table: dict[str, dict[str, float]] = {}
    evaluation = context.evaluation
    if evaluation is None:
        return table
    for control_id in (*config.required_controls, *config.optional_controls):
        linked = config.control_metrics[control_id]
        entry: dict[str, float] = {}
        kind = config.control_kinds[control_id]
        source_flag = evaluation.test_flag_rate
        source_gap = evaluation.drift_gap
        if kind in _SUPPLIED_KINDS:
            source_flag = evaluation.negative_flag_rate
            source_gap = evaluation.negative_drift_gap
        elif context.null_evaluation is not None:
            source_flag = context.null_evaluation.shuffled_flag_rate
            source_gap = context.null_evaluation.shuffled_drift_gap
        if "ood-flag-rate" in linked:
            entry["ood-flag-rate"] = source_flag
        if "density-drift-gap" in linked:
            entry["density-drift-gap"] = source_gap
        if entry:
            table[control_id] = entry
    return table


def evaluate_detection(
    pair: ReferenceTestPair | LatentValue,
    config: DetectionConfig,
    controls: Mapping[str, ReferenceTestPair | LatentValue],
) -> tuple[tuple[FamilyDetection, ...], dict[str, object]]:
    """Evaluate one pair plus controls exactly once and return decisions plus payload.

    This is the single-evaluation seam every caller must use: the density is
    fitted once and calibrated once inside one :class:`_DetectionContext`, then
    shared by family decisions and payload assembly. Uncertainty resamples
    fitted calibrated scores without refitting.
    """
    if not isinstance(config, DetectionConfig):
        raise DetectionError("config must be a DetectionConfig")
    if not isinstance(pair, (ReferenceTestPair, LatentValue)):
        raise DetectionError("pair must be a ReferenceTestPair or LatentValue")
    supplied = dict(controls)
    context = _evaluate_context(pair, config, supplied)
    detections = _decide(context, config)
    payload = detection_payload(detections, config, pair, control_metrics=_control_table(context, config), context=context)
    return detections, payload


def make_detect_executor(
    pair: ReferenceTestPair | LatentValue,
    config: DetectionConfig,
    *,
    controls: Mapping[str, ReferenceTestPair | LatentValue] | None = None,
    version: str = "density-ood-drift-detector-v1",
) -> Any:
    """Build a supplied ``detect``-stage executor bound to one pair and config.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable verifies the invocation targets ``detect`` with the expected
    request/manifest identity, evaluates :func:`evaluate_detection` once, and
    returns a ``completed`` :class:`StageOutput` whose payload carries
    observations, family evidence, control outcomes, uncertainty, and claim
    gating. No algorithm enters ``DiagnosticWorkflow`` itself.
    """
    _non_empty_string(version, name="version")
    if not isinstance(pair, (ReferenceTestPair, LatentValue)):
        raise DetectionError("pair must be a ReferenceTestPair or LatentValue")
    if not isinstance(config, DetectionConfig):
        raise DetectionError("config must be a DetectionConfig")
    frozen_pair = pair
    frozen_controls = dict(controls or {})

    def _execute(invocation: StageInvocation) -> StageOutput:
        from latent_anything._diagnostic_workflow import StageContractError as _ContractError

        if invocation.stage != "detect":
            raise _ContractError(f"detect executor received stage {invocation.stage!r}")
        if invocation.request.manifest_id != config.manifest_id:
            raise _ContractError("detect executor manifest identity mismatch")
        _, payload = evaluate_detection(frozen_pair, config, frozen_controls)
        aggregate_outcome: Literal["completed", "unsupported"] = "completed"
        return StageOutput(stage="detect", outcome=aggregate_outcome, payload=payload, artifact_refs=())

    _execute.detector_version = version  # type: ignore[attr-defined]
    return _execute


__all__ = [
    "CALIBRATION_RULE",
    "METRIC_FAMILY",
    "SUPPORTED_FAMILIES",
    "ClaimOutcome",
    "DetectionConfig",
    "DetectionError",
    "FamilyDetection",
    "FamilyThreshold",
    "Provenance",
    "ReferenceTestPair",
    "detect_families",
    "detection_config_from_manifest",
    "detection_payload",
    "evaluate_detection",
    "make_detect_executor",
]
