"""Sparse-feature instability and sequence/trajectory drift detection (Sprint 80.12).

Private detector-family adapter behind the frozen taxonomy/workflow contracts
for ``sparse_feature_instability`` and ``sequence_trajectory_drift``. It
consumes bound representation batches plus predeclared metric/control/threshold
configuration and emits machine-readable observations, taxonomy
family-evidence statuses (``evaluate_claim``), control outcomes, seeded
uncertainty metadata, and honest ``supported``/``inconclusive``/``unsupported``
outcomes.

Applicability is derived from bound axis metadata and explicit evidence, never
assumed. Sparse evaluation requires a ``seed`` axis plus caller-supplied
per-seed sparse-feature batches (dictionary atoms); temporal evaluation
requires a ``sequence_or_time`` axis plus caller-supplied ordered trajectories.
A family with no applicable input is omitted from evaluated family evidence and
reported with an honest omission reason. Where the manifest predeclares no
promotable declaration for an applicable family, the detector returns honest
``unsupported`` rather than borrowing another family's metrics.

Reused existing primitives (no new estimators):

- ``DictionaryLearning`` is the single shared sparse convention: one fit per
  seed with the manifest training seed as the base plus declared seed offsets.
- ``match_by_decoder_cosine`` is the single shared permutation-invariant
  feature-alignment convention: decoder-direction matching across seeds, never
  raw feature indices.
- ``LatentSpace.distance`` dispatch (the point-cost convention shared with
  ``indexwise_distance``) is the single shared stepwise drift convention over
  declared ordered trajectories; the
  repeatability statistic is ordered drift minus the no-drift negative
  (``drift - negative_drift``), isolating true ordered change.
- Taxonomy evidence gating reuses ``evaluate_claim``; manifest validity reuses
  ``validate_manifest``; canonical JSON reuses the shared ``canonical_json``
  contract.

Single-evaluation seam: :func:`evaluate_detection` fits every dictionary and
scores every trajectory exactly once per executor call, then shares one
context between family decisions and payload assembly. Sparse uncertainty
resamples fitted matched-cosine values without refitting and temporal
uncertainty resamples fitted stepwise distances without recomputing
alignments, both through the central ``_statistical_controls`` executor
(identity-derived ``evaluation`` streams); the shuffled-sequence null runs
as one central ``shuffled`` control.
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
from latent_anything._sae_metrics import match_by_decoder_cosine
from latent_anything.diagnostics import DiagnosticRequest
from latent_anything.dictionary_learning import DictionaryLearning, DictionaryLearningConfig
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.trajectory import Trajectory

SUPPORTED_FAMILIES: tuple[str, ...] = ("sparse_feature_instability", "sequence_trajectory_drift")
"""Detector scope for this task. Exact taxonomy identifiers; nothing else is evaluated."""

METRIC_FAMILY: Mapping[str, str] = MappingProxyType(
    {
        "sparse-cross-seed-stability": "sparse_feature_instability",
        "sparse-reconstruction-quality": "sparse_feature_instability",
        "trajectory-stepwise-drift": "sequence_trajectory_drift",
        "trajectory-repeatability": "sequence_trajectory_drift",
    }
)
"""Predeclared metric-to-family wiring. Unknown metrics reject."""

_SUPPLIED_KINDS = frozenset({"counterexample", "negative"})
"""Manifest control kinds that require caller-supplied batch data.

``seed``-randomized and ``shuffled``/``null`` controls are derived by the
detector under manifest seeds, so supplying batch data for them is a
fail-closed error rather than silently ignored input.
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
    seed_axis: tuple[int, ...]
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
        object.__setattr__(self, "seed_axis", tuple(self.seed_axis))
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
            "confidence_level": float(self.confidence_level),
            "control_metrics": {key: list(value) for key, value in self.control_metrics.items()},
            "control_kinds": dict(self.control_kinds),
            "control_seed": int(self.control_seed),
            "evaluation_seed": int(self.evaluation_seed),
            "family_ids": list(self.family_ids),
            "manifest_id": self.manifest_id,
            "manifest_metric_families": list(self.manifest_metric_families),
            "metric_ids": list(self.metric_ids),
            "optional_controls": list(self.optional_controls),
            "repetitions": int(self.repetitions),
            "required_controls": list(self.required_controls),
            "seed_axis": list(self.seed_axis),
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
    seed_axis: tuple[int, ...] = (11, 12),
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
    seeds_tuple = tuple(seed_axis)
    if len(seeds_tuple) < 2:
        raise DetectionError("seed_axis must hold at least two seeds")
    for seed in seeds_tuple:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise DetectionError("seed_axis must hold non-negative integers")
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
        seed_axis=seeds_tuple,
        repetitions=int(repetitions),
        confidence_level=_finite_number(uncertainty.get("confidence_level"), name="uncertainty.confidence_level"),
        training_seed=_non_negative_int(training[0], name="seeds.training[0]"),
        evaluation_seed=_non_negative_int(evaluation[0], name="seeds.evaluation[0]"),
        control_seed=_non_negative_int(controls_seed[0], name="seeds.controls[0]"),
    )


@dataclass(frozen=True)
class FamilyDetection:
    """Evidence-bound outcome for one taxonomy family on one bound input."""

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
class SparseInput:
    """Bound sparse-feature input: one batch plus per-seed and negative batches.

    ``seeds`` declares the seed axis; ``seed_batches`` maps each seed to a
    2D batch fitted independently. ``negative_batches`` maps a control name to
    a batch evaluated under the same comparison protocol. Sample identities
    must be unique within each batch.
    """

    value: LatentValue
    sample_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    seed_batches: Mapping[str, LatentValue]
    seed_sample_ids: Mapping[str, tuple[str, ...]] | None = None
    negative_batches: Mapping[str, LatentValue] | None = None
    representation_identity: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.value, LatentValue):
            raise DetectionError("value must be a LatentValue")
        _batch_matrix(self.value, name="sparse batch")
        sample_ids = tuple(self.sample_ids)
        object.__setattr__(self, "sample_ids", sample_ids)
        if len(sample_ids) != int(np.asarray(self.value.to_numpy()).shape[0]):
            raise DetectionError("sample_ids must cover every sparse sample")
        for position, sample_id in enumerate(sample_ids):
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise DetectionError(f"sample_ids[{position}] must be a non-empty string")
        if len(set(sample_ids)) != len(sample_ids):
            raise DetectionError("sample identities must be unique within one batch")
        seeds = tuple(self.seeds)
        object.__setattr__(self, "seeds", seeds)
        if len(seeds) < 2:
            raise DetectionError("sparse evaluation requires at least two seeds")
        seed_batches = dict(self.seed_batches)
        object.__setattr__(self, "seed_batches", MappingProxyType(seed_batches))
        raw_seed_ids = dict(self.seed_sample_ids) if self.seed_sample_ids is not None else {}
        seed_ids: dict[str, tuple[str, ...]] = {}
        for seed in seeds:
            key = str(seed)
            if key not in seed_batches or not isinstance(seed_batches[key], LatentValue):
                raise DetectionError(f"seed_batches is missing batch data for seed {seed}")
            batch_matrix = _batch_matrix(seed_batches[key], name=f"seed {seed} batch")
            ids = tuple(raw_seed_ids[key]) if key in raw_seed_ids else sample_ids
            if len(ids) != int(batch_matrix.shape[0]):
                raise DetectionError(f"seed {seed} sample identities must cover every row")
            for position, sample_id in enumerate(ids):
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise DetectionError(f"seed {seed} sample_ids[{position}] must be a non-empty string")
            if len(set(ids)) != len(ids):
                raise DetectionError(f"seed {seed} sample identities must be unique")
            seed_ids[key] = ids
        if set(seed_ids[str(seeds[0])]) != set(sample_ids):
            raise DetectionError("seed batches must cover the same samples as the bound batch")
        for seed in seeds[1:]:
            if set(seed_ids[str(seed)]) != set(seed_ids[str(seeds[0])]):
                raise DetectionError("seed batches must cover the same samples")
        object.__setattr__(self, "seed_sample_ids", MappingProxyType(seed_ids))
        negative_batches = dict(self.negative_batches or {})
        object.__setattr__(self, "negative_batches", MappingProxyType(negative_batches))
        if not negative_batches:
            raise DetectionError("sparse evaluation requires at least one negative batch")
        for name, batch in negative_batches.items():
            if not isinstance(batch, LatentValue):
                raise DetectionError(f"negative batch {name!r} must be a LatentValue")
            _batch_matrix(batch, name=f"negative {name!r} batch")
        _non_empty_string(self.representation_identity, name="representation_identity")


@dataclass(frozen=True)
class TemporalInput:
    """Bound temporal input: ordered reference/test trajectories plus controls.

    ``reference`` and ``test`` are ordered ``(n_steps, dim)`` trajectories with
    declared step identities; ``negative`` is a no-drift trajectory under the
    same alignment. Shuffled/reversed controls are derived by the detector.
    """

    reference: Trajectory
    test: Trajectory
    negative: Trajectory
    step_ids: tuple[str, ...]
    space: LatentSpace
    representation_identity: str

    def __post_init__(self) -> None:
        for name in ("reference", "test", "negative"):
            if not isinstance(getattr(self, name), Trajectory):
                raise DetectionError(f"{name} must be a Trajectory")
        step_ids = tuple(self.step_ids)
        object.__setattr__(self, "step_ids", step_ids)
        if len(step_ids) != len(self.test):
            raise DetectionError("step_ids must cover every test step")
        for position, step_id in enumerate(step_ids):
            if not isinstance(step_id, str) or not step_id.strip():
                raise DetectionError(f"step_ids[{position}] must be a non-empty string")
        if len(set(step_ids)) != len(step_ids):
            raise DetectionError("step identities must be unique within one trajectory")
        if not isinstance(self.space, LatentSpace):
            raise DetectionError("space must be a LatentSpace")
        if self.space.geometry not in ("euclidean", "unit_norm"):
            raise DetectionError(f"unsupported geometry for trajectory drift: {self.space.geometry!r}")
        for name, trajectory in (("reference", self.reference), ("test", self.test), ("negative", self.negative)):
            points = trajectory.to_numpy()
            if points.ndim != 2 or points.shape[1] != self.space.dim:
                raise DetectionError(f"{name} trajectory width {points.shape} does not match space dim {self.space.dim}")
            if not np.isfinite(points).all():
                raise DetectionError(f"{name} trajectory contains non-finite values")
        if len(self.reference) != len(self.test) or len(self.negative) != len(self.test):
            raise DetectionError("reference/test/negative trajectories must share one horizon")
        if len(self.test) < 2:
            raise DetectionError("temporal evaluation requires at least two steps")
        _non_empty_string(self.representation_identity, name="representation_identity")


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
    detector's one-fit call sites unchanged while the central executor owns
    the interval contract.
    """
    from latent_anything._statistical_controls import summarize_interval as _summarize_central

    return dict(
        _summarize_central(
            samples, repetitions=repetitions, seed=seed, confidence_level=confidence_level
        )
    )


@dataclass(frozen=True)
class _SparseEvaluation:
    """One sparse input evaluated once: one dictionary fit per seed, no refits."""

    n_samples: int
    dim: int
    n_components: int
    seeds: tuple[int, ...]
    stability: float
    matched_cosines: tuple[float, ...]
    quality: float
    negative_stability: float
    seed_control_stability: float
    estimator_digests: tuple[str, ...]
    uncertainty: Mapping[str, dict[str, object]]
    seed_control_fit_seed: int = 0


@dataclass(frozen=True)
class _TemporalEvaluation:
    """One temporal input evaluated once: one drift scoring per trajectory."""

    n_steps: int
    dim: int
    drift: float
    negative_drift: float
    shuffled_drift: float
    reversed_drift: float
    stepwise: tuple[float, ...]
    uncertainty: Mapping[str, dict[str, object]]


@dataclass(frozen=True)
class _DetectionContext:
    """One executor call evaluated exactly once and shared by decisions and payload."""

    representation_identity: str
    sparse: _SparseEvaluation | None
    temporal: _TemporalEvaluation | None
    control_outcomes: Mapping[str, str]
    central_outcomes: Mapping[str, object]
    omissions: Mapping[str, str]
    distribution_free_sparse: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_outcomes", dict(self.control_outcomes))
        object.__setattr__(self, "central_outcomes", dict(self.central_outcomes))
        object.__setattr__(self, "omissions", dict(self.omissions))


def _fit_sparse_once(matrix: np.ndarray, *, seed: int, n_components: int = 4, max_iter: int = 100) -> tuple[np.ndarray, float, str]:
    """Fit one dictionary and return atoms, held-out quality gain, and digest."""
    learner = DictionaryLearning(
        DictionaryLearningConfig(n_components=n_components, max_iter=max_iter, random_state=seed)
    )
    evaluation = learner.fit(matrix)
    baseline = float(evaluation.val_baseline_mse)
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise DetectionError("sparse comparison baseline is degenerate")
    quality = float(np.clip(1.0 - float(evaluation.val_reconstruction_mse) / baseline, 0.0, 1.0))
    atoms = np.asarray(learner.components_, dtype=np.float64)
    digest = __import__("hashlib").sha256(np.ascontiguousarray(atoms, dtype=np.float64).tobytes()).hexdigest()
    return atoms, quality, digest


def _stability_of(atoms_a: np.ndarray, atoms_b: np.ndarray) -> tuple[float, tuple[float, ...]]:
    """Align two atom sets permutation-invariantly and return matched fraction.

    Stability is the fraction of features matched at cosine >= 0.8 under
    globally ranked decoder-direction matching (``match_by_decoder_cosine``),
    never raw feature indices. Sparse ground truth re-matches its atoms
    across seeds; unstructured noise cannot.
    """
    n = int(atoms_a.shape[0])
    matched = match_by_decoder_cosine(atoms_a.T, atoms_b.T, 0.85)
    cosines = tuple(float(cosine) for _, cosine in matched)
    return float(len(cosines) / n) if n else 0.0, cosines


def _evaluate_sparse(pair: SparseInput, config: DetectionConfig) -> _SparseEvaluation:
    """Fit one dictionary per seed and align features permutation-invariantly."""
    seeds = tuple(pair.seeds)
    matrices = {seed: _batch_matrix(pair.seed_batches[str(seed)], name=f"seed {seed} batch") for seed in seeds}
    # Same-sample cross-seed comparison: canonicalize every seed batch to
    # the bound sample-identity order before fitting. Row permutation is a
    # sample artifact; feature stability must not depend on it.
    canonical = list(pair.sample_ids)
    position_of = {sample_id: position for position, sample_id in enumerate(canonical)}
    aligned: dict[int, np.ndarray] = {}
    for seed in seeds:
        ids = pair.seed_sample_ids[str(seed)]
        order = np.asarray([position_of[sample_id] for sample_id in ids], dtype=np.int64)
        aligned[seed] = matrices[seed][np.argsort(order, kind="stable")]
    matrices = aligned
    widths = {int(matrix.shape[1]) for matrix in matrices.values()}
    if len(widths) != 1:
        raise DetectionError(f"seed batch feature widths diverge: {sorted(widths)}")
    dim = int(next(iter(widths)))
    atoms: dict[int, np.ndarray] = {}
    qualities: dict[int, float] = {}
    digests: list[str] = []
    for seed in seeds:
        fitted, quality, digest = _fit_sparse_once(matrices[seed], seed=int(seed))
        atoms[seed] = fitted
        qualities[seed] = quality
        digests.append(digest)
    reference = atoms[seeds[0]]
    pair_cosines: list[float] = []
    pair_fractions: list[float] = []
    for seed in seeds[1:]:
        fraction, cosines = _stability_of(reference, atoms[seed])
        pair_fractions.append(fraction)
        pair_cosines.extend(cosines)
    # Quality floor: stability is only meaningful when both fits reconstruct;
    # failed reconstructions cannot promote regardless of matching fraction.
    if not pair_cosines:
        raise DetectionError("sparse alignment produced no matched features")
    stability = float(sum(pair_fractions) / len(pair_fractions))
    quality = float(sum(qualities.values()) / len(qualities))
    # Cross-seed control: the actual fit-seed draw plus the single predeclared
    # extra dictionary fit and stability scoring runs inside the
    # executor-supplied stream. Observed fit seed and metrics bind directly.
    def _cross_seed_statistic(rng: np.random.Generator) -> Mapping[str, object]:
        fit_seed = int(rng.integers(0, 2**31 - 1))
        fitted_atoms, _, _ = _fit_sparse_once(matrices[seeds[0]], seed=fit_seed)
        control_stability, _ = _stability_of(reference, fitted_atoms)
        return {
            "fit_seed": float(fit_seed),
            "sparse-cross-seed-stability": float(control_stability),
        }

    _seed_control = _central_control(
        control_id="control-seed-noise",
        base_seed=config.control_seed,
        seed_role="control",
        required=False,
        kind="cross_seed",
        metric_ids=("sparse-cross-seed-stability",),
        expected_behavior="first-seed refit under the control stream",
        statistic=_cross_seed_statistic,  # type: ignore[arg-type]
    )
    if _seed_control.status != "recorded":
        raise DetectionError("central cross-seed control must record, not gate")
    seed_control_stability = float(_seed_control.observed["sparse-cross-seed-stability"])
    _recorded_fit_seed = int(_seed_control.observed["fit_seed"])
    # Negative control: evaluate the caller-supplied negative batch with the
    # target reference atoms under the same permutation-invariant protocol. A
    # genuinely stable negative (same sparse family) aligns to the reference
    # and fails the control; unstructured negatives do not align and pass.
    # No raw-index comparison anywhere.
    negative_name = next(iter(pair.negative_batches), "")
    negative_matrix = _batch_matrix(pair.negative_batches[negative_name], name=f"negative {negative_name!r} batch")
    negative_atoms, _, _ = _fit_sparse_once(negative_matrix, seed=int(seeds[0]))
    negative_stability, _ = _stability_of(reference, negative_atoms)
    cosine_array = np.asarray(pair_cosines, dtype=np.float64)

    def _stability_draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, cosine_array.shape[0], size=cosine_array.shape[0])
        return float(np.mean(cosine_array[positions]))

    _, stability_interval = _central_bootstrap(
        control_id="bootstrap:sparse-cross-seed-stability",
        base_seed=config.evaluation_seed,
        seed_role="evaluation",
        repetitions=config.repetitions,
        confidence_level=config.confidence_level,
        required=False,
        kind="bootstrap",
        metric_ids=("sparse-cross-seed-stability",),
        expected_behavior="seeded resampling of fitted matched cosines",
        draw=_stability_draw,
    )
    uncertainty = {
        "sparse-cross-seed-stability": dict(stability_interval),
    }
    return _SparseEvaluation(
        n_samples=int(matrices[seeds[0]].shape[0]),
        dim=dim,
        n_components=int(atoms[seeds[0]].shape[0]),
        seeds=seeds,
        stability=stability,
        matched_cosines=tuple(pair_cosines),
        quality=quality,
        negative_stability=negative_stability,
        seed_control_stability=seed_control_stability,
        estimator_digests=tuple(digests),
        uncertainty=uncertainty,
        seed_control_fit_seed=_recorded_fit_seed,
    )

def _stepwise_drift(reference: np.ndarray, test: np.ndarray, space: LatentSpace) -> tuple[float, tuple[float, ...]]:
    """Score one ordered pair once with stepwise index-wise distances.

    Uses the ``indexwise_distance`` point-cost convention (``LatentSpace``
    geometry dispatch per step); the stepwise vector is retained so
    uncertainty resamples fitted distances without recomputing alignments.
    """
    from latent_anything.dtw import indexwise_distance as _indexwise

    points_ref = np.asarray(reference, dtype=np.float64)
    points_test = np.asarray(test, dtype=np.float64)
    stepwise = tuple(float(space.distance(a, b)) for a, b in zip(points_ref, points_test, strict=True))
    mean = _indexwise(points_ref, points_test, space)
    assert abs(mean - float(sum(stepwise) / len(stepwise))) < 1e-9
    return mean, stepwise


def _shuffled_sequence_statistic(
    rng: np.random.Generator, reference: np.ndarray, test: np.ndarray, space: LatentSpace
) -> Mapping[str, object]:
    """Permute the sequence with the supplied central RNG and score once."""
    order = rng.permutation(test.shape[0])
    shuffled = test[order]
    drift, _ = _stepwise_drift(reference, shuffled, space)
    return {
        "trajectory-stepwise-drift": float(drift),
        "shuffled_order": [int(item) for item in order],
        "shuffled_steps": [float(item) for item in np.asarray(shuffled).ravel()],
    }


def _evaluate_temporal(pair: TemporalInput, config: DetectionConfig) -> _TemporalEvaluation:
    """Score ordered, shuffled, and reversed trajectories with one pass each.

    Central schedule: the shuffled-sequence null permutes under the
    executor's identity-derived control stream; the drift interval resamples
    fitted stepwise distances without recomputing alignments.
    """
    reference = pair.reference.to_numpy()
    test = pair.test.to_numpy()
    negative = pair.negative.to_numpy()
    drift, stepwise = _stepwise_drift(reference, test, pair.space)
    negative_drift, _ = _stepwise_drift(reference, negative, pair.space)
    _sequence_shuffle = _central_control(
        control_id="control-shuffled-sequence",
        base_seed=config.control_seed,
        seed_role="control",
        required=False,
        kind="shuffled",
        metric_ids=("trajectory-stepwise-drift",),
        expected_behavior="sequence permutation under the control stream",
        statistic=lambda rng: _shuffled_sequence_statistic(rng, reference, test, pair.space),
    )
    if _sequence_shuffle.status != "recorded":
        raise DetectionError("central shuffled control must record, not gate")
    shuffled_drift = float(_sequence_shuffle.observed["trajectory-stepwise-drift"])
    _shuffled_detail = dict(_sequence_shuffle.detail or {})
    shuffled_steps = tuple(float(item) for item in _shuffled_detail["shuffled_steps"])  # type: ignore[union-attr]
    reversed_drift, _ = _stepwise_drift(reference, test[::-1], pair.space)
    stepwise_array = np.asarray(stepwise, dtype=np.float64)

    def _drift_draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, stepwise_array.shape[0], size=stepwise_array.shape[0])
        return float(np.mean(stepwise_array[positions]))

    _, drift_interval = _central_bootstrap(
        control_id="bootstrap:trajectory-stepwise-drift",
        base_seed=config.evaluation_seed,
        seed_role="evaluation",
        repetitions=config.repetitions,
        confidence_level=config.confidence_level,
        required=False,
        kind="bootstrap",
        metric_ids=("trajectory-stepwise-drift",),
        expected_behavior="seeded resampling of fitted stepwise distances",
        draw=_drift_draw,
    )
    uncertainty = {
        "trajectory-stepwise-drift": dict(drift_interval),
    }
    for metric_id, number in (("trajectory-stepwise-drift", drift), ("negative", negative_drift)):
        if not np.isfinite(number):
            raise DetectionError(f"metric {metric_id!r} produced a non-finite value")
    return _TemporalEvaluation(
        n_steps=int(test.shape[0]),
        dim=int(test.shape[1]),
        drift=drift,
        negative_drift=negative_drift,
        shuffled_drift=shuffled_drift,
        reversed_drift=reversed_drift,
        stepwise=stepwise,
        uncertainty=uncertainty,
    )


def _evaluate_context(
    target: SparseInput | TemporalInput | LatentValue,
    config: DetectionConfig,
    supplied: Mapping[str, SparseInput | TemporalInput | LatentValue],
) -> _DetectionContext:
    """Validate and evaluate the target plus every control batch exactly once."""
    for control_id in supplied:
        if control_id not in (*config.required_controls, *config.optional_controls):
            raise DetectionError(f"unknown control reference: {control_id!r}")
        kind = config.control_kinds[control_id]
        if kind not in _SUPPLIED_KINDS:
            raise DetectionError(f"control {control_id!r} is derived by the detector; no batch data is accepted")
    wants_sparse = "sparse_feature_instability" in config.family_ids
    wants_temporal = "sequence_trajectory_drift" in config.family_ids
    sparse_metrics = [m for m in config.metric_ids if METRIC_FAMILY.get(m) == "sparse_feature_instability"]
    temporal_metrics = [m for m in config.metric_ids if METRIC_FAMILY.get(m) == "sequence_trajectory_drift"]
    manifest_declared = [f for f in config.manifest_metric_families if isinstance(f, str)]
    sparse_covered = "sparse_feature_instability" in manifest_declared and set(sparse_metrics) == {
        m for m in config.metric_ids if METRIC_FAMILY.get(m) == "sparse_feature_instability"
    }
    temporal_covered = "sequence_trajectory_drift" in manifest_declared and set(temporal_metrics) == {
        m for m in config.metric_ids if METRIC_FAMILY.get(m) == "sequence_trajectory_drift"
    }

    sparse_input = target if isinstance(target, SparseInput) else None
    temporal_input = target if isinstance(target, TemporalInput) else None
    if wants_sparse and wants_temporal:
        raise DetectionError("one executor call evaluates one family input; sparse and temporal need separate calls")
    if wants_sparse and sparse_input is None:
        raise DetectionError("sparse evaluation requires a bound SparseInput with seed axis and per-seed batches")
    if wants_temporal and temporal_input is None:
        raise DetectionError("temporal evaluation requires a bound TemporalInput with a sequence_or_time axis")

    sparse: _SparseEvaluation | None = None
    temporal: _TemporalEvaluation | None = None
    omissions: dict[str, str] = {}
    control_outcomes: dict[str, str] = {}
    distribution_free_sparse = False

    _sparse_thresholds = {item.metric_id: item for item in config.thresholds}
    _sparse_specs: list[_ControlSpec] = []
    _sparse_supplied: dict[str, Mapping[str, float] | None] = {}
    _sparse_derived: dict[str, str] = {}

    if wants_sparse:
        assert sparse_input is not None
        if not sparse_metrics or (manifest_declared and not sparse_covered):
            omissions["sparse_feature_instability"] = (
                "missing-declaration:sparse_feature_instability"
                if not sparse_metrics or (manifest_declared and not sparse_covered)
                else "not-applicable:sparse_feature_instability"
            )
        else:
            sparse = _evaluate_sparse(sparse_input, config)
            for control_id in (*config.required_controls, *config.optional_controls):
                linked = tuple(config.control_metrics[control_id])
                if not set(linked) & set(sparse_metrics):
                    continue
                required = control_id in config.required_controls
                kind = config.control_kinds[control_id]
                if kind in _SUPPLIED_KINDS:
                    # The stable negative must NOT reach the stability
                    # threshold; quality is a positive-evidence metric and is
                    # not gated by the negative. One central part per linked
                    # stability metric.
                    stability_linked = [m for m in linked if m == "sparse-cross-seed-stability"]
                    if not stability_linked:
                        _sparse_specs.append(
                            _ControlSpec(
                                control_id=control_id, kind="counterexample", required=required,
                                metric_ids=linked,
                                expected_behavior="stable negative recorded without stability gating",
                            )
                        )
                        _sparse_supplied[control_id] = {}
                        continue
                    for metric_id in stability_linked:
                        gate_threshold = _sparse_thresholds[metric_id]
                        gate = "below_threshold" if gate_threshold.comparator == ">=" else "meets_threshold"
                        part_id = f"{control_id}::{metric_id}"
                        _sparse_derived[part_id] = control_id
                        _sparse_specs.append(
                            _ControlSpec(
                                control_id=part_id, kind="counterexample", required=required,
                                metric_ids=(metric_id,),
                                expected_behavior="stable negative must not reach stability threshold",
                                comparator=gate,  # type: ignore[arg-type]
                                threshold_value=float(gate_threshold.value),
                            )
                        )
                        _sparse_supplied[part_id] = {metric_id: float(sparse.negative_stability)}
                    continue
                _sparse_specs.append(
                    _ControlSpec(
                        control_id=control_id, kind="seed" if kind == "seed" else "null",
                        required=required, metric_ids=linked,
                        expected_behavior="seed/shuffled control recorded without gating",
                    )
                )
                _sparse_supplied[control_id] = {}

    if wants_temporal:
        assert temporal_input is not None
        if not temporal_metrics or (manifest_declared and not temporal_covered):
            omissions["sequence_trajectory_drift"] = "missing-declaration:sequence_trajectory_drift"
        else:
            temporal = _evaluate_temporal(temporal_input, config)
            drift_threshold = _sparse_thresholds.get("trajectory-stepwise-drift")
            for control_id in (*config.required_controls, *config.optional_controls):
                linked = tuple(config.control_metrics[control_id])
                if not set(linked) & set(temporal_metrics):
                    continue
                required = control_id in config.required_controls
                kind = config.control_kinds[control_id]
                if kind in _SUPPLIED_KINDS:
                    # The no-drift negative must NOT reach the drift
                    # threshold; repeatability metrics on the control are not
                    # gated (the control has no control of its own).
                    if drift_threshold is None:
                        part_id = f"{control_id}::trajectory-stepwise-drift"
                        _sparse_derived[part_id] = control_id
                        _sparse_specs.append(
                            _ControlSpec(
                                control_id=part_id, kind="counterexample", required=required,
                                metric_ids=("trajectory-stepwise-drift",),
                                expected_behavior="no-drift negative declares no drift threshold",
                                comparator="below_threshold", threshold_value=0.0,
                            )
                        )
                        _sparse_supplied[part_id] = {"trajectory-stepwise-drift": float("inf")}
                    else:
                        gate = "below_threshold" if drift_threshold.comparator == ">=" else "meets_threshold"
                        part_id = f"{control_id}::trajectory-stepwise-drift"
                        _sparse_derived[part_id] = control_id
                        _sparse_specs.append(
                            _ControlSpec(
                                control_id=part_id, kind="counterexample", required=required,
                                metric_ids=("trajectory-stepwise-drift",),
                                expected_behavior="no-drift negative must not reach drift threshold",
                                comparator=gate,  # type: ignore[arg-type]
                                threshold_value=float(drift_threshold.value),
                            )
                        )
                        _sparse_supplied[part_id] = {"trajectory-stepwise-drift": float(temporal.negative_drift)}
                    continue
                _sparse_specs.append(
                    _ControlSpec(
                        control_id=control_id, kind="shuffled" if kind == "shuffled" else "null",
                        required=required, metric_ids=linked,
                        expected_behavior="shuffled sequence recorded without gating",
                    )
                )
                _sparse_supplied[control_id] = {}
    if _sparse_specs:
        _sparse_statistics: dict[str, object] = {}
        if sparse is not None:
            for control_id in (*config.required_controls, *config.optional_controls):
                if config.control_kinds.get(control_id) == "seed" and set(config.control_metrics[control_id]) & set(sparse_metrics):
                    _sparse_statistics[control_id] = lambda rng, _s=sparse: {  # type: ignore[misc]
                        "sparse-cross-seed-stability": float(_s.seed_control_stability),
                        "fit_seed": float(_s.seed_control_fit_seed),
                    }
        if temporal is not None:
            for control_id in (*config.required_controls, *config.optional_controls):
                if config.control_kinds.get(control_id) == "shuffled" and set(config.control_metrics[control_id]) & set(temporal_metrics):
                    _sparse_statistics[control_id] = lambda rng, _t=temporal: {  # type: ignore[misc]
                        "trajectory-stepwise-drift": float(_t.shuffled_drift),
                    }
        _sparse_gating = _execute_plan(
            _ControlPlan(
                plan_id=f"{config.manifest_id}:sparse-temporal",
                repetitions=config.repetitions,
                confidence_level=config.confidence_level,
                evaluation_seed=config.evaluation_seed,
                control_seed=config.control_seed,
                training_seed=config.training_seed,
                controls=tuple(_sparse_specs),
            ),
            _sparse_supplied,
            statistics=_sparse_statistics or None,  # type: ignore[arg-type]
        )
        _sparse_central: dict[str, object] = {
            control_id: outcome.to_dict() for control_id, outcome in _sparse_gating.items()
        }
        for control_id in (*config.required_controls, *config.optional_controls):
            parts = [part for part, parent in _sparse_derived.items() if parent == control_id]
            if parts:
                statuses = [_sparse_gating[part].status for part in parts]
                control_outcomes[control_id] = "passed" if all(status in ("passed", "recorded") for status in statuses) else "failed"
                continue
            outcome = _sparse_gating.get(control_id)
            if outcome is not None:
                control_outcomes[control_id] = "passed" if outcome.status in ("passed", "recorded") else "failed"
        _sparse_blocked = sorted({_sparse_derived.get(part, part) for part in _failed_required(_sparse_gating)} & set(config.required_controls))
        if _sparse_blocked:
            control_outcomes.update({control_id: "failed" for control_id in _sparse_blocked})
    else:
        _sparse_central = {}
    identity = (
        sparse_input.representation_identity if sparse_input is not None
        else (temporal_input.representation_identity if temporal_input is not None else "")
    )
    return _DetectionContext(
        representation_identity=identity,
        sparse=sparse,
        temporal=temporal,
        control_outcomes=control_outcomes,
        central_outcomes=_sparse_central,
        omissions=omissions,
        distribution_free_sparse=distribution_free_sparse,
    )


def _decide(context: _DetectionContext, config: DetectionConfig) -> tuple[FamilyDetection, ...]:
    """Decide every configured family from one shared evaluation context."""
    thresholds = {item.metric_id: item for item in config.thresholds}
    detections: list[FamilyDetection] = []
    for family_id in config.family_ids:
        family_metrics = [m for m in config.metric_ids if METRIC_FAMILY.get(m) == family_id]
        if family_id in context.omissions and (
            (family_id == "sparse_feature_instability" and context.sparse is None)
            or (family_id == "sequence_trajectory_drift" and context.temporal is None)
        ):
            code = context.omissions[family_id]
            detections.append(
                FamilyDetection(
                    family_id=family_id,
                    outcome="unsupported",
                    claim_allowed=False,
                    observed_metrics={},
                    threshold_pass={},
                    control_outcomes=dict(context.control_outcomes),
                    evidence_status=(
                        {
                            "sparse_cross_seed_stability": "not_evaluated",
                            "sparse_feature_alignment": "not_evaluated",
                            "sparse_quality_measure": "not_evaluated",
                            "sparse_seed_control": "not_evaluated",
                            "sparse_non_applicable_negative_control": "not_evaluated",
                            "sparse_provenance": "not_evaluated",
                        }
                        if family_id == "sparse_feature_instability"
                        else {
                            "trajectory_alignment_identity": "not_evaluated",
                            "trajectory_stepwise_drift_metric": "not_evaluated",
                            "trajectory_null_sequence_control": "not_evaluated",
                            "trajectory_no_drift_negative_control": "not_evaluated",
                            "trajectory_repeatability": "not_evaluated",
                        }
                    ),
                    missing_evidence=(code,),
                    reason=(
                        "manifest predeclares no sparse/temporal metrics; refusing to borrow another family's label"
                        if code.startswith("missing-declaration")
                        else "family omitted: required axis/evidence not applicable to this input"
                    ),
                )
            )
            continue
        if family_id == "sparse_feature_instability":
            evaluation = context.sparse
            assert evaluation is not None
            observed = {
                "sparse-cross-seed-stability": evaluation.stability,
                "sparse-reconstruction-quality": evaluation.quality,
            }
            observed = {m: observed[m] for m in family_metrics if m in observed}
            verdicts = {m: thresholds[m].passes(observed[m]) for m in observed}
            evidence = {
                "sparse_cross_seed_stability": "observed",
                "sparse_feature_alignment": "observed",
                "sparse_quality_measure": "observed",
                "sparse_seed_control": "observed",
                "sparse_non_applicable_negative_control": "observed",
                "sparse_provenance": "observed",
            }
        else:
            evaluation = context.temporal
            assert evaluation is not None
            observed = {
                "trajectory-stepwise-drift": evaluation.drift,
                "trajectory-repeatability": evaluation.drift,
            }
            repeatability = float(evaluation.drift - evaluation.negative_drift)
            observed["trajectory-repeatability"] = repeatability
            observed = {m: observed[m] for m in family_metrics if m in observed}
            verdicts = {m: thresholds[m].passes(observed[m]) for m in observed}
            evidence = {
                "trajectory_alignment_identity": "observed",
                "trajectory_stepwise_drift_metric": "observed",
                "trajectory_null_sequence_control": "observed",
                "trajectory_no_drift_negative_control": "observed",
                "trajectory_repeatability": "observed",
            }
        decision = evaluate_claim(family_id, applicability="applicable", evidence_status=evidence)
        failed = sorted(
            control_id
            for control_id in config.required_controls
            if control_id in context.control_outcomes
            and context.control_outcomes[control_id] == "failed"
            and bool(set(config.control_metrics[control_id]) & set(family_metrics))
        )
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
        unmet = sorted(m for m in family_metrics if m in verdicts and not verdicts[m])
        if unmet:
            detections.append(
                FamilyDetection(
                    family_id=family_id,
                    outcome="inconclusive",
                    claim_allowed=False,
                    observed_metrics=dict(observed),
                    threshold_pass=dict(verdicts),
                    control_outcomes=dict(context.control_outcomes),
                    evidence_status=dict(evidence),
                    missing_evidence=tuple(f"threshold:{m}" for m in unmet),
                    reason=f"predeclared thresholds unmet; a sparse/drift score alone is not a diagnosis: {', '.join(unmet)}",
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
                    missing_evidence=tuple(decision.missing_evidence),
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
    target: SparseInput | TemporalInput | LatentValue,
    config: DetectionConfig,
    *,
    controls: Mapping[str, SparseInput | TemporalInput | LatentValue] | None = None,
) -> tuple[FamilyDetection, ...]:
    """Evaluate the configured family on one bound input plus named controls.

    Focused unit-check seam: evaluates its own context, so composing it with a
    separate :func:`detection_payload` call evaluates twice. Production paths
    (including :func:`make_detect_executor`) must use :func:`evaluate_detection`.
    """
    if not isinstance(config, DetectionConfig):
        raise DetectionError("config must be a DetectionConfig")
    if not isinstance(target, (SparseInput, TemporalInput, LatentValue)):
        raise DetectionError("target must be a SparseInput, TemporalInput, or LatentValue")
    supplied = dict(controls or {})
    if supplied:
        raise DetectionError("this detector carries negative controls inside the bound input; no external control batches are accepted")
    return _decide(_evaluate_context(target, config, supplied), config)


def detection_payload(
    detections: Sequence[FamilyDetection],
    config: DetectionConfig,
    target: SparseInput | TemporalInput | LatentValue,
    *,
    control_metrics: Mapping[str, Mapping[str, float]] | None = None,
    context: _DetectionContext | None = None,
) -> dict[str, object]:
    """Assemble the canonical machine-readable detect-stage payload."""
    if not detections:
        raise DetectionError("detections must not be empty")
    resolved = context if context is not None else _evaluate_context(target, config, {})
    try:
        canonical_json([detection.to_dict() for detection in detections])
    except PortableNodeError as exc:
        raise DetectionError(f"detections are not canonical JSON: {exc}") from exc
    supplied_metrics = {control_id: dict(metrics) for control_id, metrics in (control_metrics or {}).items()}
    for control_id in supplied_metrics:
        if control_id not in config.control_metrics:
            raise DetectionError(f"unknown control reference: {control_id!r}")
    measurements: dict[str, object] = {"representation_identity": resolved.representation_identity}
    if resolved.sparse is not None:
        evaluation = resolved.sparse
        measurements["sparse"] = {
            "alignment_method": "decoder-cosine-matching",
            "estimator_digests": list(evaluation.estimator_digests),
            "matched_cosines": list(evaluation.matched_cosines),
            "n_components": int(evaluation.n_components),
            "negative_stability": float(evaluation.negative_stability),
            "seed_axis": list(evaluation.seeds),
            "seed_control_fit_seed": int(evaluation.seed_control_fit_seed),
            "seed_control_stability": float(evaluation.seed_control_stability),
            "stability": float(evaluation.stability),
            "quality": float(evaluation.quality),
            "uncertainty": {key: dict(item) for key, item in evaluation.uncertainty.items()},
        }
    if resolved.temporal is not None:
        evaluation = resolved.temporal
        measurements["temporal"] = {
            "drift_metric": "indexwise-distance",
            "n_steps": int(evaluation.n_steps),
            "drift": float(evaluation.drift),
            "negative_drift": float(evaluation.negative_drift),
            "shuffled_drift": float(evaluation.shuffled_drift),
            "reversed_drift": float(evaluation.reversed_drift),
            "stepwise": list(evaluation.stepwise),
            "uncertainty": {key: dict(item) for key, item in evaluation.uncertainty.items()},
        }
    if resolved.omissions:
        measurements["omissions"] = dict(resolved.omissions)
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
    for control_id in (*config.required_controls, *config.optional_controls):
        linked = config.control_metrics[control_id]
        entry: dict[str, float] = {}
        kind = config.control_kinds[control_id]
        if context.sparse is not None and kind in _SUPPLIED_KINDS:
            if "sparse-cross-seed-stability" in linked:
                entry["sparse-cross-seed-stability"] = context.sparse.negative_stability
            if "sparse-reconstruction-quality" in linked:
                entry["sparse-reconstruction-quality"] = context.sparse.quality
        if context.temporal is not None and kind in _SUPPLIED_KINDS:
            if "trajectory-stepwise-drift" in linked:
                entry["trajectory-stepwise-drift"] = context.temporal.negative_drift
            if "trajectory-repeatability" in linked:
                entry["trajectory-repeatability"] = context.temporal.drift - context.temporal.negative_drift
        if entry:
            table[control_id] = entry
    return table


def evaluate_detection(
    target: SparseInput | TemporalInput | LatentValue,
    config: DetectionConfig,
    controls: Mapping[str, SparseInput | TemporalInput | LatentValue],
) -> tuple[tuple[FamilyDetection, ...], dict[str, object]]:
    """Evaluate one input plus controls exactly once and return decisions plus payload.

    This is the single-evaluation seam every caller must use: every dictionary
    fit and every trajectory scoring is computed once inside one
    :class:`_DetectionContext`, then shared by family decisions and payload
    assembly. Sparse uncertainty resamples fitted matched cosines without
    refitting; temporal uncertainty resamples fitted stepwise distances.
    """
    if not isinstance(config, DetectionConfig):
        raise DetectionError("config must be a DetectionConfig")
    if not isinstance(target, (SparseInput, TemporalInput, LatentValue)):
        raise DetectionError("target must be a SparseInput, TemporalInput, or LatentValue")
    supplied = dict(controls)
    context = _evaluate_context(target, config, supplied)
    detections = _decide(context, config)
    payload = detection_payload(detections, config, target, control_metrics=_control_table(context, config), context=context)
    return detections, payload


def make_detect_executor(
    target: SparseInput | TemporalInput | LatentValue,
    config: DetectionConfig,
    *,
    controls: Mapping[str, SparseInput | TemporalInput | LatentValue] | None = None,
    version: str = "sparse-temporal-detector-v1",
) -> Any:
    """Build a supplied ``detect``-stage executor bound to one input and config.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable verifies the invocation targets ``detect`` with the expected
    request/manifest identity, evaluates :func:`evaluate_detection` once, and
    returns a ``completed`` :class:`StageOutput` whose payload carries
    observations, family evidence, control outcomes, uncertainty, and claim
    gating. No algorithm enters ``DiagnosticWorkflow`` itself.
    """
    _non_empty_string(version, name="version")
    if not isinstance(target, (SparseInput, TemporalInput, LatentValue)):
        raise DetectionError("target must be a SparseInput, TemporalInput, or LatentValue")
    if not isinstance(config, DetectionConfig):
        raise DetectionError("config must be a DetectionConfig")
    frozen_target = target
    frozen_controls = dict(controls or {})

    def _execute(invocation: StageInvocation) -> StageOutput:
        from latent_anything._diagnostic_workflow import StageContractError as _ContractError

        if invocation.stage != "detect":
            raise _ContractError(f"detect executor received stage {invocation.stage!r}")
        if invocation.request.manifest_id != config.manifest_id:
            raise _ContractError("detect executor manifest identity mismatch")
        _, payload = evaluate_detection(frozen_target, config, frozen_controls)
        aggregate_outcome: Literal["completed", "unsupported"] = "completed"
        return StageOutput(stage="detect", outcome=aggregate_outcome, payload=payload, artifact_refs=())

    _execute.detector_version = version  # type: ignore[attr-defined]
    return _execute


__all__ = [
    "METRIC_FAMILY",
    "SUPPORTED_FAMILIES",
    "ClaimOutcome",
    "DetectionConfig",
    "DetectionError",
    "FamilyDetection",
    "FamilyThreshold",
    "SparseInput",
    "TemporalInput",
    "detect_families",
    "detection_config_from_manifest",
    "detection_payload",
    "evaluate_detection",
    "make_detect_executor",
]
