"""Redundancy, superposition, separability, and probe-leakage detection (Sprint 80.10).

Private detector-family adapter behind the frozen taxonomy/workflow contracts for
``redundancy_superposition`` and ``separability_probe_leakage``. It consumes bound
representations plus predeclared metric/control/threshold configuration and emits
machine-readable observations, taxonomy family-evidence statuses
(``evaluate_claim``), control outcomes, seeded uncertainty metadata, and honest
``supported``/``inconclusive``/``unsupported`` outcomes.

Separability is never promoted from held-out accuracy alone: a supported claim
requires leakage-safe disjoint train/eval splits with distinct split identities,
a bounded-capacity declaration, a seeded label-randomization gap, a
non-separable negative control, and every required control passed. Any leakage,
overlap, missing/failed control, memorization/capacity violation, or weak
randomization gap blocks promotion (``inconclusive``, ``claim_allowed=False``).

Redundancy/superposition reports declared correlation and sparse-overcomplete
evidence with an explicit counterexample and a randomized-feature control under
predeclared decision rules. Correlation alone can never promote the claim: both
metrics must agree, otherwise the outcome is ``inconclusive`` and no semantic
superposition is inferred. Where the manifest predeclares no metrics for this
family, the detector returns honest ``unsupported`` rather than borrowing
another family's thresholds.

Reused existing primitives (no new estimators):

- ``probes._fast_probe`` is the single shared probe-fitting seam: training-only
  ``StandardScaler`` plus ``LogisticRegression`` (C=1.0, lbfgs, balanced); the
  detector's splits are caller-declared and validated disjoint, never
  recomputed, and accuracy is recomputed against leakage-safe labels.
- ``fit_covariance`` from ``geometry`` supplies the single regularized covariance
  behind the correlation measure (one fit per evaluated batch, plus one fit per
  bootstrap draw for the correlation interval).
- ``DictionaryLearning`` supplies the sparse overcomplete comparison as
  dictionary-atom coherence (one fit per evaluated batch, ``max_iter=100``,
  seeded by the manifest control seed); the sharing interval resamples the
  fitted pair-coherence values without refitting.
- ``evaluate_claim`` gates taxonomy evidence; ``validate_manifest`` and
  ``canonical_json`` enforce the frozen contracts.

Single-evaluation seam: :func:`evaluate_detection` fits every probe and every
dictionary exactly once per executor call. Separability bootstrap resamples the
fitted probe predictions without refitting through the central
``_statistical_controls`` executor (identity-derived ``evaluation`` streams);
correlation uncertainty refits ``fit_covariance`` once per central draw (no
dictionary refit); sharing uncertainty resamples the fitted coherence values
without refitting. Label-randomization and column-shuffle nulls run as central
``randomized``/``shuffled`` controls. One shared context feeds family
decisions and payload assembly.

The redundancy column-shuffle null executes only under an explicitly declared
config control linked to the redundancy metrics; with no such declaration the
null never runs and no invented identity enters the payload controls table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np

from latent_anything.probes import _fast_probe

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
from latent_anything.diagnostics import DiagnosticRequest
from latent_anything.dictionary_learning import DictionaryLearning, DictionaryLearningConfig
from latent_anything.geometry import fit_covariance
from latent_anything.latent_value import LatentValue

SUPPORTED_FAMILIES: tuple[str, ...] = ("redundancy_superposition", "separability_probe_leakage")
"""Detector scope for this task. Exact taxonomy identifiers; nothing else is evaluated."""

METRIC_FAMILY: Mapping[str, str] = MappingProxyType(
    {
        "heldout-probe-accuracy": "separability_probe_leakage",
        "probe-leakage-gap": "separability_probe_leakage",
        "feature-max-abs-correlation": "redundancy_superposition",
        "sparse-feature-sharing": "redundancy_superposition",
    }
)
"""Predeclared metric-to-family wiring. Unknown metrics reject."""

DECLARED_CAPACITY = "linear-logreg-C1.0-standardized"
"""The only probe capacity this detector may promote.

The headline claim is bound to a linear logistic probe (C=1.0, lbfgs,
training-only standardization). Any other declaration fails the capacity
control instead of widening expressivity silently.
"""

_SUPPLIED_KINDS = frozenset({"counterexample", "negative"})
"""Manifest control kinds that require caller-supplied batch data.

``capacity``, ``randomized``, and ``null`` controls are derived by the detector
from the target batch under manifest seeds, so supplying batch data for them is
a fail-closed error rather than silently ignored input.
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
    repetitions: int
    confidence_level: float
    training_seed: int
    evaluation_seed: int
    control_seed: int

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


def detection_config_from_manifest(request: DiagnosticRequest, manifest: Mapping[str, object]) -> DetectionConfig:
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
        repetitions=int(repetitions),
        confidence_level=_finite_number(uncertainty.get("confidence_level"), name="uncertainty.confidence_level"),
        training_seed=_non_negative_int(training[0], name="seeds.training[0]"),
        evaluation_seed=_non_negative_int(evaluation[0], name="seeds.evaluation[0]"),
        control_seed=_non_negative_int(controls_seed[0], name="seeds.controls[0]"),
    )


@dataclass(frozen=True)
class FamilyDetection:
    """Evidence-bound outcome for one taxonomy family on one batch."""

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


def _batch_matrix(value: LatentValue) -> np.ndarray:
    if not isinstance(value, LatentValue):
        raise DetectionError("value must be a LatentValue")
    data = np.asarray(value.to_numpy(), dtype=np.float64)
    if data.ndim != 2:
        raise DetectionError(f"incompatible rank: detection requires one 2D (n_samples, dim) batch, got {data.ndim}D")
    if data.shape[0] < 2 or data.shape[1] < 1:
        raise DetectionError(f"undersampled batch: detection requires at least two samples, got shape {data.shape}")
    if not np.isfinite(data).all():
        raise DetectionError("batch contains non-finite values")
    return data


@dataclass(frozen=True)
class LabeledBatch:
    """Bound labeled representation with immutable labels, sample identities, and declared splits.

    Train/eval partitions are caller-declared (never recomputed): index sets must
    be non-empty, in range, and disjoint, sample identities must be unique across
    the batch, and the split identities must be non-empty and distinct. Any
    leakage or overlap fails closed at construction.
    """

    value: LatentValue
    labels: tuple[object, ...]
    sample_ids: tuple[str, ...]
    train_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    train_split_identity: str
    eval_split_identity: str
    capacity: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, LatentValue):
            raise DetectionError("value must be a LatentValue")
        matrix = _batch_matrix(self.value)
        n = int(matrix.shape[0])
        labels = tuple(self.labels) if not isinstance(self.labels, tuple) else self.labels
        sample_ids = tuple(self.sample_ids) if not isinstance(self.sample_ids, tuple) else self.sample_ids
        train_indices = tuple(self.train_indices) if not isinstance(self.train_indices, tuple) else self.train_indices
        eval_indices = tuple(self.eval_indices) if not isinstance(self.eval_indices, tuple) else self.eval_indices
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "train_indices", train_indices)
        object.__setattr__(self, "eval_indices", eval_indices)
        if len(labels) != n:
            raise DetectionError(f"labels cover {len(labels)} samples but the batch holds {n}")
        if len(sample_ids) != n:
            raise DetectionError(f"sample_ids cover {len(sample_ids)} samples but the batch holds {n}")
        for position, sample_id in enumerate(sample_ids):
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise DetectionError(f"sample_ids[{position}] must be a non-empty string")
        if len(set(sample_ids)) != n:
            raise DetectionError("sample identities must be unique within one batch")
        label_array = np.asarray(labels)
        if label_array.ndim != 1 or label_array.shape[0] != n:
            raise DetectionError("labels must be a one-dimensional vector covering every sample")
        if np.issubdtype(label_array.dtype, np.floating) and not np.isfinite(label_array).all():
            raise DetectionError("labels contain non-finite values")
        if len(np.unique(label_array)) < 2:
            raise DetectionError("labels must contain at least two classes")
        for name, indices in (("train_indices", train_indices), ("eval_indices", eval_indices)):
            if not indices:
                raise DetectionError(f"{name} must not be empty")
            if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in indices):
                raise DetectionError(f"{name} must hold integer positions")
            if any(int(item) < 0 or int(item) >= n for item in indices):
                raise DetectionError(f"{name} holds a position outside the batch of {n} samples")
            if len(set(int(item) for item in indices)) != len(indices):
                raise DetectionError(f"{name} must not repeat a sample position")
        if set(int(item) for item in train_indices) & set(int(item) for item in eval_indices):
            raise DetectionError("train/eval split leaks: a sample position appears on both sides")
        train_ids = {sample_ids[int(item)] for item in train_indices}
        eval_ids = {sample_ids[int(item)] for item in eval_indices}
        if train_ids & eval_ids:
            raise DetectionError("train/eval split leaks: a sample identity appears on both sides")
        _non_empty_string(self.train_split_identity, name="train_split_identity")
        _non_empty_string(self.eval_split_identity, name="eval_split_identity")
        if self.train_split_identity == self.eval_split_identity:
            raise DetectionError("train/eval split identities must be distinct")
        _non_empty_string(self.capacity, name="capacity")


def _probe_predictions(
    train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, query_y: np.ndarray, *, seed: int
) -> np.ndarray:
    """Fit one bounded linear probe on train rows and predict query rows.

    Thin wrapper over the shared ``probes._fast_probe`` seam (identical
    estimator: training-only ``StandardScaler``, ``LogisticRegression``
    C=1.0/lbfgs/balanced). ``query_y`` is used only for the shared seam's
    accuracy bookkeeping; this detector recomputes accuracy against its own
    leakage-safe labels. Exactly one classifier fit per call; callers must not
    refit to obtain intervals.
    """
    try:
        result = _fast_probe(
            np.asarray(train_x, dtype=np.float64),
            np.asarray(train_y),
            np.asarray(query_x, dtype=np.float64),
            np.asarray(query_y),
            seed,
        )
    except ValueError as exc:
        raise DetectionError(f"probe fit failed: {exc}") from exc
    return np.asarray(result.predictions)


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
class _SeparabilityEvaluation:
    """One separability input evaluated once: four probe fits, no refits after."""

    n_train: int
    n_eval: int
    dim: int
    n_classes: int
    n_params: int
    capacity_declared: str
    capacity_passed: bool
    heldout_accuracy: float
    randomized_accuracy: float
    leakage_gap: float
    swap_accuracy: float
    negative_accuracy: float
    eval_predictions: tuple[float, ...]
    eval_labels: tuple[float, ...]
    randomized_predictions: tuple[float, ...]
    uncertainty: Mapping[str, dict[str, object]]


@dataclass(frozen=True)
class _RedundancyEvaluation:
    """One redundancy batch decomposed once: one covariance fit, one dictionary fit."""

    n_samples: int
    dim: int
    max_abs_correlation: float
    sparse_sharing: float
    uncertainty: Mapping[str, dict[str, object]]


@dataclass(frozen=True)
class _DetectionContext:
    """One executor call evaluated exactly once and shared by decisions and payload."""

    representation_identity: str
    separability: _SeparabilityEvaluation | None
    redundancy: _RedundancyEvaluation | None
    null_redundancy: _RedundancyEvaluation | None
    counterexample_redundancy: _RedundancyEvaluation | None
    control_outcomes: Mapping[str, str]
    central_outcomes: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_outcomes", dict(self.control_outcomes))
        object.__setattr__(self, "central_outcomes", dict(self.central_outcomes))


def _accuracy_of(predictions: np.ndarray, truths: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictions).ravel() == np.asarray(truths).ravel()))


def _randomized_statistic(
    rng: np.random.Generator,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    heldout: float,
    training_seed: int,
) -> Mapping[str, float]:
    """Permute labels with the supplied central RNG, refit, and score once."""
    shuffled_train_y = np.asarray(rng.permutation(np.asarray(train_y).ravel()))
    randomized_predictions = _probe_predictions(
        train_x, shuffled_train_y, eval_x, eval_y, seed=training_seed
    )
    randomized = _accuracy_of(randomized_predictions, eval_y)
    return {
        "heldout_accuracy": float(heldout),
        "leakage_gap": float(float(heldout) - randomized),
        "randomized_accuracy": float(randomized),
        "randomized_predictions": [float(item) for item in np.asarray(randomized_predictions).ravel()],
    }


def _evaluate_separability(
    batch: LabeledBatch, negative: LabeledBatch, config: DetectionConfig
) -> _SeparabilityEvaluation:
    """Run the headline probe, its seeded controls, and the split-swap null exactly once.

    Four probe fits via the shared ``probes._fast_probe`` seam; uncertainty
    resamples the fitted predictions without refitting.
    """
    matrix = _batch_matrix(batch.value)
    labels = np.asarray(batch.labels)
    train = np.asarray([int(item) for item in batch.train_indices], dtype=np.int64)
    evaluation = np.asarray([int(item) for item in batch.eval_indices], dtype=np.int64)
    train_x, train_y = matrix[train], labels[train]
    eval_x, eval_y = matrix[evaluation], labels[evaluation]
    if len(np.unique(train_y)) < 2:
        raise DetectionError("train split must contain at least two classes")
    if len(np.unique(eval_y)) < 2:
        raise DetectionError("eval split must contain at least two classes")

    eval_predictions = _probe_predictions(train_x, train_y, eval_x, eval_y, seed=config.training_seed)
    heldout = _accuracy_of(eval_predictions, eval_y)
    # Central randomized control: the actual label permutation plus the probe
    # refit/evaluation runs inside the executor-supplied stream. Observed
    # accuracy/gap bind directly from the returned ControlOutcome.
    _shuffle_outcome = _central_control(
        control_id="control-label-randomization",
        base_seed=config.control_seed,
        seed_role="control",
        required=False,
        kind="randomized",
        metric_ids=("heldout-probe-accuracy", "probe-leakage-gap"),
        expected_behavior="one seeded permutation of the training labels",
        statistic=lambda rng: _randomized_statistic(rng, train_x, train_y, eval_x, eval_y, heldout, config.training_seed),
    )
    if _shuffle_outcome.status != "recorded":
        raise DetectionError("central randomized control must record, not gate")
    randomized = float(_shuffle_outcome.observed["randomized_accuracy"])
    gap = float(_shuffle_outcome.observed["leakage_gap"])
    _randomized_detail = dict(_shuffle_outcome.detail or {})
    randomized_predictions = np.asarray(
        [float(item) for item in _randomized_detail["randomized_predictions"]], dtype=np.float64
    )


    swap_predictions = _probe_predictions(eval_x, eval_y, train_x, train_y, seed=config.training_seed)
    swap = _accuracy_of(swap_predictions, train_y)

    negative_matrix = _batch_matrix(negative.value)
    if negative_matrix.shape[1] != matrix.shape[1]:
        raise DetectionError(
            f"negative control has {negative_matrix.shape[1]} features but the target has {matrix.shape[1]}"
        )
    negative_labels = np.asarray(negative.labels)
    negative_train = np.asarray([int(item) for item in negative.train_indices], dtype=np.int64)
    negative_eval = np.asarray([int(item) for item in negative.eval_indices], dtype=np.int64)
    negative_predictions = _probe_predictions(
        negative_matrix[negative_train],
        negative_labels[negative_train],
        negative_matrix[negative_eval],
        negative_labels[negative_eval],
        seed=config.training_seed,
    )
    negative_accuracy = _accuracy_of(negative_predictions, negative_labels[negative_eval])

    n_classes = int(len(np.unique(labels)))
    n_params = int(matrix.shape[1] * (1 if n_classes == 2 else n_classes) + (1 if n_classes == 2 else n_classes))
    capacity_passed = bool(batch.capacity == DECLARED_CAPACITY and n_params <= int(train.shape[0]))

    n_eval = int(evaluation.shape[0])
    flat_eval = np.asarray(eval_predictions).ravel()
    flat_randomized = np.asarray(randomized_predictions).ravel()
    flat_truth = np.asarray(eval_y).ravel()

    def _heldout_draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, n_eval, size=n_eval)
        return float(np.mean(flat_eval[positions] == flat_truth[positions]))

    def _gap_draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, n_eval, size=n_eval)
        held_draw = float(np.mean(flat_eval[positions] == flat_truth[positions]))
        random_draw = float(np.mean(flat_randomized[positions] == flat_truth[positions]))
        return float(held_draw - random_draw)

    _, held_interval = _central_bootstrap(
        control_id="bootstrap:heldout-probe-accuracy",
        base_seed=config.evaluation_seed,
        seed_role="evaluation",
        repetitions=config.repetitions,
        confidence_level=config.confidence_level,
        required=False,
        kind="bootstrap",
        metric_ids=("heldout-probe-accuracy",),
        expected_behavior="seeded resampling of fitted probe predictions",
        draw=_heldout_draw,
    )
    _, gap_interval = _central_bootstrap(
        control_id="bootstrap:probe-leakage-gap",
        base_seed=config.evaluation_seed,
        seed_role="evaluation",
        repetitions=config.repetitions,
        confidence_level=config.confidence_level,
        required=False,
        kind="bootstrap",
        metric_ids=("probe-leakage-gap",),
        expected_behavior="seeded resampling of fitted probe predictions",
        draw=_gap_draw,
    )
    uncertainty = {
        "heldout-probe-accuracy": dict(held_interval),
        "probe-leakage-gap": dict(gap_interval),
    }
    for metric_id, number in (
        ("heldout-probe-accuracy", heldout),
        ("probe-leakage-gap", gap),
        ("swap-accuracy", swap),
        ("negative-accuracy", negative_accuracy),
    ):
        if not np.isfinite(number):
            raise DetectionError(f"metric {metric_id!r} produced a non-finite value")
    return _SeparabilityEvaluation(
        n_train=int(train.shape[0]),
        n_eval=n_eval,
        dim=int(matrix.shape[1]),
        n_classes=n_classes,
        n_params=n_params,
        capacity_declared=batch.capacity,
        capacity_passed=capacity_passed,
        heldout_accuracy=heldout,
        randomized_accuracy=randomized,
        leakage_gap=gap,
        swap_accuracy=swap,
        negative_accuracy=negative_accuracy,
        eval_predictions=tuple(float(item) for item in flat_eval),
        eval_labels=tuple(float(item) for item in flat_truth),
        randomized_predictions=tuple(float(item) for item in flat_randomized),
        uncertainty=uncertainty,
    )


def _metrics_met(verdicts: Mapping[str, bool], family_metrics: Sequence[str]) -> bool:
    return all(verdicts[metric_id] for metric_id in family_metrics if metric_id in verdicts)


def _redundancy_metrics(matrix: np.ndarray, *, control_seed: int) -> tuple[float, float, tuple[float, ...]]:
    """Compute correlation dependence and dictionary-atom sharing for one batch.

    Exactly one covariance fit and one dictionary fit; per-atom coherence gains
    are returned so uncertainty resamples statistics without refitting.
    Sharing is the maximum absolute cosine between distinct normalized
    dictionary atoms (atom coherence): near-duplicate atoms mean the sparse
    overcomplete basis reuses the same direction, the observable signature of
    feature sharing. Bounded in ``[0, 1]``; healthy independent features keep
    atoms incoherent.
    """
    dim = int(matrix.shape[1])
    if dim < 2:
        raise DetectionError("redundancy requires at least two feature dimensions")
    variances = np.var(matrix, axis=0)
    if not np.isfinite(variances).all() or bool((variances <= 0.0).any()):
        raise DetectionError("redundancy requires strictly positive variance on every feature")
    _, covariance = fit_covariance(matrix, reg_coef=1e-6)
    scales = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scales, scales)
    if not np.isfinite(correlation).all():
        raise DetectionError("correlation profile is non-finite")
    off_diagonal = np.abs(correlation - np.diag(np.diag(correlation)))
    max_abs = float(np.max(off_diagonal))

    learner = DictionaryLearning(
        DictionaryLearningConfig(n_components=2 * dim, max_iter=100, random_state=control_seed),
    )
    learner.fit(matrix)
    atoms = np.asarray(learner.components_, dtype=np.float64)
    norms = np.linalg.norm(atoms, axis=1)
    if not np.isfinite(norms).all() or bool((norms <= 0.0).any()):
        raise DetectionError("sparse comparison produced degenerate atoms")
    normalized = atoms / norms[:, np.newaxis]
    gram = np.abs(normalized @ normalized.T)
    np.fill_diagonal(gram, 0.0)
    sharing = float(np.max(gram))
    gains = tuple(float(gram[i, j]) for i in range(gram.shape[0]) for j in range(i + 1, gram.shape[0]))
    if not np.isfinite(max_abs) or not np.isfinite(sharing):
        raise DetectionError("redundancy metrics produced a non-finite value")
    return max_abs, sharing, gains


def _evaluate_redundancy(matrix: np.ndarray, config: DetectionConfig) -> _RedundancyEvaluation:
    """Decompose one batch once and summarize both redundancy statistics.

    The dictionary is fitted exactly once; the sharing interval resamples the
    fitted pair-coherence values without refitting. The correlation interval
    refits ``fit_covariance`` once per bootstrap draw on resampled rows.
    """
    max_abs, sharing, gains = _redundancy_metrics(matrix, control_seed=config.control_seed)
    n = int(matrix.shape[0])
    gain_array = np.asarray(gains, dtype=np.float64)

    def _correlation_draw(rng: np.random.Generator) -> float:
        rows = rng.integers(0, n, size=n)
        block = matrix[rows]
        _, covariance = fit_covariance(block, reg_coef=1e-6)
        scales = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scales, scales)
        off_diagonal = np.abs(correlation - np.diag(np.diag(correlation)))
        return float(np.max(off_diagonal))

    def _sharing_draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, gain_array.shape[0], size=gain_array.shape[0])
        return float(np.max(gain_array[positions]))

    _, correlation_interval = _central_bootstrap(
        control_id="bootstrap:feature-max-abs-correlation",
        base_seed=config.evaluation_seed,
        seed_role="evaluation",
        repetitions=config.repetitions,
        confidence_level=config.confidence_level,
        required=False,
        kind="bootstrap",
        metric_ids=("feature-max-abs-correlation",),
        expected_behavior="seeded resampling of the fitted batch",
        draw=_correlation_draw,
    )
    _, sharing_interval = _central_bootstrap(
        control_id="bootstrap:sparse-feature-sharing",
        base_seed=config.evaluation_seed,
        seed_role="evaluation",
        repetitions=config.repetitions,
        confidence_level=config.confidence_level,
        required=False,
        kind="bootstrap",
        metric_ids=("sparse-feature-sharing",),
        expected_behavior="seeded resampling of fitted coherence values",
        draw=_sharing_draw,
    )
    uncertainty = {
        "feature-max-abs-correlation": dict(correlation_interval),
        "sparse-feature-sharing": dict(sharing_interval),
    }
    return _RedundancyEvaluation(
        n_samples=n,
        dim=int(matrix.shape[1]),
        max_abs_correlation=max_abs,
        sparse_sharing=sharing,
        uncertainty=uncertainty,
    )


def _evaluate_context(
    target: LabeledBatch | LatentValue,
    config: DetectionConfig,
    supplied: Mapping[str, LabeledBatch | LatentValue],
) -> _DetectionContext:
    """Validate and evaluate the target plus every control batch exactly once."""
    for control_id in supplied:
        if control_id not in (*config.required_controls, *config.optional_controls):
            raise DetectionError(f"unknown control reference: {control_id!r}")
        kind = config.control_kinds[control_id]
        if kind not in _SUPPLIED_KINDS:
            raise DetectionError(f"control {control_id!r} is derived by the detector; no batch data is accepted")
    for control_id in config.required_controls:
        if config.control_kinds[control_id] in _SUPPLIED_KINDS and control_id not in supplied:
            raise DetectionError(f"required controls are missing batch data: {control_id}")

    wants_separability = "separability_probe_leakage" in config.family_ids
    wants_redundancy = "redundancy_superposition" in config.family_ids
    labeled = target if isinstance(target, LabeledBatch) else None
    if wants_separability and labeled is None:
        raise DetectionError("separability evaluation requires bound labels, sample identities, and declared splits")
    matrix = _batch_matrix(labeled.value if labeled is not None else cast(LatentValue, target))
    separability: _SeparabilityEvaluation | None = None
    control_outcomes: dict[str, str] = {}
    _sep_central: dict[str, object] = {}
    _central_folded: dict[str, object] = {}
    _all_central: dict[str, object] = {}
    if wants_separability:
        assert labeled is not None
        negative_id: str | None = None
        for control_id in (*config.required_controls, *config.optional_controls):
            if config.control_kinds[control_id] in _SUPPLIED_KINDS and METRIC_FAMILY.get(
                config.control_metrics[control_id][0], ""
            ) == "separability_probe_leakage":
                negative_id = control_id
        if negative_id is None:
            raise DetectionError("separability evaluation requires a supplied non-separable negative control")
        negative = supplied[negative_id]
        if not isinstance(negative, LabeledBatch):
            raise DetectionError(f"control {negative_id!r} must carry bound labels and declared splits")
        separability = _evaluate_separability(labeled, negative, config)
        # Central status/gating for separability controls: capacity gates on
        # the declared bound, randomized gates on the executed leakage gap,
        # supplied negatives gate per metric (must NOT meet headline
        # thresholds), and null/split-swap controls record without gating.
        _sep_thresholds = {item.metric_id: item for item in config.thresholds}
        _sep_specs: list[_ControlSpec] = []
        _sep_supplied: dict[str, Mapping[str, float] | None] = {}
        for control_id in (*config.required_controls, *config.optional_controls):
            linked = tuple(config.control_metrics[control_id])
            if not set(linked) & {"heldout-probe-accuracy", "probe-leakage-gap"}:
                continue
            required = control_id in config.required_controls
            kind = config.control_kinds[control_id]
            if kind == "capacity":
                gate_value = 1.0 if separability.capacity_passed else 0.0
                _sep_supplied[control_id] = {"capacity": gate_value}
                _sep_specs.append(
                    _ControlSpec(
                        control_id=control_id, kind="capacity", required=required,
                        metric_ids=("capacity",),
                        expected_behavior="declared probe capacity bound",
                        comparator="meets_threshold", threshold_value=1.0,
                    )
                )
                continue
            if kind == "randomized":
                gate_threshold = _sep_thresholds.get("probe-leakage-gap")
                gate = "meets_threshold" if gate_threshold is None or gate_threshold.comparator == ">=" else "below_threshold"
                bound = float(gate_threshold.value) if gate_threshold is not None else 0.0
                _sep_supplied[control_id] = {"probe-leakage-gap": float(separability.leakage_gap)}
                _sep_specs.append(
                    _ControlSpec(
                        control_id=control_id, kind="randomized", required=required,
                        metric_ids=("probe-leakage-gap",),
                        expected_behavior="label-randomization gap under predeclared threshold",
                        comparator=gate,  # type: ignore[arg-type]
                        threshold_value=bound,
                    )
                )
                continue
            if kind in _SUPPLIED_KINDS:
                held_threshold = _sep_thresholds.get("heldout-probe-accuracy")
                if held_threshold is None:
                    _sep_supplied[control_id] = {"heldout-probe-accuracy": float(separability.negative_accuracy)}
                    _sep_specs.append(
                        _ControlSpec(
                            control_id=control_id, kind="counterexample", required=required,
                            metric_ids=("heldout-probe-accuracy",),
                            expected_behavior="non-separable negative must not meet headline accuracy",
                            comparator="below_threshold", threshold_value=0.0,
                        )
                    )
                    continue
                gate = "below_threshold" if held_threshold.comparator == ">=" else "meets_threshold"
                _sep_supplied[control_id] = {"heldout-probe-accuracy": float(separability.negative_accuracy)}
                _sep_specs.append(
                    _ControlSpec(
                        control_id=control_id, kind="counterexample", required=required,
                        metric_ids=("heldout-probe-accuracy",),
                        expected_behavior="non-separable negative must not meet headline accuracy",
                        comparator=gate,  # type: ignore[arg-type]
                        threshold_value=float(held_threshold.value),
                    )
                )
                continue
            _sep_specs.append(
                _ControlSpec(
                    control_id=control_id, kind="null", required=required,
                    metric_ids=linked,
                    expected_behavior="split direction recorded without gating",
                )
            )
            _sep_supplied[control_id] = {}
        _sep_gating = _execute_plan(
            _ControlPlan(
                plan_id=f"{config.manifest_id}:separability",
                repetitions=config.repetitions,
                confidence_level=config.confidence_level,
                evaluation_seed=config.evaluation_seed,
                control_seed=config.control_seed,
                training_seed=config.training_seed,
                controls=tuple(_sep_specs),
            ),
            _sep_supplied,
        )
        for control_id, outcome in _sep_gating.items():
            control_outcomes[control_id] = "passed" if outcome.status in ("passed", "recorded") else "failed"
        _sep_central: dict[str, object] = {control_id: outcome.to_dict() for control_id, outcome in _sep_gating.items()}
        _sep_blocked = set(_failed_required(_sep_gating))
        if _sep_blocked:
            control_outcomes.update({control_id: "failed" for control_id in _sep_blocked})

    redundancy: _RedundancyEvaluation | None = None
    null_redundancy: _RedundancyEvaluation | None = None
    counterexample_redundancy: _RedundancyEvaluation | None = None
    if wants_redundancy and any(
        METRIC_FAMILY.get(metric_id) == "redundancy_superposition" for metric_id in config.metric_ids
    ):
        redundancy = _evaluate_redundancy(matrix, config)
        # Central null control: the actual column permutation plus the full
        # redundancy evaluation runs inside ``execute_plan`` under the exact
        # declared manifest control identity. Observed metrics bind directly
        # from that ControlOutcome (no replay, no second transform).
        _null_declared: str | None = None
        for _candidate in (*config.required_controls, *config.optional_controls):
            if config.control_kinds[_candidate] == "randomized" and set(config.control_metrics[_candidate]) & {"feature-max-abs-correlation", "sparse-feature-sharing"}:
                _null_declared = _candidate
                break
        _null_statistic_holder: dict[str, object] = {}

        def _null_statistic(rng: np.random.Generator) -> Mapping[str, object]:
            shuffled = np.column_stack([rng.permutation(matrix[:, j]) for j in range(matrix.shape[1])])
            evaluated = _evaluate_redundancy(shuffled, config)
            # Store the already-evaluated result: the single central transform
            # plus its full evaluation is consumed directly below (one
            # DictionaryLearning fit, one covariance schedule per bootstrap
            # draw, no replay, no second transform).
            _null_statistic_holder["evaluated"] = evaluated
            return {
                "feature-max-abs-correlation": float(evaluated.max_abs_correlation),
                "sparse-feature-sharing": float(evaluated.sparse_sharing),
            }

        if _null_declared is None:
            # No declared randomized/null control covers the redundancy
            # metrics: the null transform does not execute and no invented
            # identity enters the payload controls table. Gating below treats
            # the absent null as recorded-without-values, never as support.
            null_redundancy = None
            _null_outcome = None
        else:
            _null_declared_id = _null_declared
            _null_via_plan = _execute_plan(
                _ControlPlan(
                    plan_id=f"{config.manifest_id}:redundancy-null",
                    repetitions=config.repetitions,
                    confidence_level=config.confidence_level,
                    evaluation_seed=config.evaluation_seed,
                    control_seed=config.control_seed,
                    training_seed=config.training_seed,
                    controls=(_ControlSpec(
                        control_id=_null_declared_id, kind="randomized", required=False,
                        metric_ids=("feature-max-abs-correlation", "sparse-feature-sharing"),
                        expected_behavior="independent-per-column permutation destroys joint structure",
                    ),),
                ),
                {},
                statistics={_null_declared_id: _null_statistic},  # type: ignore[dict-item]
            )
            _null_outcome = _null_via_plan[_null_declared_id]
            if _null_outcome.status != "recorded":
                raise DetectionError("central null control must record, not gate")
            evaluated_null = _null_statistic_holder.get("evaluated")
            if not isinstance(evaluated_null, _RedundancyEvaluation):
                raise DetectionError("central null control did not evaluate")
            null_redundancy = evaluated_null
        # Central status/gating: every declared control is evaluated by
        # ``execute_plan``. Supplied counterexamples gate per linked metric
        # (counterexample values must NOT meet defect thresholds, one central
        # part per metric); capacity gates on the declared capacity outcome;
        # randomized gates on the executed null evaluation; null/split-swap
        # controls record without gating. Parts fold back to declared
        # identities (all parts must pass/record).
        specs: list[_ControlSpec] = []
        supplied_parts: dict[str, Mapping[str, float] | None] = {}
        derived_ids: dict[str, str] = {}
        _thresholds = {item.metric_id: item for item in config.thresholds}
        for control_id in (*config.required_controls, *config.optional_controls):
            linked = tuple(config.control_metrics[control_id])
            required = control_id in config.required_controls
            kind = config.control_kinds[control_id]
            if kind in ("null",) or "swap" in control_id or "shuffle" in control_id.lower():
                specs.append(
                    _ControlSpec(
                        control_id=control_id, kind="null", required=required,
                        metric_ids=linked,
                        expected_behavior="split/sequence direction recorded without gating",
                    )
                )
                supplied_parts[control_id] = {}
                continue
            if kind == "capacity":
                if separability is None:
                    supplied_parts[control_id] = None
                    specs.append(
                        _ControlSpec(
                            control_id=control_id, kind="capacity", required=required,
                            metric_ids=linked,
                            expected_behavior="declared probe capacity bound",
                            comparator="meets_threshold", threshold_value=1.0,
                        )
                    )
                    continue
                gate_value = 1.0 if separability.capacity_passed else 0.0
                supplied_parts[control_id] = {"capacity": gate_value}
                specs.append(
                    _ControlSpec(
                        control_id=control_id, kind="capacity", required=required,
                        metric_ids=("capacity",),
                        expected_behavior="declared probe capacity bound",
                        comparator="meets_threshold", threshold_value=1.0,
                    )
                )
                continue
            if kind == "randomized":
                if "heldout-probe-accuracy" in linked or "probe-leakage-gap" in linked:
                    assert separability is not None
                    gate_metric = "probe-leakage-gap" if "probe-leakage-gap" in linked else "heldout-probe-accuracy"
                    gate_threshold = _thresholds[gate_metric]
                    gate = "meets_threshold" if gate_threshold.comparator == ">=" else "below_threshold"
                    part_id = f"{control_id}::{gate_metric}"
                    derived_ids[part_id] = control_id
                    observed_gap = {
                        "probe-leakage-gap": float(separability.leakage_gap),
                        "heldout-probe-accuracy": float(separability.heldout_accuracy),
                    }
                    specs.append(
                        _ControlSpec(
                            control_id=part_id, kind="randomized", required=required,
                            metric_ids=(gate_metric,),
                            expected_behavior="label-randomization gap under predeclared threshold",
                            comparator=gate,  # type: ignore[arg-type]
                            threshold_value=float(gate_threshold.value),
                        )
                    )
                    supplied_parts[part_id] = {gate_metric: observed_gap[gate_metric]}
                    continue
                if null_redundancy is None:
                    raise DetectionError(
                        f"randomized control {control_id!r} requires a declared null evaluation, "
                        "but no randomized/null control covers the redundancy metrics"
                    )
                for metric_id in linked:
                    gate_threshold = _thresholds[metric_id]
                    gate = "below_threshold" if gate_threshold.comparator == ">=" else "meets_threshold"
                    part_id = f"{control_id}::{metric_id}"
                    derived_ids[part_id] = control_id
                    observed_null = {
                        "feature-max-abs-correlation": float(null_redundancy.max_abs_correlation),
                        "sparse-feature-sharing": float(null_redundancy.sparse_sharing),
                    }
                    specs.append(
                        _ControlSpec(
                            control_id=part_id, kind="randomized", required=required,
                            metric_ids=(metric_id,),
                            expected_behavior="shuffled null must not meet defect thresholds",
                            comparator=gate,  # type: ignore[arg-type]
                            threshold_value=float(gate_threshold.value),
                        )
                    )
                    supplied_parts[part_id] = {metric_id: observed_null[metric_id]}
                continue
            if kind in _SUPPLIED_KINDS:
                control_value = supplied[control_id]
                control_matrix = _batch_matrix(control_value.value if isinstance(control_value, LabeledBatch) else control_value)
                if control_matrix.shape[1] != matrix.shape[1]:
                    raise DetectionError(
                        f"control {control_id!r} has {control_matrix.shape[1]} features but the target has {matrix.shape[1]}"
                    )
                evaluated = _evaluate_redundancy(control_matrix, config)
                if counterexample_redundancy is None:
                    counterexample_redundancy = evaluated
                observed_counter = {
                    "feature-max-abs-correlation": float(evaluated.max_abs_correlation),
                    "sparse-feature-sharing": float(evaluated.sparse_sharing),
                    "heldout-probe-accuracy": float(separability.negative_accuracy) if separability is not None else 0.0,
                }
                for metric_id in linked:
                    if metric_id == "heldout-probe-accuracy":
                        gate_threshold = _thresholds[metric_id]
                        gate = "below_threshold" if gate_threshold.comparator == ">=" else "meets_threshold"
                    elif metric_id == "probe-leakage-gap" and separability is not None:
                        gap_value = float(separability.heldout_accuracy - separability.negative_accuracy)
                        gate_threshold = _thresholds[metric_id]
                        gate = "meets_threshold" if gate_threshold.comparator == ">=" else "below_threshold"
                        observed_counter[metric_id] = gap_value
                    else:
                        gate_threshold = _thresholds[metric_id]
                        gate = "below_threshold" if gate_threshold.comparator == ">=" else "meets_threshold"
                    part_id = f"{control_id}::{metric_id}"
                    derived_ids[part_id] = control_id
                    specs.append(
                        _ControlSpec(
                            control_id=part_id, kind="counterexample", required=required,
                            metric_ids=(metric_id,),
                            expected_behavior="counterexample must not exhibit the claimed pattern",
                            comparator=gate,  # type: ignore[arg-type]
                            threshold_value=float(gate_threshold.value),
                        )
                    )
                    supplied_parts[part_id] = {metric_id: float(observed_counter[metric_id])}
                continue
            specs.append(
                _ControlSpec(
                    control_id=control_id, kind=config.control_kinds[control_id]
                    if config.control_kinds[control_id] in ("bootstrap", "null", "shuffled", "randomized", "cross_seed", "counterexample", "negative", "capacity", "seed")
                    else "null",
                    required=required, metric_ids=linked,
                    expected_behavior="recorded control; not threshold-gated",
                )
            )
            supplied_parts[control_id] = {}
        _central_gating = _execute_plan(
            _ControlPlan(
                plan_id=f"{config.manifest_id}:redundancy-separability",
                repetitions=config.repetitions,
                confidence_level=config.confidence_level,
                evaluation_seed=config.evaluation_seed,
                control_seed=config.control_seed,
                training_seed=config.training_seed,
                controls=tuple(specs),
            ),
            supplied_parts,
        )
        # Merge the executed null outcome only when a declared control bound
        # it: with no declared randomized/null control the null never ran and
        # no invented identity may enter the payload controls table.
        if _null_declared is not None and _null_outcome is not None:
            _central_gating[_null_declared] = _null_outcome
        for control_id in (*config.required_controls, *config.optional_controls):
            parts = [part for part, parent in derived_ids.items() if parent == control_id]
            if not parts:
                if _null_declared is not None and control_id == _null_declared and _null_outcome is not None:
                    control_outcomes[control_id] = "passed" if _null_outcome.status in ("passed", "recorded") else "failed"
                    continue
                outcome = _central_gating[control_id]
                control_outcomes[control_id] = "passed" if outcome.status in ("passed", "recorded") else "failed"
                continue
            statuses = [_central_gating[part].status for part in parts]
            control_outcomes[control_id] = "passed" if all(status in ("passed", "recorded") for status in statuses) else "failed"
        _central_folded: dict[str, object] = {control_id: outcome.to_dict() for control_id, outcome in _central_gating.items()}
        _blocked = sorted({derived_ids.get(part, part) for part in _failed_required(_central_gating)} & set(config.required_controls))
        if _blocked:
            control_outcomes.update({control_id: "failed" for control_id in _blocked})
    _all_central.update(_sep_central)
    _all_central.update(_central_folded)
    bound_identity = labeled.value.identity if labeled is not None else cast(LatentValue, target).identity
    return _DetectionContext(
        representation_identity=bound_identity,
        separability=separability,
        redundancy=redundancy,
        null_redundancy=null_redundancy,
        counterexample_redundancy=counterexample_redundancy,
        control_outcomes=control_outcomes,
        central_outcomes=_all_central,
    )


def _decide(context: _DetectionContext, config: DetectionConfig) -> tuple[FamilyDetection, ...]:
    """Decide every configured family from one shared evaluation context."""
    thresholds = {item.metric_id: item for item in config.thresholds}
    detections: list[FamilyDetection] = []
    for family_id in config.family_ids:
        family_metrics = [metric_id for metric_id in config.metric_ids if METRIC_FAMILY.get(metric_id) == family_id]
        if family_id == "separability_probe_leakage":
            evaluation = context.separability
            assert evaluation is not None
            observed = {
                "heldout-probe-accuracy": evaluation.heldout_accuracy,
                "probe-leakage-gap": evaluation.leakage_gap,
            }
            observed = {metric_id: observed[metric_id] for metric_id in family_metrics if metric_id in observed}
            verdicts = {metric_id: thresholds[metric_id].passes(observed[metric_id]) for metric_id in observed}
            evidence = {
                "probe_split_provenance": "observed",
                "probe_heldout_metric": "observed",
                "probe_capacity_control": "observed",
                "probe_label_randomization_control": "observed",
                "probe_nonseparable_negative_control": "observed",
            }
        elif family_id == "redundancy_superposition" and context.redundancy is None:
            # Honest unsupported path: the manifest predeclares no promotable
            # redundancy metrics, so no other family's thresholds may substitute.
            detections.append(
                FamilyDetection(
                    family_id=family_id,
                    outcome="unsupported",
                    claim_allowed=False,
                    observed_metrics={},
                    threshold_pass={},
                    control_outcomes=dict(context.control_outcomes),
                    evidence_status={
                        "redundancy_dependency_measure": "not_evaluated",
                        "superposition_sparse_or_overcomplete_comparison": "not_evaluated",
                        "redundancy_randomized_feature_control": "not_evaluated",
                        "redundancy_healthy_counterexample": "not_evaluated",
                        "redundancy_provenance": "not_evaluated",
                    },
                    missing_evidence=("missing-declaration:redundancy_superposition",),
                    reason="manifest predeclares no redundancy/superposition metrics; refusing to borrow another family's label",
                )
            )
            continue
        else:
            evaluation = cast(_RedundancyEvaluation, context.redundancy)
            observed = {
                "feature-max-abs-correlation": evaluation.max_abs_correlation,
                "sparse-feature-sharing": evaluation.sparse_sharing,
            }
            observed = {metric_id: observed[metric_id] for metric_id in family_metrics if metric_id in observed}
            verdicts = {metric_id: thresholds[metric_id].passes(observed[metric_id]) for metric_id in observed}
            evidence = {
                "redundancy_dependency_measure": "observed",
                "superposition_sparse_or_overcomplete_comparison": "observed",
                "redundancy_randomized_feature_control": "observed",
                "redundancy_healthy_counterexample": "observed",
                "redundancy_provenance": "observed",
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
        if unmet and family_id == "separability_probe_leakage":
            # Held-out accuracy alone never promotes: report the unmet
            # predeclared threshold even when a control also fails.
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
                    reason=f"predeclared thresholds unmet; probe accuracy alone is not a diagnosis: {', '.join(unmet)}",
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
        unmet = sorted(metric_id for metric_id in family_metrics if metric_id in verdicts and not verdicts[metric_id])
        if family_id == "redundancy_superposition" and unmet:
            correlation_flags = bool(
                "feature-max-abs-correlation" in unmet
                and "feature-max-abs-correlation" in family_metrics
                and "sparse-feature-sharing" in family_metrics
                and "sparse-feature-sharing" not in unmet
            )
            sharing_flags = bool(
                "sparse-feature-sharing" in unmet
                and "sparse-feature-sharing" in family_metrics
                and "feature-max-abs-correlation" in family_metrics
                and "feature-max-abs-correlation" not in unmet
            )
            if correlation_flags or sharing_flags:
                # Correlation alone (or sharing alone) can never promote a
                # redundancy/superposition claim; semantic superposition is not
                # inferred from one lens.
                detections.append(
                    FamilyDetection(
                        family_id=family_id,
                        outcome="inconclusive",
                        claim_allowed=False,
                        observed_metrics=dict(observed),
                        threshold_pass=dict(verdicts),
                        control_outcomes=dict(context.control_outcomes),
                        evidence_status=dict(evidence),
                        missing_evidence=("inconsistent-evidence:redundancy_superposition",),
                        reason="correlation without sparse-comparison support cannot promote a redundancy/superposition claim",
                    )
                )
                continue
            # A healthy negative (both lenses below a defect threshold) is a
            # negative: observable and promotable, not an inconsistency.
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
                    reason="metrics flag a defect under predeclared thresholds with required controls passed"
                    if not _metrics_met(verdicts, family_metrics)
                    else "predeclared thresholds met with required controls passed",
                )
            )
            continue
        metrics_agree = _metrics_met(verdicts, family_metrics)
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
                reason="predeclared thresholds met with required controls passed"
                if metrics_agree
                else "metrics flag a defect under predeclared thresholds with required controls passed",
            )
        )
    return tuple(detections)


def detect_families(
    target: LabeledBatch | LatentValue,
    config: DetectionConfig,
    *,
    controls: Mapping[str, LabeledBatch | LatentValue] | None = None,
) -> tuple[FamilyDetection, ...]:
    """Evaluate the configured families on one bound input plus named controls.

    Focused unit-check seam: evaluates its own context, so composing it with a
    separate :func:`detection_payload` call evaluates twice. Production paths
    (including :func:`make_detect_executor`) must use :func:`evaluate_detection`.
    """
    if not isinstance(config, DetectionConfig):
        raise DetectionError("config must be a DetectionConfig")
    if not isinstance(target, (LabeledBatch, LatentValue)):
        raise DetectionError("target must be a LabeledBatch or LatentValue")
    supplied = dict(controls or {})
    return _decide(_evaluate_context(target, config, supplied), config)


def detection_payload(
    detections: Sequence[FamilyDetection],
    config: DetectionConfig,
    target: LabeledBatch | LatentValue,
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
    evaluation = resolved.separability
    if evaluation is not None:
        measurements["separability"] = {
            "capacity": {
                "allowed": DECLARED_CAPACITY,
                "declared": evaluation.capacity_declared,
                "n_params": int(evaluation.n_params),
                "n_train": int(evaluation.n_train),
                "passed": bool(evaluation.capacity_passed),
            },
            "heldout_accuracy": float(evaluation.heldout_accuracy),
            "leakage_gap": float(evaluation.leakage_gap),
            "n_eval": int(evaluation.n_eval),
            "n_train": int(evaluation.n_train),
            "negative_accuracy": float(evaluation.negative_accuracy),
            "randomized_accuracy": float(evaluation.randomized_accuracy),
            "swap_accuracy": float(evaluation.swap_accuracy),
            "uncertainty": {key: dict(item) for key, item in evaluation.uncertainty.items()},
        }
    if resolved.redundancy is not None:
        redundancy_block: dict[str, object] = {
            "max_abs_correlation": float(resolved.redundancy.max_abs_correlation),
            "sparse_sharing": float(resolved.redundancy.sparse_sharing),
            "uncertainty": {key: dict(item) for key, item in resolved.redundancy.uncertainty.items()},
        }
        if resolved.null_redundancy is not None:
            redundancy_block["null_randomized_feature"] = {
                "max_abs_correlation": float(resolved.null_redundancy.max_abs_correlation),
                "sparse_sharing": float(resolved.null_redundancy.sparse_sharing),
            }
        if resolved.counterexample_redundancy is not None:
            redundancy_block["counterexample"] = {
                "max_abs_correlation": float(resolved.counterexample_redundancy.max_abs_correlation),
                "sparse_sharing": float(resolved.counterexample_redundancy.sparse_sharing),
            }
        measurements["redundancy"] = redundancy_block
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
    evaluation = context.separability
    for control_id in (*config.required_controls, *config.optional_controls):
        linked = config.control_metrics[control_id]
        entry: dict[str, float] = {}
        if evaluation is not None:
            if "heldout-probe-accuracy" in linked:
                kind = config.control_kinds[control_id]
                if kind == "capacity":
                    entry["heldout-probe-accuracy"] = evaluation.heldout_accuracy
                elif kind == "randomized":
                    entry["heldout-probe-accuracy"] = evaluation.randomized_accuracy
                elif kind in _SUPPLIED_KINDS:
                    entry["heldout-probe-accuracy"] = evaluation.negative_accuracy
                else:
                    entry["heldout-probe-accuracy"] = evaluation.swap_accuracy
            if "probe-leakage-gap" in linked:
                kind = config.control_kinds[control_id]
                if kind == "randomized":
                    entry["probe-leakage-gap"] = evaluation.leakage_gap
                elif kind in _SUPPLIED_KINDS:
                    entry["probe-leakage-gap"] = float(evaluation.heldout_accuracy - evaluation.negative_accuracy)
        if context.redundancy is not None:
            source = context.redundancy
            if config.control_kinds[control_id] == "randomized" and context.null_redundancy is not None:
                source = context.null_redundancy
            if config.control_kinds[control_id] in _SUPPLIED_KINDS and context.counterexample_redundancy is not None:
                source = context.counterexample_redundancy
            if "feature-max-abs-correlation" in linked:
                entry["feature-max-abs-correlation"] = source.max_abs_correlation
            if "sparse-feature-sharing" in linked:
                entry["sparse-feature-sharing"] = source.sparse_sharing
        if entry:
            table[control_id] = entry
    return table


def evaluate_detection(
    target: LabeledBatch | LatentValue,
    config: DetectionConfig,
    controls: Mapping[str, LabeledBatch | LatentValue],
) -> tuple[tuple[FamilyDetection, ...], dict[str, object]]:
    """Evaluate one input plus controls exactly once and return decisions plus payload.

    This is the single-evaluation seam every caller must use: every probe fit
    and every dictionary decomposition is computed once inside one
    :class:`_DetectionContext`, then shared by family decisions and payload
    assembly. Separability bootstrap resamples fitted probe predictions without
    refitting; correlation uncertainty refits ``fit_covariance`` once per draw
    (the only bootstrap refit); sharing uncertainty resamples fitted coherence
    values without refitting. ``detect_families``
    and :func:`detection_payload` remain available for focused unit checks but
    must not be composed naively in production paths, since that composition
    would evaluate everything twice.
    """
    if not isinstance(config, DetectionConfig):
        raise DetectionError("config must be a DetectionConfig")
    if not isinstance(target, (LabeledBatch, LatentValue)):
        raise DetectionError("target must be a LabeledBatch or LatentValue")
    supplied = dict(controls)
    context = _evaluate_context(target, config, supplied)
    detections = _decide(context, config)
    payload = detection_payload(detections, config, target, control_metrics=_control_table(context, config), context=context)
    return detections, payload


def make_detect_executor(
    target: LabeledBatch | LatentValue,
    config: DetectionConfig,
    *,
    controls: Mapping[str, LabeledBatch | LatentValue] | None = None,
    version: str = "redundancy-separability-detector-v1",
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
    if not isinstance(target, (LabeledBatch, LatentValue)):
        raise DetectionError("target must be a LabeledBatch or LatentValue")
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
    "DECLARED_CAPACITY",
    "METRIC_FAMILY",
    "SUPPORTED_FAMILIES",
    "ClaimOutcome",
    "DetectionConfig",
    "DetectionError",
    "FamilyDetection",
    "FamilyThreshold",
    "LabeledBatch",
    "detect_families",
    "detection_config_from_manifest",
    "detection_payload",
    "evaluate_detection",
    "make_detect_executor",
]
