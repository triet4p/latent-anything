"""Collapse, rank-loss, anisotropy, and inactive-dimension detection (Sprint 80.9).

Smallest detector-family adapter behind the frozen taxonomy/workflow
contracts for ``collapse_rank_loss`` and ``anisotropy_inactive_dimensions``.
It consumes bound :class:`LatentValue` batches plus predeclared
metric/control/threshold configuration and emits machine-readable
observations, taxonomy family-evidence statuses, control outcomes, and
uncertainty/repetition metadata. A supplied ``detect``-stage executor wires it
into the method-agnostic :class:`DiagnosticWorkflow` without inserting
algorithms into the coordinator.

Numerical reuse (no duplicate primitives):

- Sample covariance and the sorted eigenvalue spectrum reuse
  :func:`compute_latent_health` from ``_jepa_evaluation`` (unbiased
  ``np.cov``, participation-ratio effective rank, collapsed fraction,
  covariance condition).
- The singular-value spread (``min_sv / max_sv`` of the centered batch)
  extends that same spectrum with one thin SVD the health primitive does not
  expose; no second covariance fit or extra data copy is introduced.
- Deterministic seeded bootstrap over sample rows supplies the predeclared
  repetition/interval metadata through the central ``_statistical_controls``
  executor (identity-derived ``evaluation`` streams, exact repetition count);
  the column-shuffle null runs as one central ``shuffled`` control.

Collapse-vs-benign-scale separation: effective rank and singular spread are
scale-invariant (both normalize out global gain), while per-dimension
variance is scale-sensitive. A globally scaled but structurally healthy batch
therefore keeps healthy rank/spread values and is reported as a negative
(no defect), whereas a rank-deficient or directionally dead batch fails the
predeclared thresholds. Threshold direction and identity come from the frozen
manifest/config (``>=`` on both encoder metrics); nothing is tuned from
observed values.

Fail-closed: non-finite input, undersampled batches (``n_samples <= dim``),
non-2D or incompatible-rank/axis data, missing manifest thresholds or
manifest/request controls, unknown comparators, and ambiguous metric wiring
reject via :class:`DetectionError`. A failed required control blocks a
supported diagnostic conclusion (``claim_allowed`` is ``False``).

Non-goals: 80.10+ families, localization, explanations/interventions, and real
core benchmark proof. Repetition/interval/null scheduling is centralized in
``_statistical_controls`` (80.15); this adapter supplies only fitted-batch
draw callbacks and predeclared seeds.
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
from latent_anything._jepa_evaluation import compute_latent_health
from latent_anything._portable_contract import PortableNodeError, canonical_json
from latent_anything._representation_taxonomy import evaluate_claim
from latent_anything._statistical_controls import ControlPlan as _ControlPlan
from latent_anything._statistical_controls import ControlSpec as _ControlSpec
from latent_anything._statistical_controls import execute_plan as _execute_plan
from latent_anything._statistical_controls import failed_required as _failed_required
from latent_anything.diagnostics import DiagnosticRequest
from latent_anything.latent_value import LatentValue

SUPPORTED_FAMILIES: tuple[str, ...] = ("anisotropy_inactive_dimensions", "collapse_rank_loss")
"""Detector scope for this task. Exact taxonomy identifiers; nothing else is evaluated."""

_FEATURE_VARIANCE_RATIO_METRIC = "bottleneck-feature-variance-ratio"

METRIC_FAMILY: Mapping[str, str] = MappingProxyType(
    {
        "bottleneck-effective-rank": "collapse_rank_loss",
        "bottleneck-singular-spread": "collapse_rank_loss",
        _FEATURE_VARIANCE_RATIO_METRIC: "collapse_rank_loss",
    }
)
"""Predeclared collapse metric-to-family wiring. Unknown metrics reject."""

ANISOTROPY_METRICS: tuple[str, ...] = ("bottleneck-effective-rank", "bottleneck-singular-spread")
"""Anisotropy reads the same frozen spectrum through a directional-activity lens."""

ClaimOutcome = Literal["supported", "inconclusive", "unsupported"]


class DetectionError(ValueError):
    """Raised when detection input, config, or evidence is fail-closed invalid."""


def _require_request(value: object) -> DiagnosticRequest:
    if not isinstance(value, DiagnosticRequest):
        raise DetectionError("request must be a DiagnosticRequest")
    return value


def _require_config(value: object) -> DetectionConfig:
    if not isinstance(value, DetectionConfig):
        raise DetectionError("config must be a DetectionConfig")
    return value


def _require_batch(value: object) -> LatentValue:
    if not isinstance(value, LatentValue):
        raise DetectionError("value must be a LatentValue")
    return value


def _require_invocation(value: object) -> StageInvocation:
    from latent_anything._diagnostic_workflow import StageInvocation as _Invocation

    if not isinstance(value, _Invocation):
        from latent_anything._diagnostic_workflow import StageContractError as _ContractError

        raise _ContractError("detect executor requires a StageInvocation")
    return value


def _require_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise DetectionError("claim_allowed must be boolean")
    return value


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
    repetitions: int
    confidence_level: float
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
            if METRIC_FAMILY[metric_id] not in self.family_ids:
                raise DetectionError(f"metric {metric_id!r} belongs to an unevaluated family")
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
        if self.repetitions < 2:
            raise DetectionError("repetitions must be at least two")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise DetectionError("confidence_level must be between zero and one")
        for name in ("evaluation_seed", "control_seed"):
            candidate = getattr(self, name)
            if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 0:
                raise DetectionError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this configuration."""
        return {
            "confidence_level": float(self.confidence_level),
            "control_metrics": {key: list(value) for key, value in self.control_metrics.items()},
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


def detection_config_from_manifest(request: object, manifest: Mapping[str, object]) -> DetectionConfig:
    """Parse the predeclared detector configuration; fail closed on any gap."""
    request = _require_request(request)
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
    manifest_thresholds: list[FamilyThreshold] = []
    for index, raw in enumerate(raw_thresholds):
        item = _mapping(raw, name=f"thresholds[{index}]")
        manifest_thresholds.append(
            FamilyThreshold(
                metric_id=_non_empty_string(item.get("metric_id"), name=f"thresholds[{index}].metric_id"),
                comparator=_non_empty_string(item.get("comparator"), name=f"thresholds[{index}].comparator"),
                value=_finite_number(item.get("value"), name=f"thresholds[{index}].value"),
                tolerance=_finite_number(item.get("tolerance"), name=f"thresholds[{index}].tolerance"),
            )
        )
    if len({item.metric_id for item in manifest_thresholds}) != len(manifest_thresholds):
        raise DetectionError("every manifest metric must have exactly one predeclared threshold")
    threshold_by_metric = {item.metric_id: item for item in manifest_thresholds}
    requested_metric_ids = tuple(request.controls.metric_ids)
    missing_thresholds = [metric_id for metric_id in requested_metric_ids if metric_id not in threshold_by_metric]
    if missing_thresholds:
        raise DetectionError(f"missing predeclared thresholds for requested metrics: {', '.join(missing_thresholds)}")
    thresholds = [threshold_by_metric[metric_id] for metric_id in requested_metric_ids]

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
    evaluation = seeds.get("evaluation")
    controls_seed = seeds.get("controls")
    if isinstance(evaluation, (str, bytes)) or not isinstance(evaluation, Sequence) or not evaluation:
        raise DetectionError("seeds.evaluation must be a non-empty list")
    if isinstance(controls_seed, (str, bytes)) or not isinstance(controls_seed, Sequence) or not controls_seed:
        raise DetectionError("seeds.controls must be a non-empty list")
    return DetectionConfig(
        manifest_id=manifest_id,
        family_ids=family_ids,
        metric_ids=tuple(request.controls.metric_ids),
        thresholds=tuple(thresholds),
        required_controls=tuple(
            control_id for control_id in manifest_controls if manifest_controls[control_id]["required"] is True
        ),
        optional_controls=tuple(
            control_id for control_id in manifest_controls if manifest_controls[control_id]["required"] is not True
        ),
        control_metrics={
            control_id: tuple(str(item) for item in cast(Sequence[object], manifest_controls[control_id]["metric_ids"]))
            for control_id in manifest_controls
        },
        repetitions=int(repetitions),
        confidence_level=_finite_number(uncertainty.get("confidence_level"), name="uncertainty.confidence_level"),
        evaluation_seed=_non_negative_int(evaluation[0], name="seeds.evaluation[0]"),
        control_seed=_non_negative_int(controls_seed[0], name="seeds.controls[0]"),
    )


@dataclass(frozen=True)
class FamilyDetection:
    """Evidence-bound outcome for one taxonomy family on one batch."""

    family_id: str
    outcome: ClaimOutcome
    claim_allowed: object
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
        object.__setattr__(self, "claim_allowed", _require_bool(self.claim_allowed))
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


def _batch_matrix(value: object) -> np.ndarray:
    value = _require_batch(value)
    data = np.asarray(value.to_numpy(), dtype=np.float64)
    if data.ndim != 2:
        raise DetectionError(f"incompatible rank: detection requires one 2D (n_samples, dim) batch, got {data.ndim}D")
    if data.shape[0] <= data.shape[1]:
        raise DetectionError(
            f"undersampled batch: {data.shape[0]} samples cannot support rank estimation over {data.shape[1]} dims"
        )
    if not np.isfinite(data).all():
        raise DetectionError("batch contains non-finite values")
    return data


def _spectrum(centered: np.ndarray) -> tuple[float, ...]:
    """Return the singular values of one centered batch with one thin SVD."""
    return tuple(float(item) for item in np.linalg.svd(centered, compute_uv=False))


def _spread_of(singular: Sequence[float]) -> float:
    peak = float(singular[0])
    if peak <= 0.0 or not np.isfinite(peak):
        raise DetectionError("singular spectrum is degenerate")
    ratio = float(singular[-1] / peak)
    if not np.isfinite(ratio):
        raise DetectionError("singular spread is non-finite")
    return ratio


def _singular_spread(centered: np.ndarray) -> float:
    # Resample/null path: one thin SVD per draw. Batch paths use _spectrum
    # once and reuse the values for both spread and the reported spectrum,
    # so no batch is ever decomposed twice.
    return _spread_of(_spectrum(centered))


def _feature_variance_ratios(matrix: np.ndarray) -> tuple[float, ...]:
    """Return sample variances divided by their median for aligned features."""
    variances = np.var(matrix, axis=0, ddof=1)
    median = float(np.median(variances))
    if not np.isfinite(median) or median <= 0.0:
        raise DetectionError("feature variance ratio is undefined when median feature variance is zero")
    ratios = variances / median
    if not np.isfinite(ratios).all():
        raise DetectionError("feature variance ratio produced a non-finite value")
    return tuple(float(item) for item in ratios)


@dataclass(frozen=True)
class _EvaluatedBatch:
    """One batch decomposed once into spectrum and optional feature-variance evidence."""

    n_samples: int
    dim: int
    effective_rank: float
    singular_spread: float
    min_variance: float
    feature_variance_ratio: float | None
    feature_variance_ratios: tuple[float, ...]
    inactive_fraction: float
    covariance_condition: float
    singular: tuple[float, ...]


def _evaluate_batch(matrix: np.ndarray, *, include_feature_variance: bool = False) -> _EvaluatedBatch:
    """Decompose one validated batch once and derive configured metrics from it."""
    health = compute_latent_health(matrix)
    centered = matrix - np.mean(matrix, axis=0)
    singular = _spectrum(centered)
    spread = _spread_of(singular)
    variances = np.var(matrix, axis=0)
    feature_ratios = _feature_variance_ratios(matrix) if include_feature_variance else ()
    feature_ratio = min(feature_ratios) if feature_ratios else None
    observed = {
        "bottleneck-effective-rank": float(health.effective_rank),
        "bottleneck-singular-spread": spread,
        "bottleneck-min-variance": float(np.min(variances)),
        "bottleneck-inactive-fraction": float(health.collapsed_fraction),
    }
    if feature_ratio is not None:
        observed[_FEATURE_VARIANCE_RATIO_METRIC] = feature_ratio
    for metric_id, number in observed.items():
        if not np.isfinite(number):
            raise DetectionError(f"metric {metric_id!r} produced a non-finite value")
    return _EvaluatedBatch(
        n_samples=int(matrix.shape[0]),
        dim=int(matrix.shape[1]),
        effective_rank=observed["bottleneck-effective-rank"],
        singular_spread=observed["bottleneck-singular-spread"],
        min_variance=observed["bottleneck-min-variance"],
        feature_variance_ratio=feature_ratio,
        feature_variance_ratios=feature_ratios,
        inactive_fraction=observed["bottleneck-inactive-fraction"],
        covariance_condition=float(health.covariance_condition),
        singular=singular,
    )


def _bootstrap_both(
    data: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence_level: float,
    include_feature_variance: bool = False,
) -> dict[str, dict[str, object]]:
    """Resample once per draw and record rank, spread, and configured feature variance ratio."""
    from latent_anything._statistical_controls import derive_stream_seed as _derive
    from latent_anything._statistical_controls import summarize_interval as _interval

    n = data.shape[0]
    stream = _derive(seed, "bootstrap:collapse-both", role="evaluation")
    rng = np.random.default_rng(stream)
    rank_samples: list[float] = []
    spread_samples: list[float] = []
    feature_ratio_samples: list[float] | None = [] if include_feature_variance else None
    for _ in range(repetitions):
        rows = rng.integers(0, n, size=n)
        block = data[rows]
        rank_samples.append(float(compute_latent_health(block).effective_rank))
        spread_samples.append(float(_singular_spread(block - np.mean(block, axis=0))))
        if feature_ratio_samples is not None:
            feature_ratio_samples.append(min(_feature_variance_ratios(block)))
    results = {
        "bottleneck-effective-rank": dict(
            _interval(rank_samples, repetitions=repetitions, seed=stream, confidence_level=confidence_level)
        ),
        "bottleneck-singular-spread": dict(
            _interval(spread_samples, repetitions=repetitions, seed=stream, confidence_level=confidence_level)
        ),
    }
    if feature_ratio_samples is not None:
        results[_FEATURE_VARIANCE_RATIO_METRIC] = dict(
            _interval(
                feature_ratio_samples,
                repetitions=repetitions,
                seed=stream,
                confidence_level=confidence_level,
            )
        )
    return results


@dataclass(frozen=True)
class _DetectionContext:
    """One executor call evaluated exactly once and shared by decisions and payload."""

    representation_identity: str
    evaluated: _EvaluatedBatch
    singular: tuple[float, ...]
    control_evaluated: Mapping[str, _EvaluatedBatch]
    control_outcomes: Mapping[str, str]
    central_outcomes: Mapping[str, object]
    null_metrics: Mapping[str, float]
    uncertainty: Mapping[str, dict[str, object]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_evaluated", dict(self.control_evaluated))
        object.__setattr__(self, "control_outcomes", dict(self.control_outcomes))
        object.__setattr__(self, "central_outcomes", dict(self.central_outcomes))
        object.__setattr__(self, "null_metrics", dict(self.null_metrics))
        object.__setattr__(self, "uncertainty", {key: dict(item) for key, item in self.uncertainty.items()})


def _evaluate_context(
    value: LatentValue,
    config: DetectionConfig,
    supplied: Mapping[str, LatentValue],
) -> _DetectionContext:
    """Validate and decompose the target plus every control batch exactly once."""
    data = _batch_matrix(value)
    matrices: dict[str, np.ndarray] = {}
    for control_id, control_value in supplied.items():
        if control_id not in (*config.required_controls, *config.optional_controls):
            raise DetectionError(f"unknown control reference: {control_id!r}")
        matrices[control_id] = _batch_matrix(control_value)
    missing = [control_id for control_id in config.required_controls if control_id not in supplied]
    if missing:
        raise DetectionError(f"required controls are missing batch data: {', '.join(missing)}")

    include_feature_variance = _FEATURE_VARIANCE_RATIO_METRIC in config.metric_ids
    evaluated = _evaluate_batch(data, include_feature_variance=include_feature_variance)
    control_evaluated = {
        control_id: _evaluate_batch(matrix, include_feature_variance=include_feature_variance)
        for control_id, matrix in matrices.items()
    }
    thresholds = {item.metric_id: item for item in config.thresholds}
    control_metrics: dict[str, dict[str, float]] = {}
    for control_id, item in control_evaluated.items():
        metrics = {
            "bottleneck-effective-rank": item.effective_rank,
            "bottleneck-singular-spread": item.singular_spread,
            "bottleneck-min-variance": item.min_variance,
            "bottleneck-inactive-fraction": item.inactive_fraction,
        }
        if item.feature_variance_ratio is not None:
            metrics[_FEATURE_VARIANCE_RATIO_METRIC] = item.feature_variance_ratio
        control_metrics[control_id] = metrics

    def _null_statistic(rng: np.random.Generator) -> Mapping[str, float]:
        # Central transform: the single seeded column permutation runs inside
        # the executor-owned stream; the fitted batch is evaluated once from
        # that permuted draw (no second permutation schedule, no refit loop).
        shuffled = np.column_stack([rng.permutation(data[:, j]) for j in range(data.shape[1])])
        shuffled_evaluated = _evaluate_batch(shuffled)
        return {
            "bottleneck-effective-rank": float(shuffled_evaluated.effective_rank),
            "bottleneck-singular-spread": float(shuffled_evaluated.singular_spread),
        }

    # Central status/gating: every declared control is evaluated by
    # ``execute_plan`` under identity-derived streams. Supplied controls gate
    # on their predeclared thresholds (healthy references must meet them);
    # null/shuffle-named controls record without gating. The derived
    # column-shuffle null executes its transform inside the central callback
    # and binds observed values from that exact ControlOutcome.
    specs: list[_ControlSpec] = []
    supplied_metrics: dict[str, Mapping[str, float] | None] = {}
    derived_ids: dict[str, str] = {}
    for control_id in (*config.required_controls, *config.optional_controls):
        full_linked = tuple(config.control_metrics[control_id])
        linked = (
            tuple(
                metric_id
                for metric_id in full_linked
                if metric_id
                in (
                    "bottleneck-effective-rank",
                    "bottleneck-singular-spread",
                    _FEATURE_VARIANCE_RATIO_METRIC,
                )
            )
            or full_linked
        )
        required = control_id in config.required_controls
        if "null" in control_id or "shuffle" in control_id:
            specs.append(
                _ControlSpec(
                    control_id=control_id,
                    kind="shuffled",
                    required=required,
                    metric_ids=linked,
                    expected_behavior="independent-per-column permutation destroys joint structure",
                )
            )
            supplied_metrics[control_id] = {}
            continue
        for metric_id in linked:
            gate = "meets_threshold" if thresholds[metric_id].comparator == ">=" else "below_threshold"
            part_id = f"{control_id}::{metric_id}"
            derived_ids[part_id] = control_id
            specs.append(
                _ControlSpec(
                    control_id=part_id,
                    kind="counterexample",
                    required=required,
                    metric_ids=(metric_id,),
                    expected_behavior=f"supplied reference evaluated against predeclared {metric_id} threshold",
                    comparator=gate,
                    threshold_value=float(thresholds[metric_id].value),
                )
            )
            observed_value = control_metrics.get(control_id, {}).get(metric_id)
            supplied_metrics[part_id] = None if observed_value is None else {metric_id: float(observed_value)}
    _null_spec_id = "central:column-shuffle-null"
    specs.append(
        _ControlSpec(
            control_id=_null_spec_id,
            kind="shuffled",
            required=False,
            metric_ids=("bottleneck-effective-rank", "bottleneck-singular-spread"),
            expected_behavior="independent-per-column permutation destroys joint structure",
        )
    )
    _central_gating = _execute_plan(
        _ControlPlan(
            plan_id=f"{config.manifest_id}:collapse",
            repetitions=config.repetitions,
            confidence_level=config.confidence_level,
            evaluation_seed=config.evaluation_seed,
            control_seed=config.control_seed,
            controls=tuple(specs),
        ),
        supplied_metrics,
        statistics={_null_spec_id: _null_statistic},
    )
    # The derived null transform executes centrally; bind its observed values
    # from that exact ControlOutcome (fixed derived identity, never colliding
    # with the manifest's own null/shuffle-named recorded control).
    _derived_null = _central_gating.get(_null_spec_id)
    if _derived_null is None:
        raise DetectionError("central column-shuffle null did not execute")
    if _derived_null.status == "failed":
        raise DetectionError(f"central null control failed: {_derived_null.reason}")
    _null_observed: object = _derived_null.observed
    if not isinstance(_null_observed, Mapping):
        raise DetectionError("central null control observed must be a mapping")
    _null_values = {key: float(value) for key, value in _null_observed.items()}
    # Fold per-metric parts back to declared control identities: a declared
    # control passes iff every central part passes or records; any failed
    # part fails the declared control. Central parts stay visible in the
    # payload under their derived identities.
    control_outcomes: dict[str, str] = {}
    for control_id in (*config.required_controls, *config.optional_controls):
        parts = [part for part, parent in derived_ids.items() if parent == control_id]
        if not parts:
            outcome = _central_gating[control_id]
            control_outcomes[control_id] = "passed" if outcome.status in ("passed", "recorded") else "failed"
            continue
        statuses = [_central_gating[part].status for part in parts]
        control_outcomes[control_id] = (
            "passed" if all(status in ("passed", "recorded") for status in statuses) else "failed"
        )
    central_outcomes = {control_id: outcome.to_dict() for control_id, outcome in _central_gating.items()}
    _blocked_parts = set(_failed_required(_central_gating))
    _blocked = sorted({derived_ids.get(part, part) for part in _blocked_parts} & set(config.required_controls))
    if _blocked:
        control_outcomes.update({control_id: "failed" for control_id in _blocked})
    return _DetectionContext(
        representation_identity=value.identity,
        evaluated=evaluated,
        singular=evaluated.singular,
        control_evaluated=control_evaluated,
        control_outcomes=control_outcomes,
        central_outcomes=central_outcomes,
        null_metrics=_null_values,
        uncertainty=_bootstrap_both(
            data,
            repetitions=config.repetitions,
            seed=config.evaluation_seed,
            confidence_level=config.confidence_level,
            include_feature_variance=include_feature_variance,
        ),
    )


def _decide(context: _DetectionContext, config: DetectionConfig) -> tuple[FamilyDetection, ...]:
    """Decide every configured family from one shared evaluation context."""
    evaluated = context.evaluated
    observed: dict[str, float] = {
        "bottleneck-effective-rank": evaluated.effective_rank,
        "bottleneck-singular-spread": evaluated.singular_spread,
        "bottleneck-min-variance": evaluated.min_variance,
        "bottleneck-inactive-fraction": evaluated.inactive_fraction,
    }
    if _FEATURE_VARIANCE_RATIO_METRIC in config.metric_ids:
        if evaluated.feature_variance_ratio is None:
            raise DetectionError("configured feature variance ratio was not evaluated")
        observed[_FEATURE_VARIANCE_RATIO_METRIC] = evaluated.feature_variance_ratio
    thresholds = {item.metric_id: item for item in config.thresholds}
    verdicts: dict[str, bool] = {}
    for metric_id in config.metric_ids:
        threshold = thresholds.get(metric_id)
        if threshold is None:
            raise DetectionError(f"missing threshold for metric {metric_id!r}")
        verdicts[metric_id] = threshold.passes(observed[metric_id])
    control_outcomes = dict(context.control_outcomes)
    detections: list[FamilyDetection] = []
    for family_id in config.family_ids:
        if family_id == "collapse_rank_loss":
            family_metrics = [metric_id for metric_id in config.metric_ids if METRIC_FAMILY[metric_id] == family_id]
        else:
            # Anisotropy/inactive-dimension reads the frozen collapse spectrum
            # through a directional-activity lens: same predeclared metrics and
            family_metrics = [metric_id for metric_id in config.metric_ids if metric_id in ANISOTROPY_METRICS]
        failed = sorted(
            control_id
            for control_id in config.required_controls
            if control_id in control_outcomes
            and control_outcomes[control_id] == "failed"
            and bool(set(config.control_metrics[control_id]) & set(family_metrics))
        )
        if family_id == "collapse_rank_loss":
            evidence = {
                "collapse_rank_profile": "observed",
                "collapse_variance_or_singular_values": "observed",
                "collapse_healthy_counterexample": "observed"
                if "control-healthy-counterexample" in control_outcomes
                else "missing",
                "collapse_negative_control": "observed"
                if "control-benign-low-variance" in control_outcomes
                else "missing",
                "collapse_provenance": "observed",
            }
        else:
            evidence = {
                "anisotropy_activity_spectrum": "observed",
                "anisotropy_predeclared_threshold": "observed",
                "anisotropy_isotropy_negative_control": (
                    "observed"
                    if "control-benign-low-variance" in control_outcomes
                    or "control-healthy-counterexample" in control_outcomes
                    else "missing"
                ),
                "anisotropy_provenance": "observed",
            }
        decision = evaluate_claim(family_id, applicability="applicable", evidence_status=evidence)
        metrics_pass = all(verdicts[metric_id] for metric_id in family_metrics)
        if failed:
            detections.append(
                FamilyDetection(
                    family_id=family_id,
                    outcome="inconclusive",
                    claim_allowed=False,
                    observed_metrics={metric_id: observed[metric_id] for metric_id in family_metrics},
                    threshold_pass={metric_id: verdicts[metric_id] for metric_id in family_metrics},
                    control_outcomes=dict(control_outcomes),
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
                    observed_metrics={metric_id: observed[metric_id] for metric_id in family_metrics},
                    threshold_pass={metric_id: verdicts[metric_id] for metric_id in family_metrics},
                    control_outcomes=dict(control_outcomes),
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
                observed_metrics={metric_id: observed[metric_id] for metric_id in family_metrics},
                threshold_pass={metric_id: verdicts[metric_id] for metric_id in family_metrics},
                control_outcomes=dict(control_outcomes),
                evidence_status=dict(evidence),
                missing_evidence=(),
                reason=(
                    "metrics pass predeclared thresholds with required controls passed"
                    if metrics_pass
                    else "metrics flag a defect predeclared thresholds with required controls passed"
                ),
            )
        )
    return tuple(detections)


def detect_families(
    value: LatentValue,
    config: DetectionConfig,
    *,
    controls: Mapping[str, LatentValue] | None = None,
) -> tuple[FamilyDetection, ...]:
    """Evaluate the configured families on one bound batch plus named controls.

    Focused unit-check seam: evaluates its own context, so composing it with a
    separate :func:`detection_payload` call evaluates twice. Production paths
    (including :func:`make_detect_executor`) must use :func:`evaluate_detection`.
    """
    config = _require_config(config)
    supplied = dict(controls or {})
    return _decide(_evaluate_context(value, config, supplied), config)


def detection_payload(
    detections: Sequence[FamilyDetection],
    config: DetectionConfig,
    value: LatentValue,
    *,
    control_metrics: Mapping[str, Mapping[str, float]] | None = None,
    context: _DetectionContext | None = None,
) -> dict[str, object]:
    """Assemble the canonical machine-readable detect-stage payload."""
    if not detections:
        raise DetectionError("detections must not be empty")
    resolved = context if context is not None else _evaluate_context(value, config, {})
    evaluated = resolved.evaluated
    try:
        canonical_json([detection.to_dict() for detection in detections])
    except PortableNodeError as exc:
        raise DetectionError(f"detections are not canonical JSON: {exc}") from exc
    supplied_metrics = {control_id: dict(metrics) for control_id, metrics in (control_metrics or {}).items()}
    for control_id in supplied_metrics:
        if control_id not in config.control_metrics:
            raise DetectionError(f"unknown control reference: {control_id!r}")

    measurements: dict[str, object] = {
        "covariance_condition": float(evaluated.covariance_condition)
        if np.isfinite(evaluated.covariance_condition)
        else None,
        "effective_rank": float(evaluated.effective_rank),
        "inactive_fraction": float(evaluated.inactive_fraction),
        "n_samples": int(evaluated.n_samples),
        "null_shuffled": dict(resolved.null_metrics),
        "representation_dim": int(evaluated.dim),
        "representation_identity": resolved.representation_identity,
        "singular_spread": float(evaluated.singular_spread),
        "singular_values": list(resolved.singular),
        "uncertainty": {key: dict(item) for key, item in resolved.uncertainty.items()},
    }
    if _FEATURE_VARIANCE_RATIO_METRIC in config.metric_ids:
        if evaluated.feature_variance_ratio is None:
            raise DetectionError("configured feature variance ratio was not evaluated")
        measurements[_FEATURE_VARIANCE_RATIO_METRIC] = evaluated.feature_variance_ratio
        measurements["feature_variance_ratios"] = list(evaluated.feature_variance_ratios)
    return {
        "config": config.to_dict(),
        "control_metrics": supplied_metrics,
        "controls": {
            control_id: dict(cast(Mapping[str, object], outcome))
            for control_id, outcome in resolved.central_outcomes.items()
        },
        "families": [detection.to_dict() for detection in detections],
        "family_evidence": {detection.family_id: dict(detection.evidence_status) for detection in detections},
        "measurements": measurements,
    }


def evaluate_detection(
    value: LatentValue,
    config: DetectionConfig,
    controls: Mapping[str, LatentValue],
) -> tuple[tuple[FamilyDetection, ...], dict[str, object]]:
    """Evaluate one batch plus controls exactly once and return decisions plus payload.

    This is the single-evaluation seam every caller must use: the target batch,
    each control batch, the shuffled null, and the seeded bootstrap are each
    computed once inside one :class:`_DetectionContext`, then shared by family
    decisions and payload assembly. ``detect_families`` and
    :func:`detection_payload` remain available for focused unit checks but must
    not be composed naively in production paths, since that composition would
    evaluate everything twice.
    """
    config = _require_config(config)
    supplied = dict(controls)
    context = _evaluate_context(value, config, supplied)
    detections = _decide(context, config)
    control_metrics: dict[str, dict[str, float]] = {}
    for control_id, item in context.control_evaluated.items():
        metrics = {
            "bottleneck-effective-rank": item.effective_rank,
            "bottleneck-singular-spread": item.singular_spread,
            "bottleneck-min-variance": item.min_variance,
            "bottleneck-inactive-fraction": item.inactive_fraction,
        }
        if item.feature_variance_ratio is not None:
            metrics[_FEATURE_VARIANCE_RATIO_METRIC] = item.feature_variance_ratio
        control_metrics[control_id] = metrics
    payload = detection_payload(detections, config, value, control_metrics=control_metrics, context=context)
    return detections, payload


def make_detect_executor(
    value: LatentValue,
    config: DetectionConfig,
    *,
    controls: Mapping[str, LatentValue] | None = None,
    version: str = "collapse-anisotropy-detector-v1",
) -> Any:
    """Build a supplied ``detect``-stage executor bound to one batch and config.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable verifies the invocation targets ``detect`` with the expected
    request/manifest identity, evaluates :func:`evaluate_detection` once, and
    returns a ``completed`` :class:`StageOutput` whose payload carries
    observations, family evidence, control outcomes, uncertainty, and claim
    gating. No algorithm enters ``DiagnosticWorkflow`` itself.
    """
    _non_empty_string(version, name="version")
    value = _require_batch(value)
    config = _require_config(config)
    frozen_controls = dict(controls or {})

    def _execute(invocation: object) -> StageOutput:
        from latent_anything._diagnostic_workflow import StageContractError as _ContractError

        invocation = _require_invocation(invocation)
        if invocation.stage != "detect":
            raise _ContractError(f"detect executor received stage {invocation.stage!r}")
        from latent_anything.diagnostics import DiagnosticRequest as _Request

        request = invocation.request
        if not isinstance(request, _Request):
            raise _ContractError("detect executor requires a DiagnosticRequest")
        if request.manifest_id != config.manifest_id:
            raise _ContractError("detect executor manifest identity mismatch")
        _, payload = evaluate_detection(value, config, frozen_controls)
        aggregate_outcome: Literal["completed", "unsupported"] = "completed"
        return StageOutput(stage="detect", outcome=aggregate_outcome, payload=payload, artifact_refs=())

    _execute.detector_version = version  # type: ignore[attr-defined]
    return _execute


__all__ = [
    "ANISOTROPY_METRICS",
    "METRIC_FAMILY",
    "SUPPORTED_FAMILIES",
    "ClaimOutcome",
    "DetectionConfig",
    "DetectionError",
    "FamilyDetection",
    "FamilyThreshold",
    "detect_families",
    "detection_config_from_manifest",
    "detection_payload",
    "evaluate_detection",
    "make_detect_executor",
]
