"""Centralized statistical-control executor (Sprint 80.15).

One boring private convention for immutable/prevalidated control plans and
deterministic execution results. Detectors bind repetitions, confidence level,
and seed roles from predeclared manifest/config inputs; the executor owns seed
streams, exact repetition counts, resampling/permutation scheduling, percentile
interval summarization, control evaluation, missing/error handling, and
canonical result/provenance. Estimator-specific logic stays in caller-supplied
fitted scores/statistics or a transform callback; no estimator is recreated
here and no expensive model is refit unless a predeclared control requires it.

Non-goals: detector algorithms, explanations/interventions, benchmark proof,
public API growth, or frozen-contract changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Any, Literal, cast

import numpy as np

from latent_anything._portable_contract import PortableNodeError, canonical_json

CONTROL_KINDS: tuple[str, ...] = (
    "bootstrap",
    "null",
    "shuffled",
    "randomized",
    "cross_seed",
    "counterexample",
    "negative",
    "capacity",
    "seed",
)
"""Control kinds the central executor can schedule. Anything else rejects."""

ControlStatus = Literal["passed", "failed", "recorded", "unsupported"]


class ControlExecutionError(ValueError):
    """Raised when a control plan, input, or execution result is fail-closed invalid."""


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlExecutionError(f"{name} must be a non-empty string")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ControlExecutionError(f"{name} must be a finite number")
    return float(value)


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ControlExecutionError(f"{name} must be a non-negative integer")
    return value


def derive_stream_seed(base_seed: int, control_id: str, *, role: str) -> int:
    """Derive a deterministic per-control stream seed from identity, never ``hash``.

    The derivation is ``sha256(role || base_seed || control_id)`` truncated to
    63 bits: stable across processes, declaration order, and execution order.
    """
    _non_empty_string(control_id, name="control_id")
    _non_empty_string(role, name="role")
    digest = sha256(
        f"{role}\x00{int(base_seed)}\x00{control_id}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % (2**63)


def summarize_interval(
    draws: Sequence[float], *, repetitions: int, seed: int, confidence_level: float
) -> dict[str, object]:
    """Summarize exactly ``repetitions`` draws with deterministic percentile bounds."""
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise ControlExecutionError("repetitions must be at least two")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ControlExecutionError("confidence_level must be between zero and one")
    values = [float(item) for item in draws]
    if len(values) != repetitions:
        raise ControlExecutionError(
            f"expected exactly {repetitions} draws, got {len(values)}"
        )
    for position, item in enumerate(values):
        if not isfinite(item):
            raise ControlExecutionError(f"draw[{position}] is non-finite")
    ordered = sorted(values)
    lower_index = int(((1.0 - confidence_level) / 2.0) * repetitions)
    upper_index = min(repetitions - 1, int((1.0 - (1.0 - confidence_level) / 2.0) * repetitions))
    return {
        "confidence_level": float(confidence_level),
        "lower": float(ordered[lower_index]),
        "mean": float(sum(values) / len(values)),
        "method": "bootstrap",
        "repetitions": int(repetitions),
        "seed": int(seed),
        "upper": float(ordered[upper_index]),
    }


@dataclass(frozen=True)
class ControlSpec:
    """One immutable predeclared control: identity, kind, gating, and comparator."""

    control_id: str
    kind: str
    required: bool
    metric_ids: tuple[str, ...]
    expected_behavior: str = ""
    comparator: str = "record"
    threshold_value: float | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.control_id, name="control_id")
        if self.kind not in CONTROL_KINDS:
            raise ControlExecutionError(f"unsupported control kind: {self.kind!r}")
        if not isinstance(self.required, bool):
            raise ControlExecutionError("required must be boolean")
        metrics = tuple(self.metric_ids)
        if not metrics:
            raise ControlExecutionError(f"control {self.control_id!r} must link declared metrics")
        for position, metric_id in enumerate(metrics):
            if not isinstance(metric_id, str) or not metric_id.strip():
                raise ControlExecutionError(f"control {self.control_id!r} metric_ids[{position}] must be a non-empty string")
        object.__setattr__(self, "metric_ids", metrics)
        if not isinstance(self.expected_behavior, str):
            raise ControlExecutionError("expected_behavior must be a string")
        if self.comparator not in ("record", "below_threshold", "meets_threshold"):
            raise ControlExecutionError(f"unsupported control comparator: {self.comparator!r}")
        if self.threshold_value is not None:
            _finite_number(self.threshold_value, name="threshold_value")
        if self.comparator != "record" and self.threshold_value is None:
            raise ControlExecutionError(f"gated control {self.control_id!r} requires a threshold_value")

    def to_dict(self) -> dict[str, object]:
        return {
            "comparator": self.comparator,
            "control_id": self.control_id,
            "expected_behavior": self.expected_behavior,
            "kind": self.kind,
            "metric_ids": list(self.metric_ids),
            "required": self.required,
            "threshold_value": None if self.threshold_value is None else float(self.threshold_value),
        }


@dataclass(frozen=True)
class ControlPlan:
    """Immutable prevalidated execution plan: seeds, repetitions, and control specs."""

    plan_id: str
    repetitions: int
    confidence_level: float
    evaluation_seed: int
    control_seed: int
    training_seed: int | None = None
    controls: tuple[ControlSpec, ...] = ()

    def __post_init__(self) -> None:
        _non_empty_string(self.plan_id, name="plan_id")
        if isinstance(self.repetitions, bool) or not isinstance(self.repetitions, int) or self.repetitions < 2:
            raise ControlExecutionError("repetitions must be at least two")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ControlExecutionError("confidence_level must be between zero and one")
        object.__setattr__(self, "evaluation_seed", _non_negative_int(self.evaluation_seed, name="evaluation_seed"))
        object.__setattr__(self, "control_seed", _non_negative_int(self.control_seed, name="control_seed"))
        if self.training_seed is not None:
            object.__setattr__(self, "training_seed", _non_negative_int(self.training_seed, name="training_seed"))
        specs = tuple(self.controls)
        if not specs:
            raise ControlExecutionError("controls must not be empty")
        for spec in specs:
            if not isinstance(spec, ControlSpec):
                raise ControlExecutionError("controls must hold ControlSpec items")
        identities = [spec.control_id for spec in specs]
        if len(set(identities)) != len(identities):
            raise ControlExecutionError("control identifiers must be unique")
        object.__setattr__(self, "controls", specs)

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence_level": float(self.confidence_level),
            "control_seed": int(self.control_seed),
            "controls": [spec.to_dict() for spec in self.controls],
            "evaluation_seed": int(self.evaluation_seed),
            "plan_id": self.plan_id,
            "repetitions": int(self.repetitions),
            "training_seed": None if self.training_seed is None else int(self.training_seed),
        }


@dataclass(frozen=True)
class ControlOutcome:
    """One deterministic execution result with full seed/interval provenance."""

    control_id: str
    kind: str
    status: ControlStatus
    required: bool
    observed: Mapping[str, float]
    expected_behavior: str
    repetitions: int
    confidence_level: float
    stream_seed: int
    seed_role: str
    interval: Mapping[str, object] | None = None
    reason: str = ""
    detail: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.control_id, name="control_id")
        if self.kind not in CONTROL_KINDS:
            raise ControlExecutionError(f"unsupported control kind: {self.kind!r}")
        if self.status not in ("passed", "failed", "recorded", "unsupported"):
            raise ControlExecutionError(f"unsupported control status: {self.status!r}")
        if not isinstance(self.required, bool):
            raise ControlExecutionError("required must be boolean")
        object.__setattr__(self, "observed", dict(self.observed))
        for metric_id, value in self.observed.items():
            if not isinstance(metric_id, str) or not metric_id:
                raise ControlExecutionError("observed metric identities must be non-empty strings")
            _finite_number(value, name=f"control {self.control_id!r} observed[{metric_id}]")
        if not isinstance(self.expected_behavior, str):
            raise ControlExecutionError("expected_behavior must be a string")
        if isinstance(self.repetitions, bool) or not isinstance(self.repetitions, int) or self.repetitions < 2:
            raise ControlExecutionError("repetitions must be at least two")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ControlExecutionError("confidence_level must be between zero and one")
        _non_negative_int(self.stream_seed, name="stream_seed")
        _non_empty_string(self.seed_role, name="seed_role")
        if self.interval is not None:
            object.__setattr__(self, "interval", dict(self.interval))
        if self.detail is not None:
            if not isinstance(self.detail, Mapping):
                raise ControlExecutionError("detail must be a mapping")
            object.__setattr__(self, "detail", dict(self.detail))
        _non_empty_string(self.reason, name="reason")
        if self.required and self.status in ("recorded", "unsupported"):
            raise ControlExecutionError(f"required control {self.control_id!r} must be passed or failed")
        if self.status == "unsupported":
            raise ControlExecutionError(f"control {self.control_id!r} is unsupported; it must never read as pass")

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence_level": float(self.confidence_level),
            "control_id": self.control_id,
            "detail": None if self.detail is None else dict(self.detail),
            "expected_behavior": self.expected_behavior,
            "interval": None if self.interval is None else dict(self.interval),
            "kind": self.kind,
            "observed": dict(self.observed),
            "reason": self.reason,
            "repetitions": int(self.repetitions),
            "required": self.required,
            "seed_role": self.seed_role,
            "status": self.status,
            "stream_seed": int(self.stream_seed),
        }


def _string_tuple(value: object, *, name: str, minimum: int = 1) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ControlExecutionError(f"{name} must be a list of strings")
    items = tuple(value)
    for position, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ControlExecutionError(f"{name}[{position}] must be a non-empty string")
    if len(items) < minimum:
        raise ControlExecutionError(f"{name} must hold at least {minimum} item(s)")
    return items


def plan_from_manifest(
    *,
    plan_id: str,
    manifest: Mapping[str, object],
    metric_ids: Sequence[str],
    thresholds: Mapping[str, tuple[str, float]] | None = None,
    evaluation_seed: int,
    control_seed: int,
    training_seed: int | None = None,
) -> ControlPlan:
    """Bind repetitions, confidence level, seeds, kinds, and thresholds from a manifest.

    No observed-data tuning and no implicit defaults: repetitions and confidence
    come from ``manifest.uncertainty``, seeds from ``manifest.seeds``, kinds and
    required flags from ``manifest.controls``. Comparators default to ``record``
    unless ``thresholds`` supplies an explicit ``(comparator, value)`` pair for
    a linked metric.
    """
    from latent_anything._benchmark_manifest import BenchmarkManifestValidationError, validate_manifest

    _non_empty_string(plan_id, name="plan_id")
    if not isinstance(manifest, Mapping):
        raise ControlExecutionError("manifest must be a mapping")
    try:
        validate_manifest(manifest)
    except BenchmarkManifestValidationError as exc:
        raise ControlExecutionError(f"invalid benchmark manifest: {exc}") from exc
    requested = _string_tuple(metric_ids, name="metric_ids", minimum=1)
    declared_thresholds = dict(thresholds or {})

    uncertainty = manifest.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise ControlExecutionError("manifest uncertainty must be an object")
    repetitions = uncertainty.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise ControlExecutionError("uncertainty.repetitions must be at least two")
    confidence = uncertainty.get("confidence_level")
    confidence_level = _finite_number(confidence, name="uncertainty.confidence_level")
    if not 0.0 < confidence_level < 1.0:
        raise ControlExecutionError("uncertainty.confidence_level must be between zero and one")

    seeds = manifest.get("seeds")
    if not isinstance(seeds, Mapping):
        raise ControlExecutionError("manifest seeds must be an object")
    for role in ("evaluation", "controls"):
        rows = seeds.get(role)
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
            raise ControlExecutionError(f"seeds.{role} must be a non-empty list")
    manifest_evaluation = seeds.get("evaluation")
    manifest_controls = seeds.get("controls")
    assert isinstance(manifest_evaluation, Sequence) and not isinstance(manifest_evaluation, (str, bytes))
    assert isinstance(manifest_controls, Sequence) and not isinstance(manifest_controls, (str, bytes))
    if int(manifest_evaluation[0]) != int(evaluation_seed) or int(manifest_controls[0]) != int(control_seed):
        raise ControlExecutionError("seed roles must match the manifest seeds.evaluation/controls entries")
    if training_seed is not None and "training" in seeds:
        rows = seeds.get("training")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows:
            if int(rows[0]) != int(training_seed):
                raise ControlExecutionError("training seed must match the manifest seeds.training entry")

    raw_controls = manifest.get("controls")
    if isinstance(raw_controls, (str, bytes)) or not isinstance(raw_controls, Sequence) or not raw_controls:
        raise ControlExecutionError("manifest controls must be a non-empty list")
    specs: list[ControlSpec] = []
    for index, raw in enumerate(raw_controls):
        if not isinstance(raw, Mapping):
            raise ControlExecutionError(f"controls[{index}] must be an object")
        control = cast(Mapping[str, object], raw)
        control_id = control.get("id")
        kind = control.get("kind")
        _non_empty_string(control_id, name=f"controls[{index}].id")
        _non_empty_string(kind, name=f"controls[{index}].kind")
        assert isinstance(control_id, str) and isinstance(kind, str)
        if kind not in CONTROL_KINDS:
            raise ControlExecutionError(f"unsupported manifest control kind: {kind!r}")
        linked = control.get("metric_ids")
        linked_ids = _string_tuple(linked, name=f"controls[{index}].metric_ids", minimum=1)
        required = control.get("required")
        if not isinstance(required, bool):
            raise ControlExecutionError(f"controls[{index}].required must be boolean")
        expected = control.get("expected_behavior")
        if not isinstance(expected, str):
            raise ControlExecutionError(f"controls[{index}].expected_behavior must be a string")
        comparator = "record"
        threshold_value: float | None = None
        for metric_id in linked_ids:
            if metric_id in declared_thresholds:
                pair = declared_thresholds[metric_id]
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ControlExecutionError(f"threshold for {metric_id!r} must be a (comparator, value) pair")
                gate, target = pair
                if gate == "record":
                    continue
                if gate not in ("below_threshold", "meets_threshold"):
                    raise ControlExecutionError(f"unsupported control comparator: {gate!r}")
                comparator = gate
                threshold_value = _finite_number(target, name=f"threshold for {metric_id!r}")
        if comparator != "record" and threshold_value is None:
            raise ControlExecutionError(f"gated control {control_id!r} requires a threshold_value")
        specs.append(
            ControlSpec(
                control_id=control_id,
                kind=kind,
                required=required,
                metric_ids=tuple(m for m in linked_ids if m in requested) or linked_ids,
                expected_behavior=expected,
                comparator=comparator,
                threshold_value=threshold_value,
            )
        )
    if not specs:
        raise ControlExecutionError("manifest declares no controls")
    return ControlPlan(
        plan_id=plan_id,
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        evaluation_seed=int(evaluation_seed),
        control_seed=int(control_seed),
        training_seed=None if training_seed is None else int(training_seed),
        controls=tuple(specs),
    )


DrawFn = Callable[[np.random.Generator], float]
"""One resample draw: consumes the caller's fitted state, returns one float."""


def run_bootstrap(
    *,
    control_id: str,
    base_seed: int,
    seed_role: str,
    repetitions: int,
    confidence_level: float,
    required: bool,
    kind: str,
    metric_ids: Sequence[str],
    expected_behavior: str,
    draw: DrawFn,
    observed: Mapping[str, float] | None = None,
) -> tuple[ControlOutcome, dict[str, object]]:
    """Run exactly ``repetitions`` draws under an identity-derived stream.

    The caller supplies already-fitted state via ``draw``; the executor owns
    the RNG stream, repetition count, finiteness checks, and percentile summary.
    """
    _non_empty_string(control_id, name="control_id")
    stream = derive_stream_seed(int(base_seed), control_id, role=seed_role)
    rng = np.random.default_rng(stream)
    draws: list[float] = []
    try:
        for _ in range(int(repetitions)):
            draws.append(float(draw(rng)))
    except ControlExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - execution failures block, never pass
        return ControlOutcome(
            control_id=control_id, kind=kind, status="failed", required=bool(required),
            observed={}, expected_behavior=expected_behavior, repetitions=int(repetitions),
            confidence_level=float(confidence_level), stream_seed=stream, seed_role=seed_role,
            interval=None, reason=f"control draw failed: {exc}",
        ), {"control_id": control_id, "status": "failed", "stream_seed": stream}
    if len(draws) != int(repetitions):
        return ControlOutcome(
            control_id=control_id, kind=kind, status="failed", required=bool(required),
            observed={}, expected_behavior=expected_behavior, repetitions=int(repetitions),
            confidence_level=float(confidence_level), stream_seed=stream, seed_role=seed_role,
            interval=None, reason=f"wrong repetition count: got {len(draws)}, expected {repetitions}",
        ), {"control_id": control_id, "status": "failed", "stream_seed": stream}
    non_finite = [position for position, item in enumerate(draws) if not isfinite(item)]
    if non_finite:
        return ControlOutcome(
            control_id=control_id, kind=kind, status="failed", required=bool(required),
            observed={}, expected_behavior=expected_behavior, repetitions=int(repetitions),
            confidence_level=float(confidence_level), stream_seed=stream, seed_role=seed_role,
            interval=None, reason=f"non-finite draws at positions {non_finite[:4]}",
        ), {"control_id": control_id, "status": "failed", "stream_seed": stream}
    interval = summarize_interval(
        draws, repetitions=int(repetitions), seed=stream, confidence_level=float(confidence_level)
    )
    observed_map = dict(observed or {})
    status: ControlStatus = "recorded" if not bool(required) else "passed"
    outcome = ControlOutcome(
        control_id=control_id, kind=kind, status=status, required=bool(required),
        observed=observed_map, expected_behavior=expected_behavior,
        repetitions=int(repetitions), confidence_level=float(confidence_level),
        stream_seed=stream, seed_role=seed_role, interval=interval,
        reason="bootstrap interval recorded under the identity-derived stream",
    )
    return outcome, dict(interval)


def run_permutation_control(
    *,
    control_id: str,
    base_seed: int,
    seed_role: str,
    required: bool,
    kind: str,
    metric_ids: Sequence[str],
    expected_behavior: str,
    statistic: Callable[[np.random.Generator], Mapping[str, float]],
    comparator: str = "record",
    threshold_value: float | None = None,
) -> ControlOutcome:
    """Evaluate one null/shuffled/randomized/cross-seed control under its own stream.

    ``statistic`` runs the caller-supplied transform exactly once (already-fitted
    scores are preferred; one seeded permutation callback is accepted where the
    predeclared control requires it). Gated comparators compare the observed
    value against the predeclared threshold; ``record`` stays informational.
    """
    _non_empty_string(control_id, name="control_id")
    if kind not in ("null", "shuffled", "randomized", "cross_seed", "seed", "counterexample", "negative", "capacity", "bootstrap"):
        raise ControlExecutionError(f"unsupported permutation control kind: {kind!r}")
    stream = derive_stream_seed(int(base_seed), control_id, role=seed_role)
    rng = np.random.default_rng(stream)
    try:
        observed_raw = statistic(rng)
    except ControlExecutionError as exc:
        return ControlOutcome(
            control_id=control_id, kind=kind, status="failed", required=bool(required),
            observed={}, expected_behavior=expected_behavior, repetitions=2,
            confidence_level=0.95, stream_seed=stream, seed_role=seed_role,
            interval=None, reason=f"control statistic failed: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - execution failures block, never pass
        return ControlOutcome(
            control_id=control_id, kind=kind, status="failed", required=bool(required),
            observed={}, expected_behavior=expected_behavior, repetitions=2,
            confidence_level=0.95, stream_seed=stream, seed_role=seed_role,
            interval=None, reason=f"control statistic failed: {exc}",
        )
    if not isinstance(observed_raw, Mapping):
        return ControlOutcome(
            control_id=control_id, kind=kind, status="failed", required=bool(required),
            observed={}, expected_behavior=expected_behavior, repetitions=2,
            confidence_level=0.95, stream_seed=stream, seed_role=seed_role,
            interval=None, reason="control statistic must return a metric mapping",
        )
    observed: dict[str, float] = {}
    detail: dict[str, object] = {}
    for key, value in observed_raw.items():
        name = str(key)
        if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
            detail[name] = dict(value) if isinstance(value, Mapping) else list(value)
            continue
        observed[name] = float(value)
    for metric_id, value in observed.items():
        if not isfinite(value):
            return ControlOutcome(
                control_id=control_id, kind=kind, status="failed", required=bool(required),
                observed={}, expected_behavior=expected_behavior, repetitions=2,
                confidence_level=0.95, stream_seed=stream, seed_role=seed_role,
                interval=None, reason=f"non-finite control result for {metric_id!r}",
            )
    linked = [metric_id for metric_id in metric_ids if metric_id in observed]
    if comparator == "record":
        status: ControlStatus = "recorded" if not bool(required) else "passed"
        return ControlOutcome(
            control_id=control_id, kind=kind, status=status, required=bool(required),
            observed=observed, expected_behavior=expected_behavior, repetitions=2,
            confidence_level=0.95, stream_seed=stream, seed_role=seed_role,
            interval=None, detail=detail or None,
            reason="recorded control; not threshold-gated",
        )
    if threshold_value is None or not linked:
        return ControlOutcome(
            control_id=control_id, kind=kind, status="failed", required=bool(required),
            observed=observed, expected_behavior=expected_behavior, repetitions=2,
            confidence_level=0.95, stream_seed=stream, seed_role=seed_role,
            interval=None, detail=detail or None,
            reason="gated control is missing its threshold or linked metric",
        )
    threshold = float(threshold_value)
    if comparator == "below_threshold":
        gated = all(observed[metric_id] < threshold for metric_id in linked)
    elif comparator == "meets_threshold":
        gated = all(observed[metric_id] >= threshold for metric_id in linked)
    else:
        raise ControlExecutionError(f"unsupported control comparator: {comparator!r}")
    return ControlOutcome(
        control_id=control_id, kind=kind, status="passed" if gated else "failed",
        required=bool(required), observed=observed, expected_behavior=expected_behavior,
        repetitions=2, confidence_level=0.95, stream_seed=stream, seed_role=seed_role,
        interval=None, detail=detail or None,
        reason=f"control comparator {comparator} against threshold {threshold}",
    )


def execute_plan(
    plan: ControlPlan,
    supplied: Mapping[str, Mapping[str, float] | None],
    *,
    statistics: Mapping[str, Callable[[np.random.Generator], Mapping[str, float]] | None] | None = None,
    intervals: Mapping[str, dict[str, object]] | None = None,
) -> dict[str, ControlOutcome]:
    """Evaluate every control in a plan with order-invariant identity-derived streams.

    ``supplied`` maps each control identity to its already-computed observed
    metric mapping (or ``None`` when missing). ``statistics`` optionally carries
    one transform callback per control for derived null/shuffled/randomized
    kinds. ``intervals`` optionally carries precomputed central summaries for
    bootstrap-kind controls. Controls are executed in sorted identity order, so
    declaration/execution order cannot change results. Missing required data,
    execution failure, non-finite results, or a failed comparator yields
    ``failed``; missing/failed required controls block downstream conclusions.
    Optional controls without gating stay ``recorded``.
    """
    if not isinstance(plan, ControlPlan):
        raise ControlExecutionError("plan must be a ControlPlan")
    if not isinstance(supplied, Mapping):
        raise ControlExecutionError("supplied must be a mapping")
    callbacks = dict(statistics or {})
    summaries = dict(intervals or {})
    ordered = sorted(plan.controls, key=lambda spec: spec.control_id)
    outcomes: dict[str, ControlOutcome] = {}
    for spec in ordered:
        present = spec.control_id in supplied and supplied[spec.control_id] is not None
        if spec.control_id in callbacks:
            outcomes[spec.control_id] = run_permutation_control(
                control_id=spec.control_id,
                base_seed=plan.control_seed,
                seed_role="control",
                required=spec.required,
                kind=spec.kind,
                metric_ids=list(spec.metric_ids),
                expected_behavior=spec.expected_behavior,
                statistic=callbacks[spec.control_id],
                comparator=spec.comparator,
                threshold_value=spec.threshold_value,
            )
            continue
        if spec.kind == "bootstrap" and spec.control_id in summaries:
            summary = summaries[spec.control_id]
            if not isinstance(summary, Mapping):
                raise ControlExecutionError(f"interval for {spec.control_id!r} must be a mapping")
            outcomes[spec.control_id] = ControlOutcome(
                control_id=spec.control_id, kind=spec.kind,
                status="passed" if spec.required else "recorded", required=spec.required,
                observed=dict(supplied.get(spec.control_id) or {}),
                expected_behavior=spec.expected_behavior,
                repetitions=plan.repetitions, confidence_level=plan.confidence_level,
                stream_seed=derive_stream_seed(plan.control_seed, spec.control_id, role="control"),
                seed_role="control", interval=dict(summary),
                reason="bootstrap interval supplied from the central repetition schedule",
            )
            continue
        if not present:
            outcomes[spec.control_id] = ControlOutcome(
                control_id=spec.control_id, kind=spec.kind,
                status="failed", required=spec.required,
                observed={}, expected_behavior=spec.expected_behavior,
                repetitions=plan.repetitions, confidence_level=plan.confidence_level,
                stream_seed=derive_stream_seed(plan.control_seed, spec.control_id, role="control"),
                seed_role="control", interval=None,
                reason="required control is missing data" if spec.required else "optional control has no data; recorded as failed-optional",
            )
            if not spec.required:
                outcomes[spec.control_id] = ControlOutcome(
                    control_id=spec.control_id, kind=spec.kind,
                    status="recorded", required=False,
                    observed={}, expected_behavior=spec.expected_behavior,
                    repetitions=plan.repetitions, confidence_level=plan.confidence_level,
                    stream_seed=derive_stream_seed(plan.control_seed, spec.control_id, role="control"),
                    seed_role="control", interval=None,
                    reason="optional control has no data; recorded without gating",
                )
            continue
        observed_raw = cast(Mapping[str, float], supplied[spec.control_id])
        observed = {str(key): float(value) for key, value in observed_raw.items()}
        non_finite = [metric_id for metric_id, value in observed.items() if not isfinite(value)]
        if non_finite:
            outcomes[spec.control_id] = ControlOutcome(
                control_id=spec.control_id, kind=spec.kind,
                status="failed", required=spec.required,
                observed={}, expected_behavior=spec.expected_behavior,
                repetitions=plan.repetitions, confidence_level=plan.confidence_level,
                stream_seed=derive_stream_seed(plan.control_seed, spec.control_id, role="control"),
                seed_role="control", interval=None,
                reason=f"non-finite control result for {', '.join(sorted(non_finite))}",
            )
            continue
        if spec.comparator == "record":
            outcomes[spec.control_id] = ControlOutcome(
                control_id=spec.control_id, kind=spec.kind,
                status="passed" if spec.required else "recorded", required=spec.required,
                observed=observed, expected_behavior=spec.expected_behavior,
                repetitions=plan.repetitions, confidence_level=plan.confidence_level,
                stream_seed=derive_stream_seed(plan.control_seed, spec.control_id, role="control"),
                seed_role="control", interval=None,
                reason="recorded control; not threshold-gated",
            )
            continue
        linked = [metric_id for metric_id in spec.metric_ids if metric_id in observed]
        if spec.threshold_value is None or not linked:
            outcomes[spec.control_id] = ControlOutcome(
                control_id=spec.control_id, kind=spec.kind,
                status="failed", required=spec.required,
                observed=observed, expected_behavior=spec.expected_behavior,
                repetitions=plan.repetitions, confidence_level=plan.confidence_level,
                stream_seed=derive_stream_seed(plan.control_seed, spec.control_id, role="control"),
                seed_role="control", interval=None,
                reason="gated control is missing its threshold or linked metric",
            )
            continue
        threshold = float(spec.threshold_value)
        if spec.comparator == "below_threshold":
            gated = all(observed[metric_id] < threshold for metric_id in linked)
        elif spec.comparator == "meets_threshold":
            gated = all(observed[metric_id] >= threshold for metric_id in linked)
        else:
            raise ControlExecutionError(f"unsupported control comparator: {spec.comparator!r}")
        outcomes[spec.control_id] = ControlOutcome(
            control_id=spec.control_id, kind=spec.kind,
            status="passed" if gated else "failed", required=spec.required,
            observed=observed, expected_behavior=spec.expected_behavior,
            repetitions=plan.repetitions, confidence_level=plan.confidence_level,
            stream_seed=derive_stream_seed(plan.control_seed, spec.control_id, role="control"),
            seed_role="control", interval=None,
            reason=f"control comparator {spec.comparator} against threshold {threshold}",
        )
    try:
        canonical_json({key: value.to_dict() for key, value in outcomes.items()})
    except PortableNodeError as exc:
        raise ControlExecutionError(f"control outcomes are not canonical JSON: {exc}") from exc
    return outcomes


def failed_required(outcomes: Mapping[str, ControlOutcome]) -> tuple[str, ...]:
    """Return sorted identities of required controls whose status is ``failed``."""
    return tuple(sorted(
        control_id for control_id, outcome in outcomes.items()
        if outcome.required and outcome.status == "failed"
    ))


__all__ = [
    "CONTROL_KINDS",
    "ControlExecutionError",
    "ControlOutcome",
    "ControlPlan",
    "ControlSpec",
    "derive_stream_seed",
    "execute_plan",
    "failed_required",
    "plan_from_manifest",
    "run_bootstrap",
    "run_permutation_control",
    "summarize_interval",
]
