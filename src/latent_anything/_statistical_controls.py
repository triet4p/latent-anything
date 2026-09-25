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
from typing import Literal, cast

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


def _require_repetitions(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ControlExecutionError("repetitions must be at least two")
    return int(value)


def _require_confidence(value: object) -> float:
    try:
        level = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ControlExecutionError("confidence_level must be between zero and one") from exc
    if not 0.0 < level < 1.0:
        raise ControlExecutionError("confidence_level must be between zero and one")
    return level


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ControlExecutionError(f"{name} must be boolean")
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ControlExecutionError(f"{name} must be a string")
    return value


def _require_metric_ids(value: object, *, control_id: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ControlExecutionError(f"control {control_id!r} must link declared metrics")
    items = tuple(value)
    if not items:
        raise ControlExecutionError(f"control {control_id!r} must link declared metrics")
    for position, metric_id in enumerate(items):
        if not isinstance(metric_id, str) or not metric_id.strip():
            raise ControlExecutionError(f"control {control_id!r} metric_ids[{position}] must be a non-empty string")
    return items


def _require_specs(value: object) -> tuple[ControlSpec, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ControlExecutionError("controls must hold ControlSpec items")
    items = tuple(value)
    if not items:
        raise ControlExecutionError("controls must not be empty")
    for spec in items:
        if not isinstance(spec, ControlSpec):
            raise ControlExecutionError("controls must hold ControlSpec items")
    return items


def _require_observed(value: object) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ControlExecutionError("observed must be a mapping")
    return value


_StatisticFn = Callable[..., object]
"""One caller-supplied control transform over the identity-derived stream."""


def _require_statistic(value: object, *, name: str) -> _StatisticFn:
    if value is None or not callable(value):
        raise ControlExecutionError(f"control {name!r} statistic must be callable")
    return value


def _require_supplied_observed(value: object, *, name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ControlExecutionError(f"control {name!r} observed must be a mapping")
    return value


def _require_section(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ControlExecutionError(f"{name} must be an object")
    return value


def _require_seed_rows(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ControlExecutionError(f"{name} must be a non-empty list")
    return value


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
    digest = sha256(f"{role}\x00{int(base_seed)}\x00{control_id}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**63)


def summarize_interval(
    draws: Sequence[float], *, repetitions: object, seed: object, confidence_level: object
) -> dict[str, object]:
    """Summarize exactly ``repetitions`` draws with deterministic percentile bounds."""
    repetitions = _require_repetitions(repetitions)
    confidence_level = _require_confidence(confidence_level)
    seed = _non_negative_int(seed, name="seed")
    values = [float(item) for item in draws]
    if len(values) != repetitions:
        raise ControlExecutionError(f"expected exactly {repetitions} draws, got {len(values)}")
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
    required: object
    metric_ids: object
    expected_behavior: str = ""
    comparator: str = "record"
    threshold_value: float | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.control_id, name="control_id")
        if self.kind not in CONTROL_KINDS:
            raise ControlExecutionError(f"unsupported control kind: {self.kind!r}")
        required = _require_bool(self.required, name="required")
        object.__setattr__(self, "required", required)
        metrics = _require_metric_ids(self.metric_ids, control_id=self.control_id)
        object.__setattr__(self, "metric_ids", metrics)
        behavior = _require_text(self.expected_behavior, name="expected_behavior")
        object.__setattr__(self, "expected_behavior", behavior)
        if self.comparator not in ("record", "below_threshold", "meets_threshold"):
            raise ControlExecutionError(f"unsupported control comparator: {self.comparator!r}")
        if self.threshold_value is not None:
            _finite_number(self.threshold_value, name="threshold_value")
        if self.comparator != "record" and self.threshold_value is None:
            raise ControlExecutionError(f"gated control {self.control_id!r} requires a threshold_value")

    def to_dict(self) -> dict[str, object]:
        required = _require_bool(self.required, name="required")
        metrics = _require_metric_ids(self.metric_ids, control_id=self.control_id)
        behavior = _require_text(self.expected_behavior, name="expected_behavior")
        threshold = self.threshold_value
        return {
            "comparator": self.comparator,
            "control_id": self.control_id,
            "expected_behavior": behavior,
            "kind": self.kind,
            "metric_ids": list(metrics),
            "required": required,
            "threshold_value": None if threshold is None else float(threshold),
        }


@dataclass(frozen=True)
class ControlPlan:
    """Immutable prevalidated execution plan: seeds, repetitions, and control specs."""

    plan_id: str
    repetitions: object
    confidence_level: object
    evaluation_seed: object
    control_seed: object
    training_seed: object = None
    controls: object = ()

    def __post_init__(self) -> None:
        _non_empty_string(self.plan_id, name="plan_id")
        repetitions = _require_repetitions(self.repetitions)
        confidence = _require_confidence(self.confidence_level)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "evaluation_seed", _non_negative_int(self.evaluation_seed, name="evaluation_seed"))
        object.__setattr__(self, "control_seed", _non_negative_int(self.control_seed, name="control_seed"))
        if self.training_seed is not None:
            object.__setattr__(self, "training_seed", _non_negative_int(self.training_seed, name="training_seed"))
        specs = _require_specs(self.controls)
        identities = [spec.control_id for spec in specs]
        if len(set(identities)) != len(identities):
            raise ControlExecutionError("control identifiers must be unique")
        object.__setattr__(self, "controls", specs)

    def to_dict(self) -> dict[str, object]:
        repetitions = _require_repetitions(self.repetitions)
        confidence = _require_confidence(self.confidence_level)
        evaluation = _non_negative_int(self.evaluation_seed, name="evaluation_seed")
        control = _non_negative_int(self.control_seed, name="control_seed")
        specs = _require_specs(self.controls)
        training = None if self.training_seed is None else _non_negative_int(self.training_seed, name="training_seed")
        return {
            "confidence_level": confidence,
            "control_seed": control,
            "controls": [spec.to_dict() for spec in specs],
            "evaluation_seed": evaluation,
            "plan_id": self.plan_id,
            "repetitions": repetitions,
            "training_seed": training,
        }


@dataclass(frozen=True)
class ControlOutcome:
    """One deterministic execution result with full seed/interval provenance."""

    control_id: str
    kind: str
    status: ControlStatus
    required: object
    observed: object
    expected_behavior: object
    repetitions: object
    confidence_level: object
    stream_seed: object
    seed_role: str
    interval: object = None
    reason: str = ""
    detail: object = None

    def __post_init__(self) -> None:
        _non_empty_string(self.control_id, name="control_id")
        if self.kind not in CONTROL_KINDS:
            raise ControlExecutionError(f"unsupported control kind: {self.kind!r}")
        if self.status not in ("passed", "failed", "recorded", "unsupported"):
            raise ControlExecutionError(f"unsupported control status: {self.status!r}")
        required = _require_bool(self.required, name="required")
        object.__setattr__(self, "required", required)
        observed = _require_observed(self.observed)
        object.__setattr__(self, "observed", dict(observed))
        for metric_id, value in observed.items():
            identity = _non_empty_string(metric_id, name="observed metric identities")
            _finite_number(value, name=f"control {self.control_id!r} observed[{identity}]")
        behavior = _require_text(self.expected_behavior, name="expected_behavior")
        object.__setattr__(self, "expected_behavior", behavior)
        repetitions = _require_repetitions(self.repetitions)
        confidence = _require_confidence(self.confidence_level)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "confidence_level", confidence)
        stream = _non_negative_int(self.stream_seed, name="stream_seed")
        object.__setattr__(self, "stream_seed", stream)
        _non_empty_string(self.seed_role, name="seed_role")
        if self.interval is not None:
            interval = _require_observed(self.interval)
            object.__setattr__(self, "interval", dict(interval))
        if self.detail is not None:
            detail = _require_observed(self.detail)
            object.__setattr__(self, "detail", dict(detail))
        _non_empty_string(self.reason, name="reason")
        if self.required and self.status in ("recorded", "unsupported"):
            raise ControlExecutionError(f"required control {self.control_id!r} must be passed or failed")
        if self.status == "unsupported":
            raise ControlExecutionError(f"control {self.control_id!r} is unsupported; it must never read as pass")

    def to_dict(self) -> dict[str, object]:
        required = _require_bool(self.required, name="required")
        observed = _require_observed(self.observed)
        behavior = _require_text(self.expected_behavior, name="expected_behavior")
        repetitions = _require_repetitions(self.repetitions)
        confidence = _require_confidence(self.confidence_level)
        stream = _non_negative_int(self.stream_seed, name="stream_seed")
        interval = None if self.interval is None else dict(_require_observed(self.interval))
        detail = None if self.detail is None else dict(_require_observed(self.detail))
        return {
            "confidence_level": confidence,
            "control_id": self.control_id,
            "detail": detail,
            "expected_behavior": behavior,
            "interval": interval,
            "kind": self.kind,
            "observed": dict(observed),
            "reason": self.reason,
            "repetitions": repetitions,
            "required": required,
            "seed_role": self.seed_role,
            "status": self.status,
            "stream_seed": stream,
        }


def _require_threshold_pair(value: object, *, name: str) -> tuple[str, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ControlExecutionError(f"threshold for {name!r} must be a (comparator, value) pair")
    gate, target = value
    if not isinstance(gate, str):
        raise ControlExecutionError(f"threshold for {name!r} must be a (comparator, value) pair")
    return gate, _finite_number(target, name=f"threshold for {name!r}")


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
    try:
        validate_manifest(manifest)
    except BenchmarkManifestValidationError as exc:
        raise ControlExecutionError(f"invalid benchmark manifest: {exc}") from exc
    requested = _string_tuple(metric_ids, name="metric_ids", minimum=1)
    declared_thresholds = dict(thresholds or {})

    uncertainty = _require_section(manifest.get("uncertainty"), name="manifest uncertainty")
    repetitions = _require_repetitions(uncertainty.get("repetitions"))
    confidence_raw = _finite_number(uncertainty.get("confidence_level"), name="uncertainty.confidence_level")
    confidence_level = _require_confidence(confidence_raw)

    seeds = _require_section(manifest.get("seeds"), name="manifest seeds")
    for role in ("evaluation", "controls"):
        _require_seed_rows(seeds.get(role), name=f"seeds.{role}")
    manifest_evaluation = seeds.get("evaluation")
    manifest_controls = seeds.get("controls")
    assert isinstance(manifest_evaluation, Sequence) and not isinstance(manifest_evaluation, (str, bytes))
    assert isinstance(manifest_controls, Sequence) and not isinstance(manifest_controls, (str, bytes))
    if int(manifest_evaluation[0]) != int(evaluation_seed) or int(manifest_controls[0]) != int(control_seed):
        raise ControlExecutionError("seed roles must match the manifest seeds.evaluation/controls entries")
    if training_seed is not None and "training" in seeds:
        rows = seeds.get("training")
        if (
            isinstance(rows, Sequence)
            and not isinstance(rows, (str, bytes))
            and rows
            and int(rows[0]) != int(training_seed)
        ):
            raise ControlExecutionError("training seed must match the manifest seeds.training entry")

    raw_controls = manifest.get("controls")
    if isinstance(raw_controls, (str, bytes)) or not isinstance(raw_controls, Sequence) or not raw_controls:
        raise ControlExecutionError("manifest controls must be a non-empty list")
    specs: list[ControlSpec] = []
    for index, raw in enumerate(raw_controls):
        if not isinstance(raw, Mapping):
            raise ControlExecutionError(f"controls[{index}] must be an object")
        control = cast(Mapping[str, object], raw)
        control_id = _non_empty_string(control.get("id"), name=f"controls[{index}].id")
        kind = _non_empty_string(control.get("kind"), name=f"controls[{index}].kind")
        if kind not in CONTROL_KINDS:
            raise ControlExecutionError(f"unsupported manifest control kind: {kind!r}")
        linked = control.get("metric_ids")
        linked_ids = _string_tuple(linked, name=f"controls[{index}].metric_ids", minimum=1)
        required = _require_bool(control.get("required"), name=f"controls[{index}].required")
        expected = _require_text(control.get("expected_behavior"), name=f"controls[{index}].expected_behavior")
        comparator = "record"
        threshold_value: float | None = None
        for metric_id in linked_ids:
            if metric_id in declared_thresholds:
                gate, target = _require_threshold_pair(declared_thresholds[metric_id], name=metric_id)
                if gate == "record":
                    continue
                if gate not in ("below_threshold", "meets_threshold"):
                    raise ControlExecutionError(f"unsupported control comparator: {gate!r}")
                comparator = gate
                threshold_value = target
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


DrawFn = Callable[[object], float]
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
            control_id=control_id,
            kind=kind,
            status="failed",
            required=bool(required),
            observed={},
            expected_behavior=expected_behavior,
            repetitions=int(repetitions),
            confidence_level=float(confidence_level),
            stream_seed=stream,
            seed_role=seed_role,
            interval=None,
            reason=f"control draw failed: {exc}",
        ), {"control_id": control_id, "status": "failed", "stream_seed": stream}
    if len(draws) != int(repetitions):
        return ControlOutcome(
            control_id=control_id,
            kind=kind,
            status="failed",
            required=bool(required),
            observed={},
            expected_behavior=expected_behavior,
            repetitions=int(repetitions),
            confidence_level=float(confidence_level),
            stream_seed=stream,
            seed_role=seed_role,
            interval=None,
            reason=f"wrong repetition count: got {len(draws)}, expected {repetitions}",
        ), {"control_id": control_id, "status": "failed", "stream_seed": stream}
    non_finite = [position for position, item in enumerate(draws) if not isfinite(item)]
    if non_finite:
        return ControlOutcome(
            control_id=control_id,
            kind=kind,
            status="failed",
            required=bool(required),
            observed={},
            expected_behavior=expected_behavior,
            repetitions=int(repetitions),
            confidence_level=float(confidence_level),
            stream_seed=stream,
            seed_role=seed_role,
            interval=None,
            reason=f"non-finite draws at positions {non_finite[:4]}",
        ), {"control_id": control_id, "status": "failed", "stream_seed": stream}
    interval = summarize_interval(
        draws, repetitions=int(repetitions), seed=stream, confidence_level=float(confidence_level)
    )
    declared = _string_tuple(metric_ids, name="metric_ids", minimum=1)
    observed_map = {name: float(value) for name, value in (observed or {}).items() if name in declared}
    observed_map = observed_map or dict(observed or {})
    status: ControlStatus = "recorded" if not bool(required) else "passed"
    outcome = ControlOutcome(
        control_id=control_id,
        kind=kind,
        status=status,
        required=bool(required),
        observed=observed_map,
        expected_behavior=expected_behavior,
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        stream_seed=stream,
        seed_role=seed_role,
        interval=interval,
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
    statistic: _StatisticFn,
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
    if kind not in (
        "null",
        "shuffled",
        "randomized",
        "cross_seed",
        "seed",
        "counterexample",
        "negative",
        "capacity",
        "bootstrap",
    ):
        raise ControlExecutionError(f"unsupported permutation control kind: {kind!r}")
    stream = derive_stream_seed(int(base_seed), control_id, role=seed_role)
    rng = np.random.default_rng(stream)
    try:
        observed_raw = statistic(rng)
    except ControlExecutionError as exc:
        return ControlOutcome(
            control_id=control_id,
            kind=kind,
            status="failed",
            required=bool(required),
            observed={},
            expected_behavior=expected_behavior,
            repetitions=2,
            confidence_level=0.95,
            stream_seed=stream,
            seed_role=seed_role,
            interval=None,
            reason=f"control statistic failed: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - execution failures block, never pass
        return ControlOutcome(
            control_id=control_id,
            kind=kind,
            status="failed",
            required=bool(required),
            observed={},
            expected_behavior=expected_behavior,
            repetitions=2,
            confidence_level=0.95,
            stream_seed=stream,
            seed_role=seed_role,
            interval=None,
            reason=f"control statistic failed: {exc}",
        )
    if not isinstance(observed_raw, Mapping):
        return ControlOutcome(
            control_id=control_id,
            kind=kind,
            status="failed",
            required=bool(required),
            observed={},
            expected_behavior=expected_behavior,
            repetitions=2,
            confidence_level=0.95,
            stream_seed=stream,
            seed_role=seed_role,
            interval=None,
            reason="control statistic must return a metric mapping",
        )
    observed: dict[str, float] = {}
    detail: dict[str, object] = {}
    for key, value in observed_raw.items():
        name = str(key)
        if isinstance(value, (Mapping, list, tuple)):
            detail[name] = dict(value) if isinstance(value, Mapping) else list(value)
            continue
        observed[name] = float(value)
    for metric_id, value in observed.items():
        if not isfinite(value):
            return ControlOutcome(
                control_id=control_id,
                kind=kind,
                status="failed",
                required=bool(required),
                observed={},
                expected_behavior=expected_behavior,
                repetitions=2,
                confidence_level=0.95,
                stream_seed=stream,
                seed_role=seed_role,
                interval=None,
                reason=f"non-finite control result for {metric_id!r}",
            )
    linked = [metric_id for metric_id in metric_ids if metric_id in observed]
    if comparator == "record":
        status: ControlStatus = "recorded" if not bool(required) else "passed"
        return ControlOutcome(
            control_id=control_id,
            kind=kind,
            status=status,
            required=bool(required),
            observed=observed,
            expected_behavior=expected_behavior,
            repetitions=2,
            confidence_level=0.95,
            stream_seed=stream,
            seed_role=seed_role,
            interval=None,
            detail=detail or None,
            reason="recorded control; not threshold-gated",
        )
    if threshold_value is None or not linked:
        return ControlOutcome(
            control_id=control_id,
            kind=kind,
            status="failed",
            required=bool(required),
            observed=observed,
            expected_behavior=expected_behavior,
            repetitions=2,
            confidence_level=0.95,
            stream_seed=stream,
            seed_role=seed_role,
            interval=None,
            detail=detail or None,
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
        control_id=control_id,
        kind=kind,
        status="passed" if gated else "failed",
        required=bool(required),
        observed=observed,
        expected_behavior=expected_behavior,
        repetitions=2,
        confidence_level=0.95,
        stream_seed=stream,
        seed_role=seed_role,
        interval=None,
        detail=detail or None,
        reason=f"control comparator {comparator} against threshold {threshold}",
    )


def execute_plan(
    plan: ControlPlan,
    supplied: Mapping[str, Mapping[str, float] | None],
    *,
    statistics: Mapping[str, _StatisticFn | None] | None = None,
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
    callbacks = dict(statistics or {})
    summaries = dict(intervals or {})
    specs = _require_specs(plan.controls)
    control_seed = _non_negative_int(plan.control_seed, name="control_seed")
    repetitions = _require_repetitions(plan.repetitions)
    confidence = _require_confidence(plan.confidence_level)
    ordered = sorted(specs, key=lambda spec: spec.control_id)
    outcomes: dict[str, ControlOutcome] = {}
    for spec in ordered:
        present = spec.control_id in supplied and supplied[spec.control_id] is not None
        if spec.control_id in callbacks:
            statistic = _require_statistic(callbacks[spec.control_id], name=spec.control_id)
            outcomes[spec.control_id] = run_permutation_control(
                control_id=spec.control_id,
                base_seed=_non_negative_int(plan.control_seed, name="control_seed"),
                seed_role="control",
                required=_require_bool(spec.required, name="required"),
                kind=spec.kind,
                metric_ids=list(_require_metric_ids(spec.metric_ids, control_id=spec.control_id)),
                expected_behavior=_require_text(spec.expected_behavior, name="expected_behavior"),
                statistic=statistic,
                comparator=spec.comparator,
                threshold_value=spec.threshold_value,
            )
            continue
        if spec.kind == "bootstrap" and spec.control_id in summaries:
            summary = summaries[spec.control_id]
            outcomes[spec.control_id] = ControlOutcome(
                control_id=spec.control_id,
                kind=spec.kind,
                status="passed" if spec.required else "recorded",
                required=spec.required,
                observed=dict(supplied.get(spec.control_id) or {}),
                expected_behavior=spec.expected_behavior,
                repetitions=repetitions,
                confidence_level=confidence,
                stream_seed=derive_stream_seed(control_seed, spec.control_id, role="control"),
                seed_role="control",
                interval=dict(summary),
                reason="bootstrap interval supplied from the central repetition schedule",
            )
            continue
        if not present:
            outcomes[spec.control_id] = ControlOutcome(
                control_id=spec.control_id,
                kind=spec.kind,
                status="failed",
                required=spec.required,
                observed={},
                expected_behavior=spec.expected_behavior,
                repetitions=repetitions,
                confidence_level=confidence,
                stream_seed=derive_stream_seed(control_seed, spec.control_id, role="control"),
                seed_role="control",
                interval=None,
                reason="required control is missing data"
                if spec.required
                else "optional control has no data; recorded as failed-optional",
            )
            if not spec.required:
                outcomes[spec.control_id] = ControlOutcome(
                    control_id=spec.control_id,
                    kind=spec.kind,
                    status="recorded",
                    required=False,
                    observed={},
                    expected_behavior=spec.expected_behavior,
                    repetitions=repetitions,
                    confidence_level=confidence,
                    stream_seed=derive_stream_seed(control_seed, spec.control_id, role="control"),
                    seed_role="control",
                    interval=None,
                    reason="optional control has no data; recorded without gating",
                )
            continue
        observed_raw = _require_supplied_observed(supplied[spec.control_id], name=spec.control_id)
        observed = {str(key): float(value) for key, value in observed_raw.items()}
        non_finite = [metric_id for metric_id, value in observed.items() if not isfinite(value)]
        if non_finite:
            outcomes[spec.control_id] = ControlOutcome(
                control_id=spec.control_id,
                kind=spec.kind,
                status="failed",
                required=spec.required,
                observed={},
                expected_behavior=spec.expected_behavior,
                repetitions=repetitions,
                confidence_level=confidence,
                stream_seed=derive_stream_seed(control_seed, spec.control_id, role="control"),
                seed_role="control",
                interval=None,
                reason=f"non-finite control result for {', '.join(sorted(non_finite))}",
            )
            continue
        if spec.comparator == "record":
            outcomes[spec.control_id] = ControlOutcome(
                control_id=spec.control_id,
                kind=spec.kind,
                status="passed" if spec.required else "recorded",
                required=spec.required,
                observed=observed,
                expected_behavior=spec.expected_behavior,
                repetitions=repetitions,
                confidence_level=confidence,
                stream_seed=derive_stream_seed(control_seed, spec.control_id, role="control"),
                seed_role="control",
                interval=None,
                reason="recorded control; not threshold-gated",
            )
            continue
        spec_metrics = _require_metric_ids(spec.metric_ids, control_id=spec.control_id)
        linked = [metric_id for metric_id in spec_metrics if metric_id in observed]
        if spec.threshold_value is None or not linked:
            outcomes[spec.control_id] = ControlOutcome(
                control_id=spec.control_id,
                kind=spec.kind,
                status="failed",
                required=spec.required,
                observed=observed,
                expected_behavior=spec.expected_behavior,
                repetitions=repetitions,
                confidence_level=confidence,
                stream_seed=derive_stream_seed(control_seed, spec.control_id, role="control"),
                seed_role="control",
                interval=None,
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
            control_id=spec.control_id,
            kind=spec.kind,
            status="passed" if gated else "failed",
            required=spec.required,
            observed=observed,
            expected_behavior=spec.expected_behavior,
            repetitions=repetitions,
            confidence_level=confidence,
            stream_seed=derive_stream_seed(control_seed, spec.control_id, role="control"),
            seed_role="control",
            interval=None,
            reason=f"control comparator {spec.comparator} against threshold {threshold}",
        )
    try:
        canonical_json({key: value.to_dict() for key, value in outcomes.items()})
    except PortableNodeError as exc:
        raise ControlExecutionError(f"control outcomes are not canonical JSON: {exc}") from exc
    return outcomes


def failed_required(outcomes: Mapping[str, ControlOutcome]) -> tuple[str, ...]:
    """Return sorted identities of required controls whose status is ``failed``."""
    return tuple(
        sorted(
            control_id for control_id, outcome in outcomes.items() if outcome.required and outcome.status == "failed"
        )
    )


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
