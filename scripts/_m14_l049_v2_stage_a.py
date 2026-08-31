"""Offline train-only candidate selection for the L04.9 v2 addendum."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from scripts._m14_l049_v2_attestation import build_runtime_attestation
from scripts._m14_l049_v2_schema import (
    BOOTSTRAP_REPLICATES,
    EXPECTED_RUNTIME_MODEL,
    FINALIZER_REJECTION_CODES,
    OOF_RECOVERY_THRESHOLD,
    PARENT_PLAN_SHA256,
    PUBLIC_TRAIN_SEED,
    TRAIN_GROUP_COUNT,
    V2_ADDENDUM_SCHEMA,
    V2_STAGE_A_SCHEMA,
    VALIDATION_REJECTION_CODES,
    candidate_grid,
    canonical_digest,
    canonical_fixture_bytes,
    canonical_json_bytes,
    digest_bytes,
    directional_recovery,
    top_level_cli_sha256,
)

ScoreValue = float | Mapping[str, Any]
ScoreFunction = Callable[[Mapping[str, Any], int, int], ScoreValue]

_CLEANUP_ERROR_TYPES = frozenset(
    {"CleanupError", "Exception", "MemoryError", "OSError", "RuntimeError", "TimeoutError", "TypeError", "ValueError"}
)
_FINALIZER_RESOURCE_FIELDS = frozenset(
    {
        "stage",
        "execution_attempted",
        "execution_backend",
        "model",
        "model_revision",
        "integration",
        "model_adapter",
        "device",
        "backend",
        "dtype",
        "hook",
        "intervention",
        "operation_counts",
        "cleanup",
        "resource_peak",
        "no_mutation",
    }
)
_COUNTER_FIELDS = frozenset(("candidate_evaluations", "hooks", "captures", "patches", "controls", "forwards"))
_MAX_RESOURCE_VALUE = 6_000_000_000
_FINALIZER_IDENTITY_VALUES: Mapping[str, object] = {
    "stage": "real_runtime",
    "execution_attempted": True,
    "execution_backend": "cuda",
    "model": EXPECTED_RUNTIME_MODEL,
    "model_revision": EXPECTED_RUNTIME_MODEL,
    "integration": "TransformerLMIntegration",
    "model_adapter": "N/A",
    "device": "cuda",
    "backend": "cuda",
    "dtype": "float32",
    "no_mutation": True,
}
_FINALIZER_HOOK_FIELDS = frozenset(("registered", "capture_calls", "removed"))
_FINALIZER_INTERVENTION_FIELDS = frozenset(("patch_calls", "control_calls", "forward_calls"))
_FINALIZER_PEAK_FIELDS = frozenset(
    (
        "peak_cpu_bytes",
        "peak_gpu_bytes",
        "peak_gpu_reserved_bytes",
        "unit",
        "budget_cpu_bytes",
        "budget_gpu_bytes",
        "measurement_status",
        "measurement_reason",
        "elapsed_seconds",
        "elapsed_source",
        "cpu_source",
        "gpu_source",
        "gpu_reserved_source",
        "gpu_device",
    )
)
_FINALIZER_MEASUREMENT_REASONS = frozenset(
    {
        "cuda_unavailable",
        "cuda_reset_failed",
        "cuda_peak_query_failed",
        "cuda_zero_peak",
        "rss_unavailable",
        "clock_invalid",
        "tracker_unstarted",
        "resource_measurement_invalid",
    }
)
_FINALIZER_CPU_SOURCES = frozenset(
    {
        "resource.ru_maxrss_linux_kib",
        "resource.ru_maxrss_macos_bytes",
        "psutil.Process.memory_info.rss",
        "unavailable",
    }
)


def _finite_positive_float_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) and converted > 0 else None


def _bounded_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    try:
        converted = int(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if 0 <= converted <= _MAX_RESOURCE_VALUE else None


def attempted_real_resources() -> dict[str, Any]:
    """Return a complete attempted-CUDA envelope when setup fails before tracking."""
    zero_counts = dict.fromkeys(_COUNTER_FIELDS, 0)
    return {
        "stage": "real_runtime",
        "execution_attempted": True,
        "execution_backend": "cuda",
        "model": EXPECTED_RUNTIME_MODEL,
        "model_revision": EXPECTED_RUNTIME_MODEL,
        "integration": "TransformerLMIntegration",
        "model_adapter": "N/A",
        "device": "cuda",
        "backend": "cuda",
        "dtype": "float32",
        "hook": {"registered": 0, "capture_calls": 0, "removed": 0},
        "intervention": {"patch_calls": 0, "control_calls": 0, "forward_calls": 0},
        "operation_counts": zero_counts,
        "cleanup": {"hook_count": 0, "completed": True},
        "resource_peak": {
            "peak_cpu_bytes": 0,
            "peak_gpu_bytes": 0,
            "peak_gpu_reserved_bytes": 0,
            "unit": "bytes",
            "budget_cpu_bytes": 6_000_000_000,
            "budget_gpu_bytes": 6_000_000_000,
            "measurement_status": "unavailable",
            "measurement_reason": "tracker_unstarted",
            "elapsed_seconds": None,
            "elapsed_source": "time.perf_counter",
            "cpu_source": "unavailable",
            "gpu_source": "unavailable",
            "gpu_reserved_source": "unavailable",
            "gpu_device": "unavailable",
        },
        "no_mutation": True,
    }


def synthetic_resources() -> dict[str, Any]:
    """Return the deliberately minimal non-runtime resource envelope."""
    return {
        "stage": "protocol_fixture",
        "execution_backend": "cpu",
        "execution_attempted": False,
        "no_mutation": True,
    }


def normalize_attempted_real_resources(resources: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project an untrusted runtime envelope onto the canonical resource schema."""
    baseline = attempted_real_resources()
    if not isinstance(resources, Mapping):
        return baseline
    counters = resources.get("operation_counts")
    counter_values = (
        {key: _bounded_nonnegative_int(counters.get(key)) for key in _COUNTER_FIELDS}
        if isinstance(counters, Mapping) and set(counters) == set(_COUNTER_FIELDS)
        else {}
    )
    has_valid_counters = bool(counter_values) and all(value is not None for value in counter_values.values())
    if has_valid_counters:
        copied = {key: cast(int, counter_values[key]) for key in _COUNTER_FIELDS}
        baseline["operation_counts"] = copied
        baseline["hook"] = {
            "registered": copied["hooks"],
            "capture_calls": copied["captures"],
            "removed": copied["hooks"],
        }
        baseline["intervention"] = {
            "patch_calls": copied["patches"],
            "control_calls": copied["controls"],
            "forward_calls": copied["forwards"],
        }
    hook = resources.get("hook")
    if isinstance(hook, Mapping):
        registered = hook.get("registered")
        capture_calls = hook.get("capture_calls")
        removed = hook.get("removed")
        registered_value = _bounded_nonnegative_int(registered)
        capture_value = _bounded_nonnegative_int(capture_calls)
        removed_value = _bounded_nonnegative_int(removed)
        if (
            registered_value is not None
            and capture_value is not None
            and removed_value is not None
            and removed_value <= registered_value
        ):
            baseline["hook"]["registered"] = registered_value
            baseline["hook"]["capture_calls"] = capture_value
            baseline["hook"]["removed"] = removed_value
            baseline["operation_counts"]["hooks"] = registered_value
    projected_peak: dict[str, Any] | None = None
    peak = resources.get("resource_peak")
    if isinstance(peak, Mapping):
        reasons = {
            "cuda_unavailable",
            "cuda_reset_failed",
            "cuda_peak_query_failed",
            "cuda_zero_peak",
            "rss_unavailable",
            "clock_invalid",
            "tracker_unstarted",
            "resource_measurement_invalid",
        }
        int_fields = (
            "peak_cpu_bytes",
            "peak_gpu_bytes",
            "peak_gpu_reserved_bytes",
            "budget_cpu_bytes",
            "budget_gpu_bytes",
        )
        scalar_ints = {key: _bounded_nonnegative_int(peak.get(key)) for key in int_fields}
        basic_peak = peak.get("unit") == "bytes" and all(value is not None for value in scalar_ints.values())
        status = peak.get("measurement_status")
        reason = peak.get("measurement_reason")
        if basic_peak and status == "available" and reason is None:
            elapsed = _finite_positive_float_value(peak.get("elapsed_seconds"))
            peak_cpu = cast(int, scalar_ints["peak_cpu_bytes"])
            peak_gpu = cast(int, scalar_ints["peak_gpu_bytes"])
            peak_reserved = cast(int, scalar_ints["peak_gpu_reserved_bytes"])
            budget_cpu = cast(int, scalar_ints["budget_cpu_bytes"])
            budget_gpu = cast(int, scalar_ints["budget_gpu_bytes"])
            source_ok = (
                peak.get("elapsed_source") == "time.perf_counter"
                and elapsed is not None
                and peak.get("cpu_source")
                in {"resource.ru_maxrss_linux_kib", "resource.ru_maxrss_macos_bytes", "psutil.Process.memory_info.rss"}
                and peak.get("gpu_source") == "torch.cuda.max_memory_allocated"
                and peak.get("gpu_reserved_source") == "torch.cuda.max_memory_reserved"
                and isinstance(peak.get("gpu_device"), str)
                and peak["gpu_device"].startswith("cuda:")
                and peak["gpu_device"][5:].isdigit()
                and peak_cpu > 0
                and peak_gpu > 0
                and peak_reserved > 0
                and peak_cpu <= budget_cpu
                and peak_gpu <= budget_gpu
                and peak_reserved <= budget_gpu
                and peak_reserved >= peak_gpu
            )
            if source_ok:
                projected_peak = {
                    **{key: cast(int, scalar_ints[key]) for key in int_fields},
                    "unit": "bytes",
                    "measurement_status": "available",
                    "measurement_reason": None,
                    "elapsed_seconds": elapsed,
                    "elapsed_source": "time.perf_counter",
                    "cpu_source": str(peak["cpu_source"]),
                    "gpu_source": "torch.cuda.max_memory_allocated",
                    "gpu_reserved_source": "torch.cuda.max_memory_reserved",
                    "gpu_device": f"cuda:{str(peak['gpu_device'])[5:]}",
                }
        elif basic_peak and status == "unavailable" and reason in reasons:
            projected_peak = dict(baseline["resource_peak"])
            projected_peak["measurement_reason"] = reason
            projected_peak["elapsed_seconds"] = None
            projected_peak["elapsed_source"] = "time.perf_counter"
            if reason in {"cuda_unavailable", "cuda_reset_failed", "cuda_peak_query_failed", "cuda_zero_peak"}:
                projected_peak["gpu_device"] = "unavailable"
                projected_peak["gpu_source"] = "unavailable"
                projected_peak["gpu_reserved_source"] = "unavailable"
            if reason == "rss_unavailable":
                projected_peak["cpu_source"] = "unavailable"
        elif basic_peak:
            projected_peak = dict(baseline["resource_peak"])
            projected_peak["measurement_reason"] = "resource_measurement_invalid"
        else:
            projected_peak = dict(baseline["resource_peak"])
            projected_peak["measurement_reason"] = "resource_measurement_invalid"
    if projected_peak is not None:
        baseline["resource_peak"] = projected_peak
    cleanup = resources.get("cleanup")
    if isinstance(cleanup, Mapping):
        if cleanup == {"hook_count": 0, "completed": True}:
            baseline["cleanup"] = {"hook_count": 0, "completed": True}
        elif (
            cleanup.get("attempted") is True
            and cleanup.get("completed") is False
            and cleanup.get("error_type") in _CLEANUP_ERROR_TYPES
            and cleanup.get("reason") in {"finalizer_exception", "finalizer_invalid_result"}
            and cleanup.get("stage") == "cleanup"
        ):
            registered = int(baseline["hook"]["registered"])
            removed = int(baseline["hook"]["removed"])
            baseline["cleanup"] = {
                "attempted": True,
                "completed": False,
                "hooks_remaining": registered - removed,
                "error_type": cleanup["error_type"],
                "reason": cleanup["reason"],
                "stage": "cleanup",
            }
    return baseline


def _finalizer_rejection_code(value: object) -> str | None:
    """Validate and classify a finalizer result without exposing values or keys.

    This is the single producer-independent checker used at the finalizer
    boundary.  The public artifact validator intentionally remains a separate
    consumer-side check; keeping this function as the only producer-side
    predicate prevents acceptance and diagnostic paths from drifting apart.
    """
    if not isinstance(value, Mapping):
        return "finalizer_not_mapping"
    missing = _FINALIZER_RESOURCE_FIELDS - set(value)
    extra = set(value) - _FINALIZER_RESOURCE_FIELDS
    if missing and extra:
        return "finalizer_top_level_fields"
    if missing:
        return "finalizer_top_level_missing_fields"
    if extra:
        return "finalizer_top_level_extra_fields"

    for field, expected in _FINALIZER_IDENTITY_VALUES.items():
        actual = value.get(field)
        valid = actual is expected if isinstance(expected, bool) else isinstance(actual, str) and actual == expected
        if not valid:
            return "finalizer_identity_fields"

    counters = value.get("operation_counts")
    if (
        not isinstance(counters, Mapping)
        or set(counters) != _COUNTER_FIELDS
        or any(_bounded_nonnegative_int(item) is None for item in counters.values())
    ):
        return "finalizer_operation_counts"

    hook = value.get("hook")
    if (
        not isinstance(hook, Mapping)
        or set(hook) != _FINALIZER_HOOK_FIELDS
        or any(_bounded_nonnegative_int(hook.get(field)) is None for field in _FINALIZER_HOOK_FIELDS)
    ):
        return "finalizer_hook_fields"
    if (
        hook["registered"] != counters["hooks"]
        or hook["capture_calls"] != counters["captures"]
        or hook["removed"] != hook["registered"]
    ):
        return "finalizer_cross_field_invariants"

    intervention = value.get("intervention")
    if (
        not isinstance(intervention, Mapping)
        or set(intervention) != _FINALIZER_INTERVENTION_FIELDS
        or any(_bounded_nonnegative_int(intervention.get(field)) is None for field in _FINALIZER_INTERVENTION_FIELDS)
    ):
        return "finalizer_intervention_fields"
    if (
        intervention["patch_calls"] != counters["patches"]
        or intervention["control_calls"] != counters["controls"]
        or intervention["forward_calls"] != counters["forwards"]
    ):
        return "finalizer_cross_field_invariants"

    cleanup = value.get("cleanup")
    if (
        not isinstance(cleanup, Mapping)
        or set(cleanup) != {"hook_count", "completed"}
        or cleanup.get("hook_count") != 0
        or cleanup.get("completed") is not True
    ):
        return "finalizer_cleanup_fields"

    peak = value.get("resource_peak")
    if not isinstance(peak, Mapping) or set(peak) != _FINALIZER_PEAK_FIELDS:
        return "finalizer_resource_peak_fields"
    integer_fields = (
        "peak_cpu_bytes",
        "peak_gpu_bytes",
        "peak_gpu_reserved_bytes",
        "budget_cpu_bytes",
        "budget_gpu_bytes",
    )
    if (
        peak.get("unit") != "bytes"
        or any(_bounded_nonnegative_int(peak.get(field)) is None for field in integer_fields)
        or peak.get("measurement_status") not in {"available", "unavailable"}
        or (
            peak.get("measurement_reason") is not None
            and (
                not isinstance(peak.get("measurement_reason"), str)
                or peak.get("measurement_reason") not in _FINALIZER_MEASUREMENT_REASONS
            )
        )
        or peak.get("elapsed_seconds") is not None
        and _finite_nonnegative_float_value(peak.get("elapsed_seconds")) is None
        or peak.get("elapsed_source") != "time.perf_counter"
        or peak.get("cpu_source") not in _FINALIZER_CPU_SOURCES
        or peak.get("gpu_source") not in {"torch.cuda.max_memory_allocated", "unavailable"}
        or peak.get("gpu_reserved_source") not in {"torch.cuda.max_memory_reserved", "unavailable"}
        or not isinstance(peak.get("gpu_device"), str)
    ):
        return "finalizer_resource_peak_fields"

    status = peak["measurement_status"]
    reason = peak["measurement_reason"]
    if status == "available":
        if (
            reason is not None
            or _finite_positive_float_value(peak.get("elapsed_seconds")) is None
            or peak["peak_cpu_bytes"] <= 0
            or peak["peak_gpu_bytes"] <= 0
            or peak["peak_gpu_reserved_bytes"] <= 0
            or peak["cpu_source"] == "unavailable"
            or peak["gpu_source"] == "unavailable"
            or peak["gpu_reserved_source"] == "unavailable"
            or not peak["gpu_device"].startswith("cuda:")
            or not peak["gpu_device"][5:].isdigit()
        ):
            return "finalizer_resource_peak_fields"
    else:
        if reason is None:
            return "finalizer_resource_peak_fields"
        cpu_source = peak["cpu_source"]
        gpu_source = peak["gpu_source"]
        reserved_source = peak["gpu_reserved_source"]
        source_value_valid = (
            (
                cpu_source == "unavailable"
                and peak["peak_cpu_bytes"] == 0
                or cpu_source != "unavailable"
                and peak["peak_cpu_bytes"] > 0
            )
            and (
                gpu_source == "unavailable"
                and peak["peak_gpu_bytes"] == 0
                or gpu_source != "unavailable"
                and peak["peak_gpu_bytes"] > 0
            )
            and (
                reserved_source == "unavailable"
                and peak["peak_gpu_reserved_bytes"] == 0
                or reserved_source != "unavailable"
                and peak["peak_gpu_reserved_bytes"] > 0
            )
        )
        cuda_reason = reason in {
            "cuda_unavailable",
            "cuda_reset_failed",
            "cuda_peak_query_failed",
            "cuda_zero_peak",
        }
        if (
            not source_value_valid
            or (cuda_reason and (peak["gpu_source"] != "unavailable" or peak["gpu_reserved_source"] != "unavailable"))
            or (reason == "rss_unavailable" and peak["cpu_source"] != "unavailable")
            or (
                reason in {"tracker_unstarted", "resource_measurement_invalid"}
                and (
                    peak["cpu_source"] != "unavailable"
                    or peak["gpu_source"] != "unavailable"
                    or peak["gpu_reserved_source"] != "unavailable"
                )
            )
            or (cuda_reason and peak["gpu_device"] != "unavailable" and not peak["gpu_device"].startswith("cuda:"))
        ):
            return "finalizer_resource_peak_fields"

    if (
        peak["peak_cpu_bytes"] > peak["budget_cpu_bytes"]
        or peak["peak_gpu_bytes"] > peak["budget_gpu_bytes"]
        or peak["peak_gpu_reserved_bytes"] > peak["budget_gpu_bytes"]
        or peak["peak_gpu_reserved_bytes"] < peak["peak_gpu_bytes"]
    ):
        return "finalizer_cross_field_invariants"
    return None


def _finite_nonnegative_float_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) and converted >= 0 else None


def _valid_finalizer_resources(value: object) -> bool:
    """Return whether the canonical producer-side finalizer check succeeds."""
    return _finalizer_rejection_code(value) is None


class _ResourceFinalizerError(RuntimeError):
    """Internal boundary preserving a finalizer failure separately."""

    def __init__(
        self,
        cleanup_error: BaseException,
        *,
        reason: str,
        finalizer_rejection_code: str | None = None,
    ) -> None:
        self.cleanup_error = cleanup_error
        self.reason = reason
        self.finalizer_rejection_code = finalizer_rejection_code
        super().__init__("runtime resource finalizer failed")


def _candidate_key(candidate: Mapping[str, int]) -> tuple[int, int]:
    return int(candidate["layer"]), (0, -1, -2).index(int(candidate["offset"]))


def outer_folds(group_ids: Sequence[str]) -> list[list[str]]:
    groups = sorted(str(group) for group in group_ids)
    if len(groups) != 36 or len(set(groups)) != 36:
        raise ValueError("stage A requires exactly 36 unique train groups")
    return [groups[index : index + 6] for index in range(0, 36, 6)]


def default_train_score(row: Mapping[str, Any], layer: int, offset: int) -> float:
    """Deterministic synthetic sensitivity score; no model or holdout data."""
    token = f"{row['row_id']}|{int(layer)}|{int(offset)}".encode()
    noise = (int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64 - 0.5) * 0.02
    signal = 0.16 if (int(layer), int(offset)) == (6, 0) else 0.0
    return float(signal + noise)


def _lower_ci(values: Sequence[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite and non-empty")
    rng = np.random.default_rng(int(seed))
    draws = array[rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))]
    return float(np.quantile(np.mean(draws, axis=1), 0.025))


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("stage A accepts train rows only")
        groups.setdefault(str(row["group_id"]), []).append(row)
    if any(len(value) != 2 for value in groups.values()):
        raise ValueError("stage A requires two rows per train group")
    return dict(sorted(groups.items()))


def _score_value(value: ScoreValue) -> tuple[float, dict[str, float]]:
    """Normalize a scorer value and retain primitive margins for validation."""
    if isinstance(value, Mapping):
        clean = value.get("clean_margin")
        corrupted = value.get("corrupted_margin")
        patched = value.get("patched_margin")
        recovery = directional_recovery(clean, corrupted, patched)
        if recovery is None:
            raise ValueError("scorer produced an invalid directional recovery")
        return recovery, {
            "clean_margin": float(clean),  # type: ignore[arg-type]
            "corrupted_margin": float(corrupted),  # type: ignore[arg-type]
            "patched_margin": float(patched),  # type: ignore[arg-type]
        }
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("scorer produced a non-finite recovery")
    # Offline/injected scorers already return a recovery scalar.  Bind it to
    # deterministic primitive endpoints so the validator can recompute rather
    # than trusting the declared group score.
    return numeric, {"clean_margin": 1.0, "corrupted_margin": 0.0, "patched_margin": numeric}


def _score_records(rows: Sequence[Mapping[str, Any]], scorer: ScoreFunction) -> list[dict[str, Any]]:
    groups = _group_rows(rows)
    records: list[dict[str, Any]] = []
    for group_id, group_rows in groups.items():
        for candidate in candidate_grid():
            normalized = [_score_value(scorer(row, candidate["layer"], candidate["offset"])) for row in group_rows]
            values = [value for value, _primitive in normalized]
            if not np.isfinite(values).all():
                raise ValueError("train candidate scores must be finite")
            primitives = [item for _value, item in normalized]
            records.append(
                {
                    "group_id": group_id,
                    "layer": candidate["layer"],
                    "offset": candidate["offset"],
                    "row_scores": values,
                    "group_score": float(np.mean(values)),
                    "primitive_margins": primitives,
                }
            )
    return records


def _rank(records: Sequence[Mapping[str, Any]], groups: Sequence[str]) -> list[dict[str, Any]]:
    selected = [record for record in records if str(record["group_id"]) in set(groups)]
    ranked: list[dict[str, Any]] = []
    for candidate in candidate_grid():
        values = [
            float(record["group_score"])
            for record in selected
            if record["layer"] == candidate["layer"] and record["offset"] == candidate["offset"]
        ]
        if len(values) != len(groups):
            raise ValueError("candidate fold scores are incomplete")
        ranked.append(
            {
                "layer": candidate["layer"],
                "offset": candidate["offset"],
                "mean_recovery": float(np.mean(values)),
                "lower_ci": _lower_ci(values, seed=PUBLIC_TRAIN_SEED + len(groups) + candidate["layer"] * 3),
            }
        )
    return sorted(ranked, key=lambda item: (-item["mean_recovery"], -item["lower_ci"], _candidate_key(item)))


def select_stage_a(rows: Sequence[Mapping[str, Any]], scorer: ScoreFunction = default_train_score) -> dict[str, Any]:
    """Select a candidate using six outer folds and train groups only."""
    groups = _group_rows(rows)
    folds = outer_folds(list(groups))
    records = _score_records(rows, scorer)
    fold_records: list[dict[str, Any]] = []
    wins: dict[tuple[int, int], int] = {}
    oof: list[dict[str, Any]] = []
    for fold_index, validation_groups in enumerate(folds):
        fit_groups = [group for group in groups if group not in validation_groups]
        ranking = _rank(records, fit_groups)
        winner = {"layer": int(ranking[0]["layer"]), "offset": int(ranking[0]["offset"])}
        key = (winner["layer"], winner["offset"])
        wins[key] = wins.get(key, 0) + 1
        for group in validation_groups:
            match = next(
                record
                for record in records
                if record["group_id"] == group
                and record["layer"] == winner["layer"]
                and record["offset"] == winner["offset"]
            )
            oof.append(
                {
                    "fold": fold_index,
                    "group_id": group,
                    "layer": winner["layer"],
                    "offset": winner["offset"],
                    "recovery": float(match["group_score"]),
                }
            )
        fold_records.append(
            {
                "fold": fold_index,
                "fit_groups": fit_groups,
                "validation_groups": validation_groups,
                "ranking": ranking,
                "winner": winner,
            }
        )
    consensus = max(
        wins.items(), key=lambda item: (-item[1], _candidate_key({"layer": item[0][0], "offset": item[0][1]}))
    )
    oof_values = [float(item["recovery"]) for item in oof]
    lower = _lower_ci(oof_values, seed=PUBLIC_TRAIN_SEED)
    positive = sum(value > 0.0 for value in oof_values)
    passed = bool(
        consensus[1] >= 4
        and lower > OOF_RECOVERY_THRESHOLD
        and all(float(np.mean([item["recovery"] for item in oof if item["fold"] == fold])) > 0.0 for fold in range(6))
        and positive >= 24
    )
    return {
        "candidate_grid": candidate_grid(),
        "score_records": records,
        "folds": fold_records,
        "consensus_candidate": ({"layer": consensus[0][0], "offset": consensus[0][1]} if consensus[1] >= 4 else None),
        "consensus_wins": int(consensus[1]),
        "oof_evidence": oof,
        "oof_metric": {
            "point_estimate": float(np.mean(oof_values)),
            "lower_ci_95": lower,
            "threshold": OOF_RECOVERY_THRESHOLD,
            "all_fold_means_positive": all(
                float(np.mean([item["recovery"] for item in oof if item["fold"] == fold])) > 0.0 for fold in range(6)
            ),
            "positive_groups": positive,
            "required_positive_groups": 24,
            "pass": passed,
        },
        "train_group_ids": list(groups),
    }


def build_stage_a_artifact(
    rows: Sequence[Mapping[str, Any]],
    addendum: Mapping[str, Any],
    *,
    source_sha256: str,
    scorer: ScoreFunction = default_train_score,
    resources: Mapping[str, Any] | None = None,
    execution_mode: str = "synthetic",
    cli_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a train-only artifact for offline or injected real execution."""
    if execution_mode not in {"synthetic", "real"}:
        raise ValueError("Stage A execution mode is invalid")
    selection = select_stage_a(rows, scorer)
    source_resources = resources if isinstance(resources, Mapping) else {}
    finalizer = source_resources.get("finalize")
    resource_payload = (
        normalize_attempted_real_resources(source_resources) if execution_mode == "real" else synthetic_resources()
    )
    if callable(finalizer):
        try:
            finalized = finalizer()
        except Exception as error:  # noqa: BLE001 - promote as a D0 cleanup failure
            raise _ResourceFinalizerError(error, reason="finalizer_exception") from error
        if not _valid_finalizer_resources(finalized):
            error = TypeError("runtime resource finalizer returned an invalid mapping")
            raise _ResourceFinalizerError(
                error,
                reason="finalizer_invalid_result",
                finalizer_rejection_code=_finalizer_rejection_code(finalized),
            ) from error
        resource_payload = normalize_attempted_real_resources(cast(Mapping[str, Any], finalized))
    raw = canonical_fixture_bytes(rows)
    artifact: dict[str, Any] = {
        "schema_version": V2_STAGE_A_SCHEMA,
        "stage": "stage_a_train_selection",
        "status": (
            "stage_a_complete"
            if execution_mode == "real" and selection["oof_metric"]["pass"]
            else "protocol_fixture"
            if execution_mode == "synthetic" and selection["oof_metric"]["pass"]
            else "stage_a_failed"
        ),
        "evidence_level": "D1" if execution_mode == "real" and selection["oof_metric"]["pass"] else "D0",
        "evidence_eligible": bool(execution_mode == "real" and selection["oof_metric"]["pass"]),
        "repository_promotion": False,
        "failure_kind": "semantic_gate" if not selection["oof_metric"]["pass"] else None,
        "selection_complete": True,
        "parent_plan_sha256": PARENT_PLAN_SHA256,
        "addendum_schema": V2_ADDENDUM_SCHEMA,
        "addendum_sha256": canonical_digest(addendum, "addendum_sha256"),
        "source_sha256": str(source_sha256),
        "public_train_seed": PUBLIC_TRAIN_SEED,
        "train_fixture_sha256": digest_bytes(raw),
        "holdout_commitment": dict(addendum["fixture"]),
        "selection": selection,
        "resources": resource_payload,
    }
    selection_sha = digest_bytes(canonical_json_bytes(selection))
    resolved_cli_sha = cli_sha256 or top_level_cli_sha256("stage_a_train_selection")
    if resolved_cli_sha is None:
        raise ValueError("Stage A top-level CLI digest is unavailable")
    attestation = build_runtime_attestation(
        stage="stage_a_train_selection",
        mode=execution_mode,
        group_count=TRAIN_GROUP_COUNT,
        pair_count=TRAIN_GROUP_COUNT,
        candidate_count=len(candidate_grid()),
        seed_count=1,
        fixture_sha256=artifact["train_fixture_sha256"],
        candidate_sha256=selection_sha,
        source_sha256=str(source_sha256),
        addendum_sha256=artifact["addendum_sha256"],
        cli_sha256=resolved_cli_sha,
        resources=artifact["resources"],
        operation_counts=(
            artifact["resources"].get("operation_counts")
            if execution_mode == "real" and isinstance(artifact["resources"], Mapping)
            else None
        ),
    )
    artifact["runtime_attestation"] = attestation
    artifact["attestation_sha256"] = attestation["attestation_sha256"]
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    return artifact


def _safe_runtime_error(error: BaseException) -> dict[str, Any]:
    """Return failure metadata without serializing prompts or tensor payloads."""
    details: dict[str, Any] = {"exception_type": type(error).__name__}
    field = getattr(error, "field", None)
    expected = getattr(error, "expected_shape", None)
    actual = getattr(error, "actual_shape", None)
    if isinstance(field, str) and isinstance(expected, tuple) and isinstance(actual, tuple):
        details["shape_field"] = field
        details["expected_shape"] = [int(value) for value in expected]
        details["actual_shape"] = [int(value) for value in actual]
    return details


def _cleanup_error_type(error: BaseException) -> str:
    name = type(error).__name__
    return name if name in _CLEANUP_ERROR_TYPES else "CleanupError"


def _cleanup_failure(
    error: BaseException,
    prior: Mapping[str, Any],
    *,
    reason: str,
    finalizer_rejection_code: str | None = None,
) -> dict[str, Any]:
    hook = prior.get("hook")
    registered = hook.get("registered") if isinstance(hook, Mapping) else 0
    removed = hook.get("removed") if isinstance(hook, Mapping) else 0
    if not isinstance(registered, int) or isinstance(registered, bool) or registered < 0:
        registered = 0
    if not isinstance(removed, int) or isinstance(removed, bool) or removed < 0:
        removed = 0
    hooks_remaining = max(registered - removed, 0)
    cleanup: dict[str, Any] = {
        "attempted": True,
        "completed": False,
        "hooks_remaining": hooks_remaining,
        "error_type": _cleanup_error_type(error),
        "reason": reason,
        "stage": "cleanup",
    }
    if reason == "finalizer_invalid_result":
        cleanup["finalizer_rejection_code"] = (
            finalizer_rejection_code
            if finalizer_rejection_code in FINALIZER_REJECTION_CODES
            else "finalizer_top_level_fields"
        )
    return cleanup


def _failure_resources(
    resources: Mapping[str, Any] | None,
    *,
    cleanup_error: BaseException | None = None,
    cleanup_reason: str = "finalizer_exception",
    finalizer_rejection_code: str | None = None,
) -> dict[str, Any]:
    source: Mapping[str, Any] = resources if isinstance(resources, Mapping) else {}
    finalizer = source.get("finalize")
    payload = (
        normalize_attempted_real_resources(source)
        if source.get("execution_backend") == "cuda"
        else synthetic_resources()
    )
    if cleanup_error is None and callable(finalizer):
        try:
            finalized = finalizer()
        except Exception as error:  # noqa: BLE001 - preserve primary runtime failure
            cleanup_error = error
        else:
            if _valid_finalizer_resources(finalized):
                payload = normalize_attempted_real_resources(cast(Mapping[str, Any], finalized))
            else:
                cleanup_error = TypeError("runtime resource finalizer returned an invalid mapping")
                cleanup_reason = "finalizer_invalid_result"
                finalizer_rejection_code = _finalizer_rejection_code(finalized)
    attempted = payload.get("execution_attempted") is True
    if attempted:
        payload["stage"] = "cleanup"
        if cleanup_error is not None:
            payload["cleanup"] = _cleanup_failure(
                cleanup_error,
                payload,
                reason=cleanup_reason,
                finalizer_rejection_code=finalizer_rejection_code,
            )
        else:
            cleanup = payload.get("cleanup")
            if isinstance(cleanup, Mapping):
                cleanup_payload = dict(cleanup)
                cleanup_payload.setdefault("hook_count", 0)
                cleanup_payload.setdefault("completed", True)
                payload["cleanup"] = cleanup_payload
    else:
        payload.setdefault("stage", "preflight")
        payload.setdefault("execution_attempted", False)
        payload.setdefault("execution_backend", "none")
        payload.setdefault("device", "not used")
        payload.setdefault("no_mutation", True)
    return payload


def build_stage_a_failure_artifact(
    rows: Sequence[Mapping[str, Any]],
    addendum: Mapping[str, Any],
    *,
    source_sha256: str,
    error: BaseException,
    resources: Mapping[str, Any] | None = None,
    cleanup_error: BaseException | None = None,
    cleanup_reason: str = "finalizer_exception",
    finalizer_rejection_code: str | None = None,
    cli_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a truthful non-promoting artifact after an incomplete real run."""
    resource_payload = _failure_resources(
        resources,
        cleanup_error=cleanup_error,
        cleanup_reason=cleanup_reason,
        finalizer_rejection_code=finalizer_rejection_code,
    )
    attempted = resource_payload.get("execution_backend") == "cuda"
    resource_payload.setdefault("model", EXPECTED_RUNTIME_MODEL)
    resource_payload.setdefault("model_revision", EXPECTED_RUNTIME_MODEL)
    resource_payload.setdefault("integration", "TransformerLMIntegration")
    resource_payload.setdefault("model_adapter", "N/A")
    resource_payload.setdefault("backend", "cuda" if attempted else "none")
    resource_payload.setdefault("dtype", "float32")
    resource_payload.setdefault("resource_peak", {"peak_cpu_bytes": 0, "peak_gpu_bytes": 0, "unit": "bytes"})
    resource_payload.setdefault("hook", {"registered": 0, "capture_calls": 0, "removed": 0})
    resource_payload.setdefault("intervention", {"patch_calls": 0, "control_calls": 0, "forward_calls": 0})
    resource_payload.setdefault(
        "operation_counts",
        dict.fromkeys(("candidate_evaluations", "hooks", "captures", "patches", "controls", "forwards"), 0),
    )
    resource_payload.setdefault("cleanup", {"hook_count": 0, "completed": True})
    resource_payload.setdefault("no_mutation", True)
    raw = canonical_fixture_bytes(rows)
    selection: dict[str, Any] = {
        "candidate_grid": candidate_grid(),
        "score_records": [],
        "folds": [],
        "consensus_candidate": None,
        "consensus_wins": 0,
        "oof_evidence": [],
        "oof_metric": None,
        "train_group_ids": sorted({str(row["group_id"]) for row in rows}),
        "failure": _safe_runtime_error(error),
    }
    selection_sha = digest_bytes(canonical_json_bytes(selection))
    resolved_cli_sha = cli_sha256 or top_level_cli_sha256("stage_a_train_selection")
    if resolved_cli_sha is None:
        raise ValueError("Stage A top-level CLI digest is unavailable")
    mode = "real" if attempted else "synthetic"
    attestation = build_runtime_attestation(
        stage="stage_a_train_selection",
        mode=mode,
        group_count=TRAIN_GROUP_COUNT,
        pair_count=TRAIN_GROUP_COUNT,
        candidate_count=len(candidate_grid()),
        seed_count=1,
        fixture_sha256=digest_bytes(raw),
        candidate_sha256=selection_sha,
        source_sha256=str(source_sha256),
        addendum_sha256=canonical_digest(addendum, "addendum_sha256"),
        cli_sha256=resolved_cli_sha,
        resources=resource_payload,
        operation_counts=resource_payload["operation_counts"],
    )
    artifact: dict[str, Any] = {
        "schema_version": V2_STAGE_A_SCHEMA,
        "stage": "stage_a_train_selection",
        "status": "stage_a_failed",
        "evidence_level": "D0",
        "evidence_eligible": False,
        "repository_promotion": False,
        "failure_kind": "runtime_exception",
        "selection_complete": False,
        "parent_plan_sha256": PARENT_PLAN_SHA256,
        "addendum_schema": V2_ADDENDUM_SCHEMA,
        "addendum_sha256": canonical_digest(addendum, "addendum_sha256"),
        "source_sha256": str(source_sha256),
        "public_train_seed": PUBLIC_TRAIN_SEED,
        "train_fixture_sha256": digest_bytes(raw),
        "holdout_commitment": dict(addendum["fixture"]),
        "selection": selection,
        "resources": resource_payload,
        "runtime_attestation": attestation,
        "attestation_sha256": attestation["attestation_sha256"],
    }
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    return artifact


def build_stage_a_validation_rejected_artifact(
    rows: Sequence[Mapping[str, Any]],
    addendum: Mapping[str, Any],
    *,
    source_sha256: str,
    resources: Mapping[str, Any] | None,
    validation_codes: Sequence[str],
    cli_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a sanitized D0 after the primary artifact fails validation."""
    codes = [code for code in validation_codes if code in VALIDATION_REJECTION_CODES]
    if not codes:
        codes = ["validation_rejected_contract"]
    artifact = build_stage_a_failure_artifact(
        rows,
        addendum,
        source_sha256=source_sha256,
        error=RuntimeError("validation rejected"),
        resources=normalize_attempted_real_resources(resources),
        cli_sha256=cli_sha256,
    )
    selection = dict(artifact["selection"])
    selection["failure"] = {"validation_codes": codes}
    artifact["selection"] = selection
    artifact["failure_kind"] = "validation_rejected"
    artifact["selection_complete"] = False
    artifact["runtime_attestation"] = build_runtime_attestation(
        stage="stage_a_train_selection",
        mode="real",
        group_count=TRAIN_GROUP_COUNT,
        pair_count=TRAIN_GROUP_COUNT,
        candidate_count=len(candidate_grid()),
        seed_count=1,
        fixture_sha256=artifact["train_fixture_sha256"],
        candidate_sha256=digest_bytes(canonical_json_bytes(selection)),
        source_sha256=source_sha256,
        addendum_sha256=artifact["addendum_sha256"],
        cli_sha256=cli_sha256 or top_level_cli_sha256("stage_a_train_selection") or "",
        resources=artifact["resources"],
        operation_counts=artifact["resources"]["operation_counts"],
    )
    artifact["attestation_sha256"] = artifact["runtime_attestation"]["attestation_sha256"]
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    return artifact


def run_real_stage_a(
    rows: Sequence[Mapping[str, Any]],
    addendum: Mapping[str, Any],
    *,
    source_sha256: str,
    runtime: Mapping[str, Any],
    cli_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the train-only real boundary through an injected runtime.

    The runtime owns model loading, hooks, captures, interventions, and
    cleanup.  Keeping it injected makes behavioral tests deterministic while
    the CLI can refuse to fabricate a result when CUDA is unavailable.
    """
    scorer = runtime.get("score")
    resources = runtime.get("resources")
    runtime_error = runtime.get("error")
    if not isinstance(runtime_error, BaseException):
        runtime_error = RuntimeError("real Stage A runtime was not available")
    if not callable(scorer) or not isinstance(resources, Mapping):
        return build_stage_a_failure_artifact(
            rows,
            addendum,
            source_sha256=source_sha256,
            error=runtime_error,
            resources=resources,
            cli_sha256=cli_sha256,
        )
    finalizer = resources.get("finalize")
    finalizer_called = False
    builder_resources: Mapping[str, Any] = resources
    if callable(finalizer):
        builder_resources = dict(resources)

        def _tracked_finalizer() -> object:
            nonlocal finalizer_called
            finalizer_called = True
            return cast(Callable[[], object], finalizer)()

        builder_resources["finalize"] = _tracked_finalizer  # type: ignore[index]
    try:
        return build_stage_a_artifact(
            rows,
            addendum,
            source_sha256=source_sha256,
            scorer=cast(ScoreFunction, scorer),
            resources=builder_resources,
            execution_mode="real",
            cli_sha256=cli_sha256,
        )
    except _ResourceFinalizerError as error:
        return build_stage_a_failure_artifact(
            rows,
            addendum,
            source_sha256=source_sha256,
            error=error.cleanup_error,
            resources=resources,
            cleanup_error=error.cleanup_error,
            cleanup_reason=error.reason,
            finalizer_rejection_code=error.finalizer_rejection_code,
            cli_sha256=cli_sha256,
        )
    except Exception as error:  # noqa: BLE001 - every runtime failure is a D0 triad
        fallback_resources: Mapping[str, Any] = resources
        if finalizer_called:
            # A late artifact/attestation failure must not invoke a side-effectful
            # runtime closure for a second time while producing the D0 fallback.
            fallback_resources = {key: item for key, item in resources.items() if key != "finalize"}
        return build_stage_a_failure_artifact(
            rows,
            addendum,
            source_sha256=source_sha256,
            error=error,
            resources=fallback_resources,
            cli_sha256=cli_sha256,
        )


__all__ = [
    "ScoreFunction",
    "attempted_real_resources",
    "build_stage_a_artifact",
    "build_stage_a_failure_artifact",
    "build_stage_a_validation_rejected_artifact",
    "default_train_score",
    "outer_folds",
    "normalize_attempted_real_resources",
    "run_real_stage_a",
    "select_stage_a",
]
