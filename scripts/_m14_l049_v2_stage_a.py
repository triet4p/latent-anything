"""Offline train-only candidate selection for the L04.9 v2 addendum."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from scripts._m14_l049_v2_attestation import build_runtime_attestation
from scripts._m14_l049_v2_schema import (
    BOOTSTRAP_REPLICATES,
    EXPECTED_RUNTIME_MODEL,
    OOF_RECOVERY_THRESHOLD,
    PARENT_PLAN_SHA256,
    PUBLIC_TRAIN_SEED,
    TRAIN_GROUP_COUNT,
    V2_ADDENDUM_SCHEMA,
    V2_STAGE_A_SCHEMA,
    candidate_grid,
    canonical_digest,
    canonical_fixture_bytes,
    canonical_json_bytes,
    digest_bytes,
    top_level_cli_sha256,
)

ScoreFunction = Callable[[Mapping[str, Any], int, int], float]

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


def _valid_finalizer_resources(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(_FINALIZER_RESOURCE_FIELDS):
        return False
    if (
        value.get("stage") != "real_runtime"
        or value.get("execution_attempted") is not True
        or value.get("execution_backend") != "cuda"
        or value.get("model") != EXPECTED_RUNTIME_MODEL
        or value.get("model_revision") != EXPECTED_RUNTIME_MODEL
        or value.get("integration") != "TransformerLMIntegration"
        or value.get("model_adapter") != "N/A"
        or value.get("device") != "cuda"
        or value.get("backend") != "cuda"
        or value.get("dtype") != "float32"
        or value.get("no_mutation") is not True
    ):
        return False
    counters = value.get("operation_counts")
    if (
        not isinstance(counters, Mapping)
        or set(counters) != set(_COUNTER_FIELDS)
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in counters.values())
    ):
        return False
    hook = value.get("hook")
    if (
        not isinstance(hook, Mapping)
        or set(hook) != {"registered", "capture_calls", "removed"}
        or hook.get("registered") != counters["hooks"]
        or hook.get("capture_calls") != counters["captures"]
        or hook.get("removed") != hook.get("registered")
    ):
        return False
    intervention = value.get("intervention")
    if (
        not isinstance(intervention, Mapping)
        or set(intervention) != {"patch_calls", "control_calls", "forward_calls"}
        or intervention.get("patch_calls") != counters["patches"]
        or intervention.get("control_calls") != counters["controls"]
        or intervention.get("forward_calls") != counters["forwards"]
    ):
        return False
    cleanup = value.get("cleanup")
    if not isinstance(cleanup, Mapping) or cleanup != {"hook_count": 0, "completed": True}:
        return False
    peak = value.get("resource_peak")
    peak_fields = (
        "peak_cpu_bytes",
        "peak_gpu_bytes",
        "peak_gpu_reserved_bytes",
        "budget_cpu_bytes",
        "budget_gpu_bytes",
    )

    def _valid_nonnegative(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    return not (
        not isinstance(peak, Mapping)
        or set(peak)
        != {
            *peak_fields,
            "unit",
            "measurement_status",
            "measurement_reason",
            "elapsed_seconds",
            "elapsed_source",
            "cpu_source",
            "gpu_source",
            "gpu_reserved_source",
            "gpu_device",
        }
        or peak.get("unit") != "bytes"
        or any(not _valid_nonnegative(peak.get(key)) for key in peak_fields)
        or peak["peak_cpu_bytes"] > peak["budget_cpu_bytes"]
        or peak["peak_gpu_bytes"] > peak["budget_gpu_bytes"]
        or (
            isinstance(peak["peak_gpu_reserved_bytes"], int)
            and isinstance(peak["peak_gpu_bytes"], int)
            and peak["peak_gpu_reserved_bytes"] < peak["peak_gpu_bytes"]
        )
        or peak["measurement_status"] not in {"available", "unavailable"}
        or peak["measurement_reason"] is not None
        and not isinstance(peak["measurement_reason"], str)
        or peak["elapsed_seconds"] is not None
        and (not isinstance(peak["elapsed_seconds"], (int, float)) or isinstance(peak["elapsed_seconds"], bool))
    )


class _ResourceFinalizerError(RuntimeError):
    """Internal boundary preserving a finalizer failure separately."""

    def __init__(self, cleanup_error: BaseException, *, reason: str) -> None:
        self.cleanup_error = cleanup_error
        self.reason = reason
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


def _score_records(rows: Sequence[Mapping[str, Any]], scorer: ScoreFunction) -> list[dict[str, Any]]:
    groups = _group_rows(rows)
    records: list[dict[str, Any]] = []
    for group_id, group_rows in groups.items():
        for candidate in candidate_grid():
            values = [float(scorer(row, candidate["layer"], candidate["offset"])) for row in group_rows]
            if not np.isfinite(values).all():
                raise ValueError("train candidate scores must be finite")
            records.append(
                {
                    "group_id": group_id,
                    "layer": candidate["layer"],
                    "offset": candidate["offset"],
                    "row_scores": values,
                    "group_score": float(np.mean(values)),
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
    resource_payload = (
        dict(resources)
        if resources is not None
        else {
            "stage": "protocol_fixture",
            "execution_backend": "cpu",
            "execution_attempted": False,
            "no_mutation": True,
        }
    )
    finalizer = resource_payload.pop("finalize", None)
    if callable(finalizer):
        try:
            finalized = finalizer()
        except Exception as error:  # noqa: BLE001 - promote as a D0 cleanup failure
            raise _ResourceFinalizerError(error, reason="finalizer_exception") from error
        if not _valid_finalizer_resources(finalized):
            error = TypeError("runtime resource finalizer returned an invalid mapping")
            raise _ResourceFinalizerError(error, reason="finalizer_invalid_result") from error
        resource_payload = dict(cast(Mapping[str, Any], finalized))
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


def _cleanup_failure(error: BaseException, prior: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    hook = prior.get("hook")
    registered = hook.get("registered") if isinstance(hook, Mapping) else 0
    removed = hook.get("removed") if isinstance(hook, Mapping) else 0
    if not isinstance(registered, int) or isinstance(registered, bool) or registered < 0:
        registered = 0
    if not isinstance(removed, int) or isinstance(removed, bool) or removed < 0:
        removed = 0
    hooks_remaining = max(registered - removed, 0)
    return {
        "attempted": True,
        "completed": False,
        "hooks_remaining": hooks_remaining,
        "error_type": _cleanup_error_type(error),
        "reason": reason,
        "stage": "cleanup",
    }


def _failure_resources(
    resources: Mapping[str, Any] | None,
    *,
    cleanup_error: BaseException | None = None,
    cleanup_reason: str = "finalizer_exception",
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(cast(Mapping[str, Any], resources or {}))
    finalizer = payload.pop("finalize", None)
    if cleanup_error is None and callable(finalizer):
        try:
            finalized = finalizer()
        except Exception as error:  # noqa: BLE001 - preserve primary runtime failure
            cleanup_error = error
        else:
            if _valid_finalizer_resources(finalized):
                payload = dict(cast(Mapping[str, Any], finalized))
            else:
                cleanup_error = TypeError("runtime resource finalizer returned an invalid mapping")
                cleanup_reason = "finalizer_invalid_result"
    payload.pop("finalize", None)
    attempted = payload.get("execution_attempted") is True
    if attempted:
        payload["stage"] = "cleanup"
        if cleanup_error is not None:
            payload["cleanup"] = _cleanup_failure(cleanup_error, payload, reason=cleanup_reason)
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
    cli_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a truthful non-promoting artifact after an incomplete real run."""
    resource_payload = _failure_resources(resources, cleanup_error=cleanup_error, cleanup_reason=cleanup_reason)
    attempted = resource_payload.get("execution_backend") == "cuda"
    resource_payload.setdefault("model", EXPECTED_RUNTIME_MODEL)
    resource_payload.setdefault("model_revision", EXPECTED_RUNTIME_MODEL)
    resource_payload.setdefault("integration", "TransformerLMIntegration")
    resource_payload.setdefault("model_adapter", "N/A")
    resource_payload.setdefault("backend", "cuda" if attempted else "none")
    resource_payload.setdefault("dtype", "float32")
    resource_payload.setdefault("network", "enabled" if attempted else "not attempted")
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
    try:
        return build_stage_a_artifact(
            rows,
            addendum,
            source_sha256=source_sha256,
            scorer=cast(ScoreFunction, scorer),
            resources=resources,
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
            cli_sha256=cli_sha256,
        )
    except Exception as error:  # noqa: BLE001 - every runtime failure is a D0 triad
        return build_stage_a_failure_artifact(
            rows,
            addendum,
            source_sha256=source_sha256,
            error=error,
            resources=resources,
            cli_sha256=cli_sha256,
        )


__all__ = [
    "ScoreFunction",
    "attempted_real_resources",
    "build_stage_a_artifact",
    "build_stage_a_failure_artifact",
    "default_train_score",
    "outer_folds",
    "run_real_stage_a",
    "select_stage_a",
]
