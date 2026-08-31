"""Shared fail-closed primitives for the L04.9 v2 validators."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from scripts._m14_l049_v2_fixture import validate_rows
from scripts._m14_l049_v2_schema import (
    BOOTSTRAP_REPLICATES,
    CANDIDATE_OFFSETS,
    DENOMINATOR_EPSILON,
    EXPECTED_ADDENDUM_SHA256,
    EXPECTED_AUTHORING_MANIFEST_SHA256,
    EXPECTED_HOLDOUT_CONTENT_SHA256,
    EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256,
    EXPECTED_RUNTIME_DTYPE,
    EXPECTED_RUNTIME_INTEGRATION,
    EXPECTED_RUNTIME_MODEL,
    HOLDOUT_GROUP_COUNT,
    OOF_RECOVERY_THRESHOLD,
    RUNTIME_ATTESTATION_SCHEMA,
    RUNTIME_EVENT_CODES,
    STAGE_B_SEEDS,
    TRAIN_GROUP_COUNT,
    V2_ADDENDUM_SCHEMA,
    V2_ROW_KEYS,
    CommitmentPolicy,
    canonical_digest,
    canonical_fixture_bytes,
    canonical_json_bytes,
    digest_bytes,
    directional_recovery,
    fixture_digest,
    is_digest,
    top_level_cli_sha256,
)

EXPECTED_MODEL = "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
REAL_RESOURCE_FIELDS = {
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
    "cleanup",
    "resource_peak",
    "no_mutation",
}


def safe_float(value: object) -> float | None:
    """Return a finite scalar float, rejecting bools, containers and huge ints."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def safe_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if -(1 << 63) <= result < (1 << 63) else None


def finite_values(values: object, *, expected_len: int | None = None) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or (expected_len is not None and len(values) != expected_len):
        return None
    result = [safe_float(value) for value in values]
    if any(value is None for value in result):
        return None
    return [value for value in result if value is not None]


def lower_ci(values: object, seed: object) -> float | None:
    numbers = finite_values(values)
    seed_value = safe_int(seed)
    if not numbers or seed_value is None:
        return None
    try:
        array = np.asarray(numbers, dtype=np.float64)
        rng = np.random.default_rng(seed_value)
        draws = array[rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))]
        result = float(np.quantile(np.mean(draws, axis=1), 0.025))
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def metric(values: object, seed: object, threshold: float) -> dict[str, Any] | None:
    numbers = finite_values(values)
    lower = lower_ci(numbers, seed)
    if not numbers or lower is None:
        return None
    try:
        point = float(np.mean(np.asarray(numbers, dtype=np.float64)))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(point):
        return None
    return {
        "point_estimate": point,
        "lower_ci_95": lower,
        "threshold": float(threshold),
        "comparator": ">",
        "aggregation_unit": "independent causal group",
        "pass": bool(lower > threshold),
    }


def same_metric(declared: object, expected: object) -> bool:
    return isinstance(declared, Mapping) and isinstance(expected, Mapping) and dict(declared) == dict(expected)


def candidate_key(candidate: Mapping[str, object]) -> tuple[int, int]:
    layer = safe_int(candidate.get("layer"))
    offset = safe_int(candidate.get("offset"))
    if layer is None or offset not in CANDIDATE_OFFSETS:
        return (10**9, 10**9)
    return (layer, CANDIDATE_OFFSETS.index(offset))


def groups(
    rows: Sequence[Mapping[str, Any]], split: str, count: int
) -> tuple[dict[str, list[Mapping[str, Any]]], list[str]]:
    errors = list(validate_rows(rows, split, count))
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    row_ids: set[str] = set()
    pair_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or list(row) != list(V2_ROW_KEYS):  # pyright: ignore[reportUnnecessaryIsInstance]
            errors.append("v2 fixture row schema is invalid")
            continue
        row_id, pair_id, group_id = row.get("row_id"), row.get("causal_pair_id"), row.get("group_id")
        if not all(isinstance(value, str) and value for value in (row_id, pair_id, group_id)):
            errors.append("v2 fixture identifiers are invalid")
            continue
        row_id = cast(str, row_id)
        pair_id = cast(str, pair_id)
        group_id = cast(str, group_id)
        if row_id in row_ids:
            errors.append("v2 fixture row IDs are duplicated")
        row_ids.add(row_id)
        pair_ids.add(pair_id)
        grouped.setdefault(group_id, []).append(row)
    if len(rows) != count * 2 or len(grouped) != count or any(len(value) != 2 for value in grouped.values()):
        errors.append("v2 fixture group cardinality is invalid")
    for pair in grouped.values():
        if {row.get("condition") for row in pair} != {"clean", "corrupted"}:
            errors.append("v2 fixture pair conditions are invalid")
        if len({row.get("causal_pair_id") for row in pair}) != 1:
            errors.append("v2 fixture group contains multiple causal pairs")
    if len(pair_ids) != count:
        errors.append("v2 fixture causal-pair IDs are duplicated or missing")
    return dict(sorted(grouped.items())), errors


def addendum_errors(addendum: Mapping[str, Any], policy: CommitmentPolicy | None = None) -> list[str]:
    resolved_policy = policy
    if resolved_policy is None:
        from scripts._m14_l049_v2_schema import pinned_commitment_policy

        resolved_policy = pinned_commitment_policy()
    try:
        expected_addendum = resolved_policy.expected_addendum()
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError):
        return ["v2 commitment policy is malformed"]
    # The policy is immutable and carries the complete canonical addendum,
    # rather than allowing a caller to swap only a digest or one sub-policy.
    errors: list[str] = []
    if dict(addendum) != expected_addendum:
        errors.append("v2 addendum does not match the immutable commitment policy")
    try:
        recomputed = canonical_digest(addendum, "addendum_sha256")
    except (TypeError, ValueError, OverflowError):
        recomputed = None
    if (
        addendum.get("addendum_sha256") != resolved_policy.addendum_sha256
        or recomputed != resolved_policy.addendum_sha256
    ):
        errors.append("v2 addendum digest is not the immutable policy digest")
    if addendum.get("parent_plan_sha256") != resolved_policy.parent_plan_sha256:
        errors.append("v2 addendum parent plan digest is not the immutable policy value")
    authoring = addendum.get("authoring")
    if not isinstance(authoring, Mapping):
        return errors + ["v2 authoring manifest is missing"]
    expected_authoring = expected_addendum.get("authoring")
    expected_manifest_sha = (
        expected_authoring.get("manifest_sha256") if isinstance(expected_authoring, Mapping) else None
    )
    if authoring.get("manifest_sha256") != expected_manifest_sha:
        errors.append("v2 authoring manifest digest is not committed")
    manifest = authoring.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("manifest_sha256") != expected_manifest_sha:
        errors.append("v2 authoring manifest self-binding is invalid")
    fixture = addendum.get("fixture")
    if (
        not isinstance(fixture, Mapping)
        or fixture.get("holdout_content_sha256") != resolved_policy.holdout_content_sha256
    ):
        errors.append("v2 holdout content commitment is invalid")
    if (
        not isinstance(fixture, Mapping)
        or fixture.get("holdout_seed_commitment_sha256") != resolved_policy.holdout_seed_commitment_sha256
    ):
        errors.append("v2 holdout seed commitment is invalid")
    return errors


def mapping_digest(mapping: object) -> str | None:
    if not isinstance(mapping, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()
    ):
        return None
    try:
        return digest_bytes(canonical_json_bytes(dict(mapping)))
    except (TypeError, ValueError, OverflowError):
        return None


def real_resources(resources: object, *, allow_failure: bool = False, require_measured: bool = False) -> list[str]:
    if not isinstance(resources, Mapping):
        return ["Stage B real-runtime resources are missing"]
    errors: list[str] = []

    def _nonnegative(value: object) -> bool:
        parsed = safe_int(value)
        return parsed is not None and parsed >= 0

    missing = REAL_RESOURCE_FIELDS - set(resources)
    if missing:
        errors.append("Stage B real-runtime resource provenance is incomplete")
    if resources.get("execution_attempted") is not True or resources.get("execution_backend") != "cuda":
        errors.append("Stage B real-runtime execution was not evidenced")
    if resources.get("stage") != "real_runtime" and not (allow_failure and resources.get("stage") == "cleanup"):
        errors.append("real-runtime stage marker is invalid")
    if resources.get("model") != EXPECTED_MODEL or resources.get("model_revision") != EXPECTED_MODEL:
        errors.append("Stage B GPT2 revision is not pinned")
    if resources.get("integration") != EXPECTED_RUNTIME_INTEGRATION or resources.get("model_adapter") != "N/A":
        errors.append("Stage B integration boundary is invalid")
    if (
        resources.get("device") != "cuda"
        or resources.get("backend") != "cuda"
        or resources.get("dtype") != EXPECTED_RUNTIME_DTYPE
    ):
        errors.append("Stage B CUDA device/backend evidence is invalid")
    hook = resources.get("hook")
    intervention = resources.get("intervention")
    cleanup = resources.get("cleanup")
    peak = resources.get("resource_peak")
    if (
        not isinstance(hook, Mapping)
        or set(hook) != {"registered", "capture_calls", "removed"}
        or any(not _nonnegative(hook.get(key)) for key in hook)
    ):
        errors.append("Stage B hook execution evidence is missing or malformed")
    if (
        not isinstance(intervention, Mapping)
        or set(intervention) != {"patch_calls", "control_calls", "forward_calls"}
        or any(not _nonnegative(intervention.get(key)) for key in intervention)
    ):
        errors.append("Stage B intervention execution evidence is missing or malformed")
    cleanup_success = (
        isinstance(cleanup, Mapping)
        and set(cleanup) == {"hook_count", "completed"}
        and cleanup.get("completed") is True
        and cleanup.get("hook_count") == 0
    )
    cleanup_failure = (
        allow_failure
        and isinstance(cleanup, Mapping)
        and set(cleanup) == {"attempted", "completed", "hooks_remaining", "error_type", "reason", "stage"}
        and cleanup.get("attempted") is True
        and cleanup.get("completed") is False
        and _nonnegative(cleanup.get("hooks_remaining"))
        and cleanup.get("error_type")
        in {
            "CleanupError",
            "Exception",
            "MemoryError",
            "OSError",
            "RuntimeError",
            "TimeoutError",
            "TypeError",
            "ValueError",
        }
        and cleanup.get("reason") in {"finalizer_exception", "finalizer_invalid_result"}
        and cleanup.get("stage") == "cleanup"
    )
    if not cleanup_success and not cleanup_failure:
        errors.append("Stage B cleanup execution evidence is missing or malformed")

    def _bounded_peak(value: object, budget: object) -> bool:
        peak_value = safe_int(value)
        budget_value = safe_int(budget)
        if peak_value is None or budget_value is None:
            return False
        return peak_value >= 0 and budget_value >= 0 and peak_value <= budget_value

    expected_peak_fields = {
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
    }
    if (
        not isinstance(peak, Mapping)
        or set(peak) != expected_peak_fields
        or peak.get("unit") != "bytes"
        or not _bounded_peak(peak.get("peak_cpu_bytes"), peak.get("budget_cpu_bytes"))
        or not _bounded_peak(peak.get("peak_gpu_bytes"), peak.get("budget_gpu_bytes"))
        or not _bounded_peak(peak.get("peak_gpu_reserved_bytes"), peak.get("budget_gpu_bytes"))
    ):
        errors.append("Stage B resource peak evidence is missing or exceeds budget")
    elif peak.get("peak_gpu_reserved_bytes", 0) < peak.get("peak_gpu_bytes", 0):
        errors.append("Stage B reserved GPU peak is below allocated GPU peak")
    elif peak.get("measurement_status") not in {"available", "unavailable"}:
        errors.append("Stage B resource measurement status is invalid")
    elif peak.get("measurement_status") == "available":

        def _positive_int(value: object) -> bool:
            parsed = safe_int(value)
            return parsed is not None and parsed > 0

        def _positive_float(value: object) -> bool:
            parsed = safe_float(value)
            return parsed is not None and parsed > 0

        if (
            not _positive_int(peak.get("peak_cpu_bytes"))
            or not _positive_int(peak.get("peak_gpu_bytes"))
            or not _positive_int(peak.get("peak_gpu_reserved_bytes"))
            or peak.get("measurement_reason") is not None
            or peak.get("elapsed_source") != "time.perf_counter"
            or not _positive_float(peak.get("elapsed_seconds"))
            or peak.get("cpu_source")
            not in {
                "resource.ru_maxrss_linux_kib",
                "resource.ru_maxrss_macos_bytes",
                "psutil.Process.memory_info.rss",
            }
            or peak.get("gpu_source") != "torch.cuda.max_memory_allocated"
            or peak.get("gpu_reserved_source") != "torch.cuda.max_memory_reserved"
            or not isinstance(peak.get("gpu_device"), str)
            or not peak.get("gpu_device", "").startswith("cuda:")
        ):
            errors.append("Stage B measured resource provenance is invalid")
    else:
        reason = peak.get("measurement_reason")

        def _positive_int(value: object) -> bool:
            parsed = safe_int(value)
            return parsed is not None and parsed > 0

        valid_reasons = {
            "cuda_unavailable",
            "cuda_reset_failed",
            "cuda_peak_query_failed",
            "cuda_zero_peak",
            "rss_unavailable",
            "clock_invalid",
            "tracker_unstarted",
        }
        cuda_unavailable = reason in {
            "cuda_unavailable",
            "cuda_reset_failed",
            "cuda_peak_query_failed",
            "cuda_zero_peak",
        }
        rss_unavailable = reason == "rss_unavailable"
        source_shape_valid = (
            reason in valid_reasons
            and (
                not cuda_unavailable
                or (peak.get("gpu_source") == "unavailable" and peak.get("gpu_reserved_source") == "unavailable")
            )
            and (not rss_unavailable or peak.get("cpu_source") == "unavailable")
            and (
                reason != "tracker_unstarted"
                or (
                    peak.get("cpu_source") == "unavailable"
                    and peak.get("gpu_source") == "unavailable"
                    and peak.get("gpu_reserved_source") == "unavailable"
                )
            )
            and (
                not cuda_unavailable
                or peak.get("gpu_device") == "unavailable"
                or (isinstance(peak.get("gpu_device"), str) and peak.get("gpu_device", "").startswith("cuda:"))
            )
        )
        cpu_source = peak.get("cpu_source")
        gpu_source = peak.get("gpu_source")
        reserved_source = peak.get("gpu_reserved_source")
        source_value_valid = (
            (
                (cpu_source == "unavailable" and peak.get("peak_cpu_bytes") == 0)
                or (
                    cpu_source
                    in {
                        "resource.ru_maxrss_linux_kib",
                        "resource.ru_maxrss_macos_bytes",
                        "psutil.Process.memory_info.rss",
                    }
                    and _positive_int(peak.get("peak_cpu_bytes"))
                )
            )
            and (
                (gpu_source == "unavailable" and peak.get("peak_gpu_bytes") == 0)
                or (gpu_source == "torch.cuda.max_memory_allocated" and _positive_int(peak.get("peak_gpu_bytes")))
            )
            and (
                (reserved_source == "unavailable" and peak.get("peak_gpu_reserved_bytes") == 0)
                or (
                    reserved_source == "torch.cuda.max_memory_reserved"
                    and _positive_int(peak.get("peak_gpu_reserved_bytes"))
                )
            )
        )
        if not source_shape_valid or not source_value_valid:
            errors.append("Stage B unavailable resource provenance is invalid")
        if require_measured:
            errors.append("measured resource provenance is required for eligible evidence")
    if resources.get("no_mutation") is not True:
        errors.append("Stage B no-mutation gate failed")
    counters = resources.get("operation_counts")
    counter_values = [safe_int(counters.get(key)) for key in counters] if isinstance(counters, Mapping) else []
    if (
        not isinstance(counters, Mapping)
        or set(counters) != {"candidate_evaluations", "hooks", "captures", "patches", "controls", "forwards"}
        or any(value is None or value < 0 for value in counter_values)
    ):
        errors.append("real-runtime operation counters are missing or malformed")
    if isinstance(hook, Mapping) and isinstance(counters, Mapping):
        registered = safe_int(hook.get("registered"))
        removed = safe_int(hook.get("removed"))
        hook_total = safe_int(counters.get("hooks"))
        if (
            registered is None
            or removed is None
            or hook_total is None
            or registered != hook_total
            or removed > registered
        ):
            errors.append("real-runtime hook registration/removal counters are inconsistent")
        else:
            remaining = registered - removed
            if cleanup_failure and isinstance(cleanup, Mapping) and cleanup.get("hooks_remaining") != remaining:
                errors.append("real-runtime cleanup remaining hooks are not counter-derived")
            if (
                cleanup_success
                and isinstance(cleanup, Mapping)
                and (cleanup.get("hook_count") != remaining or remaining != 0)
            ):
                errors.append("real-runtime successful cleanup hooks are not fully removed")
    return errors


def expected_runtime_counts(
    *, stage: str, mode: str, group_count: int, pair_count: int, candidate_count: int, seed_count: int
) -> dict[str, int]:
    """Derive operation counts from the stage protocol, independently of resources."""
    if mode != "real":
        return {
            "groups": group_count,
            "pairs": pair_count,
            "candidates": candidate_count,
            "seeds": seed_count,
            "candidate_evaluations": group_count * 2 * candidate_count * seed_count,
            "hooks": group_count * seed_count,
            "captures": group_count * 2 * seed_count,
            "patches": group_count * seed_count,
            "controls": group_count * 4 * seed_count,
            "forwards": group_count * 2 * seed_count + group_count * seed_count + group_count * 4 * seed_count,
        }
    if stage == "stage_a_train_selection":
        # One clean and one corrupted base forward per pair; every corrupted
        # row/candidate receives one patched forward.  The clean rows still
        # count as candidate evaluations but reuse their captured base output.
        evaluations = pair_count * 2 * candidate_count
        patches = pair_count * candidate_count
        forwards = pair_count * 2 + patches
        return {
            "groups": group_count,
            "pairs": pair_count,
            "candidates": candidate_count,
            "seeds": seed_count,
            "candidate_evaluations": evaluations,
            "hooks": patches,
            "captures": forwards,
            "patches": patches,
            "controls": 0,
            "forwards": forwards,
        }
    # Stage B executes one selected-candidate patch per causal pair, then four
    # independently forwarded controls for every declared seed.
    patches = pair_count
    controls = pair_count * 4 * seed_count
    forwards = pair_count * 3 + controls
    return {
        "groups": group_count,
        "pairs": pair_count,
        "candidates": candidate_count,
        "seeds": seed_count,
        "candidate_evaluations": pair_count * seed_count,
        "hooks": patches,
        "captures": forwards,
        "patches": patches,
        "controls": controls,
        "forwards": forwards,
    }


def canonical_artifact_digest(value: Mapping[str, Any], field: str) -> str | None:
    try:
        return canonical_digest(value, field)
    except (TypeError, ValueError, OverflowError):
        return None


def runtime_attestation_errors(
    attestation: object,
    *,
    stage: str,
    mode: str,
    group_count: int,
    pair_count: int,
    candidate_count: int,
    seed_count: int,
    fixture_sha256: object,
    candidate_sha256: object,
    source_sha256: object,
    addendum_sha256: object,
    cli_sha256: object,
    execution_resources: object = None,
    partial_failure: bool = False,
) -> list[str]:
    """Independently validate every structured attestation primitive."""
    errors: list[str] = []
    if not isinstance(attestation, Mapping):
        return ["runtime attestation is missing"]
    expected_keys = {
        "schema_version",
        "stage",
        "mode",
        "model",
        "commitments",
        "events",
        "counts",
        "parameters",
        "cleanup_hook_count",
        "resources",
        "cli_sha256",
        "runtime_module_digests",
        "hash_chain",
        "hash_chain_sha256",
        "transcript_sha256",
        "execution_backend",
        "attestation_sha256",
    }
    if set(attestation) != expected_keys:
        errors.append("runtime attestation fields are invalid")
    if attestation.get("schema_version") != RUNTIME_ATTESTATION_SCHEMA:
        errors.append("runtime attestation schema is invalid")
    if attestation.get("stage") != stage or attestation.get("mode") != mode:
        errors.append("runtime attestation stage or mode is invalid")
    backend = "cuda" if mode == "real" else "synthetic"
    model = attestation.get("model")
    expected_model = {
        "name": EXPECTED_RUNTIME_MODEL,
        "revision": EXPECTED_RUNTIME_MODEL,
        "integration": EXPECTED_RUNTIME_INTEGRATION,
        "model_adapter": "N/A",
        "device": "cuda" if mode == "real" else "cpu",
        "backend": backend,
        "dtype": EXPECTED_RUNTIME_DTYPE,
    }
    if model != expected_model:
        errors.append("runtime attestation model boundary is invalid")
    expected_commitments = {
        "fixture_sha256": fixture_sha256,
        "candidate_sha256": candidate_sha256,
        "source_sha256": source_sha256,
        "addendum_sha256": addendum_sha256,
    }
    if attestation.get("commitments") != expected_commitments or any(
        not is_digest(value) for value in expected_commitments.values()
    ):
        errors.append("runtime attestation primitive commitments are invalid")
    expected_events = [{"ordinal": index, "code": code} for index, code in enumerate(RUNTIME_EVENT_CODES)]
    if attestation.get("events") != expected_events:
        errors.append("runtime attestation event order is invalid")
    expected_counts = expected_runtime_counts(
        stage=stage,
        mode=mode,
        group_count=group_count,
        pair_count=pair_count,
        candidate_count=candidate_count,
        seed_count=seed_count,
    )
    protocol_counts = dict(expected_counts)
    if partial_failure and isinstance(execution_resources, Mapping):
        partial_counts = execution_resources.get("operation_counts")
        if isinstance(partial_counts, Mapping) and set(partial_counts) == {
            "candidate_evaluations",
            "hooks",
            "captures",
            "patches",
            "controls",
            "forwards",
        }:
            for key in partial_counts:
                value = safe_int(partial_counts.get(key))
                if value is not None and 0 <= value <= protocol_counts[key]:
                    expected_counts[key] = value
                else:
                    errors.append("partial runtime operation counter exceeds protocol bound")
    if attestation.get("counts") != expected_counts:
        errors.append("runtime attestation execution counts are invalid")
    expected_chain = [
        {"name": "fixture", "sha256": fixture_sha256},
        {"name": "candidate", "sha256": candidate_sha256},
        {"name": "source", "sha256": source_sha256},
        {"name": "addendum", "sha256": addendum_sha256},
    ]
    if attestation.get("hash_chain") != expected_chain:
        errors.append("runtime attestation hash chain is invalid")
    try:
        expected_chain_sha = digest_bytes(canonical_json_bytes({"items": expected_chain}))
    except (TypeError, ValueError, OverflowError):
        expected_chain_sha = None
    if attestation.get("hash_chain_sha256") != expected_chain_sha:
        errors.append("runtime attestation hash-chain digest is invalid")
    parameters = attestation.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or set(parameters) != {"before_sha256", "after_sha256"}
        or not is_digest(parameters.get("before_sha256"))
        or parameters.get("before_sha256") != parameters.get("after_sha256")
    ):
        errors.append("runtime attestation parameter identity is invalid")
    expected_cleanup_hook_count = 0
    if partial_failure and isinstance(execution_resources, Mapping):
        cleanup = execution_resources.get("cleanup")
        if isinstance(cleanup, Mapping) and cleanup.get("completed") is False:
            parsed_remaining = safe_int(cleanup.get("hooks_remaining"))
            expected_cleanup_hook_count = -1 if parsed_remaining is None else parsed_remaining
    if attestation.get("cleanup_hook_count") != expected_cleanup_hook_count:
        errors.append("runtime attestation cleanup count is invalid")
    expected_resources = {
        "peak_cpu_bytes": 0,
        "peak_gpu_bytes": 0,
        "unit": "bytes",
        "source": "synthetic_fixture",
        "budget_cpu_bytes": 0,
        "budget_gpu_bytes": 0,
    }
    if partial_failure and isinstance(execution_resources, Mapping):
        counters = execution_resources.get("operation_counts")
        if not isinstance(counters, Mapping) or any(
            attestation.get("counts", {}).get(key) != counters.get(key)
            for key in ("candidate_evaluations", "hooks", "captures", "patches", "controls", "forwards")
        ):
            errors.append("partial runtime operation counters are not execution-bound")
    if mode == "real" and isinstance(execution_resources, Mapping):
        peak = execution_resources.get("resource_peak")
        if isinstance(peak, Mapping):
            expected_resources = {
                "peak_cpu_bytes": peak.get("peak_cpu_bytes"),
                "peak_gpu_bytes": peak.get("peak_gpu_bytes"),
                "peak_gpu_reserved_bytes": peak.get("peak_gpu_reserved_bytes"),
                "unit": peak.get("unit"),
                "source": peak.get("gpu_source"),
                "budget_cpu_bytes": peak.get("budget_cpu_bytes"),
                "budget_gpu_bytes": peak.get("budget_gpu_bytes"),
                "measurement_status": peak.get("measurement_status"),
                "measurement_reason": peak.get("measurement_reason"),
                "elapsed_seconds": peak.get("elapsed_seconds"),
                "elapsed_source": peak.get("elapsed_source"),
                "cpu_source": peak.get("cpu_source"),
                "gpu_source": peak.get("gpu_source"),
                "gpu_reserved_source": peak.get("gpu_reserved_source"),
                "gpu_device": peak.get("gpu_device"),
            }
    resources = attestation.get("resources")
    if resources != expected_resources:
        errors.append("runtime attestation resource peaks or units are invalid")
    if mode == "real" and isinstance(execution_resources, Mapping) and not partial_failure:
        hook = execution_resources.get("hook")
        intervention = execution_resources.get("intervention")
        cleanup = execution_resources.get("cleanup")
        counters = execution_resources.get("operation_counts")
        if not isinstance(counters, Mapping) or counters != {
            key: expected_counts[key]
            for key in ("candidate_evaluations", "hooks", "captures", "patches", "controls", "forwards")
        }:
            errors.append("runtime operation counters are not independently derived")
        if not isinstance(hook, Mapping) or {
            "registered": hook.get("registered"),
            "capture_calls": hook.get("capture_calls"),
            "removed": hook.get("removed"),
        } != {
            "registered": expected_counts["hooks"],
            "capture_calls": expected_counts["captures"],
            "removed": expected_counts["hooks"],
        }:
            errors.append("runtime attestation hook counts are not execution-bound")
        if not isinstance(intervention, Mapping) or {
            "patch_calls": intervention.get("patch_calls"),
            "control_calls": intervention.get("control_calls"),
            "forward_calls": intervention.get("forward_calls"),
        } != {
            "patch_calls": expected_counts["patches"],
            "control_calls": expected_counts["controls"],
            "forward_calls": expected_counts["forwards"],
        }:
            errors.append("runtime attestation intervention counts are not execution-bound")
        if not isinstance(cleanup, Mapping) or cleanup != {"hook_count": 0, "completed": True}:
            errors.append("runtime attestation cleanup is not execution-bound")
    expected_cli = top_level_cli_sha256(stage) or cli_sha256
    if attestation.get("cli_sha256") != expected_cli or not is_digest(expected_cli):
        errors.append("runtime attestation CLI digest is invalid")
    modules = attestation.get("runtime_module_digests")
    if modules != {"producer": source_sha256, "integration": source_sha256} or not is_digest(source_sha256):
        errors.append("runtime attestation module digests are invalid")
    try:
        expected_transcript = digest_bytes(canonical_json_bytes({"items": expected_events}))
    except (TypeError, ValueError, OverflowError):
        expected_transcript = None
    if attestation.get("transcript_sha256") != expected_transcript:
        errors.append("runtime attestation transcript digest is invalid")
    if attestation.get("execution_backend") != backend:
        errors.append("runtime attestation backend is invalid")
    if attestation.get("attestation_sha256") != canonical_artifact_digest(attestation, "attestation_sha256"):
        errors.append("runtime attestation self-digest is invalid")
    return errors


__all__ = [
    "EXPECTED_MODEL",
    "addendum_errors",
    "candidate_key",
    "canonical_artifact_digest",
    "runtime_attestation_errors",
    "finite_values",
    "fixture_digest",
    "groups",
    "lower_ci",
    "mapping_digest",
    "metric",
    "real_resources",
    "same_metric",
    "safe_float",
    "safe_int",
    "canonical_fixture_bytes",
    "is_digest",
    "directional_recovery",
    "expected_runtime_counts",
    "EXPECTED_ADDENDUM_SHA256",
    "EXPECTED_AUTHORING_MANIFEST_SHA256",
    "EXPECTED_HOLDOUT_CONTENT_SHA256",
    "EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256",
    "V2_ADDENDUM_SCHEMA",
    "OOF_RECOVERY_THRESHOLD",
    "STAGE_B_SEEDS",
    "TRAIN_GROUP_COUNT",
    "HOLDOUT_GROUP_COUNT",
    "DENOMINATOR_EPSILON",
]
