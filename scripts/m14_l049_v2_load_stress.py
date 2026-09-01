"""Train-only model-load stress diagnostic for the L04.9 v2 runtime.

This entry point deliberately exercises a bounded, fixed candidate workload
through the production Stage A scorer using the repository-owned train
fixture.  It emits no candidate, selection, fold, OOF, fixture, prompt, or
artifact data; the output is a fixed marker grammar containing only sanitized
resource/finalizer categories and booleans.  It must never be used with
holdout or Stage B inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts._m14_l049_v2_fixture import TRAIN_FIXTURE_PATH, read_rows
from scripts._m14_l049_v2_real_runtime import (
    RealRuntimeError,
    attempted_runtime_resources,
    build_stage_a_runtime,
)
from scripts._m14_l049_v2_schema import FINALIZER_REJECTION_CODES
from scripts._m14_l049_v2_stage_a import run_stage_a_candidate_workload, validate_finalizer_resources

_MARKERS = (
    "L049_V2_LOAD_STRESS_STATUS",
    "L049_V2_LOAD_STRESS_FINALIZER_CODE",
    "L049_V2_LOAD_STRESS_MEASUREMENT_STATUS",
    "L049_V2_LOAD_STRESS_MEASUREMENT_REASON",
    "L049_V2_LOAD_STRESS_CPU_PROVENANCE",
    "L049_V2_LOAD_STRESS_GPU_PROVENANCE",
    "L049_V2_LOAD_STRESS_DEVICE_CANONICAL",
    "L049_V2_LOAD_STRESS_COUNTERS_COMPLETE",
    "L049_V2_LOAD_STRESS_CPU_BUDGET_OK",
    "L049_V2_LOAD_STRESS_GPU_ALLOCATED_BUDGET_OK",
    "L049_V2_LOAD_STRESS_GPU_RESERVED_BUDGET_OK",
    "L049_V2_LOAD_STRESS_CLEANUP",
)
_REASONS = frozenset(
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
_COUNTERS = frozenset(("candidate_evaluations", "hooks", "captures", "patches", "controls", "forwards"))
_EXPECTED_COUNTERS = {
    "candidate_evaluations": 2592,
    "hooks": 1296,
    "captures": 1368,
    "patches": 1296,
    "controls": 0,
    "forwards": 1368,
}


def _bool_marker(value: object) -> str:
    return "true" if value else "false"


def _counter_complete(resources: Mapping[str, Any]) -> bool:
    counters = resources.get("operation_counts")
    return (
        isinstance(counters, Mapping)
        and set(counters) == set(_COUNTERS)
        and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counters.values())
        and all(counters[key] == value for key, value in _EXPECTED_COUNTERS.items())
    )


def _safe_markers(
    resources: Mapping[str, Any],
    *,
    workload_ok: bool,
    cleanup_ok: bool,
    rejection_code: str | None = None,
) -> tuple[tuple[str, str], ...]:
    peak = resources.get("resource_peak")
    if not isinstance(peak, Mapping):
        peak = {}
    raw_status = peak.get("measurement_status")
    status = raw_status if isinstance(raw_status, str) and raw_status in {"available", "unavailable"} else "unknown"
    raw_reason = peak.get("measurement_reason")
    reason = (
        raw_reason
        if isinstance(raw_reason, str) and raw_reason in _REASONS
        else "none"
        if raw_reason is None
        else "unknown"
    )
    code = rejection_code
    safe_code = code if isinstance(code, str) and code in FINALIZER_REJECTION_CODES else "finalizer_not_mapping"
    if (not workload_ok or not cleanup_ok) and code is None:
        safe_code = "finalizer_top_level_fields"
    cpu_source = peak.get("cpu_source")
    gpu_source = peak.get("gpu_source")
    reserved_source = peak.get("gpu_reserved_source")
    device = peak.get("gpu_device")
    canonical_device = isinstance(device, str) and (
        device == "unavailable" or (device.startswith("cuda:") and device[5:].isdigit())
    )
    cpu_peak = peak.get("peak_cpu_bytes")
    gpu_peak = peak.get("peak_gpu_bytes")
    reserved_peak = peak.get("peak_gpu_reserved_bytes")
    cpu_budget = peak.get("budget_cpu_bytes")
    gpu_budget = peak.get("budget_gpu_bytes")

    def budgets_ok(value: object, budget: object) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and isinstance(budget, int)
            and not isinstance(budget, bool)
            and 0 <= value <= budget
        )

    status_ok = (
        workload_ok
        and cleanup_ok
        and code is None
        and status == "available"
        and reason == "none"
        and cpu_source != "unavailable"
        and gpu_source != "unavailable"
        and reserved_source != "unavailable"
        and canonical_device
        and _counter_complete(resources)
        and budgets_ok(cpu_peak, cpu_budget)
        and budgets_ok(gpu_peak, gpu_budget)
        and budgets_ok(reserved_peak, gpu_budget)
    )
    return (
        ("L049_V2_LOAD_STRESS_STATUS", "PASS" if status_ok else "FAIL"),
        ("L049_V2_LOAD_STRESS_FINALIZER_CODE", "NONE" if status_ok else safe_code),
        ("L049_V2_LOAD_STRESS_MEASUREMENT_STATUS", status),
        ("L049_V2_LOAD_STRESS_MEASUREMENT_REASON", reason),
        ("L049_V2_LOAD_STRESS_CPU_PROVENANCE", _bool_marker(cpu_source != "unavailable")),
        (
            "L049_V2_LOAD_STRESS_GPU_PROVENANCE",
            _bool_marker(gpu_source != "unavailable" and reserved_source != "unavailable"),
        ),
        ("L049_V2_LOAD_STRESS_DEVICE_CANONICAL", _bool_marker(canonical_device)),
        ("L049_V2_LOAD_STRESS_COUNTERS_COMPLETE", _bool_marker(_counter_complete(resources))),
        ("L049_V2_LOAD_STRESS_CPU_BUDGET_OK", _bool_marker(budgets_ok(cpu_peak, cpu_budget))),
        ("L049_V2_LOAD_STRESS_GPU_ALLOCATED_BUDGET_OK", _bool_marker(budgets_ok(gpu_peak, gpu_budget))),
        ("L049_V2_LOAD_STRESS_GPU_RESERVED_BUDGET_OK", _bool_marker(budgets_ok(reserved_peak, gpu_budget))),
        ("L049_V2_LOAD_STRESS_CLEANUP", "PASS" if cleanup_ok else "FAIL"),
    )


def validate_load_stress_output(output: str) -> list[str]:
    """Validate fixed stress markers without accepting arbitrary free text."""
    lines = output.splitlines()
    if len(lines) != len(_MARKERS):
        return ["load stress marker count is invalid"]
    records: dict[str, str] = {}
    for line in lines:
        name, separator, value = line.partition("=")
        if not separator or name not in _MARKERS or name in records:
            return ["load stress marker names are invalid"]
        records[name] = value
    if tuple(records) != _MARKERS:
        return ["load stress marker order is invalid"]
    if records[_MARKERS[0]] not in {"PASS", "FAIL"}:
        return ["load stress status is invalid"]
    code = records[_MARKERS[1]]
    if code != "NONE" and code not in FINALIZER_REJECTION_CODES:
        return ["load stress finalizer code is invalid"]
    if (records[_MARKERS[0]] == "PASS") != (code == "NONE"):
        return ["load stress status and finalizer code disagree"]
    status = records[_MARKERS[2]]
    reason = records[_MARKERS[3]]
    if status not in {"available", "unavailable", "unknown"}:
        return ["load stress measurement status is invalid"]
    if reason not in _REASONS and reason not in {"none", "unknown"}:
        return ["load stress measurement reason is invalid"]
    for name in _MARKERS[4:11]:
        if records[name] not in {"true", "false"}:
            return ["load stress boolean marker is invalid"]
    cleanup = records[_MARKERS[11]]
    if cleanup not in {"PASS", "FAIL"}:
        return ["load stress cleanup marker is invalid"]
    if records[_MARKERS[0]] == "PASS":
        if (
            status != "available"
            or reason != "none"
            or cleanup != "PASS"
            or any(records[name] != "true" for name in _MARKERS[4:11])
        ):
            return ["load stress PASS markers are inconsistent"]
    elif code == "NONE":
        return ["load stress FAIL cannot claim no finalizer rejection"]
    if (status == "unknown" or reason == "unknown") and (records[_MARKERS[0]] != "FAIL" or code == "NONE"):
        return ["unknown load stress metadata must fail closed"]
    return []


def run_load_stress() -> tuple[tuple[tuple[str, str], ...], int]:
    """Execute one fixed train-only workload and return sanitized markers."""
    resources: Mapping[str, Any] = attempted_runtime_resources()
    finalizer: Any = None
    workload_ok = True
    cleanup_ok = False
    rejection_code: str | None = None
    try:
        _header, rows = read_rows(TRAIN_FIXTURE_PATH)
        scorer, runtime_resources = build_stage_a_runtime(rows)
        resources = runtime_resources
        finalizer = runtime_resources.get("finalize")
        # Reuse the exact production candidate loop. Records are discarded at
        # the callback boundary, so no candidate/selection artifact is emitted
        # while scorer cache lifetime and operation counters remain identical.
        workload_ok = run_stage_a_candidate_workload(rows, scorer) == 36 * 36
    except RealRuntimeError as error:
        resources = error.resources
        workload_ok = False
    except Exception:  # noqa: BLE001 - fixed markers only
        workload_ok = False
    finally:
        if callable(finalizer):
            try:
                finalized = finalizer()
                cleanup_ok = validate_finalizer_resources(finalized)
                if cleanup_ok and isinstance(finalized, Mapping):
                    resources = finalized
                else:
                    rejection_code = "finalizer_top_level_fields"
            except Exception:  # noqa: BLE001 - fixed markers only
                workload_ok = False
                rejection_code = "finalizer_not_mapping"
        else:
            rejection_code = "finalizer_not_mapping"
    markers = _safe_markers(
        resources,
        workload_ok=workload_ok,
        cleanup_ok=cleanup_ok,
        rejection_code=rejection_code,
    )
    marker_errors = validate_load_stress_output("\n".join(f"{k}={v}" for k, v in markers))
    if marker_errors:
        return markers, 1
    return markers, 0 if markers[0][1] == "PASS" else 1


def main() -> int:
    markers, code = run_load_stress()
    for name, value in markers:
        print(f"{name}={value}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
