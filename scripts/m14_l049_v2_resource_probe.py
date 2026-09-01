"""Minimal CUDA resource probe for the L04.9 v2 runtime boundary.

This command deliberately does not load a model, read a fixture, score a
candidate, or access holdout material.  It exercises only the production
``ResourceTracker`` around one optional CUDA allocation and emits a fixed,
sanitized marker set for a later owner-authorized remote run.
"""

# This standalone probe intentionally composes private runtime seams; it is
# not part of the public package surface.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any

from scripts._m14_l049_v2_real_runtime import ResourceTracker, _new_counters, _runtime_resources
from scripts._m14_l049_v2_schema import FINALIZER_REJECTION_CODES
from scripts._m14_l049_v2_stage_a import _finalizer_rejection_code

_MEASUREMENT_STATUSES = frozenset(("available", "unavailable"))
_MEASUREMENT_REASONS = frozenset(
    (
        "cuda_unavailable",
        "cuda_reset_failed",
        "cuda_peak_query_failed",
        "cuda_zero_peak",
        "rss_unavailable",
        "clock_invalid",
        "tracker_unstarted",
        "resource_measurement_invalid",
    )
)
_PROBE_MARKER_NAMES = (
    "L049_V2_RESOURCE_PROBE_STATUS",
    "L049_V2_RESOURCE_PROBE_FINALIZER_CODE",
    "L049_V2_RESOURCE_PROBE_MEASUREMENT_STATUS",
    "L049_V2_RESOURCE_PROBE_MEASUREMENT_REASON",
    "L049_V2_RESOURCE_PROBE_CPU_PROVENANCE",
    "L049_V2_RESOURCE_PROBE_GPU_PROVENANCE",
    "L049_V2_RESOURCE_PROBE_DEVICE_CANONICAL",
    "L049_V2_RESOURCE_PROBE_CLEANUP",
)


def validate_resource_probe_output(output: str) -> list[str]:
    """Validate the probe's fixed-marker grammar without reading free text."""
    lines = output.splitlines()
    if len(lines) != len(_PROBE_MARKER_NAMES):
        return ["resource probe marker count is invalid"]
    records: dict[str, str] = {}
    for line in lines:
        name, separator, value = line.partition("=")
        if not separator or name not in _PROBE_MARKER_NAMES or name in records:
            return ["resource probe marker names are invalid"]
        records[name] = value
    if tuple(records) != _PROBE_MARKER_NAMES:
        return ["resource probe marker order is invalid"]
    if records["L049_V2_RESOURCE_PROBE_STATUS"] not in {"PASS", "FAIL"}:
        return ["resource probe status is invalid"]
    code = records["L049_V2_RESOURCE_PROBE_FINALIZER_CODE"]
    if code != "NONE" and code not in FINALIZER_REJECTION_CODES:
        return ["resource probe finalizer code is invalid"]
    marker_status = records["L049_V2_RESOURCE_PROBE_STATUS"]
    measurement_status = records["L049_V2_RESOURCE_PROBE_MEASUREMENT_STATUS"]
    reason = records["L049_V2_RESOURCE_PROBE_MEASUREMENT_REASON"]
    if (marker_status == "PASS") != (code == "NONE"):
        return ["resource probe status and finalizer code disagree"]
    if measurement_status not in {"available", "unavailable", "unknown"}:
        return ["resource probe measurement status is invalid"]
    if reason not in _MEASUREMENT_REASONS and reason not in {"none", "unknown"}:
        return ["resource probe measurement reason is invalid"]
    for field in ("CPU_PROVENANCE", "GPU_PROVENANCE", "DEVICE_CANONICAL"):
        if records[f"L049_V2_RESOURCE_PROBE_{field}"] not in {"true", "false"}:
            return ["resource probe provenance is invalid"]
    if records["L049_V2_RESOURCE_PROBE_CLEANUP"] != "PASS":
        return ["resource probe cleanup is invalid"]
    if measurement_status == "unknown" or reason == "unknown":
        if marker_status != "FAIL" or code == "NONE":
            return ["unknown resource probe metadata must fail closed"]
    elif measurement_status == "available":
        if reason != "none":
            return ["available resource probe status requires no measurement reason"]
        if marker_status == "PASS" and (
            records["L049_V2_RESOURCE_PROBE_CPU_PROVENANCE"] != "true"
            or records["L049_V2_RESOURCE_PROBE_GPU_PROVENANCE"] != "true"
            or records["L049_V2_RESOURCE_PROBE_DEVICE_CANONICAL"] != "true"
        ):
            return ["available resource probe status has inconsistent provenance"]
    elif reason == "none":
        return ["unavailable resource probe status requires a measurement reason"]
    if reason in {"cuda_unavailable", "cuda_reset_failed", "cuda_peak_query_failed", "cuda_zero_peak"} and (
        records["L049_V2_RESOURCE_PROBE_GPU_PROVENANCE"] != "false"
    ):
        return ["CUDA-unavailable resource probe status has inconsistent GPU provenance"]
    if reason in {"tracker_unstarted", "resource_measurement_invalid"} and (
        records["L049_V2_RESOURCE_PROBE_CPU_PROVENANCE"] != "false"
        or records["L049_V2_RESOURCE_PROBE_GPU_PROVENANCE"] != "false"
        or records["L049_V2_RESOURCE_PROBE_DEVICE_CANONICAL"] != "false"
    ):
        return ["fully unavailable resource probe status has inconsistent provenance"]
    if reason == "rss_unavailable" and records["L049_V2_RESOURCE_PROBE_CPU_PROVENANCE"] != "false":
        return ["RSS-unavailable resource probe status has inconsistent CPU provenance"]
    return []


def run_resource_probe(
    torch_module: Any | None = None,
    *,
    resource_module: Any | None = None,
    psutil_module: Any | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Run the resource-only probe and return resources plus its safe code."""
    tracker = ResourceTracker(
        torch_module=torch_module,
        resource_module=resource_module,
        psutil_module=psutil_module,
    )
    tracker.start()
    try:
        cuda = getattr(tracker._torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            allocation = tracker._torch.empty((1,), device="cuda")
            cuda.synchronize(int(cuda.current_device()))
            del allocation
    except Exception:  # noqa: BLE001 - the tracker publishes sanitized status
        pass
    finally:
        tracker.finish()
    resources = _runtime_resources(_new_counters(), tracker)
    return resources, _finalizer_rejection_code(resources)


def emit_resource_probe(resources: dict[str, Any], rejection_code: str | None) -> None:
    """Emit only fixed markers and allowlisted provenance booleans."""
    peak = resources["resource_peak"]
    status = peak.get("measurement_status")
    safe_status = status if isinstance(status, str) and status in _MEASUREMENT_STATUSES else "unknown"
    reason = peak.get("measurement_reason")
    safe_reason = (
        reason
        if isinstance(reason, str) and reason in _MEASUREMENT_REASONS
        else "none"
        if reason is None
        else "unknown"
    )
    safe_rejection = (
        rejection_code
        if isinstance(rejection_code, str) and rejection_code in FINALIZER_REJECTION_CODES
        else "finalizer_not_mapping"
    )
    unknown_metadata = safe_status == "unknown" or safe_reason == "unknown"
    effective_rejection = rejection_code
    if effective_rejection is None and unknown_metadata:
        effective_rejection = "finalizer_resource_peak_status_reason"
    cpu_provenance = peak["cpu_source"] != "unavailable"
    gpu_provenance = peak["gpu_source"] != "unavailable" and peak["gpu_reserved_source"] != "unavailable"
    device = peak["gpu_device"]
    canonical_device = device == "unavailable" or (
        isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit()
    )
    markers = (
        ("L049_V2_RESOURCE_PROBE_STATUS", "PASS" if effective_rejection is None else "FAIL"),
        ("L049_V2_RESOURCE_PROBE_FINALIZER_CODE", "NONE" if effective_rejection is None else safe_rejection),
        ("L049_V2_RESOURCE_PROBE_MEASUREMENT_STATUS", safe_status),
        ("L049_V2_RESOURCE_PROBE_MEASUREMENT_REASON", safe_reason),
        ("L049_V2_RESOURCE_PROBE_CPU_PROVENANCE", str(cpu_provenance).lower()),
        ("L049_V2_RESOURCE_PROBE_GPU_PROVENANCE", str(gpu_provenance).lower()),
        ("L049_V2_RESOURCE_PROBE_DEVICE_CANONICAL", str(canonical_device).lower()),
        ("L049_V2_RESOURCE_PROBE_CLEANUP", "PASS"),
    )
    for name, value in markers:
        print(f"{name}={value}")


def main() -> int:
    resources, rejection_code = run_resource_probe()
    emit_resource_probe(resources, rejection_code)
    return 0 if rejection_code is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
