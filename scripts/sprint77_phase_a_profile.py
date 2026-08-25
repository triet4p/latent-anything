#!/usr/bin/env python3
"""Collect cProfile attribution for Sprint 77 framework-owned workloads."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
from pathlib import Path

try:
    from scripts.sprint77_phase_a_benchmark import build_cases
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from sprint77_phase_a_benchmark import build_cases


def profile_cases(*, limit: int = 12, output: Path | None = None) -> dict[str, object]:
    """Profile selected representative operations and return top cumulative rows."""

    selected = {
        "trajectory_dtw",
        "density_geodesic",
        "activation_capture",
        "cem_planning",
        "mppi_planning",
        "portable_encode_decode",
        "artifact_and_disk_cache",
        "local_recorder",
    }
    profiles: list[dict[str, object]] = []
    for case in build_cases():
        if case.name not in selected:
            continue
        profiler = cProfile.Profile()
        profiler.enable()
        case.operation()
        profiler.disable()
        stats = pstats.Stats(profiler)
        rows: list[dict[str, object]] = []
        raw_stats = getattr(stats, "stats", {})
        for (filename, line, function), (primitive_calls, calls, total, cumulative, callers) in sorted(
            raw_stats.items(), key=lambda item: item[1][3], reverse=True
        )[:limit]:
            del callers
            rows.append(
                {
                    "location": f"{filename}:{line}({function})",
                    "primitive_calls": primitive_calls,
                    "calls": calls,
                    "tottime_seconds": total,
                    "cumtime_seconds": cumulative,
                }
            )
        profiles.append({"name": case.name, "attribution": case.attribution, "top_cumulative": rows})
    report: dict[str, object] = {
        "schema_version": "sprint77-phase-a-profile-v1",
        "profile": "cProfile cumulative attribution",
        "selected_cases": profiles,
        "interpretation": {
            "framework_owned": (
                "Functions under src/latent_anything are framework time; NumPy, Torch, "
                "SQLite, and filesystem rows are dependencies or I/O and are reported separately."
            ),
            "rust_decision": "This artifact is evidence only; it does not decide the Rust task.",
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint77_phase_a_profile.json"))
    args = parser.parse_args()
    report = profile_cases(limit=args.limit, output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
