"""Structural checks for Phase-A profile attribution."""

from __future__ import annotations

from typing import cast

from scripts.sprint77_phase_a_profile import profile_cases


def test_profile_reports_framework_and_dependency_attribution() -> None:
    report = profile_cases(limit=4)
    profiles = cast(list[dict[str, object]], report["selected_cases"])
    names = {str(profile["name"]) for profile in profiles}
    assert {"trajectory_dtw", "density_geodesic", "activation_capture", "cem_planning"} <= names
    for profile in profiles:
        rows = cast(list[dict[str, object]], profile["top_cumulative"])
        assert rows
        assert all(float(cast(float, row["cumtime_seconds"])) >= 0.0 for row in rows)
