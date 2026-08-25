"""Structural tests for Sprint 77 Phase-A benchmark evidence."""

from __future__ import annotations

from typing import cast

from scripts.sprint77_phase_a_benchmark import SCHEMA_VERSION, run_suite


def test_phase_a_suite_has_fixed_workloads_and_correctness_digests() -> None:
    report = run_suite(warmups=0, repetitions=2)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["workload_contract"] == {
        "offline": True,
        "seed": 771,
        "warmups": 0,
        "repetitions": 2,
        "default_ci_policy": "semantic digest checks only; latency is advisory and environment-scoped",
    }
    cases = cast(list[dict[str, object]], report["cases"])
    names = {str(case["name"]) for case in cases}
    assert {
        "geometry_distance",
        "trajectory_dtw",
        "density_geodesic",
        "activation_capture",
        "rollout",
        "cem_planning",
        "mppi_planning",
        "portable_encode_decode",
        "artifact_and_disk_cache",
        "bounded_streaming",
        "local_recorder",
        "plugin_listing",
        "lerobot_numpy_boundary",
    } <= names
    for case in cases:
        latency = cast(dict[str, object], case["latency_us"])
        median = float(cast(float, latency["median"]))
        p95 = float(cast(float, latency["p95"]))
        assert median >= 0.0
        assert p95 >= median
        assert len(str(case["correctness_digest"])) == 64


def test_phase_a_repeated_suite_is_semantically_reproducible() -> None:
    first = run_suite(warmups=0, repetitions=2)
    second = run_suite(warmups=0, repetitions=2)
    first_cases = cast(list[dict[str, object]], first["cases"])
    second_cases = cast(list[dict[str, object]], second["cases"])
    first_digests = {str(case["name"]): case["correctness_digest"] for case in first_cases}
    second_digests = {str(case["name"]): case["correctness_digest"] for case in second_cases}
    assert first_digests == second_digests
