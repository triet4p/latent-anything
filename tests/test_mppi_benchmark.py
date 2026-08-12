from __future__ import annotations

from typing import cast

from scripts.mppi_planning_benchmark import run_benchmark


def test_controlled_benchmark_compares_mppi_cem_and_random_with_evidence() -> None:
    report = run_benchmark(population_size=64, iterations=5)
    result_rows = cast(list[dict[str, object]], report["results"])
    rows = {str(row["method"]): row for row in result_rows}

    assert report["acceptance"] == {
        "mppi_model_return_gt_random": True,
        "mppi_realized_return_gt_random": True,
        "mppi_smoothness_reported": True,
        "mppi_sample_count_reported": True,
        "mppi_latency_reported": True,
        "mppi_transition_robustness_reported": True,
    }
    assert {"fixed_zero", "random_shooting", "cem", "mppi"} == set(rows)
    for row in rows.values():
        assert float(cast(float, row["action_smoothness"])) >= 0.0
        assert int(cast(int, row["sample_count"])) >= 0
        assert float(cast(float, row["latency_seconds"])) >= 0.0
        assert float(cast(float, row["transition_error_robustness"])) == float(
            cast(float, row["realized_return"]) / (1.0 + abs(cast(float, row["predicted_return"])))
        )
