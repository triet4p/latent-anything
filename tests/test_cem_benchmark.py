from __future__ import annotations

from typing import cast

from scripts.cem_planning_benchmark import run_benchmark


def test_controlled_benchmark_shows_cem_improvement_and_model_bias() -> None:
    report = run_benchmark(population_size=64, iterations=5)
    result_rows = cast(list[dict[str, object]], report["results"])
    rows = {str(row["method"]): row for row in result_rows}
    assert report["acceptance"] == {
        "cem_model_return_gt_random": True,
        "cem_realized_return_gt_random": True,
        "cem_model_bias_is_reported": True,
    }
    assert float(cast(float, rows["cem"]["predicted_return"])) > float(cast(float, rows["cem"]["realized_return"]))
    assert float(cast(float, rows["cem"]["model_bias"])) > 0.0
    assert float(cast(float, rows["cem"]["realized_return"])) > float(
        cast(float, rows["fixed_zero"]["realized_return"])
    )
