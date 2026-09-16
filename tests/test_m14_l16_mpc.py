from __future__ import annotations

from scripts.m14_l16_mpc import run_benchmark


def test_mpc_receding_horizon_beats_fixed_baseline_and_respects_controls() -> None:
    report = run_benchmark(population_size=16, iterations=2)

    assert report["record_id"] == "THY-T07-MODEL-PREDICTIVE-CONTROL-MPC"
    assert report["accepted"] is True
    acceptance = report["acceptance"]
    assert acceptance == {
        "finite_returns": True,
        "mpc_improves_fixed_zero": True,
        "receding_horizon_replans": True,
        "executed_prefix_only": True,
        "actions_within_bounds": True,
        "sample_budget_respected": True,
        "state_trace_finite": True,
    }
    metrics = report["metrics"]
    assert float(metrics["mpc_return"]) > float(metrics["fixed_zero_return"])  # type: ignore[index]
    assert int(metrics["replans"]) == 3  # type: ignore[index]
    assert int(metrics["executed_steps"]) == 3  # type: ignore[index]
