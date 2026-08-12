"""Regression tests for the reward/value reproduction script."""

from __future__ import annotations

from scripts.reward_value_benchmark import run_benchmark


def test_reward_value_benchmark_runs_with_source_bound_trajectories() -> None:
    result = run_benchmark(episodes=16, horizon=2)

    assert result["source_space_identity"] == "synthetic-reward-value-system-v1"
    assert result["real_trajectory"]["valid_steps"] == 2  # type: ignore[index]
    assert result["imagined_trajectory"]["valid_steps"] == 2  # type: ignore[index]
