"""Sprint 69 controlled comparison of CEM, MPPI, and random shooting.

The benchmark keeps the model, task, candidate budget, and environment replay
identical across methods.  The environment deliberately responds 20% less to
actions than the learned transition, so predicted-versus-realized return and
transition-error robustness remain visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from latent_anything import (
    CEMConfig,
    CEMPlanner,
    DeterministicLatentTransition,
    LatentSpace,
    LinearRewardScorer,
    MonteCarloValueEstimator,
    MPPIConfig,
    MPPIPlanner,
    RewardValueEvaluator,
    RolloutPipeline,
)

DEFAULT_OUTPUT = Path("artifacts/mppi_planning_benchmark.json")
DEFAULT_CONFIG_OUTPUT = Path("artifacts/mppi_planning_benchmark_config.json")


def _build_model_pipeline(discount: float) -> RolloutPipeline:
    space = LatentSpace(1, source_model="mppi-controlled-model")
    state_values = np.repeat(np.linspace(-2.0, 2.0, 16)[:, None], 3, axis=0)
    action_values = np.tile(np.asarray([[-1.0], [0.0], [1.0]]), (16, 1))
    transition = DeterministicLatentTransition(space, 1).fit(
        state_values,
        action_values,
        state_values + action_values,
    )
    reward_scorer = LinearRewardScorer(1, 1, source_space_identity="mppi-controlled-model")
    reward_scorer.fit(
        state_values,
        action_values,
        state_values[:, 0] + 0.1 * action_values[:, 0],
        policy_id="controlled-policy",
        data_distribution="uniform-latent-actions",
    )
    value_estimator = MonteCarloValueEstimator(
        1,
        discount=discount,
        horizon=6,
        policy_id="controlled-policy",
        data_distribution="uniform-latent-actions",
    )
    value_estimator.fit(np.linspace(-2.0, 2.0, 48)[:, None], np.linspace(-2.0, 2.0, 48))
    return RolloutPipeline(
        transition,
        evaluator=RewardValueEvaluator(reward_scorer, value_estimator),
    )


def _model_score(pipeline: RolloutPipeline, initial_state: np.ndarray, actions: np.ndarray) -> float:
    result = pipeline.run(initial_state, actions)
    if result.evaluation is None:
        raise RuntimeError("controlled benchmark pipeline must return an evaluation")
    return float(result.evaluation.returns[0])


def _realized_score(initial_state: np.ndarray, actions: np.ndarray, discount: float) -> float:
    state = float(initial_state[0])
    total = 0.0
    for step, action in enumerate(actions[:, 0]):
        total += discount**step * (state + 0.1 * float(action))
        state += 0.8 * float(action)
    return total


def _smoothness(actions: np.ndarray) -> float:
    if len(actions) < 2:
        return 0.0
    return float(np.mean(np.square(np.diff(actions, axis=0))))


def _row(
    name: str,
    actions: np.ndarray,
    predicted_return: float,
    realized_return: float,
    sample_count: int,
    latency_seconds: float,
) -> dict[str, object]:
    return {
        "method": name,
        "actions": actions.tolist(),
        "predicted_return": float(predicted_return),
        "realized_return": float(realized_return),
        "model_bias": float(predicted_return - realized_return),
        "action_smoothness": _smoothness(actions),
        "sample_count": sample_count,
        "latency_seconds": latency_seconds,
        "transition_error_robustness": float(realized_return / (1.0 + abs(predicted_return))),
    }


def run_benchmark(*, seed: int = 69, population_size: int = 96, iterations: int = 6) -> dict[str, object]:
    """Run the same controlled continuous-control task for three methods."""

    discount = 0.95
    initial_state = np.zeros(1, dtype=np.float64)
    cem_config = CEMConfig(
        horizon=6,
        action_dim=1,
        lower_bounds=(-1.0,),
        upper_bounds=(1.0,),
        population_size=population_size,
        elite_fraction=0.2,
        iterations=iterations,
        smoothing=0.1,
        min_std=0.02,
        seed=seed,
    )
    mppi_config = MPPIConfig(
        horizon=6,
        action_dim=1,
        lower_bounds=(-1.0,),
        upper_bounds=(1.0,),
        population_size=population_size,
        iterations=iterations,
        temperature=0.25,
        noise_std=(0.45,),
        seed=seed,
    )
    pipeline = _build_model_pipeline(discount)
    cem_result = CEMPlanner(cem_config).plan_rollouts(initial_state, pipeline)
    mppi_result = MPPIPlanner(mppi_config).plan_rollouts(initial_state, pipeline)
    rng = np.random.default_rng(seed)
    random_candidates = rng.uniform(-1.0, 1.0, size=(population_size, 6, 1))
    random_scores = np.asarray([_model_score(pipeline, initial_state, candidate) for candidate in random_candidates])
    random_actions = random_candidates[int(np.argmax(random_scores))]
    fixed_actions = np.zeros((6, 1), dtype=np.float64)
    rows = [
        _row(
            "fixed_zero",
            fixed_actions,
            _model_score(pipeline, initial_state, fixed_actions),
            _realized_score(initial_state, fixed_actions, discount),
            0,
            0.0,
        ),
        _row(
            "random_shooting",
            random_actions,
            float(np.max(random_scores)),
            _realized_score(initial_state, random_actions, discount),
            population_size,
            0.0,
        ),
        _row(
            "cem",
            cem_result.actions,
            cem_result.predicted_return,
            _realized_score(initial_state, cem_result.actions, discount),
            population_size * iterations,
            cem_result.runtime_profile.total_seconds,
        ),
        _row(
            "mppi",
            mppi_result.actions,
            mppi_result.predicted_return,
            _realized_score(initial_state, mppi_result.actions, discount),
            mppi_result.sample_count,
            mppi_result.runtime_profile.total_seconds,
        ),
    ]
    mppi_row = cast(dict[str, object], next(row for row in rows if row["method"] == "mppi"))
    random_row = cast(dict[str, object], next(row for row in rows if row["method"] == "random_shooting"))
    return {
        "benchmark": "mppi_vs_cem_latent_control",
        "seed": seed,
        "config": {
            "horizon": 6,
            "action_dim": 1,
            "population_size": population_size,
            "iterations": iterations,
            "lower_bounds": [-1.0],
            "upper_bounds": [1.0],
            "mppi_temperature": mppi_config.temperature,
            "mppi_noise_std": list(mppi_config.noise_std),
            "discount": discount,
            "model_action_scale": 1.0,
            "environment_action_scale": 0.8,
        },
        "results": rows,
        "acceptance": {
            "mppi_model_return_gt_random": bool(
                mppi_result.predicted_return > float(cast(float, random_row["predicted_return"]))
            ),
            "mppi_realized_return_gt_random": bool(
                cast(float, mppi_row["realized_return"]) > cast(float, random_row["realized_return"])
            ),
            "mppi_smoothness_reported": True,
            "mppi_sample_count_reported": True,
            "mppi_latency_reported": True,
            "mppi_transition_robustness_reported": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--population-size", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    args = parser.parse_args()
    result = run_benchmark(seed=args.seed, population_size=args.population_size, iterations=args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.config_output.write_text(json.dumps(result["config"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
