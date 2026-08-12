"""Sprint 68 controlled CEM planning benchmark.

The benchmark compares fixed actions, random shooting, and CEM on a one-
dimensional latent-control task.  The learned model deliberately uses a
stronger action response than the environment, so model-space optimization and
environment-realized return are reported separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from latent_anything import (
    DeterministicLatentTransition,
    LatentSpace,
    LinearRewardScorer,
    MonteCarloValueEstimator,
    RewardValueEvaluator,
    RolloutPipeline,
)
from latent_anything.cem import CEMConfig, CEMPlanner

DEFAULT_OUTPUT = Path("artifacts/cem_planning_benchmark.json")
DEFAULT_CONFIG_OUTPUT = Path("artifacts/cem_planning_benchmark_config.json")


def _build_model_pipeline(discount: float) -> RolloutPipeline:
    space = LatentSpace(1, source_model="cem-controlled-model")
    states = np.linspace(-2.0, 2.0, 48)[:, None]
    action_values = np.tile(np.asarray([[-1.0], [0.0], [1.0]]), (16, 1))
    state_values = np.repeat(states[:16], 3, axis=0)
    transition = DeterministicLatentTransition(space, 1).fit(
        state_values,
        action_values,
        state_values + action_values,
    )
    reward_scorer = LinearRewardScorer(1, 1, source_space_identity="cem-controlled-model")
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
    value_estimator.fit(states, np.zeros(len(states)))
    evaluator = RewardValueEvaluator(reward_scorer, value_estimator)
    return RolloutPipeline(transition, evaluator=evaluator)


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


def run_benchmark(*, seed: int = 68, population_size: int = 96, iterations: int = 6) -> dict[str, object]:
    """Run the controlled comparison and return JSON-serializable evidence."""

    discount = 0.95
    initial_state = np.zeros(1, dtype=np.float64)
    config = CEMConfig(
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
    pipeline = _build_model_pipeline(discount)
    planner = CEMPlanner(config)
    cem_result = planner.plan_rollouts(initial_state, pipeline)
    rng = np.random.default_rng(seed)
    random_candidates = rng.uniform(-1.0, 1.0, size=(population_size, config.horizon, config.action_dim))
    random_scores = np.asarray([_model_score(pipeline, initial_state, candidate) for candidate in random_candidates])
    random_actions = random_candidates[int(np.argmax(random_scores))]
    fixed_actions = np.zeros((config.horizon, config.action_dim), dtype=np.float64)

    rows: list[dict[str, object]] = []
    for name, actions, predicted in (
        ("fixed_zero", fixed_actions, _model_score(pipeline, initial_state, fixed_actions)),
        ("random_shooting", random_actions, float(np.max(random_scores))),
        ("cem", cem_result.actions, cem_result.predicted_return),
    ):
        realized = _realized_score(initial_state, actions, discount)
        rows.append(
            {
                "method": name,
                "actions": actions.tolist(),
                "predicted_return": float(predicted),
                "realized_return": realized,
                "model_bias": float(predicted - realized),
            }
        )
    cem_row = cast(dict[str, object], next(row for row in rows if row["method"] == "cem"))
    random_row = cast(dict[str, object], next(row for row in rows if row["method"] == "random_shooting"))
    return {
        "benchmark": "cem_latent_control",
        "seed": seed,
        "config": {
            "horizon": config.horizon,
            "action_dim": config.action_dim,
            "population_size": config.population_size,
            "elite_count": config.resolved_elite_count,
            "elite_fraction": config.elite_fraction,
            "iterations": config.iterations,
            "smoothing": config.smoothing,
            "min_std": config.min_std,
            "lower_bounds": list(config.lower_bounds),
            "upper_bounds": list(config.upper_bounds),
            "discount": discount,
            "model_action_scale": 1.0,
            "environment_action_scale": 0.8,
        },
        "results": rows,
        "cem_convergence_history": list(cem_result.convergence_history),
        "cem_runtime_profile": {
            "total_seconds": cem_result.runtime_profile.total_seconds,
            "stage_totals": cem_result.runtime_profile.stage_totals(),
        },
        "acceptance": {
            "cem_model_return_gt_random": bool(
                cem_result.predicted_return > float(cast(float, random_row["predicted_return"]))
            ),
            "cem_realized_return_gt_random": bool(
                cast(float, cem_row["realized_return"]) > cast(float, random_row["realized_return"])
            ),
            "cem_model_bias_is_reported": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=68)
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
