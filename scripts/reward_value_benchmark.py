"""Sprint 67 held-out reward/value and imagined-trajectory benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latent_anything import (
    LatentSpace,
    LinearRewardScorer,
    MonteCarloValueEstimator,
    RewardValueEvaluator,
    RolloutPipeline,
    Trajectory,
)
from latent_anything.transition import DeterministicLatentTransition

DEFAULT_OUTPUT = Path("artifacts/reward_value_evaluation.json")
DEFAULT_CONFIG_OUTPUT = Path("artifacts/reward_value_evaluation_config.json")


def _generate_dataset(*, seed: int, episodes: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.empty((episodes, horizon + 1, 2), dtype=np.float64)
    actions = rng.normal(scale=0.35, size=(episodes, horizon, 1))
    states[:, 0] = rng.normal(scale=0.7, size=(episodes, 2))
    for step in range(horizon):
        states[:, step + 1, 0] = (
            0.82 * states[:, step, 0] + 0.28 * actions[:, step, 0] + rng.normal(scale=0.02, size=episodes)
        )
        states[:, step + 1, 1] = 0.9 * states[:, step, 1] + rng.normal(scale=0.02, size=episodes)
    rewards = states[:, :-1, 0] + 0.4 * actions[:, :, 0]
    return states, actions, rewards


def run_benchmark(*, seed: int = 67, episodes: int = 96, horizon: int = 8) -> dict[str, object]:
    """Fit on one split and score the held-out split plus imagination bias."""

    if episodes < 16 or horizon < 2:
        raise ValueError("episodes must be >= 16 and horizon must be >= 2")
    states, actions, rewards = _generate_dataset(seed=seed, episodes=episodes, horizon=horizon)
    split = max(8, min(episodes - 4, int(episodes * 0.75)))
    identity = "synthetic-reward-value-system-v1"
    discount = 0.85

    reward_scorer = LinearRewardScorer(2, 1, source_space_identity=identity, ridge=1e-8)
    reward_scorer.fit(
        states[:split, :-1].reshape(-1, 2),
        actions[:split].reshape(-1, 1),
        rewards[:split].reshape(-1),
        policy_id="gaussian-behavior-v1",
        data_distribution="synthetic-held-out-dynamics",
    )
    value_estimator = MonteCarloValueEstimator(
        2,
        discount=discount,
        horizon=horizon,
        policy_id="gaussian-behavior-v1",
        data_distribution="synthetic-held-out-dynamics",
        ridge=1e-8,
    )
    value_estimator.fit_trajectories(states[:split], rewards[:split])
    evaluator = RewardValueEvaluator(reward_scorer, value_estimator)
    holdout = evaluator.evaluate_holdout(states[split:], actions[split:], rewards[split:], source="real-holdout")

    transition = DeterministicLatentTransition(
        LatentSpace(2, source_model="synthetic-reward-value-system"),
        1,
        source_space_identity=identity,
        ridge=1e-8,
    ).fit(
        states[:split, :-1].reshape(-1, 2),
        actions[:split].reshape(-1, 1),
        states[:split, 1:].reshape(-1, 2),
    )
    pipeline = RolloutPipeline(transition, evaluator=evaluator)
    imagined_result = pipeline.run(states[split, 0], actions[split])
    imagined = imagined_result.evaluation
    if imagined is None:
        raise RuntimeError("configured rollout evaluator did not produce an imagined score")
    real = evaluator.evaluate(
        Trajectory(
            states[split],
            metadata={"state_source": "observed", "source_space_identity": identity},
        ),
        actions[split],
        source="real",
    )
    comparison = evaluator.compare_real_imagined(
        Trajectory(
            states[split],
            metadata={"state_source": "observed", "source_space_identity": identity},
        ),
        imagined_result.trajectory,
        actions[split],
    )
    return {
        "evidence_status": "D2",
        "benchmark": "reward_value_evaluation",
        "seed": seed,
        "source_space_identity": identity,
        "episodes": episodes,
        "horizon": horizon,
        "train_episodes": split,
        "test_episodes": episodes - split,
        "discount": discount,
        "policy_id": value_estimator.policy_id,
        "data_distribution": value_estimator.data_distribution,
        "heldout": holdout.diagnostics.to_dict(),
        "real_trajectory": real.to_metrics(),
        "imagined_trajectory": imagined.to_metrics(),
        "real_vs_imagined": comparison.to_dict(),
        "failure_analysis": [
            "This is a controlled synthetic D2 benchmark, not evidence on a real pretrained world model.",
            "The reward and value heads are linear NumPy baselines; nonlinear and distributional "
            "heads remain future work.",
            "Value targets are finite-horizon Monte-Carlo returns under the declared behavior policy and discount.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    args = parser.parse_args()
    result = run_benchmark(seed=args.seed, episodes=args.episodes, horizon=args.horizon)
    config = {
        "benchmark": result["benchmark"],
        "seed": result["seed"],
        "episodes": result["episodes"],
        "horizon": result["horizon"],
        "discount": result["discount"],
        "train_fraction": 0.75,
        "reward_head": "linear_state_action",
        "value_head": "linear_monte_carlo",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
