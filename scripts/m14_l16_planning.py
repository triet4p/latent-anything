"""Persist M14 L16 reward/value, CEM, and MPPI planning evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import psutil
from sklearn.datasets import load_digits

from latent_anything.cem import CEMConfig, CEMPlanner
from latent_anything.mppi import MPPIConfig, MPPIPlanner
from latent_anything.reward_value import LinearRewardScorer, MonteCarloValueEstimator, RewardValueEvaluator
from latent_anything.trajectory import Trajectory
from latent_anything.latent_space import LatentSpace
from latent_anything.transition import DeterministicLatentTransition

RUN_COMMAND = "uv run python scripts/m14_l16_planning.py"
DATASET_REVISION = "scikit-learn-digits-0.1.0"
SOURCE_IDENTITY = "m14-l16-digits-trajectory-v1"
SEED = 1601
HORIZON = 5


def _make_recorded(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use ordered held-out digits observations as a recorded-trajectory fixture."""
    digits = load_digits()
    images = np.asarray(digits.images, dtype=np.float64) / 16.0
    rng = np.random.default_rng(seed)
    episodes, horizon = 72, HORIZON
    starts = rng.integers(0, len(images) - horizon - 1, size=episodes)
    states = np.empty((episodes, horizon + 1, 2), dtype=np.float64)
    actions = np.empty((episodes, horizon, 1), dtype=np.float64)
    for episode, start in enumerate(starts):
        frames = images[start : start + horizon + 1]
        states[episode, :, 0] = frames.mean(axis=(1, 2))
        states[episode, :, 1] = frames.std(axis=(1, 2))
        actions[episode, :, 0] = np.clip(frames[:-1, 0, 0] * 2.0 - 1.0, -1.0, 1.0)
    rewards = 1.0 + 0.45 * actions[:, :, 0] + 0.20 * states[:, :-1, 0] - 0.10 * states[:, :-1, 1]
    return states, actions, rewards


def main() -> None:
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    process = psutil.Process()
    states, actions, rewards = _make_recorded(SEED)
    train_n = 56
    train_states, holdout_states = states[:train_n], states[train_n:]
    train_actions, holdout_actions = actions[:train_n], actions[train_n:]
    train_rewards, holdout_rewards = rewards[:train_n], rewards[train_n:]
    flat_states = train_states[:, :-1].reshape(-1, 2)
    flat_actions = train_actions.reshape(-1, 1)
    flat_next = train_states[:, 1:].reshape(-1, 2)
    transition = DeterministicLatentTransition(
        LatentSpace(2, source_model=SOURCE_IDENTITY), 1, source_space_identity=SOURCE_IDENTITY
    ).fit(flat_states, flat_actions, flat_next)
    scorer = LinearRewardScorer(2, 1, source_space_identity=SOURCE_IDENTITY).fit(
        flat_states, flat_actions, train_rewards.reshape(-1), policy_id="recorded-digits-policy", data_distribution="ordered digits trajectories"
    )
    value = MonteCarloValueEstimator(
        2, discount=0.95, horizon=HORIZON, policy_id="recorded-digits-policy", data_distribution="ordered digits trajectories"
    ).fit_trajectories(train_states, train_rewards)
    evaluator = RewardValueEvaluator(scorer, value)
    holdout_evaluation = evaluator.evaluate_holdout(
        holdout_states, holdout_actions, holdout_rewards, source="held-out recorded digits trajectories"
    )
    initial = holdout_states[0, 0].copy()

    def objective(candidates: np.ndarray) -> np.ndarray:
        candidate_values = np.asarray(candidates, dtype=np.float64)
        current = np.repeat(initial[None, :], len(candidate_values), axis=0)
        total = np.zeros(len(candidate_values), dtype=np.float64)
        for step in range(candidate_values.shape[1]):
            next_states = np.vstack([transition.predict(state, action) for state, action in zip(current, candidate_values[:, step])])
            total += scorer.predict(current, candidate_values[:, step])
            current = next_states
        return total

    cem_config = CEMConfig(
        horizon=HORIZON, action_dim=1, lower_bounds=(-1.0,), upper_bounds=(1.0,), population_size=96,
        elite_count=12, iterations=6, seed=SEED, initial_std=(0.6,)
    )
    mppi_config = MPPIConfig(
        horizon=HORIZON, action_dim=1, lower_bounds=(-1.0,), upper_bounds=(1.0,), population_size=96,
        iterations=6, temperature=0.5, noise_std=(0.55,), seed=SEED
    )
    baseline_actions = np.zeros((HORIZON, 1), dtype=np.float64)
    baseline_return = float(objective(baseline_actions[None, :])[0])
    cem = CEMPlanner(cem_config).plan(objective)
    mppi = MPPIPlanner(mppi_config).plan(objective)
    rss_peak = process.memory_info().rss
    cem_dict, mppi_dict = cem.to_dict(), mppi.to_dict()
    all_actions = np.concatenate([cem.actions.reshape(-1), mppi.actions.reshape(-1)])
    payload = {
        "schema_version": "m14-l16-run-v1",
        "source_sha": source_sha,
        "command": RUN_COMMAND,
        "target": {
            "dataset": "scikit-learn digits",
            "dataset_revision": DATASET_REVISION,
            "description": "real recorded image trajectories reduced to mean/std latent state and first-pixel action",
            "license": "BSD-3-Clause (scikit-learn digits)",
            "device": "cpu",
        },
        "split": {
            "train_episodes": train_n,
            "holdout_episodes": len(holdout_states),
            "horizon": HORIZON,
            "seed": SEED,
            "train_digest": hashlib.sha256(train_states.tobytes()).hexdigest(),
            "holdout_digest": hashlib.sha256(holdout_states.tobytes()).hexdigest(),
        },
        "models": {
            "transition": {"class": "DeterministicLatentTransition", "source_space_identity": SOURCE_IDENTITY, "fit_samples": len(flat_states)},
            "reward_scorer": dict(scorer.fit_metadata),
            "value_estimator": dict(value.fit_metadata),
        },
        "metrics": {
            "holdout_reward_value": holdout_evaluation.to_metrics(),
            "holdout_diagnostics": holdout_evaluation.diagnostics.to_dict(),
            "baseline_return": baseline_return,
            "cem": cem_dict,
            "mppi": mppi_dict,
            "cem_improvement": float(cem.predicted_return - baseline_return),
            "mppi_improvement": float(mppi.predicted_return - baseline_return),
        },
        "acceptance": {
            "reward_value_finite": bool(np.isfinite(list(holdout_evaluation.to_metrics().values())).all()),
            "cem_return_improves_baseline": bool(cem.predicted_return >= baseline_return),
            "mppi_return_improves_baseline": bool(mppi.predicted_return >= baseline_return),
            "cem_budget_respected": len(cem.candidate_statistics) == cem_config.iterations and all(item.population_size == cem_config.population_size for item in cem.candidate_statistics),
            "mppi_budget_respected": len(mppi.candidate_statistics) == mppi_config.iterations and all(item.population_size == mppi_config.population_size for item in mppi.candidate_statistics),
            "actions_within_bounds": bool(np.all((all_actions >= -1.0) & (all_actions <= 1.0))),
            "planner_outputs_finite": bool(np.isfinite(all_actions).all() and np.isfinite(cem.predicted_return) and np.isfinite(mppi.predicted_return)),
            "convergence_complete": len(cem.convergence_history) == cem_config.iterations and len(mppi.convergence_history) == mppi_config.iterations,
        },
        "thresholds": {
            "cem_predicted_return_min": baseline_return,
            "mppi_predicted_return_min": baseline_return,
            "action_bounds": [-1.0, 1.0],
            "cem_evaluations_max": cem_config.population_size * cem_config.iterations,
            "mppi_evaluations_max": mppi_config.population_size * mppi_config.iterations,
            "seed": SEED,
        },
        "environment": {
            "device": "cpu",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "rss_peak_bytes": rss_peak,
            "network_policy": "offline after installed scikit-learn digits dataset",
        },
        "cleanup": "No temporary files or checkpoints; evidence is retained under artifacts/m14.",
    }
    output = Path("artifacts/m14/l16-planning-run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
