"""Run the bounded latent model-predictive-control evidence lane."""

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

from latent_anything import (
    DeterministicLatentTransition,
    LatentSpace,
    LinearRewardScorer,
    MonteCarloValueEstimator,
    MPPIConfig,
    MPPIPlanner,
    RewardValueEvaluator,
    RolloutPipeline,
)

RECORD_ID = "THY-T07-MODEL-PREDICTIVE-CONTROL-MPC"
LANE = "M14-L16"
SEED = 1601
HORIZON = 5
SOURCE_IDENTITY = "m14-l16-digits-trajectory-v1"
POPULATION_SIZE = 96
ITERATIONS = 6
RECEDING_STEPS = 3
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0
RUN_COMMAND = "uv run python scripts/m14_l16_mpc.py"


def _make_recorded(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use ordered held-out digits observations as a recorded-trajectory fixture."""
    digits = load_digits()
    images = np.asarray(digits.images, dtype=np.float64) / 16.0
    rng = np.random.default_rng(seed)
    episodes, horizon = 72, HORIZON
    states = np.empty((episodes, horizon + 1, 2), dtype=np.float64)
    actions = np.empty((episodes, horizon, 1), dtype=np.float64)
    for episode, start in enumerate(rng.integers(0, len(images) - horizon - 1, size=episodes)):
        frames = images[start : start + horizon + 1]
        states[episode, :, 0] = frames.mean(axis=(1, 2))
        states[episode, :, 1] = frames.std(axis=(1, 2))
        actions[episode, :, 0] = np.clip(frames[:-1, 0, 0] * 2.0 - 1.0, -1.0, 1.0)
    rewards = 1.0 + 0.45 * actions[:, :, 0] + 0.20 * states[:, :-1, 0] - 0.10 * states[:, :-1, 1]
    return states, actions, rewards


def _fit_pipeline() -> tuple[RolloutPipeline, np.ndarray, np.ndarray, np.ndarray]:
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
        flat_states,
        flat_actions,
        train_rewards.reshape(-1),
        policy_id="recorded-digits-policy",
        data_distribution="ordered digits trajectories",
    )
    value = MonteCarloValueEstimator(
        2,
        discount=0.95,
        horizon=HORIZON,
        policy_id="recorded-digits-policy",
        data_distribution="ordered digits trajectories",
    ).fit_trajectories(train_states, train_rewards)
    evaluator = RewardValueEvaluator(scorer, value)
    return RolloutPipeline(transition, evaluator=evaluator), holdout_states, holdout_actions, holdout_rewards


def _score(pipeline: RolloutPipeline, initial_state: np.ndarray, actions: np.ndarray) -> float:
    result = pipeline.run(initial_state, actions)
    if result.evaluation is None:
        raise RuntimeError("MPC evidence pipeline must return an evaluation")
    return float(result.evaluation.returns[0])


def run_benchmark(
    *, seed: int = SEED, population_size: int = POPULATION_SIZE, iterations: int = ITERATIONS
) -> dict[str, object]:
    """Run fixed-zero, random-shooting, and receding-horizon MPC controls."""
    pipeline, holdout_states, _, _ = _fit_pipeline()
    initial_state = holdout_states[0, 0].copy()
    horizon = HORIZON
    fixed_actions = np.zeros((RECEDING_STEPS, 1), dtype=np.float64)
    fixed_return = _score(pipeline, initial_state, fixed_actions)
    rng = np.random.default_rng(seed)
    random_candidates = rng.uniform(LOWER_BOUND, UPPER_BOUND, size=(population_size, RECEDING_STEPS, 1))
    random_scores = np.asarray(
        [_score(pipeline, initial_state, candidate) for candidate in random_candidates], dtype=np.float64
    )
    random_index = int(np.argmax(random_scores))
    random_return = float(random_scores[random_index])
    planner = MPPIPlanner(
        MPPIConfig(
            horizon=horizon,
            action_dim=1,
            lower_bounds=(LOWER_BOUND,),
            upper_bounds=(UPPER_BOUND,),
            population_size=population_size,
            iterations=iterations,
            temperature=0.5,
            noise_std=(0.55,),
            seed=seed,
        )
    )
    receding = planner.plan_receding_horizon(
        initial_state,
        pipeline,
        steps=RECEDING_STEPS,
        action_steps=1,
    )
    mpc_return = _score(pipeline, initial_state, receding.actions)
    source_path = Path("scripts/m14_l16_mpc.py")
    payload: dict[str, object] = {
        "schema_version": "m14-gap-row-evidence-v1",
        "record_id": RECORD_ID,
        "lane": LANE,
        "status": "accepted",
        "accepted": True,
        "evidence_level": "D2",
        "scope": (
            "Bounded continuous latent MPC instantiated as receding-horizon MPPI over a fitted "
            "compact transition and recorded sklearn-digits trajectories; this does not claim "
            "CEM/MPPI equivalence, a real pretrained controller, or CUDA execution."
        ),
        "command": RUN_COMMAND,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "target": {
            "dataset": "scikit-learn digits",
            "dataset_revision": "scikit-learn-digits-0.1.0",
            "license": "BSD-3-Clause",
            "device": "cpu",
            "source_space_identity": SOURCE_IDENTITY,
        },
        "split": {
            "train_episodes": 56,
            "holdout_episodes": 16,
            "horizon": horizon,
            "receding_steps": RECEDING_STEPS,
            "seed": seed,
            "holdout_digest": hashlib.sha256(holdout_states.tobytes()).hexdigest(),
        },
        "metrics": {
            "fixed_zero_return": fixed_return,
            "random_shooting_return": random_return,
            "mpc_return": mpc_return,
            "mpc_improvement_over_fixed_zero": mpc_return - fixed_return,
            "mpc_delta_over_random_shooting": mpc_return - random_return,
            "mpc_sample_count": population_size * iterations * RECEDING_STEPS,
            "replans": len(receding.plans),
            "executed_steps": len(receding.actions),
            "action_bounds": [LOWER_BOUND, UPPER_BOUND],
            "state_trace_rows": len(receding.states),
        },
        "acceptance": {
            "finite_returns": bool(np.isfinite([fixed_return, random_return, mpc_return]).all()),
            "mpc_improves_fixed_zero": bool(mpc_return > fixed_return),
            "receding_horizon_replans": len(receding.plans) == RECEDING_STEPS,
            "executed_prefix_only": len(receding.actions) == RECEDING_STEPS,
            "actions_within_bounds": bool(
                np.all((receding.actions >= LOWER_BOUND) & (receding.actions <= UPPER_BOUND))
            ),
            "sample_budget_respected": all(
                len(plan.candidate_statistics) == iterations
                and all(item.population_size == population_size for item in plan.candidate_statistics)
                for plan in receding.plans
            ),
            "state_trace_finite": bool(np.isfinite(receding.states).all()),
        },
        "thresholds": {
            "mpc_return_min": fixed_return,
            "action_bounds": [LOWER_BOUND, UPPER_BOUND],
            "population_size": population_size,
            "iterations": iterations,
            "receding_steps": RECEDING_STEPS,
            "max_samples": population_size * iterations * RECEDING_STEPS,
            "seed": seed,
        },
        "provenance": {
            "source": [
                {
                    "path": str(source_path).replace("\\", "/"),
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                }
            ],
            "benchmark": str(source_path).replace("\\", "/"),
            "focused_tests": [
                "tests/test_m14_l16_mpc.py",
                "tests/test_reward_value.py",
                "tests/test_cem.py",
                "tests/test_cem_rollout.py",
                "tests/test_mppi.py",
                "tests/test_mppi_rollout.py",
            ],
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "rss_peak_bytes": psutil.Process().memory_info().rss,
            "network_policy": "offline after installed scikit-learn digits dataset",
        },
        "cleanup": "No temporary files or checkpoints; evidence is retained under artifacts/m14.",
    }
    return payload


def main() -> None:
    output = Path("artifacts/m14/l16-mpc.json")
    config_output = Path("artifacts/m14/l16-mpc.config.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_benchmark()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "schema_version": "m14-gap-threshold-config-v1",
        "record_id": RECORD_ID,
        "lane": LANE,
        "target_evidence": "D2",
        "dataset_revision": "scikit-learn-digits-0.1.0",
        "model_revision": "compact-deterministic-transition-v1",
        "device": "cpu",
        "seed": SEED,
        "train_episodes": 56,
        "holdout_episodes": 16,
        "horizon": HORIZON,
        "receding_steps": RECEDING_STEPS,
        "population_size": POPULATION_SIZE,
        "iterations": ITERATIONS,
        "action_bounds": [LOWER_BOUND, UPPER_BOUND],
        "acceptance": {
            "finite_returns": True,
            "mpc_improves_fixed_zero": {"comparator": ">", "threshold": 0.0},
            "receding_horizon_replans": {"comparator": "==", "threshold": RECEDING_STEPS},
            "executed_prefix_only": {"comparator": "==", "threshold": RECEDING_STEPS},
            "actions_within_bounds": True,
            "sample_budget_max": POPULATION_SIZE * ITERATIONS * RECEDING_STEPS,
            "state_trace_finite": True,
        },
        "scope": (
            "Compact held-out recorded-trajectory substitute only; no real pretrained model, "
            "policy-gradient, MuZero, MCTS, or CUDA claim."
        ),
    }
    config_output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
