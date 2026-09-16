"""Persist M14 L15 deterministic, stochastic, and RSSM transition evidence."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import psutil

from latent_anything.latent_space import LatentSpace
from latent_anything.rssm import RSSMLatentTransition, RSSMTransitionConfig
from latent_anything.transition import DeterministicLatentTransition, StochasticGaussianLatentTransition

RUN_COMMAND = "uv run python scripts/m14_l15_transitions.py"
DATASET_REVISION = "m14-transition-fixture-v1"
TRAIN_SEED = 650
HOLDOUT_SEED = 651


def _fixture(seed: int, episodes: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.empty((episodes, horizon + 1, 2), dtype=np.float64)
    actions = rng.normal(scale=0.3, size=(episodes, horizon, 1))
    states[:, 0] = rng.normal(scale=0.5, size=(episodes, 2))
    matrix = np.array([[0.72, 0.08], [-0.10, 0.84]])
    action_matrix = np.array([[0.24], [-0.18]])
    for step in range(horizon):
        states[:, step + 1] = states[:, step] @ matrix.T + actions[:, step] @ action_matrix.T
        states[:, step + 1] += rng.normal(scale=0.015, size=(episodes, 2))
    return states, actions, np.ones((episodes, horizon), dtype=np.float64)


def main() -> None:
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    process = psutil.Process()
    train_states, train_actions, train_mask = _fixture(TRAIN_SEED, 24, 6)
    holdout_states, holdout_actions, holdout_mask = _fixture(HOLDOUT_SEED, 8, 6)
    train_flat_states = train_states[:, :-1].reshape(-1, 2)
    train_flat_actions = train_actions.reshape(-1, 1)
    train_flat_next = train_states[:, 1:].reshape(-1, 2)
    holdout_flat_states = holdout_states[:, :-1].reshape(-1, 2)
    holdout_flat_actions = holdout_actions.reshape(-1, 1)
    holdout_flat_next = holdout_states[:, 1:].reshape(-1, 2)
    space = LatentSpace(2, source_model=DATASET_REVISION)
    deterministic = DeterministicLatentTransition(space, 1, source_space_identity=DATASET_REVISION).fit(
        train_flat_states, train_flat_actions, train_flat_next
    )
    stochastic = StochasticGaussianLatentTransition(
        space, 1, source_space_identity=DATASET_REVISION, variance_floor=1e-8
    ).fit(train_flat_states, train_flat_actions, train_flat_next)
    rssm = RSSMLatentTransition(
        space,
        1,
        source_space_identity=DATASET_REVISION,
        config=RSSMTransitionConfig(hidden_dim=6, epochs=35, learning_rate=0.03, seed=65),
    ).fit(train_states, train_actions, sequence_mask=train_mask)
    deterministic_one = deterministic.evaluate_one_step(holdout_flat_states, holdout_flat_actions, holdout_flat_next)
    deterministic_rollout = deterministic.evaluate_rollout(holdout_states[:, 0], holdout_actions, holdout_states)
    stochastic_one = stochastic.evaluate_one_step(
        holdout_flat_states, holdout_flat_actions, holdout_flat_next, n_diversity_samples=32, seed=123
    )
    stochastic_rollout = stochastic.evaluate_rollout(
        holdout_states[:, 0], holdout_actions, holdout_states, n_samples=64, seed=123
    )
    rssm_one = rssm.evaluate_one_step(holdout_states, holdout_actions, holdout_states, sequence_mask=holdout_mask)
    rssm_rollout = rssm.evaluate_rollout(
        holdout_states[:, 0], holdout_actions, holdout_states, sequence_mask=holdout_mask, n_samples=64, seed=123
    )
    stochastic_a = stochastic.rollout(holdout_states[0, 0], holdout_actions[0], n_samples=16, seed=777)
    stochastic_b = stochastic.rollout(holdout_states[0, 0], holdout_actions[0], n_samples=16, seed=777)
    rssm.reset()
    rssm_a = rssm.step(holdout_states[0, 0], holdout_actions[0, 0])
    carry_hidden = rssm.hidden_state
    rssm.reset()
    rssm_b = rssm.step(holdout_states[0, 0], holdout_actions[0, 0])
    rssm_rollout_a = rssm.rollout(holdout_states[0, 0], holdout_actions[0], n_samples=8, seed=888)
    rssm.reset()
    rssm_rollout_b = rssm.rollout(holdout_states[0, 0], holdout_actions[0], n_samples=8, seed=888)
    rss_peak = process.memory_info().rss
    payload = {
        "schema_version": "m14-l15-run-v1",
        "source_sha": source_sha,
        "command": RUN_COMMAND,
        "target": {
            "dataset_revision": DATASET_REVISION,
            "description": "seeded compact 2D latent transition fixture with deterministic process noise",
            "license": "project-authored MIT-repository content",
            "device": "cpu",
        },
        "split": {
            "train_episodes": 24,
            "holdout_episodes": 8,
            "horizon": 6,
            "train_seed": TRAIN_SEED,
            "holdout_seed": HOLDOUT_SEED,
            "train_transition_count": int(len(train_flat_states)),
            "holdout_transition_count": int(len(holdout_flat_states)),
            "train_digest": hashlib.sha256(train_states.tobytes()).hexdigest(),
            "holdout_digest": hashlib.sha256(holdout_states.tobytes()).hexdigest(),
        },
        "models": {
            "deterministic": {"source_space_identity": DATASET_REVISION, "ridge": deterministic.ridge},
            "stochastic": {"source_space_identity": DATASET_REVISION, "variance_floor": stochastic.variance_floor},
            "rssm": rssm.to_config().model_dump(mode="json"),
        },
        "metrics": {
            "deterministic_one_step": asdict(deterministic_one),
            "deterministic_rollout": asdict(deterministic_rollout),
            "stochastic_one_step": asdict(stochastic_one),
            "stochastic_rollout": asdict(stochastic_rollout),
            "rssm_one_step": asdict(rssm_one),
            "rssm_rollout": asdict(rssm_rollout),
            "rssm_hidden_shape": list(rssm.hidden_shape),
            "rssm_scale": rssm.scale.tolist(),
        },
        "acceptance": {
            "deterministic_one_step_finite": bool(np.isfinite(deterministic_one.rmse)),
            "deterministic_rollout_finite": bool(np.isfinite(deterministic_rollout.mean_error)),
            "deterministic_rmse_threshold": deterministic_one.rmse < 0.15,
            "stochastic_distribution_finite": bool(np.isfinite(stochastic_one.negative_log_likelihood)),
            "stochastic_seeded_reproducible": bool(np.array_equal(stochastic_a.samples, stochastic_b.samples)),
            "stochastic_rollout_shape": stochastic_a.samples.shape == (16, 7, 2),
            "rssm_one_step_finite": bool(np.isfinite(rssm_one.rmse) and np.isfinite(rssm_one.negative_log_likelihood)),
            "rssm_rollout_finite": bool(np.isfinite(rssm_rollout.mean_error)),
            "rssm_state_carry": bool(np.isfinite(rssm_a).all() and np.linalg.norm(carry_hidden) > 0.0),
            "rssm_seeded_rollout_reproducible": bool(np.array_equal(rssm_rollout_a.samples, rssm_rollout_b.samples)),
            "rollout_horizon_complete": deterministic_rollout.horizon == 6 and stochastic_rollout.horizon == 6 and rssm_rollout.horizon == 6,
            "shape_safe": deterministic.rollout(holdout_states[0, 0], holdout_actions[0]).shape == (7, 2),
            "finite_scales": bool(np.isfinite(stochastic.scale).all() and np.isfinite(rssm.scale).all()),
        },
        "thresholds": {
            "deterministic_one_step_rmse_max_exclusive": 0.15,
            "rollout_horizon": 6,
            "finite_distribution_metrics": True,
            "seeded_stochastic_and_rssm_reproducibility": True,
        },
        "environment": {
            "device": "cpu",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": metadata.version("torch"),
            "rss_peak_bytes": rss_peak,
            "network_policy": "offline; project-authored fixture only",
        },
        "cleanup": "No temporary state/checkpoint files; all outputs retained under artifacts/m14.",
    }
    output = Path("artifacts/m14/l15-transitions-run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda value: value.tolist()) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, default=lambda value: value.tolist()))


if __name__ == "__main__":
    main()
