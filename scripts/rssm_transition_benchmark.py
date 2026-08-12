"""Sprint 65 comparison of deterministic, memoryless, and RSSM transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from latent_anything import (
    DeterministicLatentTransition,
    LatentSpace,
    RSSMLatentTransition,
    RSSMTransitionConfig,
    StochasticGaussianLatentTransition,
)

DEFAULT_OUTPUT = Path("artifacts/rssm_transition_comparison.json")
DEFAULT_CONFIG_OUTPUT = Path("artifacts/rssm_transition_comparison_config.json")
DEFAULT_PLOT_OUTPUT = Path("artifacts/rssm_transition_comparison.png")


def _generate_system(*, seed: int, episodes: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a compact partially-observed system with one-step memory."""

    rng = np.random.default_rng(seed)
    states = np.empty((episodes, horizon + 1, 2), dtype=np.float64)
    actions = rng.normal(scale=0.55, size=(episodes, horizon, 1))
    states[:, 0] = rng.normal(scale=0.6, size=(episodes, 2))
    memory = rng.normal(scale=0.2, size=(episodes, 1))
    for index in range(horizon):
        memory = 0.84 * memory + 0.28 * states[:, index, :1] + 0.22 * actions[:, index]
        states[:, index + 1, 0] = (
            0.72 * states[:, index, 0]
            + 0.38 * memory[:, 0]
            + 0.18 * actions[:, index, 0]
            + rng.normal(scale=0.025, size=episodes)
        )
        states[:, index + 1, 1] = (
            0.91 * states[:, index, 1] - 0.12 * actions[:, index, 0] + rng.normal(scale=0.035, size=episodes)
        )
    return states, actions, np.ones((episodes, horizon), dtype=np.float64)


def run_benchmark(*, seed: int = 65, episodes: int = 64, horizon: int = 16) -> dict[str, object]:
    """Fit all three transition instances and return comparable D2 evidence."""

    if episodes < 12:
        raise ValueError("episodes must be >= 12")
    if horizon < 2:
        raise ValueError("horizon must be >= 2")
    states, actions, mask = _generate_system(seed=seed, episodes=episodes, horizon=horizon)
    split = max(8, min(episodes - 4, int(episodes * 0.75)))
    identity = "synthetic-memory-system-v1"
    space = LatentSpace(2, source_model="synthetic-memory-system")
    train_states = states[:split]
    train_actions = actions[:split]
    test_states = states[split:]
    test_actions = actions[split:]

    deterministic = DeterministicLatentTransition(space, 1, source_space_identity=identity, ridge=1e-8)
    stochastic = StochasticGaussianLatentTransition(
        space, 1, source_space_identity=identity, ridge=1e-8, variance_floor=1e-6
    )
    rssm = RSSMLatentTransition(
        space,
        1,
        source_space_identity=identity,
        config=RSSMTransitionConfig(hidden_dim=10, epochs=90, learning_rate=0.025, seed=seed),
    )
    flat_states = train_states[:, :-1].reshape(-1, 2)
    flat_actions = train_actions.reshape(-1, 1)
    flat_targets = train_states[:, 1:].reshape(-1, 2)
    deterministic.fit(flat_states, flat_actions, flat_targets)
    stochastic.fit(flat_states, flat_actions, flat_targets)
    rssm.fit(train_states, train_actions, sequence_mask=mask[:split])

    deterministic_one = deterministic.evaluate_one_step(
        test_states[:, :-1].reshape(-1, 2), test_actions.reshape(-1, 1), test_states[:, 1:].reshape(-1, 2)
    )
    stochastic_one = stochastic.evaluate_one_step(
        test_states[:, :-1].reshape(-1, 2),
        test_actions.reshape(-1, 1),
        test_states[:, 1:].reshape(-1, 2),
        n_diversity_samples=64,
        seed=seed,
    )
    rssm_one = rssm.evaluate_one_step(test_states, test_actions, test_states, sequence_mask=mask[split:])
    deterministic_rollout = deterministic.evaluate_rollout(test_states[:, 0], test_actions, test_states)
    stochastic_rollout = stochastic.evaluate_rollout(
        test_states[:, 0], test_actions, test_states, n_samples=96, seed=seed
    )
    rssm_rollout = rssm.evaluate_rollout(
        test_states[:, 0], test_actions, test_states, sequence_mask=mask[split:], n_samples=96, seed=seed
    )
    result: dict[str, object] = {
        "evidence_status": "D2",
        "benchmark": "rssm_transition_comparison",
        "seed": seed,
        "source_space_identity": identity,
        "episodes": episodes,
        "horizon": horizon,
        "train_episodes": split,
        "test_episodes": episodes - split,
        "dataset": {
            "kind": "controlled_partially_observed_temporal_system",
            "noise": "independent Gaussian observation/process noise",
            "variable_length_mask_supported": True,
        },
        "deterministic": {
            "one_step_rmse": deterministic_one.rmse,
            "rollout_mean_error": deterministic_rollout.mean_error,
            "rollout_final_error": deterministic_rollout.final_error,
            "rollout_stable": deterministic_rollout.stable,
        },
        "stochastic_memoryless": {
            "one_step_nll": stochastic_one.negative_log_likelihood,
            "one_step_coverage": stochastic_one.coverage,
            "rollout_mean_error": stochastic_rollout.mean_error,
            "rollout_final_error": stochastic_rollout.final_error,
            "rollout_mean_coverage": stochastic_rollout.mean_coverage,
            "rollout_stable": stochastic_rollout.stable,
        },
        "rssm": {
            "config": rssm.to_config().model_dump(mode="json"),
            "one_step_mse": rssm_one.mse,
            "one_step_rmse": rssm_one.rmse,
            "one_step_nll": rssm_one.negative_log_likelihood,
            "one_step_kl": rssm_one.kl_divergence,
            "one_step_coverage": rssm_one.coverage,
            "rollout_errors_by_horizon": list(rssm_rollout.errors_by_horizon),
            "rollout_kl_by_horizon": list(rssm_rollout.kl_by_horizon),
            "rollout_coverage_by_horizon": list(rssm_rollout.coverage_by_horizon),
            "rollout_mean_error": rssm_rollout.mean_error,
            "rollout_final_error": rssm_rollout.final_error,
            "rollout_mean_kl": rssm_rollout.mean_kl,
            "rollout_mean_coverage": rssm_rollout.mean_coverage,
            "rollout_stable": rssm_rollout.stable,
        },
        "failure_analysis": [
            "The deterministic and memoryless stochastic baselines do not receive recurrent history, "
            "so this benchmark is intentionally partially observed.",
            "RSSM does not outperform either baseline on open-loop rollout error: its mean error is "
            f"{rssm_rollout.mean_error:.3f} versus {deterministic_rollout.mean_error:.3f} and "
            f"{stochastic_rollout.mean_error:.3f}; no superiority claim is made.",
            "RSSM KL is against an observation-centred posterior proxy; a learned posterior encoder "
            "and free-bits objective remain outside this compact NumPy-facing instance.",
            "All claims are synthetic D2 evidence; no real pretrained world-model or CUDA claim is promoted.",
        ],
    }
    return result


def _write_plot(result: dict[str, object], output: Path) -> None:
    rssm = cast(dict[str, object], result["rssm"])
    errors = np.asarray(rssm["rollout_errors_by_horizon"], dtype=np.float64)
    coverage = np.asarray(rssm["rollout_coverage_by_horizon"], dtype=np.float64)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    horizon = np.arange(1, len(errors) + 1)
    axes[0].plot(horizon, errors, marker="o", color="#264653", label="RSSM mean-path error")
    axes[0].set_ylabel("Latent error")
    axes[0].set_title("RSSM-style transition: masked temporal rollout")
    axes[0].grid(alpha=0.3)
    axes[1].plot(horizon, coverage, marker="o", color="#2a9d8f", label="RSSM interval coverage")
    axes[1].axhline(0.95, color="#6c757d", linestyle="--", label="nominal 95%")
    axes[1].set_xlabel("Rollout horizon")
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_OUTPUT)
    args = parser.parse_args()
    result = run_benchmark(seed=args.seed, episodes=args.episodes, horizon=args.horizon)
    config = {
        "benchmark": result["benchmark"],
        "seed": result["seed"],
        "episodes": result["episodes"],
        "horizon": result["horizon"],
        "train_fraction": 0.75,
        "rssm_hidden_dim": 10,
        "rssm_epochs": 90,
        "sequence_mask": "(episodes, horizon), 0/1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    _write_plot(result, args.plot_output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
