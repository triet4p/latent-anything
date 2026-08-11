"""Sprint 64 benchmark for calibrated stochastic latent transition rollouts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from latent_anything import DeterministicLatentTransition, LatentSpace, StochasticGaussianLatentTransition

DEFAULT_OUTPUT = Path("artifacts/stochastic_transition_rollout.json")
DEFAULT_CONFIG_OUTPUT = Path("artifacts/stochastic_transition_rollout_config.json")
DEFAULT_PLOT_OUTPUT = Path("artifacts/stochastic_transition_uncertainty_band.png")


def _generate_system(
    *, seed: int, episodes: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate controlled 2D linear dynamics with independent process noise."""

    rng = np.random.default_rng(seed)
    dynamics = np.array([[0.93, 0.08], [-0.05, 0.90]], dtype=np.float64)
    control = np.array([[0.22, -0.31]], dtype=np.float64)
    noise_scale = np.array([0.07, 0.16], dtype=np.float64)
    initial_states = rng.normal(scale=0.8, size=(episodes, 2))
    actions = rng.normal(scale=0.65, size=(episodes, horizon, 1))
    states = np.empty((episodes, horizon + 1, 2), dtype=np.float64)
    states[:, 0, :] = initial_states
    for step in range(horizon):
        mean = states[:, step, :] @ dynamics.T + actions[:, step, :] @ control
        states[:, step + 1, :] = mean + rng.normal(scale=noise_scale, size=(episodes, 2))
    return states, actions, initial_states, dynamics, control, noise_scale


def run_benchmark(
    *, seed: int = 64, episodes: int = 96, horizon: int = 24, train_fraction: float = 0.75
) -> dict[str, object]:
    """Fit deterministic and stochastic transitions and return D2 evidence."""

    if episodes < 8:
        raise ValueError("episodes must be >= 8")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    states, actions, initial_states, dynamics, control, noise_scale = _generate_system(
        seed=seed, episodes=episodes, horizon=horizon
    )
    split = max(4, min(episodes - 4, int(episodes * train_fraction)))
    train_states = states[:split, :-1].reshape(-1, 2)
    train_actions = actions[:split].reshape(-1, 1)
    train_next_states = states[:split, 1:].reshape(-1, 2)
    test_states = states[split:, :-1].reshape(-1, 2)
    test_actions = actions[split:].reshape(-1, 1)
    test_next_states = states[split:, 1:].reshape(-1, 2)

    stochastic = StochasticGaussianLatentTransition(
        LatentSpace(2, source_model="synthetic-stochastic-linear-system"),
        action_dim=1,
        source_space_identity="synthetic-stochastic-linear-system-v1",
        ridge=1e-10,
        variance_floor=1e-6,
    )
    deterministic = DeterministicLatentTransition(
        LatentSpace(2, source_model="synthetic-stochastic-linear-system"),
        action_dim=1,
        source_space_identity="synthetic-stochastic-linear-system-v1",
        ridge=1e-10,
    )
    fit_start = time.perf_counter()
    stochastic.fit(train_states, train_actions, train_next_states)
    deterministic.fit(train_states, train_actions, train_next_states)
    fit_runtime = time.perf_counter() - fit_start

    stochastic_one_step = stochastic.evaluate_one_step(
        test_states, test_actions, test_next_states, n_diversity_samples=96, seed=seed
    )
    stochastic_rollout = stochastic.evaluate_rollout(
        initial_states[split:], actions[split:], states[split:], n_samples=512, seed=seed
    )
    deterministic_one_step = deterministic.evaluate_one_step(test_states, test_actions, test_next_states)
    deterministic_rollout = deterministic.evaluate_rollout(initial_states[split:], actions[split:], states[split:])
    sampled = stochastic.rollout(initial_states[split], actions[split], n_samples=512, seed=seed)
    mean_path = stochastic.mean_rollout(initial_states[split], actions[split])

    return {
        "evidence_status": "D2",
        "benchmark": "stochastic_gaussian_latent_transition_rollout",
        "seed": seed,
        "source_space_identity": stochastic.source_space_identity,
        "state_dim": stochastic.state_dim,
        "action_dim": stochastic.action_dim,
        "episodes": episodes,
        "horizon": horizon,
        "train_episodes": split,
        "test_episodes": episodes - split,
        "training_horizon": stochastic.fit_metadata["training_horizon"],
        "synthetic_system": {
            "dynamics": dynamics.tolist(),
            "control": control.tolist(),
            "noise_scale": noise_scale.tolist(),
        },
        "fitted_scale": stochastic.scale.tolist(),
        "one_step": {
            "negative_log_likelihood": stochastic_one_step.negative_log_likelihood,
            "coverage": stochastic_one_step.coverage,
            "interval_width": stochastic_one_step.interval_width,
            "sample_diversity": stochastic_one_step.sample_diversity,
            "mean_error": stochastic_one_step.mean_error,
            "n_samples": stochastic_one_step.n_samples,
            "runtime_seconds": stochastic_one_step.runtime_seconds,
        },
        "rollout": {
            "negative_log_likelihood_by_horizon": list(stochastic_rollout.nll_by_horizon),
            "coverage_by_horizon": list(stochastic_rollout.coverage_by_horizon),
            "sample_diversity_by_horizon": list(stochastic_rollout.sample_diversity_by_horizon),
            "mean_error_by_horizon": list(stochastic_rollout.mean_error_by_horizon),
            "mean_negative_log_likelihood": stochastic_rollout.mean_negative_log_likelihood,
            "mean_coverage": stochastic_rollout.mean_coverage,
            "mean_sample_diversity": stochastic_rollout.mean_sample_diversity,
            "final_error": stochastic_rollout.final_error,
            "runtime_seconds": stochastic_rollout.runtime_seconds,
            "stable": stochastic_rollout.stable,
        },
        "deterministic_control": {
            "one_step_rmse": deterministic_one_step.rmse,
            "rollout_mean_error": deterministic_rollout.mean_error,
            "rollout_final_error": deterministic_rollout.final_error,
            "rollout_max_error": deterministic_rollout.max_error,
        },
        "comparison": {
            "sampled_rollout_mean_error": float(np.mean(np.linalg.norm(sampled.mean[1:] - states[split, 1:], axis=1))),
            "sampled_rollout_final_scale": sampled.scale[-1].tolist(),
            "mean_rollout_final_state": mean_path.to_numpy()[-1].tolist(),
            "sampled_rollout_final_state": sampled.mean[-1].tolist(),
            "sampled_rollout_mean_differs_from_deterministic": bool(
                not np.allclose(
                    sampled.mean, deterministic.mean_rollout(initial_states[split], actions[split]).to_numpy()
                )
            ),
        },
        "fit_runtime_seconds": fit_runtime,
    }


def _write_plot(result: dict[str, object], output: Path) -> None:
    """Write uncertainty bands and horizon calibration plots."""

    rollout = result["rollout"]
    assert isinstance(rollout, dict)
    errors = np.asarray(rollout["mean_error_by_horizon"], dtype=np.float64)
    coverage = np.asarray(rollout["coverage_by_horizon"], dtype=np.float64)
    diversity = np.asarray(rollout["sample_diversity_by_horizon"], dtype=np.float64)
    horizon = np.arange(1, len(errors) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(horizon, errors, color="#264653", marker="o", label="mean-path error")
    axes[0].plot(horizon, diversity, color="#e76f51", marker="s", label="sample diversity")
    axes[0].set_ylabel("Latent distance / scale")
    axes[0].set_title("Stochastic Gaussian latent transition: uncertainty growth")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(horizon, coverage, color="#2a9d8f", marker="o", label="empirical 95% coverage")
    axes[1].axhline(0.95, color="#6c757d", linestyle="--", label="nominal 95%")
    axes[1].set_xlabel("Rollout horizon")
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
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
        "rollout_samples": 512,
        "interval_level": 0.95,
        "source_space_identity": result["source_space_identity"],
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
