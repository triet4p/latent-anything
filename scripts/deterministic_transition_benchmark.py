"""Sprint 63 benchmark for deterministic latent transition and rollout drift."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from latent_anything import DeterministicLatentTransition, LatentSpace

DEFAULT_OUTPUT = Path("artifacts/deterministic_transition_rollout.json")
DEFAULT_CONFIG_OUTPUT = Path("artifacts/deterministic_transition_rollout_config.json")
DEFAULT_PLOT_OUTPUT = Path("artifacts/deterministic_transition_rollout.png")


def _generate_system(
    *, seed: int, episodes: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate controlled 2D linear dynamics with reproducible episodes."""

    rng = np.random.default_rng(seed)
    dynamics = np.array([[0.97, 0.12], [-0.08, 0.94]], dtype=np.float64)
    control = np.array([[0.25, -0.18]], dtype=np.float64)
    initial_states = rng.normal(scale=0.8, size=(episodes, 2))
    actions = rng.normal(scale=0.7, size=(episodes, horizon, 1))
    states = np.empty((episodes, horizon + 1, 2), dtype=np.float64)
    states[:, 0, :] = initial_states
    for step in range(horizon):
        states[:, step + 1, :] = states[:, step, :] @ dynamics.T + actions[:, step, :] @ control
    return states, actions, initial_states, dynamics, control


def run_benchmark(
    *, seed: int = 63, episodes: int = 64, horizon: int = 24, train_fraction: float = 0.75
) -> dict[str, object]:
    """Fit on held-out episodes and return one-step/open-loop evidence."""

    if episodes < 4:
        raise ValueError("episodes must be >= 4")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    states, actions, initial_states, dynamics, control = _generate_system(seed=seed, episodes=episodes, horizon=horizon)
    split = max(2, min(episodes - 2, int(episodes * train_fraction)))
    train_states = states[:split, :-1].reshape(-1, 2)
    train_actions = actions[:split].reshape(-1, 1)
    train_next_states = states[:split, 1:].reshape(-1, 2)
    test_states = states[split:, :-1].reshape(-1, 2)
    test_actions = actions[split:].reshape(-1, 1)
    test_next_states = states[split:, 1:].reshape(-1, 2)

    transition = DeterministicLatentTransition(
        LatentSpace(2, source_model="synthetic-linear-system"),
        action_dim=1,
        source_space_identity="synthetic-linear-system-v1",
        ridge=1e-10,
    )
    fit_start = time.perf_counter()
    transition.fit(train_states, train_actions, train_next_states)
    fit_runtime = time.perf_counter() - fit_start
    one_step = transition.evaluate_one_step(test_states, test_actions, test_next_states)
    rollout = transition.evaluate_rollout(initial_states[split:], actions[split:], states[split:])
    return {
        "evidence_status": "D2",
        "benchmark": "deterministic_latent_transition_rollout",
        "seed": seed,
        "source_space_identity": transition.source_space_identity,
        "state_dim": transition.state_dim,
        "action_dim": transition.action_dim,
        "episodes": episodes,
        "horizon": horizon,
        "train_episodes": split,
        "test_episodes": episodes - split,
        "training_horizon": transition.fit_metadata["training_horizon"],
        "synthetic_system": {"dynamics": dynamics.tolist(), "control": control.tolist()},
        "one_step": {
            "mse": one_step.mse,
            "rmse": one_step.rmse,
            "max_error": one_step.max_error,
            "n_samples": one_step.n_samples,
            "runtime_seconds": one_step.runtime_seconds,
        },
        "rollout": {
            "errors_by_horizon": list(rollout.errors_by_horizon),
            "mean_error": rollout.mean_error,
            "final_error": rollout.final_error,
            "max_error": rollout.max_error,
            "max_state_norm": rollout.max_state_norm,
            "runtime_seconds": rollout.runtime_seconds,
            "stable": rollout.stable,
        },
        "fit_runtime_seconds": fit_runtime,
    }


def _write_plot(result: dict[str, object], output: Path) -> None:
    """Write a compact error-versus-horizon plot for the evidence artifact."""

    rollout = result["rollout"]
    assert isinstance(rollout, dict)
    errors = np.asarray(rollout["errors_by_horizon"], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(np.arange(1, len(errors) + 1), errors, marker="o", color="#264653", linewidth=2)
    axis.set_title("Deterministic latent transition: open-loop error")
    axis.set_xlabel("Rollout horizon")
    axis.set_ylabel("Mean Euclidean latent error")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--episodes", type=int, default=64)
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
