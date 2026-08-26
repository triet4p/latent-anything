"""Private NumPy recurrent math and rollout assembly for RSSM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def recurrent_step(
    hidden: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    *,
    recurrent_weights: np.ndarray,
    recurrent_bias: np.ndarray,
    emission_weights: np.ndarray,
    emission_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one or a batch of recurrent states and emit predictive means."""

    recurrent_input = np.concatenate((hidden, state, action), axis=-1)
    next_hidden = np.tanh(recurrent_input @ recurrent_weights + recurrent_bias)
    ones = np.ones((*next_hidden.shape[:-1], 1), dtype=np.float64)
    emission_input = np.concatenate((next_hidden, state, action, ones), axis=-1)
    mean = emission_input @ emission_weights + emission_bias
    return next_hidden, mean


def teacher_forced_distribution_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    recurrent_weights: np.ndarray,
    recurrent_bias: np.ndarray,
    emission_weights: np.ndarray,
    emission_bias: np.ndarray,
    hidden_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return masked teacher-forced means and hidden paths for public wrapping."""

    hidden = np.zeros((states.shape[0], hidden_dim), dtype=np.float64)
    means = np.empty((states.shape[0], actions.shape[1], emission_weights.shape[1]), dtype=np.float64)
    hidden_paths = np.empty((states.shape[0], actions.shape[1], hidden_dim), dtype=np.float64)
    for index in range(actions.shape[1]):
        recurrent_input = np.concatenate((hidden, states[:, index], actions[:, index]), axis=-1)
        proposed_hidden = np.tanh(recurrent_input @ recurrent_weights + recurrent_bias)
        hidden = np.where(mask[:, index, None], proposed_hidden, hidden)
        ones = np.ones((states.shape[0], 1), dtype=np.float64)
        emission_input = np.concatenate((hidden, states[:, index], actions[:, index], ones), axis=-1)
        means[:, index] = emission_input @ emission_weights + emission_bias
        hidden_paths[:, index] = hidden
    return means, hidden_paths


def sample_recurrent_rollout(
    initial: np.ndarray,
    actions: np.ndarray,
    *,
    n_samples: int,
    hidden_dim: int,
    scale: np.ndarray,
    seed: int | None,
    recurrent_weights: np.ndarray,
    recurrent_bias: np.ndarray,
    emission_weights: np.ndarray,
    emission_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a seeded particle rollout and return samples, paths, final hidden state."""

    generator = np.random.default_rng(seed)
    samples = np.empty((n_samples, actions.shape[0] + 1, initial.shape[0]), dtype=np.float64)
    hidden = np.zeros((n_samples, hidden_dim), dtype=np.float64)
    deterministic = np.empty((n_samples, actions.shape[0] + 1, hidden_dim), dtype=np.float64)
    samples[:, 0] = initial
    deterministic[:, 0] = hidden
    for index, action in enumerate(actions):
        repeated_action = np.broadcast_to(action, (n_samples, action.shape[0]))
        hidden, means = recurrent_step(
            hidden,
            samples[:, index],
            repeated_action,
            recurrent_weights=recurrent_weights,
            recurrent_bias=recurrent_bias,
            emission_weights=emission_weights,
            emission_bias=emission_bias,
        )
        samples[:, index + 1] = means + scale * generator.normal(size=means.shape)
        deterministic[:, index + 1] = hidden
    return samples, deterministic, np.mean(hidden, axis=0)


def build_rollout_metadata(
    *,
    state_source: str,
    source_space_identity: str,
    transition_name: str,
    horizon: int,
    action_shape: tuple[int],
    state_shape: tuple[int],
    hidden_shape: tuple[int],
    metadata: Mapping[str, Any] | None,
    n_samples: int | None = None,
    seed: int | None = None,
    interval_level: float | None = None,
) -> dict[str, Any]:
    """Assemble the established public rollout metadata without model state."""

    values: dict[str, Any] = {
        "state_source": state_source,
        "source_space_identity": source_space_identity,
        "transition": transition_name,
        "rollout_horizon": int(horizon),
        "action_shape": action_shape,
        "state_shape": state_shape,
        "deterministic_state_shape": hidden_shape,
        "stateful": True,
    }
    if n_samples is not None:
        values.update({"n_samples": int(n_samples), "seed": seed, "interval_level": interval_level})
    if metadata is not None:
        values.update(dict(metadata))
    return values
