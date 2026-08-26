"""Private NumPy validation helpers for the decoder-free JEPA adapter."""

from __future__ import annotations

import numpy as np


def finite_array(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(value).__name__}")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(value, dtype=np.float64)


def validate_point(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
    values = finite_array(value, name=name)
    if values.ndim != 1 or values.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {values.shape}")
    return values


def validate_batch(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
    values = finite_array(value, name=name)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width}), got {values.shape}")
    if values.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    return values


def validate_mask(sequence_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if sequence_mask is None:
        return np.ones(shape, dtype=bool)
    raw = finite_array(sequence_mask, name="sequence_mask")
    if raw.shape != shape or not np.isin(raw, [0.0, 1.0]).all():
        raise ValueError(f"sequence_mask must have shape {shape} and contain only 0/1 values")
    return raw.astype(bool)


def validate_sequences(
    observations: np.ndarray,
    actions: np.ndarray,
    sequence_mask: np.ndarray | None,
    *,
    observation_dim: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation_values = finite_array(observations, name="observations")
    action_values = finite_array(actions, name="actions")
    if observation_values.ndim != 3 or observation_values.shape[2] != observation_dim:
        raise ValueError(
            f"observations must have shape (episodes, horizon + 1, {observation_dim}), got {observation_values.shape}"
        )
    expected = (observation_values.shape[0], observation_values.shape[1] - 1, action_dim)
    if action_values.shape != expected:
        raise ValueError(f"actions must have shape {expected}, got {action_values.shape}")
    return observation_values, action_values, validate_mask(sequence_mask, expected[:2])


def validate_rollout_inputs(
    initial_state: np.ndarray,
    actions: np.ndarray,
    target_states: np.ndarray,
    sequence_mask: np.ndarray | None,
    *,
    latent_dim: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initial = finite_array(initial_state, name="initial_state")
    action_values = finite_array(actions, name="actions")
    targets = finite_array(target_states, name="target_states")
    if initial.ndim == 1:
        initial = initial[None, :]
    if action_values.ndim == 2:
        action_values = action_values[None, :, :]
    if targets.ndim == 2:
        targets = targets[None, :, :]
    if initial.ndim != 2 or action_values.ndim != 3 or targets.ndim != 3:
        raise ValueError("invalid rollout input dimensions")
    expected_targets = (initial.shape[0], action_values.shape[1] + 1, latent_dim)
    if (
        initial.shape[1] != latent_dim
        or action_values.shape[0] != initial.shape[0]
        or action_values.shape[2] != action_dim
    ):
        raise ValueError("rollout inputs have incompatible state/action shapes")
    if targets.shape != expected_targets:
        raise ValueError(f"target_states must have shape {expected_targets}, got {targets.shape}")
    if not np.array_equal(initial, targets[:, 0, :]):
        raise ValueError("initial_state must equal target_states[:, 0, :]")
    return initial, action_values, targets, validate_mask(sequence_mask, action_values.shape[:2])
