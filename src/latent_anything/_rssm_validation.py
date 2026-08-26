"""Private NumPy validation helpers for the RSSM facade."""

from __future__ import annotations

import numpy as np


def finite_array(value: object, *, name: str) -> np.ndarray:
    """Validate and copy a finite numeric NumPy array."""

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
    return values


def _validate_mask(sequence_mask: np.ndarray | None, *, shape: tuple[int, int]) -> np.ndarray:
    if sequence_mask is None:
        return np.ones(shape, dtype=bool)
    raw_mask = finite_array(sequence_mask, name="sequence_mask")
    if raw_mask.shape != shape or not np.isin(raw_mask, [0.0, 1.0]).all():
        raise ValueError(f"sequence_mask must have shape {shape} and contain only 0/1 values")
    return raw_mask.astype(bool)


def validate_sequences(
    states: np.ndarray,
    actions: np.ndarray,
    sequence_mask: np.ndarray | None,
    *,
    state_dim: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_values = finite_array(states, name="states")
    action_values = finite_array(actions, name="actions")
    if state_values.ndim != 3 or state_values.shape[2] != state_dim:
        raise ValueError(f"states must have shape (episodes, horizon + 1, {state_dim}), got {state_values.shape}")
    if (
        action_values.ndim != 3
        or action_values.shape[2] != action_dim
        or action_values.shape[:2] != (state_values.shape[0], state_values.shape[1] - 1)
    ):
        raise ValueError(f"actions must have shape (episodes, horizon, {action_dim}), got {action_values.shape}")
    return state_values, action_values, _validate_mask(sequence_mask, shape=action_values.shape[:2])


def validate_one_step_sequences(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    sequence_mask: np.ndarray | None,
    *,
    state_dim: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_values, action_values, mask = validate_sequences(
        states, actions, sequence_mask, state_dim=state_dim, action_dim=action_dim
    )
    target_values = finite_array(next_states, name="next_states")
    if target_values.shape != state_values.shape:
        raise ValueError(f"next_states must have shape {state_values.shape}, got {target_values.shape}")
    return state_values, action_values, mask


def validate_rollout_inputs(
    initial_state: np.ndarray,
    actions: np.ndarray,
    target_states: np.ndarray,
    sequence_mask: np.ndarray | None,
    *,
    state_dim: int,
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
    if initial.ndim != 2 or initial.shape[1] != state_dim or action_values.ndim != 3 or targets.ndim != 3:
        raise ValueError("invalid rollout input dimensions")
    if (
        action_values.shape[0] != initial.shape[0]
        or targets.shape[:2] != (initial.shape[0], action_values.shape[1] + 1)
        or action_values.shape[2] != action_dim
        or targets.shape[2] != state_dim
    ):
        raise ValueError("rollout inputs have incompatible batch, horizon, or feature shapes")
    if not np.array_equal(initial, targets[:, 0]):
        raise ValueError("initial_state must equal target_states[:, 0, :]")
    return initial, action_values, targets, _validate_mask(sequence_mask, shape=action_values.shape[:2])
