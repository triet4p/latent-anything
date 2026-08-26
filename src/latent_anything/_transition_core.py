"""Private validation and affine-fit helpers shared by concrete transitions.

The deterministic and diagonal-Gaussian transitions intentionally keep their
different public lifecycles.  This module contains only the behavior proven
identical by both implementations: finite NumPy validation, constructor
guards, residual affine fitting, and rollout-shape normalization.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from latent_anything.latent_space import LatentSpace


def as_finite_array(value: object, *, name: str) -> np.ndarray:
    """Return *value* as an array and reject non-finite numeric data."""

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(value).__name__}")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(value, dtype=np.float64)


def validate_constructor(
    latent_space: LatentSpace,
    action_dim: int,
    *,
    transition_name: str,
    ridge: float,
    stability_norm_limit: float,
    variance_floor: float | None = None,
) -> None:
    """Apply the exact constructor guards shared by both transitions."""

    if latent_space.geometry != "euclidean" or latent_space.shape != (latent_space.dim,):
        raise ValueError(f"{transition_name} requires a flat Euclidean LatentSpace")
    if action_dim < 1:
        raise ValueError(f"action_dim must be >= 1, got {action_dim}")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError(f"ridge must be finite and >= 0, got {ridge}")
    if variance_floor is not None and (variance_floor < 0 or not np.isfinite(variance_floor)):
        raise ValueError(f"variance_floor must be finite and >= 0, got {variance_floor}")
    if stability_norm_limit <= 0 or not np.isfinite(stability_norm_limit):
        raise ValueError(f"stability_norm_limit must be finite and > 0, got {stability_norm_limit}")


def resolve_source_identity(
    latent_space: LatentSpace,
    source_space_identity: str | None,
    *,
    empty_uses_fallback: bool = False,
) -> str:
    """Resolve a source identity while preserving each class's empty-string semantics."""

    if source_space_identity is None or (empty_uses_fallback and not source_space_identity):
        resolved = latent_space.source_model or f"{latent_space.geometry}:{latent_space.dim}"
    else:
        resolved = source_space_identity
    if not resolved.strip():
        raise ValueError("source_space_identity must be a non-empty string")
    return resolved


def validate_batch(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
    """Validate a two-dimensional state/action batch."""

    values = as_finite_array(value, name=name)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width}), got {values.shape}")
    return values


def validate_point(value: np.ndarray, *, name: str, width: int | None = None) -> np.ndarray:
    """Validate a one-dimensional state/action point."""

    values = as_finite_array(value, name=name)
    if values.ndim != 1 or (width is not None and values.shape != (width,)):
        if width is None:
            raise ValueError(f"{name} must have shape (state_dim,), got {values.shape}")
        raise ValueError(f"{name} must have shape ({width},), got {values.shape}")
    return values


def validate_fit_samples(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    *,
    state_dim: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the common one-step fit sample contract."""

    state_values = validate_batch(states, name="states", width=state_dim)
    action_values = validate_batch(actions, name="actions", width=action_dim)
    target_values = validate_batch(next_states, name="next_states", width=state_dim)
    if state_values.shape[0] != action_values.shape[0] or state_values.shape[0] != target_values.shape[0]:
        raise ValueError("states, actions, and next_states must have the same number of samples")
    if state_values.shape[0] < 1:
        raise ValueError("at least one transition sample is required")
    return state_values, action_values, target_values


def fit_affine_residual(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit shared residual-affine weights and return design, weights, residuals."""

    design = np.concatenate(
        [states, actions, np.ones((states.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    residual_targets = next_states - states
    if ridge == 0:
        weights = np.linalg.lstsq(design, residual_targets, rcond=None)[0]
    else:
        gram = design.T @ design
        regularizer = np.eye(gram.shape[0], dtype=np.float64) * ridge
        weights = np.linalg.solve(gram + regularizer, design.T @ residual_targets)
    mean_predictions = states + design @ weights
    residuals = next_states - mean_predictions
    return design, np.asarray(weights, dtype=np.float64), np.asarray(residuals, dtype=np.float64)


def validate_rollout_inputs(
    initial_state: np.ndarray,
    actions: np.ndarray,
    target_states: np.ndarray,
    *,
    state_dim: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize and validate deterministic/stochastic rollout evaluation inputs."""

    initial = as_finite_array(initial_state, name="initial_state")
    action_values = as_finite_array(actions, name="actions")
    targets = as_finite_array(target_states, name="target_states")
    if initial.ndim == 1:
        initial = initial[None, :]
    if action_values.ndim == 2:
        action_values = action_values[None, :, :]
    if targets.ndim == 2:
        targets = targets[None, :, :]
    if initial.ndim != 2 or initial.shape[1] != state_dim:
        raise ValueError(f"initial_state must have shape (n, {state_dim}) or ({state_dim},), got {initial.shape}")
    if action_values.ndim != 3 or action_values.shape[2] != action_dim:
        raise ValueError(
            f"actions must have shape (n, h, {action_dim}) or (h, {action_dim}), got {action_values.shape}"
        )
    if targets.ndim != 3 or targets.shape[2] != state_dim:
        raise ValueError(
            f"target_states must have shape (n, h+1, {state_dim}) or (h+1, {state_dim}), got {targets.shape}"
        )
    if initial.shape[0] != action_values.shape[0] or initial.shape[0] != targets.shape[0]:
        raise ValueError("initial_state, actions, and target_states must have matching batch sizes")
    if action_values.shape[1] + 1 != targets.shape[1]:
        raise ValueError("target_states must contain the initial state plus one state per action")
    if not np.array_equal(initial, targets[:, 0, :]):
        raise ValueError("initial_state must equal target_states[:, 0, :] for rollout evaluation")
    return initial, action_values, targets


def build_rollout_metadata(
    *,
    state_source: str,
    source_space_identity: str,
    transition_name: str,
    rollout_horizon: int,
    action_shape: tuple[int],
    state_shape: tuple[int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build common immutable-trajectory provenance before caller overrides."""

    metadata: dict[str, Any] = {
        "state_source": state_source,
        "source_space_identity": source_space_identity,
        "transition": transition_name,
        "rollout_horizon": rollout_horizon,
        "action_shape": action_shape,
        "state_shape": state_shape,
    }
    if extra is not None:
        metadata.update(extra)
    return metadata
