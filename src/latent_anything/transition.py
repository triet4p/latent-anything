"""A first concrete deterministic latent transition and rollout evaluator.

Sprint 63 deliberately keeps this module narrow.  It models flat Euclidean
states with an action-conditioned affine residual update:

    z_next = z + [z, action, 1] @ weights

The class is intentionally concrete; Sprints 64 and 65 will provide the
evidence needed before a shared transition contract is considered.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


def _as_finite_array(value: object, *, name: str) -> np.ndarray:
    """Return *value* as an array and reject non-finite numeric data."""

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(value).__name__}")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class OneStepMetrics:
    """Error and runtime measurements for teacher-forced one-step prediction."""

    mse: float
    rmse: float
    max_error: float
    n_samples: int
    runtime_seconds: float


@dataclass(frozen=True)
class RolloutMetrics:
    """Open-loop error and stability measurements indexed by rollout horizon."""

    errors_by_horizon: tuple[float, ...]
    mean_error: float
    final_error: float
    max_error: float
    max_state_norm: float
    runtime_seconds: float
    stable: bool

    @property
    def horizon(self) -> int:
        """Return the number of predicted transitions evaluated."""

        return len(self.errors_by_horizon)


class DeterministicLatentTransition:
    """Fit and recursively apply one concrete deterministic latent dynamics model.

    Parameters
    ----------
    latent_space : LatentSpace
        The source space.  Sprint 63 accepts only flat Euclidean states.
    action_dim : int
        Number of scalar action coordinates.
    source_space_identity : str, optional
        Stable caller-provided identity for the state representation.  When
        omitted, the source model name is used, followed by a deterministic
        geometry/dimension fallback.
    ridge : float, default=1e-8
        Non-negative Tikhonov regularization used by the deterministic least
        squares fit.
    stability_norm_limit : float, default=1e6
        Maximum finite state norm considered stable during evaluation.

    Notes
    -----
    ``fit`` learns one-step residual dynamics from ``(z_t, a_t, z_{t+1})``.
    ``rollout`` then feeds each predicted state into the next step and returns
    an immutable :class:`~latent_anything.trajectory.Trajectory` containing
    the initial state plus one state per action.
    """

    def __init__(
        self,
        latent_space: LatentSpace,
        action_dim: int,
        *,
        source_space_identity: str | None = None,
        ridge: float = 1e-8,
        stability_norm_limit: float = 1e6,
    ) -> None:
        if latent_space.geometry != "euclidean" or latent_space.shape != (latent_space.dim,):
            raise ValueError("DeterministicLatentTransition requires a flat Euclidean LatentSpace")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if ridge < 0 or not np.isfinite(ridge):
            raise ValueError(f"ridge must be finite and >= 0, got {ridge}")
        if stability_norm_limit <= 0 or not np.isfinite(stability_norm_limit):
            raise ValueError(f"stability_norm_limit must be finite and > 0, got {stability_norm_limit}")

        self.latent_space = latent_space
        self.action_dim = action_dim
        self.ridge = float(ridge)
        self.stability_norm_limit = float(stability_norm_limit)
        resolved_identity = (
            source_space_identity
            if source_space_identity is not None
            else (latent_space.source_model or f"{latent_space.geometry}:{latent_space.dim}")
        )
        self.source_space_identity = resolved_identity
        if not self.source_space_identity.strip():
            raise ValueError("source_space_identity must be a non-empty string")
        self._weights: np.ndarray | None = None
        self._fit_metadata: Mapping[str, Any] = MappingProxyType({})

    @property
    def state_dim(self) -> int:
        """Return the flat latent state dimension."""

        return self.latent_space.dim

    @property
    def state_shape(self) -> tuple[int]:
        """Return the required shape of one latent state."""

        return (self.state_dim,)

    @property
    def action_shape(self) -> tuple[int]:
        """Return the required shape of one action."""

        return (self.action_dim,)

    @property
    def is_fitted(self) -> bool:
        """Whether one-step coefficients have been fitted."""

        return self._weights is not None

    @property
    def fit_metadata(self) -> Mapping[str, Any]:
        """Return immutable metadata describing the last fit."""

        return self._fit_metadata

    @property
    def coefficients(self) -> np.ndarray:
        """Return a defensive copy of the fitted residual coefficients."""

        self._require_fitted()
        return self._weights.copy()  # type: ignore[union-attr]

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        *,
        training_horizon: int = 1,
        source_space_identity: str | None = None,
    ) -> DeterministicLatentTransition:
        """Fit residual affine dynamics on one-step transition samples.

        ``states``, ``actions`` and ``next_states`` must have matching first
        dimensions and shapes ``(n, state_dim)``, ``(n, action_dim)`` and
        ``(n, state_dim)`` respectively.  The ``training_horizon`` is
        recorded as provenance; this first instance fits one-step targets and
        does not silently claim multi-step training.
        """

        if training_horizon < 1:
            raise ValueError(f"training_horizon must be >= 1, got {training_horizon}")
        if source_space_identity is not None and source_space_identity != self.source_space_identity:
            raise ValueError(
                "source_space_identity does not match the transition: "
                f"expected {self.source_space_identity!r}, got {source_space_identity!r}"
            )
        state_values = self._validate_batch(states, name="states", width=self.state_dim)
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        target_values = self._validate_batch(next_states, name="next_states", width=self.state_dim)
        if state_values.shape[0] != action_values.shape[0] or state_values.shape[0] != target_values.shape[0]:
            raise ValueError("states, actions, and next_states must have the same number of samples")
        if state_values.shape[0] < 1:
            raise ValueError("at least one transition sample is required")

        design = np.concatenate(
            [state_values, action_values, np.ones((state_values.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        residual_targets = target_values - state_values
        if self.ridge == 0:
            weights = np.linalg.lstsq(design, residual_targets, rcond=None)[0]
        else:
            gram = design.T @ design
            regularizer = np.eye(gram.shape[0], dtype=np.float64) * self.ridge
            weights = np.linalg.solve(gram + regularizer, design.T @ residual_targets)
        self._weights = np.asarray(weights, dtype=np.float64)
        self._fit_metadata = MappingProxyType(
            {
                "source_space_identity": self.source_space_identity,
                "state_shape": self.state_shape,
                "action_shape": self.action_shape,
                "n_samples": int(state_values.shape[0]),
                "training_horizon": training_horizon,
                "fit_kind": "one_step_residual_affine",
                "ridge": self.ridge,
            }
        )
        return self

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict one next latent state from one state/action pair."""

        self._require_fitted()
        state_value = self._validate_point(state, name="state", width=self.state_dim)
        action_value = self._validate_point(action, name="action", width=self.action_dim)
        design = np.concatenate([state_value, action_value, np.ones(1, dtype=np.float64)])
        next_state = state_value + design @ self._weights  # type: ignore[operator]
        if not np.isfinite(next_state).all():
            raise FloatingPointError("transition produced a non-finite next state")
        return np.asarray(next_state, dtype=np.float64)

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Alias for :meth:`step` using prediction terminology."""

        return self.step(state, action)

    def rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Trajectory:
        """Recursively predict a trajectory for an action sequence.

        ``actions`` has shape ``(horizon, action_dim)``.  The result has shape
        ``(horizon + 1, state_dim)`` and carries source, horizon, action-shape,
        and predicted-state provenance in immutable trajectory metadata.
        """

        self._require_fitted()
        initial = self._validate_point(initial_state, name="initial_state")
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        states = np.empty((action_values.shape[0] + 1, self.state_dim), dtype=np.float64)
        states[0] = initial
        for index, action in enumerate(action_values):
            states[index + 1] = self.step(states[index], action)
        rollout_metadata: dict[str, Any] = {
            "state_source": "predicted",
            "source_space_identity": self.source_space_identity,
            "transition": self.__class__.__name__,
            "rollout_horizon": int(action_values.shape[0]),
            "action_shape": self.action_shape,
            "state_shape": self.state_shape,
        }
        if metadata is not None:
            rollout_metadata.update(dict(metadata))
        return Trajectory(states, metadata=rollout_metadata)

    def evaluate_one_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> OneStepMetrics:
        """Measure teacher-forced prediction error and runtime."""

        start = time.perf_counter()
        state_values = self._validate_batch(states, name="states", width=self.state_dim)
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        target_values = self._validate_batch(next_states, name="next_states", width=self.state_dim)
        if not (state_values.shape[0] == action_values.shape[0] == target_values.shape[0]):
            raise ValueError("states, actions, and next_states must have the same number of samples")
        predictions = np.vstack([self.step(state, action) for state, action in zip(state_values, action_values)])
        errors = np.linalg.norm(predictions - target_values, axis=1)
        runtime = time.perf_counter() - start
        mse = float(np.mean((predictions - target_values) ** 2))
        return OneStepMetrics(
            mse=mse,
            rmse=float(np.sqrt(mse)),
            max_error=float(np.max(errors)),
            n_samples=int(state_values.shape[0]),
            runtime_seconds=float(runtime),
        )

    def evaluate_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        target_states: np.ndarray,
    ) -> RolloutMetrics:
        """Measure open-loop error by horizon against known target states.

        A single episode uses shapes ``(d,)``, ``(h, a)``, and ``(h+1, d)``.
        Batched evaluation is also supported with shapes ``(n, d)``,
        ``(n, h, a)``, and ``(n, h+1, d)``; errors are averaged over episodes.
        """

        start = time.perf_counter()
        initial_values, action_values, target_values = self._validate_rollout_inputs(
            initial_state, actions, target_states
        )
        batch_size, horizon, _ = action_values.shape
        predictions = np.empty_like(target_values)
        predictions[:, 0] = initial_values
        for episode in range(batch_size):
            for index in range(horizon):
                predictions[episode, index + 1] = self.step(predictions[episode, index], action_values[episode, index])
        errors = np.linalg.norm(predictions[:, 1:] - target_values[:, 1:], axis=2)
        state_norms = np.linalg.norm(predictions, axis=2)
        runtime = time.perf_counter() - start
        if horizon == 0:
            empty = tuple[float, ...]()
            return RolloutMetrics(empty, 0.0, 0.0, 0.0, float(np.max(state_norms)), float(runtime), True)
        errors_by_horizon = tuple(float(value) for value in np.mean(errors, axis=0))
        max_state_norm = float(np.max(state_norms))
        return RolloutMetrics(
            errors_by_horizon=errors_by_horizon,
            mean_error=float(np.mean(errors)),
            final_error=errors_by_horizon[-1],
            max_error=float(np.max(errors)),
            max_state_norm=max_state_norm,
            runtime_seconds=float(runtime),
            stable=bool(np.isfinite(predictions).all() and max_state_norm <= self.stability_norm_limit),
        )

    def _require_fitted(self) -> None:
        if self._weights is None:
            raise RuntimeError("transition must be fitted before prediction")

    @staticmethod
    def _validate_batch(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
        values = _as_finite_array(value, name=name)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError(f"{name} must have shape (n, {width}), got {values.shape}")
        return values

    @staticmethod
    def _validate_point(value: np.ndarray, *, name: str, width: int | None = None) -> np.ndarray:
        expected_width = width
        values = _as_finite_array(value, name=name)
        if values.ndim != 1 or (expected_width is not None and values.shape != (expected_width,)):
            if expected_width is None:
                raise ValueError(f"{name} must have shape (state_dim,), got {values.shape}")
            raise ValueError(f"{name} must have shape ({expected_width},), got {values.shape}")
        return values

    def _validate_rollout_inputs(
        self, initial_state: np.ndarray, actions: np.ndarray, target_states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        initial = _as_finite_array(initial_state, name="initial_state")
        action_values = _as_finite_array(actions, name="actions")
        targets = _as_finite_array(target_states, name="target_states")
        if initial.ndim == 1:
            initial = initial[None, :]
        if action_values.ndim == 2:
            action_values = action_values[None, :, :]
        if targets.ndim == 2:
            targets = targets[None, :, :]
        if initial.ndim != 2 or initial.shape[1] != self.state_dim:
            raise ValueError(
                f"initial_state must have shape (n, {self.state_dim}) or ({self.state_dim},), got {initial.shape}"
            )
        if action_values.ndim != 3 or action_values.shape[2] != self.action_dim:
            raise ValueError(
                f"actions must have shape (n, h, {self.action_dim}) or "
                f"(h, {self.action_dim}), got {action_values.shape}"
            )
        if targets.ndim != 3 or targets.shape[2] != self.state_dim:
            raise ValueError(
                f"target_states must have shape (n, h+1, {self.state_dim}) or "
                f"(h+1, {self.state_dim}), got {targets.shape}"
            )
        if initial.shape[0] != action_values.shape[0] or initial.shape[0] != targets.shape[0]:
            raise ValueError("initial_state, actions, and target_states must have matching batch sizes")
        if action_values.shape[1] + 1 != targets.shape[1]:
            raise ValueError("target_states must contain the initial state plus one state per action")
        if not np.array_equal(initial, targets[:, 0, :]):
            raise ValueError("initial_state must equal target_states[:, 0, :] for rollout evaluation")
        return initial, action_values, targets
