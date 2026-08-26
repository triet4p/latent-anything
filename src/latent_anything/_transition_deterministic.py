"""Concrete deterministic transition lifecycle."""

from __future__ import annotations

import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from latent_anything._transition_core import (
    build_rollout_metadata,
    fit_affine_residual,
    resolve_source_identity,
    validate_batch,
    validate_constructor,
    validate_fit_samples,
    validate_point,
    validate_rollout_inputs,
)
from latent_anything._transition_types import OneStepMetrics, RolloutMetrics
from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


class DeterministicLatentTransition:
    """Fit and recursively apply one concrete deterministic latent dynamics model."""

    stream_state_contract = "explicit"

    def __init__(
        self,
        latent_space: LatentSpace,
        action_dim: int,
        *,
        source_space_identity: str | None = None,
        ridge: float = 1e-8,
        stability_norm_limit: float = 1e6,
    ) -> None:
        validate_constructor(
            latent_space,
            action_dim,
            transition_name="DeterministicLatentTransition",
            ridge=ridge,
            stability_norm_limit=stability_norm_limit,
        )
        self.latent_space = latent_space
        self.action_dim = action_dim
        self.ridge = float(ridge)
        self.stability_norm_limit = float(stability_norm_limit)
        self.source_space_identity = resolve_source_identity(latent_space, source_space_identity)
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
        """Fit residual affine dynamics on one-step transition samples."""

        if training_horizon < 1:
            raise ValueError(f"training_horizon must be >= 1, got {training_horizon}")
        if source_space_identity is not None and source_space_identity != self.source_space_identity:
            raise ValueError(
                "source_space_identity does not match the transition: "
                f"expected {self.source_space_identity!r}, got {source_space_identity!r}"
            )
        state_values, action_values, target_values = validate_fit_samples(
            states,
            actions,
            next_states,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )
        _design, weights, _residuals = fit_affine_residual(
            state_values,
            action_values,
            target_values,
            ridge=self.ridge,
        )
        self._weights = weights
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
        state_value = validate_point(state, name="state", width=self.state_dim)
        action_value = validate_point(action, name="action", width=self.action_dim)
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
        """Recursively predict a trajectory for an action sequence."""

        self._require_fitted()
        initial = validate_point(initial_state, name="initial_state")
        action_values = validate_batch(actions, name="actions", width=self.action_dim)
        states = np.empty((action_values.shape[0] + 1, self.state_dim), dtype=np.float64)
        states[0] = initial
        for index, action in enumerate(action_values):
            states[index + 1] = self.step(states[index], action)
        rollout_metadata = build_rollout_metadata(
            state_source="predicted",
            source_space_identity=self.source_space_identity,
            transition_name=self.__class__.__name__,
            rollout_horizon=int(action_values.shape[0]),
            action_shape=self.action_shape,
            state_shape=self.state_shape,
        )
        if metadata is not None:
            rollout_metadata.update(dict(metadata))
        return Trajectory(states, metadata=rollout_metadata)

    def mean_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Trajectory:
        """Return the deterministic rollout through the shared transition surface."""

        return self.rollout(initial_state, actions, metadata=metadata)

    def evaluate_one_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> OneStepMetrics:
        """Measure teacher-forced prediction error and runtime."""

        start = time.perf_counter()
        state_values = validate_batch(states, name="states", width=self.state_dim)
        action_values = validate_batch(actions, name="actions", width=self.action_dim)
        target_values = validate_batch(next_states, name="next_states", width=self.state_dim)
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
        """Measure open-loop error by horizon against known target states."""

        start = time.perf_counter()
        initial_values, action_values, target_values = validate_rollout_inputs(
            initial_state,
            actions,
            target_states,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
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

    @staticmethod
    def _validate_batch(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
        return validate_batch(value, name=name, width=width)

    @staticmethod
    def _validate_point(value: np.ndarray, *, name: str, width: int | None = None) -> np.ndarray:
        return validate_point(value, name=name, width=width)

    def _validate_rollout_inputs(
        self, initial_state: np.ndarray, actions: np.ndarray, target_states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return validate_rollout_inputs(
            initial_state,
            actions,
            target_states,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    def _require_fitted(self) -> None:
        if self._weights is None:
            raise RuntimeError("transition must be fitted before prediction")


__all__ = ["DeterministicLatentTransition"]
