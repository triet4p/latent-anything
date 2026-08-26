"""Concrete memoryless diagonal-Gaussian transition lifecycle."""

from __future__ import annotations

import time
from collections.abc import Mapping
from statistics import NormalDist
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
from latent_anything._transition_types import (
    GaussianPrediction,
    StochasticOneStepMetrics,
    StochasticRollout,
    StochasticRolloutMetrics,
)
from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


class StochasticGaussianLatentTransition:
    """Fit and sample one concrete diagonal-Gaussian latent transition."""

    stream_state_contract = "explicit"

    def __init__(
        self,
        latent_space: LatentSpace,
        action_dim: int,
        *,
        source_space_identity: str | None = None,
        ridge: float = 1e-8,
        variance_floor: float = 1e-8,
        stability_norm_limit: float = 1e6,
    ) -> None:
        validate_constructor(
            latent_space,
            action_dim,
            transition_name="StochasticGaussianLatentTransition",
            ridge=ridge,
            variance_floor=variance_floor,
            stability_norm_limit=stability_norm_limit,
        )
        self.latent_space = latent_space
        self.action_dim = action_dim
        self.ridge = float(ridge)
        self.variance_floor = float(variance_floor)
        self.stability_norm_limit = float(stability_norm_limit)
        self.source_space_identity = resolve_source_identity(
            latent_space,
            source_space_identity,
            empty_uses_fallback=True,
        )
        self._weights: np.ndarray | None = None
        self._scale: np.ndarray | None = None
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
        """Whether mean and predictive scale have been fitted."""

        return self._weights is not None and self._scale is not None

    @property
    def fit_metadata(self) -> Mapping[str, Any]:
        """Return immutable metadata describing the last fit."""

        return self._fit_metadata

    @property
    def coefficients(self) -> np.ndarray:
        """Return a defensive copy of the fitted residual coefficients."""

        self._require_fitted()
        return self._weights.copy()  # type: ignore[union-attr]

    @property
    def scale(self) -> np.ndarray:
        """Return a defensive copy of the fitted predictive standard deviation."""

        self._require_fitted()
        return self._scale.copy()  # type: ignore[union-attr]

    @property
    def variance(self) -> np.ndarray:
        """Return a defensive copy of the fitted predictive variance."""

        return np.square(self.scale)

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        *,
        training_horizon: int = 1,
        source_space_identity: str | None = None,
    ) -> StochasticGaussianLatentTransition:
        """Fit affine mean dynamics and diagonal residual variance."""

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
        _design, weights, residuals = fit_affine_residual(
            state_values,
            action_values,
            target_values,
            ridge=self.ridge,
        )
        variances = np.maximum(np.mean(np.square(residuals), axis=0), self.variance_floor)
        self._weights = weights
        self._scale = np.sqrt(np.asarray(variances, dtype=np.float64))
        self._fit_metadata = MappingProxyType(
            {
                "source_space_identity": self.source_space_identity,
                "state_shape": self.state_shape,
                "action_shape": self.action_shape,
                "n_samples": int(state_values.shape[0]),
                "training_horizon": training_horizon,
                "fit_kind": "one_step_residual_affine_diagonal_gaussian",
                "distribution_family": "diagonal_gaussian",
                "uncertainty_type": "aleatoric_residual",
                "covariance_parameterization": "diagonal_scale",
                "variance_floor": self.variance_floor,
                "ridge": self.ridge,
                "residual_variance": tuple(float(value) for value in variances),
            }
        )
        return self

    def predict(self, state: np.ndarray, action: np.ndarray) -> GaussianPrediction:
        """Return an explicit Gaussian prediction for one state/action pair."""

        self._require_fitted()
        state_value = validate_point(state, name="state", width=self.state_dim)
        action_value = validate_point(action, name="action", width=self.action_dim)
        design = np.concatenate([state_value, action_value, np.ones(1, dtype=np.float64)])
        mean = state_value + design @ self._weights  # type: ignore[operator]
        prediction = GaussianPrediction(mean=np.asarray(mean, dtype=np.float64), scale=self.scale)
        if not np.isfinite(prediction.mean).all() or not np.isfinite(prediction.scale).all():
            raise FloatingPointError("transition produced a non-finite Gaussian prediction")
        return prediction

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Return the predictive mean for deterministic-compatible callers."""

        return self.predict(state, action).mean.copy()

    def mean_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Trajectory:
        """Recursively roll out predictive means as an uncertainty-free baseline."""

        self._require_fitted()
        initial = validate_point(initial_state, name="initial_state")
        action_values = validate_batch(actions, name="actions", width=self.action_dim)
        states = np.empty((action_values.shape[0] + 1, self.state_dim), dtype=np.float64)
        states[0] = initial
        for index, action in enumerate(action_values):
            states[index + 1] = self.step(states[index], action)
        rollout_metadata = build_rollout_metadata(
            state_source="predictive_mean",
            source_space_identity=self.source_space_identity,
            transition_name=self.__class__.__name__,
            rollout_horizon=int(action_values.shape[0]),
            action_shape=self.action_shape,
            state_shape=self.state_shape,
            extra={"distribution_family": "diagonal_gaussian"},
        )
        if metadata is not None:
            rollout_metadata.update(dict(metadata))
        return Trajectory(states, metadata=rollout_metadata)

    def rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        n_samples: int = 128,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        interval_level: float = 0.95,
        metadata: Mapping[str, Any] | None = None,
    ) -> StochasticRollout:
        """Sample a particle rollout while retaining mean and uncertainty bands."""

        self._require_fitted()
        if rng is not None and seed is not None:
            raise ValueError("pass either rng or seed, not both")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if not 0.0 < interval_level < 1.0 or not np.isfinite(interval_level):
            raise ValueError(f"interval_level must be finite and between 0 and 1, got {interval_level}")
        initial = validate_point(initial_state, name="initial_state")
        action_values = validate_batch(actions, name="actions", width=self.action_dim)
        generator = rng if rng is not None else np.random.default_rng(seed)
        samples = np.empty((n_samples, action_values.shape[0] + 1, self.state_dim), dtype=np.float64)
        samples[:, 0, :] = initial
        for index, action in enumerate(action_values):
            predictions = self._predict_batch(
                samples[:, index, :], np.broadcast_to(action, (n_samples, self.action_dim))
            )
            means = np.vstack([prediction.mean for prediction in predictions])
            samples[:, index + 1, :] = means + self.scale * generator.normal(size=(n_samples, self.state_dim))
        rollout_metadata = build_rollout_metadata(
            state_source="sampled",
            source_space_identity=self.source_space_identity,
            transition_name=self.__class__.__name__,
            rollout_horizon=int(action_values.shape[0]),
            action_shape=self.action_shape,
            state_shape=self.state_shape,
            extra={
                "distribution_family": "diagonal_gaussian",
                "n_samples": int(n_samples),
                "seed": seed,
                "interval_level": interval_level,
            },
        )
        if metadata is not None:
            rollout_metadata.update(dict(metadata))
        return StochasticRollout(samples, interval_level=interval_level, metadata=rollout_metadata)

    def evaluate_one_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        *,
        interval_level: float = 0.95,
        n_diversity_samples: int = 64,
        seed: int = 0,
    ) -> StochasticOneStepMetrics:
        """Measure NLL, interval coverage, diversity, and mean error."""

        start = time.perf_counter()
        state_values = validate_batch(states, name="states", width=self.state_dim)
        action_values = validate_batch(actions, name="actions", width=self.action_dim)
        target_values = validate_batch(next_states, name="next_states", width=self.state_dim)
        if not (state_values.shape[0] == action_values.shape[0] == target_values.shape[0]):
            raise ValueError("states, actions, and next_states must have the same number of samples")
        if n_diversity_samples < 2:
            raise ValueError(f"n_diversity_samples must be >= 2, got {n_diversity_samples}")
        predictions = self._predict_batch(state_values, action_values)
        log_probs = np.asarray([prediction.log_prob(target) for prediction, target in zip(predictions, target_values)])
        lower, upper = zip(*(prediction.interval(interval_level) for prediction in predictions))
        lower_values = np.asarray(lower)
        upper_values = np.asarray(upper)
        coverage = np.mean((target_values >= lower_values) & (target_values <= upper_values))
        widths = upper_values - lower_values
        diversity = np.mean(
            [
                np.mean(np.std(prediction.sample(seed=seed + index, n_samples=n_diversity_samples), axis=0))
                for index, prediction in enumerate(predictions)
            ]
        )
        errors = np.linalg.norm(np.asarray([prediction.mean for prediction in predictions]) - target_values, axis=1)
        runtime = time.perf_counter() - start
        return StochasticOneStepMetrics(
            negative_log_likelihood=float(-np.mean(log_probs)),
            coverage=float(coverage),
            interval_width=float(np.mean(widths)),
            sample_diversity=float(diversity),
            mean_error=float(np.mean(errors)),
            n_samples=int(state_values.shape[0]),
            runtime_seconds=float(runtime),
        )

    def evaluate_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        target_states: np.ndarray,
        *,
        n_samples: int = 256,
        seed: int = 0,
        interval_level: float = 0.95,
    ) -> StochasticRolloutMetrics:
        """Measure NLL, coverage, diversity, and mean drift by horizon."""

        start = time.perf_counter()
        initial_values, action_values, target_values = validate_rollout_inputs(
            initial_state,
            actions,
            target_states,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        if not 0.0 < interval_level < 1.0 or not np.isfinite(interval_level):
            raise ValueError(f"interval_level must be finite and between 0 and 1, got {interval_level}")
        batch_size, horizon, _ = action_values.shape
        if horizon == 0:
            return StochasticRolloutMetrics((), (), (), (), 0.0, 1.0, 0.0, 0.0, time.perf_counter() - start, True)

        means = np.empty((batch_size, horizon, self.state_dim), dtype=np.float64)
        scales = np.empty_like(means)
        for episode in range(batch_size):
            rollout = self.rollout(
                initial_values[episode],
                action_values[episode],
                n_samples=n_samples,
                seed=seed + episode,
                interval_level=interval_level,
            )
            means[episode] = rollout.mean[1:]
            scales[episode] = rollout.scale[1:]
        targets = target_values[:, 1:, :]
        effective_scales = np.maximum(scales, max(self.variance_floor**0.5, np.finfo(np.float64).eps))
        differences = targets - means
        log_density = -0.5 * (
            np.square(differences / effective_scales) + np.log(2.0 * np.pi) + 2.0 * np.log(effective_scales)
        )
        quantile = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
        lower = means - quantile * scales
        upper = means + quantile * scales
        nll_by_horizon = tuple(float(value) for value in np.mean(-np.sum(log_density, axis=2), axis=0))
        coverage_by_horizon = tuple(
            float(value) for value in np.mean((targets >= lower) & (targets <= upper), axis=(0, 2))
        )
        diversity_by_horizon = tuple(float(value) for value in np.mean(scales, axis=(0, 2)))
        errors_by_horizon = tuple(float(value) for value in np.mean(np.linalg.norm(differences, axis=2), axis=0))
        max_norm = float(np.max(np.linalg.norm(means, axis=2)))
        runtime = time.perf_counter() - start
        return StochasticRolloutMetrics(
            negative_log_likelihood_by_horizon=nll_by_horizon,
            coverage_by_horizon=coverage_by_horizon,
            sample_diversity_by_horizon=diversity_by_horizon,
            mean_error_by_horizon=errors_by_horizon,
            mean_negative_log_likelihood=float(np.mean(nll_by_horizon)),
            mean_coverage=float(np.mean(coverage_by_horizon)),
            mean_sample_diversity=float(np.mean(diversity_by_horizon)),
            final_error=errors_by_horizon[-1],
            runtime_seconds=float(runtime),
            stable=bool(np.isfinite(means).all() and max_norm <= self.stability_norm_limit),
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

    def _predict_batch(self, states: np.ndarray, actions: np.ndarray) -> tuple[GaussianPrediction, ...]:
        """Return explicit predictions for a validated batch."""

        return tuple(self.predict(state, action) for state, action in zip(states, actions))

    def _require_fitted(self) -> None:
        if self._weights is None or self._scale is None:
            raise RuntimeError("transition must be fitted before prediction")


__all__ = ["StochasticGaussianLatentTransition"]
