"""Concrete deterministic and stochastic latent transitions.

Sprint 63 deliberately keeps this module narrow.  It models flat Euclidean
states with an action-conditioned affine residual update:

    z_next = z + [z, action, 1] @ weights

The transition classes are intentionally concrete.  Sprint 64 adds a
diagonal-Gaussian residual model without freezing a cross-transition
interface; Sprint 65 will provide the evidence needed before that decision.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
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


@dataclass(frozen=True, slots=True)
class GaussianPrediction:
    """One explicit diagonal-Gaussian prediction.

    ``mean`` and ``scale`` are event-shaped arrays.  The scale is the
    predictive standard deviation, not the covariance of a latent Gaussian
    primitive; ``covariance`` exposes the diagonal predictive covariance for
    callers that need it.  A zero scale is supported for degenerate,
    deterministic noise tests and is sampled exactly.
    """

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = _as_finite_array(self.mean, name="mean")
        scale = _as_finite_array(self.scale, name="scale")
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError(f"mean and scale must have matching shape (state_dim,), got {mean.shape}, {scale.shape}")
        if np.any(scale < 0):
            raise ValueError("scale must be non-negative")
        mean = mean.copy()
        scale = scale.copy()
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @property
    def variance(self) -> np.ndarray:
        """Return the diagonal predictive variance."""

        return np.square(self.scale)

    @property
    def std(self) -> np.ndarray:
        """Return the predictive standard deviation (an alias for ``scale``)."""

        return self.scale

    @property
    def covariance(self) -> np.ndarray:
        """Return the full diagonal predictive covariance matrix."""

        return np.diag(self.variance)

    @property
    def event_shape(self) -> tuple[int]:
        """Return the Gaussian event shape."""

        return self.mean.shape

    @property
    def distribution_family(self) -> str:
        """Return the explicit distribution family name."""

        return "diagonal_gaussian"

    def sample(
        self,
        rng: np.random.Generator | None = None,
        *,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Draw reproducible samples without hiding the predictive scale."""

        if rng is not None and seed is not None:
            raise ValueError("pass either rng or seed, not both")
        if n_samples is not None and n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        generator = rng if rng is not None else np.random.default_rng(seed)
        if n_samples is None:
            return self.mean + self.scale * generator.normal(size=self.mean.shape)
        return self.mean + self.scale * generator.normal(size=(n_samples, *self.mean.shape))

    def log_prob(self, value: np.ndarray) -> float | np.ndarray:
        """Evaluate the diagonal-Gaussian log density for one or many values."""

        values = _as_finite_array(value, name="value")
        if values.shape == self.mean.shape:
            differences = values - self.mean
            return float(np.sum(self._log_density(differences)))
        if values.ndim == 2 and values.shape[1:] == self.mean.shape:
            differences = values - self.mean[None, :]
            return np.sum(self._log_density(differences), axis=1)
        raise ValueError(f"value must have shape {self.mean.shape} or (n, {self.mean.size})")

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Return central coordinate-wise Gaussian interval bounds."""

        if not 0.0 < level < 1.0 or not np.isfinite(level):
            raise ValueError(f"level must be finite and between 0 and 1, got {level}")
        quantile = NormalDist().inv_cdf(0.5 + level / 2.0)
        return self.mean - quantile * self.scale, self.mean + quantile * self.scale

    def _log_density(self, difference: np.ndarray) -> np.ndarray:
        # A tiny effective scale keeps log_prob finite for a point-mass
        # prediction while sample() still preserves exact zero-noise behavior.
        effective_scale = np.maximum(self.scale, 1e-12)
        standardized = np.clip(difference / effective_scale, -1e150, 1e150)
        return -0.5 * (
            np.square(standardized)
            + np.log(2.0 * np.pi)
            + 2.0 * np.log(effective_scale)
        )


@dataclass(frozen=True, slots=True)
class StochasticRollout:
    """Particle rollout plus immutable uncertainty summaries."""

    samples: np.ndarray
    interval_level: float = 0.95
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        values = _as_finite_array(self.samples, name="samples")
        if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] < 1:
            raise ValueError(f"samples must have shape (n_samples, horizon + 1, state_dim), got {values.shape}")
        if not 0.0 < self.interval_level < 1.0 or not np.isfinite(self.interval_level):
            raise ValueError(f"interval_level must be finite and between 0 and 1, got {self.interval_level}")
        copied = values.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "samples", copied)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def mean(self) -> np.ndarray:
        """Return the particle mean path."""

        return np.mean(self.samples, axis=0)

    @property
    def scale(self) -> np.ndarray:
        """Return the particle standard-deviation path."""

        return np.std(self.samples, axis=0)

    @property
    def lower(self) -> np.ndarray:
        """Return the lower uncertainty band using a Gaussian summary."""

        quantile = NormalDist().inv_cdf(0.5 + self.interval_level / 2.0)
        return self.mean - quantile * self.scale

    @property
    def upper(self) -> np.ndarray:
        """Return the upper uncertainty band using a Gaussian summary."""

        quantile = NormalDist().inv_cdf(0.5 + self.interval_level / 2.0)
        return self.mean + quantile * self.scale

    def to_numpy(self) -> np.ndarray:
        """Return a defensive copy of the particle tensor."""

        return self.samples.copy()


@dataclass(frozen=True, slots=True)
class StochasticOneStepMetrics:
    """Likelihood, calibration, and diversity metrics for one-step predictions."""

    negative_log_likelihood: float
    coverage: float
    interval_width: float
    sample_diversity: float
    mean_error: float
    n_samples: int
    runtime_seconds: float

    @property
    def nll(self) -> float:
        """Short alias for negative log-likelihood."""

        return self.negative_log_likelihood


@dataclass(frozen=True, slots=True)
class StochasticRolloutMetrics:
    """Stochastic open-loop metrics indexed by rollout horizon."""

    negative_log_likelihood_by_horizon: tuple[float, ...]
    coverage_by_horizon: tuple[float, ...]
    sample_diversity_by_horizon: tuple[float, ...]
    mean_error_by_horizon: tuple[float, ...]
    mean_negative_log_likelihood: float
    mean_coverage: float
    mean_sample_diversity: float
    final_error: float
    runtime_seconds: float
    stable: bool

    @property
    def horizon(self) -> int:
        """Return the number of predicted transitions evaluated."""

        return len(self.mean_error_by_horizon)

    @property
    def errors_by_horizon(self) -> tuple[float, ...]:
        """Alias matching deterministic rollout metrics."""

        return self.mean_error_by_horizon

    @property
    def nll_by_horizon(self) -> tuple[float, ...]:
        """Short alias for per-horizon negative log-likelihood."""

        return self.negative_log_likelihood_by_horizon

    @property
    def mean_error(self) -> float:
        """Return mean Euclidean error over the evaluated horizons."""

        return float(np.mean(self.mean_error_by_horizon)) if self.mean_error_by_horizon else 0.0


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


class StochasticGaussianLatentTransition:
    """Fit and sample one concrete diagonal-Gaussian latent transition.

    The mean uses the same action-conditioned affine residual family as
    :class:`DeterministicLatentTransition`.  The diagonal scale is fitted
    from held-out residual variation, so uncertainty is returned directly by
    :meth:`predict` as a :class:`GaussianPrediction` rather than being hidden
    in metadata.  This is a deliberately memoryless, flat-Euclidean instance;
    recurrent state and a shared transition protocol remain Sprint 65 scope.
    """

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
        if latent_space.geometry != "euclidean" or latent_space.shape != (latent_space.dim,):
            raise ValueError("StochasticGaussianLatentTransition requires a flat Euclidean LatentSpace")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if ridge < 0 or not np.isfinite(ridge):
            raise ValueError(f"ridge must be finite and >= 0, got {ridge}")
        if variance_floor < 0 or not np.isfinite(variance_floor):
            raise ValueError(f"variance_floor must be finite and >= 0, got {variance_floor}")
        if stability_norm_limit <= 0 or not np.isfinite(stability_norm_limit):
            raise ValueError(f"stability_norm_limit must be finite and > 0, got {stability_norm_limit}")

        self.latent_space = latent_space
        self.action_dim = action_dim
        self.ridge = float(ridge)
        self.variance_floor = float(variance_floor)
        self.stability_norm_limit = float(stability_norm_limit)
        self.source_space_identity = source_space_identity or (
            latent_space.source_model or f"{latent_space.geometry}:{latent_space.dim}"
        )
        if not self.source_space_identity.strip():
            raise ValueError("source_space_identity must be a non-empty string")
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
        state_values = self._validate_batch(states, name="states", width=self.state_dim)
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        target_values = self._validate_batch(next_states, name="next_states", width=self.state_dim)
        if state_values.shape[0] != action_values.shape[0] or state_values.shape[0] != target_values.shape[0]:
            raise ValueError("states, actions, and next_states must have the same number of samples")
        if state_values.shape[0] < 1:
            raise ValueError("at least one transition sample is required")

        design = np.concatenate(
            [state_values, action_values, np.ones((state_values.shape[0], 1), dtype=np.float64)], axis=1
        )
        residual_targets = target_values - state_values
        if self.ridge == 0:
            weights = np.linalg.lstsq(design, residual_targets, rcond=None)[0]
        else:
            gram = design.T @ design
            regularizer = np.eye(gram.shape[0], dtype=np.float64) * self.ridge
            weights = np.linalg.solve(gram + regularizer, design.T @ residual_targets)
        mean_predictions = state_values + design @ weights
        residuals = target_values - mean_predictions
        variances = np.maximum(np.mean(np.square(residuals), axis=0), self.variance_floor)
        self._weights = np.asarray(weights, dtype=np.float64)
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
        state_value = self._validate_point(state, name="state", width=self.state_dim)
        action_value = self._validate_point(action, name="action", width=self.action_dim)
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
        initial = self._validate_point(initial_state, name="initial_state")
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        states = np.empty((action_values.shape[0] + 1, self.state_dim), dtype=np.float64)
        states[0] = initial
        for index, action in enumerate(action_values):
            states[index + 1] = self.step(states[index], action)
        rollout_metadata: dict[str, Any] = {
            "state_source": "predictive_mean",
            "source_space_identity": self.source_space_identity,
            "transition": self.__class__.__name__,
            "rollout_horizon": int(action_values.shape[0]),
            "action_shape": self.action_shape,
            "state_shape": self.state_shape,
            "distribution_family": "diagonal_gaussian",
        }
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
        initial = self._validate_point(initial_state, name="initial_state")
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        generator = rng if rng is not None else np.random.default_rng(seed)
        samples = np.empty((n_samples, action_values.shape[0] + 1, self.state_dim), dtype=np.float64)
        samples[:, 0, :] = initial
        for index, action in enumerate(action_values):
            predictions = self._predict_batch(
                samples[:, index, :], np.broadcast_to(action, (n_samples, self.action_dim))
            )
            means = np.vstack([prediction.mean for prediction in predictions])
            # Keeping the one-point distribution API as the source of scale
            # ensures sample() and rollout() have identical semantics.
            samples[:, index + 1, :] = means + self.scale * generator.normal(
                size=(n_samples, self.state_dim)
            )
        rollout_metadata: dict[str, Any] = {
            "state_source": "sampled",
            "source_space_identity": self.source_space_identity,
            "transition": self.__class__.__name__,
            "rollout_horizon": int(action_values.shape[0]),
            "action_shape": self.action_shape,
            "state_shape": self.state_shape,
            "distribution_family": "diagonal_gaussian",
            "n_samples": int(n_samples),
            "seed": seed,
            "interval_level": interval_level,
        }
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
        state_values = self._validate_batch(states, name="states", width=self.state_dim)
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        target_values = self._validate_batch(next_states, name="next_states", width=self.state_dim)
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
        initial_values, action_values, target_values = self._validate_rollout_inputs(
            initial_state, actions, target_states
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
            np.square(differences / effective_scales)
            + np.log(2.0 * np.pi)
            + 2.0 * np.log(effective_scales)
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

    def _predict_batch(self, states: np.ndarray, actions: np.ndarray) -> tuple[GaussianPrediction, ...]:
        """Return explicit predictions for a validated batch."""

        return tuple(self.predict(state, action) for state, action in zip(states, actions))

    def _require_fitted(self) -> None:
        if self._weights is None or self._scale is None:
            raise RuntimeError("transition must be fitted before prediction")

    @staticmethod
    def _validate_batch(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
        values = _as_finite_array(value, name=name)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError(f"{name} must have shape (n, {width}), got {values.shape}")
        return values

    @staticmethod
    def _validate_point(value: np.ndarray, *, name: str, width: int | None = None) -> np.ndarray:
        values = _as_finite_array(value, name=name)
        if values.ndim != 1 or (width is not None and values.shape != (width,)):
            if width is None:
                raise ValueError(f"{name} must have shape (state_dim,), got {values.shape}")
            raise ValueError(f"{name} must have shape ({width},), got {values.shape}")
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
