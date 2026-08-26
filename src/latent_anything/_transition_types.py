"""Public transition result values kept in a cycle-free internal module."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from types import MappingProxyType
from typing import Any

import numpy as np

from latent_anything._transition_core import as_finite_array


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
    """One explicit diagonal-Gaussian prediction."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = as_finite_array(self.mean, name="mean")
        scale = as_finite_array(self.scale, name="scale")
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

        values = as_finite_array(value, name="value")
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
        effective_scale = np.maximum(self.scale, 1e-12)
        standardized = np.clip(difference / effective_scale, -1e150, 1e150)
        return -0.5 * (np.square(standardized) + np.log(2.0 * np.pi) + 2.0 * np.log(effective_scale))


@dataclass(frozen=True, slots=True)
class StochasticRollout:
    """Particle rollout plus immutable uncertainty summaries."""

    samples: np.ndarray
    interval_level: float = 0.95
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        values = as_finite_array(self.samples, name="samples")
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


__all__ = [
    "GaussianPrediction",
    "OneStepMetrics",
    "RolloutMetrics",
    "StochasticOneStepMetrics",
    "StochasticRollout",
    "StochasticRolloutMetrics",
]
