"""A compact NumPy-facing RSSM-style latent transition.

The implementation uses a learned tanh deterministic state and a diagonal
Gaussian stochastic next-latent head.  Torch is used only for the bounded
internal fit; callers exchange NumPy arrays, just like the earlier transition
instances.  This is intentionally a small RSSM-style model rather than a
claim of full Dreamer/RSSM posterior inference: the KL diagnostic uses an
observation-centred posterior proxy and is reported explicitly as such.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

from latent_anything._rssm_checkpoint import read_rssm_checkpoint, write_rssm_checkpoint
from latent_anything._rssm_evaluation import aggregate_one_step_metrics, aggregate_rollout_metrics
from latent_anything._rssm_runtime import (
    build_rollout_metadata,
    recurrent_step,
    sample_recurrent_rollout,
    teacher_forced_distribution_arrays,
)
from latent_anything._rssm_training import fit_rssm_parameters, teacher_forced_predictions
from latent_anything._rssm_validation import (
    finite_array as _finite_array,
)
from latent_anything._rssm_validation import (
    validate_batch,
    validate_one_step_sequences,
    validate_point,
    validate_rollout_inputs,
    validate_sequences,
)
from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


class RSSMTransitionConfig(BaseModel):
    """Reproducible fit/runtime configuration for :class:`RSSMLatentTransition`."""

    hidden_dim: int = Field(default=16, gt=0)
    epochs: int = Field(default=160, gt=0)
    learning_rate: float = Field(default=0.01, gt=0)
    variance_floor: float = Field(default=1e-6, ge=0)
    posterior_scale_factor: float = Field(default=0.5, gt=0)
    stability_norm_limit: float = Field(default=1e6, gt=0)
    seed: int = 65
    device: str = "cpu"

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("device must be a non-empty string")
        return value


@dataclass(frozen=True, slots=True)
class RSSMPrediction:
    """One RSSM-style Gaussian prediction with its deterministic state."""

    mean: np.ndarray
    scale: np.ndarray
    deterministic_state: np.ndarray

    def __post_init__(self) -> None:
        mean = _finite_array(self.mean, name="mean")
        scale = _finite_array(self.scale, name="scale")
        deterministic = _finite_array(self.deterministic_state, name="deterministic_state")
        if mean.ndim != 1 or scale.shape != mean.shape or deterministic.ndim != 1:
            raise ValueError("RSSMPrediction arrays must be one-dimensional with matching mean/scale shapes")
        if np.any(scale < 0):
            raise ValueError("scale must be non-negative")
        for name, value in (("mean", mean), ("scale", scale), ("deterministic_state", deterministic)):
            immutable = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)

    @property
    def variance(self) -> np.ndarray:
        """Return the elementwise predictive variance, ``scale ** 2``."""
        return np.square(self.scale)

    @property
    def covariance(self) -> np.ndarray:
        """Return the diagonal covariance matrix for this prediction."""
        return np.diag(self.variance)

    @property
    def event_shape(self) -> tuple[int]:
        """Return the one-dimensional event shape of the predicted state."""
        return self.mean.shape

    @property
    def distribution_family(self) -> str:
        """Return the distribution label used by this diagonal Gaussian."""
        return "diagonal_gaussian"

    def sample(self, *, seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw one state sample, using either a seed or an existing generator."""
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        generator = rng if rng is not None else np.random.default_rng(seed)
        return self.mean + self.scale * generator.normal(size=self.mean.shape)

    def log_prob(self, value: np.ndarray) -> float:
        """Return the diagonal-Gaussian log probability of a matching state."""
        values = _finite_array(value, name="value")
        if values.shape != self.mean.shape:
            raise ValueError(f"value must have shape {self.mean.shape}, got {values.shape}")
        effective_scale = np.maximum(self.scale, 1e-12)
        difference = values - self.mean
        return float(
            np.sum(
                -0.5 * (np.square(difference / effective_scale) + np.log(2.0 * np.pi) + 2.0 * np.log(effective_scale))
            )
        )

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Return elementwise normal interval bounds for a level in ``(0, 1)``."""
        if not 0.0 < level < 1.0 or not np.isfinite(level):
            raise ValueError("level must be finite and between 0 and 1")
        quantile = 1.959963984540054
        if level != 0.95:
            from statistics import NormalDist

            quantile = NormalDist().inv_cdf(0.5 + level / 2.0)
        return self.mean - quantile * self.scale, self.mean + quantile * self.scale

    def kl_to_observation(self, observation: np.ndarray, *, posterior_scale_factor: float = 0.5) -> float:
        """Return KL from an observation-centred proxy posterior to the prior."""

        target = _finite_array(observation, name="observation")
        if target.shape != self.mean.shape:
            raise ValueError(f"observation must have shape {self.mean.shape}, got {target.shape}")
        if posterior_scale_factor <= 0 or not np.isfinite(posterior_scale_factor):
            raise ValueError("posterior_scale_factor must be finite and > 0")
        prior_variance = np.maximum(self.variance, 1e-12)
        posterior_scale = np.maximum(self.scale * posterior_scale_factor, 1e-6)
        posterior_variance = np.square(posterior_scale)
        return float(
            0.5
            * np.sum(
                np.log(prior_variance / posterior_variance)
                + (posterior_variance + np.square(target - self.mean)) / prior_variance
                - 1.0
            )
        )


@dataclass(frozen=True, slots=True)
class RSSMRollout:
    """Particle rollout retaining both stochastic and deterministic paths."""

    samples: np.ndarray
    deterministic_states: np.ndarray
    interval_level: float = 0.95
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        samples = _finite_array(self.samples, name="samples")
        deterministic = _finite_array(self.deterministic_states, name="deterministic_states")
        if samples.ndim != 3 or deterministic.ndim != 3 or samples.shape[:2] != deterministic.shape[:2]:
            raise ValueError("samples and deterministic_states must be shaped (n_samples, horizon + 1, dim)")
        if not 0.0 < self.interval_level < 1.0:
            raise ValueError("interval_level must be between 0 and 1")
        for name, value in (("samples", samples), ("deterministic_states", deterministic)):
            immutable = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def mean(self) -> np.ndarray:
        """Return the particle mean for each rollout time and state dimension."""
        return np.mean(self.samples, axis=0)

    @property
    def scale(self) -> np.ndarray:
        """Return particle standard deviation for each rollout time and dimension."""
        return np.std(self.samples, axis=0)

    @property
    def lower(self) -> np.ndarray:
        """Return the fixed 95% lower normal bound across rollout particles."""
        return self.mean - 1.959963984540054 * self.scale

    @property
    def upper(self) -> np.ndarray:
        """Return the fixed 95% upper normal bound across rollout particles."""
        return self.mean + 1.959963984540054 * self.scale

    def to_numpy(self) -> np.ndarray:
        """Return a writable copy of samples shaped ``(samples, time, dim)``."""
        return self.samples.copy()


@dataclass(frozen=True, slots=True)
class RSSMOneStepMetrics:
    """Temporal one-step prediction, calibration, and KL diagnostics."""

    mse: float
    rmse: float
    negative_log_likelihood: float
    kl_divergence: float
    coverage: float
    mean_error: float
    n_samples: int
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class RSSMRolloutMetrics:
    """Masked open-loop RSSM metrics indexed by valid horizon."""

    errors_by_horizon: tuple[float, ...]
    kl_by_horizon: tuple[float, ...]
    coverage_by_horizon: tuple[float, ...]
    mean_error: float
    final_error: float
    mean_kl: float
    mean_coverage: float
    runtime_seconds: float
    stable: bool

    @property
    def horizon(self) -> int:
        """Return the number of evaluated rollout steps."""
        return len(self.errors_by_horizon)


class RSSMLatentTransition:
    """Fit and execute a compact recurrent stochastic latent transition.

    ``states`` are shaped ``(episodes, horizon + 1, state_dim)`` and actions
    ``(episodes, horizon, action_dim)``.  ``sequence_mask`` is optional and is
    shaped ``(episodes, horizon)``; false entries are padding and do not update
    recurrent state or contribute to the fit/evaluation loss.
    """

    def __init__(
        self,
        latent_space: LatentSpace,
        action_dim: int,
        *,
        source_space_identity: str | None = None,
        config: RSSMTransitionConfig | None = None,
        hidden_dim: int = 16,
        epochs: int = 160,
        learning_rate: float = 0.01,
        variance_floor: float = 1e-6,
        posterior_scale_factor: float = 0.5,
        stability_norm_limit: float = 1e6,
        seed: int = 65,
        device: str = "cpu",
    ) -> None:
        if latent_space.geometry != "euclidean" or latent_space.shape != (latent_space.dim,):
            raise ValueError("RSSMLatentTransition requires a flat Euclidean LatentSpace")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        supplied = config or RSSMTransitionConfig(
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            variance_floor=variance_floor,
            posterior_scale_factor=posterior_scale_factor,
            stability_norm_limit=stability_norm_limit,
            seed=seed,
            device=device,
        )
        self.latent_space = latent_space
        self.action_dim = action_dim
        self.config = supplied
        self.device = self._resolve_device(supplied.device)
        self.source_space_identity = source_space_identity or (
            latent_space.source_model or f"{latent_space.geometry}:{latent_space.dim}"
        )
        if not self.source_space_identity.strip():
            raise ValueError("source_space_identity must be a non-empty string")
        self._recurrent_weights: np.ndarray | None = None
        self._recurrent_bias: np.ndarray | None = None
        self._emission_weights: np.ndarray | None = None
        self._emission_bias: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._fit_metadata: Mapping[str, Any] = MappingProxyType({})
        self._hidden_state: np.ndarray | None = None

    @property
    def state_dim(self) -> int:
        """Return the flat Euclidean latent-state width."""
        return self.latent_space.dim

    @property
    def state_shape(self) -> tuple[int]:
        """Return the latent-state shape expected by transition calls."""
        return (self.state_dim,)

    @property
    def action_shape(self) -> tuple[int]:
        """Return the action shape expected by transition calls."""
        return (self.action_dim,)

    @property
    def hidden_shape(self) -> tuple[int]:
        """Return the recurrent hidden-state shape configured for this model."""
        return (self.config.hidden_dim,)

    @property
    def is_fitted(self) -> bool:
        """Return whether fitted emission scale parameters are available."""
        return self._scale is not None

    @property
    def fit_metadata(self) -> Mapping[str, Any]:
        """Return immutable provenance and training metadata for the fitted model."""
        return self._fit_metadata

    @property
    def scale(self) -> np.ndarray:
        """Return a copy of fitted per-dimension predictive standard deviations."""
        self._require_fitted()
        return self._scale.copy()  # type: ignore[union-attr]

    @property
    def hidden_state(self) -> np.ndarray:
        """Return a copy of the current hidden state, or its zero initialization."""
        if self._hidden_state is None:
            return np.zeros(self.hidden_shape, dtype=np.float64)
        return self._hidden_state.copy()

    def reset(self, hidden_state: np.ndarray | None = None) -> None:
        """Reset the recurrent state, optionally to a validated hidden vector."""

        if hidden_state is None:
            self._hidden_state = np.zeros(self.hidden_shape, dtype=np.float64)
            return
        value = _finite_array(hidden_state, name="hidden_state")
        if value.shape != self.hidden_shape:
            raise ValueError(f"hidden_state must have shape {self.hidden_shape}, got {value.shape}")
        self._hidden_state = value.copy()

    def to_config(self) -> RSSMTransitionConfig:
        """Return the effective serializable configuration with resolved device."""
        return self.config.model_copy(update={"device": self.device})

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        sequence_mask: np.ndarray | None = None,
        seed: int | None = None,
    ) -> RSSMLatentTransition:
        """Fit recurrent dynamics with masked variable-length sequences."""

        state_values, action_values, mask = self._validate_sequences(states, actions, sequence_mask)
        if not np.any(mask):
            raise ValueError("sequence_mask must contain at least one valid transition")
        fit_seed = self.config.seed if seed is None else seed
        fitted = fit_rssm_parameters(
            state_values,
            action_values,
            mask,
            hidden_dim=self.config.hidden_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            variance_floor=self.config.variance_floor,
            device=self.device,
            seed=fit_seed,
        )
        self._recurrent_weights = fitted.recurrent_weights
        self._recurrent_bias = fitted.recurrent_bias
        self._emission_weights = fitted.emission_weights
        self._emission_bias = fitted.emission_bias
        self._scale = fitted.scale
        self._fit_metadata = MappingProxyType(
            {
                "source_space_identity": self.source_space_identity,
                "state_shape": self.state_shape,
                "action_shape": self.action_shape,
                "hidden_shape": self.hidden_shape,
                "episodes": int(state_values.shape[0]),
                "sequence_length": int(state_values.shape[1] - 1),
                "valid_transitions": int(np.sum(mask)),
                "fit_kind": "rssm_style_tanh_recurrent_diagonal_gaussian",
                "model_family": "rssm_style",
                "posterior": "observation_centered_proxy",
                "device": self.device,
                "seed": int(fit_seed),
                "epochs": self.config.epochs,
                "final_training_mse": fitted.final_loss,
            }
        )
        self.reset()
        return self

    def predict(self, state: np.ndarray, action: np.ndarray) -> RSSMPrediction:
        """Advance the stateful recurrent model and return a Gaussian prediction."""

        self._require_fitted()
        state_value = self._validate_point(state, name="state", width=self.state_dim)
        action_value = self._validate_point(action, name="action", width=self.action_dim)
        if self._hidden_state is None:
            self.reset()
        assert self._hidden_state is not None
        assert self._recurrent_weights is not None and self._recurrent_bias is not None
        assert self._emission_weights is not None and self._emission_bias is not None
        hidden, mean = recurrent_step(
            self._hidden_state,
            state_value,
            action_value,
            recurrent_weights=self._recurrent_weights,
            recurrent_bias=self._recurrent_bias,
            emission_weights=self._emission_weights,
            emission_bias=self._emission_bias,
        )
        self._hidden_state = hidden
        return RSSMPrediction(mean=mean, scale=self.scale, deterministic_state=hidden)

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Advance the fitted model and return the predictive mean state vector."""
        return self.predict(state, action).mean.copy()

    def mean_rollout(
        self, initial_state: np.ndarray, actions: np.ndarray, *, metadata: Mapping[str, Any] | None = None
    ) -> Trajectory:
        """Return a deterministic predictive-mean trajectory for an action sequence."""
        self._require_fitted()
        assert self._recurrent_weights is not None and self._recurrent_bias is not None
        assert self._emission_weights is not None and self._emission_bias is not None
        initial = self._validate_point(initial_state, name="initial_state", width=self.state_dim)
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        self.reset()
        states = np.empty((action_values.shape[0] + 1, self.state_dim), dtype=np.float64)
        states[0] = initial
        for index, action in enumerate(action_values):
            states[index + 1] = self.step(states[index], action)
        values = build_rollout_metadata(
            state_source="predictive_mean",
            source_space_identity=self.source_space_identity,
            transition_name=self.__class__.__name__,
            horizon=action_values.shape[0],
            action_shape=self.action_shape,
            state_shape=self.state_shape,
            hidden_shape=self.hidden_shape,
            metadata=metadata,
        )
        return Trajectory(states, metadata=values)

    def rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        n_samples: int = 128,
        seed: int | None = None,
        interval_level: float = 0.95,
        metadata: Mapping[str, Any] | None = None,
    ) -> RSSMRollout:
        """Reset and recursively sample a recurrent particle rollout."""

        self._require_fitted()
        assert self._recurrent_weights is not None and self._recurrent_bias is not None
        assert self._emission_weights is not None and self._emission_bias is not None
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if not 0.0 < interval_level < 1.0:
            raise ValueError("interval_level must be between 0 and 1")
        initial = self._validate_point(initial_state, name="initial_state", width=self.state_dim)
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        samples, deterministic, final_hidden = sample_recurrent_rollout(
            initial,
            action_values,
            n_samples=n_samples,
            hidden_dim=self.config.hidden_dim,
            scale=self.scale,
            seed=seed,
            recurrent_weights=self._recurrent_weights,
            recurrent_bias=self._recurrent_bias,
            emission_weights=self._emission_weights,
            emission_bias=self._emission_bias,
        )
        self._hidden_state = final_hidden
        values = build_rollout_metadata(
            state_source="sampled",
            source_space_identity=self.source_space_identity,
            transition_name=self.__class__.__name__,
            horizon=action_values.shape[0],
            action_shape=self.action_shape,
            state_shape=self.state_shape,
            hidden_shape=self.hidden_shape,
            metadata=metadata,
            n_samples=n_samples,
            seed=seed,
            interval_level=interval_level,
        )
        return RSSMRollout(samples, deterministic, interval_level=interval_level, metadata=values)

    def evaluate_one_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        *,
        sequence_mask: np.ndarray | None = None,
        interval_level: float = 0.95,
    ) -> RSSMOneStepMetrics:
        """Evaluate masked teacher-forced Gaussian predictions against next states."""
        start = time.perf_counter()
        state_values, action_values, mask = self._validate_one_step_sequences(
            states, actions, next_states, sequence_mask
        )
        predictions = self._teacher_forced_distribution_predictions(state_values, action_values, mask)
        evaluated = aggregate_one_step_metrics(
            predictions,
            next_states,
            mask,
            interval_level=interval_level,
            posterior_scale_factor=self.config.posterior_scale_factor,
        )
        self.reset()
        return RSSMOneStepMetrics(
            evaluated.mse,
            evaluated.rmse,
            evaluated.negative_log_likelihood,
            evaluated.kl_divergence,
            evaluated.coverage,
            evaluated.mean_error,
            evaluated.n_samples,
            time.perf_counter() - start,
        )

    def evaluate_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        target_states: np.ndarray,
        *,
        sequence_mask: np.ndarray | None = None,
        n_samples: int = 128,
        seed: int = 0,
        interval_level: float = 0.95,
    ) -> RSSMRolloutMetrics:
        """Evaluate masked open-loop particle rollouts and horizon-wise diagnostics."""
        start = time.perf_counter()
        initial, action_values, targets, mask = self._validate_rollout_inputs(
            initial_state, actions, target_states, sequence_mask
        )
        batch, horizon, _ = action_values.shape
        if horizon == 0:
            return RSSMRolloutMetrics((), (), (), 0.0, 0.0, 0.0, 1.0, time.perf_counter() - start, True)
        means: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        for episode in range(batch):
            length = int(np.sum(mask[episode]))
            if length == 0:
                means.append(np.empty((0, self.state_dim), dtype=np.float64))
                scales.append(np.empty((0, self.state_dim), dtype=np.float64))
                continue
            rollout = self.rollout(
                initial[episode],
                action_values[episode, :length],
                n_samples=n_samples,
                seed=seed + episode,
                interval_level=interval_level,
            )
            means.append(rollout.mean)
            scales.append(rollout.scale)
        evaluated = aggregate_rollout_metrics(
            targets,
            mask,
            means,
            scales,
            variance_floor=self.config.variance_floor,
            stability_norm_limit=self.config.stability_norm_limit,
        )
        return RSSMRolloutMetrics(
            evaluated.errors_by_horizon,
            evaluated.kl_by_horizon,
            evaluated.coverage_by_horizon,
            evaluated.mean_error,
            evaluated.final_error,
            evaluated.mean_kl,
            evaluated.mean_coverage,
            time.perf_counter() - start,
            evaluated.stable,
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a portable checkpoint; in-flight recurrent state is not persisted."""

        self._require_fitted()
        assert self._recurrent_weights is not None and self._recurrent_bias is not None
        assert self._emission_weights is not None and self._emission_bias is not None and self._scale is not None
        write_rssm_checkpoint(
            path,
            recurrent_weights=self._recurrent_weights,
            recurrent_bias=self._recurrent_bias,
            emission_weights=self._emission_weights,
            emission_bias=self._emission_bias,
            scale=self._scale,
            config=self.to_config().model_dump(mode="json"),
            source_space_identity=self.source_space_identity,
            fit_metadata=dict(self._fit_metadata),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, device: str | None = None) -> RSSMLatentTransition:
        """Load a fitted RSSM checkpoint and reset its transient hidden state."""
        checkpoint = read_rssm_checkpoint(path)
        config_values = checkpoint.metadata["config"]
        if device is not None:
            config_values["device"] = device
        config = RSSMTransitionConfig(**config_values)
        source_identity = checkpoint.metadata["source_space_identity"]
        state_dim = checkpoint.emission_weights.shape[1]
        action_dim = checkpoint.recurrent_weights.shape[0] - config.hidden_dim - state_dim
        model = cls(
            LatentSpace(state_dim, source_model=source_identity),
            action_dim,
            source_space_identity=source_identity,
            config=config,
        )
        model._recurrent_weights = checkpoint.recurrent_weights
        model._recurrent_bias = checkpoint.recurrent_bias
        model._emission_weights = checkpoint.emission_weights
        model._emission_bias = checkpoint.emission_bias
        model._scale = checkpoint.scale
        fit_metadata = checkpoint.metadata.get("fit_metadata", {})
        model._fit_metadata = MappingProxyType(dict(fit_metadata))
        model.reset()
        return model

    @property
    def hidden_dim_input(self) -> int:
        """Return the recurrent input width: hidden, state, and action dimensions."""
        return self.config.hidden_dim + self.state_dim + self.action_dim

    def _resolve_device(self, device: str) -> str:
        try:
            resolved = str(torch.device(device))
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"invalid RSSM device {device!r}") from exc
        if resolved.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"RSSM device {device!r} requested but CUDA is unavailable")
        return resolved

    def _require_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self._recurrent_weights,
                self._recurrent_bias,
                self._emission_weights,
                self._emission_bias,
                self._scale,
            )
        ):
            raise RuntimeError("transition must be fitted before prediction")

    @staticmethod
    def _validate_point(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
        return validate_point(value, name=name, width=width)

    @staticmethod
    def _validate_batch(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
        return validate_batch(value, name=name, width=width)

    def _validate_sequences(
        self, states: np.ndarray, actions: np.ndarray, sequence_mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return validate_sequences(states, actions, sequence_mask, state_dim=self.state_dim, action_dim=self.action_dim)

    def _validate_one_step_sequences(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, sequence_mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return validate_one_step_sequences(
            states,
            actions,
            next_states,
            sequence_mask,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    def _validate_rollout_inputs(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        target_states: np.ndarray,
        sequence_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return validate_rollout_inputs(
            initial_state,
            actions,
            target_states,
            sequence_mask,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    def _concat_hidden_input(self, hidden: np.ndarray, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate((hidden, state, action))

    def _concat_emission_input(self, hidden: np.ndarray, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate((hidden, state, action, np.ones(1)))

    def _teacher_forced_predictions(
        self, states: np.ndarray, actions: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._recurrent_weights is not None and self._recurrent_bias is not None
        assert self._emission_weights is not None and self._emission_bias is not None
        return teacher_forced_predictions(
            states,
            actions,
            mask,
            recurrent_weights=self._recurrent_weights,
            recurrent_bias=self._recurrent_bias,
            emission_weights=self._emission_weights,
            emission_bias=self._emission_bias,
            hidden_dim=self.config.hidden_dim,
            state_dim=self.state_dim,
        )

    def _teacher_forced_distribution_predictions(
        self, states: np.ndarray, actions: np.ndarray, mask: np.ndarray
    ) -> list[list[RSSMPrediction]]:
        self._require_fitted()
        assert self._recurrent_weights is not None and self._recurrent_bias is not None
        assert self._emission_weights is not None and self._emission_bias is not None
        means, hidden_paths = teacher_forced_distribution_arrays(
            states,
            actions,
            mask,
            recurrent_weights=self._recurrent_weights,
            recurrent_bias=self._recurrent_bias,
            emission_weights=self._emission_weights,
            emission_bias=self._emission_bias,
            hidden_dim=self.config.hidden_dim,
        )
        result: list[list[RSSMPrediction]] = [[] for _ in range(states.shape[0])]
        for episode in range(states.shape[0]):
            result[episode] = [
                RSSMPrediction(means[episode, index], self.scale, hidden_paths[episode, index])
                for index in range(actions.shape[1])
            ]
        return result
