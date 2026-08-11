"""A compact NumPy-facing RSSM-style latent transition.

The implementation uses a learned tanh deterministic state and a diagonal
Gaussian stochastic next-latent head.  Torch is used only for the bounded
internal fit; callers exchange NumPy arrays, just like the earlier transition
instances.  This is intentionally a small RSSM-style model rather than a
claim of full Dreamer/RSSM posterior inference: the KL diagnostic uses an
observation-centred posterior proxy and is reported explicitly as such.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


def _finite_array(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(value).__name__}")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(value, dtype=np.float64)


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
        return np.square(self.scale)

    @property
    def covariance(self) -> np.ndarray:
        return np.diag(self.variance)

    @property
    def event_shape(self) -> tuple[int]:
        return self.mean.shape

    @property
    def distribution_family(self) -> str:
        return "diagonal_gaussian"

    def sample(self, *, seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        generator = rng if rng is not None else np.random.default_rng(seed)
        return self.mean + self.scale * generator.normal(size=self.mean.shape)

    def log_prob(self, value: np.ndarray) -> float:
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
        return np.mean(self.samples, axis=0)

    @property
    def scale(self) -> np.ndarray:
        return np.std(self.samples, axis=0)

    @property
    def lower(self) -> np.ndarray:
        return self.mean - 1.959963984540054 * self.scale

    @property
    def upper(self) -> np.ndarray:
        return self.mean + 1.959963984540054 * self.scale

    def to_numpy(self) -> np.ndarray:
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
        return self.latent_space.dim

    @property
    def state_shape(self) -> tuple[int]:
        return (self.state_dim,)

    @property
    def action_shape(self) -> tuple[int]:
        return (self.action_dim,)

    @property
    def hidden_shape(self) -> tuple[int]:
        return (self.config.hidden_dim,)

    @property
    def is_fitted(self) -> bool:
        return self._scale is not None

    @property
    def fit_metadata(self) -> Mapping[str, Any]:
        return self._fit_metadata

    @property
    def scale(self) -> np.ndarray:
        self._require_fitted()
        return self._scale.copy()  # type: ignore[union-attr]

    @property
    def hidden_state(self) -> np.ndarray:
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
        torch.manual_seed(fit_seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(fit_seed)
        torch_device = torch.device(self.device)
        recurrent = torch.nn.Linear(self.hidden_dim_input, self.config.hidden_dim, device=torch_device)
        emission = torch.nn.Linear(
            self.config.hidden_dim + self.state_dim + self.action_dim + 1, self.state_dim, device=torch_device
        )
        optimizer = torch.optim.Adam([*recurrent.parameters(), *emission.parameters()], lr=self.config.learning_rate)
        states_tensor = torch.as_tensor(state_values, dtype=torch.float32, device=torch_device)
        actions_tensor = torch.as_tensor(action_values, dtype=torch.float32, device=torch_device)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=torch_device)
        final_loss = float("nan")
        for _ in range(self.config.epochs):
            optimizer.zero_grad()
            hidden = torch.zeros((state_values.shape[0], self.config.hidden_dim), device=torch_device)
            total_loss = torch.zeros((), device=torch_device)
            valid_count = torch.zeros((), device=torch_device)
            for index in range(action_values.shape[1]):
                valid = mask_tensor[:, index]
                proposed = torch.tanh(
                    recurrent(torch.cat((hidden, states_tensor[:, index], actions_tensor[:, index]), dim=1))
                )
                hidden = torch.where(valid[:, None], proposed, hidden)
                prediction = emission(
                    torch.cat(
                        (
                            hidden,
                            states_tensor[:, index],
                            actions_tensor[:, index],
                            torch.ones((state_values.shape[0], 1), device=torch_device),
                        ),
                        dim=1,
                    )
                )
                residual = torch.square(prediction - states_tensor[:, index + 1])
                total_loss = total_loss + torch.sum(residual * valid[:, None])
                valid_count = valid_count + torch.sum(valid)
            loss = total_loss / torch.clamp(valid_count * self.state_dim, min=1.0)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

        self._recurrent_weights = recurrent.weight.detach().cpu().numpy().T.astype(np.float64)
        self._recurrent_bias = recurrent.bias.detach().cpu().numpy().astype(np.float64)
        self._emission_weights = emission.weight.detach().cpu().numpy().T.astype(np.float64)
        self._emission_bias = emission.bias.detach().cpu().numpy().astype(np.float64)
        _, predictions = self._teacher_forced_predictions(state_values, action_values, mask)
        residuals = states[:, 1:, :] - predictions
        valid_residuals = residuals[mask]
        variances = np.maximum(np.mean(np.square(valid_residuals), axis=0), self.config.variance_floor)
        self._scale = np.sqrt(variances)
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
                "final_training_mse": final_loss,
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
        hidden = np.tanh(
            self._concat_hidden_input(self._hidden_state, state_value, action_value) @ self._recurrent_weights
            + self._recurrent_bias
        )  # type: ignore[operator]
        mean = (
            self._concat_emission_input(hidden, state_value, action_value) @ self._emission_weights
            + self._emission_bias
        )  # type: ignore[operator]
        self._hidden_state = hidden
        return RSSMPrediction(mean=mean, scale=self.scale, deterministic_state=hidden)

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.predict(state, action).mean.copy()

    def mean_rollout(
        self, initial_state: np.ndarray, actions: np.ndarray, *, metadata: Mapping[str, Any] | None = None
    ) -> Trajectory:
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
        values: dict[str, Any] = {
            "state_source": "predictive_mean",
            "source_space_identity": self.source_space_identity,
            "transition": self.__class__.__name__,
            "rollout_horizon": int(action_values.shape[0]),
            "action_shape": self.action_shape,
            "state_shape": self.state_shape,
            "deterministic_state_shape": self.hidden_shape,
            "stateful": True,
        }
        if metadata is not None:
            values.update(dict(metadata))
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
        generator = np.random.default_rng(seed)
        samples = np.empty((n_samples, action_values.shape[0] + 1, self.state_dim), dtype=np.float64)
        hidden = np.zeros((n_samples, self.config.hidden_dim), dtype=np.float64)
        deterministic = np.empty((n_samples, action_values.shape[0] + 1, self.config.hidden_dim), dtype=np.float64)
        samples[:, 0] = initial
        deterministic[:, 0] = hidden
        for index, action in enumerate(action_values):
            repeated_action = np.broadcast_to(action, (n_samples, self.action_dim))
            recurrent_input = np.concatenate((hidden, samples[:, index], repeated_action), axis=1)
            hidden = np.tanh(recurrent_input @ self._recurrent_weights + self._recurrent_bias)  # type: ignore[operator]
            emission_input = np.concatenate(
                (hidden, samples[:, index], repeated_action, np.ones((n_samples, 1))), axis=1
            )
            means = emission_input @ self._emission_weights + self._emission_bias  # type: ignore[operator]
            samples[:, index + 1] = means + self.scale * generator.normal(size=(n_samples, self.state_dim))
            deterministic[:, index + 1] = hidden
        self._hidden_state = np.mean(hidden, axis=0)
        values: dict[str, Any] = {
            "state_source": "sampled",
            "source_space_identity": self.source_space_identity,
            "transition": self.__class__.__name__,
            "rollout_horizon": int(action_values.shape[0]),
            "action_shape": self.action_shape,
            "state_shape": self.state_shape,
            "deterministic_state_shape": self.hidden_shape,
            "stateful": True,
            "n_samples": int(n_samples),
            "seed": seed,
            "interval_level": interval_level,
        }
        if metadata is not None:
            values.update(dict(metadata))
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
        start = time.perf_counter()
        state_values, action_values, mask = self._validate_one_step_sequences(
            states, actions, next_states, sequence_mask
        )
        predictions = self._teacher_forced_distribution_predictions(state_values, action_values, mask)
        targets = next_states[:, 1:, :][mask]
        selected = [
            prediction for row, row_mask in zip(predictions, mask) for prediction, valid in zip(row, row_mask) if valid
        ]
        means = np.asarray([prediction.mean for prediction in selected])
        errors = means - targets
        lower = np.asarray([prediction.interval(interval_level)[0] for prediction in selected])
        upper = np.asarray([prediction.interval(interval_level)[1] for prediction in selected])
        nll = float(-np.mean([prediction.log_prob(target) for prediction, target in zip(selected, targets)]))
        kl = float(
            np.mean(
                [
                    prediction.kl_to_observation(target, posterior_scale_factor=self.config.posterior_scale_factor)
                    for prediction, target in zip(selected, targets)
                ]
            )
        )
        coverage = float(np.mean((targets >= lower) & (targets <= upper)))
        mse = float(np.mean(np.square(errors)))
        self.reset()
        return RSSMOneStepMetrics(
            mse,
            float(np.sqrt(mse)),
            nll,
            kl,
            coverage,
            float(np.mean(np.linalg.norm(errors, axis=1))),
            len(selected),
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
        start = time.perf_counter()
        initial, action_values, targets, mask = self._validate_rollout_inputs(
            initial_state, actions, target_states, sequence_mask
        )
        batch, horizon, _ = action_values.shape
        if horizon == 0:
            return RSSMRolloutMetrics((), (), (), 0.0, 0.0, 0.0, 1.0, time.perf_counter() - start, True)
        errors: list[list[float]] = []
        kls: list[list[float]] = []
        coverages: list[list[float]] = []
        predicted_norms: list[float] = []
        for episode in range(batch):
            length = int(np.sum(mask[episode]))
            if length == 0:
                continue
            rollout = self.rollout(
                initial[episode],
                action_values[episode, :length],
                n_samples=n_samples,
                seed=seed + episode,
                interval_level=interval_level,
            )
            mean = rollout.mean[1:]
            scale = np.maximum(rollout.scale[1:], np.sqrt(self.config.variance_floor))
            target = targets[episode, 1 : length + 1]
            differences = target - mean
            errors.append([float(np.linalg.norm(value)) for value in differences])
            predicted_norms.append(float(np.max(np.linalg.norm(mean, axis=1))))
            lower = mean - 1.959963984540054 * scale
            upper = mean + 1.959963984540054 * scale
            coverages.append(
                [
                    float(np.mean((target[index] >= lower[index]) & (target[index] <= upper[index])))
                    for index in range(length)
                ]
            )
            kls.append([float(0.5 * np.sum(np.square(differences[index] / scale[index]))) for index in range(length)])
        errors_by_horizon = tuple(
            float(np.mean([row[index] for row in errors if index < len(row)]))
            for index in range(horizon)
            if any(index < len(row) for row in errors)
        )
        kl_by_horizon = tuple(
            float(np.mean([row[index] for row in kls if index < len(row)])) for index in range(len(errors_by_horizon))
        )
        coverage_by_horizon = tuple(
            float(np.mean([row[index] for row in coverages if index < len(row)]))
            for index in range(len(errors_by_horizon))
        )
        max_state_norm = max(predicted_norms, default=0.0)
        return RSSMRolloutMetrics(
            errors_by_horizon,
            kl_by_horizon,
            coverage_by_horizon,
            float(np.mean(errors_by_horizon)),
            errors_by_horizon[-1] if errors_by_horizon else 0.0,
            float(np.mean(kl_by_horizon)) if kl_by_horizon else 0.0,
            float(np.mean(coverage_by_horizon)) if coverage_by_horizon else 1.0,
            time.perf_counter() - start,
            bool(np.isfinite(max_state_norm) and max_state_norm <= self.config.stability_norm_limit),
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a portable checkpoint; in-flight recurrent state is not persisted."""

        self._require_fitted()
        assert self._recurrent_weights is not None and self._recurrent_bias is not None
        assert self._emission_weights is not None and self._emission_bias is not None and self._scale is not None
        metadata = {
            "config": self.to_config().model_dump(mode="json"),
            "source_space_identity": self.source_space_identity,
            "fit_metadata": dict(self._fit_metadata),
        }
        np.savez(
            path,
            recurrent_weights=self._recurrent_weights,
            recurrent_bias=self._recurrent_bias,
            emission_weights=self._emission_weights,
            emission_bias=self._emission_bias,
            scale=self._scale,
            metadata_json=json.dumps(metadata),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, device: str | None = None) -> RSSMLatentTransition:
        with np.load(path, allow_pickle=False) as data:  # pyright: ignore[reportUnknownMemberType]
            metadata_raw = data["metadata_json"].item()
            if not isinstance(metadata_raw, str):
                raise ValueError("RSSM checkpoint has no metadata_json string")
            metadata = cast(dict[str, Any], json.loads(metadata_raw))
            config_values = cast(dict[str, Any], metadata["config"])
            if device is not None:
                config_values["device"] = device
            config = RSSMTransitionConfig(**config_values)
            source_identity = metadata["source_space_identity"]
            recurrent_weights = np.asarray(data["recurrent_weights"], dtype=np.float64)
            emission_weights = np.asarray(data["emission_weights"], dtype=np.float64)
            state_dim = emission_weights.shape[1]
            action_dim = recurrent_weights.shape[0] - config.hidden_dim - state_dim
            model = cls(
                LatentSpace(state_dim, source_model=source_identity),
                action_dim,
                source_space_identity=source_identity,
                config=config,
            )
            model._recurrent_weights = recurrent_weights
            model._recurrent_bias = np.asarray(data["recurrent_bias"], dtype=np.float64)
            model._emission_weights = emission_weights
            model._emission_bias = np.asarray(data["emission_bias"], dtype=np.float64)
            model._scale = np.asarray(data["scale"], dtype=np.float64)
            fit_metadata = metadata.get("fit_metadata", {})
            model._fit_metadata = MappingProxyType(dict(fit_metadata))
            model.reset()
            return model

    @property
    def hidden_dim_input(self) -> int:
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
        values = _finite_array(value, name=name)
        if values.ndim != 1 or values.shape != (width,):
            raise ValueError(f"{name} must have shape ({width},), got {values.shape}")
        return values

    @staticmethod
    def _validate_batch(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
        values = _finite_array(value, name=name)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError(f"{name} must have shape (n, {width}), got {values.shape}")
        return values

    def _validate_sequences(
        self, states: np.ndarray, actions: np.ndarray, sequence_mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_values = _finite_array(states, name="states")
        action_values = _finite_array(actions, name="actions")
        if state_values.ndim != 3 or state_values.shape[2] != self.state_dim:
            raise ValueError(
                f"states must have shape (episodes, horizon + 1, {self.state_dim}), got {state_values.shape}"
            )
        if (
            action_values.ndim != 3
            or action_values.shape[2] != self.action_dim
            or action_values.shape[:2] != (state_values.shape[0], state_values.shape[1] - 1)
        ):
            raise ValueError(
                f"actions must have shape (episodes, horizon, {self.action_dim}), got {action_values.shape}"
            )
        if sequence_mask is None:
            mask = np.ones(action_values.shape[:2], dtype=bool)
        else:
            raw_mask = _finite_array(sequence_mask, name="sequence_mask")
            if raw_mask.shape != action_values.shape[:2] or not np.isin(raw_mask, [0.0, 1.0]).all():
                raise ValueError(f"sequence_mask must have shape {action_values.shape[:2]} and contain only 0/1 values")
            mask = raw_mask.astype(bool)
        return state_values, action_values, mask

    def _validate_one_step_sequences(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, sequence_mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_values, action_values, mask = self._validate_sequences(states, actions, sequence_mask)
        target_values = _finite_array(next_states, name="next_states")
        if target_values.shape != state_values.shape:
            raise ValueError(f"next_states must have shape {state_values.shape}, got {target_values.shape}")
        return state_values, action_values, mask

    def _validate_rollout_inputs(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        target_states: np.ndarray,
        sequence_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        initial = _finite_array(initial_state, name="initial_state")
        action_values = _finite_array(actions, name="actions")
        targets = _finite_array(target_states, name="target_states")
        if initial.ndim == 1:
            initial = initial[None, :]
        if action_values.ndim == 2:
            action_values = action_values[None, :, :]
        if targets.ndim == 2:
            targets = targets[None, :, :]
        if initial.ndim != 2 or initial.shape[1] != self.state_dim or action_values.ndim != 3 or targets.ndim != 3:
            raise ValueError("invalid rollout input dimensions")
        if (
            action_values.shape[0] != initial.shape[0]
            or targets.shape[:2] != (initial.shape[0], action_values.shape[1] + 1)
            or action_values.shape[2] != self.action_dim
            or targets.shape[2] != self.state_dim
        ):
            raise ValueError("rollout inputs have incompatible batch, horizon, or feature shapes")
        if not np.array_equal(initial, targets[:, 0]):
            raise ValueError("initial_state must equal target_states[:, 0, :]")
        if sequence_mask is None:
            mask = np.ones(action_values.shape[:2], dtype=bool)
        else:
            raw_mask = _finite_array(sequence_mask, name="sequence_mask")
            if raw_mask.shape != action_values.shape[:2] or not np.isin(raw_mask, [0.0, 1.0]).all():
                raise ValueError(f"sequence_mask must have shape {action_values.shape[:2]} and contain only 0/1 values")
            mask = raw_mask.astype(bool)
        return initial, action_values, targets, mask

    def _concat_hidden_input(self, hidden: np.ndarray, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate((hidden, state, action))

    def _concat_emission_input(self, hidden: np.ndarray, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate((hidden, state, action, np.ones(1)))

    def _teacher_forced_predictions(
        self, states: np.ndarray, actions: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._recurrent_weights is not None and self._recurrent_bias is not None
        assert self._emission_weights is not None and self._emission_bias is not None
        hidden = np.zeros((states.shape[0], self.config.hidden_dim), dtype=np.float64)
        predictions = np.empty((states.shape[0], actions.shape[1], self.state_dim), dtype=np.float64)
        hidden_paths = np.empty((states.shape[0], actions.shape[1], self.config.hidden_dim), dtype=np.float64)
        for index in range(actions.shape[1]):
            proposed = np.tanh(
                np.concatenate((hidden, states[:, index], actions[:, index]), axis=1) @ self._recurrent_weights
                + self._recurrent_bias
            )  # type: ignore[operator]
            hidden = np.where(mask[:, index, None], proposed, hidden)
            predictions[:, index] = (
                np.concatenate((hidden, states[:, index], actions[:, index], np.ones((states.shape[0], 1))), axis=1)
                @ self._emission_weights
                + self._emission_bias
            )  # type: ignore[operator]
            hidden_paths[:, index] = hidden
        return hidden_paths, predictions

    def _teacher_forced_distribution_predictions(
        self, states: np.ndarray, actions: np.ndarray, mask: np.ndarray
    ) -> list[list[RSSMPrediction]]:
        self._require_fitted()
        assert self._recurrent_weights is not None and self._recurrent_bias is not None
        assert self._emission_weights is not None and self._emission_bias is not None
        hidden = np.zeros((states.shape[0], self.config.hidden_dim), dtype=np.float64)
        result: list[list[RSSMPrediction]] = [[] for _ in range(states.shape[0])]
        for index in range(actions.shape[1]):
            proposed = np.tanh(
                np.concatenate((hidden, states[:, index], actions[:, index]), axis=1) @ self._recurrent_weights
                + self._recurrent_bias
            )  # type: ignore[operator]
            hidden = np.where(mask[:, index, None], proposed, hidden)
            means = (
                np.concatenate((hidden, states[:, index], actions[:, index], np.ones((states.shape[0], 1))), axis=1)
                @ self._emission_weights
                + self._emission_bias
            )  # type: ignore[operator]
            for episode in range(states.shape[0]):
                result[episode].append(RSSMPrediction(means[episode], self.scale, hidden[episode]))
        return result
