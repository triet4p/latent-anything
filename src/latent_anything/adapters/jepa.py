"""Compact decoder-free JEPA/LeWM-style world-model adapter.

This module is a reproducible CPU reference implementation rather than a
claim about a particular upstream checkpoint.  It follows the important
JEPA contract: a context encoder predicts a stop-gradient target-encoder
representation, and no pixel/data-space decoder is exposed.  The same
predictor is also a mean latent transition so it can be consumed by the
existing rollout pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator
from torch import nn

from latent_anything._jepa_checkpoint import read_jepa_checkpoint, write_jepa_checkpoint
from latent_anything._jepa_evaluation import (
    JEPALatentHealthValues,
    aggregate_prediction_metrics,
    aggregate_rollout_metrics,
    compute_latent_health,
)
from latent_anything._jepa_training import fit_jepa_parameters
from latent_anything._jepa_validation import (
    finite_array as _finite_array,
)
from latent_anything._jepa_validation import (
    validate_batch,
    validate_mask,
    validate_point,
    validate_rollout_inputs,
    validate_sequences,
)
from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


def _immutable(value: np.ndarray) -> np.ndarray:
    result = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    result.setflags(write=False)
    return result


class JEPAWorldModelConfig(BaseModel):
    """Reproducible fit and runtime settings for the compact JEPA model."""

    hidden_dim: int = Field(default=32, gt=0)
    epochs: int = Field(default=120, gt=0)
    learning_rate: float = Field(default=0.01, gt=0)
    ema_momentum: float = Field(default=0.95, gt=0, lt=1)
    variance_loss_weight: float = Field(default=0.01, ge=0)
    minimum_latent_std: float = Field(default=0.05, ge=0)
    variance_floor: float = Field(default=1e-6, ge=0)
    stability_norm_limit: float = Field(default=1e6, gt=0)
    seed: int = 71
    device: str = "cpu"

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("device must be a non-empty string")
        return value


@dataclass(frozen=True, slots=True)
class JEPAPrediction:
    """One predictive latent distribution with diagonal uncertainty."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = _finite_array(self.mean, name="mean")
        scale = _finite_array(self.scale, name="scale")
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("mean and scale must be one-dimensional arrays with matching shapes")
        if np.any(scale < 0):
            raise ValueError("scale must be non-negative")
        object.__setattr__(self, "mean", _immutable(mean))
        object.__setattr__(self, "scale", _immutable(scale))

    @property
    def variance(self) -> np.ndarray:
        """Return elementwise predictive variance, computed as ``scale ** 2``."""
        return np.square(self.scale)

    @property
    def covariance(self) -> np.ndarray:
        """Return the diagonal covariance matrix of the prediction."""
        return np.diag(self.variance)


@dataclass(frozen=True, slots=True)
class JEPALatentHealth:
    """Variance, covariance, and collapse diagnostics for a latent batch."""

    mean_variance: float
    min_variance: float
    max_variance: float
    covariance_condition: float
    effective_rank: float
    participation_ratio: float
    collapsed_fraction: float
    collapse_score: float
    n_samples: int
    latent_dim: int

    def to_dict(self) -> dict[str, object]:
        """Return latent-health values in a JSON-compatible mapping."""
        return {
            "mean_variance": self.mean_variance,
            "min_variance": self.min_variance,
            "max_variance": self.max_variance,
            "covariance_condition": self.covariance_condition,
            "effective_rank": self.effective_rank,
            "participation_ratio": self.participation_ratio,
            "collapsed_fraction": self.collapsed_fraction,
            "collapse_score": self.collapse_score,
            "n_samples": self.n_samples,
            "latent_dim": self.latent_dim,
        }


@dataclass(frozen=True, slots=True)
class JEPAPredictionMetrics:
    """Teacher-forced latent prediction and collapsed-baseline comparison."""

    mse: float
    rmse: float
    mean_error: float
    collapsed_baseline_mse: float
    improvement_over_collapsed: float
    target_health: JEPALatentHealth
    n_samples: int
    runtime_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Return prediction metrics and nested representation-health evidence."""
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mean_error": self.mean_error,
            "collapsed_baseline_mse": self.collapsed_baseline_mse,
            "improvement_over_collapsed": self.improvement_over_collapsed,
            "target_health": self.target_health.to_dict(),
            "n_samples": self.n_samples,
            "runtime_seconds": self.runtime_seconds,
        }

    def to_metrics(self) -> dict[str, float]:
        """Return the flat metric names used by evaluation consumers."""
        return {
            "latent_prediction_mse": self.mse,
            "latent_prediction_rmse": self.rmse,
            "collapsed_baseline_mse": self.collapsed_baseline_mse,
            "improvement_over_collapsed": self.improvement_over_collapsed,
            "latent_effective_rank": self.target_health.effective_rank,
            "latent_collapsed_fraction": self.target_health.collapsed_fraction,
        }


@dataclass(frozen=True, slots=True)
class JEPARolloutMetrics:
    """Open-loop latent error and horizon-drift diagnostics."""

    errors_by_horizon: tuple[float, ...]
    mean_error: float
    final_error: float
    horizon_drift: float
    error_growth_ratio: float
    n_episodes: int
    runtime_seconds: float
    stable: bool

    @property
    def horizon(self) -> int:
        """Return the number of open-loop horizon entries."""
        return len(self.errors_by_horizon)

    def to_dict(self) -> dict[str, object]:
        """Return rollout metrics as JSON-compatible scalar and list values."""
        return {
            "errors_by_horizon": list(self.errors_by_horizon),
            "mean_error": self.mean_error,
            "final_error": self.final_error,
            "horizon_drift": self.horizon_drift,
            "error_growth_ratio": self.error_growth_ratio,
            "n_episodes": self.n_episodes,
            "runtime_seconds": self.runtime_seconds,
            "stable": self.stable,
        }

    def to_metrics(self) -> dict[str, float]:
        """Return the flat rollout metric names used by evaluation consumers."""
        return {
            "rollout_mean_error": self.mean_error,
            "rollout_final_error": self.final_error,
            "rollout_horizon_drift": self.horizon_drift,
            "rollout_error_growth_ratio": self.error_growth_ratio,
            "rollout_stable": float(self.stable),
        }


@dataclass(frozen=True, slots=True)
class JEPAEvaluationReport:
    """Combined typed evidence payload for a JEPA world-model run."""

    prediction: JEPAPredictionMetrics
    rollout: JEPARolloutMetrics
    provenance: Mapping[str, object] = MappingProxyType({})

    def to_dict(self) -> dict[str, object]:
        """Return prediction, rollout, and provenance evidence as a mapping."""
        return {
            "prediction": self.prediction.to_dict(),
            "rollout": self.rollout.to_dict(),
            "provenance": dict(self.provenance),
        }

    def to_metrics(self) -> dict[str, float]:
        """Return the merged flat prediction and rollout metrics."""
        metrics = self.prediction.to_metrics()
        metrics.update(self.rollout.to_metrics())
        return metrics


class _MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Map a torch observation batch of width ``input_dim`` to latent values."""
        return self.layers(values)


class _Predictor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Map latent/action torch batches to predicted next-latent values."""
        return self.layers(torch.cat((latent, action), dim=1))


class JEPAWorldModelAdapter:
    """Compact action-conditioned decoder-free JEPA/LeWM-style adapter.

    The adapter is deliberately a ``ModelAdapter`` without ``decode``.  It
    learns ``predict(context_encoder(obs_t), action_t)`` to match the
    stop-gradient ``target_encoder(obs_{t+1})``.  After fitting, the
    predictor also implements the shared mean-transition surface used by
    :class:`~latent_anything.rollout_pipeline.RolloutPipeline`.

    ``dataset_revision`` and ``model_revision`` identify this reproducible
    compact reference lane.  They are not claims of fidelity to an upstream
    I-JEPA or LeWM checkpoint.
    """

    stream_state_contract = "explicit"
    model_revision = "compact-jepa-lewm-v1"
    dataset_revision = "synthetic-controlled-latent-dynamics-v1"

    def __init__(
        self,
        observation_dim: int,
        latent_dim: int,
        action_dim: int = 1,
        *,
        source_space_identity: str | None = None,
        config: JEPAWorldModelConfig | None = None,
        hidden_dim: int = 32,
        epochs: int = 120,
        learning_rate: float = 0.01,
        ema_momentum: float = 0.95,
        variance_loss_weight: float = 0.01,
        minimum_latent_std: float = 0.05,
        variance_floor: float = 1e-6,
        stability_norm_limit: float = 1e6,
        seed: int = 71,
        device: str = "cpu",
    ) -> None:
        if observation_dim < 1 or latent_dim < 1:
            raise ValueError("observation_dim and latent_dim must be >= 1")
        if action_dim < 0:
            raise ValueError("action_dim must be >= 0")
        supplied = config or JEPAWorldModelConfig(
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            ema_momentum=ema_momentum,
            variance_loss_weight=variance_loss_weight,
            minimum_latent_std=minimum_latent_std,
            variance_floor=variance_floor,
            stability_norm_limit=stability_norm_limit,
            seed=seed,
            device=device,
        )
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.config = supplied
        self.device = self._resolve_device(supplied.device)
        self.source_space_identity = source_space_identity or f"{self.model_revision}:latent-{latent_dim}"
        if not self.source_space_identity.strip():
            raise ValueError("source_space_identity must be a non-empty string")

        torch.manual_seed(supplied.seed)
        torch_device = torch.device(self.device)
        self._context_encoder = _MLPEncoder(observation_dim, supplied.hidden_dim, latent_dim).to(torch_device)
        self._target_encoder = _MLPEncoder(observation_dim, supplied.hidden_dim, latent_dim).to(torch_device)
        self._predictor = _Predictor(latent_dim, action_dim, supplied.hidden_dim).to(torch_device)
        self._target_encoder.load_state_dict(self._context_encoder.state_dict())
        for parameter in self._target_encoder.parameters():
            parameter.requires_grad_(False)
        self._scale: np.ndarray | None = None
        self._fit_metadata: Mapping[str, Any] = MappingProxyType({})
        self._training_steps = 0

    @property
    def latent_space(self) -> LatentSpace:
        """Return the decoder-free Euclidean latent metadata."""

        return LatentSpace(
            dim=self.latent_dim,
            source_model=self.source_space_identity,
            metadata={
                "model_family": "jepa_lewm_style",
                "model_revision": self.model_revision,
                "dataset_revision": self.dataset_revision,
                "exposure_mode": "no_explicit_latent",
                "decoder": "absent",
                "prediction_target": "stop_gradient_target_encoder",
                "action_conditioned": self.action_dim > 0,
                "interpolation": "euclidean",
            },
        )

    @property
    def state_dim(self) -> int:
        """Return the exposed latent-state width."""
        return self.latent_dim

    @property
    def state_shape(self) -> tuple[int]:
        """Return the one-dimensional latent-state shape."""
        return (self.latent_dim,)

    @property
    def action_shape(self) -> tuple[int]:
        """Return the one-dimensional action shape."""
        return (self.action_dim,)

    @property
    def is_fitted(self) -> bool:
        """Return whether fitted predictive scale parameters are available."""
        return self._scale is not None

    @property
    def fit_metadata(self) -> Mapping[str, Any]:
        """Return immutable fit provenance, diagnostics, and revision metadata."""
        return self._fit_metadata

    @property
    def scale(self) -> np.ndarray:
        """Return a copy of fitted per-latent predictive standard deviations."""
        self._require_fitted()
        assert self._scale is not None
        return self._scale.copy()

    @property
    def target_encoder_requires_grad(self) -> bool:
        """Whether any target-encoder parameter accidentally receives grad."""

        return any(parameter.requires_grad for parameter in self._target_encoder.parameters())

    @property
    def target_encoder_has_gradients(self) -> bool:
        """Return whether target parameters currently carry a gradient."""

        return any(parameter.grad is not None for parameter in self._target_encoder.parameters())

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode input data with the trainable context encoder."""

        values = self._validate_observations(data, name="data")
        return self._encode_tensor(values, self._context_encoder)

    def encode_target(self, observations: np.ndarray) -> np.ndarray:
        """Encode full observations with the stop-gradient target encoder."""

        values = self._validate_observations(observations, name="observations")
        return self._encode_tensor(values, self._target_encoder)

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        sequence_mask: np.ndarray | None = None,
        seed: int | None = None,
    ) -> JEPAWorldModelAdapter:
        """Fit latent prediction on masked observation/action sequences."""

        observation_values, action_values, mask = self._validate_sequences(observations, actions, sequence_mask)
        if not np.any(mask):
            raise ValueError("sequence_mask must contain at least one valid transition")
        fit_seed = self.config.seed if seed is None else seed
        fitted = fit_jepa_parameters(
            self._context_encoder,
            self._target_encoder,
            self._predictor,
            observation_values,
            action_values,
            mask,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            ema_momentum=self.config.ema_momentum,
            variance_loss_weight=self.config.variance_loss_weight,
            minimum_latent_std=self.config.minimum_latent_std,
            variance_floor=self.config.variance_floor,
            device=self.device,
            seed=fit_seed,
            initial_training_steps=self._training_steps,
        )
        self._training_steps = fitted.training_steps
        residual = fitted.final_target - fitted.final_prediction
        self._scale = np.sqrt(np.maximum(np.mean(np.square(residual), axis=0), self.config.variance_floor))
        health = self.latent_health(fitted.final_target)
        self._fit_metadata = MappingProxyType(
            {
                "model_family": "jepa_lewm_style",
                "model_revision": self.model_revision,
                "dataset_revision": self.dataset_revision,
                "source_space_identity": self.source_space_identity,
                "observation_shape": (self.observation_dim,),
                "state_shape": self.state_shape,
                "action_shape": self.action_shape,
                "latent_shape": self.state_shape,
                "exposure_mode": "no_explicit_latent",
                "decoder": "absent",
                "target_encoder": "stop_gradient_ema",
                "target_encoder_updates": self._training_steps,
                "episodes": int(observation_values.shape[0]),
                "sequence_length": int(observation_values.shape[1] - 1),
                "valid_transitions": int(np.sum(mask)),
                "seed": int(fit_seed),
                "device": self.device,
                "epochs": self.config.epochs,
                "final_training_loss": fitted.final_loss,
                "target_effective_rank": health.effective_rank,
                "target_collapsed_fraction": health.collapsed_fraction,
                "variance_regularizer_source": "trainable_context_encoder",
            }
        )
        return self

    def predict(self, state: np.ndarray, action: np.ndarray) -> JEPAPrediction:
        """Predict the next latent distribution from a latent/action pair."""

        self._require_fitted()
        state_value = self._validate_point(state, name="state", width=self.latent_dim)
        action_value = self._validate_point(action, name="action", width=self.action_dim)
        torch_device = torch.device(self.device)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state_value[None, :], dtype=torch.float32, device=torch_device)
            action_tensor = torch.as_tensor(action_value[None, :], dtype=torch.float32, device=torch_device)
            predicted = self._predictor(state_tensor, action_tensor).cpu().numpy()[0].astype(np.float64)
        return JEPAPrediction(predicted, self.scale)

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Return the predictive mean for the shared transition contract."""

        return self.predict(state, action).mean.copy()

    def mean_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Trajectory:
        """Recursively roll out predictive means without decoding to data space."""

        self._require_fitted()
        initial = self._validate_point(initial_state, name="initial_state", width=self.latent_dim)
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        states = np.empty((action_values.shape[0] + 1, self.latent_dim), dtype=np.float64)
        states[0] = initial
        for index, action in enumerate(action_values):
            states[index + 1] = self.step(states[index], action)
        values: dict[str, object] = {
            "state_source": "jepa_predictive_mean",
            "source_space_identity": self.source_space_identity,
            "transition": self.__class__.__name__,
            "rollout_horizon": int(action_values.shape[0]),
            "action_shape": self.action_shape,
            "state_shape": self.state_shape,
            "exposure_mode": "no_explicit_latent",
            "decoder": "absent",
        }
        if metadata is not None:
            values.update(dict(metadata))
        return Trajectory(states, metadata=values)

    def evaluate_one_step(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
    ) -> JEPAPredictionMetrics:
        """Measure teacher-forced latent prediction and collapsed-baseline gain."""

        start = time.perf_counter()
        current = self._validate_batch(observations, name="observations", width=self.observation_dim)
        action_values = self._validate_batch(actions, name="actions", width=self.action_dim)
        next_values = self._validate_batch(next_observations, name="next_observations", width=self.observation_dim)
        if current.shape != next_values.shape or current.shape[0] != action_values.shape[0]:
            raise ValueError("observations, actions, and next_observations must have matching sample counts")
        predictions = np.vstack(
            [
                self.predict(state, action).mean
                for state, action in zip(self.encode(current), action_values, strict=True)
            ]
        )
        targets = self.encode_target(next_values)
        evaluated = aggregate_prediction_metrics(predictions, targets, variance_floor=self.config.variance_floor)
        return JEPAPredictionMetrics(
            mse=evaluated.mse,
            rmse=evaluated.rmse,
            mean_error=evaluated.mean_error,
            collapsed_baseline_mse=evaluated.collapsed_baseline_mse,
            improvement_over_collapsed=evaluated.improvement_over_collapsed,
            target_health=self._public_health(evaluated.target_health),
            n_samples=evaluated.n_samples,
            runtime_seconds=time.perf_counter() - start,
        )

    def evaluate_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        target_states: np.ndarray,
        *,
        sequence_mask: np.ndarray | None = None,
    ) -> JEPARolloutMetrics:
        """Measure masked open-loop latent error and horizon drift."""

        start = time.perf_counter()
        initial, action_values, targets, mask = self._validate_rollout_inputs(
            initial_state, actions, target_states, sequence_mask
        )
        predictions: list[np.ndarray] = []
        for episode in range(initial.shape[0]):
            length = int(np.sum(mask[episode]))
            if length == 0:
                predictions.append(np.empty((0, self.latent_dim), dtype=np.float64))
                continue
            predictions.append(self.mean_rollout(initial[episode], action_values[episode, :length]).to_numpy())
        evaluated = aggregate_rollout_metrics(
            targets,
            mask,
            predictions,
            variance_floor=self.config.variance_floor,
            stability_norm_limit=self.config.stability_norm_limit,
        )
        return JEPARolloutMetrics(
            errors_by_horizon=evaluated.errors_by_horizon,
            mean_error=evaluated.mean_error,
            final_error=evaluated.final_error,
            horizon_drift=evaluated.horizon_drift,
            error_growth_ratio=evaluated.error_growth_ratio,
            n_episodes=evaluated.n_episodes,
            runtime_seconds=time.perf_counter() - start,
            stable=evaluated.stable,
        )

    @staticmethod
    def latent_health(latents: np.ndarray, *, collapse_variance_threshold: float = 1e-5) -> JEPALatentHealth:
        """Compute covariance/effective-rank diagnostics without a decoder."""

        values = _finite_array(latents, name="latents")
        health_values = compute_latent_health(values, collapse_variance_threshold=collapse_variance_threshold)
        return JEPAWorldModelAdapter._public_health(health_values)

    @staticmethod
    def _public_health(values: JEPALatentHealthValues) -> JEPALatentHealth:
        return JEPALatentHealth(
            mean_variance=values.mean_variance,
            min_variance=values.min_variance,
            max_variance=values.max_variance,
            covariance_condition=values.covariance_condition,
            effective_rank=values.effective_rank,
            participation_ratio=values.participation_ratio,
            collapsed_fraction=values.collapsed_fraction,
            collapse_score=values.collapse_score,
            n_samples=values.n_samples,
            latent_dim=values.latent_dim,
        )

    def evaluate_latent_health(self, observations: np.ndarray, *, target: bool = False) -> JEPALatentHealth:
        """Encode observations and return representation-health diagnostics."""

        return self.latent_health(self.encode_target(observations) if target else self.encode(observations))

    def to_config(self) -> JEPAWorldModelConfig:
        """Return the effective serializable configuration with resolved device."""
        return self.config.model_copy(update={"device": self.device})

    def save(self, path: str) -> None:
        """Save a portable tensor checkpoint with explicit provenance."""

        self._require_fitted()
        payload = {
            "observation_dim": self.observation_dim,
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "source_space_identity": self.source_space_identity,
            "config": self.to_config().model_dump(mode="json"),
            "fit_metadata": dict(self._fit_metadata),
        }
        context_state = {
            name: value.detach().cpu().numpy() for name, value in self._context_encoder.state_dict().items()
        }
        target_state = {name: value.detach().cpu().numpy() for name, value in self._target_encoder.state_dict().items()}
        predictor_state = {name: value.detach().cpu().numpy() for name, value in self._predictor.state_dict().items()}
        write_jepa_checkpoint(
            path,
            metadata=payload,
            scale=self.scale,
            context_state=context_state,
            target_state=target_state,
            predictor_state=predictor_state,
        )

    @classmethod
    def load(cls, path: str, *, device: str | None = None) -> JEPAWorldModelAdapter:
        """Load a checkpoint and restore the target encoder without gradients."""

        checkpoint = read_jepa_checkpoint(path)
        metadata = checkpoint.metadata
        config_values = dict(metadata["config"])
        if device is not None:
            config_values["device"] = device
        model = cls(
            int(metadata["observation_dim"]),
            int(metadata["latent_dim"]),
            int(metadata["action_dim"]),
            source_space_identity=str(metadata["source_space_identity"]),
            config=JEPAWorldModelConfig(**config_values),
        )
        for module, state in (
            (model._context_encoder, checkpoint.context_state),
            (model._target_encoder, checkpoint.target_state),
            (model._predictor, checkpoint.predictor_state),
        ):
            module_state = {name: torch.as_tensor(state[name], device=model.device) for name in module.state_dict()}
            module.load_state_dict(module_state)
        model._scale = checkpoint.scale
        model._fit_metadata = MappingProxyType(dict(metadata.get("fit_metadata", {})))
        return model

    def _update_target_encoder(self) -> None:
        momentum = self.config.ema_momentum
        with torch.no_grad():
            for target, context in zip(
                self._target_encoder.parameters(), self._context_encoder.parameters(), strict=True
            ):
                target.mul_(momentum).add_(context, alpha=1.0 - momentum)

    def _encode_tensor(self, values: np.ndarray, encoder: _MLPEncoder) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            result = encoder(tensor).cpu().numpy().astype(np.float64)
        return result

    def _resolve_device(self, device: str) -> str:
        try:
            resolved = str(torch.device(device))
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"invalid JEPA device {device!r}") from exc
        if resolved.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"JEPA device {device!r} requested but CUDA is unavailable")
        return resolved

    def _require_fitted(self) -> None:
        if self._scale is None:
            raise RuntimeError("JEPA world model must be fitted before prediction")

    def _validate_observations(self, values: np.ndarray, *, name: str) -> np.ndarray:
        return self._validate_batch(values, name=name, width=self.observation_dim)

    @staticmethod
    def _validate_point(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
        return validate_point(value, name=name, width=width)

    @staticmethod
    def _validate_batch(value: np.ndarray, *, name: str, width: int) -> np.ndarray:
        return validate_batch(value, name=name, width=width)

    def _validate_sequences(
        self, observations: np.ndarray, actions: np.ndarray, sequence_mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return validate_sequences(
            observations,
            actions,
            sequence_mask,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
        )

    @staticmethod
    def _validate_mask(sequence_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
        return validate_mask(sequence_mask, shape)

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
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
        )


__all__ = [
    "JEPAWorldModelAdapter",
    "JEPAWorldModelConfig",
    "JEPAEvaluationReport",
    "JEPALatentHealth",
    "JEPAPrediction",
    "JEPAPredictionMetrics",
    "JEPARolloutMetrics",
]
