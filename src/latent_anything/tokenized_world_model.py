"""Compact autoregressive dynamics over VQ code sequences.

The implementation is deliberately small and diagnostic.  A fitted
``VQVAE`` supplies the frozen observation tokenizer and decoder; the dynamics
model predicts the next frame's code IDs one token at a time with an
action-conditioned GRU encoder/decoder.  Integer IDs remain categorical at
the public boundary and the class also implements the mean-transition seam
used by :class:`latent_anything.rollout_pipeline.RolloutPipeline`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from latent_anything._tokenized_dynamics import AutoregressiveDynamics as _AutoregressiveDynamics
from latent_anything._tokenized_evaluation import free_running_metrics, teacher_forced_metrics
from latent_anything._tokenized_integrity import (
    validate_codebook_version,
    validate_sequence_codebook_version,
    validate_tokenizer_binding,
)
from latent_anything._tokenized_training import fit_token_dynamics, sample_next_tokens
from latent_anything._tokenized_validation import (
    finite_array as _finite_array,
)
from latent_anything._tokenized_validation import (
    validate_actions,
    validate_sequence_tokens,
    validate_tokens,
    validate_training_actions,
)
from latent_anything.adapters.vq_vae import VQVAE
from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory

SamplingMode = Literal["greedy", "sample"]


@dataclass(frozen=True, slots=True)
class TokenPrediction:
    """One categorical next-frame prediction for a batch of contexts."""

    tokens: np.ndarray
    token_log_likelihood: np.ndarray
    sampling: SamplingMode

    def __post_init__(self) -> None:
        tokens = np.asarray(self.tokens)
        likelihood = _finite_array(self.token_log_likelihood, name="token_log_likelihood")
        if tokens.ndim != 2 or not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError("tokens must be a two-dimensional integer array")
        if likelihood.shape != tokens.shape:
            raise ValueError("token_log_likelihood must match tokens")
        frozen_tokens = tokens.astype(np.int64, copy=True)
        frozen_likelihood = likelihood.astype(np.float64, copy=True)
        frozen_tokens.setflags(write=False)
        frozen_likelihood.setflags(write=False)
        object.__setattr__(self, "tokens", frozen_tokens)
        object.__setattr__(self, "token_log_likelihood", frozen_likelihood)

    @property
    def mean_log_likelihood(self) -> float:
        """Return the mean token log likelihood over the batch and frame."""

        return float(np.mean(self.token_log_likelihood))


@dataclass(frozen=True, slots=True)
class TokenPredictionMetrics:
    """Teacher-forced categorical likelihood and accuracy diagnostics."""

    cross_entropy: float
    perplexity: float
    token_accuracy: float
    n_tokens: int

    def to_dict(self) -> dict[str, float | int]:
        """Return teacher-forced token metrics as JSON-compatible scalars."""
        return {
            "cross_entropy": self.cross_entropy,
            "perplexity": self.perplexity,
            "token_accuracy": self.token_accuracy,
            "n_tokens": self.n_tokens,
        }


@dataclass(frozen=True, slots=True)
class TokenRolloutMetrics:
    """Free-running token drift and optional decoded/task diagnostics."""

    token_error_by_horizon: tuple[float, ...]
    exact_frame_accuracy_by_horizon: tuple[float, ...]
    decoded_mse_by_horizon: tuple[float, ...] | None
    task_proxy_accuracy_by_horizon: tuple[float, ...] | None
    failure_horizon: int | None

    @property
    def horizon(self) -> int:
        """Return the number of free-running horizon entries."""
        return len(self.token_error_by_horizon)

    def to_dict(self) -> dict[str, object]:
        """Return token rollout metrics with optional diagnostics as lists."""
        return {
            "token_error_by_horizon": list(self.token_error_by_horizon),
            "exact_frame_accuracy_by_horizon": list(self.exact_frame_accuracy_by_horizon),
            "decoded_mse_by_horizon": None
            if self.decoded_mse_by_horizon is None
            else list(self.decoded_mse_by_horizon),
            "task_proxy_accuracy_by_horizon": None
            if self.task_proxy_accuracy_by_horizon is None
            else list(self.task_proxy_accuracy_by_horizon),
            "failure_horizon": self.failure_horizon,
        }


@dataclass(frozen=True, slots=True)
class TokenizedEvaluationReport:
    """Combined teacher-forced and free-running held-out evidence."""

    teacher_forced: TokenPredictionMetrics
    free_running: TokenRolloutMetrics
    codebook: Mapping[str, object]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "codebook", MappingProxyType(dict(self.codebook)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_dict(self) -> dict[str, object]:
        """Return teacher-forced, free-running, codebook, and provenance evidence."""
        return {
            "teacher_forced": self.teacher_forced.to_dict(),
            "free_running": self.free_running.to_dict(),
            "codebook": dict(self.codebook),
            "provenance": dict(self.provenance),
        }


class TokenizedWorldModelConfig(BaseModel):
    """Serializable training configuration for the compact token model."""

    model_config = ConfigDict(extra="forbid")

    action_dim: int = Field(ge=1)
    hidden_dim: int = Field(default=32, ge=2)
    epochs: int = Field(default=40, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0)
    seed: int = 0
    model_revision: str = "compact-tokenized-world-model-v1"
    codebook_version: str


class TokenizedWorldModel:
    """Action-conditioned next-frame prediction over frozen VQ token IDs.

    ``encode`` and ``decode`` expose the tokenizer boundary.  ``fit`` trains
    only the dynamics model after the supplied VQ-VAE has been fitted.  The
    model accepts either raw image sequences shaped ``(episodes, time, 1, 8,
    8)`` or already-tokenized sequences shaped ``(episodes, time, K)``.
    """

    stream_state_contract = "explicit"

    def __init__(
        self,
        tokenizer: VQVAE,
        action_dim: int,
        *,
        hidden_dim: int = 32,
        epochs: int = 40,
        learning_rate: float = 0.01,
        seed: int = 0,
        model_revision: str = "compact-tokenized-world-model-v1",
        codebook_version: str | None = None,
    ) -> None:
        if action_dim < 1 or hidden_dim < 2 or epochs < 1 or learning_rate <= 0.0:
            raise ValueError("action_dim and hidden_dim must be positive; epochs and learning_rate must be positive")
        tokenizer_version = tokenizer.codebook_version
        validate_codebook_version(codebook_version, tokenizer_version)
        self.tokenizer = tokenizer
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.model_revision = model_revision
        self.codebook_version = tokenizer_version
        self.vocab_size = tokenizer.codebook_size
        self.tokens_per_frame = tokenizer.sequence_length
        self.pad_token_id = self.vocab_size
        self._dynamics = _AutoregressiveDynamics(self.vocab_size, action_dim, hidden_dim, self.pad_token_id)
        self._fitted = False
        self._fit_metadata: dict[str, object] = {}

    @property
    def latent_space(self) -> LatentSpace:
        """Return the discrete frame-token geometry and version binding."""

        self._validate_tokenizer_version()
        return LatentSpace(
            dim=self.tokens_per_frame,
            geometry="discrete_code",
            source_model=self.model_revision,
            codebook_size=self.vocab_size,
            metadata={
                "representation": "autoregressive_integer_code_sequence",
                "tokens_per_frame": self.tokens_per_frame,
                "codebook_version": self.codebook_version,
                "tokenizer_revision": self.tokenizer.model_revision,
                "action_dim": self.action_dim,
                "padding_token_id": self.pad_token_id,
                "interpolation": "unsupported",
            },
        )

    @property
    def state_dim(self) -> int:
        """Return the number of token IDs representing one frame."""
        return self.tokens_per_frame

    @property
    def action_shape(self) -> tuple[int]:
        """Return the one-dimensional action shape for dynamics inputs."""
        return (self.action_dim,)

    @property
    def action_dim(self) -> int:
        """Return the configured action width."""
        return self._action_dim

    @action_dim.setter
    def action_dim(self, value: int) -> None:
        """Set the action width used to construct the dynamics model."""
        self._action_dim = value

    @property
    def state_shape(self) -> tuple[int]:
        """Return the token-ID state shape for one frame."""
        return (self.tokens_per_frame,)

    @property
    def source_space_identity(self) -> str:
        """Return the tokenizer revision and codebook identity for this state space."""
        self._validate_tokenizer_version()
        return f"{self.tokenizer.model_revision}:{self.codebook_version}:{self.tokens_per_frame}tokens"

    @property
    def is_fitted(self) -> bool:
        """Return whether the autoregressive dynamics model has been fitted."""
        return self._fitted

    @property
    def fit_metadata(self) -> Mapping[str, object]:
        """Return a defensive mapping of fit, tokenizer, and provenance metadata."""
        return MappingProxyType(dict(self._fit_metadata))

    def to_config(self) -> TokenizedWorldModelConfig:
        """Return the serializable dynamics configuration and codebook binding."""
        return TokenizedWorldModelConfig(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            seed=self.seed,
            model_revision=self.model_revision,
            codebook_version=self.codebook_version,
        )

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Tokenize a batch of raw observations with the frozen VQ-VAE."""

        self._validate_tokenizer_version()
        return self.tokenizer.encode(data)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode an integer token batch through the frozen VQ-VAE."""

        self._validate_tokenizer_version()
        self._validate_tokens(latent, name="latent")
        return self.tokenizer.decode(latent)

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        sequence_mask: np.ndarray | None = None,
        codebook_version: str | None = None,
    ) -> None:
        """Fit next-frame dynamics from raw image or integer-token sequences."""

        self._validate_tokenizer_version()
        if observations.ndim == 3:
            self.fit_tokens(observations, actions, sequence_mask=sequence_mask, codebook_version=codebook_version)
            return
        if observations.ndim != 5:
            raise ValueError("observations must be (episodes, time, 1, 8, 8) or (episodes, time, tokens)")
        flattened = observations.reshape((-1, *observations.shape[2:]))
        tokens = self.encode(flattened).reshape(observations.shape[0], observations.shape[1], self.tokens_per_frame)
        self.fit_tokens(tokens, actions, sequence_mask=sequence_mask, codebook_version=codebook_version)
        self._fit_metadata["input_representation"] = "raw_observations"

    def fit_tokens(
        self,
        token_sequences: np.ndarray,
        actions: np.ndarray,
        *,
        sequence_mask: np.ndarray | None = None,
        codebook_version: str | None = None,
    ) -> None:
        """Fit teacher-forced next-token prediction on a masked sequence batch."""

        self._validate_tokenizer_version()
        tokens = self._validate_sequence_tokens(token_sequences, allow_padding=True)
        action_values, mask = self._validate_training_actions(actions, sequence_mask, tokens.shape[:2])
        validate_sequence_codebook_version(codebook_version, self.codebook_version)
        valid = mask.astype(bool)
        current = tokens[:, :-1][valid]
        target = tokens[:, 1:][valid]
        current = self._validate_tokens(current, name="current tokens")
        target = self._validate_tokens(target, name="target tokens")
        flat_actions = action_values[valid]
        if current.shape[0] == 0:
            raise ValueError("sequence_mask must select at least one transition")

        result = fit_token_dynamics(
            self._dynamics,
            current,
            target,
            flat_actions,
            vocab_size=self.vocab_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )
        self._fitted = True
        self._fit_metadata = {
            "valid_transitions": int(current.shape[0]),
            "masked_transitions": int(mask.size - mask.sum()),
            "last_cross_entropy": result.loss,
            "last_token_accuracy": result.accuracy,
            "codebook_version": self.codebook_version,
            "input_representation": "token_sequences",
        }

    def predict_next(
        self,
        tokens: np.ndarray,
        action: np.ndarray,
        *,
        sampling: SamplingMode = "greedy",
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> TokenPrediction:
        """Generate one next frame autoregressively with optional seeded sampling."""

        self._require_fitted()
        current = self._validate_tokens(tokens, name="tokens", allow_single=True)
        actions = self._validate_actions(action, batch_size=current.shape[0])
        if sampling not in {"greedy", "sample"}:
            raise ValueError("sampling must be 'greedy' or 'sample'")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and not 1 <= top_k <= self.vocab_size:
            raise ValueError(f"top_k must be between 1 and {self.vocab_size}")
        result = sample_next_tokens(
            self._dynamics,
            current,
            actions,
            vocab_size=self.vocab_size,
            tokens_per_frame=self.tokens_per_frame,
            sampling=sampling,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        return TokenPrediction(result.tokens, result.token_log_likelihood, sampling)

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Return one greedy next token frame for the transition contract."""

        return self.predict_next(state, action).tokens[0]

    def mean_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Trajectory:
        """Run the deterministic greedy token rollout used by ``RolloutPipeline``."""

        return self.rollout(initial_state, actions, metadata=metadata)

    def rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        sampling: SamplingMode = "greedy",
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Trajectory:
        """Roll out a token sequence with a deterministic or seeded sampler."""

        self._require_fitted()
        state = self._validate_tokens(initial_state, name="initial_state", allow_single=True)[0]
        action_values = _finite_array(actions, name="actions")
        if action_values.ndim != 2 or action_values.shape[1] != self.action_dim:
            raise ValueError(f"actions must have shape (horizon, {self.action_dim})")
        values = [state.copy()]
        for index, action in enumerate(action_values):
            prediction = self.predict_next(
                state[None, :],
                action[None, :],
                sampling=sampling,
                temperature=temperature,
                top_k=top_k,
                seed=None if seed is None else seed + index,
            )
            state = prediction.tokens[0]
            values.append(state.copy())
        rollout_metadata = dict(metadata or {})
        rollout_metadata.update(
            {
                "model_revision": self.model_revision,
                "codebook_version": self.codebook_version,
                "sampling": sampling,
                "seed": seed,
                "padding_token_id": self.pad_token_id,
            }
        )
        return Trajectory(np.asarray(values, dtype=np.int64), metadata=rollout_metadata)

    def evaluate(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        sequence_mask: np.ndarray | None = None,
        task_proxy: Callable[[np.ndarray], np.ndarray] | None = None,
        failure_threshold: float = 0.5,
    ) -> TokenizedEvaluationReport:
        """Compare teacher-forced likelihood with free-running drift."""

        self._require_fitted()
        token_sequences, decoded_targets = self._prepare_evaluation_observations(observations)
        action_values, mask = self._validate_training_actions(actions, sequence_mask, token_sequences.shape[:2])
        teacher = self._teacher_forced_metrics(token_sequences, action_values, mask)
        free = self._free_running_metrics(
            token_sequences,
            action_values,
            mask,
            decoded_targets=decoded_targets,
            task_proxy=task_proxy,
            failure_threshold=failure_threshold,
        )
        codebook = self.tokenizer.codebook_metadata()
        codebook["codebook_version"] = self.codebook_version
        return TokenizedEvaluationReport(
            teacher_forced=teacher,
            free_running=free,
            codebook=codebook,
            provenance={
                "model_revision": self.model_revision,
                "tokenizer_revision": self.tokenizer.model_revision,
                "codebook_version": self.codebook_version,
                "source_space_identity": self.source_space_identity,
                "input_representation": "token_sequences" if observations.ndim == 3 else "raw_observations",
            },
        )

    def code_usage(self, tokens: np.ndarray) -> dict[str, object]:
        """Return code counts, perplexity, and dead-code rate for token IDs."""

        self._validate_tokenizer_version()
        values = self._validate_tokens(tokens, name="tokens", allow_sequence=True)
        counts = np.bincount(values.reshape(-1), minlength=self.vocab_size).astype(np.int64)
        probabilities = counts / max(float(counts.sum()), 1.0)
        active = probabilities > 0
        entropy = -float(np.sum(probabilities[active] * np.log(probabilities[active])))
        return {
            "counts": counts.tolist(),
            "perplexity": float(np.exp(entropy)),
            "dead_code_rate": float(np.mean(counts == 0)),
        }

    def _teacher_forced_metrics(
        self, tokens: np.ndarray, actions: np.ndarray, mask: np.ndarray
    ) -> TokenPredictionMetrics:
        cross_entropy, perplexity, accuracy, n_tokens = teacher_forced_metrics(
            self._dynamics,
            tokens,
            actions,
            mask,
            vocab_size=self.vocab_size,
        )
        return TokenPredictionMetrics(cross_entropy, perplexity, accuracy, n_tokens)

    def _free_running_metrics(
        self,
        tokens: np.ndarray,
        actions: np.ndarray,
        mask: np.ndarray,
        *,
        decoded_targets: np.ndarray | None,
        task_proxy: Callable[[np.ndarray], np.ndarray] | None,
        failure_threshold: float,
    ) -> TokenRolloutMetrics:
        errors, exact, decoded_errors, task_accuracy, failure_horizon = free_running_metrics(
            self._dynamics,
            tokens,
            actions,
            mask,
            vocab_size=self.vocab_size,
            tokens_per_frame=self.tokens_per_frame,
            pad_token_id=self.pad_token_id,
            decode=self.decode if decoded_targets is not None else None,
            decoded_targets=decoded_targets,
            task_proxy=task_proxy,
            failure_threshold=failure_threshold,
        )
        return TokenRolloutMetrics(
            tuple(errors),
            tuple(exact),
            None if decoded_errors is None else tuple(decoded_errors),
            None if task_accuracy is None else tuple(task_accuracy),
            failure_horizon,
        )

    def _prepare_evaluation_observations(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if observations.ndim == 3:
            tokens = self._validate_sequence_tokens(observations, allow_padding=False)
            decoded = self.decode(tokens.reshape(-1, self.tokens_per_frame)).reshape(
                tokens.shape[0], tokens.shape[1], *self.tokenizer.input_shape
            )
            return tokens, decoded
        if observations.ndim != 5:
            raise ValueError("observations must be a token sequence or raw image sequence")
        flattened = observations.reshape((-1, *observations.shape[2:]))
        decoded = observations.astype(np.float64, copy=True)
        tokens = self.encode(flattened).reshape(observations.shape[0], observations.shape[1], self.tokens_per_frame)
        return tokens, decoded

    def _validate_sequence_tokens(self, tokens: np.ndarray, *, allow_padding: bool) -> np.ndarray:
        return validate_sequence_tokens(
            tokens,
            tokens_per_frame=self.tokens_per_frame,
            pad_token_id=self.pad_token_id,
            vocab_size=self.vocab_size,
            allow_padding=allow_padding,
        )

    def _validate_tokens(
        self,
        tokens: np.ndarray,
        *,
        name: str,
        allow_single: bool = False,
        allow_sequence: bool = False,
    ) -> np.ndarray:
        return validate_tokens(
            tokens,
            name=name,
            tokens_per_frame=self.tokens_per_frame,
            vocab_size=self.vocab_size,
            allow_single=allow_single,
            allow_sequence=allow_sequence,
        )

    def _validate_actions(self, actions: np.ndarray, *, batch_size: int) -> np.ndarray:
        return validate_actions(actions, batch_size=batch_size, action_dim=self.action_dim)

    def _validate_training_actions(
        self,
        actions: np.ndarray,
        sequence_mask: np.ndarray | None,
        sequence_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_training_actions(
            actions,
            sequence_mask,
            sequence_shape,
            action_dim=self.action_dim,
        )

    def _require_fitted(self) -> None:
        self._validate_tokenizer_version()
        if not self._fitted:
            raise RuntimeError("TokenizedWorldModel must be fitted before prediction")

    def _validate_tokenizer_version(self) -> None:
        validate_tokenizer_binding(self.tokenizer.codebook_version, self.codebook_version)


__all__ = [
    "TokenPrediction",
    "TokenPredictionMetrics",
    "TokenRolloutMetrics",
    "TokenizedEvaluationReport",
    "TokenizedWorldModel",
    "TokenizedWorldModelConfig",
]
