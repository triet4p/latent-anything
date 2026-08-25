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
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn, optim

from latent_anything.adapters.vq_vae import VQVAE
from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory

SamplingMode = Literal["greedy", "sample"]


def _finite_array(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return value


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
        return len(self.token_error_by_horizon)

    def to_dict(self) -> dict[str, object]:
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


class _AutoregressiveDynamics(nn.Module):
    """Private GRU encoder/decoder used by ``TokenizedWorldModel``."""

    def __init__(self, vocab_size: int, action_dim: int, hidden_dim: int, pad_token_id: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=pad_token_id)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.bos = nn.Parameter(torch.zeros(hidden_dim))

    def encode_context(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_embedding = self.action_projection(actions).unsqueeze(1)
        values = self.token_embedding(tokens) + action_embedding
        _, hidden = self.encoder(values)
        return hidden

    def decode_teacher_forced(
        self, hidden: torch.Tensor, actions: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        action_embedding = self.action_projection(actions).unsqueeze(1)
        prefix = torch.cat(
            (self.bos.expand(target_tokens.shape[0], 1, -1), self.token_embedding(target_tokens[:, :-1])), dim=1
        )
        values = prefix + action_embedding
        decoded, _ = self.decoder(values, hidden)
        return self.output(decoded)

    def decode_one(self, hidden: torch.Tensor, actions: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        action_embedding = self.action_projection(actions)
        value = self.bos.expand(prefix.shape[0], -1) if prefix.shape[1] == 0 else self.token_embedding(prefix[:, -1])
        decoded, _ = self.decoder((value + action_embedding).unsqueeze(1), hidden)
        return self.output(decoded[:, 0])


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
        if codebook_version is not None and codebook_version != tokenizer_version:
            raise ValueError(f"codebook_version {codebook_version!r} does not match tokenizer {tokenizer_version!r}")
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
        return self.tokens_per_frame

    @property
    def action_shape(self) -> tuple[int]:
        return (self.action_dim,)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @action_dim.setter
    def action_dim(self, value: int) -> None:
        self._action_dim = value

    @property
    def state_shape(self) -> tuple[int]:
        return (self.tokens_per_frame,)

    @property
    def source_space_identity(self) -> str:
        self._validate_tokenizer_version()
        return f"{self.tokenizer.model_revision}:{self.codebook_version}:{self.tokens_per_frame}tokens"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def fit_metadata(self) -> Mapping[str, object]:
        return MappingProxyType(dict(self._fit_metadata))

    def to_config(self) -> TokenizedWorldModelConfig:
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
        if codebook_version is not None and codebook_version != self.codebook_version:
            raise ValueError("token sequence codebook_version does not match the frozen tokenizer")
        valid = mask.astype(bool)
        current = tokens[:, :-1][valid]
        target = tokens[:, 1:][valid]
        current = self._validate_tokens(current, name="current tokens")
        target = self._validate_tokens(target, name="target tokens")
        flat_actions = action_values[valid]
        if current.shape[0] == 0:
            raise ValueError("sequence_mask must select at least one transition")

        torch.manual_seed(self.seed)  # pyright: ignore[reportUnknownMemberType]
        torch.set_num_threads(1)
        source_tensor = torch.from_numpy(current.astype(np.int64))  # pyright: ignore[reportUnknownMemberType]
        target_tensor = torch.from_numpy(target.astype(np.int64))  # pyright: ignore[reportUnknownMemberType]
        action_tensor = torch.from_numpy(flat_actions.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
        optimizer = optim.Adam(self._dynamics.parameters(), lr=self.learning_rate)
        last_loss = 0.0
        last_accuracy = 0.0
        for _ in range(self.epochs):
            hidden = self._dynamics.encode_context(source_tensor, action_tensor)
            logits = self._dynamics.decode_teacher_forced(hidden, action_tensor, target_tensor)
            loss = nn.functional.cross_entropy(logits.reshape(-1, self.vocab_size), target_tensor.reshape(-1))
            optimizer.zero_grad()
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            with torch.no_grad():
                last_loss = float(loss.detach())  # pyright: ignore[reportUnknownMemberType]
                last_accuracy = float((logits.argmax(dim=-1) == target_tensor).float().mean())  # pyright: ignore[reportUnknownMemberType]
        self._fitted = True
        self._fit_metadata = {
            "valid_transitions": int(current.shape[0]),
            "masked_transitions": int(mask.size - mask.sum()),
            "last_cross_entropy": last_loss,
            "last_token_accuracy": last_accuracy,
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
        generator = np.random.default_rng(seed)
        current_tensor = torch.from_numpy(current.astype(np.int64))  # pyright: ignore[reportUnknownMemberType]
        action_tensor = torch.from_numpy(actions.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
        with torch.no_grad():
            hidden = self._dynamics.encode_context(current_tensor, action_tensor)
            prefix = torch.empty((current.shape[0], 0), dtype=torch.long)
            output = np.empty((current.shape[0], self.tokens_per_frame), dtype=np.int64)
            log_likelihood = np.empty_like(output, dtype=np.float64)
            for position in range(self.tokens_per_frame):
                logits = self._dynamics.decode_one(hidden, action_tensor, prefix)
                scaled = logits / temperature
                if top_k is not None and top_k < self.vocab_size:
                    values, indices = torch.topk(scaled, top_k, dim=-1)
                    filtered = torch.full_like(scaled, -torch.inf)
                    filtered.scatter_(1, indices, values)
                    scaled = filtered
                log_probs = torch.log_softmax(scaled, dim=-1)
                if sampling == "greedy":
                    next_token = torch.argmax(log_probs, dim=-1)
                else:
                    probabilities = torch.softmax(scaled, dim=-1).cpu().numpy()
                    sampled = [generator.choice(self.vocab_size, p=row) for row in probabilities]
                    next_token = torch.from_numpy(np.asarray(sampled, dtype=np.int64))
                output[:, position] = next_token.cpu().numpy()
                log_likelihood[:, position] = log_probs.gather(1, next_token[:, None]).squeeze(1).cpu().numpy()
                prefix = torch.cat((prefix, next_token[:, None]), dim=1)
        return TokenPrediction(output, log_likelihood, sampling)

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
        current = self._validate_tokens(tokens[:, :-1][mask.astype(bool)], name="current tokens")
        target = self._validate_tokens(tokens[:, 1:][mask.astype(bool)], name="target tokens")
        flat_actions = actions[mask.astype(bool)]
        current_tensor = torch.from_numpy(current.astype(np.int64))  # pyright: ignore[reportUnknownMemberType]
        target_tensor = torch.from_numpy(target.astype(np.int64))  # pyright: ignore[reportUnknownMemberType]
        action_tensor = torch.from_numpy(flat_actions.astype(np.float32))  # pyright: ignore[reportUnknownMemberType]
        with torch.no_grad():
            hidden = self._dynamics.encode_context(current_tensor, action_tensor)
            logits = self._dynamics.decode_teacher_forced(hidden, action_tensor, target_tensor)
            loss = nn.functional.cross_entropy(logits.reshape(-1, self.vocab_size), target_tensor.reshape(-1))
            accuracy = (logits.argmax(dim=-1) == target_tensor).float().mean()
        n_tokens = int(target.size)
        cross_entropy = float(loss)  # pyright: ignore[reportUnknownArgumentType]
        return TokenPredictionMetrics(cross_entropy, float(np.exp(cross_entropy)), float(accuracy), n_tokens)  # pyright: ignore[reportUnknownArgumentType]

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
        if not 0.0 < failure_threshold <= 1.0:
            raise ValueError("failure_threshold must be in (0, 1]")
        horizon = actions.shape[1]
        errors: list[float] = []
        exact: list[float] = []
        decoded_errors: list[float] | None = [] if decoded_targets is not None else None
        task_accuracy: list[float] | None = [] if task_proxy is not None and decoded_targets is not None else None
        current = tokens[:, 0].copy()
        for index in range(horizon):
            valid = mask[:, index].astype(bool)
            target = tokens[:, index + 1]
            prediction = self.predict_next(current, actions[:, index])
            predicted = prediction.tokens
            if not np.any(valid):
                errors.append(float("nan"))
                exact.append(float("nan"))
            else:
                errors.append(float(np.mean(predicted[valid] != target[valid])))
                exact.append(float(np.mean(np.all(predicted[valid] == target[valid], axis=1))))
                if decoded_errors is not None and decoded_targets is not None:
                    decoded = self.decode(predicted[valid])
                    decoded_errors.append(float(np.mean(np.square(decoded - decoded_targets[valid, index + 1]))))
                    if task_accuracy is not None and task_proxy is not None:
                        predicted_labels = np.asarray(task_proxy(decoded))
                        target_labels = np.asarray(task_proxy(decoded_targets[valid, index + 1]))
                        if predicted_labels.shape != target_labels.shape:
                            raise ValueError("task_proxy must return matching label shapes")
                        task_accuracy.append(float(np.mean(predicted_labels == target_labels)))
            current = predicted
        failure_horizon = next(
            (index + 1 for index, value in enumerate(errors) if np.isfinite(value) and value > failure_threshold),
            None,
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
        values = _finite_array(tokens, name="token_sequences")
        if values.ndim != 3 or values.shape[2] != self.tokens_per_frame:
            raise ValueError(f"token_sequences must have shape (episodes, time, {self.tokens_per_frame})")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("token_sequences must preserve integer token IDs")
        if allow_padding:
            if np.any(values < 0) or np.any(values > self.pad_token_id):
                raise ValueError("token IDs must be within the codebook or padding token")
        else:
            self._validate_tokens(values.reshape(-1, self.tokens_per_frame), name="token_sequences")
        if values.shape[1] < 2:
            raise ValueError("token sequences need at least an initial and next frame")
        return values.astype(np.int64, copy=False)

    def _validate_tokens(
        self,
        tokens: np.ndarray,
        *,
        name: str,
        allow_single: bool = False,
        allow_sequence: bool = False,
    ) -> np.ndarray:
        values = _finite_array(tokens, name=name)
        expected_ndim = 2
        if allow_sequence:
            expected_ndim = 3
        valid_single = allow_single and values.ndim in {1, 2}
        if (values.ndim != expected_ndim and not valid_single) or values.shape[-1] != self.tokens_per_frame:
            raise ValueError(f"{name} must end with token shape ({self.tokens_per_frame},)")
        if not np.issubdtype(values.dtype, np.integer):
            if not np.all(values == np.floor(values)):
                raise TypeError(f"{name} must contain integer token IDs")
            values = values.astype(np.int64)
        if np.any(values < 0) or np.any(values >= self.vocab_size):
            raise ValueError(f"{name} contains invalid token IDs; expected [0, {self.vocab_size - 1}]")
        if allow_single and values.ndim == 1:
            values = values[None, :]
        return values.astype(np.int64, copy=False)

    def _validate_actions(self, actions: np.ndarray, *, batch_size: int) -> np.ndarray:
        values = _finite_array(actions, name="action")
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape != (batch_size, self.action_dim):
            raise ValueError(f"action must have shape ({batch_size}, {self.action_dim})")
        return values.astype(np.float64, copy=False)

    def _validate_training_actions(
        self,
        actions: np.ndarray,
        sequence_mask: np.ndarray | None,
        sequence_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        values = _finite_array(actions, name="actions")
        expected_shape = (sequence_shape[0], sequence_shape[1] - 1, self.action_dim)
        if values.ndim != 3 or values.shape != expected_shape:
            raise ValueError(f"actions must have shape {expected_shape}")
        if sequence_mask is None:
            mask = np.ones(values.shape[:2], dtype=bool)
        else:
            mask_values = _finite_array(sequence_mask, name="sequence_mask")
            if mask_values.shape != values.shape[:2] or not np.all(np.isin(mask_values, [0, 1])):
                raise ValueError("sequence_mask must be a binary array matching the transition batch")
            mask = mask_values.astype(bool)
        return values.astype(np.float64, copy=False), mask

    def _require_fitted(self) -> None:
        self._validate_tokenizer_version()
        if not self._fitted:
            raise RuntimeError("TokenizedWorldModel must be fitted before prediction")

    def _validate_tokenizer_version(self) -> None:
        if self.tokenizer.codebook_version != self.codebook_version:
            raise ValueError("tokenizer checkpoint changed after TokenizedWorldModel construction")


__all__ = [
    "TokenPrediction",
    "TokenPredictionMetrics",
    "TokenRolloutMetrics",
    "TokenizedEvaluationReport",
    "TokenizedWorldModel",
    "TokenizedWorldModelConfig",
]
