"""Revision-pinned decoder-only transformer integration with native hidden-state
observation and a direct logit lens.

Design
------
This is a **concrete integration**, not a ``ModelAdapter`` implementation. The
full decoder-only transformer lifecycle (tokenization, embedding, forward pass,
hidden-state capture, logit-lens projection) does not fit the ``encode()`` /
``decode()`` / ``latent_space`` contract — it owns tokenized inputs, an LM head,
a final normalisation layer, and multiple hidden-state representations. Collapsing
into ``encode()`` would hide meaningful layer-by-layer semantics.

**No generative protocol is introduced.** Sharing a generative interface
requires >=3 differing integrations (per Rule of Three). This second concrete
integration proves the common shape; extraction waits for repetitions.

Native hidden-state outputs (``output_hidden_states=True``) are the canonical
observation path — hooks are used only for intervention.

The direct logit lens applies the model's own final-layer normalisation and LM
head to pre-final native hidden states, while applying only the LM head to
the terminal native state because Hugging Face GPT-2 already applies ``ln_f``
before storing it. Learned/tuned translators are explicitly deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from latent_anything._transformer_analysis import (
    apply_logit_lens,
    compute_token_rank_trajectories,
    compute_top_tokens,
    softmax,
)
from latent_anything._transformer_backend import load_backend
from latent_anything._transformer_backend import tokenize as tokenize_backend
from latent_anything._transformer_runtime import run_generation
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue

# ---------------------------------------------------------------------------
# Pinned model identity   (Task 1)
# ---------------------------------------------------------------------------

TRANSFORMER_MODEL_ID = "openai-community/gpt2"
"""HuggingFace model ID for the pinned decoder-only transformer."""

TRANSFORMER_MODEL_REVISION = "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
"""Pinned revision for reproducible behaviour across installations.

This is the canonical ``openai-community/gpt2`` model checkpoint (124M
parameters). Pinned by its immutable Hugging Face commit hash so that offline
tests and benchmarks are reproducible across time.
"""

# Tested dependency ranges (installation guards)
TESTED_TRANSFORMERS_RANGE = ">=4.45,<5.0"

# GPT-2 configuration constants (gpt2-small / 124M).
GPT2_NUM_LAYERS = 12
GPT2_NUM_HEADS = 12
GPT2_HIDDEN_DIM = 768
GPT2_VOCAB_SIZE = 50257
GPT2_MAX_POSITION_EMBEDDINGS = 1024

# ---------------------------------------------------------------------------
# Public data types   (Task 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformerInput:
    """Typed input for one transformer forward pass.

    All arrays are non-writable NumPy arrays with explicit batch and sequence
    axes.

    Parameters
    ----------
    input_ids:
        Token IDs as ``(batch_size, seq_len)`` int64 array.
    attention_mask:
        Attention mask as ``(batch_size, seq_len)`` int64 array (1 = real
        token, 0 = padding).
    """

    input_ids: np.ndarray
    attention_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D (batch, seq), got {self.input_ids.ndim}D")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError(
                f"attention_mask shape {self.attention_mask.shape} must match input_ids shape {self.input_ids.shape}"
            )


@dataclass(frozen=True)
class LayerIndex:
    """Typed native output index selection for hidden-state capture.

    Parameters
    ----------
    layer:
        Layer index (0-based). 0 is the embedding output; 1 is the first
        transformer block output, etc.
    """

    layer: int

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError(f"layer must be >= 0, got {self.layer}")


@dataclass(frozen=True)
class TokenMask:
    """Mask indicating which tokens are real vs padding.

    Parameters
    ----------
    mask:
        Boolean array of shape ``(batch_size, seq_len)`` where ``True``
        means a real (non-padded) token.
    """

    mask: np.ndarray

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            object.__setattr__(self, "mask", self.mask.astype(np.bool_))
        if self.mask.ndim != 2:
            raise ValueError(f"mask must be 2D (batch, seq), got {self.mask.ndim}D")


@dataclass(frozen=True)
class HiddenState:
    """A single hidden state captured from a transformer layer.

    Parameters
    ----------
    layer:
        0-based layer index this hidden state came from.
    values:
        Hidden state as ``(batch_size, seq_len, hidden_dim)`` non-writable
        NumPy array.
    provenance:
        The model/tokenizer provenance string.
    metadata:
        Additional metadata (device, dtype, etc.).
    """

    layer: int
    values: np.ndarray
    provenance: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(f"hidden state must be 3D (batch, seq, hidden), got {self.values.ndim}D")
        if self.layer < 0:
            raise ValueError(f"layer must be >= 0, got {self.layer}")


@dataclass(frozen=True)
class LogitLensResult:
    """Result of applying the direct logit lens at one layer.

    The direct logit lens applies the model's final LayerNorm and LM head
    to the hidden state at a given layer, producing token logits and
    probabilities.

    Parameters
    ----------
    layer:
        0-based layer index.
    logits:
        Raw logits as ``(batch_size, seq_len, vocab_size)`` float32 array.
    probabilities:
        Softmax probabilities as ``(batch_size, seq_len, vocab_size)``
        float32 array.
    top_tokens:
        List of (token_id, probability) for the top-K predicted tokens at
        each sequence position, shaped as ``(batch_size, seq_len, top_k)``
        list of tuples. Only populated if ``top_k > 0``.
    top_k:
        Number of top tokens stored.
    """

    layer: int
    logits: np.ndarray
    probabilities: np.ndarray
    top_tokens: list[list[list[tuple[int, float]]]] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    top_k: int = 0

    def __post_init__(self) -> None:
        if self.logits.ndim != 3:
            raise ValueError(f"logits must be 3D (batch, seq, vocab), got {self.logits.ndim}D")
        if self.probabilities.shape != self.logits.shape:
            raise ValueError("probabilities shape must match logits shape")


@dataclass(frozen=True)
class TokenRankTrajectory:
    """Rank and probability trajectory for a selected token across layers.

    Parameters
    ----------
    token_id:
        The token ID being tracked.
    token_str:
        The decoded string form of the token.
    ranks:
        Rank of this token at each layer (1 = most probable).
    probabilities:
        Probability of this token at each layer.
    layers:
        The layer indices corresponding to each entry.
    """

    token_id: int
    token_str: str
    ranks: list[int]
    probabilities: list[float]
    layers: list[int]


@dataclass(frozen=True)
class TransformerGenerationRequest:
    """Typed request for one transformer forward pass with hidden-state capture.

    Parameters
    ----------
    prompt:
        Text prompt or tuple of prompts (batched generation).
    max_length:
        Maximum total length (prompt + generated tokens).
    seed:
        Random seed for reproducibility.
    capture_hidden_states:
        If True, record all hidden states from all layers.
    capture_layers:
        Specific native ``output_hidden_states`` indices to capture. Native
        index 0 is the embedding output and transformer block ``L``'s output
        is index ``L + 1``. If empty and ``capture_hidden_states`` is True,
        captures all native indices (0 to n_layers).
    top_k_logit_lens:
        Number of top tokens to store in each LogitLensResult (0 = skip).
    """

    prompt: str | tuple[str, ...]
    max_length: int = 128
    seed: int = 42
    capture_hidden_states: bool = True
    capture_layers: tuple[int, ...] = ()
    top_k_logit_lens: int = 5

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {self.max_length}")


@dataclass(frozen=True)
class TransformerGenerationResult:
    """Complete typed result of one transformer forward pass.

    Parameters
    ----------
    input_ids:
        Input token IDs used for the forward pass.
    attention_mask:
        Attention mask used.
    logits:
        Final-layer logits as ``(batch_size, seq_len, vocab_size)``.
    hidden_states:
        All captured hidden states, one per layer.
    lens_results:
        Results of applying the direct logit lens to each captured layer.
    token_rank_trajectories:
        Rank/probability trajectories for selected tokens.
    prompt:
        The original prompt string(s).
    provenance:
        Model/tokenizer provenance string.
    """

    input_ids: np.ndarray
    attention_mask: np.ndarray
    logits: np.ndarray
    hidden_states: tuple[HiddenState, ...]
    lens_results: tuple[LogitLensResult, ...]
    token_rank_trajectories: tuple[TokenRankTrajectory, ...]
    prompt: str | tuple[str, ...]
    provenance: str = ""


@dataclass(frozen=True)
class HiddenStateIntervention:
    """Intervention on a specific layer's hidden state during forward pass.

    Applies ``hidden ← hidden + strength * direction`` at the specified
    layer and token positions.

    Parameters
    ----------
    layer:
        Zero-based transformer block index to intervene on. The runtime maps
        this value to ``transformer.h.<layer>``; the corresponding native
        ``output_hidden_states`` block output is at index ``layer + 1``.
    direction:
        Direction vector as ``(1, 1, hidden_dim)`` or
        ``(batch_size, seq_len, hidden_dim)`` non-writable NumPy array.
    strength:
        Strength multiplier (>= 0). Zero means no effect.
    token_indices:
        If not None, only intervene on these token positions
        ``(batch_idx, seq_idx)`` pairs.
    """

    layer: int
    direction: np.ndarray
    strength: float
    token_indices: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        if self.direction.ndim != 3:
            raise ValueError(f"direction must be 3D (batch, seq, hidden), got {self.direction.ndim}D")
        if self.strength < 0:
            raise ValueError(f"strength must be >= 0, got {self.strength}")
        if self.layer < 0:
            raise ValueError(f"layer must be >= 0, got {self.layer}")


# ---------------------------------------------------------------------------
# Transformer integration   (Tasks 1, 3, 4, 5)
# ---------------------------------------------------------------------------


class TransformerLMIntegration:
    """Revision-pinned decoder-only transformer integration.

    Provides a concrete forward-pass lifecycle with optional hidden-state
    capture, direct logit lens analysis, and bounded activation intervention.

    Parameters
    ----------
    model_id:
        HuggingFace model ID (default: pinned ``gpt2``).
    revision:
        Git revision (commit hash or tag; default: pinned revision).
    device:
        Torch device string (``"cpu"``, ``"cuda"``, …).
    dtype:
        NumPy dtype for the public boundary.
    """

    def __init__(
        self,
        model_id: str = TRANSFORMER_MODEL_ID,
        revision: str = TRANSFORMER_MODEL_REVISION,
        *,
        device: str = "cpu",
        dtype: np.dtype | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = np.dtype(np.float32 if dtype is None else dtype)
        self._model: Any = None
        self._tokenizer: Any = None
        self._config: Any = None

    # -- internal helpers ---------------------------------------------------

    def _torch_dtype(self) -> Any:
        import torch

        return torch.float32

    @property
    def provenance(self) -> str:
        """Return a provenance string for this integration instance."""
        return f"{self.model_id}@{self.revision}"

    def _backend(self) -> tuple[Any, Any, Any]:
        """Lazy import and construct the transformer model and tokenizer."""
        if self._model is not None:
            return self._model, self._tokenizer, self._config
        self._model, self._tokenizer, self._config = load_backend(
            self.model_id,
            self.revision,
            device=self.device,
            torch_dtype=self._torch_dtype(),
        )
        return self._model, self._tokenizer, self._config

    def tokenize(
        self,
        prompt: str | tuple[str, ...],
        max_length: int = 128,
        return_tensors: str = "pt",
    ) -> dict[str, Any]:
        """Tokenize a prompt string or tuple of prompts.

        Parameters
        ----------
        prompt:
            Text prompt or tuple of prompts.
        max_length:
            Maximum sequence length.
        return_tensors:
            Return format (``"pt"`` for PyTorch tensors).

        Returns
        -------
        dict
            Tokenizer output with ``input_ids`` and ``attention_mask``.
        """
        _, tokenizer, _ = self._backend()

        return tokenize_backend(
            tokenizer,
            prompt,
            max_length=max_length,
            return_tensors=return_tensors,
        )

    # -- public LatentSpace descriptors   (Task 3) --------------------------

    @property
    def hidden_state_space(self) -> LatentSpace:
        """Descriptor for transformer hidden states.

        Hidden states have shape ``(batch_size, seq_len, hidden_dim)`` where
        ``hidden_dim`` is the model's hidden dimension (768 for GPT-2 small).
        """
        return LatentSpace(
            dim=GPT2_HIDDEN_DIM,
            source_model=self.provenance,
            metadata={
                "role": "transformer_hidden_state",
                "model_type": "decoder_only",
                "num_layers": GPT2_NUM_LAYERS,
                "num_heads": GPT2_NUM_HEADS,
                "vocab_size": GPT2_VOCAB_SIZE,
            },
        )

    @property
    def logit_space(self) -> LatentSpace:
        """Descriptor for logit space (vocabulary distribution).

        Logits have shape ``(batch_size, seq_len, vocab_size)``.
        """
        return LatentSpace(
            dim=GPT2_VOCAB_SIZE,
            source_model=self.provenance,
            metadata={
                "role": "logits",
                "model_type": "decoder_only",
                "vocab_size": GPT2_VOCAB_SIZE,
            },
        )

    # -- Transformer forward pass with capture   (Task 3) -------------------

    def generate(
        self,
        request: TransformerGenerationRequest,
        intervention: HiddenStateIntervention | None = None,
    ) -> TransformerGenerationResult:
        """Run a transformer forward pass with optional hidden-state capture
        and direct logit lens analysis.

        Parameters
        ----------
        request:
            Typed generation parameters (prompt, max_length, capture config).
        intervention:
            Optional hidden-state intervention to apply during the forward pass.

        Returns
        -------
        TransformerGenerationResult
            Input IDs, logits, captured hidden states, and lens results.

        Raises
        ------
        ImportError
            If ``transformers`` is not installed.
        """
        model, tokenizer, config = self._backend()
        runtime_result = run_generation(
            model,
            tokenizer,
            config,
            request,
            intervention,
            device=self.device,
            provenance=self.provenance,
            default_num_layers=GPT2_NUM_LAYERS,
        )
        return TransformerGenerationResult(
            input_ids=runtime_result.input_ids,
            attention_mask=runtime_result.attention_mask,
            logits=runtime_result.logits,
            hidden_states=tuple(
                HiddenState(layer=layer, values=values, provenance=self.provenance, metadata=metadata)
                for layer, values, metadata in runtime_result.hidden_states
            ),
            lens_results=tuple(
                LogitLensResult(
                    layer=layer,
                    logits=logits,
                    probabilities=probabilities,
                    top_tokens=top_tokens,
                    top_k=top_k,
                )
                for layer, logits, probabilities, top_tokens, top_k in runtime_result.lens_results
            ),
            token_rank_trajectories=tuple(
                TokenRankTrajectory(
                    token_id=token_id,
                    token_str=token_str,
                    ranks=ranks,
                    probabilities=probabilities,
                    layers=layers,
                )
                for token_id, token_str, ranks, probabilities, layers in runtime_result.token_rank_trajectories
            ),
            prompt=request.prompt,
            provenance=self.provenance,
        )

    # -- Logit lens implementation   (Task 4) -------------------------------

    def _apply_logit_lens(
        self,
        model: Any,
        hidden_state: Any,
        *,
        apply_final_norm: bool = True,
    ) -> np.ndarray:
        """Apply the direct logit lens with optional final LayerNorm.

        Uses the model's own final normalisation layer when requested and
        always uses its language-modelling head to project hidden states into
        vocabulary space. Native terminal hidden states that already passed
        the final normalization must set ``apply_final_norm=False``. This is
        a direct (untrained) lens — no learned translator is involved.

        Parameters
        ----------
        model:
            The HuggingFace model.
        hidden_state:
            PyTorch tensor of shape ``(batch_size, seq_len, hidden_dim)``.
        apply_final_norm:
            Whether to apply the model's final normalization before the head.

        Returns
        -------
        np.ndarray
            Logits of shape ``(batch_size, seq_len, vocab_size)``.
        """
        return apply_logit_lens(model, hidden_state, apply_final_norm=apply_final_norm)

    # -- Helper methods -----------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along the specified axis."""
        return softmax(logits, axis=axis)

    @staticmethod
    def _compute_top_tokens(
        probabilities: np.ndarray,
        top_k: int,
        tokenizer: Any,  # noqa: ARG004
    ) -> list[list[list[tuple[int, float]]]]:
        """Compute top-k token IDs and probabilities for each position."""
        return compute_top_tokens(probabilities, top_k)

    @staticmethod
    def _compute_token_rank_trajectories(
        lens_results: list[LogitLensResult],
        seq_len: int,
        _batch_size: int,
        tokenizer: Any,
    ) -> tuple[TokenRankTrajectory, ...]:
        """Compute rank/probability trajectories for the most-probable token
        at each sequence position."""
        raw = compute_token_rank_trajectories(lens_results, seq_len, tokenizer)
        return tuple(
            TokenRankTrajectory(
                token_id=token_id,
                token_str=token_str,
                ranks=ranks,
                probabilities=probabilities,
                layers=layers,
            )
            for token_id, token_str, ranks, probabilities, layers in raw
        )

    # -- Hook-based intervention helpers   (Task 6) -------------------------

    @staticmethod
    def random_intervention_direction(
        hidden_dim: int = GPT2_HIDDEN_DIM,
        seed: int | None = None,
        *,
        strength: float = 1.0,
        layer: int = 6,
    ) -> HiddenStateIntervention:
        """Create a random hidden-state intervention direction.

        Parameters
        ----------
        hidden_dim:
            Hidden dimension size.
        seed:
            Optional RNG seed for reproducibility.
        strength:
            Intervention strength multiplier.
        layer:
            Target layer index.

        Returns
        -------
        HiddenStateIntervention
        """
        rng = np.random.default_rng(seed)
        direction = rng.normal(0, 1, size=(1, 1, hidden_dim)).astype(np.float32)
        direction.setflags(write=False)
        return HiddenStateIntervention(layer=layer, direction=direction, strength=strength)

    # -- Convenience wrappers -----------------------------------------------

    def hidden_state_value(self, hidden_state: np.ndarray) -> LatentValue:
        """Wrap a hidden state array in a :class:`LatentValue`."""
        return LatentValue(
            hidden_state,
            self.hidden_state_space,
            metadata={"role": "transformer_hidden_state"},
        )

    def logit_value(self, logits: np.ndarray) -> LatentValue:
        """Wrap a logits array in a :class:`LatentValue`."""
        return LatentValue(logits, self.logit_space, metadata={"role": "logits"})

    def decode_tokens(self, token_ids: np.ndarray) -> list[str]:
        """Decode token IDs to strings.

        Parameters
        ----------
        token_ids:
            Array of token IDs.

        Returns
        -------
        list[str]
            Decoded token strings.
        """
        _, tokenizer, _ = self._backend()
        flattened = token_ids.flatten()
        return [tokenizer.decode([int(tid)]) for tid in flattened]

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        _, _, config = self._backend()
        return int(getattr(config, "num_hidden_layers", GPT2_NUM_LAYERS))

    @property
    def hidden_dim(self) -> int:
        """Return the hidden dimension."""
        _, _, config = self._backend()
        return int(getattr(config, "hidden_size", GPT2_HIDDEN_DIM))

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        _, _, config = self._backend()
        return int(getattr(config, "vocab_size", GPT2_VOCAB_SIZE))
