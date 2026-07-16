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
head to each layer's hidden state, producing token probability distributions
from intermediate layers. Learned/tuned translators are explicitly deferred.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from latent_anything.capture import ActivationCaptureSession
from latent_anything.integrations import require_optional
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue

# ---------------------------------------------------------------------------
# Pinned model identity   (Task 1)
# ---------------------------------------------------------------------------

TRANSFORMER_MODEL_ID = "gpt2"
"""HuggingFace model ID for the pinned decoder-only transformer."""

TRANSFORMER_MODEL_REVISION = "e7da7f221d5bf496a4811970ad59b19a5b3ff2a4"
"""Pinned revision for reproducible behaviour across installations.

This is the original ``gpt2`` model checkpoint (124M parameters). Pinned by
commit hash so that offline tests and benchmarks are reproducible across time.
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
    """Typed layer index selection for hidden-state capture.

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
        Specific layer indices to capture. If empty and
        ``capture_hidden_states`` is True, captures all layers (0 to n_layers).
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
        0-based layer index to intervene on.
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

        transformers = require_optional("transformers", extra="transformers")

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
        )

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=self._torch_dtype(),
        )
        self._model = self._model.to(self.device)
        self._model.eval()

        self._config = self._model.config

        # Set pad token if not set (GPT-2 doesn't have one by default).
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

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

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        return dict(encoded)

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
        import torch

        model, tokenizer, config = self._backend()

        # Tokenize
        encoded = self.tokenize(request.prompt, max_length=request.max_length)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        batch_size, seq_len = input_ids.shape

        # Determine which layers to capture.
        num_layers = int(getattr(config, "num_hidden_layers", GPT2_NUM_LAYERS))
        if request.capture_layers:
            capture_layers = sorted(request.capture_layers)
        elif request.capture_hidden_states:
            # Layers 0..num_layers: 0 = embedding, 1..num_layers = transformer blocks
            capture_layers = tuple(range(num_layers + 1))
        else:
            capture_layers = ()

        # Containers for captured hidden states and lens results.
        captured_states: list[HiddenState] = []
        lens_results: list[LogitLensResult] = []

        # ------------------------------------------------------------------
        # Intervention via ActivationCaptureSession   (Task 6)
        # ------------------------------------------------------------------
        need_intervention = intervention is not None
        need_capture = request.capture_hidden_states or len(capture_layers) > 0

        # Build the list of module locations to hook.
        module_locations: list[str] = []

        if need_intervention:
            # Hook the specific transformer block for intervention.
            # GPT-2 block naming: ``transformer.h.{layer}``
            location = f"transformer.h.{intervention.layer}"  # type: ignore[union-attr]
            if location not in module_locations:
                module_locations.append(location)

        if need_capture:
            # Hook each layer we want to capture from.
            # For native hidden states via output_hidden_states=True, we don't
            # need hooks for observation — we read from the model output directly.
            # But for intervention we need hooks.
            pass  # We use native output_hidden_states for observation

        # Build intervention callback if needed.
        intervention_fn: Any = None
        if need_intervention:
            direction_t = torch.tensor(intervention.direction, dtype=torch.float32)  # type: ignore[union-attr]
            strength_val = intervention.strength
            token_indices = intervention.token_indices
            target_dtype = direction_t.dtype

            def _intervene_cb(tensor: torch.Tensor, metadata: Any) -> torch.Tensor:  # noqa: ARG001
                modified = tensor.clone()
                delta = strength_val * direction_t.to(device=tensor.device, dtype=tensor.dtype)
                if token_indices is not None:
                    for b_idx, s_idx in token_indices:
                        if b_idx < modified.shape[0] and s_idx < modified.shape[1]:
                            modified[b_idx, s_idx] = modified[b_idx, s_idx] + delta[b_idx, s_idx]
                else:
                    modified = modified + delta
                return modified.to(dtype=target_dtype)

            intervention_fn = _intervene_cb

        # Open the capture session for intervention if needed.
        capture_session = nullcontext()
        if module_locations and intervention_fn is not None:
            capture_session = ActivationCaptureSession(
                model,
                module_locations,
                source_model_version=self.provenance,
                intervention=intervention_fn,
            )

        # ------------------------------------------------------------------
        # Forward pass   (Task 3: native output_hidden_states=True)
        # ------------------------------------------------------------------
        import torch

        with capture_session, torch.no_grad():  # type: ignore[union-attr]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,  # native hidden-state path
            )

        # Extract final logits.
        final_logits = outputs.logits.detach().cpu().numpy().copy()
        final_logits.setflags(write=False)

        # Extract native hidden states from the model output.
        # outputs.hidden_states is a tuple of length num_layers+1:
        #   (embedding_output, layer_0_output, layer_1_output, ..., layer_{N-1}_output)
        native_hidden_states = outputs.hidden_states
        provenance_str = self.provenance

        # -- Process hidden states from native output   (Task 3) --
        if native_hidden_states is not None and need_capture:
            for layer_idx in capture_layers:
                if layer_idx < len(native_hidden_states):
                    hs = native_hidden_states[layer_idx]
                    hs_np = hs.detach().cpu().numpy().copy()
                    hs_np.setflags(write=False)
                    captured_states.append(
                        HiddenState(
                            layer=layer_idx,
                            values=hs_np,
                            provenance=provenance_str,
                            metadata={
                                "shape": str(tuple(hs_np.shape)),
                                "source": "native_output_hidden_states",
                            },
                        )
                    )

        # -- Direct logit lens   (Task 4) --
        if native_hidden_states is not None:
            for layer_idx in capture_layers:
                if layer_idx < len(native_hidden_states):
                    hs = native_hidden_states[layer_idx]
                    lens_np = self._apply_logit_lens(model, hs)
                    probs = self._softmax(lens_np)

                    # Build top tokens if requested.
                    top_tokens: list[list[list[tuple[int, float]]]] = []
                    if request.top_k_logit_lens > 0:
                        top_tokens = self._compute_top_tokens(
                            probs,
                            request.top_k_logit_lens,
                            tokenizer,  # noqa: B038
                        )

                    lens_results.append(
                        LogitLensResult(
                            layer=layer_idx,
                            logits=lens_np,
                            probabilities=probs,
                            top_tokens=top_tokens,
                            top_k=request.top_k_logit_lens,
                        )
                    )

        # -- Token rank trajectories   (Task 7) --
        token_rank_trajectories = self._compute_token_rank_trajectories(lens_results, seq_len, batch_size, tokenizer)

        # Build the input/output arrays for the result.
        input_ids_np = input_ids.detach().cpu().numpy().copy()
        input_ids_np.setflags(write=False)
        attention_mask_np = attention_mask.detach().cpu().numpy().copy()
        attention_mask_np.setflags(write=False)

        return TransformerGenerationResult(
            input_ids=input_ids_np,
            attention_mask=attention_mask_np,
            logits=final_logits,
            hidden_states=tuple(captured_states),
            lens_results=tuple(lens_results),
            token_rank_trajectories=tuple(token_rank_trajectories),
            prompt=request.prompt,
            provenance=provenance_str,
        )

    # -- Logit lens implementation   (Task 4) -------------------------------

    def _apply_logit_lens(self, model: Any, hidden_state: Any) -> np.ndarray:
        """Apply the direct logit lens: final LayerNorm + LM head.

        Uses the model's own final normalisation layer and language
        modelling head to project intermediate hidden states into vocabulary
        space. This is a direct (untrained) lens — no learned translator is
        involved.

        Parameters
        ----------
        model:
            The HuggingFace model.
        hidden_state:
            PyTorch tensor of shape ``(batch_size, seq_len, hidden_dim)``.

        Returns
        -------
        np.ndarray
            Logits of shape ``(batch_size, seq_len, vocab_size)``.
        """
        import torch

        with torch.no_grad():
            # Apply final LayerNorm if present (GPT-2 has transformer.ln_f).
            if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                normalized = model.transformer.ln_f(hidden_state)
            else:
                normalized = hidden_state

            # Apply LM head (weight typically tied with word embeddings).
            logits = model.lm_head(normalized)

        logits_np = logits.detach().cpu().numpy().copy()
        logits_np.setflags(write=False)
        return logits_np

    # -- Helper methods -----------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along the specified axis."""
        shifted = logits - np.max(logits, axis=axis, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

    @staticmethod
    def _compute_top_tokens(
        probabilities: np.ndarray,
        top_k: int,
        tokenizer: Any,  # noqa: ARG004
    ) -> list[list[list[tuple[int, float]]]]:
        """Compute top-k token IDs and probabilities for each position."""
        batch_size, seq_len, _ = probabilities.shape
        top_tokens: list[list[list[tuple[int, float]]]] = []

        for b in range(batch_size):
            batch_tokens: list[list[tuple[int, float]]] = []
            for s in range(seq_len):
                pos_probs = probabilities[b, s]
                top_indices = np.argsort(pos_probs)[::-1][:top_k]
                pos_tokens: list[tuple[int, float]] = [(int(idx), float(pos_probs[idx])) for idx in top_indices]
                batch_tokens.append(pos_tokens)
            top_tokens.append(batch_tokens)

        return top_tokens

    @staticmethod
    def _compute_token_rank_trajectories(
        lens_results: list[LogitLensResult],
        seq_len: int,
        _batch_size: int,
        tokenizer: Any,
    ) -> tuple[TokenRankTrajectory, ...]:
        """Compute rank/probability trajectories for the most-probable token
        at each sequence position."""
        trajectories: list[TokenRankTrajectory] = []

        if not lens_results or seq_len < 1:
            return tuple(trajectories)

        final_probs = lens_results[-1].probabilities

        # Track the first batch only for the trajectory analysis.
        for pos in range(min(seq_len, 10)):  # Limit to first 10 positions
            top_token_id = int(np.argmax(final_probs[0, pos]))
            top_token_str = tokenizer.decode([top_token_id])

            ranks: list[int] = []
            probs_list: list[float] = []
            layers_list: list[int] = []

            for lr in lens_results:
                pos_probs = lr.probabilities[0, pos]
                sorted_indices = np.argsort(pos_probs)[::-1]
                matches = np.where(sorted_indices == top_token_id)[0]
                rank = int(matches[0]) + 1 if len(matches) > 0 else len(sorted_indices)
                prob = float(pos_probs[top_token_id])
                ranks.append(rank)
                probs_list.append(prob)
                layers_list.append(lr.layer)

            trajectories.append(
                TokenRankTrajectory(
                    token_id=top_token_id,
                    token_str=top_token_str,
                    ranks=ranks,
                    probabilities=probs_list,
                    layers=layers_list,
                )
            )

        return tuple(trajectories)

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
