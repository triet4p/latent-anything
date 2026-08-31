"""Offline tests for the decoder-only transformer LM integration.

These tests use either no backend (data-structure tests) or a minimal
FakeBackend that satisfies the protocol expected by
:class:`~latent_anything.integrations.transformer_lm.TransformerLMIntegration`.
No real model downloads occur.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

from latent_anything._transformer_runtime import TransformerRuntimeShapeError
from latent_anything.integrations.transformer_lm import (
    GPT2_HIDDEN_DIM,
    GPT2_NUM_LAYERS,
    GPT2_VOCAB_SIZE,
    TRANSFORMER_MODEL_ID,
    TRANSFORMER_MODEL_REVISION,
    HiddenState,
    HiddenStateIntervention,
    LayerIndex,
    LogitLensResult,
    TokenMask,
    TokenRankTrajectory,
    TransformerGenerationRequest,
    TransformerGenerationResult,
    TransformerInput,
    TransformerLMIntegration,
)

# ---------------------------------------------------------------------------
# Fake backend for offline testing
# ---------------------------------------------------------------------------


class FakeLMHead(nn.Module):
    """Fake LM head that produces vocabulary-sized output."""

    def __init__(self, vocab_size: int = 50257, hidden_dim: int = 768) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_dim))  # noqa: F821  # torch available in test

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: F821
        """Apply the LM head: a simple linear projection."""
        return x @ self.weight.T


class FakeTransformerBlock(nn.Module):
    """Fake transformer block with identity forward."""

    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: object) -> object:
        return x


class TupleTransformerBlock(FakeTransformerBlock):
    """Fake GPT-2 block whose output includes an auxiliary cache value."""

    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__(hidden_dim)
        self.auxiliary = object()

    def forward(self, x: object) -> tuple[object, object]:
        return x, self.auxiliary


class FakeTransformer(nn.Module):
    """Fake transformer with the same module path as GPT-2."""

    def __init__(self, hidden_dim: int = 768, num_layers: int = 12) -> None:
        super().__init__()
        self.wte = nn.Embedding(50257, hidden_dim)
        self.wpe = nn.Embedding(1024, hidden_dim)
        self.h = nn.ModuleList([FakeTransformerBlock(hidden_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)


class TupleFakeTransformer(FakeTransformer):
    def __init__(self, hidden_dim: int = 768, num_layers: int = 12) -> None:
        super().__init__(hidden_dim, num_layers)
        self.h = nn.ModuleList([TupleTransformerBlock(hidden_dim) for _ in range(num_layers)])


class FakeConfig:
    """Fake model configuration with typed attributes."""

    def __init__(self, num_layers: int = 12, hidden_dim: int = 768, vocab_size: int = 50257) -> None:
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_dim
        self.vocab_size = vocab_size
        self.n_positions = 1024
        self.n_embd = hidden_dim
        self.n_layer = num_layers
        self.n_head = 12


class FakeGPT2Model(nn.Module):
    """Fake GPT-2 model that mimics the HuggingFace interface."""

    def __init__(self, hidden_dim: int = 768, num_layers: int = 12, vocab_size: int = 50257) -> None:
        super().__init__()
        self.transformer = FakeTransformer(hidden_dim, num_layers)
        self.lm_head = FakeLMHead(vocab_size, hidden_dim)
        self.config = FakeConfig(num_layers, hidden_dim, vocab_size)

    def forward(  # type: ignore[reportUnknownMemberType]
        self,
        input_ids: object,
        attention_mask: object | None = None,  # noqa: ARG002
        output_hidden_states: bool = False,
    ) -> object:
        batch_size = int(input_ids.shape[0])  # type: ignore[union-attr]
        seq_len = int(input_ids.shape[1])  # type: ignore[union-attr]
        hidden_dim = int(self.transformer.wte.embedding_dim)

        # Generate deterministic fake hidden states.
        rng = np.random.RandomState(42)
        hs_shape = (batch_size, seq_len, hidden_dim)

        # Embedding output (layer 0).
        embedded = torch.tensor(rng.randn(*hs_shape).astype(np.float32))  # type: ignore[reportUnknownArgumentType]

        # Block outputs (layers 1..num_layers).
        num_layers = int(self.config.num_hidden_layers)
        block_outputs = []
        for _ in range(num_layers):
            block_outputs.append(torch.tensor(rng.randn(*hs_shape).astype(np.float32)))  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]

        # Final logits.
        logits_tensor = torch.tensor(rng.randn(batch_size, seq_len, int(self.config.vocab_size)).astype(np.float32))  # type: ignore[reportUnknownArgumentType]

        result_logits = logits_tensor
        result_hs = (embedded, *block_outputs) if output_hidden_states else None  # type: ignore[reportUnknownVariableType]

        # Use SimpleNamespace to avoid Python class-scope closure issues.
        return type("FakeOutput", (), {"logits": result_logits, "hidden_states": result_hs})()


class MismatchedLengthFakeGPT2Model(FakeGPT2Model):
    """Fake backend exposing an invalid shortened full-prompt output."""

    def forward(  # type: ignore[reportUnknownMemberType]
        self,
        input_ids: object,
        attention_mask: object | None = None,
        output_hidden_states: bool = False,
    ) -> object:
        result = cast(Any, super().forward(input_ids, attention_mask, output_hidden_states))
        if result.hidden_states is not None:
            result.hidden_states = tuple(state[:, :-1, :] for state in result.hidden_states)
        return result


class TupleFakeGPT2Model(FakeGPT2Model):
    """Small executable fake that routes structured block output downstream."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = TupleFakeTransformer()

    def forward(  # type: ignore[reportUnknownMemberType]
        self,
        input_ids: object,
        attention_mask: object | None = None,
        output_hidden_states: bool = False,
    ) -> object:
        del attention_mask
        hidden = self.transformer.wte(input_ids)  # type: ignore[arg-type]
        all_hidden: list[torch.Tensor] = [hidden]
        for block in self.transformer.h:
            block_output = block(hidden)
            hidden = block_output[0] if isinstance(block_output, tuple) else block_output  # type: ignore[assignment]
            all_hidden.append(hidden)  # type: ignore[arg-type]
        hidden = self.transformer.ln_f(hidden)  # type: ignore[arg-type]
        logits = self.lm_head(hidden)  # type: ignore[arg-type]
        return type(
            "FakeOutput", (), {"logits": logits, "hidden_states": tuple(all_hidden) if output_hidden_states else None}
        )()


class PostNormFakeConfig:
    """Small config for a fake that mirrors GPT-2 native state semantics."""

    num_hidden_layers = 2
    hidden_size = 4
    vocab_size = 8


class PostNormFakeTransformer(nn.Module):
    """Tiny decoder stack whose terminal native state is post-``ln_f``."""

    def __init__(self) -> None:
        super().__init__()
        self.wte = nn.Embedding(8, 4)
        self.h = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.ln_f = nn.LayerNorm(4, eps=1e-5)


class PostNormFakeGPT2Model(nn.Module):
    """Executable GPT-2-like fake with accurate native hidden-state ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = PostNormFakeTransformer()
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.config = PostNormFakeConfig()

        with torch.no_grad():
            self.transformer.ln_f.weight.copy_(torch.tensor([0.5, 1.0, 2.0, 3.0]))
            self.transformer.ln_f.bias.copy_(torch.tensor([0.2, -0.1, 0.3, -0.4]))

    def forward(  # type: ignore[reportUnknownMemberType]
        self,
        input_ids: object,
        attention_mask: object | None = None,  # noqa: ARG002
        output_hidden_states: bool = False,
    ) -> object:
        hidden = self.transformer.wte(input_ids)  # type: ignore[arg-type]
        all_hidden: list[torch.Tensor] = []
        for block in self.transformer.h:
            all_hidden.append(hidden)
            hidden = block(hidden)
        hidden = self.transformer.ln_f(hidden)
        all_hidden.append(hidden)
        logits = self.lm_head(hidden)
        return type(
            "FakeOutput", (), {"logits": logits, "hidden_states": tuple(all_hidden) if output_hidden_states else None}
        )()


class FakeTokenizer:
    """Fake tokenizer that mimics the HuggingFace tokenizer interface."""

    def __init__(self) -> None:
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token_id = 50256
        self.eos_token_id = 50256

    def __call__(
        self,
        texts: str | list[str],
        *,
        padding: bool = True,  # noqa: ARG002
        truncation: bool = True,  # noqa: ARG002
        max_length: int = 128,  # noqa: ARG002
        return_tensors: str | None = None,  # noqa: ARG002
    ) -> dict[str, object]:
        batch_size = len(texts) if isinstance(texts, list) else 1
        prompts = texts if isinstance(texts, list) else [texts]

        input_ids_list: list[list[int]] = []
        for p in prompts:
            ids = [50256] * max_length
            # Use token length proportional to prompt length, min 5.
            token_len = min(max(5, len(p)), max_length - 1)
            ids[:token_len] = list(range(token_len))
            input_ids_list.append(ids)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),  # noqa: F821
            "attention_mask": torch.ones((batch_size, max_length), dtype=torch.long),  # noqa: F821
        }

    def decode(self, token_ids: list[int]) -> str:
        return f"tok_{token_ids[0]}" if token_ids else ""


class PostNormFakeTokenizer:
    """Small tokenizer whose IDs fit the semantically accurate fake model."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 0

    def __call__(
        self,
        texts: str | list[str],
        *,
        padding: bool = True,  # noqa: ARG002
        truncation: bool = True,  # noqa: ARG002
        max_length: int = 128,
        return_tensors: str | None = None,  # noqa: ARG002
    ) -> dict[str, object]:
        batch_size = len(texts) if isinstance(texts, list) else 1
        ids = torch.arange(max_length, dtype=torch.long).remainder(8).repeat(batch_size, 1)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones((batch_size, max_length), dtype=torch.long),
        }

    def decode(self, token_ids: list[int]) -> str:
        return f"tok_{token_ids[0]}" if token_ids else ""


# ---------------------------------------------------------------------------
# Data structure tests   (Task 2)
# ---------------------------------------------------------------------------


class TestTransformerInput:
    def test_valid_input(self) -> None:
        input_ids = np.zeros((1, 10), dtype=np.int64)
        mask = np.ones((1, 10), dtype=np.int64)
        ti = TransformerInput(input_ids=input_ids, attention_mask=mask)
        assert ti.input_ids.shape == (1, 10)
        assert ti.attention_mask.shape == (1, 10)

    def test_rejects_non_2d_input_ids(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            TransformerInput(
                input_ids=np.zeros((10,), dtype=np.int64),
                attention_mask=np.zeros((1, 10), dtype=np.int64),
            )

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            TransformerInput(
                input_ids=np.zeros((1, 10), dtype=np.int64),
                attention_mask=np.zeros((1, 5), dtype=np.int64),
            )


class TestLayerIndex:
    def test_valid_index(self) -> None:
        li = LayerIndex(layer=5)
        assert li.layer == 5

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="layer must be >= 0"):
            LayerIndex(layer=-1)


class TestTokenMask:
    def test_valid_mask(self) -> None:
        mask = np.ones((1, 10), dtype=bool)
        tm = TokenMask(mask=mask)
        assert tm.mask.shape == (1, 10)

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            TokenMask(mask=np.ones((10,), dtype=bool))


class TestHiddenState:
    def test_valid_hidden_state(self) -> None:
        values = np.zeros((1, 10, 768), dtype=np.float32)
        hs = HiddenState(layer=5, values=values, provenance="gpt2@rev")
        assert hs.layer == 5
        assert hs.values.shape == (1, 10, 768)

    def test_rejects_non_3d(self) -> None:
        with pytest.raises(ValueError, match="must be 3D"):
            HiddenState(layer=0, values=np.zeros((10, 768)))

    def test_rejects_negative_layer(self) -> None:
        with pytest.raises(ValueError, match="layer must be >= 0"):
            HiddenState(layer=-1, values=np.zeros((1, 5, 768)))


class TestLogitLensResult:
    def test_valid_result(self) -> None:
        logits = np.zeros((1, 5, 50257), dtype=np.float32)
        probs = np.zeros((1, 5, 50257), dtype=np.float32)
        lr = LogitLensResult(layer=3, logits=logits, probabilities=probs)
        assert lr.layer == 3
        assert lr.logits.shape == (1, 5, 50257)

    def test_rejects_non_3d_logits(self) -> None:
        with pytest.raises(ValueError, match="must be 3D"):
            LogitLensResult(
                layer=0,
                logits=np.zeros((5, 50257)),
                probabilities=np.zeros((5, 50257)),
            )

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            LogitLensResult(
                layer=0,
                logits=np.zeros((1, 5, 50257)),
                probabilities=np.zeros((1, 3, 50257)),
            )


class TestTokenRankTrajectory:
    def test_valid_trajectory(self) -> None:
        traj = TokenRankTrajectory(
            token_id=42,
            token_str="hello",
            ranks=[5, 3, 1],
            probabilities=[0.01, 0.1, 0.5],
            layers=[0, 4, 8],
        )
        assert traj.token_id == 42
        assert len(traj.ranks) == 3
        assert traj.ranks[-1] == 1


class TestTransformerGenerationRequest:
    def test_defaults_are_valid(self) -> None:
        req = TransformerGenerationRequest(prompt="test prompt")
        assert req.prompt == "test prompt"
        assert req.max_length == 128
        assert req.top_k_logit_lens == 5

    def test_tuple_prompt_is_accepted(self) -> None:
        req = TransformerGenerationRequest(prompt=("a", "b"))
        assert req.prompt == ("a", "b")

    def test_rejects_invalid_max_length(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            TransformerGenerationRequest(prompt="x", max_length=0)


class TestTransformerGenerationResult:
    def test_holds_all_fields(self) -> None:
        result = TransformerGenerationResult(
            input_ids=np.zeros((1, 5), dtype=np.int64),
            attention_mask=np.ones((1, 5), dtype=np.int64),
            logits=np.zeros((1, 5, 50257), dtype=np.float32),
            hidden_states=(),
            lens_results=(),
            token_rank_trajectories=(),
            prompt="test",
        )
        assert result.logits.shape == (1, 5, 50257)
        assert result.input_ids.shape == (1, 5)


class TestHiddenStateIntervention:
    def test_valid_intervention(self) -> None:
        direction = np.zeros((1, 1, 768), dtype=np.float32)
        direction.setflags(write=False)
        intervention = HiddenStateIntervention(layer=6, direction=direction, strength=1.0)
        assert intervention.layer == 6
        assert intervention.strength == 1.0

    def test_rejects_non_3d_direction(self) -> None:
        with pytest.raises(ValueError, match="must be 3D"):
            HiddenStateIntervention(
                layer=0,
                direction=np.zeros((768,)),
                strength=1.0,
            )

    def test_rejects_negative_strength(self) -> None:
        with pytest.raises(ValueError, match="strength must be >= 0"):
            HiddenStateIntervention(
                layer=0,
                direction=np.zeros((1, 1, 768)),
                strength=-1.0,
            )

    def test_zero_strength_is_acceptable(self) -> None:
        intervention = HiddenStateIntervention(
            layer=0,
            direction=np.zeros((1, 1, 768)),
            strength=0.0,
        )
        assert intervention.strength == 0.0


# ---------------------------------------------------------------------------
# Pipeline construction & descriptor tests   (Task 3)
# ---------------------------------------------------------------------------


class TestTransformerLMIntegration:
    def test_public_api_and_result_schema_snapshot(self) -> None:
        assert tuple(inspect.signature(TransformerLMIntegration.generate).parameters) == (
            "self",
            "request",
            "intervention",
        )
        assert tuple(inspect.signature(TransformerLMIntegration.tokenize).parameters) == (
            "self",
            "prompt",
            "max_length",
            "return_tensors",
        )
        assert TransformerGenerationRequest.__module__ == "latent_anything.integrations.transformer_lm"
        assert TransformerGenerationResult.__module__ == "latent_anything.integrations.transformer_lm"
        assert tuple(TransformerGenerationRequest.__dataclass_fields__) == (
            "prompt",
            "max_length",
            "seed",
            "capture_hidden_states",
            "capture_layers",
            "top_k_logit_lens",
        )
        assert tuple(HiddenStateIntervention.__dataclass_fields__) == (
            "layer",
            "direction",
            "strength",
            "token_indices",
        )

    def test_constructor_with_defaults(self) -> None:
        pipe = TransformerLMIntegration()
        assert pipe.model_id == TRANSFORMER_MODEL_ID
        assert pipe.revision == TRANSFORMER_MODEL_REVISION

    def test_default_model_identity_is_immutable_and_canonical(self) -> None:
        assert TRANSFORMER_MODEL_ID == "openai-community/gpt2"
        assert TRANSFORMER_MODEL_REVISION == "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
        assert len(TRANSFORMER_MODEL_REVISION) == 40

    def test_provenance_format(self) -> None:
        pipe = TransformerLMIntegration()
        assert "@" in pipe.provenance
        assert TRANSFORMER_MODEL_ID in pipe.provenance

    def test_hidden_state_space_has_correct_role(self) -> None:
        pipe = TransformerLMIntegration()
        space = pipe.hidden_state_space
        assert space.dim == GPT2_HIDDEN_DIM
        assert space.metadata.get("role") == "transformer_hidden_state"

    def test_logit_space_has_correct_dim(self) -> None:
        pipe = TransformerLMIntegration()
        space = pipe.logit_space
        assert space.dim == GPT2_VOCAB_SIZE
        assert space.metadata.get("role") == "logits"

    def test_random_intervention_direction(self) -> None:
        intervention = TransformerLMIntegration.random_intervention_direction(
            hidden_dim=768, seed=42, strength=2.0, layer=4
        )
        assert intervention.direction.shape == (1, 1, 768)
        assert intervention.strength == 2.0
        assert intervention.layer == 4

    def test_random_intervention_is_deterministic(self) -> None:
        a = TransformerLMIntegration.random_intervention_direction(hidden_dim=768, seed=99)
        b = TransformerLMIntegration.random_intervention_direction(hidden_dim=768, seed=99)
        np.testing.assert_array_equal(a.direction, b.direction)


# ---------------------------------------------------------------------------
# FakeBackend generation tests   (Tasks 3, 4, 5)
# ---------------------------------------------------------------------------


class TestFakeBackendPipeline:
    def test_generate_rejects_shortened_full_prompt_outputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe = TransformerLMIntegration()
        fake_model = MismatchedLengthFakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        with pytest.raises(TransformerRuntimeShapeError, match=r"hidden_states\[0\]") as captured:
            pipe.generate(TransformerGenerationRequest(prompt="shape-secret", max_length=22, top_k_logit_lens=0))
        assert "shape-secret" not in str(captured.value)

    def test_generate_with_no_capture_returns_valid_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe = TransformerLMIntegration()

        # Monkey-patch the backend to avoid real model download.
        fake_model = FakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        req = TransformerGenerationRequest(
            prompt="hello",
            max_length=10,
            capture_hidden_states=False,
            capture_layers=(),
            top_k_logit_lens=0,
        )
        result = pipe.generate(req)
        assert isinstance(result, TransformerGenerationResult)
        assert result.logits.shape[0] == 1  # batch size
        assert len(result.hidden_states) == 0
        assert len(result.lens_results) == 0

    def test_generate_captures_hidden_states(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe = TransformerLMIntegration()
        fake_model = FakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        req = TransformerGenerationRequest(
            prompt="test",
            max_length=8,
            capture_hidden_states=True,
            top_k_logit_lens=0,
        )
        result = pipe.generate(req)
        assert isinstance(result, TransformerGenerationResult)
        # Should have layer 0 (embedding) + 12 block outputs = 13 hidden states
        assert len(result.hidden_states) == GPT2_NUM_LAYERS + 1
        for hs in result.hidden_states:
            assert hs.values.ndim == 3
            assert hs.layer >= 0

    def test_generate_with_lens_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe = TransformerLMIntegration()
        fake_model = FakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        req = TransformerGenerationRequest(
            prompt="test",
            max_length=8,
            capture_hidden_states=True,
            top_k_logit_lens=0,
        )
        result = pipe.generate(req)
        assert isinstance(result, TransformerGenerationResult)
        # Lens results should match hidden states count.
        assert len(result.lens_results) == len(result.hidden_states)
        for lr in result.lens_results:
            assert lr.logits.ndim == 3
            assert lr.logits.shape[-1] == GPT2_VOCAB_SIZE
            # Probabilities should sum to 1 over vocab.
            assert np.allclose(lr.probabilities.sum(axis=-1), 1.0)

    def test_generate_with_top_k_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe = TransformerLMIntegration()
        fake_model = FakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        req = TransformerGenerationRequest(
            prompt="test",
            max_length=8,
            capture_hidden_states=True,
            top_k_logit_lens=3,
        )
        result = pipe.generate(req)
        assert isinstance(result, TransformerGenerationResult)
        # At least the first lens result should have top_tokens.
        if result.lens_results:
            lr = result.lens_results[0]
            assert lr.top_k == 3
            # Check shape: (batch, seq, top_k)
            assert len(lr.top_tokens) == 1  # batch=1
            if len(lr.top_tokens[0]) > 0:
                assert len(lr.top_tokens[0][0]) == 3  # top_k=3

    def test_generate_with_intervention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe = TransformerLMIntegration()
        fake_model = FakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        req = TransformerGenerationRequest(
            prompt="test",
            max_length=8,
            capture_hidden_states=True,
            top_k_logit_lens=0,
        )
        intervention = HiddenStateIntervention(
            layer=6,
            direction=np.zeros((1, 1, GPT2_HIDDEN_DIM), dtype=np.float32),
            strength=0.5,
        )
        result = pipe.generate(req, intervention=intervention)
        assert isinstance(result, TransformerGenerationResult)
        # Intervention with zero direction should not affect shapes.
        assert len(result.hidden_states) == GPT2_NUM_LAYERS + 1

    def test_intervention_hooks_are_removed_when_forward_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed model call must not leave intervention hooks installed."""
        pipe = TransformerLMIntegration()
        fake_model = FakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        def fail_forward(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("fake forward failure")

        monkeypatch.setattr(fake_model, "forward", fail_forward)
        request = TransformerGenerationRequest(prompt="test", max_length=8, capture_hidden_states=False)
        intervention = HiddenStateIntervention(
            layer=6,
            direction=np.zeros((1, 1, GPT2_HIDDEN_DIM), dtype=np.float32),
            strength=0.5,
        )

        with pytest.raises(RuntimeError, match="fake forward failure"):
            pipe.generate(request, intervention=intervention)

        assert all(not getattr(module, "_forward_hooks", {}) for module in fake_model.modules())

    def test_generate_intervention_preserves_tuple_block_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe = TransformerLMIntegration()
        fake_model = TupleFakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))
        request = TransformerGenerationRequest(prompt="test", max_length=8, capture_hidden_states=True)

        baseline = pipe.generate(request)
        identity = pipe.generate(
            request,
            intervention=HiddenStateIntervention(
                layer=6,
                direction=np.ones((1, 1, GPT2_HIDDEN_DIM), dtype=np.float32),
                strength=0.0,
            ),
        )
        intervention = HiddenStateIntervention(
            layer=6,
            direction=np.ones((1, 1, GPT2_HIDDEN_DIM), dtype=np.float32),
            strength=1.0,
        )
        changed = pipe.generate(request, intervention=intervention)

        assert len(changed.hidden_states) == GPT2_NUM_LAYERS + 1
        assert all(
            np.array_equal(expected.values, actual.values)
            for expected, actual in zip(baseline.hidden_states, identity.hidden_states, strict=True)
        )
        assert not np.array_equal(baseline.hidden_states[7].values, changed.hidden_states[7].values)
        assert all(not getattr(module, "_forward_hooks", {}) for module in fake_model.modules())

    def test_token_rank_trajectories(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe = TransformerLMIntegration()
        fake_model = FakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        req = TransformerGenerationRequest(
            prompt="test",
            max_length=8,
            capture_hidden_states=True,
            top_k_logit_lens=0,
        )
        result = pipe.generate(req)
        assert isinstance(result, TransformerGenerationResult)
        # Token rank trajectories should be computed.
        assert len(result.token_rank_trajectories) > 0
        for traj in result.token_rank_trajectories:
            assert len(traj.ranks) == len(traj.layers)
            assert len(traj.probabilities) == len(traj.layers)

    def test_fake_backend_preserves_native_layer_index_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify native layer IDs remain ordered on the legacy fake backend."""
        pipe = TransformerLMIntegration()
        fake_model = FakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        req = TransformerGenerationRequest(
            prompt="test",
            max_length=8,
            capture_hidden_states=True,
            top_k_logit_lens=0,
        )
        result = pipe.generate(req)

        # The fake model generates random logits, so exact match is not
        # expected — but the final lens result layer should match the last layer.
        if result.lens_results and result.hidden_states:
            final_lens_layer = result.lens_results[-1].layer
            last_hs_layer = result.hidden_states[-1].layer
            assert final_lens_layer == last_hs_layer

    def test_lens_applies_final_norm_once_and_preserves_capture_subsets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Match every lens to a GPT-2-like fake's native states and logits."""
        pipe = TransformerLMIntegration()
        fake_model = PostNormFakeGPT2Model()
        fake_tokenizer = PostNormFakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))
        request = TransformerGenerationRequest(
            prompt="test",
            max_length=4,
            capture_hidden_states=True,
            capture_layers=(0, 1, 2),
            top_k_logit_lens=0,
        )

        result = pipe.generate(request)
        tokenized = fake_tokenizer(["test"], max_length=4, return_tensors="pt")
        direct: Any = fake_model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            output_hidden_states=True,
        )

        assert len(result.hidden_states) == 3
        assert len(result.lens_results) == 3
        native_states = direct.hidden_states
        assert native_states is not None
        for index, lens_result in enumerate(result.lens_results):
            state = native_states[index]
            expected_input = fake_model.transformer.ln_f(state) if index < 2 else state
            expected_logits = fake_model.lm_head(expected_input).detach().numpy()
            np.testing.assert_array_equal(lens_result.logits, expected_logits)

        np.testing.assert_array_equal(result.lens_results[-1].logits, result.logits)
        shifted = result.logits - np.max(result.logits, axis=-1, keepdims=True)
        expected_probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
        np.testing.assert_allclose(result.lens_results[-1].probabilities, expected_probabilities)
        for position, trajectory in enumerate(result.token_rank_trajectories):
            assert trajectory.layers[-1] == 2
            assert (
                trajectory.probabilities[-1] == result.lens_results[-1].probabilities[0, position, trajectory.token_id]
            )

        for capture_layers in ((2,), (0, 2)):
            subset = pipe.generate(
                TransformerGenerationRequest(
                    prompt="test",
                    max_length=4,
                    capture_hidden_states=False,
                    capture_layers=capture_layers,
                    top_k_logit_lens=0,
                )
            )
            assert tuple(state.layer for state in subset.hidden_states) == capture_layers
            assert tuple(lens.layer for lens in subset.lens_results) == capture_layers
            np.testing.assert_array_equal(subset.lens_results[-1].logits, subset.logits)

    def test_set_seed_determinism(self) -> None:
        """Verify that the integration produces deterministic results."""
        # This is a basic test that random_intervention_direction is deterministic.
        a = TransformerLMIntegration.random_intervention_direction(hidden_dim=768, seed=42)
        b = TransformerLMIntegration.random_intervention_direction(hidden_dim=768, seed=42)
        np.testing.assert_array_equal(a.direction, b.direction)

    def test_specific_capture_layers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that specific layer indices are captured correctly."""
        pipe = TransformerLMIntegration()
        fake_model = FakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        req = TransformerGenerationRequest(
            prompt="test",
            max_length=8,
            capture_hidden_states=False,
            capture_layers=(0, 6, 12),
            top_k_logit_lens=0,
        )
        result = pipe.generate(req)
        assert len(result.hidden_states) == 3
        assert result.hidden_states[0].layer == 0
        assert result.hidden_states[1].layer == 6
        assert result.hidden_states[2].layer == 12

    def test_decode_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipe = TransformerLMIntegration()
        fake_model = FakeGPT2Model()
        fake_tokenizer = FakeTokenizer()
        monkeypatch.setattr(pipe, "_backend", lambda: (fake_model, fake_tokenizer, fake_model.config))

        tokens = pipe.decode_tokens(np.array([[42]], dtype=np.int64))
        assert isinstance(tokens, list)
        assert len(tokens) > 0
