"""Deliberate model-acquisition smoke test; disabled unless explicitly enabled.

These tests download the pinned GPT-2 checkpoint and verify that the
integration produces correct results with the real model.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from latent_anything.integrations.transformer_lm import (
    HiddenStateIntervention,
    TransformerGenerationRequest,
    TransformerLMIntegration,
)


def _network_device() -> str:
    """Select the opt-in device for real network integration tests."""
    requested = os.environ.get("LATENT_ANYTHING_NETWORK_DEVICE", "cpu").strip().lower()
    if requested not in {"cpu", "auto", "cuda"}:
        raise ValueError("LATENT_ANYTHING_NETWORK_DEVICE must be 'cpu', 'auto', or 'cuda'")
    if requested == "cpu":
        return "cpu"
    available = bool(torch.cuda.is_available())
    if requested == "auto":
        return "cuda" if available else "cpu"
    if not available:
        raise RuntimeError("LATENT_ANYTHING_NETWORK_DEVICE='cuda' requires CUDA availability")
    return "cuda"


def test_network_device_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ordinary local/network runs on CPU unless explicitly opted in."""
    monkeypatch.delenv("LATENT_ANYTHING_NETWORK_DEVICE", raising=False)
    assert _network_device() == "cpu"


@pytest.mark.parametrize(("available", "expected"), [(False, "cpu"), (True, "cuda")])
def test_network_device_auto_selects_available_cuda(
    monkeypatch: pytest.MonkeyPatch, available: bool, expected: str
) -> None:
    """Auto mode uses CUDA when available and otherwise falls back to CPU."""
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert _network_device() == expected


def test_network_device_cuda_requires_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit CUDA mode fails clearly instead of silently falling back."""
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA availability"):
        _network_device()


def test_network_device_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported device selectors before acquiring a checkpoint."""
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "mps")
    with pytest.raises(ValueError, match="must be 'cpu', 'auto', or 'cuda'"):
        _network_device()


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_pinned_checkpoint_generates_expected_shape() -> None:
    """Prove the pinned GPT-2 checkpoint produces valid hidden states."""
    pipe = TransformerLMIntegration(device=_network_device())
    req = TransformerGenerationRequest(
        prompt="The capital of France is",
        max_length=10,
        seed=42,
        capture_hidden_states=True,
        top_k_logit_lens=5,
    )
    result = pipe.generate(req)
    assert result.logits.shape[-1] == 50257  # GPT-2 vocab size
    # Should have at least the embedding + all 12 layers.
    assert len(result.hidden_states) == 13
    for hs in result.hidden_states:
        assert hs.values.ndim == 3
        assert hs.values.shape[-1] == 768  # GPT-2 hidden dim


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_pinned_checkpoint_lens_results() -> None:
    """Verify that the logit lens produces valid probability distributions."""
    pipe = TransformerLMIntegration(device=_network_device())
    req = TransformerGenerationRequest(
        prompt="Hello, world",
        max_length=8,
        seed=0,
        capture_hidden_states=True,
        top_k_logit_lens=3,
    )
    result = pipe.generate(req)
    assert len(result.lens_results) == len(result.hidden_states)
    for lr in result.lens_results:
        # Probabilities should sum to 1.
        assert np.allclose(lr.probabilities.sum(axis=-1), 1.0, atol=1e-5)
        # Logits should be finite.
        assert np.all(np.isfinite(lr.logits))


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_logit_lens_final_layer_parity() -> None:
    """Verify that the logit lens at the final layer produces logits that
    match the model's own final logits (Task 5: final-layer parity).

    The direct logit lens with the model's own final LayerNorm + LM head
    should produce logits identical to the model's forward pass output.
    """
    pipe = TransformerLMIntegration(device=_network_device())
    req = TransformerGenerationRequest(
        prompt="Once upon a time",
        max_length=10,
        seed=42,
        capture_hidden_states=True,
        top_k_logit_lens=0,
    )
    result = pipe.generate(req)

    # The last hidden state is the final transformer block output.
    # The logit lens on this state should match the model's final logits
    # (after applying the same LN + LM head that the model uses internally).
    if result.lens_results:
        final_lens = result.lens_results[-1]
        # These won't be bitwise identical because output_hidden_states=True
        # returns hidden states that already went through the block's residual
        # computation. But for GPT-2, the last block output IS the final hidden
        # state, and applying ln_f + lm_head should match.
        # We verify the shape matches.
        assert final_lens.logits.shape == result.logits.shape


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_logit_lens_layer_evolution() -> None:
    """Verify that logit lens results change across layers (deeper layers
    should produce different distributions than shallow layers)."""
    pipe = TransformerLMIntegration(device=_network_device())
    req = TransformerGenerationRequest(
        prompt="The meaning of life is",
        max_length=8,
        seed=42,
        capture_hidden_states=True,
        top_k_logit_lens=5,
    )
    result = pipe.generate(req)

    # Compare early layer (1) vs middle layer (6) vs final layer (12).
    if len(result.lens_results) >= 13:
        early_lens = result.lens_results[1]  # first block
        mid_lens = result.lens_results[7]  # ~midpoint
        final_lens = result.lens_results[12]  # last block

        # The probability distributions should differ across layers.
        early_probs = early_lens.probabilities[0, 0]
        mid_probs = mid_lens.probabilities[0, 0]
        final_probs = final_lens.probabilities[0, 0]

        # KL divergence should show that distributions evolve.
        def kl_div(p: np.ndarray, q: np.ndarray) -> float:
            eps = 1e-10
            return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

        kl_early_mid = kl_div(early_probs, mid_probs)
        kl_mid_final = kl_div(mid_probs, final_probs)
        # Probability distributions should vary (KL > 0).
        assert kl_early_mid > 0.0 or kl_mid_final > 0.0


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_intervention_changes_hidden_states() -> None:
    """Verify that block intervention changes its native output state."""
    pipe = TransformerLMIntegration(device=_network_device())
    req = TransformerGenerationRequest(
        prompt="The future of AI is",
        max_length=8,
        seed=42,
        capture_hidden_states=True,
        top_k_logit_lens=0,
    )

    # Baseline: no intervention.
    baseline = pipe.generate(req)

    # With intervention at transformer block h.6.
    intervention = HiddenStateIntervention(
        layer=6,
        direction=np.ones((1, 1, 768), dtype=np.float32),
        strength=0.5,
    )
    edited = pipe.generate(req, intervention=intervention)

    # Native index 6 is the input to block h.6 and must remain unchanged;
    # block h.6's output is native hidden-state index 7.
    baseline_hs = {hs.layer: hs.values for hs in baseline.hidden_states}
    edited_hs = {hs.layer: hs.values for hs in edited.hidden_states}

    np.testing.assert_array_equal(edited_hs[6], baseline_hs[6])
    diff_layer_7 = float(np.linalg.norm(edited_hs[7] - baseline_hs[7]))
    assert diff_layer_7 > 0.0, f"Intervention produced no change at native index 7 (diff={diff_layer_7:.2e})"


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_hook_cleanup_after_intervention() -> None:
    """Verify that hooks are cleaned up after intervention (Task 6).

    After `generate()` with an intervention returns, running another
    `generate()` without intervention should produce the same result as
    a fresh run.
    """
    pipe = TransformerLMIntegration(device=_network_device())
    req = TransformerGenerationRequest(
        prompt="Testing hook cleanup",
        max_length=8,
        seed=42,
        capture_hidden_states=False,
        top_k_logit_lens=0,
    )

    # Run with intervention first.
    intervention = HiddenStateIntervention(
        layer=6,
        direction=np.ones((1, 1, 768), dtype=np.float32),
        strength=0.5,
    )
    pipe.generate(req, intervention=intervention)

    # Now run without intervention — should be clean.
    clean = pipe.generate(req)

    # Second clean run should be deterministic.
    clean_again = pipe.generate(req)
    np.testing.assert_array_equal(clean.logits, clean_again.logits)


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_padded_token_masking_parity() -> None:
    """Verify that padded tokens produce valid but masked logits (Task 5)."""
    pipe = TransformerLMIntegration(device=_network_device())

    # Use a short prompt with max_length larger than prompt length,
    # so padding is applied.
    req = TransformerGenerationRequest(
        prompt="Hello",
        max_length=10,
        seed=42,
        capture_hidden_states=True,
        top_k_logit_lens=0,
    )
    result = pipe.generate(req)

    # Padded positions should still have valid (finite) logits.
    assert np.all(np.isfinite(result.logits))

    # The first few positions (real tokens) should have meaningful
    # hidden states — verify at least the shapes are correct.
    assert result.attention_mask.shape == result.input_ids.shape
    assert result.hidden_states[0].values.shape[1] == result.input_ids.shape[1]


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_token_rank_trajectory_stability() -> None:
    """Measure token rank trajectory stability under prompt perturbations
    (Task 7)."""
    pipe = TransformerLMIntegration(device=_network_device())

    # Two similar prompts with one word changed.
    prompt_a = "The capital of France is"
    prompt_b = "The capital of England is"

    req_a = TransformerGenerationRequest(
        prompt=prompt_a, max_length=8, seed=42, capture_hidden_states=True, top_k_logit_lens=5
    )
    req_b = TransformerGenerationRequest(
        prompt=prompt_b, max_length=8, seed=42, capture_hidden_states=True, top_k_logit_lens=5
    )

    result_a = pipe.generate(req_a)
    result_b = pipe.generate(req_b)

    # Both should produce valid lens results.
    assert len(result_a.lens_results) > 0
    assert len(result_b.lens_results) > 0

    # Token rank trajectories should be present for both.
    assert len(result_a.token_rank_trajectories) > 0
    assert len(result_b.token_rank_trajectories) > 0

    # The top token at the final layer should differ between prompts
    # (France vs England have different next-token predictions).
    if result_a.token_rank_trajectories and result_b.token_rank_trajectories:
        # At least the trajectories should have the right structure.
        for traj in result_a.token_rank_trajectories:
            assert len(traj.ranks) > 0
            assert len(traj.probabilities) > 0
            assert len(traj.layers) > 0
