"""Deliberate model-acquisition smoke test; disabled unless explicitly enabled."""

from __future__ import annotations

import os

import numpy as np
import pytest

from latent_anything.integrations.diffusers_conditional import (
    DiffusersConditionalPipeline,
    GenerationRequest,
)


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_pinned_checkpoint_generates_expected_shape() -> None:
    """Prove the pinned SD 1.5 checkpoint produces valid 512x512 output with scheduler state capture."""
    pipe = DiffusersConditionalPipeline(device="cpu")
    req = GenerationRequest(
        prompt="a photograph of an astronaut riding a horse",
        num_inference_steps=5,  # minimal steps for smoke test
        seed=42,
        capture_scheduler_states=True,
        capture_denoiser_location=None,
    )
    result = pipe.generate(req)
    assert result.images.shape == (1, 512, 512, 3)
    # With 5 steps we expect at least some scheduler states captured.
    assert len(result.scheduler_states) > 0
    # Verify the final latent exists.
    assert result.final_vae_latent.shape == (1, 4, 64, 64)
    # Verify all pixel values are in [0, 1].
    assert result.images.min() >= 0.0
    assert result.images.max() <= 1.0


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_pinned_checkpoint_captures_denoiser_activations() -> None:
    """Verify that denoiser activation capture works with the real pipeline."""
    pipe = DiffusersConditionalPipeline(device="cpu")
    req = GenerationRequest(
        prompt="a cat",
        num_inference_steps=3,
        seed=0,
        capture_scheduler_states=True,
        capture_denoiser_location="mid_block",
    )
    result = pipe.generate(req)
    assert len(result.denoiser_captures) > 0
    for cap in result.denoiser_captures:
        assert cap.location == "mid_block"
        # SD 1.5 mid_block output has 1280 channels at 512x512.
        assert cap.values.shape[1] == 1280


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_scheduler_determinism_same_seed() -> None:
    """Same seed + same prompt should produce identical outputs."""
    pipe = DiffusersConditionalPipeline(device="cpu")
    req = GenerationRequest(
        prompt="test", num_inference_steps=5, seed=42, capture_scheduler_states=False, capture_denoiser_location=None
    )
    r1 = pipe.generate(req)
    r2 = pipe.generate(req)
    np.testing.assert_array_equal(r1.images, r2.images)
