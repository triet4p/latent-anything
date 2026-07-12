"""Deliberate model-acquisition smoke test; disabled unless explicitly enabled."""

from __future__ import annotations

import os

import numpy as np
import pytest

from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter

MODEL_ID = "stabilityai/sd-vae-ft-mse"
MODEL_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"


@pytest.mark.network
@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1",
    reason="set LATENT_ANYTHING_RUN_NETWORK=1 to acquire or validate the pinned checkpoint",
)
def test_pinned_checkpoint_acquires_or_loads_from_cache() -> None:
    """Prove model identity separately from normal offline unit tests."""
    adapter = DiffusersAutoencoderKLAdapter(MODEL_ID, MODEL_REVISION)
    decoded = adapter.decode(adapter.encode(np.zeros((1, 3, 32, 32), dtype=np.float32)))
    assert decoded.shape == (1, 3, 32, 32)
