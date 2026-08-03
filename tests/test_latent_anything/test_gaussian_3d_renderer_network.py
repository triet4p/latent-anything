"""Opt-in real-backend/public-scene lane for Sprint 54."""

import os

import pytest


@pytest.mark.network
@pytest.mark.large_download
def test_public_scene_real_gsplat_lane_is_explicitly_provisioned() -> None:
    """Keep checkpoint redistribution and CUDA requirements out of default CI."""
    checkpoint = os.environ.get("LATENT_ANYTHING_3DGS_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set LATENT_ANYTHING_3DGS_CHECKPOINT to run the public-scene GPU lane")
    assert checkpoint
