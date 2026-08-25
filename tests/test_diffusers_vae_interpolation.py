"""Regression tests for the local, revision-pinned VAE interpolation lane."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.diffusers_vae_fidelity import MODEL_REVISION, SAFE_WEIGHTS_SHA256
from scripts.diffusers_vae_interpolation import (
    DEFAULT_PNG,
    MIN_ADJACENT_DECODED_L2,
    MIN_ADJACENT_LATENT_L2,
    MIN_ENDPOINT_DECODED_L2,
    MIN_ENDPOINT_LATENT_L2,
    MODEL_ID,
    WEIGHTS,
    run_evidence,
    validate_arrays,
    validate_payload,
)


def test_interpolation_array_gate_rejects_reversed_endpoints_and_collapse() -> None:
    endpoint_latent = np.stack([np.zeros((1, 2, 2), dtype=np.float32), np.ones((1, 2, 2), dtype=np.float32)])
    interpolation_latent = np.stack(
        [(1.0 - weight) * endpoint_latent[0] + weight * endpoint_latent[1] for weight in WEIGHTS]
    )
    endpoint_decoded = endpoint_latent.copy()
    interpolation_decoded = interpolation_latent.copy()
    validate_arrays(WEIGHTS, endpoint_latent, interpolation_latent, endpoint_decoded, interpolation_decoded)
    with pytest.raises(ValueError, match="endpoints"):
        validate_arrays(WEIGHTS, endpoint_latent, interpolation_latent[::-1], endpoint_decoded, interpolation_decoded)
    collapsed = np.zeros_like(interpolation_latent)
    collapsed_endpoints = np.zeros_like(endpoint_latent)
    with pytest.raises(ValueError, match="degenerate"):
        validate_arrays(WEIGHTS, collapsed_endpoints, collapsed, collapsed_endpoints, collapsed)


def _valid_payload(tmp_path: Path) -> tuple[dict[str, object], Path]:
    png_path = tmp_path / DEFAULT_PNG.name
    plt.imsave(png_path, np.zeros((360, 2100, 3), dtype=np.uint8))
    payload: dict[str, object] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license": "mit",
        "weights_sha256": SAFE_WEIGHTS_SHA256,
        "coefficients": [float(value) for value in WEIGHTS],
        "png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
        "png_width": 2100,
        "png_height": 360,
        "network_attempts": [],
        "remote_code": False,
    }
    return payload, png_path


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_revision", "wrong", "provenance"),
        ("weights_sha256", "wrong", "hash"),
        ("coefficients", list(reversed([float(value) for value in WEIGHTS])), "coefficients"),
    ],
)
def test_payload_gate_rejects_provenance_and_order_tampering(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload, png_path = _valid_payload(tmp_path)
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        validate_payload(payload, png_path)


def test_payload_gate_rejects_missing_png(tmp_path: Path) -> None:
    payload, png_path = _valid_payload(tmp_path)
    png_path.unlink()
    with pytest.raises(ValueError, match="PNG"):
        validate_payload(payload, png_path)


@pytest.mark.large_download
@pytest.mark.skipif(
    os.environ.get("LATENT_ANYTHING_RUN_REAL_CHECKPOINT") != "1",
    reason="set LATENT_ANYTHING_RUN_REAL_CHECKPOINT=1 for the cached checkpoint lane",
)
def test_cached_interpolation_artifact_is_reproducible(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    snapshot = root / ".cache" / f"hf-sd-vae-ft-mse-{MODEL_REVISION}"
    if not snapshot.is_dir():
        pytest.skip("verified local checkpoint snapshot is absent")
    first = run_evidence(snapshot, tmp_path / "first.json", tmp_path / "first.png")
    second = run_evidence(snapshot, tmp_path / "second.json", tmp_path / "second.png")
    assert first["deterministic_content_sha256"] == second["deterministic_content_sha256"]
    assert first["png_sha256"] == second["png_sha256"]
    assert first["network_attempts"] == second["network_attempts"] == []
    assert first["movement"]["endpoint_latent_l2"] > MIN_ENDPOINT_LATENT_L2
    assert first["movement"]["min_adjacent_latent_l2"] > MIN_ADJACENT_LATENT_L2
    assert first["movement"]["endpoint_decoded_l2"] > MIN_ENDPOINT_DECODED_L2
    assert first["movement"]["min_adjacent_decoded_l2"] > MIN_ADJACENT_DECODED_L2
    assert first["endpoint_reconstruction"]["max_abs_error"] <= 1e-6
    assert first["png_width"] == 2100
    assert first["png_height"] == 360
    assert json.loads((tmp_path / "first.json").read_text(encoding="utf-8"))["model_revision"] == MODEL_REVISION
