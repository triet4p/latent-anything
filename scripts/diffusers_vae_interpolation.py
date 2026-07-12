"""Produce a pinned Diffusers VAE interpolation artifact from sklearn digits.

Run with ``uv run --extra diffusers python scripts/diffusers_vae_interpolation.py``.
The first run deliberately downloads the pinned model; normal tests never do.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter

MODEL_ID = "stabilityai/sd-vae-ft-mse"
MODEL_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
ARTIFACT_DIR = Path("artifacts")


def main() -> None:
    digits = load_digits()
    images = np.stack([digits.images[0], digits.images[1]])
    nchw = np.repeat(np.kron(images, np.ones((4, 4)))[:, None, :, :], 3, axis=1).astype(np.float32)
    nchw = nchw / 8.0 - 1.0
    adapter = DiffusersAutoencoderKLAdapter(MODEL_ID, MODEL_REVISION)
    latent = adapter.encode(nchw)
    weights = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    interpolation = np.stack([(1 - weight) * latent[0] + weight * latent[1] for weight in weights])
    decoded = adapter.decode(interpolation)
    norms = np.linalg.norm(interpolation.reshape(len(weights), -1), axis=1)
    density = np.count_nonzero(np.abs(interpolation) < 0.05, axis=(1, 2, 3)) / interpolation[0].size

    ARTIFACT_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, len(weights), figsize=(14, 2.4))
    for axis, image, weight in zip(axes, decoded, weights, strict=True):
        axis.imshow(np.moveaxis((image + 1.0) / 2.0, 0, -1).clip(0, 1))
        axis.set_title(f"t={weight:.2f}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "diffusers_vae_digits_interpolation.png", dpi=150)
    plt.close(figure)
    payload = {
        "evidence_level": "D1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "input_layout": "NCHW",
        "input_range": "[-1, 1]",
        "weights": weights.tolist(),
        "latent_norms": norms.tolist(),
        "near_zero_density": density.tolist(),
    }
    (ARTIFACT_DIR / "diffusers_vae_digits_interpolation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
