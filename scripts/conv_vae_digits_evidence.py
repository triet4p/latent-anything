"""Reproducible CPU evidence path for ConvVAE on sklearn digits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from latent_anything.adapters.conv_vae import ConvVAE
from latent_anything.methods import PCA, SAE, SteeringVector


def main() -> None:
    """Train a compact ConvVAE and record metrics plus an explicit failure caveat."""

    digits = load_digits()
    images = (digits.images[:32] / 16.0).astype(np.float64)[:, None, :, :]
    labels = digits.target[:32]
    adapter = ConvVAE(latent_dim=3, random_state=42, n_epochs=3)
    adapter.fit(images)
    latent = adapter.encode_value(images).to_numpy()
    pca = PCA(n_components=2)
    pca_projection = pca.fit_transform(latent)
    sae = SAE(n_components=3, n_epochs=2, random_state=42)
    sae.fit(latent)
    sae_projection = sae.transform(latent)
    steering = SteeringVector()
    steering.fit(latent[labels % 2 == 0], latent[labels % 2 == 1])
    reconstructed = adapter.decode(latent)
    payload = {
        "dataset": "sklearn-digits-8x8",
        "seed": 42,
        "samples": len(images),
        "metrics": adapter.metrics_,
        "pca_shape": list(pca_projection.shape),
        "sae_shape": list(sae_projection.shape),
        "steering_norm": float(np.linalg.norm(steering.direction)),
        "reconstruction_mae": float(np.mean(np.abs(images - reconstructed))),
        "failure_case": (
            "Three-epoch CPU smoke training is not a quality claim; use the full benchmark before D2 promotion."
        ),
    }
    output = Path("artifacts/conv_vae_digits_evidence.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(2, 4, figsize=(8, 4))
    for index, axis in enumerate(axes[0]):
        axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    for index, axis in enumerate(axes[1]):
        axis.imshow(reconstructed[index, 0], cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    figure.suptitle("ConvVAE digits: source (top), reconstruction (bottom)")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    main()
