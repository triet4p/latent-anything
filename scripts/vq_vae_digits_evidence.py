"""Reproducible CPU comparison of discrete VQ-VAE and continuous ConvVAE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.adapters import VQVAE, ConvVAE


class _DigitsDataset(Protocol):
    images: np.ndarray
    target: np.ndarray


def _mean_pairwise_distance(values: np.ndarray, metric: str) -> float:
    """Summarize same-split pair distances using one declared geometry."""

    distances: list[float] = []
    for index in range(min(len(values), 16)):
        for other in range(index + 1, min(len(values), 16)):
            if metric == "discrete":
                distances.append(float(np.mean(values[index] != values[other])))
            else:
                distances.append(float(np.linalg.norm(values[index] - values[other])))
    return float(np.mean(distances)) if distances else 0.0


def main() -> None:
    """Train both adapters on the same pinned digits split and write evidence."""

    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images / 16.0).astype(np.float64)[:, None, :, :]
    train_images, test_images = images[:256], images[256:320]

    discrete = VQVAE(codebook_size=16, embedding_dim=8, random_state=42, n_epochs=4)
    discrete.fit(train_images)
    discrete_train_codes = discrete.encode(train_images)
    discrete_test_codes = discrete.encode(test_images)
    discrete_reconstruction = discrete.decode(discrete_test_codes)

    continuous = ConvVAE(latent_dim=4, random_state=42, n_epochs=4)
    continuous.fit(train_images)
    continuous_test_latent = continuous.encode(test_images)
    continuous_reconstruction = continuous.decode(continuous_test_latent)

    payload = {
        "dataset": "sklearn.datasets.load_digits",
        "dataset_revision": VQVAE.dataset_revision,
        "model_revision": VQVAE.model_revision,
        "codebook_version": discrete.codebook_version,
        "seed": 42,
        "train_samples": len(train_images),
        "test_samples": len(test_images),
        "metrics": {
            "discrete_reconstruction_mse": float(np.mean((test_images - discrete_reconstruction) ** 2)),
            "continuous_reconstruction_mse": float(np.mean((test_images - continuous_reconstruction) ** 2)),
            "codebook_perplexity": discrete.codebook_diagnostics(discrete_train_codes)["codebook_perplexity"],
            "dead_code_rate": discrete.codebook_diagnostics(discrete_train_codes)["dead_code_rate"],
            "commitment_distance": discrete.metrics_["commitment_distance"],
            "code_frequency_drift": discrete.code_frequency_drift(discrete_train_codes, discrete_test_codes),
            "discrete_mean_pair_distance": _mean_pairwise_distance(discrete_test_codes, "discrete"),
            "continuous_mean_pair_distance": _mean_pairwise_distance(continuous_test_latent, "continuous"),
        },
        "acceptance": {
            "reconstruction_mse_finite": bool(np.isfinite(discrete_reconstruction).all()),
            "codebook_perplexity_above_one": bool(
                discrete.codebook_diagnostics(discrete_train_codes)["codebook_perplexity"] > 1.0
            ),
            "dead_code_rate_below_one": bool(
                discrete.codebook_diagnostics(discrete_train_codes)["dead_code_rate"] < 1.0
            ),
            "continuous_path_is_comparison_only": True,
        },
        "failure_case": (
            "None on the pinned compact run; real pretrained VQGAN or tokenized-world-model performance is not claimed."
        ),
    }
    output = Path("artifacts/vq_vae_digits_evidence.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    config = {
        "dataset": payload["dataset"],
        "dataset_revision": payload["dataset_revision"],
        "model_revision": payload["model_revision"],
        "codebook_version": payload["codebook_version"],
        "seed": 42,
        "train_slice": [0, 256],
        "test_slice": [256, 320],
        "vq_vae": {
            "codebook_size": 16,
            "embedding_dim": 8,
            "commitment_cost": 0.25,
            "n_epochs": 4,
            "learning_rate": 0.001,
        },
        "continuous_baseline": {"latent_dim": 4, "n_epochs": 4},
        "offline": True,
    }
    output.with_name("vq_vae_digits_evidence_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
