"""Measure the ConvVAE integration on a deterministic held-out digits split."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.adapters.conv_vae import ConvVAE
from latent_anything.methods import PCA, SAE, SteeringVector

SEED = 42
TRAIN_FRACTION = 0.8
LATENT_DIM = 4
EPOCHS = 8
MIN_ZERO_BASELINE_IMPROVEMENT = 0.10
MIN_LATENT_UTILIZATION = 1e-3
MIN_STEERING_DECODE_DELTA = 1e-6
RUNTIME_BUDGET_SECONDS = 30.0


class _DigitsDataset(Protocol):
    images: np.ndarray
    target: np.ndarray


def split_digits(images: np.ndarray, labels: np.ndarray, *, seed: int = SEED) -> tuple[np.ndarray, ...]:
    """Return disjoint deterministic train/test arrays and source indices."""

    if images.ndim != 4 or images.shape[1:] != (1, 8, 8):
        raise ValueError(f"images must have shape (n, 1, 8, 8), got {images.shape}")
    if labels.ndim != 1 or len(images) != len(labels):
        raise ValueError("labels must be a matching one-dimensional array")
    permutation = np.random.default_rng(seed).permutation(len(images))
    split = int(len(images) * TRAIN_FRACTION)
    train_indices, test_indices = permutation[:split], permutation[split:]
    return (
        images[train_indices],
        images[test_indices],
        labels[train_indices],
        labels[test_indices],
        train_indices,
        test_indices,
    )


def _index_digest(indices: np.ndarray) -> str:
    """Hash split indices so the exact partition is reproducible."""

    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def main(output_dir: Path = Path("artifacts")) -> dict[str, object]:
    """Run the bounded CPU benchmark and write its JSON/config artifacts."""

    started = time.perf_counter()
    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images / 16.0).astype(np.float64)[:, None, :, :]
    labels = digits.target.astype(np.int64)
    train, heldout, train_labels, heldout_labels, train_indices, test_indices = split_digits(images, labels)
    if np.intersect1d(train_indices, test_indices).size != 0:
        raise RuntimeError("train and held-out indices overlap")

    adapter = ConvVAE(latent_dim=LATENT_DIM, random_state=SEED, n_epochs=EPOCHS)
    adapter.fit(train)
    train_latents = adapter.encode(train)
    heldout_latents = adapter.encode(heldout)
    heldout_reconstruction = adapter.decode(heldout_latents)
    zero_baseline = np.zeros_like(heldout)
    train_mean_baseline = np.broadcast_to(np.mean(train, axis=0, keepdims=True), heldout.shape)
    heldout_mse = float(np.mean((heldout - heldout_reconstruction) ** 2))
    zero_mse = float(np.mean((heldout - zero_baseline) ** 2))
    train_mean_mse = float(np.mean((heldout - train_mean_baseline) ** 2))
    zero_improvement = float(1.0 - heldout_mse / zero_mse)

    pca = PCA(n_components=2)
    pca.fit(train_latents)
    heldout_pca = pca.transform(heldout_latents)
    sae = SAE(n_components=3, n_epochs=25, random_state=SEED)
    sae.fit(train_latents)
    heldout_sae = sae.transform(heldout_latents)
    steering = SteeringVector()
    steering.fit(train_latents[train_labels >= 5], train_latents[train_labels < 5])
    steered_latents = np.stack([steering(value, strength=0.25) for value in heldout_latents])
    steered_decoded = adapter.decode(steered_latents)
    steering_decode_delta = float(np.mean(np.abs(steered_decoded - heldout_reconstruction)))
    runtime_seconds = time.perf_counter() - started
    latent_utilization = float(adapter.metrics_["latent_utilization"])
    steering_norm = float(np.linalg.norm(steering.direction))

    acceptance = {
        "finite_heldout_metrics": bool(
            np.isfinite([heldout_mse, zero_mse, train_mean_mse, zero_improvement, latent_utilization]).all()
        ),
        "beats_zero_baseline_by_margin": zero_improvement >= MIN_ZERO_BASELINE_IMPROVEMENT,
        "latent_utilization_non_degenerate": latent_utilization >= MIN_LATENT_UTILIZATION,
        "pca_shape_and_finiteness": heldout_pca.shape == (len(heldout), 2) and bool(np.isfinite(heldout_pca).all()),
        "sae_shape_and_finiteness": heldout_sae.shape == (len(heldout), 3) and bool(np.isfinite(heldout_sae).all()),
        "steering_shape_norm_and_effect": (
            steered_latents.shape == heldout_latents.shape
            and abs(steering_norm - 1.0) <= 1e-12
            and steering_decode_delta >= MIN_STEERING_DECODE_DELTA
            and bool(np.isfinite(steered_decoded).all())
        ),
        "split_is_disjoint": bool(np.intersect1d(train_indices, test_indices).size == 0),
    }
    accepted = all(acceptance.values())
    payload: dict[str, object] = {
        "evidence_level": "D2" if accepted else "D1",
        "dataset": "sklearn.datasets.load_digits",
        "dataset_revision": f"scikit-learn=={package_version('scikit-learn')}",
        "dataset_license": "BSD-3-Clause (scikit-learn bundled digits dataset)",
        "seed": SEED,
        "split": {
            "algorithm": "default_rng(seed).permutation; first 80% train, remaining 20% held-out",
            "train_samples": len(train),
            "heldout_samples": len(heldout),
            "train_index_digest": _index_digest(train_indices),
            "heldout_index_digest": _index_digest(test_indices),
            "train_indices": train_indices.tolist(),
            "heldout_indices": test_indices.tolist(),
            "heldout_label_counts": np.bincount(heldout_labels, minlength=10).tolist(),
        },
        "model": {
            "adapter": "ConvVAE",
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "fit_partition": "train_only",
            "torch_version": torch.__version__,
            "device": "cpu",
        },
        "metrics": {
            "heldout_reconstruction_mse": heldout_mse,
            "zero_baseline_mse": zero_mse,
            "train_mean_baseline_mse": train_mean_mse,
            "improvement_over_zero_baseline": zero_improvement,
            "posterior_kl_train": adapter.metrics_["posterior_kl"],
            "latent_utilization_train": latent_utilization,
            "steering_direction_norm": steering_norm,
            "steering_decode_mean_absolute_delta": steering_decode_delta,
        },
        "composition": {
            "train_latent_shape": list(train_latents.shape),
            "heldout_latent_shape": list(heldout_latents.shape),
            "heldout_pca_shape": list(heldout_pca.shape),
            "heldout_sae_shape": list(heldout_sae.shape),
            "steering_factor": "digit >= 5 versus digit < 5; fit on train, apply to held-out",
        },
        "thresholds": {
            "min_zero_baseline_improvement": MIN_ZERO_BASELINE_IMPROVEMENT,
            "min_latent_utilization": MIN_LATENT_UTILIZATION,
            "min_steering_decode_delta": MIN_STEERING_DECODE_DELTA,
            "runtime_budget_seconds_advisory": RUNTIME_BUDGET_SECONDS,
        },
        "acceptance": acceptance,
        "accepted": accepted,
        "runtime_seconds": runtime_seconds,
        "runtime_within_budget": runtime_seconds <= RUNTIME_BUDGET_SECONDS,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "limitations": [
            "The train-pixel-mean baseline is stronger than the all-zero acceptance baseline and remains a diagnostic.",
            "This is a compact CPU ConvVAE benchmark, not a claim about a pretrained generative checkpoint.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conv_vae_heldout_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    config = {
        "dataset": payload["dataset"],
        "dataset_revision": payload["dataset_revision"],
        "dataset_license": payload["dataset_license"],
        "seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "latent_dim": LATENT_DIM,
        "epochs": EPOCHS,
        "thresholds": payload["thresholds"],
    }
    (output_dir / "conv_vae_heldout_benchmark_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    if not accepted:
        raise RuntimeError(f"ConvVAE held-out acceptance failed: {acceptance}")
    return payload


if __name__ == "__main__":
    main()
