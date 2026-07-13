"""Smoke tests for the concrete image ConvVAE adapter."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.adapters.conv_vae import ConvVAE


class _DigitsDataset(Protocol):
    images: np.ndarray


def test_conv_vae_roundtrip_shapes_and_numpy_boundary() -> None:
    adapter = ConvVAE(latent_dim=3)
    images = np.random.default_rng(0).random((4, 1, 8, 8))
    value = adapter.encode_value(images)
    assert value.shape == (4, 3)
    reconstruction = adapter.decode(value.to_numpy())
    assert reconstruction.shape == images.shape
    assert np.all((reconstruction >= 0) & (reconstruction <= 1))


def test_conv_vae_trains_on_real_digits_cpu_smoke_subset() -> None:
    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images[:16] / 16.0).astype(np.float64)[:, None, :, :]
    adapter = ConvVAE(latent_dim=3, n_epochs=1)
    adapter.fit(images)
    assert adapter.metrics_["reconstruction_mse"] >= 0
    assert adapter.metrics_["posterior_kl"] >= 0
