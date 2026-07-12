"""Offline fidelity tests for the lazy Diffusers AutoencoderKL adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter


class FakeBackend:
    config = SimpleNamespace(scaling_factor=0.5)

    def encode(self, values: torch.Tensor) -> object:
        latent = torch.cat((values, values[:, :1]), dim=1)
        distribution = SimpleNamespace(mean=latent, sample=lambda: latent + 1)
        return SimpleNamespace(latent_dist=distribution)

    def decode(self, latent: torch.Tensor) -> object:
        return SimpleNamespace(sample=latent[:, :3] * 2)


def test_scaling_and_mean_round_trip_match_direct_fake_backend(monkeypatch) -> None:
    adapter = DiffusersAutoencoderKLAdapter("fake/model", "revision", latent_mode="mean")
    monkeypatch.setattr(adapter, "_backend", lambda: FakeBackend())
    images = np.zeros((2, 3, 4, 4), dtype=np.float64)
    latent = adapter.encode(images)
    assert latent.shape == (2, 4, 4, 4)
    np.testing.assert_array_equal(latent, 0)
    np.testing.assert_array_equal(adapter.decode(latent), 0)
    latent_value = adapter.encode_value(images)
    assert latent_value.shape == (2, 4, 4, 4)
    assert latent_value.metadata == {"layout": "NHWC", "scaled": True}


def test_sample_mode_matches_direct_backend_and_validation(monkeypatch) -> None:
    adapter = DiffusersAutoencoderKLAdapter("fake/model", "revision", latent_mode="sample")
    monkeypatch.setattr(adapter, "_backend", lambda: FakeBackend())
    np.testing.assert_array_equal(adapter.encode(np.zeros((1, 3, 2, 2))), 0.5)
    with np.testing.assert_raises(ValueError):
        adapter.encode(np.zeros((1, 3, 4, 4)) + 2)


def test_constructor_rejects_invalid_latent_mode_or_dtype() -> None:
    with np.testing.assert_raises(ValueError):
        DiffusersAutoencoderKLAdapter("fake/model", "revision", latent_mode="invalid")  # type: ignore[arg-type]
    with np.testing.assert_raises(TypeError):
        DiffusersAutoencoderKLAdapter("fake/model", "revision", dtype=np.int64)
