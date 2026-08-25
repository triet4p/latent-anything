"""Offline fidelity tests for the lazy Diffusers AutoencoderKL adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter


class FakeBackend:
    config = SimpleNamespace(scaling_factor=0.5, latent_channels=4)

    def encode(self, values: torch.Tensor) -> object:
        latent = torch.cat((values, values[:, :1]), dim=1)
        distribution = SimpleNamespace(mean=latent, sample=lambda: latent + 1)
        return SimpleNamespace(latent_dist=distribution)

    def decode(self, latent: torch.Tensor) -> object:
        return SimpleNamespace(sample=latent[:, :3] * 2)


def test_scaling_and_mean_round_trip_match_direct_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_sample_mode_matches_direct_backend_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
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
    with np.testing.assert_raises(TypeError):
        DiffusersAutoencoderKLAdapter("fake/model", "revision", dtype=np.float64)


def test_backend_sets_eval_mode_after_device_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _Model:
        config = SimpleNamespace(scaling_factor=1.0, latent_channels=4)

        def to(self, **_: object) -> _Model:
            calls.append("to")
            return self

        def eval(self) -> _Model:
            calls.append("eval")
            return self

    class _Autoencoder:
        @staticmethod
        def from_pretrained(model_id: str, revision: str) -> _Model:
            assert model_id == "fake/model"
            assert revision == "revision"
            calls.append("load")
            return _Model()

    monkeypatch.setattr(
        "latent_anything.integrations.diffusers_vae.require_optional",
        lambda _name, **_kwargs: SimpleNamespace(AutoencoderKL=_Autoencoder),
    )
    adapter = DiffusersAutoencoderKLAdapter("fake/model", "revision")
    assert adapter._backend() is adapter._backend()  # pyright: ignore[reportPrivateUsage]
    assert calls == ["load", "to", "eval"]


def test_sample_seed_uses_local_generator_without_global_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Distribution:
        def __init__(self, latent: torch.Tensor) -> None:
            self.mean = latent

        def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
            assert generator is not None
            return torch.rand((1, 4, 2, 2), generator=generator)

    class _SeededBackend(FakeBackend):
        def encode(self, values: torch.Tensor) -> object:
            del values
            return SimpleNamespace(latent_dist=_Distribution(torch.zeros((1, 4, 2, 2))))

    adapter = DiffusersAutoencoderKLAdapter("fake/model", "revision", latent_mode="sample")
    monkeypatch.setattr(adapter, "_backend", lambda: _SeededBackend())
    first = adapter.encode(np.zeros((1, 3, 2, 2), dtype=np.float32), seed=123)
    second = adapter.encode(np.zeros((1, 3, 2, 2), dtype=np.float32), seed=123)
    third = adapter.encode(np.zeros((1, 3, 2, 2), dtype=np.float32), seed=124)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)
