"""Focused contract tests for the compact VQ-VAE adapter."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything.adapters import VQVAE, DecodableAdapter, ModelAdapter
from latent_anything.latent_value import LatentValue


def _images(n: int = 12) -> np.ndarray:
    """Return deterministic image-shaped data without a dataset download."""

    values = np.linspace(0.0, 1.0, n * 64, dtype=np.float64)
    return values.reshape(n, 1, 8, 8)


@pytest.fixture
def adapter() -> VQVAE:
    """Use a tiny one-epoch model for fast offline contract tests."""

    model = VQVAE(codebook_size=8, embedding_dim=4, random_state=7, n_epochs=1)
    model.fit(_images())
    return model


def test_vq_vae_conforms_to_model_and_decodable_adapter(adapter: VQVAE) -> None:
    """The image adapter has a decoder but is not a flat-batch adapter."""

    assert isinstance(adapter, ModelAdapter)
    assert isinstance(adapter, DecodableAdapter)
    assert adapter.latent_space.geometry == "discrete_code"


def test_encode_preserves_integer_code_sequence(adapter: VQVAE) -> None:
    """Encoding returns categorical IDs and does not expose embeddings implicitly."""

    codes = adapter.encode(_images(3))
    assert codes.shape == (3, 16)
    assert np.issubdtype(codes.dtype, np.integer)
    assert np.all((codes >= 0) & (codes < adapter.codebook_size))
    value = adapter.encode_value(_images(3))
    assert isinstance(value, LatentValue)
    assert value.to_numpy().dtype == np.int64
    assert value.space.metadata["representation"] == "integer_code_sequence"


def test_code_embeddings_are_an_explicit_conversion(adapter: VQVAE) -> None:
    """Continuous codebook vectors require an explicit caller opt-in."""

    codes = adapter.encode(_images(2))
    embeddings = adapter.code_embeddings(codes)
    assert embeddings.shape == (2, 16, adapter.embedding_dim)
    assert embeddings.dtype == np.float64


def test_decode_round_trip_and_rejects_continuous_latents(adapter: VQVAE) -> None:
    """The decoder consumes integer IDs and rejects plausible-looking floats."""

    codes = adapter.encode(_images(2))
    decoded = adapter.decode(codes)
    assert decoded.shape == (2, 1, 8, 8)
    with pytest.raises(TypeError, match="integer"):
        adapter.decode(codes.astype(np.float64))


def test_codebook_health_metrics_and_frequency_drift_are_finite(adapter: VQVAE) -> None:
    """Training and held-out code usage diagnostics remain finite and bounded."""

    train_codes = adapter.encode(_images(8))
    heldout_codes = adapter.encode(_images(4))
    diagnostics = adapter.codebook_diagnostics(train_codes)
    assert diagnostics["codebook_perplexity"] >= 1.0
    assert 0.0 <= diagnostics["dead_code_rate"] <= 1.0
    assert adapter.metrics_["commitment_distance"] >= 0.0
    assert 0.0 <= adapter.code_frequency_drift(train_codes, heldout_codes) <= 1.0
    metadata = adapter.codebook_metadata()
    assert metadata["counts"]
    assert metadata["model_revision"] == "compact-vq-vae-v1"


def test_code_replacement_is_explicit_and_validated(adapter: VQVAE) -> None:
    """Categorical edits use an integer replacement map, never arithmetic."""

    codes = adapter.encode(_images(1))
    replaced = adapter.replace_codes(codes, {int(codes[0, 0]): 1})
    assert replaced.shape == codes.shape
    assert np.issubdtype(replaced.dtype, np.integer)
    with pytest.raises(TypeError, match="integer"):
        adapter.replace_codes(codes, {0.0: 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="no continuous interpolation"):
        adapter.interpolate_codes(codes, codes, 0.5)


def test_latent_space_rejects_continuous_interpolation(adapter: VQVAE) -> None:
    """The adapter's declared geometry enforces the same policy at the facade."""

    codes = adapter.encode(_images(2))
    with pytest.raises(ValueError, match="no continuous interpolation"):
        adapter.latent_space.interpolate(codes[0], codes[1], 0.5)
