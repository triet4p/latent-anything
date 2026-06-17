"""Tests for the VAE adapter class (ModelAdapter #1)."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything.adapters import VAE
from latent_anything.latent_space import LatentSpace

# ---------------------------------------------------------------------------
# Test data fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data() -> np.ndarray:
    """Generate simple [0,1]-scaled cluster data."""
    rng = np.random.default_rng(99)
    data = rng.random((60, 6)).astype(np.float64)
    return np.clip(data, 0.0, 1.0)


@pytest.fixture
def fitted_vae(simple_data: np.ndarray) -> VAE:
    """A VAE fitted on simple_data with a small number of epochs."""
    vae = VAE(input_dim=6, latent_dim=3, n_epochs=30, random_state=42)
    vae.fit(simple_data)
    return vae


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestVAEInit:
    """Construction tests."""

    def test_default_construction(self) -> None:
        vae = VAE(input_dim=10, latent_dim=2)
        assert vae.input_dim == 10
        assert vae.latent_dim == 2
        assert vae.hidden_dim == max(2 * 4, 10)  # latent_dim*4=8, max(8,10)=10
        assert vae.learning_rate == 0.001
        assert vae.n_epochs == 200
        assert vae.beta == 1.0
        assert vae.random_state is None

    def test_construction_with_all_params(self) -> None:
        vae = VAE(
            input_dim=20,
            latent_dim=5,
            hidden_dim=64,
            learning_rate=0.01,
            n_epochs=100,
            beta=0.5,
            random_state=123,
        )
        assert vae.input_dim == 20
        assert vae.latent_dim == 5
        assert vae.hidden_dim == 64
        assert vae.learning_rate == 0.01
        assert vae.n_epochs == 100
        assert vae.beta == 0.5
        assert vae.random_state == 123

    def test_hidden_dim_heuristic_uses_max(self) -> None:
        """hidden_dim defaults to max(latent_dim*4, input_dim)."""
        # latent_dim*4 > input_dim → hidden = latent*4
        vae1 = VAE(input_dim=5, latent_dim=8)
        assert vae1.hidden_dim == 32  # 8*4 = 32 > 5
        # input_dim > latent_dim*4 → hidden = input_dim
        vae2 = VAE(input_dim=100, latent_dim=2)
        assert vae2.hidden_dim == 100  # 2*4 = 8 < 100

    def test_raises_on_invalid_input_dim(self) -> None:
        with pytest.raises(ValueError, match="input_dim must be >= 1"):
            VAE(input_dim=0, latent_dim=2)

    def test_raises_on_invalid_latent_dim(self) -> None:
        with pytest.raises(ValueError, match="latent_dim must be >= 1"):
            VAE(input_dim=5, latent_dim=0)


# ---------------------------------------------------------------------------
# latent_space property
# ---------------------------------------------------------------------------


class TestVAELatentSpace:
    """latent_space property tests."""

    def test_latent_space_returns_correct_object(self) -> None:
        vae = VAE(input_dim=8, latent_dim=3)
        ls = vae.latent_space
        assert isinstance(ls, LatentSpace)
        assert ls.dim == 3
        assert ls.geometry == "euclidean"
        assert ls.source_model == "vae"

    def test_latent_space_before_fit(self) -> None:
        """latent_space should be available even before fit."""
        vae = VAE(input_dim=8, latent_dim=4)
        ls = vae.latent_space
        assert ls.dim == 4


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestVAEFit:
    """Fit behaviour tests."""

    def test_fit_trains_and_tracks_loss(self, simple_data: np.ndarray) -> None:
        vae = VAE(input_dim=6, latent_dim=3, n_epochs=20, random_state=42)
        vae.fit(simple_data)
        assert len(vae.loss_history_) == 20
        # Loss should decrease over training
        assert vae.loss_history_[-1] < vae.loss_history_[0]

    def test_fit_raises_on_1d_data(self) -> None:
        vae = VAE(input_dim=4, latent_dim=2)
        with pytest.raises(ValueError, match="Expected 2D array"):
            vae.fit(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_fit_raises_on_empty_data(self) -> None:
        vae = VAE(input_dim=4, latent_dim=2)
        with pytest.raises(ValueError, match="at least 1 sample"):
            vae.fit(np.empty((0, 4)))

    def test_fit_raises_on_wrong_input_dim(self) -> None:
        vae = VAE(input_dim=5, latent_dim=2)
        with pytest.raises(ValueError, match="Expected input_dim=5"):
            vae.fit(np.ones((10, 3)))


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


class TestVAEEncode:
    """Encode behaviour tests."""

    def test_encode_produces_correct_shape(self, fitted_vae: VAE, simple_data: np.ndarray) -> None:
        encoded = fitted_vae.encode(simple_data)
        assert encoded.shape == (60, 3)

    def test_encode_returns_numpy_array(self, fitted_vae: VAE, simple_data: np.ndarray) -> None:
        encoded = fitted_vae.encode(simple_data)
        assert isinstance(encoded, np.ndarray)

    def test_encode_raises_before_fit(self) -> None:
        vae = VAE(input_dim=4, latent_dim=2)
        with pytest.raises(RuntimeError, match="must be fitted before encode"):
            vae.encode(np.ones((5, 4)))

    def test_encode_raises_on_wrong_shape(self, fitted_vae: VAE) -> None:
        with pytest.raises(ValueError, match="Expected data shape"):
            fitted_vae.encode(np.ones((5, 3)))  # 3 != input_dim=6


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


class TestVAEDecode:
    """Decode behaviour tests."""

    def test_decode_produces_correct_shape(self, fitted_vae: VAE, simple_data: np.ndarray) -> None:
        encoded = fitted_vae.encode(simple_data)
        decoded = fitted_vae.decode(encoded)
        assert decoded.shape == (60, 6)

    def test_decode_returns_numpy_array(self, fitted_vae: VAE, simple_data: np.ndarray) -> None:
        encoded = fitted_vae.encode(simple_data)
        decoded = fitted_vae.decode(encoded)
        assert isinstance(decoded, np.ndarray)

    def test_decode_raises_before_fit(self) -> None:
        vae = VAE(input_dim=4, latent_dim=2)
        with pytest.raises(RuntimeError, match="must be fitted before decode"):
            vae.decode(np.ones((5, 2)))

    def test_decode_raises_on_wrong_latent_dim(self, fitted_vae: VAE) -> None:
        with pytest.raises(ValueError, match="Expected latent shape"):
            fitted_vae.decode(np.ones((5, 5)))  # 5 != latent_dim=3


# ---------------------------------------------------------------------------
# Roundtrip & reconstruction sanity
# ---------------------------------------------------------------------------


class TestVAERoundtrip:
    """Reconstruction and roundtrip tests."""

    def test_roundtrip_gives_plausible_result(self, fitted_vae: VAE, simple_data: np.ndarray) -> None:
        """Encode-then-decode should reconstruct similar data in [0,1]."""
        encoded = fitted_vae.encode(simple_data)
        decoded = fitted_vae.decode(encoded)
        assert decoded.shape == simple_data.shape
        # Decoded values should be in [0, 1] (sigmoid output)
        assert decoded.min() >= 0.0
        assert decoded.max() <= 1.0
        # Reconstruction should be non-trivial (not all same value)
        assert np.std(decoded) > 0.0

    def test_loss_decreases_during_training(self, simple_data: np.ndarray) -> None:
        """Training loss should substantially decrease."""
        vae = VAE(input_dim=6, latent_dim=3, n_epochs=100, learning_rate=0.01, random_state=42)
        vae.fit(simple_data)
        first_loss = vae.loss_history_[0]
        last_loss = vae.loss_history_[-1]
        # Loss should have decreased by at least 20%
        assert last_loss < first_loss * 0.8, f"Loss did not decrease enough: {first_loss:.4f} → {last_loss:.4f}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestVAEReproducibility:
    """random_state reproducibility tests."""

    def test_same_seed_gives_same_encode(self, simple_data: np.ndarray) -> None:
        vae1 = VAE(input_dim=6, latent_dim=3, n_epochs=20, random_state=42)
        vae1.fit(simple_data)
        enc1 = vae1.encode(simple_data)

        vae2 = VAE(input_dim=6, latent_dim=3, n_epochs=20, random_state=42)
        vae2.fit(simple_data)
        enc2 = vae2.encode(simple_data)

        np.testing.assert_array_almost_equal(enc1, enc2)

    def test_different_seeds_give_different_encode(self, simple_data: np.ndarray) -> None:
        vae1 = VAE(input_dim=6, latent_dim=3, n_epochs=20, random_state=42)
        vae1.fit(simple_data)
        enc1 = vae1.encode(simple_data)

        vae2 = VAE(input_dim=6, latent_dim=3, n_epochs=20, random_state=99)
        vae2.fit(simple_data)
        enc2 = vae2.encode(simple_data)

        assert not np.allclose(enc1, enc2)
