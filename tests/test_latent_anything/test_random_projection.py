"""Tests for the RandomProjection adapter class (ModelAdapter #2)."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything.adapters import RandomProjection
from latent_anything.latent_space import LatentSpace

# ---------------------------------------------------------------------------
# Test data fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data() -> np.ndarray:
    """Generate simple random data (not normalised — JL projection works on any range)."""
    rng = np.random.default_rng(99)
    return rng.normal(size=(50, 8)).astype(np.float64)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRandomProjectionInit:
    """Construction tests."""

    def test_default_construction(self) -> None:
        rp = RandomProjection(input_dim=10, latent_dim=3)
        assert rp.input_dim == 10
        assert rp.latent_dim == 3
        assert rp.projection_matrix_.shape == (3, 10)
        assert rp.projection_matrix_.dtype == np.float64

    def test_construction_with_random_state(self) -> None:
        rp = RandomProjection(input_dim=20, latent_dim=5, random_state=42)
        assert rp.random_state == 42
        assert rp.projection_matrix_.shape == (5, 20)

    def test_projection_matrix_is_normalised(self) -> None:
        """Check that the matrix is normalised by 1/sqrt(latent_dim)."""
        rp = RandomProjection(input_dim=100, latent_dim=25)
        mat = rp.projection_matrix_
        # Mean of squared entries should be approx 1/latent_dim (since N(0,1) scaled by 1/sqrt(latent_dim))
        mean_sq = np.mean(mat**2)
        # After normalisation: Var = 1/latent_dim
        assert abs(mean_sq - 1.0 / 25) < 0.05, f"Mean sq entry {mean_sq:.4f} not near 1/25 = 0.04"

    def test_raises_on_invalid_input_dim(self) -> None:
        with pytest.raises(ValueError, match="input_dim must be >= 1"):
            RandomProjection(input_dim=0, latent_dim=2)

    def test_raises_on_invalid_latent_dim(self) -> None:
        with pytest.raises(ValueError, match="latent_dim must be >= 1"):
            RandomProjection(input_dim=5, latent_dim=0)


# ---------------------------------------------------------------------------
# latent_space property
# ---------------------------------------------------------------------------


class TestRandomProjectionLatentSpace:
    """latent_space property tests."""

    def test_latent_space_returns_correct_object(self) -> None:
        rp = RandomProjection(input_dim=8, latent_dim=3)
        ls = rp.latent_space
        assert isinstance(ls, LatentSpace)
        assert ls.dim == 3
        assert ls.geometry == "euclidean"
        assert ls.source_model == "random_projection"

    def test_latent_space_is_available_immediately(self) -> None:
        """latent_space should be available at construction (no fit needed)."""
        rp = RandomProjection(input_dim=8, latent_dim=4)
        ls = rp.latent_space
        assert ls.dim == 4


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


class TestRandomProjectionEncode:
    """Encode behaviour tests."""

    def test_encode_produces_correct_shape(self, simple_data: np.ndarray) -> None:
        rp = RandomProjection(input_dim=8, latent_dim=3, random_state=42)
        encoded = rp.encode(simple_data)
        assert encoded.shape == (50, 3)

    def test_encode_returns_numpy_array(self, simple_data: np.ndarray) -> None:
        rp = RandomProjection(input_dim=8, latent_dim=3)
        encoded = rp.encode(simple_data)
        assert isinstance(encoded, np.ndarray)

    def test_encode_is_available_without_fit(self) -> None:
        """RandomProjection does not need a fit step — encode works immediately."""
        rp = RandomProjection(input_dim=8, latent_dim=3, random_state=42)
        data = np.random.default_rng(0).normal(size=(10, 8))
        encoded = rp.encode(data)
        assert encoded.shape == (10, 3)

    def test_encode_raises_on_1d_data(self) -> None:
        rp = RandomProjection(input_dim=4, latent_dim=2)
        with pytest.raises(ValueError, match="Expected 2D array"):
            rp.encode(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_encode_raises_on_wrong_input_dim(self) -> None:
        rp = RandomProjection(input_dim=5, latent_dim=2)
        with pytest.raises(ValueError, match="Expected input_dim=5"):
            rp.encode(np.ones((10, 3)))


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


class TestRandomProjectionDecode:
    """Decode behaviour tests."""

    def test_decode_produces_correct_shape(self, simple_data: np.ndarray) -> None:
        rp = RandomProjection(input_dim=8, latent_dim=3, random_state=42)
        encoded = rp.encode(simple_data)
        decoded = rp.decode(encoded)
        assert decoded.shape == (50, 8)

    def test_decode_returns_numpy_array(self, simple_data: np.ndarray) -> None:
        rp = RandomProjection(input_dim=8, latent_dim=3, random_state=42)
        encoded = rp.encode(simple_data)
        decoded = rp.decode(encoded)
        assert isinstance(decoded, np.ndarray)

    def test_decode_raises_on_1d_latent(self) -> None:
        rp = RandomProjection(input_dim=4, latent_dim=2)
        with pytest.raises(ValueError, match="Expected 2D array"):
            rp.decode(np.array([1.0, 2.0]))

    def test_decode_raises_on_wrong_latent_dim(self) -> None:
        rp = RandomProjection(input_dim=5, latent_dim=2)
        with pytest.raises(ValueError, match="Expected latent_dim=2"):
            rp.decode(np.ones((10, 3)))


# ---------------------------------------------------------------------------
# Roundtrip & distance preservation
# ---------------------------------------------------------------------------


class TestRandomProjectionRoundtrip:
    """Roundtrip and distance preservation tests."""

    def test_roundtrip_outputs_same_shape(self, simple_data: np.ndarray) -> None:
        """Encode-then-decode should return data of the same shape."""
        rp = RandomProjection(input_dim=8, latent_dim=3, random_state=42)
        encoded = rp.encode(simple_data)
        decoded = rp.decode(encoded)
        assert decoded.shape == simple_data.shape

    def test_decode_is_not_identity(self, simple_data: np.ndarray) -> None:
        """Transpose reconstruction is approximate — output should differ from input."""
        rp = RandomProjection(input_dim=8, latent_dim=3, random_state=42)
        encoded = rp.encode(simple_data)
        decoded = rp.decode(encoded)
        # Should not be exactly the same (random projection loses information)
        assert not np.allclose(simple_data, decoded)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestRandomProjectionReproducibility:
    """random_state reproducibility tests."""

    def test_same_seed_gives_same_projection_matrix(self) -> None:
        rp1 = RandomProjection(input_dim=10, latent_dim=4, random_state=42)
        rp2 = RandomProjection(input_dim=10, latent_dim=4, random_state=42)
        np.testing.assert_array_equal(rp1.projection_matrix_, rp2.projection_matrix_)

    def test_same_seed_gives_same_encode(self) -> None:
        data = np.random.default_rng(0).normal(size=(20, 10))
        rp1 = RandomProjection(input_dim=10, latent_dim=4, random_state=42)
        rp2 = RandomProjection(input_dim=10, latent_dim=4, random_state=42)
        enc1 = rp1.encode(data)
        enc2 = rp2.encode(data)
        np.testing.assert_array_equal(enc1, enc2)

    def test_different_seeds_give_different_matrices(self) -> None:
        rp1 = RandomProjection(input_dim=10, latent_dim=4, random_state=42)
        rp2 = RandomProjection(input_dim=10, latent_dim=4, random_state=99)
        assert not np.allclose(rp1.projection_matrix_, rp2.projection_matrix_)

    def test_no_random_state_is_deterministic_per_instance(self) -> None:
        """Without random_state, each call to the same instance gives the same encode
        (matrix is fixed at construction, not regenerated per call)."""
        data = np.random.default_rng(0).normal(size=(10, 6))
        rp = RandomProjection(input_dim=6, latent_dim=2)
        enc1 = rp.encode(data)
        enc2 = rp.encode(data)
        np.testing.assert_array_equal(enc1, enc2)

    def test_no_random_state_gives_different_matrices_across_instances(self) -> None:
        """Without random_state, different instances should have different matrices."""
        data = np.random.default_rng(0).normal(size=(10, 6))
        rp1 = RandomProjection(input_dim=6, latent_dim=2)
        rp2 = RandomProjection(input_dim=6, latent_dim=2)
        enc1 = rp1.encode(data)
        enc2 = rp2.encode(data)
        # Very unlikely to be equal by chance
        assert not np.allclose(enc1, enc2)


# ---------------------------------------------------------------------------
# Distance preservation (JL lemma sanity)
# ---------------------------------------------------------------------------


class TestRandomProjectionDistancePreservation:
    """Approximate distance preservation (Johnson-Lindenstrauss lemma)."""

    def test_distances_roughly_preserved(self) -> None:
        """Pairwise Euclidean distances should be roughly preserved after projection.

        Johnson-Lindenstrauss lemma guarantees approximate distance preservation
        when projecting to a sufficiently high latent dimension. With 100 points
        in 100D → 40D, the mean distance ratio should be near 1.0.
        """
        rng = np.random.default_rng(42)
        n = 100
        input_dim = 100
        latent_dim = 40  # JL guarantee: sufficient for ~100 points

        data = rng.normal(size=(n, input_dim))
        rp = RandomProjection(input_dim=input_dim, latent_dim=latent_dim, random_state=42)
        encoded = rp.encode(data)

        # Compute pairwise distance ratios
        from scipy.spatial.distance import pdist

        orig_dists = pdist(data, metric="euclidean")
        proj_dists = pdist(encoded, metric="euclidean")

        # Protect against division by near-zero distances
        valid = orig_dists > 1e-8
        ratios = proj_dists[valid] / orig_dists[valid]
        mean_ratio = float(np.mean(ratios))
        std_ratio = float(np.std(ratios))

        # Mean ratio should be near 1.0 (approx distance preservation)
        assert abs(mean_ratio - 1.0) < 0.4, f"Mean distance ratio {mean_ratio:.4f} too far from 1.0"
        # Standard deviation should be bounded (JL lemma guarantees concentration)
        assert std_ratio < 0.35, f"Distance ratio std {std_ratio:.4f} too large"
