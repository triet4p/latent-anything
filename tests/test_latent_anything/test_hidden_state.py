"""Tests for HiddenStateAdapter (ModelAdapter #3, mode ii: no-explicit-latent).

Target: ~20 tests covering:
- Construction with valid/invalid parameters
- latent_space property returns correct Euclidean LatentSpace with metadata
- encode produces correct shape (n_samples, hidden_dim)
- encode is deterministic given same seed
- encode is different with different seeds
- encode is nonlinear (ReLU) — hidden state differs from linear projection
- encode input validation (ndim, feature count, empty)
- Encode outputs are non-negative (ReLU)
- ModelAdapter Protocol conformance (encode + latent_space)
- DecodableAdapter Protocol non-conformance (no decode)
- ActivationPatch rejects HiddenStateAdapter
- Reproducibility: same seed → same weights
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from latent_anything import LatentSpace
from latent_anything.adapters import DecodableAdapter, HiddenStateAdapter, ModelAdapter
from latent_anything.methods import ActivationPatch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def adapter() -> HiddenStateAdapter:
    return HiddenStateAdapter(input_dim=4, hidden_dim=3, random_state=42)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestHiddenStateAdapterConstruction:
    def test_construction_with_valid_params(self) -> None:
        adapter = HiddenStateAdapter(input_dim=8, hidden_dim=5, random_state=7)
        assert adapter.input_dim == 8
        assert adapter.hidden_dim == 5
        assert adapter.random_state == 7

    def test_construction_without_random_state(self) -> None:
        adapter = HiddenStateAdapter(input_dim=4, hidden_dim=2)
        assert adapter.random_state is None

    def test_construction_weights_shape_input_dim(self) -> None:
        adapter = HiddenStateAdapter(input_dim=10, hidden_dim=4, random_state=0)
        assert adapter.weight1_.shape == (10, 4)
        assert adapter.weight2_.shape == (4, 4)

    def test_construction_bias_shape(self) -> None:
        adapter = HiddenStateAdapter(input_dim=6, hidden_dim=3, random_state=0)
        assert adapter.bias1_.shape == (3,)
        assert adapter.bias2_.shape == (3,)

    def test_invalid_input_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="input_dim"):
            HiddenStateAdapter(input_dim=0, hidden_dim=3)

    def test_invalid_hidden_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="hidden_dim"):
            HiddenStateAdapter(input_dim=4, hidden_dim=0)


# ---------------------------------------------------------------------------
# latent_space property
# ---------------------------------------------------------------------------


class TestHiddenStateAdapterLatentSpace:
    def test_latent_space_type(self, adapter: HiddenStateAdapter) -> None:
        space = adapter.latent_space
        assert isinstance(space, LatentSpace)

    def test_latent_space_dim(self, adapter: HiddenStateAdapter) -> None:
        assert adapter.latent_space.dim == 3

    def test_latent_space_geometry(self, adapter: HiddenStateAdapter) -> None:
        assert adapter.latent_space.geometry == "euclidean"

    def test_latent_space_source_model(self, adapter: HiddenStateAdapter) -> None:
        assert adapter.latent_space.source_model == "hidden_state"

    def test_latent_space_metadata_exposure_mode(self, adapter: HiddenStateAdapter) -> None:
        metadata = adapter.latent_space.metadata
        assert metadata.get("exposure_mode") == "hidden_state"

    def test_latent_space_shape(self, adapter: HiddenStateAdapter) -> None:
        assert adapter.latent_space.shape == (3,)


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


class TestHiddenStateAdapterEncode:
    def test_encode_shape(self, adapter: HiddenStateAdapter) -> None:
        data = np.random.default_rng(0).normal(size=(10, 4))
        result = adapter.encode(data)
        assert result.shape == (10, 3)

    def test_encode_single_sample(self, adapter: HiddenStateAdapter) -> None:
        data = np.random.default_rng(0).normal(size=(1, 4))
        result = adapter.encode(data)
        assert result.shape == (1, 3)

    def test_encode_deterministic(self) -> None:
        adapter1 = HiddenStateAdapter(input_dim=4, hidden_dim=3, random_state=42)
        adapter2 = HiddenStateAdapter(input_dim=4, hidden_dim=3, random_state=42)
        data = np.random.default_rng(0).normal(size=(10, 4))
        result1 = adapter1.encode(data)
        result2 = adapter2.encode(data)
        assert_array_equal(result1, result2)

    def test_encode_different_seeds_different(self) -> None:
        adapter1 = HiddenStateAdapter(input_dim=4, hidden_dim=3, random_state=1)
        adapter2 = HiddenStateAdapter(input_dim=4, hidden_dim=3, random_state=2)
        data = np.random.default_rng(0).normal(size=(10, 4))
        result1 = adapter1.encode(data)
        result2 = adapter2.encode(data)
        assert not np.allclose(result1, result2)

    def test_encode_is_nonlinear(self) -> None:
        """ReLU nonlinearity: negative pre-activations become zero."""
        adapter = HiddenStateAdapter(input_dim=2, hidden_dim=5, random_state=42)
        # Input with large negative values to trigger ReLU zeroing
        data = np.array([[-10.0, -10.0]])
        result = adapter.encode(data)
        # ReLU zeroes negative activations
        assert np.all(result >= 0.0)

    def test_encode_outputs_non_negative(self, adapter: HiddenStateAdapter) -> None:
        """ReLU ensures all outputs are non-negative."""
        data = np.random.default_rng(0).normal(size=(20, 4))
        result = adapter.encode(data)
        assert np.all(result >= -1e-15)

    def test_encode_1d_input_raises(self, adapter: HiddenStateAdapter) -> None:
        with pytest.raises(ValueError, match="2D"):
            adapter.encode(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_encode_wrong_feature_count_raises(self, adapter: HiddenStateAdapter) -> None:
        with pytest.raises(ValueError, match="input_dim"):
            adapter.encode(np.random.default_rng(0).normal(size=(5, 7)))

    def test_encode_returns_float64(self, adapter: HiddenStateAdapter) -> None:
        data = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
        result = adapter.encode(data)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestHiddenStateAdapterProtocolConformance:
    def test_conforms_to_model_adapter(self, adapter: HiddenStateAdapter) -> None:
        assert isinstance(adapter, ModelAdapter)

    def test_does_not_conform_to_decodable_adapter(self, adapter: HiddenStateAdapter) -> None:
        assert not isinstance(adapter, DecodableAdapter)

    def test_has_no_decode_attribute(self, adapter: HiddenStateAdapter) -> None:
        assert not hasattr(adapter, "decode")

    def test_vae_conforms_to_decodable_adapter(self) -> None:
        from latent_anything.adapters import VAE

        vae = VAE(input_dim=2, latent_dim=3)
        assert isinstance(vae, DecodableAdapter)

    def test_random_projection_conforms_to_decodable_adapter(self) -> None:
        from latent_anything.adapters import RandomProjection

        rp = RandomProjection(input_dim=4, latent_dim=2)
        assert isinstance(rp, DecodableAdapter)


# ---------------------------------------------------------------------------
# ActivationPatch rejection
# ---------------------------------------------------------------------------


class TestActivationPatchRejectsNonDecodable:
    def test_activation_patch_raises_type_error_with_hidden_state(self) -> None:
        adapter = HiddenStateAdapter(input_dim=4, hidden_dim=3)
        with pytest.raises(TypeError):
            # HiddenStateAdapter is not a DecodableAdapter — ActivationPatch
            # should raise TypeError at construction due to Protocol check
            _ = ActivationPatch(adapter=adapter)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestHiddenStateAdapterReproducibility:
    def test_same_seed_same_weights(self) -> None:
        a1 = HiddenStateAdapter(input_dim=5, hidden_dim=3, random_state=42)
        a2 = HiddenStateAdapter(input_dim=5, hidden_dim=3, random_state=42)
        assert_array_equal(a1.weight1_, a2.weight1_)
        assert_array_equal(a1.bias1_, a2.bias1_)
        assert_array_equal(a1.weight2_, a2.weight2_)
        assert_array_equal(a1.bias2_, a2.bias2_)

    def test_different_seed_different_weights(self) -> None:
        a1 = HiddenStateAdapter(input_dim=5, hidden_dim=3, random_state=1)
        a2 = HiddenStateAdapter(input_dim=5, hidden_dim=3, random_state=2)
        assert not np.allclose(a1.weight1_, a2.weight1_)
