"""Tests for GaussianRendererAdapter (ModelAdapter #4, mode iii: deterministic renderer).

Target: ~25 tests covering:
- Construction with valid/invalid parameters
- latent_space property returns correct gaussian_set LatentSpace with metadata
- decode output shape (n_gaussians, 8) → (H, W, 3)
- decode is deterministic (same latent → same image)
- decode respects opacity (zero opacity → no contribution)
- decode respects colour (different colour channels)
- decode input validation (ndim, row count, column count, numeric constraints)
- encode output shape and constraints (opacity/colour ranges)
- encode roundtrip: encode → decode produces something image-like
- DecodableAdapter Protocol conformance
- No mutation of input arrays
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from latent_anything.adapters import DecodableAdapter, GaussianRendererAdapter, ModelAdapter
from latent_anything.latent_space import LatentSpace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_GAUSSIANS = 16
_IMG_H = 32
_IMG_W = 48
_PARAM_DIM = 8  # pos(2) + scale(2) + opacity(1) + color(3)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> GaussianRendererAdapter:
    return GaussianRendererAdapter(
        n_gaussians=_N_GAUSSIANS,
        img_height=_IMG_H,
        img_width=_IMG_W,
        random_state=42,
    )


@pytest.fixture
def simple_latent() -> np.ndarray:
    """A valid Gaussian parameter array with varied positions and colours.

    Places Gaussians at a 4×4 grid with distinct colours.
    """
    rng = np.random.default_rng(42)
    latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
    # Positions on a 4×4 grid
    rows = np.arange(4, dtype=np.float64)
    cols = np.arange(4, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(rows, cols, indexing="ij")
    latent[:, 0] = (grid_x.ravel() + 0.5) * (_IMG_W / 4)  # position x
    latent[:, 1] = (grid_y.ravel() + 0.5) * (_IMG_H / 4)  # position y
    latent[:, 2] = _IMG_W / 8  # scale x
    latent[:, 3] = _IMG_H / 8  # scale y
    latent[:, 4] = 1.0  # opacity
    # Distinct colours: vary across grid
    latent[:, 5] = rng.uniform(0.3, 1.0, size=_N_GAUSSIANS)  # R
    latent[:, 6] = rng.uniform(0.3, 1.0, size=_N_GAUSSIANS)  # G
    latent[:, 7] = rng.uniform(0.3, 1.0, size=_N_GAUSSIANS)  # B
    return latent


@pytest.fixture
def rendered_image(adapter: GaussianRendererAdapter, simple_latent: np.ndarray) -> np.ndarray:
    """A rendered image from the simple latent."""
    return adapter.decode(simple_latent)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGaussianRendererConstruction:
    def test_construction_with_valid_params(self) -> None:
        adapter = GaussianRendererAdapter(n_gaussians=10, img_height=64, img_width=64, random_state=7)
        assert adapter.n_gaussians == 10
        assert adapter.img_height == 64
        assert adapter.img_width == 64

    def test_construction_without_random_state(self) -> None:
        adapter = GaussianRendererAdapter(n_gaussians=5, img_height=16, img_width=16)
        assert adapter is not None

    def test_invalid_n_gaussians_raises(self) -> None:
        with pytest.raises(ValueError, match="n_gaussians"):
            GaussianRendererAdapter(n_gaussians=0, img_height=16, img_width=16)

    def test_invalid_img_height_raises(self) -> None:
        with pytest.raises(ValueError, match="img_height"):
            GaussianRendererAdapter(n_gaussians=5, img_height=0, img_width=16)

    def test_invalid_img_width_raises(self) -> None:
        with pytest.raises(ValueError, match="img_width"):
            GaussianRendererAdapter(n_gaussians=5, img_height=16, img_width=0)


# ---------------------------------------------------------------------------
# latent_space property
# ---------------------------------------------------------------------------


class TestGaussianRendererLatentSpace:
    def test_latent_space_type(self, adapter: GaussianRendererAdapter) -> None:
        space = adapter.latent_space
        assert isinstance(space, LatentSpace)

    def test_latent_space_geometry(self, adapter: GaussianRendererAdapter) -> None:
        assert adapter.latent_space.geometry == "gaussian_set"

    def test_latent_space_dim(self, adapter: GaussianRendererAdapter) -> None:
        assert adapter.latent_space.dim == _PARAM_DIM

    def test_latent_space_n_gaussians(self, adapter: GaussianRendererAdapter) -> None:
        assert adapter.latent_space.n_gaussians == _N_GAUSSIANS

    def test_latent_space_shape(self, adapter: GaussianRendererAdapter) -> None:
        assert adapter.latent_space.shape == (_N_GAUSSIANS, _PARAM_DIM)

    def test_latent_space_source_model(self, adapter: GaussianRendererAdapter) -> None:
        assert adapter.latent_space.source_model == "gaussian_renderer"

    def test_latent_space_metadata_exposure_mode(self, adapter: GaussianRendererAdapter) -> None:
        metadata = adapter.latent_space.metadata
        assert metadata.get("exposure_mode") == "deterministic_renderer"

    def test_latent_space_metadata_img_dims(self, adapter: GaussianRendererAdapter) -> None:
        metadata = adapter.latent_space.metadata
        assert metadata.get("img_height") == _IMG_H
        assert metadata.get("img_width") == _IMG_W

    def test_latent_space_param_layout(self, adapter: GaussianRendererAdapter) -> None:
        metadata = adapter.latent_space.metadata
        layout = metadata.get("gaussian_set_param_layout")
        assert layout is not None
        assert isinstance(layout, dict)
        assert "position" in layout
        assert "scale" in layout
        assert "opacity" in layout
        assert "color" in layout
        # position(2), scale(2), opacity(1), color(3)
        assert layout["position"] == (0, 2)
        assert layout["scale"] == (2, 2)
        assert layout["opacity"] == (4, 1)
        assert layout["color"] == (5, 3)


# ---------------------------------------------------------------------------
# Decode — output shape
# ---------------------------------------------------------------------------


class TestGaussianRendererDecodeShape:
    def test_decode_output_shape(self, rendered_image: np.ndarray) -> None:
        assert rendered_image.shape == (_IMG_H, _IMG_W, 3)

    def test_decode_output_dtype(self, rendered_image: np.ndarray) -> None:
        assert rendered_image.dtype == np.float64

    def test_decode_output_range(self, rendered_image: np.ndarray) -> None:
        assert np.all(rendered_image >= 0.0)
        assert np.all(rendered_image <= 1.0)

    def test_decode_with_single_gaussian(self) -> None:
        """A single centred Gaussian produces a visible blob."""
        adapter = GaussianRendererAdapter(n_gaussians=1, img_height=32, img_width=32, random_state=0)
        latent = np.array([[16.0, 16.0, 4.0, 4.0, 1.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        img = adapter.decode(latent)
        assert img.shape == (32, 32, 3)
        # Centre pixel should be bright (red channel)
        assert img[16, 16, 0] > 0.5
        # Corner should be dark (Gaussian with σ=4 at distance ~22.6 px)
        assert img[0, 0, 0] < 0.1


# ---------------------------------------------------------------------------
# Decode — determinism
# ---------------------------------------------------------------------------


class TestGaussianRendererDecodeDeterminism:
    def test_decode_is_deterministic(self, adapter: GaussianRendererAdapter, simple_latent: np.ndarray) -> None:
        result1 = adapter.decode(simple_latent)
        result2 = adapter.decode(simple_latent)
        assert_array_equal(result1, result2)

    def test_decode_different_adapters_same_params(self) -> None:
        """Two adapters with same n_gaussians/dims produce identical decode."""
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
        latent[:, 0] = 24.0  # centre x
        latent[:, 1] = 16.0  # centre y
        latent[:, 2] = 10.0  # scale x
        latent[:, 3] = 10.0  # scale y
        latent[:, 4] = 0.5  # opacity
        latent[:, 5] = 1.0  # R

        a1 = GaussianRendererAdapter(n_gaussians=_N_GAUSSIANS, img_height=_IMG_H, img_width=_IMG_W, random_state=0)
        a2 = GaussianRendererAdapter(n_gaussians=_N_GAUSSIANS, img_height=_IMG_H, img_width=_IMG_W, random_state=99)
        assert_array_equal(a1.decode(latent), a2.decode(latent))


# ---------------------------------------------------------------------------
# Decode — opacity and colour constraints
# ---------------------------------------------------------------------------


class TestGaussianRendererDecodeConstraints:
    def test_zero_opacity_no_contribution(self, adapter: GaussianRendererAdapter) -> None:
        """Gaussians with opacity=0 should contribute nothing."""
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
        # Set some non-zero colours but opacity = 0
        latent[:, 2] = 10.0  # scale x
        latent[:, 3] = 10.0  # scale y
        latent[:, 5:] = 1.0  # RGB = 1
        # opacity stays 0
        img = adapter.decode(latent)
        assert np.all(img == 0.0)

    def test_colour_channels_respected(self, adapter: GaussianRendererAdapter) -> None:
        """All-red Gaussians should produce a red image (G=B ≈ 0)."""
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
        # Centred Gaussians, red only
        latent[:, 0] = _IMG_W / 2
        latent[:, 1] = _IMG_H / 2
        latent[:, 2] = 15.0  # scale x
        latent[:, 3] = 15.0  # scale y
        latent[:, 4] = 1.0  # opacity
        latent[:, 5] = 1.0  # R
        latent[:, 6] = 0.0  # G
        latent[:, 7] = 0.0  # B
        img = adapter.decode(latent)
        centre = img[_IMG_H // 2, _IMG_W // 2]
        assert centre[0] > 0.5  # R is bright
        assert centre[1] < 0.1  # G is near 0
        assert centre[2] < 0.1  # B is near 0

    def test_green_channel_only(self, adapter: GaussianRendererAdapter) -> None:
        """All-green Gaussians should produce a green image (R=B ≈ 0)."""
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
        latent[:, 0] = _IMG_W / 2
        latent[:, 1] = _IMG_H / 2
        latent[:, 2] = 15.0
        latent[:, 3] = 15.0
        latent[:, 4] = 1.0
        latent[:, 5] = 0.0  # R
        latent[:, 6] = 1.0  # G
        latent[:, 7] = 0.0  # B
        img = adapter.decode(latent)
        centre = img[_IMG_H // 2, _IMG_W // 2]
        assert centre[1] > 0.5  # G is bright
        assert centre[0] < 0.1  # R is near 0
        assert centre[2] < 0.1  # B is near 0


# ---------------------------------------------------------------------------
# Decode — input validation
# ---------------------------------------------------------------------------


class TestGaussianRendererDecodeValidation:
    def test_decode_1d_input_raises(self, adapter: GaussianRendererAdapter) -> None:
        with pytest.raises(ValueError, match="2D"):
            adapter.decode(np.array([1.0, 2.0]))

    def test_decode_wrong_n_gaussians_raises(self, adapter: GaussianRendererAdapter) -> None:
        latent = np.zeros((_N_GAUSSIANS + 1, _PARAM_DIM), dtype=np.float64)
        with pytest.raises(ValueError, match="Gaussians"):
            adapter.decode(latent)

    def test_decode_wrong_param_dim_raises(self, adapter: GaussianRendererAdapter) -> None:
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="parameter columns"):
            adapter.decode(latent)

    def test_decode_negative_scale_raises(self, adapter: GaussianRendererAdapter) -> None:
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
        latent[:, 2] = -1.0  # negative scale x
        latent[:, 3] = 1.0
        with pytest.raises(ValueError, match="Scale"):
            adapter.decode(latent)

    def test_decode_zero_scale_raises(self, adapter: GaussianRendererAdapter) -> None:
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
        latent[:, 2] = 0.0  # zero scale x
        latent[:, 3] = 1.0
        with pytest.raises(ValueError, match="Scale"):
            adapter.decode(latent)

    def test_decode_negative_opacity_raises(self, adapter: GaussianRendererAdapter) -> None:
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
        latent[:, 2] = 1.0
        latent[:, 3] = 1.0
        latent[:, 4] = -0.1  # negative opacity
        with pytest.raises(ValueError, match="Opacity"):
            adapter.decode(latent)

    def test_decode_opacity_above_one_raises(self, adapter: GaussianRendererAdapter) -> None:
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
        latent[:, 2] = 1.0
        latent[:, 3] = 1.0
        latent[:, 4] = 1.5  # opacity > 1
        with pytest.raises(ValueError, match="Opacity"):
            adapter.decode(latent)

    def test_decode_colour_negative_raises(self, adapter: GaussianRendererAdapter) -> None:
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
        latent[:, 2] = 1.0
        latent[:, 3] = 1.0
        latent[:, 4] = 1.0
        latent[:, 5] = -0.1  # negative R
        with pytest.raises(ValueError, match="Colour"):
            adapter.decode(latent)

    def test_decode_colour_above_one_raises(self, adapter: GaussianRendererAdapter) -> None:
        latent = np.zeros((_N_GAUSSIANS, _PARAM_DIM), dtype=np.float64)
        latent[:, 2] = 1.0
        latent[:, 3] = 1.0
        latent[:, 4] = 1.0
        latent[:, 6] = 1.5  # G > 1
        with pytest.raises(ValueError, match="Colour"):
            adapter.decode(latent)


# ---------------------------------------------------------------------------
# Encode — output shape and constraints
# ---------------------------------------------------------------------------


class TestGaussianRendererEncode:
    def test_encode_output_shape(self, adapter: GaussianRendererAdapter) -> None:
        data = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.float64)
        latent = adapter.encode(data)
        assert latent.shape == (_N_GAUSSIANS, _PARAM_DIM)

    def test_encode_dtype(self, adapter: GaussianRendererAdapter) -> None:
        data = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.float64)
        latent = adapter.encode(data)
        assert latent.dtype == np.float64

    def test_encode_opacity_in_range(self, adapter: GaussianRendererAdapter) -> None:
        data = np.random.default_rng(0).random((_IMG_H, _IMG_W, 3))
        latent = adapter.encode(data)
        assert np.all(latent[:, 4] >= 0.0)
        assert np.all(latent[:, 4] <= 1.0)

    def test_encode_colour_in_range(self, adapter: GaussianRendererAdapter) -> None:
        data = np.random.default_rng(0).random((_IMG_H, _IMG_W, 3))
        latent = adapter.encode(data)
        assert np.all(latent[:, 5] >= 0.0)
        assert np.all(latent[:, 5] <= 1.0)
        assert np.all(latent[:, 6] >= 0.0)
        assert np.all(latent[:, 6] <= 1.0)
        assert np.all(latent[:, 7] >= 0.0)

    def test_encode_scale_positive(self, adapter: GaussianRendererAdapter) -> None:
        data = np.random.default_rng(0).random((_IMG_H, _IMG_W, 3))
        latent = adapter.encode(data)
        assert np.all(latent[:, 2] > 0.0)
        assert np.all(latent[:, 3] > 0.0)

    def test_encode_different_seeds_different(self) -> None:
        """Different random states produce different encoding (jitter varies)."""
        data = np.random.default_rng(0).random((_IMG_H, _IMG_W, 3))
        a1 = GaussianRendererAdapter(n_gaussians=_N_GAUSSIANS, img_height=_IMG_H, img_width=_IMG_W, random_state=1)
        a2 = GaussianRendererAdapter(n_gaussians=_N_GAUSSIANS, img_height=_IMG_H, img_width=_IMG_W, random_state=2)
        l1 = a1.encode(data)
        l2 = a2.encode(data)
        assert not np.allclose(l1, l2)

    def test_encode_input_validation_1d_raises(self, adapter: GaussianRendererAdapter) -> None:
        with pytest.raises(ValueError, match="3D"):
            adapter.encode(np.array([1.0, 2.0, 3.0]))

    def test_encode_input_validation_wrong_dims_raises(self, adapter: GaussianRendererAdapter) -> None:
        data = np.zeros((_IMG_H + 1, _IMG_W, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="Expected image shape"):
            adapter.encode(data)

    def test_encode_input_validation_out_of_range_raises(self, adapter: GaussianRendererAdapter) -> None:
        data = np.ones((_IMG_H, _IMG_W, 3), dtype=np.float64) * 1.5
        with pytest.raises(ValueError, match="must be in"):
            adapter.encode(data)


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


class TestGaussianRendererRoundtrip:
    def test_encode_decode_roundtrip_output_shape(self, adapter: GaussianRendererAdapter) -> None:
        """encode → decode returns an image with the correct shape."""
        data = np.random.default_rng(0).random((_IMG_H, _IMG_W, 3))
        latent = adapter.encode(data)
        reconstructed = adapter.decode(latent)
        assert reconstructed.shape == (_IMG_H, _IMG_W, 3)

    def test_encode_decode_roundtrip_range(self, adapter: GaussianRendererAdapter) -> None:
        """encode → decode output is in [0, 1]."""
        data = np.random.default_rng(0).random((_IMG_H, _IMG_W, 3))
        latent = adapter.encode(data)
        reconstructed = adapter.decode(latent)
        assert np.all(reconstructed >= 0.0)
        assert np.all(reconstructed <= 1.0)

    def test_solid_colour_roundtrip(self, adapter: GaussianRendererAdapter) -> None:
        """A solid red image should encode → decode to a recognisably red image."""
        data = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.float64)
        data[:, :, 0] = 0.8  # red channel
        # Some variation so the grid samples both red and other
        data[: _IMG_H // 2, :, 0] = 0.9
        latent = adapter.encode(data)
        reconstructed = adapter.decode(latent)
        # The centre region should be predominantly red
        centre_region = reconstructed[_IMG_H // 4 : 3 * _IMG_H // 4, _IMG_W // 4 : 3 * _IMG_W // 4]
        avg_r = centre_region[:, :, 0].mean()
        avg_g = centre_region[:, :, 1].mean()
        avg_b = centre_region[:, :, 2].mean()
        assert avg_r > avg_g + 0.1
        assert avg_r > avg_b + 0.1


# ---------------------------------------------------------------------------
# No mutation
# ---------------------------------------------------------------------------


class TestGaussianRendererNoMutation:
    def test_decode_does_not_mutate_input(self, adapter: GaussianRendererAdapter, simple_latent: np.ndarray) -> None:
        original = simple_latent.copy()
        _ = adapter.decode(simple_latent)
        assert_array_equal(simple_latent, original)

    def test_encode_does_not_mutate_input(self, adapter: GaussianRendererAdapter) -> None:
        data = np.random.default_rng(0).random((_IMG_H, _IMG_W, 3))
        original = data.copy()
        _ = adapter.encode(data)
        assert_array_equal(data, original)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestGaussianRendererProtocolConformance:
    def test_conforms_to_model_adapter(self, adapter: GaussianRendererAdapter) -> None:
        assert isinstance(adapter, ModelAdapter)

    def test_conforms_to_decodable_adapter(self, adapter: GaussianRendererAdapter) -> None:
        assert isinstance(adapter, DecodableAdapter)

    def test_has_encode(self, adapter: GaussianRendererAdapter) -> None:
        assert hasattr(adapter, "encode")

    def test_has_decode(self, adapter: GaussianRendererAdapter) -> None:
        assert hasattr(adapter, "decode")

    def test_has_latent_space(self, adapter: GaussianRendererAdapter) -> None:
        assert hasattr(adapter, "latent_space")
