"""Tests for the LatentSpace class."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything import LatentSpace


class TestLatentSpaceInit:
    """Construction and basic attributes."""

    def test_default_construction(self) -> None:
        space = LatentSpace(dim=8)
        assert space.dim == 8
        assert space.geometry == "euclidean"
        assert space.source_model == ""
        assert space.metadata == {}

    def test_with_source_model(self) -> None:
        space = LatentSpace(dim=64, source_model="test-vae")
        assert space.source_model == "test-vae"

    def test_with_metadata(self) -> None:
        meta = {"layers": ["fc1", "fc2"], "type": "conv"}
        space = LatentSpace(dim=16, metadata=meta)
        assert space.metadata == meta
        # Ensure we keep our own copy
        meta["extra"] = True  # type: ignore[assignment]
        assert "extra" not in space.metadata

    def test_shape_property(self) -> None:
        space = LatentSpace(dim=128)
        assert space.shape == (128,)

    def test_negative_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="dim must be >= 1"):
            LatentSpace(dim=0)

    def test_negative_dim_raises_negative(self) -> None:
        with pytest.raises(ValueError, match="dim must be >= 1"):
            LatentSpace(dim=-5)


class TestLatentSpaceValidatePoint:
    """Point validation."""

    def test_valid_point(self) -> None:
        space = LatentSpace(dim=4)
        point = np.array([1.0, 2.0, 3.0, 4.0])
        # Should not raise
        space.validate_point(point)

    def test_wrong_shape_raises(self) -> None:
        space = LatentSpace(dim=4)
        with pytest.raises(ValueError, match="Expected shape \\(4,\\)"):
            space.validate_point(np.array([1.0, 2.0, 3.0]))

    def test_non_array_raises(self) -> None:
        space = LatentSpace(dim=4)
        with pytest.raises((TypeError, AttributeError)):
            space.validate_point([1.0, 2.0, 3.0, 4.0])  # type: ignore[arg-type]

    def test_2d_array_raises(self) -> None:
        space = LatentSpace(dim=4)
        with pytest.raises(ValueError, match="Expected shape \\(4,\\)"):
            space.validate_point(np.ones((2, 4)))


class TestLatentSpaceRepr:
    """Representation."""

    def test_repr(self) -> None:
        space = LatentSpace(dim=32, source_model="vae")
        r = repr(space)
        assert "LatentSpace" in r
        assert "dim=32" in r
        assert "euclidean" in r
        assert "vae" in r
