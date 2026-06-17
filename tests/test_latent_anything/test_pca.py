"""Tests for the PCA method class."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from latent_anything.methods import PCA


class TestPCAInit:
    """Construction."""

    def test_default_construction(self) -> None:
        pca = PCA()
        assert pca.n_components is None

    def test_with_n_components(self) -> None:
        pca = PCA(n_components=3)
        assert pca.n_components == 3


class TestPCAFit:
    """Fit behaviour."""

    def test_fit_produces_components(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.random((50, 10))
        pca = PCA(n_components=3)
        pca.fit(data)
        assert pca.components_.shape == (3, 10)
        assert pca.explained_variance_ratio_.shape == (3,)
        assert pca.mean_.shape == (10,)

    def test_fit_raises_on_1d(self) -> None:
        pca = PCA()
        with pytest.raises(ValueError, match="Expected 2D array"):
            pca.fit(np.array([1.0, 2.0, 3.0]))

    def test_fit_raises_on_non_array(self) -> None:
        pca = PCA()
        with pytest.raises((TypeError, AttributeError)):
            pca.fit([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]

    def test_fit_raises_on_empty_samples(self) -> None:
        pca = PCA()
        with pytest.raises(ValueError, match="at least 1 sample"):
            pca.fit(np.empty((0, 5)))

    def test_fit_raises_on_empty_features(self) -> None:
        pca = PCA()
        with pytest.raises(ValueError, match="at least 1 sample"):
            pca.fit(np.empty((5, 0)))


class TestPCATransform:
    """Transform behaviour."""

    @pytest.fixture
    def fitted_pca(self) -> PCA:
        rng = np.random.default_rng(42)
        data = rng.random((50, 10))
        pca = PCA(n_components=3)
        pca.fit(data)
        return pca

    def test_transform_produces_correct_shape(self, fitted_pca: PCA) -> None:
        rng = np.random.default_rng(99)
        new_data = rng.random((20, 10))
        result = fitted_pca.transform(new_data)
        assert result.shape == (20, 3)

    def test_transform_raises_before_fit(self) -> None:
        pca = PCA(n_components=2)
        with pytest.raises(RuntimeError, match="must be fitted"):
            pca.transform(np.ones((5, 4)))


class TestPCARoundtrip:
    """Roundtrip and fit_transform."""

    def test_fit_transform_returns_correct_shape(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.random((100, 8))
        pca = PCA(n_components=2)
        projected = pca.fit_transform(data)
        assert projected.shape == (100, 2)

    def test_fit_transform_equivalent_to_fit_then_transform(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.random((100, 8))
        pca = PCA(n_components=2)
        projected = pca.fit_transform(data)

        pca2 = PCA(n_components=2)
        pca2.fit(data)
        projected2 = pca2.transform(data)

        assert_array_almost_equal(projected, projected2)

    def test_unfitted_properties_raise(self) -> None:
        pca = PCA()
        for prop in ["components_", "explained_variance_ratio_", "mean_"]:
            with pytest.raises(RuntimeError, match="must be fitted"):
                _ = getattr(pca, prop)
