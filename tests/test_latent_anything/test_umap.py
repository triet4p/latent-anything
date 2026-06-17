"""Tests for the UMAP method class."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from latent_anything.methods import UMAP


class TestUMAPInit:
    """Construction."""

    def test_default_construction(self) -> None:
        reducer = UMAP()
        assert reducer.n_neighbors == 15
        assert reducer.min_dist == 0.1
        assert reducer.n_components == 2
        assert reducer.metric == "euclidean"
        assert reducer.random_state is None

    def test_with_all_parameters(self) -> None:
        reducer = UMAP(
            n_neighbors=10,
            min_dist=0.01,
            n_components=3,
            metric="cosine",
            random_state=42,
        )
        assert reducer.n_neighbors == 10
        assert reducer.min_dist == 0.01
        assert reducer.n_components == 3
        assert reducer.metric == "cosine"
        assert reducer.random_state == 42


class TestUMAPFit:
    """Fit behaviour."""

    def test_fit_produces_embedding(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.random((100, 10))
        reducer = UMAP(n_components=2, random_state=42)
        reducer.fit(data)
        # After fit, transform should succeed without RuntimeError
        result = reducer.transform(data)
        assert result.shape == (100, 2)

    def test_fit_raises_on_1d(self) -> None:
        reducer = UMAP()
        with pytest.raises(ValueError, match="Expected 2D array"):
            reducer.fit(np.array([1.0, 2.0, 3.0]))

    def test_fit_raises_on_empty_samples(self) -> None:
        reducer = UMAP()
        with pytest.raises(ValueError, match="at least 1 sample"):
            reducer.fit(np.empty((0, 5)))

    def test_fit_raises_on_empty_features(self) -> None:
        reducer = UMAP()
        with pytest.raises(ValueError, match="at least 1 sample"):
            reducer.fit(np.empty((5, 0)))


class TestUMAPTransform:
    """Transform behaviour."""

    @pytest.fixture
    def fitted_umap(self) -> UMAP:
        rng = np.random.default_rng(42)
        data = rng.random((100, 10))
        reducer = UMAP(n_components=2, random_state=42)
        reducer.fit(data)
        return reducer

    def test_transform_produces_correct_shape(self, fitted_umap: UMAP) -> None:
        rng = np.random.default_rng(99)
        new_data = rng.random((20, 10))
        result = fitted_umap.transform(new_data)
        assert result.shape == (20, 2)

    def test_transform_raises_before_fit(self) -> None:
        reducer = UMAP(n_components=2)
        with pytest.raises(RuntimeError, match="must be fitted"):
            reducer.transform(np.ones((5, 4)))


class TestUMAPRoundtrip:
    """Roundtrip and fit_transform."""

    def test_fit_transform_returns_correct_shape(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.random((100, 8))
        reducer = UMAP(n_components=2, random_state=42)
        projected = reducer.fit_transform(data)
        assert projected.shape == (100, 2)

    def test_fit_transform_equivalent_to_fit_then_transform(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.random((100, 8))
        reducer = UMAP(n_components=2, random_state=42)
        projected = reducer.fit_transform(data)

        reducer2 = UMAP(n_components=2, random_state=42)
        reducer2.fit(data)
        projected2 = reducer2.transform(data)

        assert_array_almost_equal(projected, projected2)

    def test_random_state_reproducibility(self) -> None:
        """Same random_state produces identical embeddings."""
        rng = np.random.default_rng(42)
        data = rng.random((100, 8))

        r1 = UMAP(n_components=2, random_state=123)
        r2 = UMAP(n_components=2, random_state=123)
        p1 = r1.fit_transform(data)
        p2 = r2.fit_transform(data)
        assert_array_almost_equal(p1, p2)

    def test_different_random_state_different_output(self) -> None:
        """Different random_state values produce different embeddings."""
        rng = np.random.default_rng(42)
        data = rng.random((100, 8))

        r1 = UMAP(n_components=2, random_state=42)
        r2 = UMAP(n_components=2, random_state=99)
        p1 = r1.fit_transform(data)
        p2 = r2.fit_transform(data)
        # The outputs should differ (not exactly equal)
        with np.testing.assert_raises(AssertionError):
            assert_array_almost_equal(p1, p2)
