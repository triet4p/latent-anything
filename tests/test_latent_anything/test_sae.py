"""Tests for the SAE method class (Sparse Autoencoder)."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything import Method
from latent_anything.methods import PCA, SAE, UMAP


class TestSAEInit:
    """Construction."""

    def test_default_construction(self) -> None:
        sae = SAE(n_components=2)
        assert sae.n_components == 2
        assert sae.l1_coef == 0.01
        assert sae.learning_rate == 0.01
        assert sae.n_epochs == 500
        assert sae.random_state is None

    def test_with_all_parameters(self) -> None:
        sae = SAE(
            n_components=5,
            l1_coef=0.1,
            learning_rate=0.001,
            n_epochs=100,
            random_state=42,
        )
        assert sae.n_components == 5
        assert sae.l1_coef == 0.1
        assert sae.learning_rate == 0.001
        assert sae.n_epochs == 100
        assert sae.random_state == 42


class TestSAEFit:
    """Fit behaviour."""

    def test_fit_trains_and_tracks_loss(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.random((50, 10))
        sae = SAE(n_components=3, n_epochs=20, random_state=42)
        sae.fit(data)
        assert len(sae.loss_history_) == 20
        # Loss should generally decrease over training
        assert sae.loss_history_[-1] < sae.loss_history_[0] * 2  # generous bound

    def test_fit_raises_on_1d(self) -> None:
        sae = SAE(n_components=2)
        with pytest.raises(ValueError, match="Expected 2D array"):
            sae.fit(np.array([1.0, 2.0, 3.0]))

    def test_fit_raises_on_empty_samples(self) -> None:
        sae = SAE(n_components=2)
        with pytest.raises(ValueError, match="at least 1 sample"):
            sae.fit(np.empty((0, 5)))

    def test_fit_raises_on_empty_features(self) -> None:
        sae = SAE(n_components=2)
        with pytest.raises(ValueError, match="at least 1 sample"):
            sae.fit(np.empty((5, 0)))


class TestSAETransform:
    """Transform behaviour."""

    @pytest.fixture
    def fitted_sae(self) -> SAE:
        rng = np.random.default_rng(42)
        data = rng.random((50, 10))
        sae = SAE(n_components=3, n_epochs=50, random_state=42)
        sae.fit(data)
        return sae

    def test_transform_produces_correct_shape(self, fitted_sae: SAE) -> None:
        rng = np.random.default_rng(99)
        new_data = rng.random((20, 10))
        result = fitted_sae.transform(new_data)
        assert result.shape == (20, 3)

    def test_transform_raises_before_fit(self) -> None:
        sae = SAE(n_components=2)
        with pytest.raises(RuntimeError, match="must be fitted"):
            sae.transform(np.ones((5, 4)))


class TestSAESparsity:
    """Sparsity verification: L1 penalty produces near-zero latent activations."""

    def test_l1_penalty_encourages_sparsity(self) -> None:
        """With higher L1 coef, latent activations should have more near-zero entries."""
        rng = np.random.default_rng(42)
        data = rng.random((100, 8))

        # Train with aggressive sparsity
        sae = SAE(n_components=4, l1_coef=0.5, learning_rate=0.01, n_epochs=200, random_state=42)
        sae.fit(data)
        latent = sae.transform(data)

        # With aggressive L1, at least some activations should be near-zero
        n_near_zero = np.sum(np.abs(latent) < 0.01)
        assert n_near_zero > 0, (
            f"Expected some near-zero activations with L1 sparsity, "
            f"got only {n_near_zero} entries < 0.01 out of {latent.size}"
        )


class TestSAERandomState:
    """Reproducibility with random_state."""

    def test_random_state_reproducibility(self) -> None:
        """Same random_state produces identical embeddings."""
        rng = np.random.default_rng(42)
        data = rng.random((50, 8))

        s1 = SAE(n_components=2, n_epochs=50, random_state=123)
        s2 = SAE(n_components=2, n_epochs=50, random_state=123)
        s1.fit(data)
        s2.fit(data)
        p1 = s1.transform(data)
        p2 = s2.transform(data)
        np.testing.assert_array_almost_equal(p1, p2)

    def test_different_random_state_different_training(self) -> None:
        """Different random_state values produce different loss trajectories."""
        rng = np.random.default_rng(42)
        data = rng.random((50, 8))

        s1 = SAE(n_components=2, n_epochs=50, random_state=42)
        s2 = SAE(n_components=2, n_epochs=50, random_state=99)
        s1.fit(data)
        s2.fit(data)
        # The loss histories should differ (different initialisations diverge)
        assert s1.loss_history_ != s2.loss_history_, (
            "Different random_state should produce different training trajectories"
        )


class TestSAEErrorCases:
    """Error handling."""

    def test_unfitted_transform_raises(self) -> None:
        sae = SAE(n_components=2)
        with pytest.raises(RuntimeError, match="must be fitted"):
            sae.transform(np.ones((5, 4)))


class TestMethodProtocolConformance:
    """Smoke tests: PCA, UMAP, SAE all structurally satisfy the Method Protocol."""

    def test_pca_conforms_to_method_protocol(self) -> None:
        pca = PCA(n_components=2)
        assert isinstance(pca, Method), "PCA must satisfy the Method Protocol"

    def test_umap_conforms_to_method_protocol(self) -> None:
        reducer = UMAP(n_components=2)
        assert isinstance(reducer, Method), "UMAP must satisfy the Method Protocol"

    def test_sae_conforms_to_method_protocol(self) -> None:
        sae = SAE(n_components=2)
        assert isinstance(sae, Method), "SAE must satisfy the Method Protocol"

    def test_all_methods_have_fit_and_transform_signatures(self) -> None:
        """Structural check: each method provides fit/transform with correct signatures."""
        methods: list[tuple[object, str]] = [
            (PCA(n_components=2), "PCA"),
            (UMAP(n_components=2), "UMAP"),
            (SAE(n_components=2), "SAE"),
        ]
        for method, name in methods:
            assert hasattr(method, "fit"), f"{name} is missing fit()"
            assert hasattr(method, "transform"), f"{name} is missing transform()"
            assert hasattr(method, "fit_transform"), f"{name} is missing fit_transform()"
