"""Tests for ActivationPatch (B-Method #3, model-mediated) and BMethod Protocol.

Target: ~20 tests covering:
- ActivationPatch construction with VAE adapter
- ActivationPatch construction with RandomProjection adapter
- space property delegates to adapter.latent_space
- is_fitted initially False, True after fit
- fit computes non-zero delta
- __call__ moves toward target
- Input non-mutation
- Error cases: call before fit, delta before fit, empty source, mismatched dim
- apply_trajectory returns np.ndarray, correct shape
- BMethod Protocol: runtime-checkable, Lerp/SteeringVector/ActivationPatch conform,
  non-conforming object fails
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from latent_anything import LatentSpace, Trajectory
from latent_anything.adapters import VAE, RandomProjection
from latent_anything.methods import ActivationPatch, BMethod, Lerp, SteeringVector  # noqa: F401

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def vae_adapter() -> VAE:
    """A tiny pre-trained VAE on synthetic 2D data."""
    vae = VAE(input_dim=2, latent_dim=4, hidden_dim=16, n_epochs=50, beta=0.5, random_state=42)
    rng = np.random.default_rng(42)
    data = rng.uniform(0.0, 1.0, size=(100, 2))
    vae.fit(data)
    return vae


@pytest.fixture
def rp_adapter() -> RandomProjection:
    """A RandomProjection adapter (stateless, no fit needed)."""
    return RandomProjection(input_dim=4, latent_dim=2, random_state=42)


@pytest.fixture
def source_target_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two small 2D clusters for fitting ActivationPatch."""
    source = rng.normal(loc=0.0, scale=0.2, size=(20, 2))
    target = rng.normal(loc=1.0, scale=0.2, size=(20, 2))
    # Scale to [0, 1] for VAE
    source = np.clip(source, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    return source, target


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestActivationPatchConstruction:
    def test_construction_with_vae(self, vae_adapter: VAE) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        assert patch._adapter is vae_adapter  # pyright: ignore[reportPrivateUsage]
        assert not patch.is_fitted

    def test_construction_with_random_projection(self, rp_adapter: RandomProjection) -> None:
        patch = ActivationPatch(adapter=rp_adapter)
        assert patch._adapter is rp_adapter  # pyright: ignore[reportPrivateUsage]
        assert not patch.is_fitted

    def test_space_delegates_to_adapter(self, vae_adapter: VAE) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        space = patch.space
        assert isinstance(space, LatentSpace)
        assert space.dim == vae_adapter.latent_space.dim


# ---------------------------------------------------------------------------
# is_fitted
# ---------------------------------------------------------------------------


class TestActivationPatchIsFitted:
    def test_is_fitted_initially_false(self, vae_adapter: VAE) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        assert not patch.is_fitted

    def test_is_fitted_after_fit(
        self,
        vae_adapter: VAE,
        source_target_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        source, target = source_target_data
        patch = ActivationPatch(adapter=vae_adapter)
        patch.fit(source, target)
        assert patch.is_fitted


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestActivationPatchFit:
    def test_fit_computes_delta(
        self,
        vae_adapter: VAE,
        source_target_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        source, target = source_target_data
        patch = ActivationPatch(adapter=vae_adapter)
        patch.fit(source, target)
        assert patch._delta is not None  # pyright: ignore[reportPrivateUsage]
        assert patch._delta.ndim == 1  # pyright: ignore[reportPrivateUsage]
        assert patch._delta.shape[0] == vae_adapter.latent_dim  # pyright: ignore[reportPrivateUsage]
        # delta should be non-zero (different source/target clusters)
        assert np.linalg.norm(patch._delta) > 0.0  # pyright: ignore[reportPrivateUsage]

    def test_fit_empty_source_raises(self, vae_adapter: VAE) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        source = np.empty((0, 2))
        target = np.random.default_rng(42).normal(size=(10, 2))
        with pytest.raises(ValueError, match="empty"):
            patch.fit(source, target)

    def test_fit_empty_target_raises(self, vae_adapter: VAE) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        source = np.random.default_rng(42).normal(size=(10, 2))
        target = np.empty((0, 2))
        with pytest.raises(ValueError, match="empty"):
            patch.fit(source, target)

    def test_fit_mismatched_dim_raises(self, vae_adapter: VAE) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        source = np.random.default_rng(42).normal(size=(10, 2))
        target = np.random.default_rng(42).normal(size=(10, 3))  # dim mismatch
        with pytest.raises(ValueError, match="dimension mismatch|dim"):
            patch.fit(source, target)

    def test_fit_1d_source_raises(self, vae_adapter: VAE) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        source = np.random.default_rng(42).normal(size=10)  # 1D
        target = np.random.default_rng(42).normal(size=(10, 2))
        with pytest.raises(ValueError, match="2D"):
            patch.fit(source, target)

    def test_fit_1d_target_raises(self, vae_adapter: VAE) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        source = np.random.default_rng(42).normal(size=(10, 2))
        target = np.random.default_rng(42).normal(size=10)  # 1D
        with pytest.raises(ValueError, match="2D"):
            patch.fit(source, target)


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


class TestActivationPatchCall:
    def test_call_moves_toward_target(
        self,
        vae_adapter: VAE,
        source_target_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        source, target = source_target_data
        patch = ActivationPatch(adapter=vae_adapter)
        patch.fit(source, target)

        # Encode source and target to get latent centroids
        source_latent = vae_adapter.encode(source)
        target_latent = vae_adapter.encode(target)
        source_center = source_latent.mean(axis=0)
        target_center = target_latent.mean(axis=0)

        # Patch source-like input
        input_data = source[:3]
        output = patch(input_data)

        # Encode output to latent space
        output_latent = vae_adapter.encode(output)
        output_center = output_latent.mean(axis=0)

        # Patched output should be closer to target than source is
        dist_source_to_target = float(np.linalg.norm(source_center - target_center))
        dist_output_to_target = float(np.linalg.norm(output_center - target_center))
        assert dist_output_to_target < dist_source_to_target

    def test_call_preserves_input(
        self,
        vae_adapter: VAE,
        source_target_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        source, target = source_target_data
        patch = ActivationPatch(adapter=vae_adapter)
        patch.fit(source, target)

        input_data = source[:3].copy()
        input_before = input_data.copy()
        _ = patch(input_data)
        assert_array_almost_equal(input_data, input_before)

    def test_call_before_fit_raises(
        self,
        vae_adapter: VAE,
    ) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        input_data = np.random.default_rng(42).normal(size=(3, 2))
        with pytest.raises(RuntimeError, match="not fitted"):
            patch(input_data)

    def test_output_shape_matches_input(
        self,
        vae_adapter: VAE,
        source_target_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        source, target = source_target_data
        patch = ActivationPatch(adapter=vae_adapter)
        patch.fit(source, target)

        input_data = source[:5]
        output = patch(input_data)
        assert output.shape == input_data.shape  # VAE input_dim == output_dim


# ---------------------------------------------------------------------------
# delta property
# ---------------------------------------------------------------------------


class TestActivationPatchDelta:
    def test_delta_before_fit_raises(self, vae_adapter: VAE) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = patch.delta

    def test_delta_after_fit(
        self,
        vae_adapter: VAE,
        source_target_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        source, target = source_target_data
        patch = ActivationPatch(adapter=vae_adapter)
        patch.fit(source, target)
        delta = patch.delta
        assert delta.shape == (vae_adapter.latent_dim,)
        assert np.linalg.norm(delta) > 0.0

    def test_delta_returns_copy(
        self,
        vae_adapter: VAE,
        source_target_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        source, target = source_target_data
        patch = ActivationPatch(adapter=vae_adapter)
        patch.fit(source, target)
        delta1 = patch.delta
        delta2 = patch.delta
        # Same values but different objects
        assert_array_almost_equal(delta1, delta2)
        assert delta1 is not delta2  # copies


# ---------------------------------------------------------------------------
# apply_trajectory
# ---------------------------------------------------------------------------


class TestActivationPatchApplyTrajectory:
    def test_apply_trajectory_returns_ndarray(
        self,
        vae_adapter: VAE,
        source_target_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        source, target = source_target_data
        patch = ActivationPatch(adapter=vae_adapter)
        patch.fit(source, target)

        traj = Trajectory(data=np.random.default_rng(42).normal(size=(5, vae_adapter.latent_dim)))
        result = patch.apply_trajectory(traj)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, Trajectory)

    def test_apply_trajectory_shape(
        self,
        vae_adapter: VAE,
        source_target_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        source, target = source_target_data
        patch = ActivationPatch(adapter=vae_adapter)
        patch.fit(source, target)

        n_points = 4
        traj = Trajectory(data=np.random.default_rng(42).normal(size=(n_points, vae_adapter.latent_dim)))
        result = patch.apply_trajectory(traj)
        assert result.shape == (n_points, vae_adapter.input_dim)

    def test_apply_trajectory_before_fit_raises(
        self,
        vae_adapter: VAE,
    ) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        traj = Trajectory(data=np.random.default_rng(42).normal(size=(3, vae_adapter.latent_dim)))
        with pytest.raises(RuntimeError, match="not fitted"):
            patch.apply_trajectory(traj)


# ---------------------------------------------------------------------------
# BMethod Protocol
# ---------------------------------------------------------------------------


class _NonConforming:
    """A class that does NOT conform to BMethod."""

    pass


class TestBMethodProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(BMethod, "__instancecheck__")

    def test_lerp_conforms_to_bmethod(self) -> None:
        lerp = Lerp()
        assert isinstance(lerp, BMethod)

    def test_steering_vector_conforms_to_bmethod(self) -> None:
        sv = SteeringVector()
        assert isinstance(sv, BMethod)

    def test_activation_patch_conforms_to_bmethod(
        self,
        vae_adapter: VAE,
    ) -> None:
        patch = ActivationPatch(adapter=vae_adapter)
        assert isinstance(patch, BMethod)

    def test_bmethod_rejects_non_conforming(self) -> None:
        obj = _NonConforming()
        assert not isinstance(obj, BMethod)

    def test_lerp_conforms_with_space(self) -> None:
        space = LatentSpace(dim=8)
        lerp = Lerp(space=space)
        assert isinstance(lerp, BMethod)
        assert lerp.is_fitted is True  # stateless

    def test_steering_vector_unfitted_conforms(self) -> None:
        sv = SteeringVector()
        assert isinstance(sv, BMethod)
        assert not sv.is_fitted
