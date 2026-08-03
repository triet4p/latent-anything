"""Offline contract tests for the 3D Gaussian renderer facade."""

import numpy as np
import pytest

from latent_anything import LatentValue
from latent_anything.adapters import Gaussian3DRendererAdapter, GaussianCamera
from latent_anything.integrations.gsplat_renderer import ReferenceGaussianBackend


def camera() -> GaussianCamera:
    return GaussianCamera(32, 24, np.array([[20.0, 0.0, 16.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]]), np.eye(4))


def latent() -> np.ndarray:
    return np.array([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.1, 0.8, 1.0, 0.2, 0.1]])


def test_3d_renderer_metadata_and_decode_shape() -> None:
    adapter = Gaussian3DRendererAdapter(1, camera(), backend=ReferenceGaussianBackend())
    image = adapter.decode(latent())
    assert image.shape == (24, 32, 3)
    assert image.min() >= 0 and image.max() <= 1
    assert adapter.latent_space.geometry == "gaussian_3d"
    assert adapter.latent_space.metadata["parameter_layout"]["spherical_harmonics"] == (11, 3)


def test_adapter_latent_value_round_trip_uses_the_14_field_space() -> None:
    adapter = Gaussian3DRendererAdapter(1, camera(), backend=ReferenceGaussianBackend())
    value = LatentValue(latent(), adapter.latent_space)
    assert value.space.shape == (1, 14)
    assert adapter.decode(value.to_numpy()).shape == (24, 32, 3)


def test_3d_renderer_camera_transform_changes_projection() -> None:
    first = Gaussian3DRendererAdapter(1, camera(), backend=ReferenceGaussianBackend()).decode(latent())
    shifted = camera()
    shifted.world_to_camera[0, 3] = 0.4
    second = Gaussian3DRendererAdapter(1, shifted, backend=ReferenceGaussianBackend()).decode(latent())
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("column", [7, 8, 9])
def test_3d_renderer_rejects_non_positive_scales(column: int) -> None:
    value = latent()
    value[0, column] = 0
    with pytest.raises(ValueError, match="scales"):
        Gaussian3DRendererAdapter(1, camera(), backend=ReferenceGaussianBackend()).decode(value)


def test_3d_renderer_rejects_zero_rotation() -> None:
    value = latent()
    value[0, 3:7] = 0
    with pytest.raises(ValueError, match="rotations"):
        Gaussian3DRendererAdapter(1, camera(), backend=ReferenceGaussianBackend()).decode(value)
