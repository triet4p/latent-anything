"""Tests for immutable flat and structured ``LatentValue`` containers."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

from latent_anything import LatentSpace, LatentValue, Trajectory
from latent_anything.adapters import VAE, GaussianRendererAdapter


@given(st.lists(st.floats(-10, 10, allow_nan=False, allow_infinity=False), min_size=3, max_size=3))
def test_flat_value_defensively_copies_input(values: list[float]) -> None:
    data = np.array(values, dtype=np.float64)
    value = LatentValue(data, LatentSpace(dim=3))
    data[:] = 0.0
    assert_array_equal(value.to_numpy(), np.array(values, dtype=np.float64))


def test_flat_batch_preserves_shape_and_slicing() -> None:
    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    value = LatentValue(data, LatentSpace(dim=4), metadata={"source": "test"})
    assert value.is_batch is True
    assert value.batch_size == 3
    assert value.batch_shape == (3,)
    assert value.item_shape == (4,)
    assert value[1].shape == (4,)
    assert value[1:].shape == (2, 4)
    with pytest.raises(TypeError):
        value[1][0]


def test_hidden_state_sequence_preserves_leading_axes() -> None:
    data = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    value = LatentValue(data, LatentSpace(dim=4))
    assert value.is_batch is True
    assert value.batch_shape == (2, 3)
    assert value[0].shape == (3, 4)
    with pytest.raises(ValueError, match="Structured"):
        value.to_trajectory()


def test_numpy_and_metadata_cannot_mutate_value() -> None:
    value = LatentValue(np.array([1.0, 2.0]), LatentSpace(dim=2), metadata={"nested": {"name": "safe"}})
    output = value.to_numpy()
    output[:] = 0.0
    assert_array_equal(value.to_numpy(), np.array([1.0, 2.0]))
    with pytest.raises(TypeError):
        value.metadata["new"] = "value"  # type: ignore[index]


def test_structured_gaussian_value_validates_shape_and_geometry() -> None:
    adapter = GaussianRendererAdapter(n_gaussians=2, img_height=8, img_width=8, random_state=0)
    image = np.full((8, 8, 3), 0.5, dtype=np.float64)
    value = adapter.encode_value(image)
    assert value.space.geometry == "gaussian_set"
    assert value.shape == (2, 8)
    assert value.is_batch is False
    with pytest.raises(ValueError, match="Trajectory"):
        value.to_trajectory()


def test_trajectory_compatibility_round_trip() -> None:
    trajectory = Trajectory(np.arange(6, dtype=np.float64).reshape(2, 3))
    value = LatentValue.from_trajectory(trajectory)
    assert value.space.dim == 3
    assert_array_equal(value.to_trajectory().to_numpy(), trajectory.to_numpy())


def test_vae_encode_value_preserves_existing_numerical_path() -> None:
    data = np.random.default_rng(1).random((6, 4))
    vae = VAE(input_dim=4, latent_dim=2, random_state=1, n_epochs=1)
    vae.fit(data)
    assert_array_equal(vae.encode_value(data).to_numpy(), vae.encode(data))


def test_discrete_value_preserves_integer_codes() -> None:
    space = LatentSpace(dim=3, geometry="discrete_code", codebook_size=4)
    value = LatentValue(np.array([[0, 1, 3], [2, 2, 0]], dtype=np.int64), space)
    assert value.to_numpy().dtype == np.int64
