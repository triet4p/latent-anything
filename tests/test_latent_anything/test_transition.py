"""Tests for the first concrete deterministic latent transition."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from latent_anything import DeterministicLatentTransition, LatentSpace, Trajectory


def _transition(state_dim: int = 2, action_dim: int = 1) -> DeterministicLatentTransition:
    return DeterministicLatentTransition(
        LatentSpace(state_dim, source_model="synthetic-linear"),
        action_dim,
        source_space_identity="synthetic-linear-v1",
    )


def test_identity_dynamics_fit_and_rollout() -> None:
    states = np.array([[0.0, 1.0], [1.0, -2.0], [2.5, 3.0], [-1.0, 0.5]])
    actions = np.zeros((len(states), 1))
    model = _transition().fit(states, actions, states)

    assert_allclose(model.step(np.array([4.0, -3.0]), np.array([0.0])), [4.0, -3.0], atol=1e-7)
    result = model.rollout(np.array([4.0, -3.0]), np.zeros((3, 1)), metadata={"case": "identity"})
    assert isinstance(result, Trajectory)
    assert_allclose(result.to_numpy(), np.tile([4.0, -3.0], (4, 1)), atol=1e-7)
    assert result.metadata["source_space_identity"] == "synthetic-linear-v1"
    assert result.metadata["rollout_horizon"] == 3


def test_action_conditioned_linear_system() -> None:
    rng = np.random.default_rng(63)
    states = rng.normal(size=(80, 2))
    actions = rng.normal(size=(80, 1))
    matrix = np.array([[0.9, 0.1], [-0.2, 0.8]])
    control = np.array([[0.4], [-0.6]])
    next_states = states @ matrix.T + actions @ control.T
    model = _transition().fit(states, actions, next_states, training_horizon=1)

    test_state = np.array([0.3, -1.1])
    test_action = np.array([0.7])
    assert_allclose(
        model.step(test_state, test_action),
        matrix @ test_state + control[:, 0] * test_action[0],
        atol=1e-7,
    )
    metrics = model.evaluate_one_step(states, actions, next_states)
    assert metrics.rmse < 1e-6
    assert model.fit_metadata["training_horizon"] == 1


def test_rollout_error_is_reported_by_horizon() -> None:
    states = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    actions = np.ones((len(states), 1))
    model = DeterministicLatentTransition(LatentSpace(1), 1, ridge=0).fit(states, actions, states + 1)
    rollout_actions = np.ones((4, 1))
    target = np.arange(5, dtype=np.float64).reshape(-1, 1)
    metrics = model.evaluate_rollout(np.array([0.0]), rollout_actions, target)

    assert metrics.horizon == 4
    assert len(metrics.errors_by_horizon) == 4
    assert metrics.runtime_seconds >= 0.0
    assert metrics.stable


@pytest.mark.parametrize(
    ("space", "action_dim", "match"),
    [
        (LatentSpace(2, geometry="unit_norm"), 1, "flat Euclidean"),
        (LatentSpace(2), 0, "action_dim"),
    ],
)
def test_constructor_validation(space: LatentSpace, action_dim: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DeterministicLatentTransition(space, action_dim)


def test_shape_validation_and_unfitted_guard() -> None:
    model = _transition()
    with pytest.raises(RuntimeError, match="fitted"):
        model.step(np.zeros(2), np.zeros(1))
    with pytest.raises(ValueError, match="shape"):
        model.fit(np.zeros((2, 3)), np.zeros((2, 1)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="same number"):
        model.fit(np.zeros((2, 2)), np.zeros((3, 1)), np.zeros((2, 2)))
    model.fit(np.zeros((2, 2)), np.zeros((2, 1)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        model.step(np.zeros(3), np.zeros(1))


def test_fit_and_rollout_are_repeatable() -> None:
    rng = np.random.default_rng(63)
    states = rng.normal(size=(30, 2))
    actions = rng.normal(size=(30, 1))
    next_states = states + 0.2 * states + actions @ np.array([[0.3, -0.4]])
    action_sequence = rng.normal(size=(6, 1))
    first = _transition().fit(states, actions, next_states).rollout(states[0], action_sequence)
    second = _transition().fit(states, actions, next_states).rollout(states[0], action_sequence)
    assert_allclose(first.to_numpy(), second.to_numpy())
