from __future__ import annotations

import numpy as np
import pytest

from latent_anything.mppi import MPPIConfig, MPPIPlanner, compute_mppi_weights


def test_mppi_weights_use_all_candidates_and_are_temperature_controlled() -> None:
    returns = np.asarray([0.0, 1.0, 2.0])
    weights = compute_mppi_weights(returns, temperature=1.0)

    np.testing.assert_allclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)
    assert compute_mppi_weights(returns, temperature=0.1).max() > weights.max()
    assert compute_mppi_weights(returns, temperature=10.0).max() < weights.max()


def test_mppi_weighting_is_stable_for_large_returns() -> None:
    weights = compute_mppi_weights(np.asarray([1e12, 1e12 - 1.0]), temperature=0.5)

    assert np.isfinite(weights).all()
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_mppi_zero_noise_returns_the_bounded_initial_nominal() -> None:
    planner = MPPIPlanner(
        MPPIConfig(
            horizon=3,
            action_dim=1,
            lower_bounds=(-1.0,),
            upper_bounds=(1.0,),
            population_size=16,
            iterations=3,
            noise_std=(0.0,),
            initial_mean=(0.25,),
            seed=69,
        )
    )

    result = planner.plan(lambda candidates: -np.sum(np.square(candidates - 0.25), axis=(1, 2)))

    np.testing.assert_allclose(result.actions, 0.25)
    assert result.effective_sample_size == pytest.approx(16.0)
    assert result.sample_count == 48


def test_mppi_is_seeded_bounded_and_improves_a_quadratic_objective() -> None:
    config = MPPIConfig(
        horizon=4,
        action_dim=2,
        lower_bounds=(-1.0, -2.0),
        upper_bounds=(1.0, 2.0),
        population_size=128,
        iterations=6,
        temperature=0.2,
        noise_std=(0.6, 0.8),
        seed=69,
    )
    planner = MPPIPlanner(config)

    def objective(candidates: np.ndarray) -> np.ndarray:
        return -np.sum(np.square(candidates - np.asarray([0.4, -0.7])), axis=(1, 2))

    first = planner.plan(objective)
    second = planner.plan(objective)

    np.testing.assert_array_equal(first.actions, second.actions)
    assert np.all(first.actions >= np.asarray(config.lower_bounds))
    assert np.all(first.actions <= np.asarray(config.upper_bounds))
    assert first.predicted_return > -2.0
    assert first.runtime_profile.stage_totals()["planning"] >= 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature", 0.0, "temperature"),
        ("population_size", 1, "population_size"),
        ("iterations", 0, "iterations"),
        ("noise_std", (-0.1,), "noise_std"),
    ],
)
def test_mppi_rejects_invalid_configuration(field: str, value: object, message: str) -> None:
    values: dict[str, object] = {
        "horizon": 2,
        "action_dim": 1,
        "lower_bounds": (-1.0,),
        "upper_bounds": (1.0,),
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        MPPIConfig.model_validate(values)


def test_mppi_rejects_invalid_objective_scores() -> None:
    planner = MPPIPlanner(MPPIConfig(horizon=2, action_dim=1, lower_bounds=(-1.0,), upper_bounds=(1.0,)))

    with pytest.raises(ValueError, match="one finite score"):
        planner.plan(lambda candidates: np.zeros((len(candidates), 1)))
    with pytest.raises(ValueError, match="only finite"):
        planner.plan(lambda candidates: np.full(len(candidates), np.nan))
