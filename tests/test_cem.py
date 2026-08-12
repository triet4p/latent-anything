from __future__ import annotations

import numpy as np
import pytest

from latent_anything.cem import CEMConfig, CEMPlanner


def test_cem_config_resolves_elite_count_and_rejects_invalid_bounds() -> None:
    config = CEMConfig(horizon=3, action_dim=2, lower_bounds=(-1.0, -2.0), upper_bounds=(1.0, 2.0))
    assert config.resolved_elite_count == 26
    with pytest.raises(ValueError, match="strictly less"):
        CEMConfig(horizon=2, action_dim=1, lower_bounds=(1.0,), upper_bounds=(1.0,))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("horizon", 0, "horizon"),
        ("action_dim", 0, "action_dim"),
        ("population_size", 1, "population_size"),
        ("elite_count", 0, "elite_count"),
        ("iterations", 0, "iterations"),
        ("smoothing", 1.0, "smoothing"),
        ("min_std", 0.0, "min_std"),
    ],
)
def test_cem_rejects_invalid_population_and_horizon_configuration(field: str, value: object, message: str) -> None:
    values: dict[str, object] = {
        "horizon": 2,
        "action_dim": 1,
        "lower_bounds": (-1.0,),
        "upper_bounds": (1.0,),
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        CEMConfig.model_validate(values)


def test_cem_is_seeded_bounded_and_improves_a_quadratic_objective() -> None:
    config = CEMConfig(
        horizon=4,
        action_dim=1,
        lower_bounds=(-1.0,),
        upper_bounds=(1.0,),
        population_size=64,
        elite_fraction=0.2,
        iterations=8,
        seed=68,
    )
    planner = CEMPlanner(config)

    def objective(candidates: np.ndarray) -> np.ndarray:
        return -np.sum(np.square(candidates[..., 0] - 0.65), axis=1)

    first = planner.plan(objective)
    second = planner.plan(objective)

    np.testing.assert_array_equal(first.actions, second.actions)
    assert np.all(first.actions >= -1.0)
    assert np.all(first.actions <= 1.0)
    assert first.predicted_return > -0.2
    assert len(first.candidate_statistics) == config.iterations
    assert first.convergence_history[-1] >= first.convergence_history[0]
    assert first.runtime_profile.stage_totals()["planning"] >= 0.0


def test_cem_rejects_objective_scores_with_wrong_shape_or_non_finite_values() -> None:
    planner = CEMPlanner(CEMConfig(horizon=2, action_dim=1, lower_bounds=(-1.0,), upper_bounds=(1.0,)))
    with pytest.raises(ValueError, match="one finite score"):
        planner.plan(lambda candidates: np.zeros((len(candidates), 1)))
    with pytest.raises(ValueError, match="only finite"):
        planner.plan(lambda candidates: np.full(len(candidates), np.nan))
