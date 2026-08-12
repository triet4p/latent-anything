from __future__ import annotations

import numpy as np

from latent_anything import (
    DeterministicLatentTransition,
    LatentSpace,
    LinearRewardScorer,
    MonteCarloValueEstimator,
    RolloutPipeline,
)
from latent_anything.mppi import MPPIConfig, MPPIPlanner
from latent_anything.reward_value import RewardValueEvaluator


class _TrackingEvaluator(RewardValueEvaluator):
    def __init__(self, base: RewardValueEvaluator) -> None:
        super().__init__(base.reward_scorer, base.value_estimator)
        self.calls = 0

    def evaluate(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().evaluate(*args, **kwargs)  # type: ignore[arg-type]


def _pipeline() -> RolloutPipeline:
    space = LatentSpace(1, source_model="mppi-rollout")
    states = np.arange(10.0)[:, None]
    actions = np.ones((10, 1))
    transition = DeterministicLatentTransition(space, 1).fit(states, actions, states + 1.0)
    scorer = LinearRewardScorer(1, 1, source_space_identity="mppi-rollout")
    scorer.fit(states, actions, actions[:, 0])
    estimator = MonteCarloValueEstimator(1, discount=0.9, horizon=3)
    estimator.fit(states, np.arange(10.0))
    return RolloutPipeline(transition, evaluator=RewardValueEvaluator(scorer, estimator))


def test_mppi_rollout_planning_reuses_pipeline_and_reward_evaluator() -> None:
    planner = MPPIPlanner(
        MPPIConfig(
            horizon=3,
            action_dim=1,
            lower_bounds=(0.0,),
            upper_bounds=(1.0,),
            population_size=24,
            iterations=4,
            temperature=0.2,
            noise_std=(0.4,),
            seed=69,
        )
    )

    result = planner.plan_rollouts(np.zeros(1), _pipeline())

    assert result.actions.shape == (3, 1)
    assert np.all((result.actions >= 0.0) & (result.actions <= 1.0))
    assert result.predicted_return > 0.0
    assert len(result.convergence_history) == 4
    assert result.runtime_profile.stage_totals()["transition"] > 0.0


def test_mppi_explicit_evaluator_overrides_pipeline_evaluation() -> None:
    pipeline = _pipeline()
    explicit = _TrackingEvaluator(pipeline.evaluator)  # type: ignore[arg-type]

    MPPIPlanner(
        MPPIConfig(
            horizon=2,
            action_dim=1,
            lower_bounds=(0.0,),
            upper_bounds=(1.0,),
            population_size=8,
            iterations=1,
            seed=69,
        )
    ).plan_rollouts(np.zeros(1), pipeline, evaluator=explicit)

    assert explicit.calls > 0


def test_mppi_receding_horizon_shifts_and_executes_actions() -> None:
    pipeline = _pipeline()
    planner = MPPIPlanner(
        MPPIConfig(
            horizon=3,
            action_dim=1,
            lower_bounds=(0.0,),
            upper_bounds=(1.0,),
            population_size=12,
            iterations=2,
            temperature=0.4,
            noise_std=(0.2,),
            seed=69,
        )
    )

    result = planner.plan_receding_horizon(np.zeros(1), pipeline, steps=3)

    assert result.actions.shape == (3, 1)
    assert result.states.shape == (4, 1)
    assert len(result.plans) == 3
    assert result.runtime_profile.stage_totals()["planning"] > 0.0
