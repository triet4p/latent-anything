from __future__ import annotations

import numpy as np

from latent_anything import (
    DeterministicLatentTransition,
    LatentSpace,
    LinearRewardScorer,
    MonteCarloValueEstimator,
    RolloutPipeline,
)
from latent_anything.cem import CEMConfig, CEMPlanner
from latent_anything.reward_value import RewardValueEvaluator


def _pipeline() -> RolloutPipeline:
    space = LatentSpace(1, source_model="cem-rollout")
    states = np.arange(9.0)[:, None]
    actions = np.ones((9, 1))
    transition = DeterministicLatentTransition(space, 1).fit(states, actions, states + 1.0)
    scorer = LinearRewardScorer(1, 1, source_space_identity="cem-rollout")
    scorer.fit(states, actions, actions[:, 0])
    estimator = MonteCarloValueEstimator(1, discount=0.9, horizon=3)
    estimator.fit(np.arange(9.0)[:, None], np.arange(9.0))
    return RolloutPipeline(transition, evaluator=RewardValueEvaluator(scorer, estimator))


def test_cem_rollout_planning_uses_pipeline_evaluator_and_returns_model_score() -> None:
    pipeline = _pipeline()
    planner = CEMPlanner(
        CEMConfig(
            horizon=3,
            action_dim=1,
            lower_bounds=(0.0,),
            upper_bounds=(1.0,),
            population_size=24,
            elite_fraction=0.25,
            iterations=4,
            seed=68,
        )
    )

    result = planner.plan_rollouts(np.zeros(1), pipeline)

    assert result.actions.shape == (3, 1)
    assert np.all((result.actions >= 0.0) & (result.actions <= 1.0))
    assert result.predicted_return > 0.0
    assert len(result.convergence_history) == 4
    assert result.runtime_profile.stage_totals()["transition"] > 0.0
