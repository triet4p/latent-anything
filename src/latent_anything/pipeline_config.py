"""Pydantic specs and builders for the concrete pipeline stories."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from latent_anything.analysis_pipeline import AnalysisPipeline
from latent_anything.cem import CEMConfig, CEMPlanner
from latent_anything.config import ObjectSpec, build_from_config
from latent_anything.manipulation_pipeline import ManipulationPipeline
from latent_anything.mppi import MPPIConfig, MPPIPlanner
from latent_anything.registry import Registry
from latent_anything.reward_value import RewardValueEvaluator
from latent_anything.rollout_pipeline import RolloutPipeline
from latent_anything.runtime.cache import InMemoryCache


class PipelineSpec(BaseModel):
    """Config spec for an adapter plus Layer A analysis method."""

    adapter: ObjectSpec = Field(..., description="Config spec for the adapter")
    method: ObjectSpec = Field(..., description="Config spec for the Layer A method")


def build_pipeline_from_config(spec: PipelineSpec, *, registry: Registry | None = None) -> AnalysisPipeline:
    """Build an :class:`AnalysisPipeline` through the registry."""

    adapter = build_from_config(spec.adapter, registry=registry)
    method = build_from_config(spec.method, registry=registry)
    return AnalysisPipeline(adapter=adapter, method=method)  # pyright: ignore[reportUnknownArgumentType]


class ManipulationPipelineSpec(BaseModel):
    """Config spec for a Layer B method and optional data-space adapter."""

    method: ObjectSpec = Field(..., description="Config spec for the Layer B method")
    adapter: ObjectSpec | None = Field(default=None, description="Optional config spec for the adapter")


def build_manipulation_pipeline_from_config(
    spec: ManipulationPipelineSpec,
    *,
    registry: Registry | None = None,
) -> ManipulationPipeline:
    """Build a :class:`ManipulationPipeline` through the registry."""

    method = build_from_config(spec.method, registry=registry)
    adapter: Any = None if spec.adapter is None else build_from_config(spec.adapter, registry=registry)
    return ManipulationPipeline(method=method, adapter=adapter)


class RewardValueEvaluationSpec(BaseModel):
    """Config spec for fitted reward/value components used by rollouts."""

    reward_scorer: ObjectSpec = Field(..., description="Config spec for a scalar latent reward scorer")
    value_estimator: ObjectSpec = Field(..., description="Config spec for a latent state-value estimator")


def build_reward_value_evaluator_from_config(
    spec: RewardValueEvaluationSpec,
    *,
    registry: Registry | None = None,
) -> RewardValueEvaluator:
    """Build a reward/value evaluator through the runtime registry."""

    reward_scorer = build_from_config(spec.reward_scorer, registry=registry)
    value_estimator = build_from_config(spec.value_estimator, registry=registry)
    return RewardValueEvaluator(reward_scorer=reward_scorer, value_estimator=value_estimator)


class RolloutPipelineSpec(BaseModel):
    """Config spec for a registered latent transition.

    ``cache`` controls whether the builder installs an in-memory cache.  It
    deliberately does not turn configuration into a workflow language.
    """

    transition: ObjectSpec = Field(..., description="Config spec for a runtime latent transition")
    cache: bool = Field(default=False, description="Cache completed mean rollouts in memory")
    reward_value: RewardValueEvaluationSpec | None = Field(
        default=None,
        description="Optional fitted reward/value evaluator applied to imagined rollouts",
    )


def build_rollout_pipeline_from_config(
    spec: RolloutPipelineSpec,
    *,
    registry: Registry | None = None,
) -> RolloutPipeline:
    """Build a :class:`RolloutPipeline` from a runtime transition spec."""

    transition = build_from_config(spec.transition, registry=registry)
    cache = InMemoryCache() if spec.cache else None
    evaluator = (
        None
        if spec.reward_value is None
        else build_reward_value_evaluator_from_config(spec.reward_value, registry=registry)
    )
    return RolloutPipeline(transition=transition, cache=cache, evaluator=evaluator)


class CEMPlannerSpec(CEMConfig):
    """Config schema for a registry- or application-built CEM planner."""


def build_cem_planner_from_config(spec: CEMPlannerSpec) -> CEMPlanner:
    """Build a bounded CEM planner from validated configuration."""

    return CEMPlanner(spec)


class MPPIPlannerSpec(MPPIConfig):
    """Config schema for a registry- or application-built MPPI planner."""


def build_mppi_planner_from_config(spec: MPPIPlannerSpec) -> MPPIPlanner:
    """Build a bounded MPPI planner from validated configuration."""

    return MPPIPlanner(spec)
