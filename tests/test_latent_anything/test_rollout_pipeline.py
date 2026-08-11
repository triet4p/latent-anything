"""Sprint 66 tests for Pipeline #3 and the shared pipeline contract."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping

import numpy as np
import pytest

from latent_anything import (
    DeterministicLatentTransition,
    InMemoryCache,
    LatentSpace,
    LatentValue,
    ObjectSpec,
    PipelineContract,
    RolloutPipeline,
    RolloutPipelineSpec,
    build_rollout_pipeline_from_config,
)
from latent_anything.registry import KIND_RUNTIME, Registry
from latent_anything.runtime import RuntimeProfiler
from latent_anything.trajectory import Trajectory


def _fitted_transition() -> DeterministicLatentTransition:
    states = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    actions = np.ones((4, 1))
    return DeterministicLatentTransition(LatentSpace(2, source_model="rollout-test"), 1).fit(
        states,
        actions,
        states + np.array([1.0, 0.0]),
    )


def test_rollout_pipeline_composes_initial_value_and_transition() -> None:
    transition = _fitted_transition()
    pipeline = RolloutPipeline(transition)
    initial = LatentValue(np.array([4.0, 0.0]), pipeline.latent_space)

    result = pipeline.run(initial, np.ones((3, 1)), metadata={"case": "unit"})

    assert isinstance(pipeline, PipelineContract)
    assert result.cache_hit is False
    np.testing.assert_allclose(result.to_numpy(), [[4, 0], [5, 0], [6, 0], [7, 0]])
    assert result.trajectory.metadata["case"] == "unit"
    assert result.states is result.trajectory


def test_rollout_pipeline_cache_and_profile_record_distinct_stages() -> None:
    cache = InMemoryCache()
    pipeline = RolloutPipeline(_fitted_transition(), cache=cache)
    profiler = RuntimeProfiler()
    initial = np.array([0.0, 0.0])
    actions = np.ones((2, 1))

    first = pipeline.run(initial, actions, profiler=profiler)
    second = pipeline.run(initial, actions, profiler=profiler)

    assert first.cache_hit is False
    assert second.cache_hit is True
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
    assert {event.stage for event in profiler.snapshot().events} == {"cache", "transition"}
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_rollout_pipeline_sync_and_async_results_match() -> None:
    pipeline = RolloutPipeline(_fitted_transition())
    initial = np.array([2.0, 0.0])
    actions = np.ones((4, 1))

    sync = pipeline.run(initial, actions)
    asynchronous = asyncio.run(pipeline.run_async(initial, actions))

    np.testing.assert_array_equal(sync.to_numpy(), asynchronous.to_numpy())
    np.testing.assert_array_equal(sync.actions, asynchronous.actions)


def test_rollout_pipeline_config_builds_runtime_transition() -> None:
    transition = _fitted_transition()
    registry = Registry("rollout-test")
    registry.register(KIND_RUNTIME, "fixture_transition", lambda: transition)
    spec = RolloutPipelineSpec(
        transition=ObjectSpec(kind=KIND_RUNTIME, name="fixture_transition"),
        cache=True,
    )

    pipeline = build_rollout_pipeline_from_config(spec, registry=registry)

    assert pipeline.transition is transition
    assert pipeline.cache is not None


class _SlowTransition:
    state_dim = 1
    action_dim = 1
    source_space_identity = "slow"
    latent_space = LatentSpace(1, source_model="slow")

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return state + action

    def mean_rollout(
        self, initial_state: np.ndarray, actions: np.ndarray, *, metadata: Mapping[str, object] | None = None
    ) -> Trajectory:
        time.sleep(0.2)
        return Trajectory(np.vstack([initial_state, initial_state + np.sum(actions, axis=0)]), metadata=metadata)


class _ErrorTransition:
    state_dim = 1
    action_dim = 1
    source_space_identity = "error"
    latent_space = LatentSpace(1, source_model="error")

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return state + action

    def mean_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Trajectory:
        del initial_state, actions, metadata
        raise ValueError("transition failed")


def test_rollout_pipeline_async_cancellation_is_explicit() -> None:
    async def scenario() -> None:
        task = asyncio.create_task(RolloutPipeline(_SlowTransition()).run_async(np.zeros(1), np.ones((1, 1))))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_rollout_pipeline_preserves_transition_errors_in_sync_and_async_paths() -> None:
    pipeline = RolloutPipeline(_ErrorTransition())
    with pytest.raises(ValueError, match="transition failed"):
        pipeline.run(np.zeros(1), np.ones((1, 1)))
    with pytest.raises(ValueError, match="transition failed"):
        asyncio.run(pipeline.run_async(np.zeros(1), np.ones((1, 1))))
