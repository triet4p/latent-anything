"""Sprint 66 tests for Pipeline #3 and the shared pipeline contract."""

from __future__ import annotations

import asyncio
import threading
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


def test_rollout_pipeline_cache_key_includes_metadata_provenance() -> None:
    cache = InMemoryCache()
    pipeline = RolloutPipeline(_fitted_transition(), cache=cache)
    initial = np.array([0.0, 0.0])
    actions = np.ones((2, 1))

    first = pipeline.run(initial, actions, metadata={"episode": "first"})
    second = pipeline.run(initial, actions, metadata={"episode": "second"})

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert first.trajectory.metadata["episode"] == "first"
    assert second.trajectory.metadata["episode"] == "second"


def test_rollout_pipeline_sync_and_async_results_match() -> None:
    pipeline = RolloutPipeline(_fitted_transition())
    initial = np.array([2.0, 0.0])
    actions = np.ones((4, 1))

    sync = pipeline.run(initial, actions)
    asynchronous = asyncio.run(pipeline.run_async(initial, actions))

    np.testing.assert_array_equal(sync.to_numpy(), asynchronous.to_numpy())
    np.testing.assert_array_equal(sync.actions, asynchronous.actions)


def test_rollout_pipeline_stream_carries_state_and_preserves_eager_order() -> None:
    pipeline = RolloutPipeline(_fitted_transition())
    initial = np.array([0.0, 0.0])
    chunks = [np.ones((2, 1)), np.empty((0, 1)), np.ones((1, 1))]
    profiler = RuntimeProfiler()

    streamed = list(pipeline.stream(initial, chunks, max_chunk_rows=2, profiler=profiler, metadata={"episode": 7}))
    eager = pipeline.run(initial, np.ones((3, 1)), metadata={"episode": 7})

    assert [chunk.shape for chunk in streamed] == [(2, 2), (1, 2)]
    np.testing.assert_array_equal(np.concatenate([chunk.to_numpy() for chunk in streamed]), eager.to_numpy()[1:])
    assert [chunk.metadata["chunk_index"] for chunk in streamed] == [0, 2]
    assert [chunk.metadata["chunk_rows"] for chunk in streamed] == [2, 1]
    profile = profiler.snapshot()
    assert len(profile.events) == 1
    assert profile.events[0].metadata["chunk_count"] == 2
    assert profile.events[0].metadata["row_count"] == 3


def test_rollout_pipeline_stream_is_bounded_and_closes_early_source() -> None:
    pipeline = RolloutPipeline(_fitted_transition())
    requested: list[int] = []
    closed = False

    def source():
        nonlocal closed
        try:
            for index in range(3):
                requested.append(index)
                yield np.ones((1, 1))
        finally:
            closed = True

    stream = pipeline.stream(np.zeros(2), source(), max_chunk_rows=1)
    first = next(stream)
    assert first.shape == (1, 2)
    assert requested == [0]
    stream.close()
    assert closed


def test_rollout_pipeline_stream_rejects_oversized_chunks_before_transition() -> None:
    pipeline = RolloutPipeline(_fitted_transition())

    with pytest.raises(ValueError, match="max_chunk_rows"):
        next(pipeline.stream(np.zeros(2), [np.ones((2, 1))], max_chunk_rows=1))


class _ExplosiveArrayLike:
    def __array__(self) -> np.ndarray:
        raise AssertionError("array conversion must not run for unsupported chunks")


def test_rollout_pipeline_stream_rejects_unbounded_array_like_before_conversion() -> None:
    pipeline = RolloutPipeline(_fitted_transition())

    with pytest.raises(TypeError, match="exact numpy.ndarray"):
        next(pipeline.stream(np.zeros(2), [[1.0], [1.0]], max_chunk_rows=1))
    with pytest.raises(TypeError, match="exact numpy.ndarray"):
        next(pipeline.stream(np.zeros(2), [_ExplosiveArrayLike()], max_chunk_rows=1))


def test_rollout_pipeline_stream_rejects_unsupported_ndarray_shapes_and_dtypes() -> None:
    pipeline = RolloutPipeline(_fitted_transition())

    with pytest.raises(ValueError, match="shape"):
        next(pipeline.stream(np.zeros(2), [np.ones(1)], max_chunk_rows=1))
    with pytest.raises(ValueError, match="finite numeric"):
        next(pipeline.stream(np.zeros(2), [np.array([["x"]], dtype=object)], max_chunk_rows=1))


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


class _StreamErrorTransition:
    state_dim = 1
    action_dim = 1
    source_space_identity = "stream-error"
    stream_state_contract = "explicit"
    latent_space = LatentSpace(1, source_model="stream-error")

    def __init__(self) -> None:
        self.calls = 0

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.calls == 2:
            raise ValueError("stream transition failed")
        return state + action

    def mean_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Trajectory:
        del metadata
        values = [initial_state]
        current = initial_state
        for action in actions:
            current = self.step(current, action)
            values.append(current)
        return Trajectory(np.stack(values))


class _ResettableStreamTransition:
    state_dim = 1
    action_dim = 1
    source_space_identity = "resettable-stream"
    latent_space = LatentSpace(1, source_model="resettable-stream")

    def __init__(self) -> None:
        self.reset_calls = 0
        self.hidden = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self.hidden = 0

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.hidden += 1
        return state + action + self.hidden

    def mean_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Trajectory:
        del metadata
        self.reset()
        values = [initial_state]
        current = initial_state
        for action in actions:
            current = self.step(current, action)
            values.append(current)
        return Trajectory(np.stack(values))


class _UnresettableStatefulStreamTransition:
    state_dim = 1
    action_dim = 1
    source_space_identity = "unresettable-stateful-stream"
    latent_space = LatentSpace(1, source_model="unresettable-stateful-stream")

    def __init__(self) -> None:
        self.hidden = 0

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.hidden += 1
        return state + action + self.hidden

    def mean_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Trajectory:
        del metadata
        values = [initial_state]
        current = initial_state
        for action in actions:
            current = self.step(current, action)
            values.append(current)
        return Trajectory(np.stack(values))


class _SlowStreamTransition:
    state_dim = 1
    action_dim = 1
    source_space_identity = "slow-stream"
    stream_state_contract = "explicit"
    latent_space = LatentSpace(1, source_model="slow-stream")

    def __init__(self) -> None:
        self.started = threading.Event()
        self.finished = threading.Event()

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.started.set()
        time.sleep(0.03)
        self.finished.set()
        return state + action

    def mean_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Trajectory:
        del metadata
        values = [initial_state]
        current = initial_state
        for action in actions:
            current = self.step(current, action)
            values.append(current)
        return Trajectory(np.stack(values))


class _BlockingStreamSource:
    def __init__(self) -> None:
        self.iter_started = threading.Event()
        self.iter_release = threading.Event()
        self.close_started = threading.Event()
        self.close_release = threading.Event()
        self.close_finished = threading.Event()
        self._yielded = False

    def __iter__(self) -> _BlockingStreamSource:
        self.iter_started.set()
        if not self.iter_release.wait(timeout=2.0):
            raise TimeoutError("blocking __iter__ was not released")
        return self

    def __next__(self) -> np.ndarray:
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return np.ones((1, 1))

    def close(self) -> None:
        self.close_started.set()
        if not self.close_release.wait(timeout=2.0):
            raise TimeoutError("blocking close was not released")
        self.close_finished.set()


class _BlockingCloseStreamSource:
    def __init__(self) -> None:
        self.close_started = threading.Event()
        self.close_release = threading.Event()
        self.close_finished = threading.Event()
        self._yielded = False

    def __iter__(self) -> _BlockingCloseStreamSource:
        return self

    def __next__(self) -> np.ndarray:
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return np.ones((1, 1))

    def close(self) -> None:
        self.close_started.set()
        if not self.close_release.wait(timeout=2.0):
            raise TimeoutError("blocking close was not released")
        self.close_finished.set()


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


def test_rollout_pipeline_stream_resets_stateful_transition_and_propagates_errors_at_chunk_boundary() -> None:
    transition = _ResettableStreamTransition()
    pipeline = RolloutPipeline(transition)
    first = list(pipeline.stream(np.zeros(1), [np.ones((1, 1)), np.ones((1, 1))]))
    second = list(pipeline.stream(np.zeros(1), [np.ones((1, 1))]))

    assert transition.reset_calls == 2
    np.testing.assert_array_equal(first[0].to_numpy(), [[2.0]])
    np.testing.assert_array_equal(first[1].to_numpy(), [[5.0]])
    np.testing.assert_array_equal(second[0].to_numpy(), [[2.0]])

    failing = RolloutPipeline(_StreamErrorTransition())
    stream = failing.stream(np.zeros(1), [np.ones((2, 1))])
    with pytest.raises(ValueError, match="stream transition failed"):
        next(stream)


def test_rollout_pipeline_stream_rejects_hidden_state_without_reset_or_contract() -> None:
    transition = _UnresettableStatefulStreamTransition()
    stream = RolloutPipeline(transition).stream(np.zeros(1), [np.ones((1, 1))])

    with pytest.raises(TypeError, match=r"reset\(\)|stream_state_contract"):
        next(stream)
    assert transition.hidden == 0


def test_rollout_pipeline_stream_producer_errors_propagate_and_close_source() -> None:
    closed = False

    def source():
        nonlocal closed
        try:
            yield np.ones((1, 1))
            raise RuntimeError("producer failed")
        finally:
            closed = True

    stream = RolloutPipeline(_fitted_transition()).stream(np.zeros(2), source())
    next(stream)
    with pytest.raises(RuntimeError, match="producer failed"):
        next(stream)
    assert closed


def test_rollout_pipeline_async_stream_preserves_order_and_does_not_block_event_loop() -> None:
    async def scenario() -> tuple[list[np.ndarray], int, int]:
        transition = _SlowStreamTransition()
        pipeline = RolloutPipeline(transition)
        ticks = 0
        running = True

        async def ticker() -> None:
            nonlocal ticks
            while running:
                ticks += 1
                await asyncio.sleep(0)

        ticker_task = asyncio.create_task(ticker())
        values: list[np.ndarray] = []
        stream = pipeline.stream_async(np.zeros(1), [np.ones((2, 1)), np.ones((1, 1))])
        first_task = asyncio.create_task(stream.__anext__())
        await asyncio.to_thread(transition.started.wait, 1.0)
        ticks_before = ticks
        await asyncio.to_thread(transition.finished.wait, 1.0)
        ticks_during = ticks
        values.append((await first_task).to_numpy())
        async for chunk in stream:
            values.append(chunk.to_numpy())
        running = False
        await ticker_task
        return values, ticks_before, ticks_during

    values, ticks_before, ticks_during = asyncio.run(scenario())
    assert [value.shape for value in values] == [(2, 1), (1, 1)]
    np.testing.assert_array_equal(np.concatenate(values), [[1.0], [2.0], [3.0]])
    assert ticks_during > ticks_before


def test_rollout_pipeline_async_stream_offloads_iterator_and_close() -> None:
    async def scenario() -> tuple[bool, bool, bool]:
        source = _BlockingStreamSource()
        stream = RolloutPipeline(_fitted_transition()).stream_async(np.zeros(2), source)
        next_task = asyncio.create_task(stream.__anext__())
        await asyncio.to_thread(source.iter_started.wait, 1.0)
        assert not next_task.done()
        source.iter_release.set()
        chunk = await next_task
        assert chunk.shape == (1, 2)

        close_task = asyncio.create_task(stream.aclose())
        await asyncio.to_thread(source.close_started.wait, 1.0)
        assert not close_task.done()
        source.close_release.set()
        await close_task
        await asyncio.sleep(0)
        return source.iter_started.is_set(), source.close_finished.is_set(), close_task.done()

    iter_started, close_finished, close_done = asyncio.run(scenario())
    assert iter_started
    assert close_finished
    assert close_done


def test_rollout_pipeline_async_stream_cancellation_settles_worker_and_closes_source() -> None:
    async def scenario() -> tuple[bool, bool]:
        transition = _SlowStreamTransition()
        closed = False

        def source():
            nonlocal closed
            try:
                yield np.ones((1, 1))
            finally:
                closed = True

        pipeline = RolloutPipeline(transition)
        stream = pipeline.stream_async(np.zeros(1), source())
        task = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await stream.aclose()
        return closed, transition.finished.is_set()

    closed, finished = asyncio.run(scenario())
    assert closed
    assert finished


def test_rollout_pipeline_async_stream_cancellation_settles_blocking_close() -> None:
    async def scenario() -> bool:
        source = _BlockingCloseStreamSource()
        transition = _SlowStreamTransition()
        stream = RolloutPipeline(transition).stream_async(np.zeros(1), source)

        async def release_close() -> None:
            await asyncio.to_thread(source.close_started.wait, 1.0)
            source.close_release.set()

        release_task = asyncio.create_task(release_close())
        task = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await release_task
        return source.close_finished.is_set()

    assert asyncio.run(scenario())


def test_rollout_pipeline_async_stream_producer_error_and_consumer_close_cleanup() -> None:
    async def scenario() -> tuple[bool, bool]:
        producer_closed = False
        consumer_closed = False

        def failing_source():
            nonlocal producer_closed
            try:
                yield np.ones((1, 1))
                raise RuntimeError("async producer failed")
            finally:
                producer_closed = True

        def consumer_source():
            nonlocal consumer_closed
            try:
                yield np.ones((1, 1))
                yield np.ones((1, 1))
            finally:
                consumer_closed = True

        pipeline = RolloutPipeline(_fitted_transition())
        producer_stream = pipeline.stream_async(np.zeros(2), failing_source())
        await producer_stream.__anext__()
        with pytest.raises(RuntimeError, match="async producer failed"):
            await producer_stream.__anext__()

        consumer_stream = pipeline.stream_async(np.zeros(2), consumer_source())
        await consumer_stream.__anext__()
        await consumer_stream.aclose()
        return producer_closed, consumer_closed

    producer_closed, consumer_closed = asyncio.run(scenario())
    assert producer_closed
    assert consumer_closed
