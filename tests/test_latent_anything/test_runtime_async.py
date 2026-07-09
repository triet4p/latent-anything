"""Tests for Sprint 24 async runtime wrappers and profiling hooks."""

from __future__ import annotations

import asyncio

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.pipeline import AnalysisPipeline, ManipulationPipeline
from latent_anything.runtime import BatchExecutor, InMemoryCache, RuntimeProfiler
from latent_anything.trajectory import Trajectory


class SimpleFlatBatchAdapter:
    """Small flat-batch adapter test double for runtime tests."""

    def __init__(self, *, input_dim: int = 4, latent_dim: int = 3, scale: float = 2.0, bias: float = -1.0) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.scale = scale
        self.bias = bias

    @property
    def latent_space(self) -> LatentSpace:
        return LatentSpace(dim=self.latent_dim, source_model="simple_runtime_adapter")

    @property
    def supports_flat_batch(self) -> bool:
        return True

    def encode(self, data: np.ndarray) -> np.ndarray:
        return data[:, : self.latent_dim] * self.scale

    def decode(self, latent: np.ndarray) -> np.ndarray:
        return latent + self.bias


class OffsetMethod:
    """Small Layer A method double with fit/transform state."""

    def __init__(self, *, offset: float = 0.5) -> None:
        self.offset = offset
        self._fitted = False

    def fit(self, data: np.ndarray) -> None:
        _ = data
        self._fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            msg = "OffsetMethod must be fitted before transform"
            raise RuntimeError(msg)
        return data + self.offset


class SimplePatchMethod:
    """Small adapter-mediated method double with latent patching."""

    def __init__(self, *, delta: np.ndarray) -> None:
        self._delta = delta

    @property
    def space(self) -> LatentSpace | None:
        return None

    @property
    def is_fitted(self) -> bool:
        return True

    def apply_latent(self, latent: np.ndarray) -> np.ndarray:
        return latent + self._delta

    def __call__(self, data: np.ndarray) -> np.ndarray:
        _ = data
        msg = "__call__ should not be used when apply_latent profiling path is available"
        raise RuntimeError(msg)

    def apply_trajectory(self, trajectory: Trajectory, **kwargs: object) -> Trajectory:
        _ = kwargs
        return Trajectory(data=self.apply_latent(trajectory.to_numpy()))


def synthetic_data() -> np.ndarray:
    """Return deterministic input data for runtime tests."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(12, 4)).astype(np.float64)


class TestAsyncParity:
    """Async wrappers should match the existing sync paths exactly."""

    def test_analysis_pipeline_run_async_matches_sync(self) -> None:
        """AnalysisPipeline.run_async returns the same result as run."""
        adapter = SimpleFlatBatchAdapter()
        method = OffsetMethod(offset=0.25)
        pipeline = AnalysisPipeline(adapter=adapter, method=method)
        data = synthetic_data()

        sync_result = pipeline.run(data)

        async_pipeline = AnalysisPipeline(adapter=SimpleFlatBatchAdapter(), method=OffsetMethod(offset=0.25))
        async_result = asyncio.run(async_pipeline.run_async(data))

        np.testing.assert_array_equal(async_result.latents, sync_result.latents)
        np.testing.assert_array_equal(async_result.transformed, sync_result.transformed)
        assert async_result.latent_space.dim == sync_result.latent_space.dim
        assert async_result.latent_space.geometry == sync_result.latent_space.geometry
        assert async_result.latent_space.source_model == sync_result.latent_space.source_model

    def test_batch_executor_async_paths_match_sync(self) -> None:
        """BatchExecutor async encode/decode/transform match sync outputs."""
        adapter = SimpleFlatBatchAdapter()
        method = OffsetMethod(offset=1.5)
        executor = BatchExecutor(batch_size=5)
        data = synthetic_data()
        latent = adapter.encode(data)
        method.fit(latent)

        sync_encode = executor.encode(adapter, data)
        async_encode = asyncio.run(executor.encode_async(adapter, data))
        np.testing.assert_array_equal(async_encode, sync_encode)

        sync_decode = executor.decode(adapter, latent)
        async_decode = asyncio.run(executor.decode_async(adapter, latent))
        np.testing.assert_array_equal(async_decode, sync_decode)

        sync_transform = executor.transform(method, latent)
        async_transform = asyncio.run(executor.transform_async(method, latent))
        np.testing.assert_array_equal(async_transform, sync_transform)

    def test_manipulation_pipeline_run_data_async_matches_sync(self) -> None:
        """ManipulationPipeline.run_data_async matches the sync staged path."""
        adapter = SimpleFlatBatchAdapter()
        method = SimplePatchMethod(delta=np.array([0.2, -0.1, 0.4], dtype=np.float64))
        pipeline = ManipulationPipeline(method=method, adapter=adapter)
        data = synthetic_data()

        sync_output = pipeline.run_data(data)
        async_output = asyncio.run(pipeline.run_data_async(data))

        np.testing.assert_array_equal(async_output, sync_output)


class TestRuntimeProfiling:
    """Profiling hooks capture the expected stage names and durations."""

    def test_profiling_records_stage_names_and_non_negative_durations(self) -> None:
        """Profiling covers cache, encode, method, and decode with non-negative timings."""
        data = synthetic_data()

        analysis_profiler = RuntimeProfiler()
        analysis_pipeline = AnalysisPipeline(
            adapter=SimpleFlatBatchAdapter(),
            method=OffsetMethod(offset=0.75),
            cache=InMemoryCache(),
        )
        analysis_pipeline.run(data, profiler=analysis_profiler)
        analysis_profile = analysis_profiler.snapshot()

        manipulation_profiler = RuntimeProfiler()
        manipulation_pipeline = ManipulationPipeline(
            method=SimplePatchMethod(delta=np.array([0.1, 0.2, -0.3], dtype=np.float64)),
            adapter=SimpleFlatBatchAdapter(),
        )
        manipulation_pipeline.run_data(data, profiler=manipulation_profiler)
        manipulation_profile = manipulation_profiler.snapshot()

        all_events = analysis_profile.events + manipulation_profile.events
        stage_names = {event.stage for event in all_events}

        assert {"cache", "encode", "method", "decode"} <= stage_names
        assert all(event.duration_seconds >= 0.0 for event in all_events)
        assert analysis_profile.total_seconds >= 0.0
        assert manipulation_profile.total_seconds >= 0.0
        assert "cache" in analysis_profile.stage_totals()
        assert {"encode", "method", "decode"} <= set(manipulation_profile.stage_totals())
