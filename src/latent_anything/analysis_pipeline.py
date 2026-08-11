"""Pipeline #1: adapter encoding followed by a Layer A analysis method."""

from __future__ import annotations

import asyncio
from time import perf_counter

import numpy as np

from latent_anything.adapters.protocols import ModelAdapter
from latent_anything.latent_space import LatentSpace
from latent_anything.methods.protocols import Method
from latent_anything.pipeline_contract import PipelineContract, PipelineKind
from latent_anything.pipeline_models import PipelineResult
from latent_anything.runtime.cache import CacheKey, InMemoryCache, make_cache_key
from latent_anything.runtime.profiling import RuntimeProfiler


class AnalysisPipeline(PipelineContract):
    """Compose ``adapter.encode`` with ``method.fit`` and ``method.transform``."""

    pipeline_kind: PipelineKind = "analysis"

    def __init__(self, adapter: ModelAdapter, method: Method, cache: InMemoryCache | None = None) -> None:
        self.adapter = adapter
        self.method = method
        self.cache = cache
        self._latent_space = adapter.latent_space

    @property
    def latent_space(self) -> LatentSpace:
        """Return the adapter's latent-space descriptor."""

        return self._latent_space

    def run(self, data: np.ndarray, *, profiler: RuntimeProfiler | None = None) -> PipelineResult:
        """Encode, fit the analysis method, and transform the same batch."""

        latents = (
            self._cached_encode(data, profiler=profiler) if self.cache is not None else self._encode(data, profiler)
        )
        transformed = self._fit_transform(latents, profiler)
        return PipelineResult(latents=latents, transformed=transformed, latent_space=self._latent_space)

    async def run_async(self, data: np.ndarray, *, profiler: RuntimeProfiler | None = None) -> PipelineResult:
        """Run the same stages using cancellable thread-backed async wrappers."""

        latents = (
            await self._cached_encode_async(data, profiler=profiler)
            if self.cache is not None
            else await self._encode_async(data, profiler=profiler)
        )
        transformed = await self._fit_transform_async(latents, profiler=profiler)
        return PipelineResult(latents=latents, transformed=transformed, latent_space=self._latent_space)

    def _encode(self, data: np.ndarray, profiler: RuntimeProfiler | None) -> np.ndarray:
        if profiler is None:
            return self.adapter.encode(data)
        return profiler.measure("encode", lambda: self.adapter.encode(data), component=type(self.adapter).__name__)

    async def _encode_async(self, data: np.ndarray, *, profiler: RuntimeProfiler | None) -> np.ndarray:
        if profiler is None:
            return await asyncio.to_thread(self.adapter.encode, data)
        start = perf_counter()
        latents = await asyncio.to_thread(self.adapter.encode, data)
        profiler.record("encode", perf_counter() - start, component=type(self.adapter).__name__)
        return latents

    def _fit_transform(self, latents: np.ndarray, profiler: RuntimeProfiler | None) -> np.ndarray:
        def operation() -> np.ndarray:
            return self._fit_transform_impl(latents)

        if profiler is None:
            return operation()
        return profiler.measure("method", operation, component=type(self.method).__name__)

    async def _fit_transform_async(self, latents: np.ndarray, *, profiler: RuntimeProfiler | None) -> np.ndarray:
        if profiler is None:
            return await asyncio.to_thread(self._fit_transform_impl, latents)
        start = perf_counter()
        transformed = await asyncio.to_thread(self._fit_transform_impl, latents)
        profiler.record("method", perf_counter() - start, component=type(self.method).__name__)
        return transformed

    def _fit_transform_impl(self, latents: np.ndarray) -> np.ndarray:
        self.method.fit(latents)
        return self.method.transform(latents)

    def _cache_key(self, data: np.ndarray) -> CacheKey:
        return make_cache_key(
            namespace="analysis_pipeline", operation="adapter.encode", component=self.adapter, data=data
        )

    def _cached_encode(self, data: np.ndarray, profiler: RuntimeProfiler | None) -> np.ndarray:
        assert self.cache is not None
        key = self._cache_key(data)
        cached = self._cache_get(key, profiler)
        if cached is not None:
            return cached
        latents = self._encode(data, profiler)
        self._cache_set(key, latents, profiler)
        return latents

    async def _cached_encode_async(self, data: np.ndarray, *, profiler: RuntimeProfiler | None) -> np.ndarray:
        assert self.cache is not None
        key = self._cache_key(data)
        cached = self._cache_get(key, profiler)
        if cached is not None:
            return cached
        latents = await self._encode_async(data, profiler=profiler)
        self._cache_set(key, latents, profiler)
        return latents

    def _cache_get(self, key: CacheKey, profiler: RuntimeProfiler | None) -> np.ndarray | None:
        assert self.cache is not None
        if profiler is None:
            return self.cache.get(key)
        start = perf_counter()
        value = self.cache.get(key)
        profiler.record("cache", perf_counter() - start, operation="get", cache_hit=value is not None)
        return value

    def _cache_set(self, key: CacheKey, value: np.ndarray, profiler: RuntimeProfiler | None) -> None:
        assert self.cache is not None
        if profiler is None:
            self.cache.set(key, value)
            return
        start = perf_counter()
        self.cache.set(key, value)
        profiler.record("cache", perf_counter() - start, operation="set", cache_hit=False)
