"""Tests for InMemoryCache (Sprint 23 — Runtime #2)."""

from __future__ import annotations

import numpy as np

from latent_anything.latent_space import LatentSpace
from latent_anything.pipeline import AnalysisPipeline
from latent_anything.runtime import CacheKey, InMemoryCache, hash_array, hash_component_config, make_cache_key


class CountingAdapter:
    """Small ModelAdapter test double with observable encode calls."""

    def __init__(self, *, input_dim: int = 4, latent_dim: int = 4, scale: float = 2.0) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.scale = scale
        self.encode_calls = 0

    @property
    def latent_space(self) -> LatentSpace:
        return LatentSpace(dim=self.latent_dim, source_model="counting_adapter")

    def encode(self, data: np.ndarray) -> np.ndarray:
        self.encode_calls += 1
        return data[:, : self.latent_dim] * self.scale


class CountingMethod:
    """Small Method test double with observable fit/transform calls."""

    def __init__(self, *, offset: float = 1.0) -> None:
        self.offset = offset
        self.fit_calls = 0
        self.transform_calls = 0
        self._fitted = False

    def fit(self, data: np.ndarray) -> None:
        _ = data
        self.fit_calls += 1
        self._fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            msg = "CountingMethod must be fitted before transform"
            raise RuntimeError(msg)
        self.transform_calls += 1
        return data + self.offset


def synthetic_data() -> np.ndarray:
    """Return deterministic input data."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(8, 4)).astype(np.float64)


class TestCacheKey:
    """Cache key structure and invalidation hashes."""

    def test_hash_array_changes_when_data_changes(self) -> None:
        """Input data changes produce different data hashes."""
        data = synthetic_data()
        changed = data.copy()
        changed[0, 0] += 1.0
        assert hash_array(data) != hash_array(changed)

    def test_hash_array_includes_shape(self) -> None:
        """Same bytes with different shape produce different hashes."""
        flat = np.arange(8, dtype=np.float64)
        matrix = flat.reshape(2, 4)
        assert hash_array(flat) != hash_array(matrix)

    def test_hash_component_config_changes_with_adapter_config(self) -> None:
        """Adapter config changes produce different config hashes."""
        adapter_a = CountingAdapter(scale=2.0)
        adapter_b = CountingAdapter(scale=3.0)
        assert hash_component_config(adapter_a) != hash_component_config(adapter_b)

    def test_hash_component_config_excludes_runtime_counters(self) -> None:
        """Runtime counters do not change the config hash."""
        adapter = CountingAdapter(scale=2.0)
        before = hash_component_config(adapter)
        adapter.encode(synthetic_data())
        after = hash_component_config(adapter)
        assert after == before

    def test_make_cache_key_contains_required_fields(self) -> None:
        """CacheKey records operation, component, hashes, and version field."""
        adapter = CountingAdapter(scale=2.0)
        key = make_cache_key(
            namespace="test",
            operation="adapter.encode",
            component=adapter,
            data=synthetic_data(),
            framework_version="0.test",
        )
        assert isinstance(key, CacheKey)
        assert key.namespace == "test"
        assert key.operation == "adapter.encode"
        assert key.component_name.endswith("CountingAdapter")
        assert len(key.config_hash) == 64
        assert len(key.data_hash) == 64
        assert key.framework_version == "0.test"


class TestInMemoryCache:
    """InMemoryCache backend behavior."""

    def test_get_miss_updates_stats(self) -> None:
        """Missing keys return None and increment misses."""
        cache = InMemoryCache()
        key = make_cache_key(namespace="test", operation="op", component=CountingAdapter(), data=synthetic_data())
        assert cache.get(key) is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0
        assert cache.stats.size == 0

    def test_set_then_get_returns_copy_and_updates_stats(self) -> None:
        """Cache get returns a defensive copy."""
        cache = InMemoryCache()
        key = make_cache_key(namespace="test", operation="op", component=CountingAdapter(), data=synthetic_data())
        value = np.ones((2, 3), dtype=np.float64)
        cache.set(key, value)
        value[0, 0] = 99.0

        cached = cache.get(key)

        assert cached is not None
        np.testing.assert_array_equal(cached, np.ones((2, 3), dtype=np.float64))
        cached[0, 0] = 42.0
        cached_again = cache.get(key)
        assert cached_again is not None
        assert cached_again[0, 0] == 1.0
        assert cache.stats.hits == 2
        assert cache.stats.sets == 1
        assert cache.stats.size == 1

    def test_clear_removes_entries_and_resets_stats(self) -> None:
        """clear() removes cached values and resets counters."""
        cache = InMemoryCache()
        key = make_cache_key(namespace="test", operation="op", component=CountingAdapter(), data=synthetic_data())
        cache.set(key, np.ones((2, 2), dtype=np.float64))
        cache.get(key)

        cache.clear()

        assert len(cache) == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
        assert cache.stats.sets == 0
        assert cache.stats.size == 0


class TestAnalysisPipelineCache:
    """AnalysisPipeline cache integration."""

    def test_pipeline_cache_hit_reuses_encode_and_method_output(self) -> None:
        """Repeated identical run hits cache for adapter encode and method output."""
        cache = InMemoryCache()
        adapter = CountingAdapter(scale=2.0)
        method = CountingMethod(offset=1.0)
        pipeline = AnalysisPipeline(adapter=adapter, method=method, cache=cache)
        data = synthetic_data()

        first = pipeline.run(data)
        second = pipeline.run(data)

        np.testing.assert_array_equal(second.latents, first.latents)
        np.testing.assert_array_equal(second.transformed, first.transformed)
        assert adapter.encode_calls == 1
        assert method.fit_calls == 1
        assert method.transform_calls == 1
        assert cache.stats.hits == 2
        assert cache.stats.misses == 2
        assert cache.stats.sets == 2
        assert cache.stats.size == 2

    def test_pipeline_cache_invalidates_when_data_changes(self) -> None:
        """Different input data misses the cache and recomputes."""
        cache = InMemoryCache()
        adapter = CountingAdapter(scale=2.0)
        method = CountingMethod(offset=1.0)
        pipeline = AnalysisPipeline(adapter=adapter, method=method, cache=cache)
        data = synthetic_data()
        changed = data.copy()
        changed[0, 0] += 1.0

        pipeline.run(data)
        pipeline.run(changed)

        assert adapter.encode_calls == 2
        assert method.fit_calls == 2
        assert method.transform_calls == 2
        assert cache.stats.misses == 4
        assert cache.stats.size == 4

    def test_pipeline_cache_invalidates_when_adapter_config_changes(self) -> None:
        """Different adapter config produces a distinct encode cache key."""
        cache = InMemoryCache()
        data = synthetic_data()
        method_a = CountingMethod(offset=1.0)
        method_b = CountingMethod(offset=1.0)
        adapter_a = CountingAdapter(scale=2.0)
        adapter_b = CountingAdapter(scale=3.0)

        AnalysisPipeline(adapter=adapter_a, method=method_a, cache=cache).run(data)
        AnalysisPipeline(adapter=adapter_b, method=method_b, cache=cache).run(data)

        assert adapter_a.encode_calls == 1
        assert adapter_b.encode_calls == 1
        assert cache.stats.misses == 4

    def test_pipeline_cache_invalidates_when_method_config_changes(self) -> None:
        """Different method config produces a distinct method-output cache key."""
        cache = InMemoryCache()
        data = synthetic_data()
        adapter = CountingAdapter(scale=2.0)
        method_a = CountingMethod(offset=1.0)
        method_b = CountingMethod(offset=2.0)

        first = AnalysisPipeline(adapter=adapter, method=method_a, cache=cache).run(data)
        second = AnalysisPipeline(adapter=adapter, method=method_b, cache=cache).run(data)

        assert adapter.encode_calls == 1
        assert method_a.fit_calls == 1
        assert method_b.fit_calls == 1
        assert cache.stats.hits == 1
        assert cache.stats.misses == 3
        assert not np.array_equal(first.transformed, second.transformed)

    def test_pipeline_cache_returned_arrays_do_not_mutate_cached_values(self) -> None:
        """Mutating returned arrays does not mutate cached arrays."""
        cache = InMemoryCache()
        adapter = CountingAdapter(scale=2.0)
        method = CountingMethod(offset=1.0)
        pipeline = AnalysisPipeline(adapter=adapter, method=method, cache=cache)
        data = synthetic_data()

        first = pipeline.run(data)
        expected_latents = first.latents.copy()
        expected_transformed = first.transformed.copy()
        first.latents[0, 0] = 999.0
        first.transformed[0, 0] = 999.0
        second = pipeline.run(data)

        np.testing.assert_array_equal(second.latents, expected_latents)
        np.testing.assert_array_equal(second.transformed, expected_transformed)
