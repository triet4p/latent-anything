# Task Summary: Sprint 50 Task 06 — Cache/Profiling Integration

**Sprint:** Sprint 50
**Task:** Add cache/profiling integration because path optimization is expensive.

## Summary of Work

`DensityGeodesic.optimize(a, b, cache=..., profiler=...)` now integrates with the runtime `InMemoryCache` and `RuntimeProfiler`. The cache key binds endpoints (via `hash_array`), the `GeodesicConfig`, and the density oracle's `state_digest`, so a refitted density never reuses a stale path. On a hit the optimized path array is served with a `"served from cache"` status; on a miss the `method` stage is profiled and the result stored. Tests verify cache hits, distinct keys for distinct configs, and both `cache`/`method` stages in the profile.

## Files Modified

- [src/latent_anything/geodesic.py](src/latent_anything/geodesic.py) - `_cache_key`, cache/profiler plumbing in `optimize`.
- [src/latent_anything/density.py](src/latent_anything/density.py) - `GaussianMixtureDensity.state_digest()`.

## Testing

- **Test File:** [tests/test_latent_anything/test_geodesic.py](tests/test_latent_anything/test_geodesic.py)
- **Status:** Passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_geodesic.py -q`

## Additional Notes

The `state_digest` approach follows the Sprint 23/24 lesson that configuration equality alone does not imply behavioral identity.
