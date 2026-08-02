# Task Summary: Sprint 50 Task 03 — Full Path Result with Diagnostics

**Sprint:** Sprint 50
**Task:** Return the full path plus length, density/reconstruction diagnostics, and optimization status.

## Summary of Work

Added the `GeodesicPath` frozen result carrying: the full optimized path `(n_points, dim)` with fixed endpoints, the density-penalized length and the Euclidean length, per-point log-density with `min_log_density`/`mean_log_density` diagnostics, optional decoded images (`decoded`) with a `reconstruction_error` (decoded total variation) when a decoder is attached, a `PathOptimizationStatus` (converged, n_iterations, initial/final energy, message), the source representation identity, and frozen provenance. `GeodesicPath` defensively owns immutable arrays and validates every invariant in `__post_init__`, and exposes `to_dict()` for serialization.

## Files Modified

- [src/latent_anything/geodesic.py](src/latent_anything/geodesic.py) - `GeodesicPath`, `PathOptimizationStatus`, `_decode_rows`, `_decoded_total_variation`.

## Testing

- **Test File:** [tests/test_latent_anything/test_geodesic.py](tests/test_latent_anything/test_geodesic.py)
- **Status:** Passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_geodesic.py -q`

## Additional Notes

The result is immutable like `CovarianceState`/`OrthonormalSubspace`, following the Sprint 48/49 value-object pattern.
