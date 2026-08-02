# Task Summary: Sprint 48 Task 03 — Interpolation Semantics Under Anisotropy

**Sprint:** Sprint 48
**Task:** Decide and document interpolation semantics instead of silently applying Euclidean lerp under anisotropy.

## Summary of Work

Decided and documented the interpolation semantics for `geometry="anisotropic"`. Under a **constant** covariance the metric is flat, so the geodesic between two points is the affine segment `(1-t)a + t b` — numerically identical to raw-coordinate lerp. Rather than silently applying Euclidean lerp, interpolation is implemented in `covariance_interpolate` as `unwhiten((1-t)whiten(a) + t whiten(b))`, routed through the declared metric: it requires a fitted covariance (metric ops raise on an unfitted space), validates endpoints, and leaves a clean seam for a future position-dependent (pullback/density-aware) metric in Sprint 50. The equality with affine lerp is documented, not hidden. Recorded as an ADR entry.

## Files Modified

- [src/latent_anything/geometry.py](src/latent_anything/geometry.py) - Added `covariance_interpolate` with the semantics decision in its docstring.
- [src/latent_anything/latent_space.py](src/latent_anything/latent_space.py) - Dispatched `interpolate` for anisotropic; metadata `interpolation="metric-geodesic"`.
- [.agents/memory/decisions.md](.agents/memory/decisions.md) - ADR for anisotropic geometry + interpolation semantics.

## Testing

- **Test File:** [tests/test_latent_anything/test_covariance_geometry.py](tests/test_latent_anything/test_covariance_geometry.py)
- **Status:** Passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_covariance_geometry.py -q`

## Additional Notes

Tests prove the constant-metric geodesic equals affine lerp (documented equality) and that interpolation on an unfitted anisotropic space raises instead of silently falling back to Euclidean.
