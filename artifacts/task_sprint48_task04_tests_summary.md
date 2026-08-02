# Task Summary: Sprint 48 Task 04 — Analytic/Property Tests

**Sprint:** Sprint 48
**Task:** Add analytic/property tests for affine invariance, singular covariance handling, and distance symmetry.

## Summary of Work

Added 55 tests in `tests/test_latent_anything/test_covariance_geometry.py` covering: positive-definite validation (wrong shape, non-symmetric, non-finite, singular, indefinite), diagonal-loading regularization of singular matrices, Mahalanobis analytic properties (symmetry, zero-on-self, inverse-variance scaling, direction-dependence under anisotropy), affine invariance (`d_M(Aa+t, Ab+t; A C A^T) == d_M(a, b; C)`), whitening round-trips and identity covariance of whitened samples, empirical fitting constraints, the declared interpolation semantics, `CovarianceConfig` validation, provenance-bound fitting, JSON/`.npz` serialization round-trips, and the full `LatentSpace(geometry="anisotropic")` facade dispatch. Hypothesis property tests back symmetry and the constant-metric-geodesic-equals-lerp invariant. Added an anisotropic conformance case to `test_latent_space.py`.

## Files Modified

- [tests/test_latent_anything/test_covariance_geometry.py](tests/test_latent_anything/test_covariance_geometry.py) - New test module.
- [tests/test_latent_anything/test_latent_space.py](tests/test_latent_anything/test_latent_space.py) - Added anisotropic conformance matrix test.
- [tests/test_api_surface.py](tests/test_api_surface.py) - Snapshot updated for new exports.

## Testing

- **Test File:** [tests/test_latent_anything/test_covariance_geometry.py](tests/test_latent_anything/test_covariance_geometry.py)
- **Status:** Passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_covariance_geometry.py tests/test_latent_anything/test_latent_space.py tests/test_api_surface.py -q`

## Additional Notes

Property tests use `hypothesis` for the core geometry primitive, matching project convention for `LatentSpace`-level invariants.
