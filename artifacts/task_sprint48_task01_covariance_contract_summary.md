# Task Summary: Sprint 48 Task 01 — Anisotropic Metadata & Covariance Ownership Contract

**Sprint:** Sprint 48
**Task:** Define the anisotropic latent-space metadata and covariance ownership/fitting contract.

## Summary of Work

Added the stateful covariance geometry contract: `geometry="anisotropic"` on `LatentSpace`, an immutable `CovarianceState` (mean, covariance, n_samples, source representation identity, reg_coef, provenance), a pydantic `CovarianceConfig`, and a `fit_covariance_state()` entry point that binds a fitted metric to a required `source_representation_identity`. `LatentSpace.fit_covariance()` mutates the space, validates dim/PD, and records `covariance_fitted`, `covariance_source_representation_identity`, `covariance_provenance`, and `interpolation="metric-geodesic"` metadata.

## Files Modified

- [src/latent_anything/covariance.py](src/latent_anything/covariance.py) - New module: `CovarianceConfig`, `CovarianceState`, `fit_covariance_state`.
- [src/latent_anything/geometry.py](src/latent_anything/geometry.py) - Added pure covariance functions used by the contract.
- [src/latent_anything/latent_space.py](src/latent_anything/latent_space.py) - Added `anisotropic` geometry, `covariance` property, `fit_covariance`, `_attach_covariance`, metadata.
- [src/latent_anything/__init__.py](src/latent_anything/__init__.py) - Exported `CovarianceConfig`, `CovarianceState`, `fit_covariance_state`.

## Testing

- **Test File:** [tests/test_latent_anything/test_covariance_geometry.py](tests/test_latent_anything/test_covariance_geometry.py)
- **Status:** Passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_covariance_geometry.py -q`

## Additional Notes

The ownership contract mirrors the Sprint 44 density ADR: a learned metric is stateful and must be bound to the dataset/model version, preventing silent cross-space reuse.
