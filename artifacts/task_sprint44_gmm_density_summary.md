# Sprint 44 — Gaussian-mixture density and calibrated OOD summary

## Summary

Added `GaussianMixtureDensity`, typed density/evaluation/stability results, held-out empirical calibration, deterministic cross-seed evaluation, a single-Gaussian Mahalanobis baseline, and explicit geometry, dimension, sample-count, covariance, and cross-space validation.

## Files

- `src/latent_anything/density.py` — estimator, result contracts, metrics, baseline, and uncertainty.
- `tests/test_density.py` — offline calibration, OOD, failure, registry, determinism, and state-snapshot coverage.
- `src/latent_anything/__init__.py` and `_plugin_builtins.py` — public exports and config/registry construction.
- `docs/sprint-plans/sprint-44.md` — all eight atomic tasks completed.

## Verification

- `uv run pytest tests/test_density.py tests/test_api_surface.py tests/test_latent_anything/test_registry.py -q` — passed.
- `uv run pytest -q` — 912 passed, 23 skipped (optional network tests), 39 warnings.
- `uv run ruff check src tests` — passed.
- `uv run ruff format --check src tests` — passed.
- `uv run pyright` — passed.

## Notes

Fit, calibration, and evaluation arrays are separate arguments. The estimator accepts only flat Euclidean or unit-norm representations and binds every score to the identity supplied at fit time.
