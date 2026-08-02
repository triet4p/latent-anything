# Task Summary: Sprint 48 Task 07 — Serialization/Config for Fitted Covariance Provenance

**Sprint:** Sprint 48
**Task:** Add serialization/config support for fitted covariance provenance.

## Summary of Work

Added portable serialization and config support for the fitted covariance geometry: `CovarianceState.to_dict()`/`from_dict()` (JSON-friendly with arrays as nested lists, full provenance) and `.npz` `save()`/`load()` (arrays plus JSON metadata), following the SAE checkpoint pattern. `CovarianceConfig` (pydantic) validates `reg_coef` and `min_samples_per_dimension` and drives `fit_covariance_state`/`LatentSpace.fit_covariance`, giving deterministic, validated, reproducible fitting. Both types are exported from the package public surface.

## Files Modified

- [src/latent_anything/covariance.py](src/latent_anything/covariance.py) - `to_dict`/`from_dict`, `save`/`load`, `CovarianceConfig`.
- [src/latent_anything/__init__.py](src/latent_anything/__init__.py) - Public exports.
- [tests/test_api_surface.py](tests/test_api_surface.py) - Snapshot updated.

## Testing

- **Test File:** [tests/test_latent_anything/test_covariance_geometry.py](tests/test_latent_anything/test_covariance_geometry.py)
- **Status:** Passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_covariance_geometry.py -q`

## Additional Notes

Serialization preserves the source representation identity and provenance so a fitted metric can be rebuilt reproducibly without losing its binding contract.
