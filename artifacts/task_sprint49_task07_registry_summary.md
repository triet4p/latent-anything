# Task Summary: Sprint 49 Task 07 — Registry/config support under canonical transformation vocabulary

**Sprint:** Sprint 49
**Task:** Add registry/config support under canonical transformation vocabulary.

## Summary of Work

Registered `SubspaceProjection` under the canonical `intervention` registry kind as `subspace_projection` (the RFC 0001 term for "deliberately transforms a latent representation"), with a pydantic `SubspaceProjectionConfig` (`n_basis`) so the transformation is constructible via `ObjectSpec` / `build_from_config` / `build_from_dict`. Added top-level exports (`OrthonormalSubspace`, `SubspaceProjection`, `SubspaceProjectionConfig`, `coordinate_identity`, `assert_arithmetic_compatible`) and updated the API-surface snapshot and registry/demo-smoke counts accordingly.

## Files Modified

- [src/latent_anything/_plugin_builtins.py](src/latent_anything/_plugin_builtins.py) - Registered `subspace_projection` under `intervention`.
- [src/latent_anything/__init__.py](src/latent_anything/__init__.py) - New public exports + `__all__`.
- [tests/test_api_surface.py](tests/test_api_surface.py), [tests/test_latent_anything/test_registry.py](tests/test_latent_anything/test_registry.py), [tests/test_latent_anything/test_demo_smoke.py](tests/test_latent_anything/test_demo_smoke.py) - Updated snapshots/counts.

## Testing

- **Execution Command:** `uv run pytest tests/test_latent_anything/test_projection.py tests/test_api_surface.py tests/test_latent_anything/test_registry.py tests/test_latent_anything/test_demo_smoke.py -v`
- **Status:** Passed

## Additional Notes

No new registry kind was introduced: projection is a latent transformation and belongs under the canonical `intervention` vocabulary per RFC 0001.
