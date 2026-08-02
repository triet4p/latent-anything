# Task Summary: Sprint 49 Task 02 — Coordinate-system-gated latent arithmetic

**Sprint:** Sprint 49
**Task:** Implement latent arithmetic only for values proven to share space identity, geometry, shape, and source-model revision.

## Summary of Work

Added `LatentValue` arithmetic (`add`, `subtract`, `add_scaled`, `scale`, and the `+`/`-` operators) that is only permitted when `assert_arithmetic_compatible` proves both operands share a coordinate system: same geometry, same point shape, same stored shape, and a matching, declared canonical identity. Added the `coordinate_identity` canonical identity (from `source_representation_identity`, `source_model`, and revision metadata) and a `LatentValue.identity` property. Cross-system arithmetic — different model, different revision, mismatched shape/geometry, or an undeclared identity — raises `ValueError` rather than silently returning a plausible-looking array.

## Files Modified

- [src/latent_anything/latent_value.py](src/latent_anything/latent_value.py) - Added `coordinate_identity`, `assert_arithmetic_compatible`, `LatentValue.identity`, and the arithmetic operations.

## Testing

- **Test File:** [tests/test_latent_anything/test_latent_arithmetic.py](tests/test_latent_anything/test_latent_arithmetic.py)
- **Status:** Passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_latent_arithmetic.py -v`

## Additional Notes

An undeclared identity (empty `source_model` and no identity metadata) is rejected because the framework cannot prove the two spaces are the same coordinate system. Existing adapters that set `source_model` (and revision metadata where available) participate without modification.
