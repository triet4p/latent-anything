# Task Summary: Sprint 49 Task 03 — Immutable outputs with operation/provenance metadata

**Sprint:** Sprint 49
**Task:** Preserve immutable latent values and attach operation/provenance metadata to outputs.

## Summary of Work

All projection/removal/transfer and arithmetic operations build on the existing immutable `LatentValue` contract (defensive array copies, frozen metadata) and return new immutable values rather than mutating inputs. Every output carries an `operation` metadata record (kind, op, basis origin / coefficients / operand identity) and grows a `provenance` chain listing the sequence of operations that produced it. Tests pin that outputs cannot be mutated, that `to_numpy()` returns writable copies, and that the provenance chain extends across chained operations.

## Files Modified

- [src/latent_anything/latent_value.py](src/latent_anything/latent_value.py) - `_arithmetic_metadata` helper that extends metadata and appends a provenance entry.
- [src/latent_anything/projection.py](src/latent_anything/projection.py) - `_transform`/`transfer` attach operation + provenance metadata to projected/removed/transferred values.

## Testing

- **Test File:** [tests/test_latent_anything/test_projection.py](tests/test_latent_anything/test_projection.py), [tests/test_latent_anything/test_latent_arithmetic.py](tests/test_latent_anything/test_latent_arithmetic.py)
- **Status:** Passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_projection.py tests/test_latent_anything/test_latent_arithmetic.py -v`

## Additional Notes

Provenance is append-only and stored as an immutable chain; consumers can trace exactly which operations and basis families produced a value.
