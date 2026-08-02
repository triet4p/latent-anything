# Task Summary: Sprint 49 Task 04 — Analytic/property tests

**Sprint:** Sprint 49
**Task:** Add analytic tests for idempotence, orthogonality, reconstruction, and invalid cross-space operations.

## Summary of Work

Added 64 focused tests. `test_projection.py` covers basis validation (orthonormality, rank, finiteness), QR orthonormalization, projection idempotence (`P P z = P z`), component orthogonality (`P z ⟂ (I-P) z`), reconstruction (`P z + (I-P) z = z`), subspace coverage, subspace alignment, `OrthonormalSubspace` serialization round-trips and derivation families, and rejection of cross-identity / wrong-shape / non-euclidean inputs. `test_latent_arithmetic.py` covers the canonical identity, geometry/shape/identity/revision rejection, add/subtract/add_scaled/scale correctness, batch arithmetic, dunder operators, immutable outputs, and the "reject, don't guess" contract for cross-space arithmetic.

## Files Modified

- [tests/test_latent_anything/test_projection.py](tests/test_latent_anything/test_projection.py) - New analytic/property test suite.
- [tests/test_latent_anything/test_latent_arithmetic.py](tests/test_latent_anything/test_latent_arithmetic.py) - New arithmetic compatibility test suite.
- [tests/test_api_surface.py](tests/test_api_surface.py), [tests/test_latent_anything/test_registry.py](tests/test_latent_anything/test_registry.py), [tests/test_latent_anything/test_demo_smoke.py](tests/test_latent_anything/test_demo_smoke.py) - Updated snapshots for new exports and the `subspace_projection` registry entry.

## Testing

- **Test File:** [tests/test_latent_anything/test_projection.py](tests/test_latent_anything/test_projection.py), [tests/test_latent_anything/test_latent_arithmetic.py](tests/test_latent_anything/test_latent_arithmetic.py)
- **Status:** Passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_projection.py tests/test_latent_anything/test_latent_arithmetic.py -v`

## Additional Notes

All tests can fail when the underlying behavior is wrong (they assert exact mathematical identities, not tautologies).
