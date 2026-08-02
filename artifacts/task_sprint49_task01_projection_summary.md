# Task Summary: Sprint 49 Task 01 — Projection onto/removal from an orthonormal subspace

**Sprint:** Sprint 49
**Task:** Implement projection onto and removal from one fitted orthonormal subspace.

## Summary of Work

Added the pure projection algorithms to `geometry.py` (`validate_orthonormal_basis`, `orthonormalize_directions`, `project_point` for `P z = U U^T z`, `remove_point` for `(I - P) z`, `concept_coverage`, `subspace_alignment`) and the stateful contract to the new `projection.py`: an immutable `OrthonormalSubspace` value (orthonormal `(dim, n_basis)` basis, bound `source_representation_identity`, explicit `origin`, JSON/`.npz` serialization, derivation helpers for PCA/probe/concept bases) plus the `SubspaceProjection` operation (`project`, `remove`, `coverage`, `transfer`) that consumes and returns immutable `LatentValue` instances, rejecting values whose geometry, shape, or coordinate identity differs from the fitted subspace.

## Files Modified

- [src/latent_anything/geometry.py](src/latent_anything/geometry.py) - Added the focused orthonormal-subspace algorithms (validation, QR orthonormalization, projection/residual, coverage, alignment).
- [src/latent_anything/projection.py](src/latent_anything/projection.py) - New module: `OrthonormalSubspace`, `SubspaceProjectionConfig`, `SubspaceProjection`.

## Testing

- **Test File:** [tests/test_latent_anything/test_projection.py](tests/test_latent_anything/test_projection.py)
- **Status:** Passed
- **Execution Command:** `uv run pytest tests/test_latent_anything/test_projection.py -v`

## Additional Notes

The subspace is bound to the value's canonical `coordinate_identity`; applying it to a value with a different identity (or undeclared identity) raises `ValueError`. Projection is restricted to `euclidean` flat vector spaces.
