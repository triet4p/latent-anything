# Task Summary: Sprint 49 Task 06 — Projection-basis comparison

**Sprint:** Sprint 49
**Task:** Compare projection bases from PCA, probe coefficients, and concept directions without treating them as interchangeable.

## Summary of Work

Added `scripts/projection_basis_comparison.py`. It derives three orthonormal subspaces from the same real ConvVAE digits latents — PCA top component (`origin="pca"`), supervised probe coefficients (`origin="probe"`), and mean-diff concept direction (`origin="concept"`) — each recorded with its origin in an `OrthonormalSubspace` so the framework never swaps families silently. Pairwise principal-angle alignment (0.353 / 0.676 / 0.765) shows the bases are not interchangeable, and removal effects differ sharply: PCA removal preserves the target concept (0.722) while the supervised probe and concept directions suppress it (0.289 / 0.306), demonstrating that variance is not semantics. Acceptance criteria pass; results are written to `artifacts/projection_basis_comparison.json`.

## Files Modified

- [scripts/projection_basis_comparison.py](scripts/projection_basis_comparison.py) - New benchmark (added to pyright include).
- [artifacts/projection_basis_comparison.json](artifacts/projection_basis_comparison.json) - Reproducible artifact.

## Testing

- **Execution Command:** `uv run python scripts/projection_basis_comparison.py`
- **Status:** Passed (all acceptance criteria met)

## Additional Notes

Provides the second D2 benchmark role for the `subspace projection` theory topic and motivates the `origin` field on `OrthonormalSubspace`.
