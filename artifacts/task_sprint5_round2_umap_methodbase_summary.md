# Task Summary: Sprint 5 Round 2 — UMAP + `_MethodBase` Sketch

**Sprint:** Sprint 5
**Task:** Round 2 — UMAP (Method #2, nonlinear, stochastic, stateful) + unstable `_MethodBase` sketch

## Summary of Work

Implemented UMAP as the second dimensionality-reduction method (nonlinear, stochastic, stateful), wrapping `umap-learn` with numpy public surface. Following the Rule of Three (INCREMENTAL.md §4a), instance #2 triggered the extraction of a tentative shared internal shape `_MethodBase` with `fit`/`transform`/`fit_transform` protocol. PCA was migrated to inherit from `_MethodBase`; UMAP also inherits from it. `_MethodBase` is explicitly marked **UNSTABLE**, internal-only (`_` prefix), not in `__all__`, and NOT exported from the top-level package. Created end-to-end demo script with side-by-side PCA vs UMAP 2D visualization. Added 12 tests covering construction, fit, transform, fit_transform, reproducibility, and error cases. All tooling gates pass clean.

### Rule-of-Three Checkpoint

| Check | Status |
|---|---|
| Method instances | PCA (#1, linear) + UMAP (#2, nonlinear, stochastic) |
| Rule branch | **Instance #2** → sketch shared shape, marked *unstable*, NOT public |
| `_MethodBase` exposure | Internal only (`_` prefix), not in `__all__`, not exported from top-level |
| ADR impact | None — all three ADRs remain `pending` |

## Files Modified

- [pyproject.toml](../pyproject.toml) — Added `umap-learn>=0.5,<1.0` dependency
- [src/latent_anything/methods/\_\_init\_\_.py](../src/latent_anything/methods/__init__.py) — Added UMAP to `__all__`; `_MethodBase` NOT exported
- [src/latent_anything/methods/pca.py](../src/latent_anything/methods/pca.py) — Migrated to inherit from `_MethodBase`, removed duplicate `fit_transform`

## Files Created

- [src/latent_anything/methods/\_base.py](../src/latent_anything/methods/_base.py) — Internal `_MethodBase` base class (UNSTABLE, not public)
- [src/latent_anything/methods/umap.py](../src/latent_anything/methods/umap.py) — UMAP concrete class wrapping `umap-learn`
- [tests/test_latent_anything/test_umap.py](../tests/test_latent_anything/test_umap.py) — 12 tests for UMAP
- [scripts/end_to_end_umap_demo.py](../scripts/end_to_end_umap_demo.py) — Side-by-side PCA vs UMAP visualization

## Testing

- **Test Count:** 53 total (12 new UMAP + 41 existing)
- **Status:** All passed
- **Lint:** `ruff check` — 0 issues in sprint 5 code
- **Format:** `ruff format --check` — 0 issues in sprint 5 code
- **Type Check:** `pyright` strict — 0 errors
- **Execution Command:** `uv run pytest -v`

## Additional Notes

- `umap-learn` installed cleanly without pulling `torch` into the dependency tree (numba-based).
- `random_state` reproducibility verified with dedicated tests.
- `_MethodBase` is intentionally minimal — no `save`/`load` or other abstractions added. Will be replaced/frozen when Method #3 (SAE) lands in Sprint 6.
- PCA's `fit_transform` was identical to `_MethodBase.fit_transform`, so it was removed from PCA in the migration.
