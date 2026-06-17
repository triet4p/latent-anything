# Task Summary: Sprint 4 — Round 1 Core Primitives

**Sprint:** Sprint 4
**Task:** Round 1 — `LatentSpace` (euclidean flat) + `Trajectory` (immutable) + PCA method + end-to-end demo

## Summary of Work

Implemented the first concrete instances of three core primitives, hardcoded (no interfaces extracted per Rule of Three §4a — Instance #1):

- **`LatentSpace`** — Euclidean flat vector space with `dim`, `geometry="euclidean"`, optional `source_model` and `metadata`. Public surface is numpy. Has `validate_point()` and `shape` property.
- **`Trajectory`** — Immutable sequence of latent states (2D numpy array). Supports `len`, `dim`, `shape`, integer/slice indexing (returns new `Trajectory`), and `to_numpy()` (returns copy). Single-point trajectories are valid.
- **PCA (method)** — Stateful dimensionality reduction wrapping scikit-learn. `fit`/`transform`/`fit_transform` with numpy in/out. Exposes `components_`, `explained_variance_ratio_`, `mean_` after fitting.
- **End-to-end demo script** (`scripts/end_to_end_pca_demo.py`) — generates synthetic 8D latent data → packs into `Trajectory` → fits PCA → projects to 2D → visualizes with matplotlib.

## Files Modified

- `src/latent_anything/__init__.py` — Added `LatentSpace` and `Trajectory` to public exports.
- `pyproject.toml` — Added `scikit-learn` and `matplotlib` as runtime dependencies.

## Files Created

- `src/latent_anything/latent_space.py` — `LatentSpace` concrete class.
- `src/latent_anything/trajectory.py` — `Trajectory` immutable sequence class.
- `src/latent_anything/methods/__init__.py` — Methods package.
- `src/latent_anything/methods/pca.py` — PCA method wrapping sklearn.
- `scripts/end_to_end_pca_demo.py` — End-to-end demo: synthetic latent → Trajectory → PCA → 2D plot.
- `tests/test_latent_anything/test_latent_space.py` — 11 tests for LatentSpace.
- `tests/test_latent_anything/test_trajectory.py` — 14 tests (incl. hypothesis property-based) for Trajectory.
- `tests/test_latent_anything/test_pca.py` — 10 tests for PCA.

## Testing

- **Test Count:** 41 tests (11 LatentSpace + 14 Trajectory + 10 PCA + 6 package smoke)
- **Status:** All passed
- **Execution Command:** `uv run pytest tests/ -v`

## Rule of Three Check (§4a)

| Abstraction | Instance count | Decision |
|---|---|---|
| `Method` | 1 (PCA) | **Keep hardcoded** — no interface extracted. |
| `LatentSpace` | 1 (euclidean) | **Keep hardcoded** — concrete class only. |

Per INCREMENTAL.md §4a, with exactly 1 instance each, all abstractions stay hardcoded. No `Protocol`/`ABC` created. Method interface extraction deferred until PCA → UMAP → SAE (instance #3, Sprint 6).

## ADR Reconciliation (§4c)

All three 2026-06-16 ADRs (`LatentSpace` geometry-keyed, `ModelAdapter` 3-mode, geometry-dispatch) remain **`pending`** — this increment touches only the simplest euclidean flat case and does not exercise the geometry keying or dispatch logic. No ADR is confirmed or refuted yet.

## Additional Notes

- The `Trajectory` immutable design is per ARCHITECTURE §7 — all operations return new instances.
- PyTorch is not used anywhere in these primitives as planned.
- PCA uses sklearn internally but all public signatures are numpy.
- Demo script uses PEP 723 inline script metadata for standalone execution.
