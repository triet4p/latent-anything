# Task Summary: Sprint 13 — End-to-End Showcase

**Sprint:** 13 (Round 10 — Composition/Validation)
**Task:** End-to-end showcase: VAE → PCA → ActivationPatch → Decode

## Summary of Work

Sprint 13 is a **composition/validation round** — it adds no new adapter or method instances and freezes no new interfaces. Instead, it composes existing, validated primitives (`LatentSpace` geometry dispatch, `BMethod` Protocol, VAE adapter, PCA, ActivationPatch, Lerp) into the first end-to-end latent-edit story.

The deliverable is:
1. **`scripts/showcase_config.py`** — a lightweight, local config dict (not a framework-wide config system) guaranteeing reproducibility via fixed seed, data generation params, VAE params, split parameters, and output paths.
2. **`scripts/end_to_end_showcase_demo.py`** — the orchestration entry point that implements the full narrative: generate synthetic 4-cluster 8D data → train VAE → encode to 3D latent → PCA Layer A introspection (source/target/failure regions) → baseline metrics (reconstruction MSE, data-space distance to target) → ActivationPatch Layer B edit → post-edit metrics (distance improvement: **68.2%**) → Lerp trajectory panel → composite 2×2 matplotlib figure → console + file summary.
3. **`tests/test_latent_anything/test_showcase.py`** — 18 tests covering config keys, data generation shapes/range/labels, split correctness, baseline metric semantics (finite, positive), post-edit improvement, input non-mutation, PCA projection shapes, and trajectory panel lengths.
4. **Artifacts** — composite figure, metric summary text file, and config snapshot all saved to `artifacts/`.

Key design decisions:
- **No new abstraction.** Config is a local dict, not a `Pipeline` API. Script imports via `sys.path.insert`, not PEP 723 inline metadata (local package not on PyPI).
- **Prefer reuse over copy-paste.** The script reuses existing `_generate_data`, `_split_data`, PCA, ActivationPatch, and Lerp patterns from earlier end-to-end demos rather than duplicating logic.
- **Metric before aesthetics.** The primary success criterion is quantitative: distance to target centroid before vs after edit. Composite figure is secondary.
- **VAE-based, not VLA.** The sprint honestly reflects that no VLA adapter exists in the codebase yet. The story proves the framework composes with the adapter currently available.

## Files Modified

- [scripts/showcase_config.py](scripts/showcase_config.py) — Lightweight config dict for reproducibility.
- [scripts/end_to_end_showcase_demo.py](scripts/end_to_end_showcase_demo.py) — Main showcase orchestration script.
- [tests/test_latent_anything/test_showcase.py](tests/test_latent_anything/test_showcase.py) — 18 tests for showcase helpers.
- [docs/sprint-plans/sprint-13.md](docs/sprint-plans/sprint-13.md) — Updated task statuses to [x] (completed).
- [CHANGELOG.md](CHANGELOG.md) — Added Sprint 13 entries.
- [docs/PLAN.md](docs/PLAN.md) — Marked Sprint 13 active.

## Testing

- **Test File:** `tests/test_latent_anything/test_showcase.py`
- **Status:** 18/18 passed
- **Full suite:** 263 passed (all existing tests + new showcase tests)
- **Execution Command:** `uv run pytest -v`

## Tooling Gate

| Tool | Status |
|---|---|
| `ruff check` | Clean |
| `ruff format` | Clean |
| `pyright` (strict, `src/`) | 0 errors |
| `pytest` | 263 passed |

## Sprint 13 Rule-of-Three Check

| Check | Status |
|---|---|
| New method instance? | **No** — only composes PCA, ActivationPatch, Lerp |
| New adapter instance? | **No** — uses VAE as primary adapter |
| Rule branch | **Composition round** → no extract/freeze |
| Public API change? | **None** — config is local artifact, not in `src/` |
| ADR impact | `ModelAdapter` 3-mode ADR still `pending`; both validated ADRs exercised |

## Metrics (reproducible, seed=42)

| Metric | Before | After | Improvement |
|---|---|---|---|
| Dist to target centroid | 1.2102 | 0.3854 | **68.2%** |
| Recon MSE (failure) | 0.1315 | — | — |

## Additional Notes

- This is the **first composition story** of the latent-anything framework, proving that primitives validated in isolation (Sprints 4–12) compose into a coherent end-to-end narrative.
- The real VLA showcase remains future ecosystem work (when a real VLA adapter exists in the codebase).
- Script uses `sys.path.insert` to import from `src/` because `latent-anything` is a local package not on PyPI (PEP 723 inline metadata cannot resolve it).
- All artifacts are saved under `artifacts/` with deterministic paths for easy CI/gallery integration.
