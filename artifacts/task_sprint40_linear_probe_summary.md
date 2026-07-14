# Sprint 40 — Linear Probe Implementation Summary

## Overview

Implemented a label-aware linear classification probe (`LinearProbe`) with leakage-guarded train/val/test splitting, training-only feature standardization, configurable regularization, control baselines, cross-seed stability reporting, and comprehensive tests. The Sprint 36 centroid-based `probe_accuracy` helper was reconciled so the project has one unambiguous meaning for linear probing.

## Files Changed

### New files
- `src/latent_anything/probes.py` — Main probe module: `LinearProbe` class, `LinearProbeConfig` (pydantic), `LinearProbeResult` (frozen dataclass), split helpers, control baselines, cross-seed evaluation, layer evaluation
- `tests/test_probes.py` — 42 offline tests + 3 marked network tests

### Modified files
- `src/latent_anything/evaluation.py` — Reconciled `probe_accuracy()` to delegate to `LinearProbe`; old centroid logic renamed to `_centroid_probe_accuracy` (internal fast path)
- `src/latent_anything/_plugin_builtins.py` — Registered `LinearProbe` under `KIND_ANALYSIS`
- `src/latent_anything/__init__.py` — Added probe exports to `__all__` and import block
- `tests/test_evaluation.py` — Updated error message regex for new `LinearProbe` validation
- `tests/test_api_surface.py` — Updated public-export snapshot
- `tests/test_latent_anything/test_demo_smoke.py` — Updated registry count from 3→4 method_a
- `tests/test_latent_anything/test_registry.py` — Updated registry count assertions
- `CHANGELOG.md` — Added Sprint 40 entries

## Key Design Decisions

1. **No Method Protocol.** `LinearProbe` is deliberately outside the `Method`/`AnalysisPipeline` lifecycle because probes need labels during fitting. It is registered under the `"analysis"` kind for semantic taxonomy only.

2. **Leakage-guarded split.** `_stratified_split()` produces deterministic, per-class proportional train/val/test partitions with boolean masks. Feature standardization fits on training statistics only.

3. **Control baselines.** `compute_controls()` computes majority-class, shuffled-label, and raw-input accuracies on the **same** train/test split as the probe, ensuring fair comparison.

4. **Migration path.** The old `probe_accuracy()` function still returns `float` for backward compatibility but now delegates to `LinearProbe`. The centroid logic is preserved as `_centroid_probe_accuracy()` (internal fast path for `evaluate_explanation`).

## Test Coverage (offline: 42 pass)

| Category | Tests |
|---|---|
| Config validation | 5 |
| Result fields | 2 |
| Stratified split | 3 |
| Fit lifecycle | 10 |
| Predict (after fit) | 5 |
| Leakage guards | 3 |
| Degenerate classes | 3 |
| Config construction | 4 |
| Control baselines | 3 |
| Cross-seed evaluation | 3 |
| Layer evaluation | 1 |
| Real integration (marked network) | 3 |

No regressions in existing test suite (762 pass, 4 expected-count updates).

## Artifact

- Task artifact: `artifacts/task_sprint40_linear_probe_summary.md`
- Changelog: Updated with Sprint 40 entries
