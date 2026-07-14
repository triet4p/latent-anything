# Sprint 41 — Nonlinear MLP Probe Implementation Summary

## Overview

Implemented a bounded nonlinear MLP probe (`MLPProbe`) as an information-accessibility upper bound complementing Sprint 40's `LinearProbe`. The MLP has configurable architecture, deterministic initialization, early stopping on validation accuracy, and full NumPy-facing typed results. Added memorization tests and linear-vs-nonlinear comparison classification.

## Files Changed

### New files
- `src/latent_anything/mlp_probe.py` — MLP probe module: `MLPProbe` class, `MLPProbeConfig` (pydantic), `MLPProbeResult` (frozen dataclass), memorization test, probe comparison
- `tests/test_mlp_probe.py` — 33 offline tests + 2 marked network tests

### Modified files
- `src/latent_anything/__init__.py` — Added MLP probe exports to `__all__` and import block
- `src/latent_anything/_plugin_builtins.py` — Registered `MLPProbe` under `KIND_ANALYSIS`
- `pyproject.toml` — Added new files to pyright include list
- `tests/test_api_surface.py` — Updated public-export snapshot
- `tests/test_latent_anything/test_demo_smoke.py` — Updated registry count from 4→5 method_a
- `tests/test_latent_anything/test_registry.py` — Updated registry count assertions
- `CHANGELOG.md` — Added Sprint 41 entries
- `docs/sprint-plans/sprint-41.md` — All tasks marked done

## Key Design Decisions

1. **Separate result type.** `MLPProbeResult` has nonlinear-specific fields (n_epochs, stopped_early, architecture, n_params, optimizer) that are not in `LinearProbeResult`. They share `to_dict()` pattern.

2. **Reused split/preprocessing.** Uses `_stratified_split()` from `probes.py` for train/val/test splitting and training-only standardization.

3. **Early stopping on validation accuracy.** Using accuracy (not loss) as the early stopping criterion, since accuracy plateaus cleanly once perfect classification is reached.

4. **Memorization test.** `nonlinear_memorization_test()` trains on shuffled labels with a separate seed and reports the memorization ratio relative to chance. Default threshold: 2× chance.

5. **Classification taxonomy.** `compare_probes()` classifies representation access as one of: `linear-only`, `nonlinear-only`, `both`, `unsupported`, `memorization-prone` under explicit accuracy and gap thresholds.

## Test Coverage (offline: 33 pass)

| Category | Tests |
|---|---|
| Config validation | 4 |
| Result fields | 2 |
| Fit lifecycle | 8 |
| Architecture reporting | 2 |
| Early stopping | 2 |
| Determinism | 2 |
| Degenerate classes | 3 |
| Predict (not implemented) | 2 |
| Config construction | 3 |
| Memorization test | 3 |
| Probe comparison | 2 |
| Real integration (marked network) | 2 |

No regressions in existing test suite (795 pass, 4 expected-count updates).

## Artifact

- Task artifact: `artifacts/task_sprint41_mlp_probe_summary.md`
- Changelog: Updated with Sprint 41 entries
