# Sprint 42 — Concept Activation Vectors and TCAV Summary

## Overview

Implemented TCAV (Testing with Concept Activation Vectors) style directional sensitivity analysis for latent representations. The module (`src/latent_anything/tcav.py`) provides concept/reference dataset handling with provenance, concept-direction learning via mean difference and regularised linear separator, gradient-based TCAV score computation for decoder-only transformers, statistical controls with random-concept baselines, and intervention cross-checks.

## Files Changed

### New files
- `src/latent_anything/tcav.py` — TCAV module (580+ lines): `ConceptDataset`, `ConceptDirectionResult`, `TransformerLogitTarget`, `TCAVScore`, `TCAVResult`, `TCAVConfig`, `TCAV` class, `compute_tcav()`, `intervention_agreement()`, `learn_mean_diff_direction()`, `learn_linear_separator_direction()`, `build_concept_dataset_from_text()`, internal gradient computation
- `tests/test_tcav.py` — 58 tests (56 offline + 2 marked network)

### Modified files
- `src/latent_anything/__init__.py` — Added TCAV exports to `__all__` and import block
- `src/latent_anything/_plugin_builtins.py` — Registered `TCAV` under `KIND_ANALYSIS`
- `tests/test_api_surface.py` — Updated public-export snapshot
- `tests/test_latent_anything/test_demo_smoke.py` — Updated registry count from 5→6 method_a
- `tests/test_latent_anything/test_registry.py` — Updated registry count assertions
- `CHANGELOG.md` — Added Sprint 42 entries
- `docs/sprint-plans/sprint-42.md` — All tasks marked done

## Key Design Decisions

1. **No Method protocol.** Like `LinearProbe` and `MLPProbe`, TCAV requires labelled concept sets and does not fit the `Method` / `AnalysisPipeline` lifecycle.

2. **Integration-specific gradient computation.** The first implementation targets decoder-only transformers (GPT-2). Internal gradient computation uses PyTorch hooks with `retain_grad()` to capture activation gradients, then computes directional derivatives. Diffusion support is deferred.

3. **Two direction-learning methods.** Mean difference (simple centroid subtraction) for speed and the interpretable baseline; regularised logistic regression (L2, balanced weights) for robustness to class imbalance.

4. **Bootstrap stability.** Direction stability is reported as mean pairwise cosine similarity across bootstrap resamples of the concept/reference sets, with 95% CI.

5. **Random-concept baselines.** The null distribution is constructed by permuting concept/reference labels and re-computing the TCAV score, with Bonferroni correction for multiple comparisons.

6. **Matched-norm intervention cross-check.** `intervention_agreement()` intervenes along `+v_c` and `-v_c` directions with a bounded strength and checks whether the sign of the output change matches the TCAV directional derivative prediction.

7. **No generic concept extraction protocol.** TCAV has different inputs, target semantics, and statistics from probes and must not be counted mechanically as a third interchangeable probe (per Sprint 41 notes).

## Test Coverage (offline: 56 pass)

| Category | Tests |
|---|---|
| ConceptDataset validation | 5 |
| ConceptDirectionResult validation | 4 |
| Mean diff direction learning | 6 |
| Linear separator direction learning | 5 |
| Direction consistency (both methods) | 1 |
| TransformerLogitTarget config | 4 |
| TCAVScore validation | 4 |
| TCAVResult serialization | 1 |
| TCAVConfig validation | 3 |
| TCAV class construction | 3 |
| Registry construction | 4 |
| Gradient computation (synthetic model) | 4 |
| Layer activation extraction (synthetic) | 3 |
| Full compute_tcav pipeline | 4 |
| Intervention agreement | 2 |
| Build from text import | 1 |
| Edge cases | 4 |
| Real integration (marked network) | 2 |

## Verification

- **857 tests pass** (including all 56 TCAV offline tests)
- **rufff check** passes (no lint errors)
- **Registry** correctly resolves `"analysis"` kind → `TCAV` factory
- All tasks documented in sprint plan marked `[x]`
