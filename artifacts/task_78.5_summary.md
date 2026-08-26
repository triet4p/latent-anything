# Task Summary: TCAV model-boundary SRP refactor (78.5)

**Sprint:** Sprint 78  
**Task:** 78.5  
**Status:** Complete

## Scope and outcome

Separated optional PyTorch model-boundary work from TCAV's statistical/public
facade. Gradient capture, layer activation extraction, and matched positive /
negative interventions now live in `src/latent_anything/_tcav_model.py`, with
PyTorch imported only inside those call paths. `src/latent_anything/tcav.py`
retains all public classes/functions, registry construction, NumPy boundaries,
and historical private helper seams used by integrations/tests.

This is a pure internal refactor: no changelog entry is required, no generic
differentiable-model Protocol was introduced, and no Method/Analysis lifecycle
was widened.

## Files

- `src/latent_anything/tcav.py` — stable statistical/result facade and narrow compatibility wrappers.
- `src/latent_anything/_tcav_model.py` — optional model hooks, gradient/activation capture, and interventions.
- `tests/test_tcav.py` — API/signature/result snapshots and hook cleanup/failure-path coverage.
- `docs/sprint-plans/sprint-78.md` — task status.

## Metrics and dependency direction

Before: `tcav.py` was 1,260 LOC, 26 functions, 7 classes, and 4,768 AST
nodes. After: `tcav.py` is 997 LOC, 20 functions, 7 classes, and 3,470 AST
nodes; `_tcav_model.py` is 213 LOC, 3 model-boundary functions, and 1,449 AST
nodes. The largest remaining facade operation is the 218-line statistical
`compute_tcav`; model execution responsibilities no longer live inside it or
the result dataclasses. The dependency direction is facade → model boundary
and domain/statistical helpers; `_tcav_model.py` has no import back-edge into
the facade and imports `TransformerLogitTarget` only under `TYPE_CHECKING`.

No public signature exposes PyTorch. The historical private imports
`_compute_transformer_layer_gradient` and `_extract_layer_activation` remain
available from `latent_anything.tcav` through typed wrappers.

## Parity and failure evidence

Captured baseline signatures for `compute_tcav` and `intervention_agreement`
are asserted exactly in `tests/test_tcav.py`. The deterministic synthetic
fixture remains unchanged:

| Result | Baseline and refactor |
| --- | ---: |
| aggregate score | `0.0` |
| aggregate CI95 | `0.0` |
| random baseline scores | five zeros |
| empirical p-value | `1.0` |
| significance | `not_significant` |
| intervention agreement | `1.0` |

Existing unknown-layer/error messages and gradient/activation shapes remain
covered. A forward-failure test proves that hook handles are removed on the
exception path; the complete module hook set is empty afterward.

## Validation

- Focused TCAV/config/registry/transformer tests: **182 passed, 2 skipped, 19 warnings**.
- Ruff scoped check: **pass**.
- Ruff format scoped check: **3 files already formatted**.
- Strict Pyright scoped check: **0 errors, 0 warnings, 0 informations**.
- Full default pytest: **1511 passed, 36 skipped, 39 warnings in 189.66s**.
- Final `git diff --check`: **pass** (only normal CRLF conversion warnings for dirty tracked files).
- Final graphify topology: **10,254 nodes / 19,857 edges / 916 communities** after the artifact update. Graphify reported the known 50 non-code JSON files with zero AST nodes; no source extraction failure occurred.

The warnings are existing registry deprecations, one sklearn convergence
warning, and UMAP random-state warnings; no new failure or behavior change was
observed. No model download, network validation, remote CUDA, commit, or push
was performed.
