# Task Summary: Sprint 78.22 — MLP probe SRP ownership

**Sprint:** Sprint 78
**Task:** 78.22 — Split MLP training, nonlinear controls, and shared probe splitting

## Summary of Work

Extracted the Torch-only MLP model/training/evaluation lifecycle into
`_mlp_training.py`, moved nonlinear control result types and comparison
classification into `_mlp_controls.py`, and centralized the deterministic
leakage-guarded split algorithm in `_probe_split.py`. The public
`latent_anything.mlp_probe` facade retains validation, NumPy standardization,
orchestration, result assembly, public signatures, registry construction, and
public module identity. `probes._stratified_split` remains an import-compatible
wrapper, while MLP and TCAV depend on the focused shared implementation.

## Files Modified

* [src/latent_anything/mlp_probe.py](../src/latent_anything/mlp_probe.py) — reduced facade to validation, preprocessing, orchestration, and public result/API compatibility.
* [src/latent_anything/_mlp_training.py](../src/latent_anything/_mlp_training.py) — extracted MLP architecture, deterministic seeding, training loop, early stopping, and evaluation.
* [src/latent_anything/_mlp_controls.py](../src/latent_anything/_mlp_controls.py) — extracted nonlinear control/result types and comparison classification.
* [src/latent_anything/_probe_split.py](../src/latent_anything/_probe_split.py) — centralized shared stratified split implementation.
* [src/latent_anything/probes.py](../src/latent_anything/probes.py) — retained `_stratified_split` compatibility wrapper.
* [src/latent_anything/tcav.py](../src/latent_anything/tcav.py) — migrated to shared split ownership.
* [tests/test_mlp_probe.py](../tests/test_mlp_probe.py) — added public signature/module identity, exact result digest, and split digest snapshots.
* [tests/test_probes.py](../tests/test_probes.py) — added shared-helper/import-parity coverage.
* [docs/sprint-plans/sprint-78.md](../docs/sprint-plans/sprint-78.md) — marked atomic task 78.22 complete.

## Metrics and Compatibility

`mlp_probe.py` changed from 744 LOC / 625 nonblank / 2,890 AST nodes / 14
functions / 6 classes to 524 LOC / 449 nonblank / 1,795 AST nodes / 9
functions / 3 classes. `MLPProbe.fit` changed from 208 LOC, complexity 28 to
131 LOC, complexity 18. New private modules are focused: training 194 LOC,
controls 68 LOC, split 50 LOC.

The exact result digest remains
`f5fc4ebd30c4240db69f95ba208c73146eb91454e5f10dee2233830799aeb580`; the
shared split digest is
`60ff848c1685f5cb185ffe4d6f4cb0cb62ca523ebc4718f906c5b54edc683ac7`.
Public `NonlinearControls` and `ProbeComparison` preserve their historical
`latent_anything.mlp_probe` module identity.

## Testing

* **Focused tests:** `uv run pytest tests/test_mlp_probe.py tests/test_probes.py tests/test_tcav.py tests/test_api_surface.py -q` — **143 passed, 7 skipped**.
* **Repository Ruff:** `uv run ruff check src tests` — **passed**.
* **Repository format:** `uv run ruff format --check src tests` — **246 files already formatted**.
* **Strict Pyright:** `uv run pyright src tests` — **0 errors, 0 warnings, 0 informations**.
* **Full suite:** `uv run pytest -q` — **1539 passed, 36 skipped, 39 warnings** in 312.09s.
* **Diff check:** `git diff --check` — **passed**.

## Graph and Review

`graphify update .` completed with topology **10,679 nodes / 20,700 edges /
936 communities**. The graph reports 50 non-code JSON files with zero AST
nodes; this is an existing graphify extraction warning and does not affect the
Python source graph.

Review verdict: **PASS-WITH-WARNINGS**. No Blocking findings, public Torch
leakage, API/schema incompatibility, split leakage, deterministic numerical
drift, or test-integrity issue was found. Remaining warnings are the existing
architectural backlog for future MLP facade evolution; no additional
remediation is required by task 78.22.
