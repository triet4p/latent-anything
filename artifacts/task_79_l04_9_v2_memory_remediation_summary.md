# Task Summary: L04.9 v2 bounded runtime memory and diagnostic hardening

## Scope

The 855f440 Stage A evidence remains pending, byte-exact, D0, and unpromoted.
This local remediation does not use SSH/CUDA/holdout, rerun semantics, delete
evidence, finalize, or commit.

## Implementation

- Added a private `_ForwardSnapshot` seam. Stage A and Stage B retain only
  attention masks, scalar target margins, and raw block states required for
  activation interventions; complete public generation results are released
  immediately.
- Runtime requests now suppress unused native hidden-state and logit-lens
  projections through a private integration seam. Public request fields and
  default behaviour remain unchanged.
- Stage A finalization clears clean/corrupt/score caches, runs Python/Torch
  cleanup best-effort, and only then records ResourceTracker peaks.
- Added allowlisted resource budget subcodes for CPU peak, allocated GPU
  peak, reserved GPU peak, and invalid budget fields while retaining historical
  categories for old sidecars.
- Added `scripts/m14_l049_v2_load_stress.py`, a train-only diagnostic that
  reuses the canonical Stage A candidate-workload runner for all 1,296
  candidate records / 2,592 scorer calls, immediately discarding records.
  It validates the single finalizer result before emitting cleanup PASS and
  has not been run remotely.

## Evidence assessment

The current 855f440 raw capture, audit, bundle members, and triad remain at
their exact recorded sizes and SHA-256 values in the source-SHA-keyed paths.
The sanitized assessment sidecar records the D0 runtime exception and marks
cache retention as a strong code-level inference, not artifact-proven root
cause. Selection evaluations are discarded and no holdout or Stage B work is
authorized.

## Verification

- `uv run pytest tests/test_m14_l049_v2.py tests/test_transformer_lm.py -q`
  passes after the snapshot and output-suppression changes.
- Sidecar canonical digest validation passes.
- Existing v2 archive/triad independent validation remains PASS.
- Full repository tests, full L04, quality/docs gates, and the final Graphify
  update pass; repository-wide Ruff still reports unrelated pre-existing
  violations outside this task's scope.
