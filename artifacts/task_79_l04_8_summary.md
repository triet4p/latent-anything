# Task 79 L04.8 Phase A Summary

## Scope

Implemented the owner-approved M14 L04.8 disentanglement protocol locally and
offline. Real model loading remains network/CUDA gated and was not run here.

## Changes

- Added pure causal-group quality, deterministic derangement/slot reversal,
  five-seed 2,000-replicate bootstrap, and mapping-digest helpers.
- Added CPU float64 balanced class-weighted logistic probes with train-only
  standardization, fixed LBFGS settings, intercept, and L2 `C=1.0`.
- Added target-token-excluded fixed-vocabulary binary input-token baseline.
- Retained sanitized per-row holdout evidence for every seed and diagnostic,
  train class counts, train-only standardization/probe digests, slot-reversal
  mappings, and exact authored-fixture row/pair/group linkage.
- Reviewer follow-up now deduplicates row IDs before group derangement, binds
  the exact fixture order and GPT-2 target IDs/strings, independently enforces
  elapsed/CUDA/RSS budgets and no-mutation, and recomputes all row/group/seed
  metrics and control verdicts in the validator. The final review pass uses
  process peak RSS (`resource.getrusage` on Linux, normalized from KiB), gates
  both CUDA allocated and reserved memory, and records canonical named-parameter
  digests before/after execution. Accepted D2 now requires complete CUDA
  provenance and normalized process-peak RSS evidence; preflight failures remain
  truthful non-promoting D0 artifacts with explicit stages.
- Added the lazy real `TransformerLMIntegration` handler for native GPT-2
  layer 6 / hidden-state index 7, with sanitized summaries and resource
  provenance. Runtime acceptance now enforces elapsed <=1800 seconds, CUDA
  allocated <=6 GiB, RSS <=4 GiB, and unchanged model-parameter digest.
- Wired dispatcher, artifact promotion (D2 only), and fail-closed validator;
  controls are recomputed from retained summaries.
- Split validator responsibilities into fixture binding, raw-token linkage,
  resource/mutation checks, metric recomputation, and control recomputation;
  handler resource finalization, budget evaluation, raw-token linkage,
  acceptance, result assembly, and failure cleanup are independently named
  helpers. Main validator/handler coordinators are now 141/146 lines, and
  malformed nested evidence fails closed with structured errors.
- Artifact provenance now carries the truthful resource device even for
  partial attempted-CUDA failures. Generic artifact/run/failure validators
  enforce a coherent stage/backend/device/network tuple; unavailable-device
  preflight failures are explicitly staged before CUDA. Legacy artifacts that
  predate the stage field remain readable without weakening validation of new
  envelopes.
- Added offline tests for group independence, imbalance and degenerate
  bootstrap, deterministic shuffle/slot reversal over duplicated row IDs,
  train-only probe repeat, padding/output-token exclusion, runner-to-validator
  linkage, fixture/probability tampering, and adversarial budget evidence.
- Added truthful D0 failure-stage/resource tests, factor-permutation supervision
  linkage tamper tests, raw-token matrix/excluded-column linkage tests, and
  reserved-memory/model-digest adversarial tests, including direct overrun
  coverage for all frozen resource caps, malformed nested payloads, and
  placeholder-device triads in artifact/failure validators.
- Recorded the protocol decision in `.agents/memory/decisions.md` and updated
  `docs/sprint-plans/sprint-79.md`.

## Verification

- `uv run pytest -q tests/test_m14_l04_disentanglement.py` — 26 passed.
- Focused L04 runner/envelope plus disentanglement suite — 44 passed.
- L04 handler/envelope suite — 139 passed.
- Full suite: `1811 passed, 36 skipped, 39 warnings`.
- Ruff checks for all changed L04 source/tests — passed.
- Explicit strict Pyright for changed source/tests — 0 errors, 0 warnings, 0 informations.
- Evidence ledger, explanation contract, `mkdocs build --strict`, and `git diff --check` — passed.
- `graphify update .` — passed; graph rebuilt with 12,865 nodes and 26,083
  edges (1,036 communities; aggregated HTML view).
- Real CUDA/remote transport — intentionally not run in Phase A.
- Generated `artifacts/_mkdocs_check/` and `artifacts/_debug_l04/` were each
  re-resolved as workspace-local, untracked directories with no symlinks;
  exact native PowerShell recursive deletion was authorized but rejected by
  the execution policy, so both remain for owner cleanup.

## Owner review risks

The real lane still requires the owner-gated PowerShell `ssh.exe` temporary-file
transport with decoder exit, decoded SHA-256 comparison, execution markers, and
cleanup. The real GPT-2 run must also confirm resource caps and the strict
per-seed gates before any D2 promotion.
