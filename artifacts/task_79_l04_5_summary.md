# Task Summary: Sprint 79 L04.5 — TCAV Phase A

## Status

Implementation-ready; owner-reviewed remote CUDA execution remains pending.
No model download, network execution, evidence artifact, ledger promotion, or
commit was performed in Phase A.

The authoritative TCAV gap is
`THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018`; the short M14 record
key is `t05_tcav`. Accepted evidence must keep these distinct in
`accepted_record_ids` versus `accepted_gap_ids`, following the L03 envelope
precedent. The M14 policy permits D3 only after the real CUDA run and a
validator-backed artifact; the authored fixture alone cannot claim D3, and any
eventual claim remains limited to this controlled fixture rather than broad
external semantic validity.

## Implementation

Added a lazy `TransformerLMIntegration` TCAV handler for the frozen
`openai-community/gpt2` revision. It requires the explicit network/CUDA gates,
resolves exactly one-token ` true`/` false` targets, uses transformer block 6
and native hidden-state index 7 at the last non-padding prompt token, fits the
`tone_positive` concept direction on train groups only, and evaluates held-out
groups/pairs without prompt text in summaries. Cached activations and one
task-margin gradient per row are reused by the shuffled-label, random,
matched-norm, and off-target controls. Genuine +/- hidden-state hooks and a
zero-strength identity hook are retained for intervention controls.

The lane records the exact five seeds `[17, 29, 41, 53, 67]`, 2,000 group
bootstrap replicates, and a fixed 99-null corrected empirical p-value using
`(1 + count(null >= observed)) / 100`; the three null families each contain
33 independently seeded draws distributed across all five seed summaries. All metrics and controls carry finite
point estimates, units, aggregation units, intervals, frozen comparators,
thresholds, and recomputed pass verdicts.

## Files and tests

- `scripts/_m14_l04_tcav.py` — lazy real handler and orchestration.
- `scripts/_m14_l04_tcav_runtime.py` — tokenization, activation/gradient,
  intervention, and resource seams.
- `scripts/_m14_l04_tcav_metrics.py` — group bootstrap, Wilson, and p-value
  helpers.
- `scripts/_m14_l04_tcav_controls.py` — five-control verdict assembly.
- `scripts/_m14_l04_validate_tcav.py` — strict TCAV schema, provenance,
  coverage, and verdict validator.
- `tests/test_m14_l04_tcav_handler.py` — deterministic fake hook path,
  bounded-forward smoke, semantic-failure triad, accepted D3 linkage, and
  mutation rejection tests.

Focused TCAV/L04 tests pass locally, as do the full repository gates: 1697
passed and 36 skipped; strict Pyright is clean. Targeted Ruff is clean; the
repository-wide Ruff command still reports pre-existing findings in
skill/notebook trees outside this change. Graphify was refreshed after the
audit (11,894 nodes and 23,212 edges). The remote CUDA run must still prove
real model semantics before any D3 record can be retained.

Policy evidence: `docs/EVIDENCE_LEDGER.md:17,22` defines D3 as a reproducible
real-model artifact; `docs/EVIDENCE_GAP_PLAN.md:120-121,215` says the authored
fixture alone cannot establish D3; and `docs/M14_REAL_SYSTEM_VALIDATION.md:49`
requires the real CUDA run plus validator-backed artifact. Therefore this
Phase A implementation is eligible to attempt D3 but does not claim it. The
later remote transport remains the explicitly frozen authenticated direct
PowerShell `ssh.exe` path in `docs/M14_REAL_SYSTEM_VALIDATION.md:49`, despite
the generic remote skill text; no remote action was taken here.
