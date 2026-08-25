# Task Summary: M12 remediation R4 — governance and documentation reconciliation

**Sprint:** Sprint 72 post-sprint remediation
**Task:** Reconcile M12 plans, ledgers, current docs, and superseded wording.

## Summary of Work

Added a bounded remediation closure to Sprint 72, marked Milestone 12 as
implementation-complete with remediation active, restored missing Sprint 64 and
70 completed-sprint entries, corrected the stale Sprint 77 no-go note, promoted
tokenized evidence to D2 only after the non-trivial usage gate passed, and kept
VQ/codebook claims at D2. Updated README, VQ/tokenized integration docs, ledger
narrative/JSON, and the Unreleased changelog without rewriting historical beta
release notes.

## Files Modified

* `docs/PLAN.md`, `docs/sprint-plans/sprint-70.md`,
  `docs/sprint-plans/sprint-72.md`, `docs/sprint-plans/sprint-77.md`
* `docs/EVIDENCE_LEDGER.md`, `docs/evidence-ledger.json`
* `docs/VQ_VAE_INTEGRATION.md`, `docs/TOKENIZED_WORLD_MODEL.md`, `README.md`,
  `CHANGELOG.md`

## Testing

* **Evidence validator:** `uv run python scripts/validate_evidence_ledger.py` —
  pass; inventory `107`, core `23/63`, overall `23/65`.

## Additional Notes

At the time of this R4 artifact, Milestone 12 remained `[~]` pending the final
closure gate (R5). R5 subsequently passed its bounded checks and
`docs/PLAN.md` now records Milestone 12 as `[x]`; this historical note is
retained for traceability.
