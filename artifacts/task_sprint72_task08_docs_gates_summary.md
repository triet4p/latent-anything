# Task Summary: Sprint 72 Task 8 — Evidence and project gates

**Sprint:** Sprint 72
**Task:** Update theory coverage, ADR/changelog/artifacts, and gates

## Summary of Work

Documented the tokenized world-model contract and benchmark, recorded the
initial D1 `THY-T09-TOKENIZED-WORLD-MODEL` evidence override, updated the evidence
ledger/changelog/global plan/sprint plan, and recorded the architectural
decision plus shape-validation lesson in project memory.

## Files Modified

* [docs/TOKENIZED_WORLD_MODEL.md](../docs/TOKENIZED_WORLD_MODEL.md) - User-facing contract and benchmark guide.
* [docs/evidence-ledger.json](../docs/evidence-ledger.json) - current D2 evidence links.
* [docs/EVIDENCE_LEDGER.md](../docs/EVIDENCE_LEDGER.md) - Evidence narrative.
* [CHANGELOG.md](../CHANGELOG.md) - User-visible Sprint 72 entry.
* [docs/PLAN.md](../docs/PLAN.md) - Sprint completion state.
* [docs/sprint-plans/sprint-72.md](../docs/sprint-plans/sprint-72.md) - Atomic task completion state.
* [.agents/memory/decisions.md](../.agents/memory/decisions.md) - Architecture decision.
* [.agents/memory/lessons-learned.md](../.agents/memory/lessons-learned.md) - Validation lesson.

## Testing

* **Test File:** [tests/test_scripts/test_validate_evidence_ledger.py](../tests/test_scripts/test_validate_evidence_ledger.py)
* **Status:** Passed
* **Execution Command:** `uv run python scripts/validate_evidence_ledger.py; uv run pytest tests/test_scripts/test_validate_evidence_ledger.py -q`

## Additional Notes

The benchmark wiring was end-to-end, but the original one-code tokenizer collapse
made its perfect token metrics degenerate. That historical D1 result is superseded
by the M12 remediation: the current compact synthetic CPU artifact is D2 with
non-trivial tokenizer usage, while greedy dynamics still fail at the first
held-out horizon. No real checkpoint or CUDA result is promoted. See
`artifacts/task_m12_remediation_tokenized_summary.md` for the measured rerun.
