# Task Summary: Sprint 28 Task 07 — ADR and Ledger Update

**Sprint:** Sprint 28
**Task:** Record the naming decision without runtime change

## Summary of Work

Linked the semantic-vocabulary contract from the evidence ledger and prepared the append-only ADR record. Runtime behavior and imports remain unchanged.

## Files Modified

* `docs/EVIDENCE_LEDGER.md` - links non-theory contract evidence.
* `.agents/memory/decisions.md` - records the naming decision.

## Testing

* **Test File:** `tests/test_api_surface.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_api_surface.py -v`

## Additional Notes

Sprint 31 is the first implementation sprint for this migration.
