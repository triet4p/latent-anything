# Task Summary: Sprint 28 Task 02 — Vocabulary Comparison

**Sprint:** Sprint 28
**Task:** Compare candidate semantic names

## Summary of Work

Compared `analysis`, `intervention`, `adapter`, `planner`, and `runtime` against the current reducers, probes, attributors, transforms, causal edits, and execution helpers. Rejected generic layer and overloaded transform terminology.

## Files Modified

* `docs/rfcs/0001-semantic-api-vocabulary.md` - records selected and rejected vocabulary.

## Testing

* **Test File:** `tests/test_api_surface.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_api_surface.py -v`

## Additional Notes

Reserved future family names do not freeze an interface before Rule-of-Three evidence.
