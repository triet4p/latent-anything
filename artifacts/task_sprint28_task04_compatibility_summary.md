# Task Summary: Sprint 28 Task 04 — Compatibility Window

**Sprint:** Sprint 28
**Task:** Define aliases and removals through 0.9.0

## Summary of Work

Defined the canonical, warning, migration-diagnostic, and removal windows from 0.1 through 1.0, with removal of legacy registry kinds at 0.9.0.

## Files Modified

* `docs/rfcs/0001-semantic-api-vocabulary.md` - compatibility contract.

## Testing

* **Test File:** `tests/test_api_surface.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_api_surface.py -v`

## Additional Notes

Legacy spellings must never choose a new behavior silently.
