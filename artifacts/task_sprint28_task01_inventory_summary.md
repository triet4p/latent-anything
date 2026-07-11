# Task Summary: Sprint 28 Task 01 — Public Naming Inventory

**Sprint:** Sprint 28
**Task:** Inventory roadmap-layer names in the beta surface

## Summary of Work

Inventoried the top-level exports, adapter and method protocols, registry constants, `ObjectSpec` usages, pipeline specs, tests, demos, and documentation references to Layer A/B/C terminology.

## Files Modified

* `docs/rfcs/0001-semantic-api-vocabulary.md` - records the inventory outcome and migration scope.

## Testing

* **Test File:** `tests/test_api_surface.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_api_surface.py -v`

## Additional Notes

The inventory is intentionally captured before runtime renaming begins in Sprint 31.
