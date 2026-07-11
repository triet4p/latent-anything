# Task Summary: Sprint 27 Task 06 — CI Evidence Validation

**Sprint:** Sprint 27
**Task:** Reject stale or broken local evidence links in CI

## Summary of Work

Added a dedicated CI step that runs the same read-only ledger validator before linting. Validation only inspects repository paths and therefore does not import optional backends or download models.

## Files Modified

* `.github/workflows/ci.yml` - runs evidence validation in CI.
* `scripts/validate_evidence_ledger.py` - rejects stale IDs and missing local evidence paths.

## Testing

* **Test File:** `tests/test_scripts/test_validate_evidence_ledger.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_scripts/test_validate_evidence_ledger.py -v`

## Additional Notes

The validator is intentionally read-only and safe for a base installation.
