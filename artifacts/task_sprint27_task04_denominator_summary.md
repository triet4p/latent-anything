# Task Summary: Sprint 27 Task 04 — Coverage Denominator

**Sprint:** Sprint 27
**Task:** Automate the 95% core and 90% overall calculation

## Summary of Work

Defined core coverage as T01–T09 (including T03B) and overall coverage as every non-contextual topic. Only D2/D3 statuses qualify; the validator reports both numerator and denominator without mutating files.

## Files Modified

* `docs/evidence-ledger.json` - declares thresholds and qualifying statuses.
* `scripts/validate_evidence_ledger.py` - calculates the coverage summary.

## Testing

* **Test File:** `tests/test_scripts/test_validate_evidence_ledger.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_scripts/test_validate_evidence_ledger.py -v`

## Additional Notes

The baseline is deliberately 0/60 stable coverage: current D1 beta evidence does not overstate release readiness.
