# Task Summary: Sprint 27 Task 01 — Capability Inventory

**Sprint:** Sprint 27
**Task:** Stable capability identifiers for the theory index

## Summary of Work

Added a read-only inventory parser that derives one stable `THY-<tier>-<slug>` capability ID for every bold checklist item in `docs/THEORY.md`. The current inventory contains 107 topics, including uniquely numbered supplementary tiers.

## Files Modified

* `scripts/validate_evidence_ledger.py` - derives and validates the inventory.
* `docs/evidence-ledger.json` - records the capability policy and per-capability overrides.

## Testing

* **Test File:** `tests/test_scripts/test_validate_evidence_ledger.py`
* **Status:** Passed
* **Execution Command:** `uv run python scripts/validate_evidence_ledger.py`

## Additional Notes

Renaming a THEORY title is an intentional capability migration and must update its ledger key in the same change.
