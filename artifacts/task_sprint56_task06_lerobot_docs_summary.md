# Task Summary: Sprint 56 Task 06 — Supported LeRobot seams

**Sprint:** Sprint 56
**Task:** Document supported seams and rejected reimplementation scope

## Summary of Work

Added the LeRobot integration contract covering installation, the 0.6.x compatibility window, lazy loading, raw policy/processor/dataset/environment/evaluation/plugin seams, bridge-owned types, explicit non-goals, resolver conflicts, and the upstream-upgrade checklist. Updated the general optional-integration guide to point at the detailed contract.

## Files Modified

* `docs/LEROBOT_INTEGRATION.md` - detailed integration contract and upgrade checklist.
* `docs/OPTIONAL_INTEGRATIONS.md` - optional extra summary and link.

## Testing

* **Test File:** `tests/test_lerobot_integration.py`
* **Status:** Passed
* **Execution Command:** `uv run --locked --extra lerobot pytest tests/test_lerobot_integration.py -v`

## Additional Notes

The document explicitly reserves dataset mapping, policy capture, intervention, and evaluation evidence for later sprints. This keeps Sprint 56 a dependency/API boundary rather than an accidental reimplementation.
