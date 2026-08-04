# Task Summary: Sprint 56 Task 04 — Compatibility smoke tests

**Sprint:** Sprint 56
**Task:** Add base-install, extra-install, unsupported-version, and CPU-only smoke tests

## Summary of Work

Added a focused offline test module covering the supported 0.6.x report, LeRobot/Torch/NumPy incompatibilities, base-package import isolation in a subprocess, and the installed-extra API seam smoke. The extra smoke imports the raw policy, processor, dataset, environment, evaluation, and plugin-registration entry points and runs without CUDA or network access.

## Files Modified

* `tests/test_lerobot_integration.py` - seven focused compatibility and smoke tests.

## Testing

* **Test File:** `tests/test_lerobot_integration.py`
* **Status:** Passed — 7 tests
* **Execution Command:** `uv run --locked --extra lerobot pytest tests/test_lerobot_integration.py -v`

## Additional Notes

The base-install assertion runs in a fresh subprocess so test ordering cannot hide an eager LeRobot import. The optional smoke is safe to skip in the base CI environment and is executed in the dedicated LeRobot compatibility lane.
