# Task Summary: Sprint 58 Task 7 — ACT fixtures and integration smoke

**Sprint:** Sprint 58
**Task:** Add tiny policy unit fixtures plus a marked checkpoint integration test.

## Summary of Work

Added a tiny torch ACT-shaped policy with a `model.decoder` seam, processor fixtures, queue-miss coverage, factory tests, base-install lazy-import coverage, and a `network`/`large_download` public checkpoint smoke.

## Files Modified

* `tests/test_lerobot_act.py` — deterministic fixture and opt-in integration tests.
* `.github/workflows/optional-extras.yml` — LeRobot compatibility lane includes ACT tests.
* `pyproject.toml` — strict Pyright includes the new test and benchmark script.

## Testing

* **Test File:** `tests/test_lerobot_act.py`
* **Status:** Passed (4 passed, 1 skipped by default)
* **Execution Command:** `uv run pytest tests/test_lerobot_act.py -q`

## Additional Notes

CUDA was not required: the deterministic fixture and checkpoint loading path are CPU-capable. The real smoke is opt-in and does not run in the default offline suite.
