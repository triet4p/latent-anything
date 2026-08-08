# Task Summary: Sprint 62 Task 4 — Inspection and replay CLI

**Sprint:** Sprint 62
**Task:** Add supported capture-point, dataset/policy inspection, replay, and comparison commands.

## Summary of Work

Added the stdlib `latent-anything` CLI entry point and module invocation. The
capture-point and policy commands stay lazy with respect to LeRobot; dataset
inspection loads the optional extra only when requested; replay materializes a
saved config for a deterministic rerun.

## Files Modified

* `src/latent_anything/cli.py` — commands and serialization.
* `pyproject.toml` — `latent-anything` entry point.
* `tests/test_cli.py` — offline command coverage.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run --no-sync pytest tests/test_cli.py -q`
