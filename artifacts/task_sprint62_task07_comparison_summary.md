# Task Summary: Sprint 62 Task 7 — Comparison report

**Sprint:** Sprint 62
**Task:** Compare at least two policies or intervention settings.

## Summary of Work

Added typed comparison-report generation and a command/script for comparing
local records. Checked in `artifacts/lerobot_policy_comparison.json`, which
compares the pinned ACT and Diffusion observational artifacts with explicit
provenance and metric deltas while avoiding a misleading cross-dataset ranking.

## Files Modified

* `src/latent_anything/run_record.py` — comparison report.
* `src/latent_anything/cli.py` — `compare-runs` command.
* `scripts/lerobot_run_comparison.py` — reusable report script.
* `artifacts/lerobot_policy_comparison.json` — checked-in comparison evidence.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run --no-sync pytest tests/test_run_record.py tests/test_cli.py -q`
