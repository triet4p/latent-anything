# Task Summary: Sprint 59 Task 7 — Offline and checkpoint tests

**Sprint:** Sprint 59
**Task:** Add offline fixtures and marked checkpoint integration tests.

## Summary of Work

Added a deterministic tiny Diffusion policy fixture covering capture, queueing, parity, factory delegation, and analysis, plus a `network`/`large_download` test for the pinned public pair with compatibility gating.

## Files Modified

* `tests/test_lerobot_diffusion.py` — offline and marked integration coverage.
* `.github/workflows/optional-extras.yml` — dedicated Diffusion optional-extra lane.

## Testing

* **Test File:** `tests/test_lerobot_diffusion.py`
* **Status:** Passed (`5 passed, 1 skipped` by default)
* **Execution Command:** `uv run pytest tests/test_lerobot_diffusion.py -q`

## Additional Notes

The public checkpoint test is opt-in through `LATENT_ANYTHING_RUN_NETWORK=1`.
