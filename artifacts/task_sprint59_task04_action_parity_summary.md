# Task Summary: Sprint 59 Task 4 — Action-chunk parity

**Sprint:** Sprint 59  
**Task:** Verify unmodified action chunks match direct LeRobot inference for fixed seeds/noise.

## Summary of Work

The adapter calls the normal preprocessor → `policy.select_action()` → postprocessor path and proves byte-equivalent fixture actions under fixed noise. Queue misses return actions without fabricating captures.

## Files Modified

* `src/latent_anything/integrations/lerobot_diffusion.py` — queue-preserving adapter lifecycle.
* `tests/test_lerobot_diffusion.py` — fixed-noise parity and queue tests.

## Testing

* **Test File:** `tests/test_lerobot_diffusion.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_diffusion.py -q`

## Additional Notes

The adapter remains observational; it does not intervene in the action path.
