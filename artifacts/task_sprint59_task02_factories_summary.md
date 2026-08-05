# Task Summary: Sprint 59 Task 2 — Official Diffusion factories

**Sprint:** Sprint 59  
**Task:** Use official LeRobot processors and policy construction to preserve input/output normalization.

## Summary of Work

Added `load_diffusion_policy()` using LeRobot's `DiffusionConfig`, `LeRobotDatasetMetadata`, `make_policy`, and `make_pre_post_processors` seams. Added the reproducible `lerobot-diffusion` profile and lock entry.

## Files Modified

* `src/latent_anything/integrations/lerobot_diffusion.py` — lazy official factory loader.
* `pyproject.toml`, `uv.lock` — isolated optional profile.
* `tests/test_lerobot_diffusion.py` — factory delegation fixture.

## Testing

* **Test File:** `tests/test_lerobot_diffusion.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_diffusion.py -q`

## Additional Notes

The adapter does not duplicate policy normalization or denoising logic.
