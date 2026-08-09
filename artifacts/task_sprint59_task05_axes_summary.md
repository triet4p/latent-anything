# Task Summary: Sprint 59 Task 5 — Separate trajectory axes

**Sprint:** Sprint 59
**Task:** Analyze latent trajectories across denoising timesteps and episode time without conflating the axes.

## Summary of Work

`DiffusionEpisodeTrace` exposes conditioning trajectories in episode order and groups denoising trajectories by exact recorded diffusion timestep. The result records episode-time and per-timestep path lengths separately.

## Files Modified

* `src/latent_anything/integrations/lerobot_diffusion.py` — explicit trajectory accessors and metrics.
* `tests/test_lerobot_diffusion.py` — two-axis analysis assertions.

## Testing

* **Test File:** `tests/test_lerobot_diffusion.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_diffusion.py -q`

## Additional Notes

Action-chunk position remains provenance metadata and is not collapsed into either trajectory axis.
