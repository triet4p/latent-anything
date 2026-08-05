# Task Summary: Sprint 59 Task 3 — Multi-axis Diffusion capture

**Sprint:** Sprint 59  
**Task:** Capture observation-conditioning and denoising/action representations with timestep metadata.

## Summary of Work

Captured one U-Net global-conditioning vector per executed chunk and each U-Net output with denoising-step and scheduler-timestep provenance. Captures retain episode step metadata and immutable NumPy latent values.

## Files Modified

* `src/latent_anything/integrations/lerobot_diffusion.py` — typed representations and hook lifecycle.
* `tests/test_lerobot_diffusion.py` — axis and metadata assertions.

## Testing

* **Test File:** `tests/test_lerobot_diffusion.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_diffusion.py -q`

## Additional Notes

The generic capture metadata is reused while Diffusion-specific axes stay in the adapter-owned representation type.
