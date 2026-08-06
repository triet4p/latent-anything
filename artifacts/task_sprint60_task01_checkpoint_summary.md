# Task Summary: Sprint 60 Task 1 — Pinned SmolVLA checkpoint pair

**Sprint:** Sprint 60
**Task:** Select a public SmolVLA checkpoint/dataset pair with feasible hardware requirements and pin revisions.

## Summary of Work

Pinned `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de` with its documented training dataset `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` for `libero` / `libero_spatial` in `SmolVLACheckpointSpec`, and recorded the reproducible hardware profile (`SmolVLAHardwareProfile`: SmolVLM2-500M backbone, 16 VLM + 16 expert layers at 75% width, bfloat16, ~450M parameters, 16 GB GPU recommended). The pair is the model card's own `train_config.json` configuration.

## Files Modified

* `src/latent_anything/integrations/lerobot_smolvla.py` — immutable checkpoint, environment, and hardware identity.
* `docs/LEROBOT_INTEGRATION.md` — public pair, hardware profile, and reproduction details.

## Testing

* **Test File:** `tests/test_lerobot_smolvla.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_smolvla.py -q`

## Additional Notes

The `lerobot/libero` dataset uses `observation.images.image`/`image2` camera names while the policy config expects `camera1`/`camera2`/`camera3`; the model card's rename map is passed to the official `make_policy` factory (verified against the real model in Task 2).
