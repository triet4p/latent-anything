# Task Summary: Sprint 60 Task 2 — Official SmolVLA factories

**Sprint:** Sprint 60
**Task:** Load the model and pre/post-processors through supported LeRobot APIs.

## Summary of Work

`load_smolvla_policy()` loads `SmolVLAConfig` and `LeRobotDatasetMetadata` through LeRobot, delegates policy construction to `LeRobotAPI.make_policy` (with the model card's official camera `rename_map`), and builds the official pre/post-processor pipelines through `make_pre_post_processors` with a device-step override for the requested device. The full pinned policy was loaded and executed on CPU locally through these factories.

## Files Modified

* `src/latent_anything/integrations/lerobot_smolvla.py` — factory loader with lazy upstream imports.
* `pyproject.toml`, `uv.lock` — new `lerobot-smolvla` optional profile (`lerobot[dataset,evaluation,smolvla]>=0.6.0,<0.7.0`) and uv conflict declarations.
* `.github/workflows/optional-extras.yml` — profile resolution and smoke lanes.

## Testing

* **Test File:** `tests/test_lerobot_smolvla.py` (factory-delegation test)
* **Status:** Passed — `uv run pytest tests/test_lerobot_smolvla.py -q`
* **Real verification:** `uv run --extra lerobot-smolvla python -c "..."` loaded `lerobot/smolvla_libero` on CPU in ~60 s: context_dim 960, expert_dim 720, action_dim 7, max_action 32, chunk 50, num_steps 10, ~450M parameters; all four capture seams resolved (`SmolVLMVisionTransformer`, `Embedding`, `Linear`, `LlamaRMSNorm`).

## Additional Notes

Two upstream quirks were resolved during verification: `make_policy` requires the official `rename_map` to skip its strict visual-feature consistency check (the preprocessor itself applies the same rename), and the bf16 backbone required a lossless bf16→float32 upcast at the NumPy boundary.
