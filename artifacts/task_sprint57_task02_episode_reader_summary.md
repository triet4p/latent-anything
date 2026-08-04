# Task Summary: Sprint 57 Task 2 — Lazy episode reader

**Sprint:** Sprint 57
**Task:** Read one `LeRobotDataset` episode without copying the dataset.

## Summary of Work

Added `LeRobotDatasetReader` and `read_lerobot_episode`, which resolve canonical episode offsets and lazily call the upstream dataset for only the requested frame range. Selected-episode relative-index mappings are supported through LeRobot's own mapping when available.

## Files Modified

* `src/latent_anything/integrations/lerobot_dataset.py` — lazy reader implementation.
* `tests/test_lerobot_dataset_bridge.py` — lazy access and selected-episode tests.

## Testing

* **Test File:** `tests/test_lerobot_dataset_bridge.py`
* **Status:** Passed
* **Execution Command:** `.venv/Scripts/python.exe -m pytest tests/test_lerobot_dataset_bridge.py -q`

## Additional Notes

Parquet and video decoding remain entirely upstream-owned.
