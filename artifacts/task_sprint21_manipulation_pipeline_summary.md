# Task Summary: Sprint 21 — ManipulationPipeline (Pipeline #2)

**Sprint:** Sprint 21 (Round 18)
**Task:** Tasks 1–9 — ManipulationPipeline, tests, demo, rule check, docs

## Summary of Work
Implemented **Pipeline #2** (`ManipulationPipeline`) in `src/latent_anything/pipeline.py` alongside a minimal `_PipelineBase` sketch shared with the existing `AnalysisPipeline` (Pipeline #1). The `ManipulationPipeline` supports two stories:
1. **Adapter-mediated (data-space output)**: encode → BMethod → decode, returning metric-ready `np.ndarray` arrays (used with `ActivationPatch`).
2. **Latent-only (trajectory output)**: BMethod.apply_trajectory on a `Trajectory`, returning a new `Trajectory` (used with `SteeringVector`, `Lerp`).

Config-backed construction added via `ManipulationPipelineSpec` + `build_manipulation_pipeline_from_config`, reusing Sprint 18 config machinery. 28 new tests cover construction (6), data-space story (3), trajectory story (4), fit delegation (3), convenience methods (3), spec model (4), and config-backed build (5). A demo script (`scripts/end_to_end_manipulation_demo.py`) reproduces the Sprint 13 showcase path through Pipeline #2 with both stories and matplotlib visualisation.

## Files Modified
* [src/latent_anything/pipeline.py](src/latent_anything/pipeline.py) — Added `_PipelineBase` (shared sketch), `ManipulationPipeline` (Pipeline #2), `ManipulationPipelineSpec`, `build_manipulation_pipeline_from_config`. Updated `AnalysisPipeline` to inherit from `_PipelineBase`.
* [src/latent_anything/__init__.py](src/latent_anything/__init__.py) — Exported `ManipulationPipeline`, `ManipulationPipelineSpec`, `build_manipulation_pipeline_from_config`; updated `__all__`.
* [tests/test_latent_anything/test_pipeline.py](tests/test_latent_anything/test_pipeline.py) — Added 28 new tests across 7 test classes covering all pipeline invariants.
* [scripts/end_to_end_manipulation_demo.py](scripts/end_to_end_manipulation_demo.py) — New demo script showing both data-space and trajectory stories through Pipeline #2.

## Testing
* **Test File:** [tests/test_latent_anything/test_pipeline.py](tests/test_latent_anything/test_pipeline.py)
* **Status:** 49 pipeline tests pass (21 existing + 28 new). Full suite: 551 tests pass.
* **Lint/Format:** `ruff check` — clean. `ruff format` — clean.
* **Type Check:** `pyright` (src only) — 0 errors, 0 warnings, 0 informations.
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_pipeline.py -v`

## Additional Notes
* `_PipelineBase` is deliberately minimal — adapter + method storage + optional `latent_space`. No abstract interface or generic `run()`. The Rule of Three says: 2 instances → sketch shape only; freeze waits for Pipeline #3 (e.g. RuntimePipeline).
* `__call__` is NOT part of the `BMethod` Protocol (by design — signatures differ per instance). The pipeline does not hide this behind a brittle generic call; it provides separate `run_data()` and `run_trajectory()` methods.
* The `ManipulationPipeline.run_trajectory()` return type is `np.ndarray | Trajectory` to accommodate both latent→latent and data→data methods.
