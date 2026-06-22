# Sprint 20 Plan

## Sprint Goal
Increment thứ mười bảy (Round 17): add **Pipeline #1**, a concrete analysis pipeline for adapter → encode → Layer A method → artifact. This is the first `Pipeline` instance, so keep it narrow.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Implement a concrete `AnalysisPipeline` or similarly narrow class in `src/latent_anything/pipeline.py`.
- [x] Task 2: Accept a decodable or non-decodable adapter plus one Layer A method.
- [x] Task 3: Execute `encode` then `fit`/`transform` with numpy-only public data.
- [x] Task 4: Return a typed result object (`PipelineResult` frozen dataclass).
- [x] Task 5: Add tests with VAE/PCA and HiddenStateAdapter/PCA (21 tests).
- [x] Task 6: Add a config-backed construction path using Sprint 18 config machinery (`PipelineSpec` + `build_pipeline_from_config`).
- [x] Task 7: Add a script that reproduces a previous analysis demo through Pipeline #1 (`scripts/end_to_end_pipeline_demo.py`).
- [x] Task 8: Run `ruff check`, `ruff format`, `pyright`, and full pytest — all 523 pass, 0 errors.
- [x] Task 9: Rule check: Pipeline instance #1 stays concrete; no broad workflow engine. No new registry kind. Generalisation waits for Pipeline #2.
- [x] Task 10: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Notes / Blockers
* This sprint should not support Layer B manipulation yet.
* Avoid a generic DAG/executor abstraction until at least two pipeline stories exist.
