# Sprint 21 Plan

## Sprint Goal
Increment thứ mười tám (Round 18): add **Pipeline #2**, a manipulation pipeline for adapter → Layer B method → decoded/data or trajectory result. With two pipeline stories, sketch a shared internal shape but do not freeze yet.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Implement a concrete `ManipulationPipeline` for `ActivationPatch` and/or `SteeringVector`.
- [ ] Task 2: Support one adapter-mediated story: fit patch, apply to held-out data, return metric-ready arrays.
- [ ] Task 3: Support one latent-only story: steering or lerp over a `Trajectory`.
- [ ] Task 4: Sketch an internal `_PipelineBase` only if `AnalysisPipeline` and `ManipulationPipeline` truly share code.
- [ ] Task 5: Add tests for data-space output, trajectory output, config construction, and no mutation.
- [ ] Task 6: Add a demo using the Sprint 13 showcase path through Pipeline #2.
- [ ] Task 7: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [ ] Task 8: Rule check: Pipeline instance #2 → sketch shared shape only; freeze waits for a third different pipeline such as runtime/streaming.
- [ ] Task 9: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Notes / Blockers
* `__call__` signatures differ across B-Methods; do not hide that with a brittle generic method call.
