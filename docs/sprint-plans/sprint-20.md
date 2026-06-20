# Sprint 20 Plan

## Sprint Goal
Increment thứ mười bảy (Round 17): add **Pipeline #1**, a concrete analysis pipeline for adapter → encode → Layer A method → artifact. This is the first `Pipeline` instance, so keep it narrow.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Implement a concrete `AnalysisPipeline` or similarly narrow class in `src/latent_anything/pipeline.py`.
- [ ] Task 2: Accept a decodable or non-decodable adapter plus one Layer A method.
- [ ] Task 3: Execute `encode` then `fit_transform`/`transform` with numpy-only public data.
- [ ] Task 4: Return a typed result object or pydantic model only if tests show dicts become unsafe.
- [ ] Task 5: Add tests with VAE/PCA and HiddenStateAdapter/PCA.
- [ ] Task 6: Add a config-backed construction path using Sprint 18 config machinery.
- [ ] Task 7: Add a script that reproduces a previous analysis demo through Pipeline #1.
- [ ] Task 8: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [ ] Task 9: Rule check: Pipeline instance #1 stays concrete; no broad workflow engine.
- [ ] Task 10: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Notes / Blockers
* This sprint should not support Layer B manipulation yet.
* Avoid a generic DAG/executor abstraction until at least two pipeline stories exist.
