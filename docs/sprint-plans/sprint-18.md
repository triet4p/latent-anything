# Sprint 18 Plan

## Sprint Goal
Increment thứ mười lăm (Round 15): add **registry-backed config instantiation** using pydantic v2. This is config for concrete objects, not yet a full `Pipeline` abstraction.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Add pydantic config models for registry object specs: kind, name, params.
- [x] Task 2: Implement `build_from_config(spec)` that resolves a registry entry and instantiates it.
- [x] Task 3: Support nested adapter-in-method references only for a real existing case such as `ActivationPatch(adapter=...)`.
- [x] Task 4: Add clear validation errors for unknown names, wrong kind, and invalid params.
- [x] Task 5: Add tests for VAE, RandomProjection, PCA, Lerp, SteeringVector, and ActivationPatch construction from config.
- [x] Task 6: Add a script that recreates a previous demo object stack from config without adding a public Pipeline.
- [x] Task 7: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [x] Task 8: Rule check: config instantiation #1 stays narrow and registry-local.
- [x] Task 9: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Notes / Blockers
* Do not introduce Hydra.
* Do not generalize into a workflow language in this sprint.
