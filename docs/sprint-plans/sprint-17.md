# Sprint 17 Plan

## Sprint Goal
Increment thứ mười tư (Round 14): start **Plugin Extraction** with an in-process registry for built-in adapters and methods. No entry points yet; first prove registry shape with local classes.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Create `src/latent_anything/registry.py` or a small package if code pressure requires it.
- [ ] Task 2: Define registry records for kind (`adapter`, `method_a`, `method_b`), name, factory, and metadata.
- [ ] Task 3: Add decorators or explicit registration helpers for built-in classes.
- [ ] Task 4: Register VAE, RandomProjection, HiddenStateAdapter, PCA, UMAP, SAE, Lerp, SteeringVector, and ActivationPatch.
- [ ] Task 5: Add lookup/list APIs with deterministic ordering.
- [ ] Task 6: Add tests for duplicate names, missing names, kind filtering, and factory retrieval.
- [ ] Task 7: Add a small demo or script that lists registry entries.
- [ ] Task 8: Run `ruff check`, `ruff format`, `pyright`, and full pytest.
- [ ] Task 9: Rule check: this is registry instance #1; keep it in-process and do not add Python entry points yet.
- [ ] Task 10: Update artifact summary, `CHANGELOG.md`, and `docs/PLAN.md`.

## Notes / Blockers
* No semantic search, no vector DB.
* This sprint must not change behavior of existing adapters/methods.
