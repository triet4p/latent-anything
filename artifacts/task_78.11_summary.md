# Sprint 78 Atomic Task 78.11 — RSSM Runtime and Validation SRP

Status: complete (pure internal refactor; no changelog entry).

## Responsibility split

- `src/latent_anything/rssm.py` remains the public facade and owns recurrent state, reset ordering, public calls, configuration/result classes, checkpoint lifecycle, and compatibility wrappers for private validation/teacher-forcing seams.
- `src/latent_anything/_rssm_runtime.py` owns pure NumPy recurrent stepping, teacher-forced distribution arrays, seeded particle rollout assembly, final hidden-state reduction, and established rollout metadata construction.
- `src/latent_anything/_rssm_validation.py` owns finite-array, point/batch, sequence, one-step, rollout, and binary-mask validation.

No state ownership moved into a generic runtime class. No Protocol/base class, public signature, torch/NumPy boundary, checkpoint behavior, mask semantics, RNG ordering, error contract, or result schema was broadened or changed.

## Metrics

78.11 baseline after task 78.10: `rssm.py` 774 LOC / 5,777 AST nodes; `RSSMLatentTransition` 553 LOC. Largest methods were `evaluate_rollout` 54 LOC, `fit` 52 LOC, and `rollout` 54 LOC.

After:

| Module | LOC | AST nodes | Classes | Functions | Largest functions |
| --- | ---: | ---: | ---: | ---: | --- |
| `rssm.py` | 730 | 4,588 | 6 | 49 | `evaluate_rollout` 54, `fit` 52, `rollout` 48 |
| `_rssm_runtime.py` | 125 | 986 | 0 | 4 | `sample_recurrent_rollout` 35 |
| `_rssm_validation.py` | 111 | 960 | 0 | 7 | `validate_rollout_inputs` 30 |

The public facade fell by 44 LOC and the stateful class by 50 LOC. Pure runtime and validation responsibilities now have one-way private dependencies; state remains exclusively on `RSSMLatentTransition`.

## Parity and tests

- Existing reset reproducibility, seeded rollout, deterministic-state retention, variable-length/partial-mask evaluation, checkpoint round-trip, cross-process loading, and public API/schema snapshots remain green.
- Added all-valid-mask parity against the unmasked fit path, direct state/rollout final-hidden consistency, and failed-step state immutability coverage.
- Focused RSSM transition suite: `26 passed`.
- Focused transition/rollout/streaming/cache suite: `72 passed`.
- Full default pytest: `1524 passed, 36 skipped, 39 warnings` in 175.56 seconds.

## Gates and review

- Ruff check: pass.
- Ruff format check: pass (`7 files already formatted`).
- Strict Pyright on `src` and RSSM transition tests: `0 errors, 0 warnings, 0 informations`.
- `git diff --check`: pass; only normal Git LF/CRLF conversion warnings were emitted.
- Final graphify: `10,415 nodes / 20,164 edges / 899 communities`; known warning: 50 JSON files produce zero nodes and remain absent from the graph.

Review verdict: PASS. The change follows RSSM/transition ADRs and the Rule of Three: private concrete helpers only, no premature lifecycle abstraction. No model download, network access, remote CUDA, commit, or push was performed.
