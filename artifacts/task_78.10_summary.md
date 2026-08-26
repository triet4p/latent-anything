# Sprint 78 Atomic Task 78.10 — RSSM SRP Refactor

Status: complete (pure internal refactor; no changelog entry).

## Scope and responsibility split

- `src/latent_anything/rssm.py` remains the public `RSSMLatentTransition` facade. It owns public configuration/result dataclasses, recurrent state/reset, NumPy validation, prediction/step/rollout orchestration, and compatibility methods.
- `src/latent_anything/_rssm_training.py` owns the bounded torch fitting path and teacher-forced hidden/prediction loops. Torch remains internal; callers and fitted parameters remain NumPy values.
- `src/latent_anything/_rssm_evaluation.py` owns one-step and masked open-loop metric aggregation.
- `src/latent_anything/_rssm_checkpoint.py` owns the existing NPZ arrays and metadata JSON read/write boundary.

No stateful base class, Protocol, broadened `LatentTransition` contract, deterministic/stochastic/JEPA/tokenized changes, public torch exposure, or checkpoint schema change was introduced.

## Metrics

Baseline `rssm.py`: 850 LOC, 7,349 AST nodes; `RSSMLatentTransition` 631 LOC. Largest methods were `fit` 86 LOC, `evaluate_rollout` 72, `rollout` 54, and `load` 30.

After:

| Module | LOC | AST nodes | Classes | Functions | Largest functions |
| --- | ---: | ---: | ---: | ---: | --- |
| `rssm.py` | 774 | 5,777 | 6 | 50 | `rollout` 54, `evaluate_rollout` 54, `fit` 52 |
| `_rssm_training.py` | 139 | 1,123 | 1 | 2 | `fit_rssm_parameters` 85 |
| `_rssm_evaluation.py` | 133 | 1,102 | 2 | 2 | `aggregate_rollout_metrics` 56 |
| `_rssm_checkpoint.py` | 70 | 354 | 1 | 2 | `write_rssm_checkpoint` 28 |

The public facade fell from 850 to 774 LOC and the RSSM class from 631 to 553 LOC. Training, evaluation, and persistence now have one-way private dependencies into the facade's public types; no new cycle was introduced.

## Compatibility and test evidence

- Public `RSSMLatentTransition` module identity and bound signatures were snapshotted (`fit`, `load`); configuration and result dataclass field order were snapshotted.
- Existing recurrent reset reproducibility, masked/partial sequence fit and evaluation, variable-length rollout, seeded sampling, and deterministic-state behavior remain covered.
- Existing checkpoint round-trip and cross-process loading tests pass. NPZ field names, metadata JSON structure, source-space identity, fit metadata, and load/reset behavior are unchanged.
- New negative coverage rejects non-binary masks and a tampered checkpoint missing required metadata fields.
- Existing synthetic D2 evidence remains unchanged, including the intentionally retained RSSM negative result in `artifacts/rssm_transition_comparison.json`; metrics were not altered to hide history.

## Gates

- Focused transition/rollout/streaming/cache suite: `72 passed`.
- Full default pytest: `1522 passed, 36 skipped, 39 warnings` in 213.80 seconds.
- Ruff check: pass.
- Ruff format check: pass (`5 files already formatted`).
- Strict Pyright on `src` and the RSSM transition tests: `0 errors, 0 warnings, 0 informations`.
- `git diff --check`: pass; Git emitted only normal LF/CRLF conversion warnings.
- Final graphify: `10,385 nodes / 20,107 edges / 926 communities`; known warning: 50 JSON source files produced zero nodes and remain absent from the code graph.

## Review

PASS. The refactor follows the RSSM/transition ADRs: the public recurrent runtime remains concrete and stateful, the narrow `LatentTransition` mean surface is untouched, torch is confined to fitting, and serialization remains the established NumPy/NPZ boundary. No blocking or advisory findings remain for this atomic task. No model download, network access, remote CUDA, commit, or push was performed.
