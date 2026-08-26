# Sprint 78 Atomic Task 78.13 — Tokenized World Model SRP Refactor

Status: complete (pure internal refactor; no changelog entry).

## Responsibility split

- `src/latent_anything/tokenized_world_model.py` remains the stable public `TokenizedWorldModel` facade. It owns the frozen VQVAE boundary, adapter/transition seams, fit/evaluate/rollout orchestration, provenance, metadata, and public result/config schemas.
- `src/latent_anything/_tokenized_dynamics.py` owns the action-conditioned autoregressive GRU dynamics.
- `src/latent_anything/_tokenized_integrity.py` owns codebook-version binding and mutation checks.
- `src/latent_anything/_tokenized_training.py` owns bounded fitting and greedy/seeded categorical sampling.
- `src/latent_anything/_tokenized_evaluation.py` owns teacher-forced and free-running metric aggregation.
- `src/latent_anything/_tokenized_validation.py` owns finite-array, token, action, padding, and sequence-mask validation.

No token-transition Protocol, distribution wrapper, separate pipeline, continuous latent exposure, or VQVAE/JEPA/RSSM change was introduced. The existing concrete integer-code model and narrow mean-transition surface remain intact. There is no public TokenizedWorldModel checkpoint save/load API; tokenizer checkpoint integrity is enforced through the existing deterministic `codebook_version` binding.

## Metrics

Baseline `tokenized_world_model.py`: 706 LOC / 5,386 AST nodes; `TokenizedWorldModel` 512 LOC. Largest baseline methods were `fit_tokens` 51 LOC, `_free_running_metrics` 50, `predict_next` 48, `rollout` 41, and `evaluate` 37.

After:

| Module | LOC | AST nodes | Main ownership |
| --- | ---: | ---: | --- |
| `tokenized_world_model.py` | 600 | 3,177 | public facade, orchestration, schemas |
| `_tokenized_dynamics.py` | 42 | 429 | autoregressive dynamics |
| `_tokenized_integrity.py` | 22 | 100 | digest/version binding |
| `_tokenized_training.py` | 100 | 755 | fitting and sampling |
| `_tokenized_evaluation.py` | 102 | 936 | teacher/free metrics |
| `_tokenized_validation.py` | 97 | 783 | token/action/mask validation |

The facade fell by 106 LOC and 2,209 AST nodes; the stateful `TokenizedWorldModel` class fell from 512 to 435 LOC. Its largest methods are now `fit_tokens` 42 LOC, `rollout` 41, `evaluate` 37, `predict_next` 33, and `_free_running_metrics` 30. Private dependencies are one-way from the facade to focused helpers; the facade retains compatibility wrappers for private validation and metric seams.

## Compatibility and test evidence

- Public integer token IDs remain categorical `int64`; no embeddings or continuous latents are exposed.
- Existing codebook-version constructor/sequence checks and exact tokenizer-mutation error remain covered.
- Padding-token bounds, binary sequence masks, seeded sampling, greedy rollout, teacher-forced metrics, free-running drift, decoded MSE, task proxy, failure horizon, adapter/transition conformance, registry lookup, and provenance remain covered.
- Added public signature/config/result-schema snapshots and same-seed fit/prediction parity tests.
- Tokenized module suite: `9 passed`.
- Focused tokenized/VQVAE/transition/rollout/cache suite: `70 passed`.
- Full default pytest: `1529 passed, 36 skipped, 39 warnings`.

## Gates and review

- Ruff check: pass.
- Ruff format check: pass.
- Strict Pyright: `0 errors, 0 warnings, 0 informations`.
- Final `git diff --check`: pass; only normal Git LF/CRLF conversion warnings were emitted.
- Final graphify: `10,496 nodes / 20,343 edges / 909 communities`; known warning: 50 JSON files produce zero nodes and remain absent from the code graph.

Review verdict: PASS. No model download, network access, remote CUDA, commit, or push was performed.
