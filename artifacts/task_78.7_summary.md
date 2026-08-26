# Atomic Task 78.7 — SAE evaluation SRP refactor

Status: complete (refactor-only; no public behavior change)

## Scope and responsibility boundary

- `src/latent_anything/sae_evaluation.py` remains the public facade. It owns the public dataclasses/configuration, registry-constructable `SAEFeatureEvaluation` lifecycle, optional model-bound cross-check orchestration, and compatibility wrappers.
- `src/latent_anything/_sae_metrics.py` owns fitted metric assembly, read-only metric arrays, decoder-direction extraction, greedy decoder-cosine matching, and cross-seed stability aggregation.
- `src/latent_anything/_sae_atlas.py` owns example-label coercion, feature ranking, atlas construction, and JSON persistence. It imports the public result types only inside functions to preserve the facade boundary and avoid a cycle.
- No `Method`/`Analysis` protocol was widened, no generic serialization protocol was introduced, and no changelog entry was required because this is a pure internal refactor.

## Metrics

Baseline `sae_evaluation.py`: 965 LOC / 5,884 AST nodes. Largest baseline responsibilities were `_evaluate_fitted` (82 LOC), `SAEFeatureEvaluation` (189 LOC), `cross_check_feature` (114 LOC), and `build_feature_atlas` (64 LOC).

After:

| Module | LOC | AST nodes | Largest function(s) |
|---|---:|---:|---|
| `sae_evaluation.py` facade | 728 | 3,765 | `cross_check_feature` 114; `SAEFeatureEvaluation` class 165 |
| `_sae_metrics.py` | 166 | 1,235 | `evaluate_fitted` 76; `assemble_stability` 34 |
| `_sae_atlas.py` | 155 | 1,126 | `build_feature_atlas` 62; `rank_feature_examples` 29 |
| Combined refactored surface | 1,049 | 6,126 | responsibilities are separated by domain |

The facade lost 237 LOC and 2,119 AST nodes; the added support code is cohesive and independently testable. The remaining 114-LOC cross-check function is intentionally model/hook orchestration and was not split into a generic model protocol.

## Compatibility and evidence

The deterministic baseline fixture remained exact: `n_train=400`, `n_val=100`, validation MSE `0.0036947252008124403`, train MSE `0.003363850090983217`, mean L0 `2.77`, mean L1 `0.9817913174629211`, and `n_dead_features=0`. The canonical atlas JSON remained 6 entries with SHA256 `fea6bc4626515d752b5470ac2c0f1d6c240b291d43cca08867b66ecd693c218d`.

Regression coverage includes the existing train/validation split, checkpoint/config/registry behavior, cross-seed decoder-direction matching, cross-check paths, atlas round-trip, and exact error cases, plus:

- public `evaluate_sae_features` signature and `SAEEvaluationResult.to_dict()` key snapshots;
- Hypothesis coverage that decoder matching is invariant to feature-slot permutations;
- fail-closed tampered/truncated atlas loading (`KeyError`);
- preservation of atlas schema, ordering, labels, provenance, and serialization fields.

## Validation

- Focused SAE/config/registry/transformer suite: `142 passed, 19 warnings`.
- Scoped Ruff check: passed.
- Scoped Ruff format check: passed (`4 files already formatted`).
- Strict scoped Pyright: `0 errors, 0 warnings, 0 informations`.
- Full default pytest: `1516 passed, 36 skipped, 39 warnings` in `185.07s`.
- `git diff --check`: passed; only normal Git LF/CRLF normalization warnings were emitted.
- Warnings are existing sklearn convergence, registry deprecation, pipeline deprecation, UMAP, and optional-lane warnings; none are caused by this refactor.

## Graph and review

Final graphify reported `10,299 nodes / 19,959 edges / 925 communities`; the known warning is 50 JSON files producing zero AST nodes. Review verdict is PASS: no public API, schema, optional-import, or security guard changes were observed.

No commit, push, model download, network access, or remote CUDA execution was performed.
