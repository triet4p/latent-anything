# Atomic Task 78.8 — transition SRP refactor

Status: complete (pure internal refactor; no public behavior change)

## Responsibility boundary

- `src/latent_anything/transition.py` remains the compatibility facade and owns all public result dataclasses (`OneStepMetrics`, `RolloutMetrics`, `GaussianPrediction`, `StochasticRollout`, and stochastic metric results), the deterministic transition lifecycle, and the stochastic Gaussian transition lifecycle.
- `src/latent_anything/_transition_core.py` owns only behavior proven identical by both concrete implementations: finite NumPy validation, constructor guards, source-identity resolution, one-step fit sample validation, affine-residual least-squares math, rollout shape normalization, and common provenance metadata construction.
- `LatentTransition` remains the existing narrow structural mean contract. No ABC, new Protocol, distribution wrapper, RSSM change, or lifecycle abstraction was introduced.

## Metrics

Baseline `transition.py`: 1,041 LOC / 7,287 AST nodes. Public class sizes were `DeterministicLatentTransition` 334 LOC and `StochasticGaussianLatentTransition` 422 LOC; duplicated fit/validation seams were present in both.

After:

| Module | LOC | AST nodes | Largest functions/classes |
|---|---:|---:|---|
| `transition.py` facade | 958 | 5,673 | `evaluate_rollout` 65; `fit` 51 each; deterministic class 288; stochastic class 384 |
| `_transition_core.py` | 195 | 1,349 | `validate_rollout_inputs` 36; `fit_affine_residual` 23; metadata builder 23 |
| Combined refactored surface | 1,153 | 7,022 | shared mechanics are isolated without merging lifecycles |

The facade dropped 83 LOC and 1,614 AST nodes. The largest duplicated fit methods dropped from 57/58 LOC to 51 LOC each, while their common math and guards are independently testable. The remaining larger evaluation/rollout methods are story-specific and were intentionally not generalized.

## Compatibility and evidence

The public classes and dataclass fields remain defined in `latent_anything.transition`, with signatures, error messages, metadata keys, immutable output behavior, identity/shape guards, seeded RNG semantics, Gaussian sampling/log-probability/interval behavior, deterministic and stochastic rollout semantics, and the narrow `LatentTransition` contract preserved.

Existing D2 fixtures remain the numerical reference: deterministic seed 63 synthetic affine system; stochastic seed 64 diagonal-Gaussian system; RSSM seed 65 comparison. Their committed artifacts remain unchanged, including deterministic/stochastic fit and rollout summaries and the retained RSSM open-loop failure evidence.

Added regression coverage:

- public import/module, method-signature, and result-dataclass field snapshots;
- Hypothesis seeded-sampling shape/reproducibility property;
- exact constructor/shape/identity/error guards through the existing suite;
- cross-process RSSM checkpoint load/config round-trip, without modifying `rssm.py`;
- deterministic/stochastic mean-vs-sampled rollout, interval, log-probability, particle, and cache/stream/planner integration parity.

## Validation

- Focused transition/rollout/planning/cache/streaming suite: `54 passed`.
- Scoped Ruff check: passed.
- Scoped Ruff format check: passed (`3 files already formatted`).
- Strict scoped Pyright: `0 errors, 0 warnings, 0 informations`.
- Full default pytest: `1519 passed, 36 skipped, 39 warnings` in `156.89s`.
- `git diff --check`: passed; only normal Git LF/CRLF normalization warnings were emitted.

Warnings are existing sklearn convergence, registry/pipeline deprecations, and UMAP warnings; no warning is caused by this refactor.

## Review and graph

Final graphify reported `10,330 nodes / 20,022 edges / 924 communities`; the known warning is the repository's 50 JSON files producing zero AST nodes. The task review confirms the transition ADRs: concrete deterministic/Gaussian/RSSM lifecycles remain separate, only the already-frozen mean contract is shared, public APIs remain NumPy-facing, and no premature abstraction was introduced.

No changelog entry is required for this pure refactor. No commit, push, model download, network access, or remote CUDA execution was performed.
