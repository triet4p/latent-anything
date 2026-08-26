# Atomic Task 78.9 — physical transition SRP split

Status: complete (pure internal refactor; no public behavior change)

## Ownership and dependency direction

- `src/latent_anything/transition.py` is a stable 37-line compatibility facade.
- `src/latent_anything/_transition_types.py` owns the public result dataclasses: `OneStepMetrics`, `RolloutMetrics`, `GaussianPrediction`, `StochasticRollout`, `StochasticOneStepMetrics`, and `StochasticRolloutMetrics`.
- `src/latent_anything/_transition_deterministic.py` owns the complete `DeterministicLatentTransition` lifecycle.
- `src/latent_anything/_transition_stochastic.py` owns the complete `StochasticGaussianLatentTransition` lifecycle.
- `src/latent_anything/_transition_core.py` owns only shared finite-array/shape/identity/constructor validation, affine residual fitting, rollout normalization, and provenance metadata.

The import graph is acyclic: `transition` facade → concrete modules/types; concrete modules → core/types; core → `LatentSpace`. No module imports back into the facade, and `rssm.py`, JEPA, and tokenized-world-model internals were not modified.

## Before/after metrics

The 78.8 monolith baseline was `transition.py`: 1,041 LOC / 7,287 AST nodes.

| File | LOC | AST nodes | Main ownership |
|---|---:|---:|---|
| `transition.py` facade | 37 | 53 | compatibility imports and module-identity restoration |
| `_transition_types.py` | 258 | 1,431 | six public result dataclasses |
| `_transition_deterministic.py` | 279 | 1,600 | deterministic class (250 LOC) |
| `_transition_stochastic.py` | 412 | 2,700 | stochastic class (377 LOC) |
| `_transition_core.py` | 195 | 1,349 | proven shared mechanics |
| Combined final surface | 1,181 | 7,133 | physically separated responsibilities |

The public facade now contains no concrete lifecycle implementation. The result types and each transition philosophy have independent ownership without introducing a base class, ABC, or new Protocol.

## Compatibility proof

- `latent_anything.transition` and top-level `latent_anything` imports remain unchanged.
- Public classes/result types have `__module__ == "latent_anything.transition"` after facade import, preserving externally observable identity and pickle import paths.
- Fresh subprocess imports pass in both orders: facade then internal types, and internal types then facade.
- Fresh subprocess pickle load passes for `GaussianPrediction` with exact public module/name/distribution family.
- `inspect.signature` snapshots pass for deterministic fit and stochastic rollout; dataclass field schemas remain exact.
- Existing registry/config construction and `isinstance(..., LatentTransition)` checks pass.
- Existing deterministic seed-63 and stochastic seed-64 numerical artifacts remain unchanged; seeded RNG, error messages, immutable outputs, rollout metadata, Gaussian sampling/log-probability/interval semantics, and checkpoint behavior remain covered.
- Cross-process RSSM checkpoint/config round-trip passes; `rssm.py` remains outside the change scope.

## Validation

- Focused transition/API/rollout/planning/cache/streaming/JEPA/tokenized suite: `73 passed`.
- Scoped Ruff check: passed.
- Scoped Ruff format check: passed (`6 files already formatted`).
- Strict scoped Pyright: `0 errors, 0 warnings, 0 informations`.
- Full default pytest: `1520 passed, 36 skipped, 39 warnings` in `174.72s`.
- `git diff --check`: passed; only normal Git LF/CRLF normalization warnings were emitted.

Warnings are existing sklearn convergence, registry/pipeline deprecations, and UMAP notices; none are caused by this refactor.

## Review verdict and graph

Latent-Anything review verdict: PASS. The transition ADRs remain satisfied: the deterministic, memoryless Gaussian, and RSSM lifecycles remain concrete and distinct; only the already-frozen mean contract is shared; public APIs remain NumPy-facing; no generic lifecycle/distribution abstraction was introduced.

Final graphify reported `10,347 nodes / 20,049 edges / 901 communities`; the known warning is the repository's 50 JSON files producing zero AST nodes. No changelog entry is required. No commit, push, model download, network access, or remote CUDA execution was performed.
