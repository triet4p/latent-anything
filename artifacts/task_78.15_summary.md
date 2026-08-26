# Sprint 78 Atomic Task 78.15 — Reward/Value Metrics SRP Refactor

Status: complete (pure internal refactor and test-only snapshots; no changelog entry).

## Responsibility split

- `src/latent_anything/reward_value.py` remains the public facade. `LinearRewardScorer` and `MonteCarloValueEstimator` retain fitted model calls and provenance; `RewardValueEvaluator` retains trajectory/holdout orchestration, model-vs-real comparison, calibration/result assembly, and planner-facing contracts.
- `src/latent_anything/_reward_value_metrics.py` owns pure finite-array/shape validation, frozen result-array helpers, discount validation, masked terminal-aware discounted returns, Bellman residual calculations, and masked summary metrics.

No actor-critic/head Protocol, planner API change, model-call widening, or public torch boundary was introduced. Existing terminal, padding, bootstrap, source-space identity, policy/data-distribution provenance, and registry/config behavior remain concrete and unchanged.

## Metrics and dependency direction

Baseline `reward_value.py`: 846 LOC / 5,894 AST nodes. `RewardValueEvaluator` was 267 LOC; the largest baseline facade methods were `evaluate_holdout` 95 LOC and `evaluate` 64 LOC.

After:

| Module | LOC | AST nodes | Main ownership |
| --- | ---: | ---: | --- |
| `reward_value.py` | 766 | 4,785 | fitted models, evaluator orchestration, public results |
| `_reward_value_metrics.py` | 154 | 1,320 | pure validation and numerical metrics |

The public facade fell by 80 LOC and 1,109 AST nodes. `RewardValueEvaluator` remains cohesive at 267 LOC because it owns model calls, trajectory identity checks, result/provenance assembly, and planner integration. The helper has no dependency on planners, transitions, or model classes; dependencies are one-way from facade to pure numerical helpers, with no new SCC.

## Compatibility and numerical evidence

- Masked, terminal-aware discounted returns preserve zeroed padding and reset bootstrapping after mask gaps or terminal transitions.
- Bellman residuals preserve terminal/no-continuation and final-step semantics for vector and batch paths.
- Finite-array, matrix/vector shape, boolean-mask, frozen-array, and discount error messages remain covered.
- Fitted reward/value numerical fixtures, uncertainty fallback, held-out reward RMSE/MAE/bias, value calibration, Bellman consistency, model-vs-real comparison, source-space identity rejection, CEM/MPPI/rollout integration, and recorder artifact persistence remain green.
- Added public evaluator signature and result-schema snapshots plus invalid-discount and mask-gap numerical assertions.

Tests and gates:

- Focused reward/value/CEM/MPPI/rollout/transition suite: `78 passed`.
- Full default pytest: `1532 passed, 36 skipped, 39 warnings`.
- Ruff check: pass.
- Ruff format: pass.
- Strict Pyright on reward/value facade, helper, and tests: `0 errors, 0 warnings, 0 informations`.
- Final `git diff --check`: pass; only normal Git LF/CRLF conversion warnings were emitted.
- Final graphify: `10,533 nodes / 20,414 edges / 924 communities`; known warning: 50 JSON files produce zero nodes and remain absent from the code graph.

## Review verdict

PASS. No model download, network access, remote CUDA, commit, or push was performed.
