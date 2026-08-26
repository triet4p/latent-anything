# Sprint 78 Atomic Task 78.20 — LeRobot Benchmark SRP Closure

Status: complete for the 78.20 scoped refactor and validation; no model download,
network, remote CUDA, commit, or push was performed.

## Responsibility map

- `src/latent_anything/integrations/lerobot_benchmark.py` remains the stable
  public facade. It owns public configuration/result schemas, validation,
  compatibility wrappers, benchmark-specific direction/action helpers, and
  the official LeRobot action-selection seam.
- `src/latent_anything/_lerobot_benchmark_environment.py` owns lazy LIBERO and
  LeRobot environment construction, processor wiring, task metadata, and
  stale-user-config repair.
- `src/latent_anything/_lerobot_benchmark_execution.py` owns one seeded episode
  rollout, queue-aware model-query accounting, action/sample capture, and
  environment cleanup.
- `src/latent_anything/_lerobot_benchmark_statistics.py` owns confidence
  intervals, summaries, Spearman/correlation analysis, acceptance checks, and
  benchmark result assembly.

## Exact metrics

| Surface | LOC | AST nodes | Functions / classes | Largest function |
| --- | ---: | ---: | ---: | --- |
| `integrations/lerobot_benchmark.py` facade | 896 | 4,308 | 49 / 12 | `_build_outcome`: 52 LOC |
| `_lerobot_benchmark_environment.py` | 83 | 417 | 2 / 0 | `build_libero_benchmark_environment`: 67 LOC |
| `_lerobot_benchmark_execution.py` | 117 | 717 | 1 / 0 | `run_episode`: 97 LOC |
| `_lerobot_benchmark_statistics.py` | 305 | 1,961 | 7 / 0 | `run_simulation_benchmark`: 154 LOC |

The facade decreased from 1,231 to 896 LOC, and its
`run_simulation_benchmark` compatibility wrapper decreased from 158 to 20
LOC. Remaining facade responsibilities are limited to 10 public/compatibility
result/config surfaces plus benchmark-specific private seams; environment,
rollout, and statistical assembly responsibilities now have one focused owner
each.

## Compatibility and parity evidence

- `tests/test_lerobot_benchmark.py::test_benchmark_public_api_and_artifact_schema_digest`
  pins public function signatures, result-type field ownership, facade module
  identity, and the schema digest
  `afa32fe10a3e9d8f31cb0d2953dbd2d64291dbf4982a45ecb118387bafe9f8c6`.
- Existing benchmark, SmolVLA, LeRobot bridge, experiment-recorder, and W&B
  recorder behavior remains covered; official preprocess/select/postprocess
  flow, baseline bit identity, intervention effects, queue cadence, cleanup,
  and artifact schemas remain unchanged.

## Validation

- Focused benchmark/SmolVLA/LeRobot/recorder suite: **73 passed, 3 skipped**.
- Full default pytest: **1,536 passed, 36 skipped, 39 warnings**.
- 78.20-scoped Ruff check: **pass**.
- 78.20-scoped Ruff format check: **pass** (`5 files already formatted`).
- Strict Pyright on `src` and `tests`: **0 errors, 0 warnings, 0 informations**.
- `git diff --check`: **pass** (only normal LF/CRLF conversion warnings).
- Final graphify: **10,648 nodes / 20,677 edges / 921 communities**; known
  warning: 50 JSON source files produce zero graph nodes and remain absent from
  the code graph.

Repository-wide Ruff currently reports one unrelated existing violation at
`tests/test_lerobot_smolvla.py:346` (`B009`, constant `getattr`) from completed
task 78.19. It was not changed or remediated in this 78.20 closure.

## Review verdict

PASS for task 78.20 scope. The unrelated repository-wide Ruff finding is
explicitly retained for the owner’s synthesis and is not evidence of a
78.20 regression.
