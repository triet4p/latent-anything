# Pipeline execution stories

Sprint 66 keeps the three concrete pipeline stories discoverable from one
stable beta import path while giving each story focused ownership.

## Stories

| Story | Implementation | Execution input | Result |
| --- | --- | --- | --- |
| Analysis | `analysis_pipeline.py` | data batch | `PipelineResult` |
| Manipulation | `manipulation_pipeline.py` | method-specific data or trajectory | data array or trajectory |
| Rollout | `rollout_pipeline.py` | initial latent plus action sequence | `RolloutResult` |

`pipeline.py` remains a compatibility shim. Existing imports such as
`from latent_anything.pipeline import AnalysisPipeline` continue to resolve;
new code may import the focused modules directly.

## Shared contract

`PipelineContract` intentionally exposes metadata only:

- `pipeline_kind` identifies `analysis`, `manipulation`, or `rollout`.
- `latent_space` describes the associated space, when the story has one.

There is no generic `run()` contract. Analysis fits a Layer A method,
Manipulation forwards method-specific calls, and Rollout consumes the frozen
`LatentTransition.mean_rollout()` surface. Keeping those signatures explicit
preserves the lifecycle differences that the earlier pipeline stories proved.

## Rollout usage

```python
from latent_anything import RolloutPipeline

pipeline = RolloutPipeline(transition, cache=InMemoryCache())
result = pipeline.run(initial_latent, actions, profiler=profiler)
states = result.to_numpy()
```

`run_async()` is a thread-backed wrapper over the same synchronous operation.
It returns the same state sequence, propagates transition exceptions unchanged,
and re-raises `asyncio.CancelledError`. Only completed mean trajectories are
cached; cached payloads are plain array/metadata dictionaries so immutable
trajectory metadata remains safe to reconstruct.

Configuration uses `RolloutPipelineSpec` with a runtime `ObjectSpec`. Built-in
deterministic, stochastic-Gaussian, and RSSM transitions are registered under
`KIND_RUNTIME` as `deterministic_transition`, `stochastic_transition`, and
`rssm_transition`.

## CEM planning

`CEMPlanner` optimizes bounded continuous action sequences through a diagonal
Gaussian population. Each iteration samples candidates, evaluates them through
`RolloutPipeline` and its optional `RewardValueEvaluator`, retains the elite
set, and smoothly refits the mean and standard deviation. `CEMPlanResult`
returns the selected sequence, model-predicted return, per-iteration candidate
statistics, convergence history, and a `RuntimeProfile`.

The planner is configured with `CEMPlannerSpec`, built with
`build_cem_planner_from_config`, and registered as the `KIND_RUNTIME` entry
`cem_planner`. The controlled CPU reproduction is:

```text
uv run python scripts/cem_planning_benchmark.py
```

The benchmark reports fixed-zero, random-shooting, and CEM returns in both
model space and a deliberately action-mismatched environment. A positive
predicted-minus-realized gap is model exploitation evidence, not task success.

## MPPI planning

`MPPIPlanner` samples bounded action perturbations around a nominal sequence
and updates that sequence with numerically stable exponential return weights.
The temperature controls how concentrated the weights are; every candidate is
retained in the update, and the result records effective sample size and weight
entropy. `plan_receding_horizon()` executes an action prefix, shifts the
nominal sequence, and repeats the plan. Rollout candidates still use the same
`RolloutPipeline` and `RewardValueEvaluator` components as CEM.

The planner is configured with `MPPIPlannerSpec`, built with
`build_mppi_planner_from_config`, and registered as the `KIND_RUNTIME` entry
`mppi_planner`. The controlled CPU reproduction is:

```text
uv run python scripts/mppi_planning_benchmark.py
```

The benchmark compares fixed-zero, random shooting, CEM, and MPPI on the same
continuous-control task. It reports predicted and realized return, action
smoothness, sample count, latency, and robustness to a deliberate transition
scale error. This is synthetic D2 evidence, not real-model or CUDA evidence.
