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
