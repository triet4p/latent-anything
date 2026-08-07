# Task Summary: Sprint 61 Task 4 — Metrics and confidence intervals

**Sprint:** Sprint 61
**Task:** Measure success rate, return/task metrics, action deviation, latency, and confidence intervals over episodes.

## Summary of Work

`EpisodeOutcome` records per episode: success, sum/max reward (return), length, natural termination flag, mean per-step action deviation against the same-seed `no_hook` trajectory (L2 over aligned steps), mean/first-query latency, total latency, and query count. `ConditionSummary` aggregates each (condition, strength) cell with Wilson 95% confidence intervals for success rate (`wilson_ci`) and normal-approximation intervals for return (`_normal_ci`). Wilson intervals are honest for the small episode counts simulation affords.

## Files Modified

* [src/latent_anything/integrations/lerobot_benchmark.py](src/latent_anything/integrations/lerobot_benchmark.py) - `EpisodeOutcome`, `ConditionSummary`, `wilson_ci`, `_normal_ci`, `_summarize`.

## Testing

* **Test File:** [tests/test_lerobot_benchmark.py](tests/test_lerobot_benchmark.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_benchmark.py -v`

## Additional Notes

Wilson bounds verified against closed-form values (4/4 → [0.5101, 1.0]; 1/4 → [0.0456, 0.6994]).
