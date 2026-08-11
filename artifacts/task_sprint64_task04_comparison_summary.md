# Task Summary: Sprint 64 Task 4 — Mean versus sampled rollout

**Sprint:** Sprint 64
**Task:** Compare deterministic mean and sampled rollouts on controlled stochastic dynamics.

## Summary of Work

Added a reproducible controlled 2D stochastic linear-system benchmark that fits both Sprint 63’s deterministic transition and Sprint 64’s Gaussian transition, then compares mean-path and particle-rollout behavior.

## Files Modified

* [scripts/stochastic_transition_benchmark.py](../scripts/stochastic_transition_benchmark.py) — controlled benchmark.
* [artifacts/stochastic_transition_rollout.json](stochastic_transition_rollout.json) — measured comparison.
* [artifacts/stochastic_transition_rollout_config.json](stochastic_transition_rollout_config.json) — reproducible configuration.

## Testing

* **Status:** Passed — stable held-out rollout with sampled mean distinct from the deterministic control.
* **Execution Command:** `uv run python scripts/stochastic_transition_benchmark.py`

## Additional Notes

This is D2 synthetic evidence, not evidence for real-model calibration or epistemic uncertainty.
