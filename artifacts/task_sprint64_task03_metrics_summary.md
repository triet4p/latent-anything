# Task Summary: Sprint 64 Task 3 — Stochastic evaluation metrics

**Sprint:** Sprint 64
**Task:** Evaluate NLL, calibration/coverage, sample diversity, and horizon drift.

## Summary of Work

Added typed one-step and rollout metrics for negative log-likelihood, interval coverage, interval width, particle diversity, mean-path error, runtime, and stability at each horizon.

## Files Modified

* [src/latent_anything/transition.py](../src/latent_anything/transition.py) — `StochasticOneStepMetrics` and `StochasticRolloutMetrics`.
* [tests/test_latent_anything/test_transition.py](../tests/test_latent_anything/test_transition.py) — metric assertions.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_transition.py -q`

## Additional Notes

Coverage is coordinate-wise central interval coverage; diversity is the mean particle standard deviation across state coordinates.
