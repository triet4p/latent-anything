# Task Summary: Sprint 67 Task 5 — Held-out diagnostics

**Sprint:** Sprint 67
**Task:** Measure reward prediction, value calibration, and Bellman residual on held-out trajectories.

## Summary of Work

Added `HoldoutEvaluation`, `ValueCalibration`, and `RewardValueDiagnostics`. The evaluator reports reward RMSE/MAE/bias, value calibration error, and Bellman residual RMSE/MAE/bias under the declared finite-horizon return contract.

## Files Modified

* [src/latent_anything/reward_value.py](/F:/ai-ml/latent-anything/src/latent_anything/reward_value.py) - Diagnostics implementation.
* [scripts/reward_value_benchmark.py](/F:/ai-ml/latent-anything/scripts/reward_value_benchmark.py) - Reproducible held-out benchmark.
* [artifacts/reward_value_evaluation.json](/F:/ai-ml/latent-anything/artifacts/reward_value_evaluation.json) - Benchmark result.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run python scripts/reward_value_benchmark.py`

## Additional Notes

The benchmark reports the finite-horizon value calibration error rather than hiding it behind reward-fit quality.
