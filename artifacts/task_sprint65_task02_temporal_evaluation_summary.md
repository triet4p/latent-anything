# Task Summary: Sprint 65 Task 2 — Temporal evaluation

**Sprint:** Sprint 65
**Task:** Train/evaluate on a compact temporal dataset and report reconstruction/prediction, KL, calibration, and horizon drift.

## Summary of Work

Added masked temporal fit/evaluation with one-step MSE/RMSE, Gaussian NLL, observation-centred KL proxy, interval coverage, and open-loop horizon drift metrics.

## Files Modified

* [src/latent_anything/rssm.py](../src/latent_anything/rssm.py) — temporal fit/evaluation APIs.
* [scripts/rssm_transition_benchmark.py](../scripts/rssm_transition_benchmark.py) — reproducible compact temporal benchmark.
* [artifacts/rssm_transition_comparison.json](rssm_transition_comparison.json) — measured report.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run python scripts/rssm_transition_benchmark.py`

## Additional Notes

The benchmark is synthetic D2 evidence and reports the RSSM long-horizon failure rather than hiding it.
