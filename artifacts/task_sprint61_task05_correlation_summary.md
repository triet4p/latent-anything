# Task Summary: Sprint 61 Task 5 — Offline-explanation correlation

**Sprint:** Sprint 61
**Task:** Correlate offline explanation scores with environment-level causal effects and report disagreements.

## Summary of Work

`run_simulation_benchmark` records policy-ready probe samples from the first `no_hook` episode and measures the offline explanation scores for every non-zero intervention cell with `measure_smolvla_intervention` (on-target fraction, action-change norm, representation drift). `CausalCorrelationCell` pairs each offline score with the environment-level mean action deviation and success delta vs the no-hook rate. `build_correlation` applies the predeclared disagreement rules — overstatement (on-target ≥ 0.8 but |success delta| < 0.2), understatement (on-target < 0.5 but |success delta| ≥ 0.2), reversal (success delta ≤ −0.2) — and reports Spearman only when ≥ 3 cells with variance exist (rank-based implementation, no scipy).

## Files Modified

* [src/latent_anything/integrations/lerobot_benchmark.py](src/latent_anything/integrations/lerobot_benchmark.py) - `OfflineExplanationScore`, `CausalCorrelationCell`, `CausalCorrelation`, `build_correlation`, `_spearman`.

## Testing

* **Test File:** [tests/test_lerobot_benchmark.py](tests/test_lerobot_benchmark.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_benchmark.py -v`

## Additional Notes

The fixture demonstrates an honest overstatement disagreement: the targeted direction is 99% on-target offline yet the environment success is unchanged, exactly the "explain for completeness" failure mode the benchmark exists to catch.
