# Task Summary: Sprint 58 Task 6 — Observational ACT analysis

**Sprint:** Sprint 58
**Task:** Run projection/probe/trajectory analysis on successful and failed episodes with controls.

## Summary of Work

Added `analyze_act_traces()` for PCA projection, label-aware linear probing, majority/shuffled-label/raw-input controls, and Euclidean trajectory length/velocity summaries. The deterministic fixture benchmark covers both outcomes.

## Files Modified

* `src/latent_anything/integrations/lerobot_act.py` — typed analysis result and analysis function.
* `scripts/act_policy_representation_benchmark.py` — reproducible offline benchmark.
* `artifacts/act_policy_representation_benchmark.json` — generated metrics and controls.

## Testing

* **Test File:** `tests/test_lerobot_act.py`
* **Status:** Passed
* **Execution Command:** `uv run python scripts/act_policy_representation_benchmark.py`

## Additional Notes

The artifact explicitly records `causal_intervention: false`; it is not environment-level policy evidence.
