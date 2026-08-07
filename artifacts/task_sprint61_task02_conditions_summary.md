# Task Summary: Sprint 61 Task 2 — Benchmark conditions

**Sprint:** Sprint 61
**Task:** Define baseline, random intervention, targeted intervention, and no-hook control conditions.

## Summary of Work

Defined the four-condition protocol in `lerobot_benchmark.py`: `no_hook` (official preprocess → `select_action` → postprocess without capture hooks), `baseline` (hooks with strength-zero intervention — bit-exact identity), `random` (seeded unit-norm action-expert direction via `np.random.default_rng(config.intervention_seed)`), and `targeted` (the expert direction inducing the largest change along `config.action_axis` through the policy's own `action_out_proj`). The `targeted` direction is the minimum-norm solution `W[:action_dim].T @ e_axis` normalized, i.e. the most on-target intervention the offline explanation can name. Directions are shared across episodes; strengths are configurable (non-zero, finite, bounded by the adapter's `max_strength`).

## Files Modified

* [src/latent_anything/integrations/lerobot_benchmark.py](src/latent_anything/integrations/lerobot_benchmark.py) - `BenchmarkCondition`, condition validation, `_random_expert_direction`, `_targeted_expert_direction`.

## Testing

* **Test File:** [tests/test_lerobot_benchmark.py](tests/test_lerobot_benchmark.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_lerobot_benchmark.py -v`

## Additional Notes

The targeted direction on the linear fixture achieves on-target fraction >= 0.99 offline, proving the direction math; the random condition provides the off-target control.
