# Task Summary: Sprint 63 Task 3 — One-step fit and recursive rollout

**Sprint:** Sprint 63
**Task:** Add one-step training/evaluation and recursive multi-step rollout behavior.

## Summary of Work

`DeterministicLatentTransition.fit()` learns one-step residual affine coefficients with deterministic ridge least squares. `evaluate_one_step()` reports teacher-forced error and runtime, while `rollout()` recursively feeds predictions back into the transition and returns the initial state plus one predicted state for every action.

## Files Modified

* `src/latent_anything/transition.py` — fit, step/predict, one-step metrics, and recursive rollout.
* `tests/test_latent_anything/test_transition.py` — identity, action-conditioned linear, and rollout tests.

## Testing

* **Test File:** `tests/test_latent_anything/test_transition.py`
* **Status:** Passed — 7 tests
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_transition.py -q`

## Additional Notes

The training contract is explicitly one-step in this first increment; `training_horizon` is provenance, not an unsupported claim of multi-step optimization.
