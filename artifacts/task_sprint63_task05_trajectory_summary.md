# Task Summary: Sprint 63 Task 5 — Immutable rollout trajectories

**Sprint:** Sprint 63
**Task:** Return immutable latent trajectories compatible with DTW/segmentation analysis.

## Summary of Work

Recursive prediction returns the existing immutable `Trajectory` primitive. Its copied state array and read-only metadata preserve compatibility with existing DTW and temporal-analysis consumers while adding explicit predicted-state, source identity, shape, transition, and horizon provenance.

## Files Modified

* `src/latent_anything/transition.py` — `rollout()` trajectory construction and provenance.
* `tests/test_latent_anything/test_transition.py` — trajectory type, values, and metadata assertions.

## Testing

* **Test File:** `tests/test_latent_anything/test_transition.py`
* **Status:** Passed — 7 tests
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_transition.py -q`

## Additional Notes

No new trajectory abstraction or mutation path was introduced.
