# Task Summary: Sprint 53 Task 01 — Geometry-Compatible Smoothing

**Sprint:** Sprint 53
**Task:** Implement geometry-compatible smoothing with immutable metadata.

## Summary of Work

Added `smooth_trajectory()` with uniform/triangular centered windows. Euclidean data uses weighted averaging; other geometries use `LatentSpace.interpolate`. `Trajectory` now carries an immutable metadata mapping through slicing and smoothing.

## Files Modified

* `src/latent_anything/trajectory.py` — immutable metadata.
* `src/latent_anything/temporal.py` — smoothing implementation and result.
* `tests/test_temporal.py` — noise and unit-sphere tests.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_temporal.py -q`

