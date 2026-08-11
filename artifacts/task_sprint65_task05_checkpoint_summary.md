# Task Summary: Sprint 65 Task 5 — Stateful checkpoint/config contract

**Sprint:** Sprint 65
**Task:** Define serialization/checkpoint/config contracts for stateful transitions.

## Summary of Work

Added validated `RSSMTransitionConfig`, `to_config()`, and portable `.npz` `save()`/`load()` methods containing learned parameters, scale, configuration, and fit provenance. Loading intentionally resets in-flight recurrent state.

## Files Modified

* [src/latent_anything/rssm.py](../src/latent_anything/rssm.py) — config and checkpoint lifecycle.
* [tests/test_latent_anything/test_transition.py](../tests/test_latent_anything/test_transition.py) — round-trip and reset assertions.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_transition.py -q`

## Additional Notes

Checkpoint loading can override the saved device when a compatible runtime is available.
