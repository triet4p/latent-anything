# Task Summary: Sprint 65 Task 6 — Sequence, device, and reproducibility tests

**Sprint:** Sprint 65
**Task:** Add sequence masks, variable-length, reset, device, and reproducibility tests.

## Summary of Work

Added masked variable-length fit/evaluation coverage, deterministic reset checks, seeded particle equality checks, checkpoint-state reset checks, and runtime device validation through the RSSM config.

## Files Modified

* [tests/test_latent_anything/test_transition.py](../tests/test_latent_anything/test_transition.py) — focused RSSM contract tests.
* [src/latent_anything/rssm.py](../src/latent_anything/rssm.py) — mask and device validation.

## Testing

* **Status:** Passed — 18 transition tests
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_transition.py -q`

## Additional Notes

CUDA is accepted only when PyTorch reports it available; the default reproducible lane is CPU.
