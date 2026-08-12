# Task Summary: Sprint 72 Task 5 — Token validation

**Sprint:** Sprint 72
**Task:** Add invalid-token, mask, padding, horizon, and codebook-version tests

## Summary of Work

Added strict integer-ID validation, an explicit padding token for masked training transitions, binary sequence-mask validation, minimum-horizon checks, and frozen tokenizer codebook-version binding. Focused tests cover invalid IDs, masked padded frames, version mismatches, and rollout shapes.

## Files Modified

* [src/latent_anything/tokenized_world_model.py](../src/latent_anything/tokenized_world_model.py) - Boundary validation and masked training support.
* [tests/test_latent_anything/test_tokenized_world_model.py](../tests/test_latent_anything/test_tokenized_world_model.py) - Failure-contract coverage.

## Testing

* **Test File:** [tests/test_latent_anything/test_tokenized_world_model.py](../tests/test_latent_anything/test_tokenized_world_model.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_tokenized_world_model.py -q`

## Additional Notes

Padding is accepted only in sequence-training inputs and is never emitted by the categorical predictor.
