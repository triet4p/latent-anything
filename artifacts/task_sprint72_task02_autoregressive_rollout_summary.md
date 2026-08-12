# Task Summary: Sprint 72 Task 2 — Autoregressive prediction and rollout

**Sprint:** Sprint 72
**Task:** Implement autoregressive next-token prediction and seeded rollout

## Summary of Work

Implemented teacher-forced training and left-to-right next-frame generation with greedy or seeded categorical sampling, temperature and top-k controls, plus deterministic greedy `step`/`mean_rollout` and seeded concrete `rollout` methods.

## Files Modified

* [src/latent_anything/tokenized_world_model.py](../src/latent_anything/tokenized_world_model.py) - GRU encoder/decoder, token sampling, and rollout paths.
* [tests/test_latent_anything/test_tokenized_world_model.py](../tests/test_latent_anything/test_tokenized_world_model.py) - Seed parity and pipeline rollout tests.

## Testing

* **Test File:** [tests/test_latent_anything/test_tokenized_world_model.py](../tests/test_latent_anything/test_tokenized_world_model.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_tokenized_world_model.py -q`

## Additional Notes

Sampling is intentionally concrete; the frozen transition contract remains predictive-mean only.
