# Task Summary: Sprint 72 Task 4 — Teacher-forced versus free-running behavior

**Sprint:** Sprint 72
**Task:** Compare teacher-forced and free-running behavior

## Summary of Work

Evaluation now reports teacher-forced cross-entropy/perplexity separately from greedy free-running horizon drift. The benchmark additionally records seeded temperature-1.8 sampling error by horizon, making the distinction between local next-token fit and rollout uncertainty explicit.

## Files Modified

* [src/latent_anything/tokenized_world_model.py](../src/latent_anything/tokenized_world_model.py) - Separate teacher-forced and recursive evaluation paths.
* [scripts/tokenized_world_model_benchmark.py](../scripts/tokenized_world_model_benchmark.py) - Sampled rollout stress comparison.
* [artifacts/tokenized_world_model_evidence.json](tokenized_world_model_evidence.json) - Per-horizon comparison and failure fields.

## Testing

* **Test File:** [tests/test_latent_anything/test_tokenized_world_model.py](../tests/test_latent_anything/test_tokenized_world_model.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_tokenized_world_model.py -q`

## Additional Notes

The compact controlled task has zero greedy drift, while sampled error rises from 0.074 at horizon 1 to 0.229 at horizon 8. The artifact records this as a limited synthetic result, not a real-model claim.
