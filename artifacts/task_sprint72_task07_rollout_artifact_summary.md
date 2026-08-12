# Task Summary: Sprint 72 Task 7 — Reproducible rollout artifact

**Sprint:** Sprint 72
**Task:** Produce a reproducible rollout artifact with failure horizons

## Summary of Work

Added an offline benchmark pinned to sklearn digits for tokenizer provenance and a deterministic synthetic codebook dynamics task. It writes a JSON evidence artifact and config containing model/codebook revisions, fit state, teacher-forced metrics, per-horizon greedy and sampled rollout errors, decoder consistency, task-proxy accuracy, seeded parity, and failure-horizon fields.

## Files Modified

* [scripts/tokenized_world_model_benchmark.py](../scripts/tokenized_world_model_benchmark.py) - Reproducible CPU benchmark.
* [artifacts/tokenized_world_model_evidence.json](tokenized_world_model_evidence.json) - Rollout evidence.
* [artifacts/tokenized_world_model_evidence_config.json](tokenized_world_model_evidence_config.json) - Pinned run configuration.

## Testing

* **Test File:** [tests/test_latent_anything/test_tokenized_world_model.py](../tests/test_latent_anything/test_tokenized_world_model.py)
* **Status:** Passed
* **Execution Command:** `uv run python scripts/tokenized_world_model_benchmark.py`

## Additional Notes

The artifact is offline D2 synthetic CPU evidence. It explicitly retains the tokenizer's dead-code diagnostic and does not claim CUDA or real-world checkpoint evidence.
