# Task Summary: Sprint 72 Task 3 — Tokenized evidence metrics

**Sprint:** Sprint 72
**Task:** Measure likelihood, code usage, drift, decoded consistency, and task proxies

## Summary of Work

Added typed teacher-forced likelihood/perplexity and token-accuracy metrics, code usage/perplexity/dead-code summaries, free-running token drift and failure horizons, decoder-backed consistency metrics, and an explicit decoded task-proxy callback. The reproducible benchmark records deterministic and sampled rollout metrics together with tokenizer health.

## Files Modified

* [src/latent_anything/tokenized_world_model.py](../src/latent_anything/tokenized_world_model.py) - Typed evidence results and evaluation helpers.
* [scripts/tokenized_world_model_benchmark.py](../scripts/tokenized_world_model_benchmark.py) - Offline benchmark and artifact writer.
* [artifacts/tokenized_world_model_evidence.json](tokenized_world_model_evidence.json) - Measured evidence.
* [artifacts/tokenized_world_model_evidence_config.json](tokenized_world_model_evidence_config.json) - Reproduction configuration.

## Testing

* **Test File:** [tests/test_latent_anything/test_tokenized_world_model.py](../tests/test_latent_anything/test_tokenized_world_model.py)
* **Status:** Passed
* **Execution Command:** `uv run python scripts/tokenized_world_model_benchmark.py`

## Additional Notes

The benchmark is D2 synthetic CPU evidence. The frozen VQ tokenizer still reports severe dead-code usage; the dynamics result does not promote healthy large-scale tokenizer evidence.
