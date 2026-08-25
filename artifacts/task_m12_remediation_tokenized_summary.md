# Task Summary: M12 remediation R3 — tokenized world-model evidence

**Sprint:** Sprint 72 post-sprint remediation
**Task:** Regenerate tokenized dynamics from the fitted non-collapsed tokenizer.

## Summary of Work

Reran the raw-observation Sprint 72 benchmark after the VQ-VAE training fix. The
tokenized model now receives eight active train and held-out codes rather than a
constant sequence. The artifact remains explicit about weak learned dynamics:
greedy token error is `0.7265625` at horizon 1 and sampled failure horizon is 1.

## Files Modified

* `scripts/tokenized_world_model_benchmark.py` — non-collapsed and collapsed
  failure-analysis branches; no acceptance threshold relaxation.
* `artifacts/tokenized_world_model_evidence.json` and config — regenerated evidence.
* `docs/TOKENIZED_WORLD_MODEL.md`, `docs/EVIDENCE_LEDGER.md`, and
  `docs/evidence-ledger.json` — D2 synthetic status and limitations.

## Testing

* **Focused tests:** `uv run pytest tests/test_latent_anything/test_tokenized_world_model.py tests/test_latent_anything/test_vq_vae.py tests/test_vq_vae_benchmark.py -q` — **18 passed**.
* **Benchmark:** `uv run python scripts/tokenized_world_model_benchmark.py` — pass.
* **Measured result:** teacher-forced perplexity `2.684111787469027`, tokenizer
  perplexity `5.50977280035793`, dead-code rate `0.0`, active train/held-out
  codes `8/8`, all acceptance fields true, evidence status `D2`.

## Additional Notes

This is meaningful compact synthetic CPU evidence, not a real checkpoint, CUDA,
GAIA, Genie, or large-scale world-model claim.
