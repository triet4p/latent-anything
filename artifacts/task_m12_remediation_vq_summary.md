# Task Summary: M12 remediation R1/R2 — non-degenerate VQ evidence

**Sprint:** Sprint 72 post-sprint remediation
**Task:** Diagnose/fix VQ-VAE collapse and regenerate Sprint 70 evidence.

## Summary of Work

The compact full-batch VQ-VAE could assign every encoded position to one random
code before unused entries received reconstruction gradients. Added deterministic
spread initialization from evenly spaced initial encoder outputs before the first
optimizer step. The acceptance gate now requires finite reconstruction,
perplexity strictly greater than `1.0`, and dead-code rate strictly below `1.0`.

## Files Modified

* `src/latent_anything/adapters/vq_vae.py` — deterministic codebook initialization.
* `scripts/vq_vae_digits_evidence.py` — explicit non-degenerate acceptance fields.
* `tests/test_latent_anything/test_vq_vae.py` — regression test for multi-code usage.
* `tests/test_vq_vae_benchmark.py` — benchmark acceptance assertions.
* `artifacts/vq_vae_digits_evidence.json` and config — regenerated pinned evidence.
* `.agents/memory/lessons-learned.md` — append-only root-cause and recurrence guidance.

## Testing

* **Focused tests:** `uv run pytest tests/test_latent_anything/test_vq_vae.py tests/test_vq_vae_benchmark.py -q` — **11 passed**.
* **Benchmark:** `uv run python scripts/vq_vae_digits_evidence.py` — pass.
* **Measured result:** perplexity `13.090496630645841`, dead-code rate `0.0`, discrete reconstruction MSE `0.17885987018827126`; all acceptance fields true.

## Additional Notes

This is D2 synthetic CPU evidence. It does not promote a real pretrained VQGAN,
healthy large-scale tokenizer, or CUDA result.
