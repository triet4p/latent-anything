# Task Summary: Sprint 70 Task 3 — Categorical code-sequence contract

**Sprint:** Sprint 70
**Task:** Preserve code sequences without silently converting to continuous values.

## Summary of Work

`encode` and `encode_value` preserve integer IDs. Continuous embeddings require
the explicitly named `code_embeddings` call, and decoding validates integer
code sequences. The latent metadata records the integer representation and
unsupported interpolation policy.

## Files Modified

* `src/latent_anything/adapters/vq_vae.py` — validation and explicit conversion APIs.
* `docs/VQ_VAE_INTEGRATION.md` — public representation contract.
* `tests/test_latent_anything/test_vq_vae.py` — dtype and conversion tests.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_vq_vae.py -q`

## Additional Notes

This follows the Sprint 30 discrete geometry decision: categorical IDs are not
treated as Euclidean coordinates.
