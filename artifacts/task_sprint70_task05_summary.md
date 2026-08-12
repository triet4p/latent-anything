# Task Summary: Sprint 70 Task 5 — Replacement and interpolation policy

**Sprint:** Sprint 70
**Task:** Add policy tests for code replacement and unsupported continuous interpolation.

## Summary of Work

Added validated integer-to-integer `replace_codes` edits and explicit
`interpolate_codes` rejection. Tests cover invalid dtypes, invalid ranges,
replacement output preservation, adapter interpolation rejection, and the
same policy through `LatentSpace`.

## Files Modified

* `src/latent_anything/adapters/vq_vae.py` — categorical edit policy.
* `tests/test_latent_anything/test_vq_vae.py` — negative and positive policy tests.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_vq_vae.py -q`

## Additional Notes

No continuous arithmetic or rounding fallback is provided.
