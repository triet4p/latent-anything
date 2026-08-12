# Task Summary: Sprint 71 Task 1 — JEPA provenance

**Sprint:** Sprint 71
**Task:** Select and pin the JEPA/LeWM-style reference.

## Summary of Work

Selected the reproducible compact reference `compact-jepa-lewm-v1` over
`synthetic-controlled-latent-dynamics-v1`, and pinned the opt-in public
I-JEPA smoke to `facebook/ijepa_vith14_1k` revision
`be440b1cac639542ae553e71a9c7afd925ab5fac`.

## Files Modified

* [src/latent_anything/adapters/jepa.py](/F:/ai-ml/latent-anything/src/latent_anything/adapters/jepa.py) — provenance constants.
* [tests/test_latent_anything/test_jepa_checkpoint.py](/F:/ai-ml/latent-anything/tests/test_latent_anything/test_jepa_checkpoint.py) — marked checkpoint smoke.
* [docs/sprint-plans/sprint-71.md](/F:/ai-ml/latent-anything/docs/sprint-plans/sprint-71.md) — pinned provenance.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_jepa.py -q`

## Additional Notes

The public checkpoint lane is network/large-download opt-in and does not define the compact adapter implementation.
