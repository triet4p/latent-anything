# Task Summary: Sprint 66 — Public exports and migration docs

**Sprint:** Sprint 66
**Task:** Update public exports/migration docs and measure module complexity reduction.

## Summary of Work

Exported Rollout classes, result/spec models, contract, and builder from the existing beta surface. Added `docs/PIPELINES.md`, linked it from the document index, and reduced `pipeline.py` from 825 lines to a 45-line compatibility shim.

## Files Modified

* [src/latent_anything/__init__.py](/F:/ai-ml/latent-anything/src/latent_anything/__init__.py) - Public exports.
* [docs/PIPELINES.md](/F:/ai-ml/latent-anything/docs/PIPELINES.md) - Migration documentation.
* [docs/INDEX.md](/F:/ai-ml/latent-anything/docs/INDEX.md) - Document map.

## Testing

* **Test File:** [tests/test_api_surface.py](/F:/ai-ml/latent-anything/tests/test_api_surface.py)
* **Status:** Pending final full gate
* **Execution Command:** `uv run pytest tests/test_api_surface.py -q`

## Additional Notes

The focused pipeline suite already verifies compatibility imports; the full API gate is run at sprint completion.
