# Task Summary: Sprint 62 Task 1 — Versioned run record

**Sprint:** Sprint 62
**Task:** Define a reproducible, versioned run record.

## Summary of Work

Added schema-v1 `RunRecord` and `ArtifactRef` values covering config,
code/framework versions, model and dataset revisions, seeds, environment,
metrics, artifacts, parent/child links, lifecycle state, runtime profile, and
theory evidence identifiers. Stable identity hashes exclude lifecycle fields.

## Files Modified

* `src/latent_anything/run_record.py` — schema, identity, serialization, and migration.
* `src/latent_anything/__init__.py` — public exports.
* `tests/test_run_record.py` — round-trip and identity coverage.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run --no-sync pytest tests/test_run_record.py -q`
