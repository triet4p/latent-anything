# Task Summary: Sprint 73 Remediation 01 — Deterministic Duplicate Ordering

**Sprint:** Sprint 73
**Task:** Audit remediation 01 — stable ordering for identical declarations

## Summary of Work

External entry-point sorting now uses canonical group order, plugin name,
stable distribution name/version metadata, and a declaration key derived only
from parsed entry-point fields. Sorting never calls ``EntryPoint.load`` or
property parsing that assumes a valid target declaration. A regression test
proves that two distributions declaring the same group/name/value produce the
same winner and issue ordering when the provider order is reversed.

## Files Modified

* `src/latent_anything/plugin_discovery.py` — metadata-only deterministic sort key.
* `tests/test_plugin_discovery.py` — distribution-aware fake metadata and reversed-provider regression.
* `docs/sprint-plans/sprint-73.md` — remediation task status.

## Testing

* **Test File:** `tests/test_plugin_discovery.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_plugin_discovery.py -q`
* **Result:** 8 passed in 3.81s
* **Additional checks:** `uv run ruff check src/latent_anything/plugin_discovery.py tests/test_plugin_discovery.py` passed; strict Pyright on both files reported 0 errors, 0 warnings, 0 informations.

## Graphify

* **Status:** Passed. `graphify update .` completed successfully after the task: 9,112 nodes, 18,013 edges, 814 communities; Graphify warned that 42 JSON source files produced zero nodes and rebuilt an aggregated view because the graph exceeded 5,000 nodes.

## Additional Notes

Distribution metadata is read without loading third-party targets. If two
metadata objects are completely identical in all supported declaration and
distribution fields, their issue content is identical; no object identity or
provider ordering is used as a tie-breaker.
