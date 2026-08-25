# Task Summary: Sprint 73 Task 02 — Entry-Point Metadata Contract

**Sprint:** Sprint 73
**Task:** Define metadata and capability-to-registry mapping without loading
plugin code.

## Summary of Work

Added `EntryPointMetadata`, a frozen metadata record extracted from standard
library `importlib.metadata.EntryPoint` declarations without calling
`EntryPoint.load()`. It records entry-point name/group/value, optional
distribution name/version, and the existing registry kind. Adapter, analysis,
and intervention map directly to their canonical registry kinds. Transition
and planner use distinct discovery groups while preserving their currently
proven `runtime` registry kind. Unsupported groups fail with the supported
group list.

## Files Modified

* `src/latent_anything/plugin_metadata.py` — metadata record and group mapping.
* `tests/test_plugin_metadata.py` — no-load extraction, runtime mapping, and
  unsupported-group tests.
* `docs/sprint-plans/sprint-73.md` — marked the metadata subtask complete.

## Testing

* **Test Files:** `tests/test_plugin_groups.py`,
  `tests/test_plugin_metadata.py`
* **Status:** Passed (5 tests)
* **Execution:** `uv run pytest tests/test_plugin_groups.py tests/test_plugin_metadata.py -q`
  — 5 passed.
* **Static checks:** Ruff check passed; Ruff format check passed after
  formatting `plugin_metadata.py`; strict Pyright passed with 0 errors, 0
  warnings, and 0 informations for the four changed source/test files.

## Graphify

* **Command:** `graphify update .`
* **Status:** Passed. AST extraction completed for 47/47 uncached code files
  (100%). Graphify warned that 42 non-code JSON files produced zero nodes and
  were absent from the graph; this expected indexing limitation did not fail
  the update.

## Additional Notes

This task adds no Protocol, dependency, top-level export, or new registry kind.
Actual plugin loading, duplicate policy, and failure isolation are the next
atomic task.
