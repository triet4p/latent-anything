# Task Summary: Sprint 73 Task 01 — Canonical Entry-Point Groups

**Sprint:** Sprint 73
**Task:** Define the canonical Python entry-point group vocabulary.

## Summary of Work

Added the five canonical external-plugin groups for adapter, analysis,
intervention, transition, and planner capabilities. The tuple of groups is
ordered and unique so later discovery can use a stable vocabulary. The
existing registry's `runtime` kind remains unchanged for transition and
planner built-ins; separate entry-point groups do not freeze a new registry
kind. RFC 0001's `intervention` term remains canonical for transformation-like
operations, with no parallel `transformation` group.

## Files Modified

* `src/latent_anything/plugin_groups.py` — canonical group constants and
  deterministic group tuple.
* `tests/test_plugin_groups.py` — uniqueness, order, namespace, and
  intervention-vocabulary tests.
* `docs/sprint-plans/sprint-73.md` — marked Task 1 complete.

## Testing

* **Test File:** `tests/test_plugin_groups.py`
* **Status:** Passed (2 tests)
* **Execution Command:** `uv run pytest tests/test_plugin_groups.py -q`
* **Additional checks:** `uv run ruff check src/latent_anything/plugin_groups.py tests/test_plugin_groups.py`; `uv run ruff format --check src/latent_anything/plugin_groups.py tests/test_plugin_groups.py` — both passed.

## Graphify

* **Command:** `graphify update .`
* **Status:** Passed. The refresh completed with AST extraction for 48/48
  uncached code files (100%). Graphify warned that 42 non-code JSON files
  produced zero nodes and were absent from the graph; this is an expected
  non-code indexing limitation and did not fail the update.

## Additional Notes

No public Protocols or top-level `latent_anything.__all__` exports were
widened. Sprint 73 Task 2 will define discovery metadata and loading behavior
against this vocabulary.
