# Task Summary: Sprint 73 Task 05 — Separately Installed Fixture

**Sprint:** Sprint 73
**Task:** Install and exercise a minimal hello-world plugin distribution
without modifying core source.

## Summary of Work

Added a standalone `latent-anything-hello-plugin` fixture with its own
`pyproject.toml`, package module, version, and `latent_anything.adapter`
entry-point declaration. The integration test installs it into a temporary
target with `uv pip install --target --no-deps`, launches a child interpreter
with that target plus the repository's `src` path, and exercises metadata
listing, explicit loading, config construction, callable execution, and
provenance/version capture. The child confirms the fixture module is absent
before/listing and imported only after explicit loading.

## Files Modified

* `tests/fixtures/sprint73_hello_plugin/pyproject.toml` — independent fixture
  distribution metadata and entry point.
* `tests/fixtures/sprint73_hello_plugin/latent_anything_hello/__init__.py` —
  deterministic callable hello adapter.
* `tests/test_plugin_installation.py` — temporary fixture copy/install and
  clean-process integration proof; build metadata stays outside the checkout.
* `docs/sprint-plans/sprint-73.md` — marked fixture task complete.

## Testing

* **Test File:** `tests/test_plugin_installation.py`
* **Status:** Passed (1 test)
* **Execution Command:** `uv run pytest tests/test_plugin_installation.py -q`
  — 1 passed; the test itself ran `uv pip install --target ... --no-deps`.
* **Static checks:** Ruff check, Ruff format check, and strict Pyright for the
  integration test and fixture module — all passed (0 Pyright
  errors/warnings/informations).

## Graphify

* **Command:** `graphify update .`
* **Status:** Passed. AST extraction completed for 48/48 uncached code files
  (100%). Graphify warned that 42 non-code JSON files produced zero nodes and
  were absent from the graph; this expected indexing limitation did not fail
  the update. After the install-source isolation fix, a follow-up
  `graphify update .` also passed with AST extraction 46/46 (100%) and the same
  non-code JSON warning.

## Additional Notes

The fixture has no dependency on latent-anything and is installed only into a
temporary test target. No core source file is changed during installation.
