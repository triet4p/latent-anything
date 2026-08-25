# Task Summary: Sprint 73 Task 07 — Plugin Compatibility Contract

**Sprint:** Sprint 73
**Task:** Add API/version compatibility checks and clear unsupported-contract
errors.

## Summary of Work

Defined the minimal external target metadata contract version (`"1"`) and
required callable attribute `__latent_anything_plugin_api_version__`. Explicit
loading rejects missing or mismatched versions as an isolated
`PluginContractError`, reports the plugin name and required value, and keeps
processing other declarations. Successful registrations persist the API
version alongside distribution and entry-point provenance. The fixture now
declares the supported version on its callable class target.

## Files Modified

* `src/latent_anything/plugin_metadata.py` — supported API version constants.
* `src/latent_anything/plugin_discovery.py` — compatibility validation and
  `PluginContractError`.
* `tests/test_plugin_discovery.py` — unsupported-version isolation proof.
* `tests/fixtures/sprint73_hello_plugin/latent_anything_hello/__init__.py` and
  `pyproject.toml` — compatible class target declaration.
* `docs/PLAN.md` and `docs/sprint-plans/sprint-73.md` — live status updates.

## Testing

* **Test Files:** `tests/test_plugin_discovery.py`,
  `tests/test_plugin_installation.py`
* **Status:** Passed (8 tests combined)
* **Execution Command:** `uv run pytest tests/test_plugin_discovery.py tests/test_plugin_installation.py -q`
  — 8 passed.
* **Static checks:** Ruff check, Ruff format check, and strict Pyright across
  changed discovery/metadata/fixture source/tests — all passed (0 Pyright
  errors/warnings/informations).

## Graphify

* **Command:** `graphify update .`
* **Status:** Passed. AST extraction completed for 50/50 uncached code files
  (100%). Graphify warned that 42 non-code JSON files produced zero nodes and
  were absent from the graph; this expected indexing limitation did not fail
  the update.

## Additional Notes

The compatibility marker is metadata on an already-callable plugin target,
not a new runtime Protocol or workflow abstraction. Built-in registrations are
not required to carry it because only explicit external entry-point loading
enforces this external-distribution contract.
