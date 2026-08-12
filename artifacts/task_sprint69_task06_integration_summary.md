# Task Summary: Sprint 69 Task 6 — Config, registry, records, and profiling

**Sprint:** Sprint 69
**Task:** Add config/registry/experiment records and profiler integration.

## Summary of Work

Added `MPPIPlannerSpec`, config construction, the `mppi_planner` runtime registry entry, top-level public exports, `FileSystemRunRecorder.complete_mppi_plan()`, content-addressed JSON plan artifacts, sample/ESS metrics, and evaluation/planning/transition profiling.

## Files Modified

* `src/latent_anything/pipeline_config.py`, `src/latent_anything/pipeline.py` — config and compatibility exports.
* `src/latent_anything/_plugin_builtins.py`, `src/latent_anything/__init__.py` — registry and API exports.
* `src/latent_anything/run_record.py` — MPPI record completion helper.
* `tests/test_mppi_integration.py` — builder, registry, artifact, and metric tests.

## Testing

* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_mppi_integration.py -q`

## Additional Notes

The existing run-record schema remains unchanged.
