# Task Summary: Sprint 79 line 599 — full contract verification matrix

**Sprint:** Sprint 79
**Task:** line 599 (Verify all 202 exports, 32 built-in registry entries, 5 entry-point groups,
12 optional profiles, CLI commands, schema migrations, negative/security cases, sync/async paths,
cross-adapter composition, external plugin install/discovery, cache, streaming, and tracking backends)

## Summary of Work

Exercised every named line-599 category end-to-end against the authoritative contracts
(`artifacts/api_freeze_snapshot_0.1.0b1.json`, `docs/API_REFERENCE.md`, `docs/MIGRATION.md`,
`docs/API_COMPATIBILITY.md`, `pyproject.toml`, `.github/workflows/optional-extras.yml`) after a
graphify-first query. Used the existing verification matrix only — no parallel convention invented.
All focused suites pass, live behavioral proofs confirm sync/async parity, cross-adapter composition,
cache round-trips, serialization migrations, CLI commands/aliases, and real configured
cache/streaming/tracking backends. The 205-current vs 202-canonical distinction is treated exactly as
documented (3 additive symbols: `AnalysisMethod`, `Intervention`, `InterventionPipeline`). Optional
profiles and external-plugin discovery are proven in isolated/disposable environments with cleanup.
No blocker found, so no source, threshold, snapshot, or ledger change was made. Lines 595/598 stay
unchecked with prior accepted counts preserved (41/63 core, 41/65 overall).

## Files Modified

* [artifacts/task_79_line599_verification_summary.md](task_79_line599_verification_summary.md) — this
  reproducible summary (new file; only file changed)
* `docs/sprint-plans/sprint-79.md` — line 599 checked (single checkbox flip)

## Testing (exact commands and results)

Authoritative snapshot gate:

* `uv run python scripts/api_freeze_snapshot.py --check` — PASS,
  `snapshot clean (48d64721b73a9d0c9e73da4a41940008c70dfa7841e500bc11bc8dcd22ddf7f6)`.
* `uv run pytest tests/test_api_freeze_snapshot.py tests/test_api_compatibility.py
  tests/test_api_surface.py -q` — **18 passed**. Pins 205 current / 202 canonical-stable / 18 alias
  rows / 32 registry / 5 plugin groups / 12 profiles / 28 configs / 9 sync-async pairs / 7 exceptions.

Category matrix (all focused, all behavior-based, none source-text):

| Category | Exact command | Result |
|---|---|---|
| 202 exports / aliases | above API trio + `test_api_surface` identity checks | 18 passed; 205-live vs 202-canonical distinction exact |
| 32 registry entries | `uv run pytest tests/test_latent_anything/test_registry.py tests/test_latent_anything/test_runtime_async.py tests/test_latent_anything/test_rollout_pipeline.py tests/test_latent_anything/test_pipeline.py -q` → **120 passed** (= registry standalone **48** + async trio **72**); live `GLOBAL_REGISTRY.list()` | live count **32** (`analysis 10, runtime 9, adapter 8, intervention 5`) |
| 5 entry-point groups | `uv run pytest tests/test_plugin_groups.py tests/test_plugin_metadata.py tests/test_plugin_discovery.py -q` + live snapshot cross-check | **14 passed**; groups `latent_anything.{adapter,analysis,intervention,transition,planner}`, API v1 |
| 12 optional profiles | committed matrix `task_sprint79_clean_environment_matrix_summary.md`: 39/39 local lanes + Ubuntu Actions run `34681280312` (required `clean-base` + `resolve-extra` 39/39 success); live `pyproject`↔snapshot order match `MATCH: True`; disposable base env `BASE_ISOLATION: OK` + `CLEANED` | PASS, no model/data/CUDA invoked |
| CLI (5 commands + 2 aliases) | `uv run pytest tests/test_cli.py -q` → **5 passed**; combined `tests/test_cli.py tests/test_plugin_groups.py tests/test_plugin_metadata.py tests/test_plugin_discovery.py -q` → **19 passed, 0 skipped**; plus live `cli --help`, `capture-points --policy act` (exit 0), `list-capture-points` alias parity, `compare-runs --help` | PASS |
| Schema migrations (2) | `uv run pytest tests/test_portable.py tests/test_portable_results.py tests/test_run_record.py tests/test_run_record_portable.py -q` + live `decode_result_envelope`/`migrate_run_record` callable + portable round-trip | **39 passed** (separate disk-cache batch: `test_disk_cache` + `test_cache` + `test_portable` → **36 passed**) |
| Negative/security | `tests/test_artifact_store.py` (**5 passed** standalone: traversal/symlink/checksum) + `tests/test_disk_cache.py` (symlink/corrupt/oversized/key-shape) + `tests/test_experiment_recorder.py` (**29 passed** standalone: unsafe names/secrets/oversize) + portable allocation guard | behavior PASS as executed code paths, not source-text checks |
| Sync/async (9 pairs) | `uv run pytest tests/test_latent_anything/test_runtime_async.py tests/test_latent_anything/test_rollout_pipeline.py tests/test_latent_anything/test_pipeline.py -q` → **72 passed**; registry standalone `tests/test_latent_anything/test_registry.py -q` → **48 passed**; plus live `AnalysisPipeline.run`/`run_async` parity `True`, `RolloutPipeline.run`/`run_async` parity `True` | PASS; live parity True |
| Cross-adapter composition | live `build_from_config(ObjectSpec(adapter, random_projection))` → `RandomProjection (20, 4)` encode; `ManipulationPipeline` staged path in suites | PASS |
| External plugin install/discovery | `uv run pytest tests/test_plugin_installation.py -q` (separate install into `tmp/installed-plugin`, clean child `PYTHONPATH`, provenance assertions) | **1 passed**; target dir is test-tmp, nothing retained |
| Cache (real backends) | `uv run pytest tests/test_disk_cache.py tests/test_latent_anything/test_cache.py tests/test_portable.py -q` → **36 passed**; live `SQLiteDiskCache` round-trip `True`, `InMemoryCache` round-trip `True` with `CacheStats(hits=1, misses=0, sets=1, size=1)` | PASS; live True |
| Streaming (sync + async) | `uv run pytest tests/test_sprint75_streaming.py -q` → **1 passed** standalone; live `stream`/`stream_async` chunk counts `1`/`1` on fitted transition | PASS |
| Tracking backends | `uv run pytest tests/test_tracking_parity.py tests/test_mlflow_recorder.py tests/test_wandb_recorder.py tests/test_experiment_recorder.py tests/test_sprint75_streaming.py -q` → **57 passed, 2 skipped**; with `tests/test_artifact_store.py` added → **62 passed, 2 skipped** | The 2 skips are the same optional real-backend tests (`test_mlflow_real_local_file_store_when_extra_is_enabled`, `test_wandb_real_offline_when_extra_is_enabled`; missing `mlflow`/`wandb` in base env); required fallback/parity contracts passed; live `LocalExperimentRecorder` run/log/finish OK (file/offline modes only, per project policy) |
| Config/planner contracts | `test_config test_mppi test_mppi_rollout test_cem test_cem_rollout test_transition` | **86 passed** |

Two live-probe syntax slips during exploration (a one-line `python -c` conditional import and a wrong
`RandomProjection` kwarg) were operator errors in throwaway probes, not product failures; corrected
probes pass and no product code was touched.

## Additional Notes

* No code, snapshot, ledger, map, or queue change: the full matrix passes on the committed surface.
* Graphify queried first for the verification matrix; no code changed, so no graph rebuild was needed.
* Branch left clean; this evidence-only commit pushed to `sprint79-local-gate-remediation`.
* Lines 595/598 remain `[ ]` unchecked; prior accepted counts unchanged (41/63 core, 41/65 overall).
* Line 599 is checked by this task since the entire stated matrix passes with explicit evidence above.
* Line 600 (performance budgets / LeRobot policy overhead vs Sprint 77 gates) is independently
  actionable and may proceed next; it was not started here.
