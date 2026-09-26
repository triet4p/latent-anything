# Task 80.6 — Define the High-Level Diagnostic Request/Result API

## Status

**Complete.** Sprint 80 tasks 80.1–80.6 are marked `[x]`; tasks 80.7–80.29 remain pending. Historical Sprint 79 evidence was not changed.

## Summary of Work

Added `src/latent_anything/diagnostics.py`, the smallest public domain-level configuration surface for the Sprint 80 workflow: `DiagnosticRequest` (request id, manifest id, capture selection, diagnostic family selection, declared controls, intervention requests, aligned comparisons, output location/artifact intent) plus `DiagnosticResult` (request/manifeset identity, resumable status, completed workflow stages, artifact references, report id, message, stage results). All selection types use domain terms only — capture identity, representation identity, axis names, family ids, control/metric ids, intervention targets, run ids, and output paths — with no transformer layers, attention heads, VAE internals, or other architecture-specific fields. Construction is fail-closed (unknown families, duplicate intervention/comparison ids, undeclared control/metric references, identical comparison runs, unknown result stages, malformed mappings, absolute or parent-escaping output locations). No stage execution, orchestration, detector, or benchmark-run logic was added; the module only constructs, validates, and round-trips requests/results. Exported the nine public symbols through `latent_anything/__init__.py` without touching existing exports, and recorded the durable contract choice in `.agents/memory/decisions.md`.

## Public Surface

- `latent_anything.diagnostics.DiagnosticRequest` with `to_dict`/`from_dict`.
- `latent_anything.diagnostics.DiagnosticResult` with `to_dict`/`from_dict`.
- `CaptureSelection`, `DiagnosticSelection`, `ControlSelection`, `InterventionRequest`, `ComparisonRequest`, `OutputSelection`, `DiagnosticRequestError` in `latent_anything.diagnostics`.
- `ResultStatus` stays available from `latent_anything.diagnostics` only and is intentionally not a top-level re-export.
- Exactly nine names (all of the above except `ResultStatus`) are re-exported from `latent_anything` and present in `__all__`.

## Validation Evidence

Focused validation only (no formatter, linter, type checker, packaging, documentation build, or project-wide suite was run):

```text
uv run pytest tests/test_diagnostics_request.py -q
14 passed in 3.03s

uv run pytest tests/test_diagnostics_request.py tests/test_sprint80_core_manifests.py -q
19 passed in 3.07s

uv run python scripts/sprint80_task80_6_smoke.py
PASS request smoke-encoder-80-6 -> sprint80-core-encoder-autoencoder-collapse-v1
PASS request smoke-transformer-80-6 -> sprint80-core-transformer-hidden-state-probe-v1
PASS result running(capture,detect)
REJECT unknown-family: unknown diagnostic families: not_a_family
REJECT undeclared-control: intervention bad references undeclared controls: control-missing
REJECT unsafe-output: output_location must be a repository-relative path
exit code 0
```

Correction evidence for this round (CrossSeedReport restoration plus exact public-surface wording):

```text
uv run python -c "import latent_anything; ..."
EXPORT CHECK PASS: CrossSeedReport restored; 9 diagnostics top-level; ResultStatus diagnostics-only; no duplicates; total 213

uv run pytest tests/test_diagnostics_request.py -q
14 passed in 3.02s

uv run python scripts/sprint80_task80_6_smoke.py
PASS request smoke-encoder-80-6 -> sprint80-core-encoder-autoencoder-collapse-v1
PASS request smoke-transformer-80-6 -> sprint80-core-transformer-hidden-state-probe-v1
PASS result running(capture,detect)
REJECT unknown-family: unknown diagnostic families: not_a_family
REJECT undeclared-control: intervention bad references undeclared controls: control-missing
REJECT unsafe-output: output_location must be a repository-relative path
smoke exit 0

uv run python scripts/api_freeze_snapshot.py --check | grep -c "^drift:"
2833
```

The freeze drift total (2833 lines) is dominated by bulk per-symbol rows for the nine new public symbols (A current_top_level, A canonical_stable_surface, G schemas, H schemas); the remaining compact signal is section digests/counts plus L-exceptions index shifts. No snapshot file was regenerated.

## Correction Round 3 — CovarianceConfig Restoration and Baseline Proof

The re-review found `CovarianceConfig` missing from `__all__` while its import (`from latent_anything.covariance import CovarianceConfig as CovarianceConfig`) remained — the same accidental pattern as the earlier `CrossSeedReport` removal. Corrected with a single `__all__` entry restoration; no import or other symbol was touched.

```text
git diff src/latent_anything/__init__.py  # vs HEAD baseline
# __all__ delta only: +9 diagnostics names, -CovarianceConfig (before fix)
# after fix: +9 diagnostics names, zero removals

uv run python -c "exec HEAD baseline __all__; compare with worktree __all__"
added: ['CaptureSelection', 'ComparisonRequest', 'ControlSelection', 'DiagnosticRequest', 'DiagnosticRequestError', 'DiagnosticResult', 'DiagnosticSelection', 'InterventionRequest', 'OutputSelection']
removed: []
added_count: 9 removed_count: 0
BASELINE PROOF PASS: only 9 diagnostics added, 0 removed; total 214

uv run pytest tests/test_diagnostics_request.py -q
14 passed in 3.05s

uv run python scripts/sprint80_task80_6_smoke.py
PASS request smoke-encoder-80-6 -> sprint80-core-encoder-autoencoder-collapse-v1
PASS request smoke-transformer-80-6 -> sprint80-core-transformer-hidden-state-probe-v1
PASS result running(capture,detect)
REJECT unknown-family: unknown diagnostic families: not_a_family
REJECT undeclared-control: intervention bad references undeclared controls: control-missing
REJECT unsafe-output: output_location must be a repository-relative path
smoke exit 0

uv run python scripts/api_freeze_snapshot.py --check | grep -c "^drift:"
2475
uv run python scripts/api_freeze_snapshot.py --check | grep "^drift:" | grep -c "CrossSeedReport\|CovarianceConfig"
0

uv run pytest tests/test_api_surface.py -q
1 failed, 3 passed
# The single failure is the pinned-__all__ ordering assertion, failing by design:
# "At index 13 diff: 'CaptureSelection' != 'CacheStats'", "Left contains 9 more items",
# with exactly the nine +lines above and zero -lines (no removed baseline symbols).
# The snapshot file itself was not regenerated, per sprint rules.
```

## Graph Refresh Record (Correction Round 3, Final)

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/255 uncached files (39%) [8 workers]
  AST extraction: 200/255 uncached files (78%) [8 workers]
  AST extraction: 255/255 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph: hooks.json, evals.json, evals.json, act_policy_representation_benchmark.json, anisotropy_benchmark.json (+243 more). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1062 saved labels, 1025 communities now; renamed 171 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 13907 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1025 community nodes, 1326 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 13907 nodes, 28840 edges, 1025 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit code 0
```

Index/freshness evidence: `graphify-out/graph.json` (2026-09-21 10:49:45 +0700) is newer than `src/latent_anything/__init__.py` (2026-09-21 10:47:25 +0700); `CovarianceConfig` resolves with 75 graph hits and `diagnostics.py` with 115 hits.

## Affected Claims

- Every named workflow choice/output (capture, diagnostics, controls, interventions, comparisons, manifest identity, output location/artifact intent, resumable result state) is declarable without executing stages.
- The API contains domain terms, no architecture-specific fields, and no duplicate `ObjectSpec`/`PipelineSpec` configuration path.
- Both frozen 80.5 core manifests are addressable by manifest id through this API.

## Negative Results and Limitations

- The API declares intent only; it does not bind capture axes to `LatentValue`/`Trajectory` (80.7), orchestrate stages (80.8), run detectors, or execute benchmarks.
- Output location is constrained to repository-relative non-escaping paths; absolute or parent-escaping locations are rejected rather than resolved.
- The 0.9 API-freeze snapshot now drifts by design (new public symbols); regeneration awaits the reviewed release-candidate gate and is explicitly not part of this task.
- No 80.7+ work was started.

## Evidence-Review Readiness

Ready: focused API tests cover construction, round-trips, domain-term surface, public exports, no-duplicate-config-path, and each fail-closed rejection; the committed smoke script reproduces both core-manifest declarations plus unknown-family, undeclared-control, and unsafe-output rejections.

## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, status, and decision log:

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/253 uncached files (39%) [8 workers]
  AST extraction: 200/253 uncached files (79%) [8 workers]
  AST extraction: 253/253 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph: hooks.json, evals.json, evals.json, act_policy_representation_benchmark.json, anisotropy_benchmark.json (+243 more). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1047 saved labels, 1062 communities now; renamed 158 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 13902 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1062 community nodes, 1365 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 13902 nodes, 28769 edges, 1062 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit code 0
```

Index/freshness evidence: `graphify-out/graph.json` (2026-09-21 10:43:09 +0700) is newer than `src/latent_anything/__init__.py` (2026-09-21 10:40:52 +0700) and `src/latent_anything/diagnostics.py` (2026-09-21 10:33:27 +0700); `diagnostics.py` has 115 graph hits and `CrossSeedReport` resolves with 2 hits after the restoration.
