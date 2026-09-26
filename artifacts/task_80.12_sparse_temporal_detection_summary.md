# Task 80.12 — Integrate Sparse-Feature and Temporal-Drift Detection

## Status

**Complete.** Sprint 80 tasks 80.1–80.12 are done; tasks 80.13–80.29 remain pending. Historical Sprint 79 evidence was not changed. Frozen taxonomy/manifests/validator were not mutated; observed results were not encoded.

## Summary of Work

Added `src/latent_anything/_sparse_temporal_detection.py`, the smallest private detector-family adapter behind the frozen taxonomy/workflow contracts for `sparse_feature_instability` and `sequence_trajectory_drift`. It consumes bound `SparseInput`/`TemporalInput` batches plus predeclared metric/control/threshold configuration and emits machine-readable observations, taxonomy family-evidence statuses (`evaluate_claim`), control outcomes, seeded uncertainty metadata, and honest `supported`/`inconclusive`/`unsupported` outcomes. A supplied detect-stage executor integrates through the method-agnostic workflow with no method logic in the coordinator and no public API widening.

Applicability is derived from bound inputs, never assumed: sparse evaluation requires a seed axis plus per-seed batches with shared sample identities; temporal evaluation requires ordered trajectories with a `sequence_or_time` axis and declared step identities. A bare `LatentValue` carries neither axis and fails closed. Families with no promotable manifest declaration return honest `missing-declaration` unsupported rather than borrowing another family's metrics.

Reused existing primitives (no new estimators):

- `DictionaryLearning` is the single shared sparse convention: one fit per seed (8 atoms, `max_iter=100`, manifest training seed as base plus declared seed offsets).
- `match_by_decoder_cosine` is the single shared permutation-invariant alignment convention: globally ranked decoder-direction matching at cosine >= 0.85, never raw feature indices. Stability is the matched fraction.
- `LatentSpace.distance` dispatch (the point-cost convention shared with `indexwise_distance`, asserted equal per call) is the single shared stepwise drift convention; repeatability is ordered drift minus the no-drift negative.
- Taxonomy evidence gating reuses `evaluate_claim`; manifest validity reuses `validate_manifest`; canonical JSON reuses the shared `canonical_json` contract.

Metric/control semantics:

- Sparse metrics are `sparse-cross-seed-stability` and `sparse-reconstruction-quality` (both `>=` in tests/smoke: 0.5 and 0.5). Same-sample cross-seed comparison canonicalizes every seed batch to the bound sample-identity order before fitting, so row permutation is a sample artifact: the permuted-but-stable positive passes (stability=0.500 in smoke). The stable negative (same sparse family) aligns to the reference and fails its control, blocking promotion; noise-on-noise aligns to nothing and fails closed with no manufactured stability.
- Temporal metrics are `trajectory-stepwise-drift` and `trajectory-repeatability` (both `>=` in tests/smoke: 0.2 and 0.1). Ordered drift (0.500) promotes only when the no-drift negative stays below threshold and the shuffled/reversed controls are recorded; no-drift negatives, malformed axes, and failed controls cannot promote. Failed required controls block before threshold reporting.
- Negative controls ride inside the bound input (`SparseInput.negative_batches`, `TemporalInput.negative`); shuffled/seed controls are derived under manifest seeds (recorded, not threshold-gated). External control batches are a fail-closed error.

Single-evaluation seam: `evaluate_detection` fits every dictionary and scores every trajectory exactly once per executor call, then shares one context between family decisions and payload assembly. Sparse uncertainty resamples fitted matched cosines without refitting; temporal uncertainty resamples fitted stepwise distances. `detect_families`/`detection_payload` remain documented unit-check seams whose naive composition is known-double.

```text
uv run pytest tests/test_sparse_temporal_detection.py -q
11 passed in 138.31s

uv run pytest tests/test_collapse_detection.py tests/test_diagnostic_workflow.py tests/test_diagnostics_request.py tests/test_sprint80_core_manifests.py tests/test_capture_binding.py tests/test_density.py -q
50 passed in 6.53s

uv run python scripts/sprint80_task80_12_smoke.py
PASS config manifest=manifest-80-12-sparse_feature_instability metrics=['sparse-cross-seed-stability', 'sparse-reconstruction-quality']
PASS permuted-but-stable sparse positive stability=0.500
REJECT unstable noise: sparse alignment produced no matched features
PASS stable negative blocks sparse promotion
PASS ordered temporal drift=0.500
PASS no-drift temporal negative cannot promote
PASS failed temporal control blocks promotion
REJECT malformed axis: step_ids must cover every test step
PASS honest missing-declaration path (no borrowed family label)
PASS deterministic sparse payload
PASS detect executor through workflow completed(7)
REJECT external control: this detector carries negative controls inside the bound input; no external control batches are accepted
exit code 0

public surface: 9/9 80.6 names intact, no detector symbols leaked
```

## Affected Claims

- Permuted-but-stable sparse positives promote with permutation-invariant decoder matching; stable negatives block; noise fails closed.
- Ordered temporal drift promotes only with no-drift negative passed and shuffled/reversed controls recorded; malformed axes and failed controls cannot promote.
- Non-applicable inputs (bare batches without seed/sequence axes) fail closed; missing declarations return honest unsupported.
- The detect executor integrates through the workflow (`completed(7)`) with no method logic in the coordinator.
- No 80.13+ localization, control centralization (80.15), explanation/intervention, or real benchmark proof claim is made.

## Negative Results and Limitations

- Detector inputs in tests/smoke are synthetic NumPy batches (sparse ground truth, noise, linear trajectories), not real core-model captures; real core benchmark proof remains 80.23/80.24.
- The frozen core manifests predeclare no sparse/temporal metrics, so tests/smoke use synthetic frozen-schema-valid manifest-shaped mappings; no frozen manifest was mutated.
- Stability is the matched fraction at cosine >= 0.85 under globally ranked decoder matching (8 atoms, `max_iter=100`); dictionary fits dominate detector cost (~35s per sparse evaluation, ~140s for the 11-test file).
- Sparse requires at least two seeds covering identical samples with per-seed identities; temporal requires equal-horizon euclidean/unit_norm trajectories with at least two steps.
- No formatter, linter, type checker, packaging, docs build, or project-wide suites were run per task constraints.

## Files Modified

- `src/latent_anything/_sparse_temporal_detection.py` — private detector plus single-evaluation seam (`SparseInput`, `TemporalInput`, `_fit_sparse_once`, `_stability_of`, `_evaluate_sparse`, `_stepwise_drift`, `_evaluate_temporal`, `_DetectionContext`, `_evaluate_context`, `_decide`, `evaluate_detection`); `make_detect_executor` calls `evaluate_detection` exactly once; private module only; zero top-level export changes.
- `tests/test_sparse_temporal_detection.py` — 11 consumer-observable tests (permuted-stable positive, noise fail-closed, stable-negative block, ordered drift positive, no-drift negative, failed-control block, malformed-axis rejection, missing-declaration unsupported, determinism, fail-closed inputs, manifest-shape workflow integration).
- `scripts/sprint80_task80_12_smoke.py` — committed direct smoke covering every acceptance shape.
- `docs/sprint-plans/sprint-80.md` — 80.12 marked `[x]`; 80.13+ pending.

## Evidence-Review Readiness

Ready: focused detector tests prove applicability-gated sparse/temporal evaluation with permutation-invariant alignment, predeclared thresholds/seeds/uncertainty/controls gate promotion, non-applicable and failed-control cases cannot promote, results are deterministic, the detect executor completes through the method-agnostic workflow, fail-closed inputs reject; the committed smoke reproduces every acceptance shape; the nine-name 80.6 surface is verified unchanged.

## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, status, and decision log:

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/256 uncached files (39%) [8 workers]
  AST extraction: 200/256 uncached files (78%) [8 workers]
  AST extraction: 256/256 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph (...). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1026 saved labels, 1044 communities now; renamed 170 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 14494 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1044 community nodes, 1120 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 14494 nodes, 31003 edges, 1044 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit code 0
```

Index/freshness evidence: `graphify-out/graph.json` (16:13:55) is newer than all indexed code/test/script inputs (`_sparse_temporal_detection.py` 16:02:30, `test_sparse_temporal_detection.py` 16:05:51, `sprint80_task80_12_smoke.py` 16:06:25) and the artifact summary (16:12:19), but older than `docs/sprint-plans/sprint-80.md` (16:12:27, status-only edit after the rebuild). Rebuilt graph: 14494 nodes, 31003 edges, 1044 communities. Final confirming run (executed after this record was written; no file writes followed it) is reported to Main via hub.
