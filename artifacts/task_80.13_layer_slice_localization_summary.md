# Task 80.13 — Localize Findings Across Layers and Dataset Slices

## Status

**Complete (provisional, pending independent evidence review).** Sprint 80 tasks 80.1–80.13 are done; tasks 80.14–80.29 remain pending. Historical Sprint 79 evidence was not changed. Frozen taxonomy/manifests/report schema/validator were not mutated; observed results were not encoded.

## Summary of Work

Added `src/latent_anything/_layer_slice_localization.py`, the smallest private reusable localization seam behind the frozen workflow contracts. It consumes machine-readable, control-qualified detection evidence (already-computed per-layer/per-sample metric values plus predeclared metric/threshold/direction/control wiring) and never recomputes a detector algorithm.

Design:

- `DetectionEvidence` carries one predeclared metric rule: threshold value/comparator/tolerance, metric direction, `affected_when` correctness rule (`threshold_fail` vs `threshold_pass`), required-control outcomes (`passed`/`failed` convention shared with the detector adapters), qualified `supported`/`inconclusive`/`unsupported` outcome, and representation identity. Comparator/direction wiring rejects mismatches (e.g. `<=` under `higher_is_better`); unqualified (`inconclusive`/non-claim-allowed) or failed-control evidence cannot localize; `unsupported` evidence returns an honest non-applicable verdict, never success.
- `LocalizationInput` requires explicit ordered layer identities (`layer_order`), explicit `LayerCell`/`SampleCell` observations covering exactly the declared layers/samples with preserved representation identity, explicit `declared_slice_ids` plus `SliceDefinition` criteria/member identities whose membership is aligned to the sample identities with full coverage. A global score is accepted only as a recorded-and-ignored field: empty layer or sample evidence rejects ("a global score alone is insufficient").
- `localize_findings` determines the earliest affected layer by declared layer order plus the declared correctness rule (never lexical names, input order, tuning, or the global aggregate); it identifies affected individual samples and affected declared slices (slice health = member-sample mean under the same frozen criterion). Tolerance-boundary values reject as ambiguous ties rather than guessing. Benign negatives return `verdict="negative"` with no location. Checkpoint/token/time behavior lives in the 80.14 axial seam of the same module, never in the layer/slice seam (the stale `unsupported_axes_80_14` marker was removed post-review).
- `localization_payload` emits deterministic canonical output compatible with the diagnostic-report localization shape (`id`/`axis`/`selection`/`evidence_refs`/`confidence`/`status` rows for the earliest layer plus every affected sample/slice; `confidence=1.0` means deterministic selection certainty under the already-qualified predeclared rule — not a probability, effect strength, or statistical interval; evidence strength travels separately in the qualified detection evidence and control outcomes that 80.22 assembly must carry). Localization `evidence_refs` (`detect:<family>:<metric>`) are shape-compatible fragments only: `validate_report_shape` passes while `validate_diagnostic_report` requires 80.21/80.22 registration and metric linkage; refs stay provisional. `make_localize_executor` binds one input to the workflow `localize` stage with identity checks and honest `completed`/`unsupported` outcomes; `DiagnosticWorkflow` carries no localization algorithm.

```text
uv run pytest tests/test_layer_slice_localization.py -q
14 passed in 3.38s

uv run python scripts/sprint80_task80_13_smoke.py
PASS config manifest=manifest-80-13-localization-v1 metrics=['layer-health']
PASS positive earliest=layer-1 samples=['s3', 's4'] slice=slice-back
PASS shuffled input order preserves earliest=layer-1
PASS benign negative produces no location
REJECT control-blocked: failed required controls block localization: control-a
REJECT undeclared slice: undeclared slices are not localizable: slice-adhoc
REJECT global-only: layer_cells must not be empty: a global score alone is insufficient
PASS localize executor through workflow completed(7)
PASS deterministic canonical payload
PASS frozen encoder manifest wiring threshold=3.0 direction=higher_is_better
public surface: 9/9 80.6 names intact, no localizer symbols leaked
exit code 0

uv run pytest tests/test_layer_slice_localization.py tests/test_collapse_detection.py tests/test_diagnostic_workflow.py tests/test_diagnostics_request.py tests/test_sprint80_core_manifests.py tests/test_capture_binding.py tests/test_redundancy_separability_detection.py tests/test_density_drift_detection.py -q
88 passed in 46.27s
```

## Behavioral Claims

- Known multi-layer defect (`layer-0` healthy 3.5, `layer-1` 1.2, `layer-2` 0.8 under `>= 3.0` with `affected_when=threshold_fail`) yields `earliest_layer="layer-1"` with `affected_layers=("layer-1", "layer-2")`.
- Affected sample IDs `s3`/`s4` and one predeclared slice `slice-back` (member mean ~1.97 vs `slice-front` ~3.8) are identified despite a benign global aggregate `3.4`, which is recorded with `global_score_ignored=true` and never consulted.
- Shuffled input order (reversed cells, reordered slice declarations) preserves the decision; ordering comes from `layer_order`/`declared_slice_ids` only.
- Control-blocked (`control-a=failed`) and below-threshold-unqualified (`inconclusive`, `claim_allowed=False`) evidence cannot localize; threshold-direction mismatch (`<=` under `higher_is_better`) rejects at construction.
- Undeclared/ad-hoc slices, duplicate/missing/misaligned layer/sample/slice identities, slice members outside the sample identities, samples without slice membership, and tolerance-boundary ties all fail closed.
- Global-only evidence (empty layer or sample cells) is insufficient and rejects.
- Workflow integration completes the `localize` stage (`completed`, `earliest_layer=layer-1`, `affected_slices=["slice-back"]`) with no coordinator algorithms.
- `evidence_from_manifest` binds the real frozen encoder manifest wiring (`bottleneck-effective-rank`, `>= 3.0`, `higher_is_better`, both required controls) to the localization evidence without mutating the manifest.
- Output is deterministic across repeated evaluation and passes the shared `canonical_json` contract; public surface holds the exact nine 80.6 names with no localizer symbols leaked; all five frozen artifacts exist and the encoder manifest digest still matches.

## Negative Results and Limitations

- Inputs in tests/smoke are synthetic already-computed metric values, not real core-model captures or detector recomputation; by design this task never runs a detector algorithm. Real core benchmark proof remains 80.23/80.24.
- Tests/smoke use a synthetic frozen-schema-valid manifest-shaped mapping plus a read-only binding check against the real frozen encoder manifest; no frozen manifest, taxonomy, report schema, or validator was mutated.
- Slice health is the unweighted member-sample mean under the same frozen criterion; per-slice thresholds, weighted slices, overlapping membership semantics beyond explicit member lists, and slice-identity ordering beyond declaration order are out of scope.
- 80.14 axes (checkpoint/token/time) are handled by the axial seam in the same module, never the layer/slice seam; statistical-control centralization (80.15), explanations (80.16–80.17), interventions/comparisons/reporting, and causal proof are untouched.
- Ambiguous tolerance-boundary values fail closed rather than localizing; sample-only movement without an affected layer yields a negative, not a layer location.
- No formatter, linter, type checker, packaging, docs build, or project-wide suites were run per task constraints.

## Files Modified

- `src/latent_anything/_layer_slice_localization.py` — private localization seam (`DetectionEvidence`, `LayerCell`, `SampleCell`, `SliceDefinition`, `LocalizationInput`, `LocalizationResult`, `evidence_from_manifest`, `localize_findings`, `localization_payload`, `evaluate_localization`, `make_localize_executor`); private module only; zero top-level export changes.
- `tests/test_layer_slice_localization.py` — 14 consumer-observable tests (earliest layer, samples/slice under benign global, order invariance, control/qualification blocks, benign negative, undeclared/identity faults, global-only insufficiency, direction mismatch, tolerance tie, unsupported honesty, determinism/canonical output, workflow integration, frozen-manifest wiring, surface/frozen-artifact invariance).
- `scripts/sprint80_task80_13_smoke.py` — committed direct smoke covering positive, negative, fail-closed, and workflow shapes.
- `docs/sprint-plans/sprint-80.md` — 80.13 marked `[x]` provisionally; 80.14+ pending.

## Scope and Public-Surface Evidence

- `graphify query` navigation used before source exploration; prior detection adapters, capture-axis metadata, frozen manifests/report schema, workflow state machine, and threshold/direction/control conventions reused; no parallel estimator or central control engine introduced.
- Nine-name 80.6 surface verified (`CaptureSelection`, `ComparisonRequest`, `ControlSelection`, `DiagnosticRequest`, `DiagnosticRequestError`, `DiagnosticResult`, `DiagnosticSelection`, `InterventionRequest`, `OutputSelection`); no `DetectionEvidence`/`LocalizationInput`/`localize_findings`/`make_localize_executor`/`LayerCell` top-level leak.
- Frozen artifacts untouched: `benchmark_manifest_schema_v1.json`, `diagnostic_report_schema_v1.json`, `representation_problem_taxonomy_v1.json`, both Sprint 80 core benchmark manifests (encoder digest re-verified against `manifest_digest`).

## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, status, and decision log:

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/256 uncached files (39%) [8 workers]
  AST extraction: 200/256 uncached files (78%) [8 workers]
  AST extraction: 256/256 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph (...). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1044 saved labels, 1047 communities now; renamed 186 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 14584 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1047 community nodes, 1219 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 14584 nodes, 31306 edges, 1047 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Wall time: 63.70 seconds
```

Index/freshness evidence: `graphify-out/graph.json` is newer than all indexed code/test/script inputs (`src/latent_anything/_layer_slice_localization.py`, `tests/test_layer_slice_localization.py`, `scripts/sprint80_task80_13_smoke.py`) and the pre-record artifact draft, but older than this status/graph-record Markdown edit. Rebuilt graph: 14584 nodes, 31306 edges, 1047 communities. A final confirming `graphify update .` follows with no file writes afterward; a no-topology-change result is reported honestly to Main via hub.
