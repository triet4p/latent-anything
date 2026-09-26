# Task 80.14 — Localize Findings Across Checkpoint, Token, and Time Axes

## Status

**Complete (provisional, pending independent evidence review).** Sprint 80 tasks 80.1–80.14 are done; tasks 80.15–80.29 remain pending. Historical Sprint 79 evidence was not changed. Frozen taxonomy/manifests/report schema/validator were not mutated; observed results were not encoded.

## Summary of Work

Extended the evidence-gated 80.13 localization convention in place inside `src/latent_anything/_layer_slice_localization.py` with a second private axial seam for checkpoint, token, and time. The existing `DetectionEvidence` threshold/direction/correctness-rule plus qualified-control seam is reused unchanged; no detector method is recomputed and no central control engine is introduced. Each seam handles only its own axes (the stale layer/slice `unsupported_axes_80_14` marker was removed post-review); the 80.14 seam implements real behavior wherever a checkpoint/token/time axis is declared.

Design:

- `manifest_axis_lookup` gives read-only binding coverage: axis applicability requires an explicit manifest representation-axis declaration plus an explicit axial declaration in the localization input. Order is never inferred lexically or from input order.
- `CheckpointAxisDeclaration`/`CheckpointCell` require an explicit monotonic checkpoint order/index plus aligned model identity, dataset slice/configuration identity, representation identity, and optional layer identity; checkpoint comparisons must share the declared slice/configuration.
- `TokenAxisDeclaration`/`TokenCell` require explicit sample/sequence binding, explicit monotonic token order/positions, tokenization/preprocessing identity (any drift refuses), and representation identity.
- `TimeAxisDeclaration`/`TimeCell` require an explicit trajectory identity, explicit monotonic step order/index, declared ordering, and representation identity.
- `AxialLocalizationInput` requires requested axes drawn only from checkpoint/token/time; declared manifest axes without declarations, requested-but-absent axes, and empty cells fail toward explicit `not_applicable` (never empty success) or fail closed for undeclared axes and global-only evidence. Identity/order/duplication/coverage faults reject at construction.
- `axial_localize` returns per-axis `localized`/`negative`/`not_applicable`/`unsupported` results with deterministic affected selections and earliest ordered coordinates, shuffled-order invariance, tolerance-boundary rejection, and honest mixed applicability (supported present axes preserved while absent ones are marked). Aggregates are fail-closed honest: `localized` iff any axis localizes, `not_applicable` iff all are absent, else explicit `negative`. Benign negatives carry no locations.
- `axial_localization_payload` emits canonical diagnostic-report-compatible rows (`id`/`axis`/`selection`/`evidence_refs`/`confidence`/`status`; `confidence=1.0` is deterministic selection certainty under the qualified predeclared rule, not a probability or statistical interval; `evidence_refs` are shape-compatible fragments whose `validate_diagnostic_report` consumption requires 80.21/80.22 registration); `make_axial_localize_executor` binds one axial input to the workflow `localize` stage with `completed`/`not_applicable`/`unsupported` outcomes; `DiagnosticWorkflow` carries no axis algorithm.

```text
uv run pytest tests/test_checkpoint_token_time_localization.py -q
12 passed in 4.61s

uv run python scripts/sprint80_task80_14_smoke.py
PASS config manifest=manifest-80-14-axial-v1 axes=['checkpoint', 'token', 'time']
PASS checkpoint earliest=ckpt-1 affected=('ckpt-1', 'ckpt-2') slice-A/config-v3 aligned
PASS token affected=('tok-2', 'tok-3') sample-7/seq-7 bound; time earliest=step-2 traj-3 ordered
PASS shuffled input order preserves earliest=step-2
PASS mixed applicability: checkpoint localized, token explicit not_applicable
REJECT misaligned checkpoint dataset: checkpoint 'ckpt-1' dataset slice is misaligned: checkpoint comparisons require the same declared dataset slice
REJECT tokenization drift: token 'tok-0' tokenization identity drifted: refusing to localize
REJECT control-blocked: failed required controls block localization: control-a
REJECT global-only: time_cells must not be empty: a global score alone is insufficient
PASS benign negative produces no location
PASS axial executor through workflow completed(7)
PASS deterministic canonical payload
PASS 80.13 layer/slice regression earliest=layer-1
public surface: 9/9 80.6 names intact, no axial symbols leaked
exit code 0

uv run pytest tests/test_checkpoint_token_time_localization.py tests/test_layer_slice_localization.py tests/test_diagnostic_workflow.py tests/test_capture_binding.py -q
43 passed in 5.55s
```

## Behavioral Claims

- Aligned checkpoint defect (`ckpt-0` 3.6 healthy, `ckpt-1` 1.4, `ckpt-2` 0.9 under `>= 3.0` threshold-fail rule, shared `slice-A`/`config-v3`/`model-rev-7`) localizes with `earliest=ckpt-1` and affected `ckpt-1`/`ckpt-2` under declared order.
- Affected token coordinates `tok-2`/`tok-3` localize bound to `sample-7`/`seq-7` and tokenization `tokenizer-bpe-v2`; ordered time coordinates `step-2`/`step-3` localize with `earliest=step-2` on `traj-3` under `time-increasing` ordering; identities preserved into report rows.
- Reversed cell input order preserves every earliest/affected decision; ordering comes from declared order/index maps only.
- Absent axes are explicit per-axis `not_applicable` (mixed present/absent preserves the supported axis and marks the absent one; all-absent yields aggregate `not_applicable` with no report rows).
- Misaligned checkpoint dataset slice/configuration, token sample/sequence/tokenization/preprocessing drift, trajectory mismatch, non-monotonic/ambiguous declared order, duplicate/missing coordinates, undeclared axes, control-blocked/unqualified evidence, tolerance-boundary values, and global-only evidence all fail closed.
- Benign negatives yield explicit `negative` results with no locations.
- Report/workflow integration completes `localize` with `completed` (absent-only yields honest `not_applicable`); output is deterministic and canonical.
- 80.13 layer/slice regression still localizes `earliest_layer=layer-1`; exact nine-name 80.6 surface holds; all five frozen artifacts exist with the transformer manifest digest re-verified.
- Read-only binding coverage: synthetic schema-valid manifest axes plus the real frozen transformer manifest (`wikitext-token-axis-128`, `validation-selected-2048`, `heldout-probe-accuracy >= 0.7`) bound through `manifest_axis_lookup`/`evidence_from_manifest` without mutation.

## Negative Results and Limitations

- Inputs in tests/smoke are synthetic already-computed metric values, not real core-model captures or detector recomputation; by design this task never runs a detector algorithm. Real core benchmark proof remains 80.23/80.24.
- Tests/smoke use a synthetic frozen-schema-valid manifest-shaped mapping plus read-only checks against the real frozen transformer manifest; no frozen manifest, taxonomy, report schema, or validator was mutated.
- Checkpoint health is per-checkpoint already-computed values under the shared frozen criterion (no cross-checkpoint statistical comparison operator); slice means remain the 80.13 unweighted convention and are untouched.
- No statistical-control centralization (80.15), explanations (80.16–80.17), interventions/comparisons/reporting, or causal proof scope was added.
- Tolerance-boundary values fail closed rather than localizing; an absent axis never becomes an empty success.
- No formatter, linter, type checker, packaging, docs build, or project-wide suites were run per task constraints.

## Files Modified

- `src/latent_anything/_layer_slice_localization.py` — refactored/extended private seam in place (80.13 `DetectionEvidence`/layer/slice/report/workflow seam reused; added `AXIAL_AXES`, `manifest_axis_lookup`, checkpoint/token/time declarations and cells, `AxialAxisResult`, `AxialLocalizationInput`, `AxialLocalizationResult`, `axial_localize`, `axial_localization_payload`, `evaluate_axial_localization`, `make_axial_localize_executor`); private module only; zero top-level export changes.
- `tests/test_checkpoint_token_time_localization.py` — 12 consumer-observable tests (checkpoint order, token/time identities, order invariance, absent/mixed non-applicability, checkpoint/token/time misalignment, control/boundary/global-only blocks, benign negative, report/workflow/canonical, manifest binding plus 80.13 regression, surface/frozen-artifact invariance).
- `scripts/sprint80_task80_14_smoke.py` — committed direct smoke covering positive, negative, fail-closed, mixed-applicability, and workflow shapes.
- `docs/sprint-plans/sprint-80.md` — 80.14 marked `[x]` provisionally; 80.15+ pending.

## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, status, and decision log:

```text
graphify update .
  AST extraction: 100/256 uncached files (39%) [8 workers]
  AST extraction: 200/256 uncached files (78%) [8 workers]
  AST extraction: 256/256 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph (...). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1047 saved labels, 1056 communities now; renamed 170 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 14678 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1056 community nodes, 1363 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 14678 nodes, 31686 edges, 1056 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Wall time: 95.79 seconds
```

Index/freshness evidence: `graphify-out/graph.json` is newer than all indexed code/test/script inputs (`src/latent_anything/_layer_slice_localization.py`, `tests/test_checkpoint_token_time_localization.py`, `scripts/sprint80_task80_14_smoke.py`) and the pre-record artifact draft, but older than this status/graph-record Markdown edit. First recorded rebuild: 14678 nodes, 31686 edges, 1056 communities (output block above). A later confirming `graphify update .` ran with no file writes afterward and rebuilt to 14677 nodes, 31685 edges, 1041 communities (trivial churn, not no-topology-change). The exact final confirming result after this artifact correction is reported to Main.
