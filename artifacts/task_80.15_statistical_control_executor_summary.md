# Task 80.15 — Central Statistical-Control Executor

## Status

**Complete (provisional, pending independent evidence review).** Sprint 80 tasks 80.1–80.15 are done; tasks 80.16–80.29 remain pending. Historical Sprint 79 evidence was not changed. Frozen taxonomy/manifests/report schema/validator were not mutated; observed results were not encoded.

## Summary of Work

Added `src/latent_anything/_statistical_controls.py`, one boring private convention for immutable/prevalidated control plans (`ControlPlan`/`ControlSpec`) and deterministic execution results (`ControlOutcome`). It binds repetitions, confidence level, and seed roles from predeclared manifest/config inputs (`plan_from_manifest`), derives per-control RNG streams from stable identities (`derive_stream_seed`, sha256-based, never Python `hash`), owns exact repetition counts and percentile interval summarization (`summarize_interval`, `run_bootstrap`), evaluates null/shuffled/randomized/cross-seed controls (`run_permutation_control`, `execute_plan`), and records exact seed/stream/repetitions/method/interval/expected-behavior/required/passed-failed provenance. Required controls fail closed on missing data, execution failure, non-finite results, wrong repetition counts, or failed comparators; optional recorded controls stay distinguishable and never gate. Unsupported controls can never read as pass.

Migrated all four 80.9–80.12 detector adapters to the central executor for seeded repetition/CI and applicable null/shuffled/randomized/cross-seed scheduling (real cutover, corrected after the evidence-gate review):

- Collapse (80.9): one shared bootstrap schedule (`bootstrap:collapse-both`, one draw records both statistics, exactly `repetitions` decompositions) under the identity-derived `evaluation` stream; the column-shuffle null runs as one central `shuffled` control transform (`central:column-shuffle-null`) whose observed values bind the `null_shuffled` metrics directly; required/optional controls gate per linked metric through `execute_plan` with per-metric parts folded back to declared identities, `failed_required` blocking, and rich `controls` provenance in the payload.
- Redundancy/separability (80.10): heldout/gap, correlation/sharing intervals run as central bootstraps over fitted predictions/covariance/coherence values; the label-randomization probe refit and the column-shuffle redundancy evaluation execute inside central `randomized`/`shuffled` callbacks under exact declared identities (`control-label-randomization`, declared null), whose measured ControlOutcome output (accuracy/gap, correlation/sharing) is consumed directly; separability and redundancy gating route through `execute_plan` per linked metric with `failed_required` and rich `controls` provenance in the payload.
- Density/drift (80.11): flag-rate and gap intervals run as central bootstraps over fitted calibrated scores; the column-shuffle null executes inside a central `shuffled` callback under the exact declared identity (`control-shuffled-null`), permuting columns and scoring through the already-fitted/calibrated shared model once, whose measured flag/gap/scores are consumed directly (one fit, one calibration, no refit); no-shift negatives gate per linked metric through `execute_plan` with `failed_required` and rich `controls` provenance in the payload; `shuffled_null_seed` keeps the predeclared manifest seed while `shuffled_null_stream_seed` records the central stream that drove the callback.
- Sparse/temporal (80.12): stability and drift intervals run as central bootstraps over fitted cosines/stepwise distances; the shuffled-sequence null executes inside a central `shuffled` callback under the exact declared identity (`control-shuffled-sequence`), permuting and scoring once, whose measured drift is consumed directly; the cross-seed callback draws the actual fit seed from the supplied central RNG, performs the single predeclared extra dictionary fit, and returns fit seed plus stability for direct consumption; sparse/temporal gating routes through `execute_plan` per linked metric with `failed_required` and rich `controls` provenance in the payload. Sparse fit bounds preserved (one fit per seed plus the one predeclared refit); temporal scoring unchanged.
- Local `_summarize` duplicates now alias the central `summarize_interval` (sparse duplicate removed); stale prose updated. Each detector keeps its evidence semantics, one-fit/one-calibration guarantees, manifest thresholds/seeds/controls, outputs, and focused tests. Estimator-specific transforms execute only inside caller-supplied central callbacks over already-fitted state; the strict seam is documented in each adapter. Localization 80.13/80.14 untouched (still consumes qualified outcomes). `DiagnosticWorkflow` remains method-agnostic.

```text
uv run pytest tests/test_statistical_controls.py -q
14 passed

uv run pytest tests/test_collapse_detection.py tests/test_density_drift_detection.py -q
24 passed

uv run pytest tests/test_redundancy_separability_detection.py -q
14 passed

uv run pytest tests/test_sparse_temporal_detection.py -q
11 passed

uv run pytest tests/test_diagnostic_workflow.py tests/test_layer_slice_localization.py tests/test_checkpoint_token_time_localization.py -q
36 passed

uv run python scripts/sprint80_task80_15_smoke.py
PASS bootstrap interval repetitions=40 lower=1.0 upper=39.0
PASS identity-derived stream seed=5473721401843784259
PASS all control kinds (bootstrap/null/shuffled/randomized/cross_seed) via central path
PASS required pass/fail/missing semantics; optional stays recorded
PASS exact repetition count with central RNG ownership
PASS failed required control blocks collapse conclusion
PASS detector/workflow blocking path with central bootstrap streams
PASS recorded stream reproduces null draw (stream=<derived>)
public surface: 9/9 80.6 names intact, no control symbols leaked
exit code 0
```

## Behavioral Claims

- Exact repetitions and percentile CI on a known statistic (0..39, reps=40: lower=1.0, upper=39.0, mean≈19.5); wrong draw counts reject.
- Identity-derived streams are deterministic across runs and independent of declaration/execution order (sorted identity execution); distinct identities/roles/seeds diverge.
- Each control kind (bootstrap/null/shuffled/randomized/cross_seed/seed) exercises the central path with `recorded` status for optional controls.
- Required pass/fail/missing/non-finite behavior holds; execution failure yields `failed`, never pass; unsupported can never read as pass.
- Optional-vs-required semantics: optional controls stay `recorded` without gating; `failed_required` returns only failed required identities.
- A failed required control blocks a real collapse-detector conclusion (`inconclusive`, `claim_allowed=False`, `missing_evidence=failed-control:...`); positives with controls passed stay `supported`.
- Migrated adapters execute inside central streams behaviorally: each callback's measured ControlOutcome output is consumed directly (no replay, no second transform — corrected post-review: the redundancy null path previously re-evaluated the reconstructed column stack a second time; it now stores and consumes the already-evaluated `_RedundancyEvaluation` from inside the central callback, so one DictionaryLearning fit and one covariance/bootstrap schedule per null evaluation); changing control identity changes the stream and draw; declaration order does not; every adapter's real control path yields non-empty central observed values wired into `controls` payload sections; one-fit/one-calibration and sparse fit bounds intact (instrumented counts prove single execution).
- Manifest binding uses only predeclared uncertainty/seeds/controls with no observed-data tuning; seed-role mismatch rejects.
- Workflow integration completes `detect` with `completed` and central bootstrap/null provenance (reps=200, derived stream seeds).
- Nine-name 80.6 surface holds; all five frozen artifacts exist with digests re-verified.

## Negative Results and Limitations

- The one callback supplied to `execute_plan`/`run_permutation_control` performs the actual transform plus all downstream estimator scoring for that control, then returns the complete measured result mapping; detection consumes `ControlOutcome.observed` directly. The separability weak-signal fixture was retuned (data seed 48, shift 0.3) to preserve the accuracy-without-gap shape under the central stream; frozen manifest thresholds were not changed.
- Cross-seed fitting draws/records the actual fit seed from the supplied central RNG, performs the single predeclared extra dictionary fit, and returns fit seed plus metrics for direct consumption.
- Inputs in tests/smoke are synthetic NumPy batches, not real core-model captures; real core benchmark proof remains 80.23/80.24.
- Tests/smoke use synthetic frozen-schema-valid manifests plus read-only checks against the real frozen encoder manifest; no frozen manifest, taxonomy, report schema, or validator was mutated.
- No statistical estimation beyond the existing detector primitives was added; no explanations (80.16–80.17), interventions/comparisons/reporting, or causal proof scope.
- No formatter, linter, type checker, packaging, docs build, or project-wide suites were run per task constraints.

## Files Modified

- `src/latent_anything/_statistical_controls.py` — new private executor (`CONTROL_KINDS`, `ControlExecutionError`, `ControlSpec`, `ControlPlan`, `ControlOutcome`, `derive_stream_seed`, `summarize_interval`, `plan_from_manifest`, `run_bootstrap`, `run_permutation_control`, `execute_plan`, `failed_required`); private module only; zero top-level exports.
- `src/latent_anything/_collapse_detection.py` — bootstrap/null scheduling migrated to central executor; `_summarize` aliases central math; prose updated; evidence semantics and thresholds unchanged.
- `src/latent_anything/_redundancy_separability_detection.py` — bootstrap/randomized/null scheduling migrated; `_summarize` aliases central math; prose updated; evidence semantics unchanged.
- `src/latent_anything/_density_drift_detection.py` — bootstrap/null scheduling migrated; `_summarize` aliases central math; prose updated; one-fit/one-calibration preserved.
- `src/latent_anything/_sparse_temporal_detection.py` — bootstrap/shuffled/cross-seed scheduling migrated; `_summarize` duplicate removed in favor of the central alias; prose updated; one-fit guarantees preserved.
- `tests/test_statistical_controls.py` — 14 consumer-observable tests (exact reps/CI, stream determinism/order invariance, all control kinds, required semantics, optional semantics, manifest binding, detector blocking, collapse replay proof, redundancy/density/sparse/temporal replay proofs, instrumented single-execution counts, surface invariance).
- `tests/test_density_drift_detection.py` — seed-provenance assertions updated to the central derived stream (behavioral, not source-text).
- `tests/test_redundancy_separability_detection.py` — weak-signal fixture retuned (data seed 48, shift 0.3) to preserve the accuracy-without-gap shape under the central stream; thresholds unchanged.
- `tests/test_collapse_detection.py` — instrumented count and null-outcome expectations updated for the shared central schedule and declared central identities (behavioral, not source-text).
- `scripts/sprint80_task80_15_smoke.py` — committed direct smoke covering all control kinds, real detector/workflow blocking paths, and recorded-stream reproduction proofs.

## Migration Map

- `_summarize` (collapse/redundancy/density duplicates) → alias of `summarize_interval`; sparse duplicate removed; single interval contract owned centrally.
- Bootstrap loops (collapse shared schedule, separability heldout/gap, redundancy correlation/sharing, density flag/gap, sparse stability, temporal drift) → `run_bootstrap` draws over already-fitted state with identity-derived `evaluation` streams.
- Null/shuffle/randomized/cross-seed transforms → `run_permutation_control`/`execute_plan` callbacks under exact declared identities; each callback performs the actual permutation plus all downstream estimator scoring once and returns the complete measured mapping, consumed directly from `ControlOutcome.observed` (plus `detail` for vector payloads). No local RNG re-instantiation, no replay, no throwaway draw, no second transform, no discarded output.
- Required/optional status for all four adapters → `execute_plan` with per-metric parts folded back to declared identities; `failed_required` blocks only required associated conclusions; optional stays `recorded`. No duplicated local required/optional status loops or hardcoded comparator gating remain. Rich `ControlOutcome` records (identity, kind, required, comparator/threshold, stream seed, repetitions, method, observed, status) are wired into each `controls` payload section; executed controls carry non-empty finite observed values.
- Estimator statistics execute only inside central callbacks over already-fitted state (health/SVD, probe fits, covariance/dictionary fits, GMM fit/calibrate/score, dictionary fits, stepwise distances); one-fit/one-calibration and sparse fit bounds preserved (instrumented counts prove single execution).

## Scope and Public-Surface Evidence

- `graphify query` navigation used before edits; prior detector/control duplication mapped across all four adapters before the cutover; no second estimator, no refit loops, no centralization of detector algorithms.
- Nine-name 80.6 surface verified (`CaptureSelection`, `ComparisonRequest`, `ControlSelection`, `DiagnosticRequest`, `DiagnosticRequestError`, `DiagnosticResult`, `DiagnosticSelection`, `InterventionRequest`, `OutputSelection`); no `ControlPlan`/`ControlOutcome`/`execute_plan`/`run_bootstrap`/`derive_stream_seed`/`summarize_interval` top-level leak.
- Frozen artifacts untouched: `benchmark_manifest_schema_v1.json`, `diagnostic_report_schema_v1.json`, `representation_problem_taxonomy_v1.json`, both Sprint 80 core benchmark manifests (encoder digest re-verified in smoke).
## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, status, and decision log. Second correction round (re-review findings: real central transforms, declared identities, non-empty observed, no replay) ran after the earlier recorded rebuild:

```text
graphify update .
  AST extraction: 100/258 uncached files (38%) [8 workers]
  AST extraction: 200/258 uncached files (77%) [8 workers]
  AST extraction: 258/258 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph (...). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1044 saved labels, 1029 communities now; renamed 147 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 14760 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1029 community nodes, 1348 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 14760 nodes, 32013 edges, 1029 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Wall time: 63.15 seconds
```

Ordering: all correction code/test/script/artifact edits above precede this rebuild; the exact final confirming result after this record is reported to Main with no writes afterward.
