# Task 80.11 — Integrate Density, OOD, and Distribution-Drift Detection

## Status

**Complete.** Sprint 80 tasks 80.1–80.11 are done; tasks 80.12–80.29 remain pending. Historical Sprint 79 evidence was not changed. Frozen taxonomy/manifests/validator were not mutated; observed results were not encoded.

## Summary of Work

Added `src/latent_anything/_density_drift_detection.py`, the smallest private detector-family adapter behind the frozen taxonomy/workflow contracts for `density_ood_distribution_drift`. It consumes a bound `ReferenceTestPair` (reference/calibration/test rows with explicit identity and provenance) plus predeclared metric/control/threshold configuration and emits machine-readable observations, taxonomy family-evidence statuses (`evaluate_claim`), control outcomes, seeded uncertainty metadata, and honest `supported`/`inconclusive`/`unsupported` outcomes. A supplied detect-stage executor integrates through the method-agnostic workflow with no method logic in the coordinator and no public API widening.

Reused existing primitives (no new estimators):

- `GaussianMixtureDensity` (`GMMConfig`) is the single shared density convention: one fit on reference/train rows only, one `calibrate` on held-out reference rows only, and `score` for every evaluation batch. No parallel estimator exists.
- The drift statistic is the mean calibrated OOD-score shift (`mean(test) - mean(reference-held-out)`) under that one fitted model; the flag rate uses the 0.9 quantile of the held-out calibration scores (`heldout-reference-quantile-0.9`). Gap uncertainty resamples fitted test and held-out reference scores jointly (exactly `repetitions` draws under the evaluation seed); `ReferenceTestPair.geometry` propagates into `GaussianMixtureDensity.fit` and the recorded fit/calibration provenance.
- The shuffled null independently permutes each test feature column under the control seed (`independent-per-column-permutation`, recorded with method and seed); it is an optional recorded control demonstrating estimator response, not a threshold gate.
- Taxonomy evidence gating reuses `evaluate_claim`; manifest validity reuses `validate_manifest`; canonical JSON reuses the shared `canonical_json` contract.

Metric/control semantics:

- Metrics are `ood-flag-rate` and `density-drift-gap` (both `>=` in tests/smoke: 0.5 and 0.2). Threshold direction/identity come from the manifest/config only; nothing is tuned from observed values.
- A supported claim requires all of: bound reference/test identities in one shared space, declared dataset/split/revision/preprocessing/axis provenance with distinct split identities, calibration on held-out reference rows only, the drift statistic clearing its threshold, the no-shift negative staying below threshold, and every required control passed. Any identity mismatch or sample overlap, missing calibration/control, or failed required control blocks promotion (`inconclusive`, `claim_allowed=False`). A drift score alone never promotes.
- The no-shift negative (`counterexample`/`negative` kind) is the only control taking caller-supplied batch data; `shuffled`/`null` controls are derived from the test batch under manifest seeds (recorded, not threshold-gated). Supplying batch data for derived controls is a fail-closed error.
- Unsupported geometries and caller-declared distribution-free requests return honest `unsupported` with reasons — never silently in-distribution. Manifests with no promotable family declaration return `missing-declaration` unsupported rather than borrowing another family's metrics.

Single-evaluation seam: `evaluate_detection` fits the density once and calibrates once per executor call, then scores every batch (target, negative, shuffled null) from that one fitted model. Flag-rate uncertainty resamples fitted calibrated scores without refitting; gap uncertainty resamples fitted test and held-out reference scores jointly (exactly `repetitions` draws). `detect_families`/`detection_payload` remain documented unit-check seams whose naive composition is known-double.

```text
uv run pytest tests/test_density_drift_detection.py -q
15 passed

uv run pytest tests/test_mlp_probe.py tests/test_probes.py tests/test_density.py tests/test_collapse_detection.py tests/test_diagnostic_workflow.py tests/test_diagnostics_request.py tests/test_sprint80_core_manifests.py tests/test_capture_binding.py -q
129 passed, 5 skipped

uv run python scripts/sprint80_task80_11_smoke.py
PASS config manifest=manifest-80-11 metrics=['ood-flag-rate', 'density-drift-gap'] reps=50
PASS shifted/OOD positive flag=1.000 gap=0.494
PASS calibrated in-distribution negative (drift score alone is not a diagnosis)
REJECT overlap: reference/test overlap: a sample identity appears on both sides
REJECT identity mismatch: test representation identity does not match the reference identity
PASS shuffled control recorded shuffle_flag=1.000
PASS failed no-shift control blocks promotion
PASS distribution-free request is unsupported, not in-distribution
PASS honest missing-declaration path (no borrowed family label)
PASS deterministic flag=1.000
PASS detect executor through workflow completed(7)
REJECT missing-control: required controls are missing batch data: control-no-shift-negative
exit code 0

public surface: 9/9 80.6 names intact, no detector symbols leaked
```

## Correction Round 2 — Evidence-Gate Findings (Same Task, No Scope Change)

1. Shuffled null now independently permutes each test feature column under the control seed (recorded as `independent-per-column-permutation` with seed); the whole-row permutation no-op is removed and control wording claims no more than the optional recorded control tests.
2. Drift-gap uncertainty now performs exactly `repetitions` joint resamples of fitted test and held-out reference scores under the evaluation seed and summarizes resampled mean gaps; raw per-sample quantiles are no longer labeled bootstrap. New observable test asserts repetition count, seed, interval shape, and determinism.
3. `ReferenceTestPair.geometry` propagates into `GaussianMixtureDensity.fit` and the recorded fit/calibration provenance; new unit_norm test proves euclidean and unit_norm coverage.

## Correction Round — Instrumented Single-Evaluation Proof (Same Task, No Scope Change)

Added `test_executor_evaluates_density_and_calibration_once`, which drives the real `detect` executor through a `StageInvocation` while counting `GaussianMixtureDensity.fit`/`calibrate` calls behaviorally: exactly 1 fit and 1 calibration per executor call, with both uncertainty blocks reporting `repetitions == 50`. Scoring is shared from the one fitted model (calibration, target, negative, shuffled null); only uncertainty resamples fitted scores without refitting.

## Affected Claims

- Shifted/OOD positives promote only with explicit reference/test identity, held-out calibration, drift thresholds met, no-shift negative passed, and shuffled null recorded.
- Calibrated in-distribution negatives do not promote; drift scores alone never promote.
- Overlap, identity/provenance mismatch, missing calibration/control, and failed no-shift controls all block fail-closed.
- Distribution-free requests and unsupported geometries return honest `unsupported`, never in-distribution.
- The detect executor integrates through the workflow (`completed(7)`) with no method logic in the coordinator.
- No 80.12+ family, localization, centralized control execution (80.15), explanation/intervention, or real benchmark proof claim is made.

## Negative Results and Limitations

- Detector inputs in tests/smoke are synthetic NumPy batches (Gaussian reference, shifted OOD, held-out ID), not real core-model captures; real core benchmark proof remains 80.23/80.24.
- The frozen core manifests predeclare no density/OOD/drift metrics, so tests/smoke use a synthetic manifest-shaped mapping with the same frozen schema semantics plus a directly constructed `DetectionConfig`; no frozen manifest was mutated.
- The drift statistic is the mean calibrated-score shift under one GMM (2 components, `random_state=training_seed`); it is not a distribution-free/nonparametric test — such requests are explicitly unsupported.
- Bootstrap: 50 flag-rate resamples reuse fitted calibrated scores; 50 drift-gap resamples jointly resample fitted test and held-out reference scores (exactly `repetitions` draws under the evaluation seed). No refits. GMM fit/calibration dominate detector cost.
- Density requires euclidean/unit_norm geometries, matching feature widths, and declared identities; `full` covariance needs more samples than dimensions.
- No formatter, linter, type checker, packaging, docs build, or project-wide suites were run per task constraints.

## Files Modified

- `src/latent_anything/_density_drift_detection.py` — private detector plus single-evaluation seam (`Provenance`, `ReferenceTestPair`, `_fit_once`, `_score_with`, `_DetectionContext`, `_evaluate_context`, `_decide`, `_control_table`, `evaluate_detection`); `make_detect_executor` calls `evaluate_detection` exactly once; private module only; zero top-level export changes.
- `tests/test_density_drift_detection.py` — 15 consumer-observable tests (prior 13 plus drift-gap bootstrap shape/determinism and unit_norm geometry/provenance propagation; shuffled test now asserts the per-column method and seed).
- `scripts/sprint80_task80_11_smoke.py` — committed direct smoke covering every acceptance shape.
- `docs/sprint-plans/sprint-80.md` — 80.11 marked `[x]`; 80.12+ pending.

## Evidence-Review Readiness

Ready: focused detector tests prove reference/test identity and held-out calibration gate the claim, uncertainty and shuffled controls are recorded, positives/negatives separate, distribution-free/failed-control cases cannot promote, one fit plus one calibration per executor call is instrumented, results are deterministic, the detect executor completes through the method-agnostic workflow, fail-closed inputs reject; the committed smoke reproduces every acceptance shape; the nine-name 80.6 surface is verified unchanged.

## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, status, and decision log:

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/255 uncached files (39%) [8 workers]
  AST extraction: 200/255 uncached files (78%) [8 workers]
  AST extraction: 255/255 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph (...). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1049 saved labels, 1026 communities now; renamed 184 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 14390 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1026 community nodes, 1164 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 14390 nodes, 30561 edges, 1026 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit code 0
```

Index/freshness evidence: `graphify-out/graph.json` (15:13:44) is newer than `docs/sprint-plans/sprint-80.md` (14:55:16) and all indexed code/test/script inputs (`_density_drift_detection.py` 15:07:13, `test_density_drift_detection.py` 15:11:06, `sprint80_task80_11_smoke.py` 15:11:53). Rebuilt graph: 14390 nodes, 30561 edges, 1026 communities. This Markdown summary postdates the graph because the AST code-graph update does not index Markdown. Final confirming run (executed after this record was written; no file writes followed it) is reported to Main via hub.
