# Task 80.9 — Integrate Collapse, Rank-Loss, Anisotropy, and Inactive-Dimension Detection

## Status

**Complete.** Sprint 80 tasks 80.1–80.9 are marked `[x]`; tasks 80.10–80.29 remain pending. Historical Sprint 79 evidence was not changed.

## Summary of Work

Added `src/latent_anything/_collapse_detection.py`, the smallest detector-family adapter behind the frozen taxonomy/workflow contracts for `collapse_rank_loss` and `anisotropy_inactive_dimensions`. It consumes bound `LatentValue` batches plus predeclared metric/control/threshold configuration and emits machine-readable observations, taxonomy family-evidence statuses (`evaluate_claim`), control outcomes, seeded bootstrap uncertainty/repetition metadata, and honest `supported`/`inconclusive` outcomes. A supplied `detect`-stage executor (`make_detect_executor`) wires it into the method-agnostic `DiagnosticWorkflow` with no algorithm in the coordinator and zero top-level export changes.

## Correction Round — Single-Evaluation Refactor (Evidence-Gate Finding)

Reviewer finding: `detect_families` computed the shuffled null plus two bootstrap intervals that went unused, then `detection_payload` recomputed them; control metrics were also recomputed via `_control_metrics_for`. One executor call therefore cost roughly twice the manifest-declared 200 resamples per metric plus duplicate control decompositions.

Correction (same task, no scope change): introduced one shared `_DetectionContext` per executor call. `_evaluate_context` validates and decomposes the target batch once (`_evaluate_batch`: one `compute_latent_health` spectrum + one thin SVD), each control batch once, the column-permuted shuffled null once, and the seeded bootstrap once via `_bootstrap_both` — exactly `repetitions` resample draws on one shared `default_rng(seed)` sequence, recording both statistics per draw. `_decide` consumes the context for family outcomes; `detection_payload(..., context=...)` reuses the same context for measurements/uncertainty instead of recomputing. New single-evaluation seam `evaluate_detection(value, config, controls)` returns decisions plus payload from one context; `make_detect_executor` calls it exactly once. `detect_families`/`detection_payload` remain as focused unit-check seams with docstring warnings that naive composition evaluates twice. Frozen thresholds, seeds, and the 200-repetition count are unchanged; all observable outputs are preserved.

Performance semantics after correction: one executor call costs 4 `compute_latent_health` evaluations (target + 2 controls + 1 shuffled null) plus exactly `repetitions` resample evaluations (200, not 400), one thin SVD per evaluated batch/draw, and one control pass. The instrumented test below observes this behaviorally (counting health/SVD calls through the executor), not by asserting on source text.

- Sample covariance, sorted eigenvalue spectrum, participation-ratio effective rank, collapsed/inactive fraction, and covariance condition reuse `compute_latent_health` from `_jepa_evaluation` (unbiased `np.cov`; the same estimator family the JEPA adapter and its tests already prove).
- The singular-value spread (`min_sv / max_sv` of the centered batch) extends that spectrum with one thin SVD the health primitive does not expose; no second covariance fit and no extra data copy beyond the centered view.
- Seeded bootstrap over sample rows (manifest `seeds.evaluation`, `uncertainty.repetitions=200`, `confidence_level=0.95`) and a column-permuted shuffled null (manifest `seeds.controls`) supply the repetition/interval/null metadata deterministically.
- Taxonomy evidence gating reuses `evaluate_claim`; manifest validity and canonical digests reuse `validate_manifest`; canonical JSON reuses the shared `canonical_json` contract.

Metrics/control semantics:

- Metrics are exactly the frozen encoder manifest pair, both `>=`: `bottleneck-effective-rank` (threshold 3.0, tolerance 0.1) and `bottleneck-singular-spread` (threshold 0.25, tolerance 0.02). Threshold direction/identity come from the manifest/config only; nothing is tuned from observed values.
- Collapse reads both metrics as rank evidence; anisotropy/inactive-dimension reads the same frozen spectrum through a directional-activity lens (same predeclared metrics/thresholds, distinct `anisotropy_*` taxonomy evidence identifiers) — no second estimator, no separate threshold path.
- Effective rank and singular spread are scale-invariant, so a globally scaled but structurally healthy batch keeps healthy values (negative, no defect promoted). Rank-deficient or directionally dead batches fail thresholds (positive, defect flagged) while required controls still pass.
- Controls share preprocessing and the declared thresholds: `control-healthy-counterexample` (required, full-rank reference must pass), `control-benign-low-variance` (required, scaled-healthy reference must pass), `control-null-shuffle` (optional, recorded not threshold-gated). A failed required control linked to a family's metrics forces `outcome=inconclusive`, `claim_allowed=False`, with `missing_evidence=(failed-control:<id>,)`.
```text
uv run pytest tests/test_collapse_detection.py -q
9 passed in 3.38s

uv run pytest tests/test_collapse_detection.py tests/test_diagnostic_workflow.py tests/test_diagnostics_request.py tests/test_sprint80_core_manifests.py tests/test_capture_binding.py -q
45 passed in 3.44s

uv run python scripts/sprint80_task80_9_smoke.py
PASS config manifest=sprint80-core-encoder-autoencoder-collapse-v1 metrics=['bottleneck-effective-rank', 'bottleneck-singular-spread'] reps=200
PASS collapse positive er=1.987 spread=0.0000
PASS anisotropy positive er=1.276 spread=0.0178
PASS benign scaled-low-variance negative (rank/spread healthy)
PASS failed-control blocks conclusion
PASS deterministic null_er=3.956
PASS detect executor through workflow completed(7)
REJECT non-finite: batch contains non-finite values
REJECT missing-control: required controls are missing batch data: control-healthy-counterexample, control-benign-low-variance
exit code 0

uv run python -c "import latent_anything as la; ..."
9 ['CaptureSelection', 'ComparisonRequest', 'ControlSelection', 'DiagnosticRequest', 'DiagnosticRequestError', 'DiagnosticResult', 'DiagnosticSelection', 'InterventionRequest', 'OutputSelection']
False False
```

Correction evidence: the new instrumented test `test_executor_evaluates_each_batch_bootstrap_and_control_exactly_once` drives the real `detect` executor through a `StageInvocation` while counting `compute_latent_health` and thin-SVD calls behaviorally. It asserts exactly `4 + repetitions` health evaluations (target + 2 controls + null + 200 resample draws, one shared draw per metric) and `4 + repetitions` SVDs, with both uncertainty blocks reporting `repetitions == 200`. The naive pre-correction composition would have cost 2x bootstrap loops (400 resample evaluations) plus duplicated control/health passes; the test fails on that composition by construction.

The `True` on the earlier probe matched only the pre-existing `detect_change_points` export; the final check confirms the 80.6 nine-name surface is intact, `ResultStatus`/`DiagnosticWorkflow`/`StageOutput` stay out of the top level, and no collapse/detector symbol was added.

## Affected Claims

- Both taxonomy families produce evidence-bound outcomes using existing primitives: collapse positives flag threshold failure with controls passed; anisotropy positives flag directional defects through the same frozen spectrum; benign scaled batches stay negative; healthy counterexample and shuffled-null controls pass with deterministic repetition metadata.
- The detect executor integrates through the workflow (`completed(7)`) with no method logic in the coordinator.
- No 80.10+ family, localization, centralized control execution (80.15), explanation/intervention, or real benchmark proof claim is made.

## Negative Results and Limitations

- Detector inputs in tests/smoke are synthetic NumPy batches (healthy/collapsed/anisotropic/scaled), not real ConvVAE/transformer captures; real core benchmark proof remains 80.23/80.24.
- The frozen encoder manifest wires both metrics to `collapse_rank_loss`; anisotropy therefore reuses the same two predeclared metrics through its own evidence identifiers rather than separate anisotropy metric ids.
- `control-null-shuffle` manifest wiring is optional and recorded, not threshold-gated; the estimator-level shuffled null is always computed in payload measurements deterministically.
- Bootstrap runs 200 resamples per metric per evaluation (manifest-predeclared); this is the dominant detector cost and is unchanged from the frozen manifest.
- Rank estimation requires `n_samples > dim` 2D finite batches; undersampled, non-2D, or non-finite inputs reject instead of estimating.
- No formatter, linter, type checker, packaging, docs build, or project-wide suites were run per task constraints.

## Files Modified

- `src/latent_anything/_collapse_detection.py` — private detector plus correction-round single-evaluation seam (`_EvaluatedBatch`, `_evaluate_batch`, `_DetectionContext`, `_evaluate_context`, `_decide`, `_bootstrap_both`, `evaluate_detection`); `make_detect_executor` now calls `evaluate_detection` exactly once; dead duplicate helpers removed; `evaluate_detection` exported from the private module only; zero top-level export changes.
- `tests/test_collapse_detection.py` — 9 consumer-observable tests (previous 8 plus the instrumented single-evaluation test); determinism and null-control tests now route through `evaluate_detection`.
- `scripts/sprint80_task80_9_smoke.py` — committed direct smoke (determinism check now routes through `evaluate_detection`); same acceptance shapes.
- `docs/sprint-plans/sprint-80.md` — 80.9 stays `[x]` provisionally; 80.10+ pending.

## Evidence-Review Readiness

Ready: focused detector tests prove both families separate genuine defects from benign low variance with shared preprocessing and predeclared thresholds, failed controls block promotion, results are deterministic, the detect executor completes through the method-agnostic workflow, single-evaluation is instrumented (200 resamples per metric, one control pass), and fail-closed inputs reject; the committed smoke reproduces every acceptance shape; the nine-name 80.6 surface is verified unchanged.

## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, status, and decision log:
```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/255 uncached files (39%) [8 workers]
  AST extraction: 200/255 uncached files (78%) [8 workers]
  AST extraction: 255/255 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph: hooks.json, evals.json, evals.json, act_policy_representation_benchmark.json, anisotropy_benchmark.json (+243 more). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] community set changed since labeling (1044 saved labels, 1064 communities now; renamed 182 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-21/
Graph has 14169 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1064 community nodes, 1280 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 14169 nodes, 29684 edges, 1064 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit code 0
```

Index/freshness evidence (correction round): `graphify-out/graph.json` (2026-09-21 05:59:03 +0000) is newer than `src/latent_anything/_collapse_detection.py` (2026-09-21 05:52:56 +0000), `tests/test_collapse_detection.py` (2026-09-21 05:53:56 +0000), `scripts/sprint80_task80_9_smoke.py` (2026-09-21 05:56:21 +0000), and the artifact summary (2026-09-21 05:57:53 +0000). Symbol indexing: `graphify query "collapse anisotropy evaluate_detection single evaluation"` resolves `evaluate_detection()`, `_evaluate_context()`, `_decide()`, `detection_payload()`, `DetectionConfig`, `DetectionError`, `FamilyDetection`, `LatentValue`, and the smoke main.
