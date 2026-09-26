# Task 80.10 — Integrate Redundancy, Superposition, Separability, and Probe-Leakage Detection

## Status

**Complete.** Sprint 80 tasks 80.1–80.10 are done; tasks 80.11–80.29 remain pending. Historical Sprint 79 evidence was not changed. Frozen taxonomy/manifests/validator were not mutated; observed results were not encoded.

## Summary of Work

Added `src/latent_anything/_redundancy_separability_detection.py`, the smallest private detector-family adapter behind the frozen taxonomy/workflow contracts for `redundancy_superposition` and `separability_probe_leakage`. It consumes bound labeled representations plus predeclared metric/control/threshold configuration and emits machine-readable observations, taxonomy family-evidence statuses (`evaluate_claim`), control outcomes, seeded uncertainty metadata, and honest `supported`/`inconclusive`/`unsupported` outcomes. A supplied detect-stage executor integrates through the method-agnostic workflow with no method logic in the coordinator and no public API widening.

- `probes._fast_probe` is the single shared probe-fitting seam (training-only `StandardScaler`, `LogisticRegression` C=1.0/lbfgs/balanced); the detector passes caller-declared disjoint splits through it and recomputes accuracy against leakage-safe labels. No parallel probe path exists.
- `fit_covariance` from `geometry` supplies the regularized covariance behind max-absolute-correlation (one fit per evaluated batch, plus one refit per bootstrap draw for the correlation interval only).
- `DictionaryLearning` supplies the sparse overcomplete comparison as dictionary-atom coherence: maximum absolute cosine between distinct normalized atoms (one fit per evaluated batch, `max_iter=100`, seeded by the manifest control seed; the sharing interval resamples fitted coherence values without refitting).
- Taxonomy evidence gating reuses `evaluate_claim`; manifest validity and canonical digests reuse `validate_manifest`; canonical JSON reuses the shared `canonical_json` contract.

Metric/control semantics:

- Separability metrics are exactly the frozen transformer manifest pair, both `>=`: `heldout-probe-accuracy` (threshold 0.7, tolerance 0.02) and `probe-leakage-gap` (threshold 0.15, tolerance 0.02). Threshold direction/identity come from the manifest/config only; nothing is tuned from observed values.
- A supported separability claim requires all of: leakage-safe disjoint train/eval splits with unique sample identities and distinct split identities, the declared bounded capacity (`linear-logreg-C1.0-standardized`, params within train size), the seeded label-randomization gap, the non-separable negative control, and every required control passed. Any leakage, overlap, missing/failed control, capacity violation, or weak gap blocks promotion (`inconclusive`, `claim_allowed=False`). Held-out accuracy alone never promotes: an unmet headline or gap threshold blocks even when every control passes.
- Capacity (`capacity` kind), label-randomization (`randomized` kind), and split-swap (`null` kind) controls are derived by the detector from the target batch under manifest seeds; supplying batch data for them is a fail-closed error. The non-separable negative (`counterexample`/`negative` kind) is the only separability control that takes caller-supplied batch data.
- Redundancy/superposition reports declared correlation and sparse-overcomplete evidence with an explicit counterexample and a randomized-feature null under predeclared decision rules. Both lenses must agree to promote; correlation alone (or sharing alone) yields `inconclusive` with `inconsistent-evidence:redundancy_superposition` and no semantic superposition is inferred. Where the manifest predeclares no promotable declaration for this family, the detector returns honest `unsupported` (`missing-declaration:redundancy_superposition`) rather than borrowing another family's label.

Single-evaluation seam: `evaluate_detection` fits every probe and every dictionary exactly once per executor call. Separability bootstrap resamples fitted probe predictions without refitting; correlation uncertainty refits `fit_covariance` once per draw (the only bootstrap refit); sharing uncertainty resamples fitted coherence values without refitting. One shared context feeds family decisions and payload assembly. `detect_families`/`detection_payload` remain documented unit-check seams whose naive composition is known-double.


## Correction Round — Evidence-Gate Findings (Same Task, No Scope Change)

1. Removed the parallel `StandardScaler`+`LogisticRegression` path: the detector now calls the shared `probes._fast_probe` seam (identical train-only scaling/estimator semantics) through a thin `_probe_predictions` wrapper; no second probe convention exists and no shared symbol was changed.
2. Added a real observable high-accuracy-alone case (weak-signal fixture acc=0.783, gap=0.100): held-out accuracy passes while the leakage gap fails, yielding `inconclusive`, `claim_allowed=False`, `missing_evidence` containing `threshold:probe-leakage-gap`, and a reason stating probe accuracy alone is insufficient. The separability threshold gate now precedes the control gate so the threshold reason is reported even when a control also fails. The misleading prior test/smoke claim is replaced.
3. Corrected bootstrap documentation: correlation uncertainty refits `fit_covariance` once per draw; only prediction/coherence bootstrap reuses fitted outputs. Module, helper, and artifact text now state this exactly. Post deep-review correction: the redundancy column-shuffle null executes only under an explicitly declared config control linked to the redundancy metrics; with no such declaration the null never runs and no invented `control-null-shuffle` identity enters the payload controls table (single-evaluation fix preserved). Synthetic fixtures/smoke predeclare `control-redundant-null-shuffle` where support needs it, plus a regression test proving an undeclared null never appears.
```text
correction validation:
uv run pytest tests/test_redundancy_separability_detection.py -q
14 passed
uv run pytest tests/test_mlp_probe.py tests/test_probes.py tests/test_collapse_detection.py tests/test_diagnostic_workflow.py tests/test_diagnostics_request.py tests/test_sprint80_core_manifests.py tests/test_capture_binding.py -q
124 passed, 5 skipped
uv run python scripts/sprint80_task80_10_smoke.py
PASS high-accuracy-alone blocked acc=0.783 gap=0.100
exit code 0

prior validation (pre-correction, retained):
uv run pytest tests/test_redundancy_separability_detection.py -q
14 passed

uv run python scripts/sprint80_task80_10_smoke.py
PASS config manifest=sprint80-core-transformer-hidden-state-probe-v1 metrics=['heldout-probe-accuracy', 'probe-leakage-gap'] reps=200
PASS separable positive acc=0.983 gap=0.517
REJECT overlap: train/eval split leaks: a sample position appears on both sides
PASS capacity/memorization control blocks promotion
PASS label-randomized control blocks promotion
PASS non-separable negative blocks when it separates
PASS high-accuracy-alone blocked (gap threshold is load-bearing)
PASS redundancy positive corr=0.997 sharing=0.759
PASS honest unsupported path (no borrowed family label)
PASS deterministic acc=0.983
PASS detect executor through workflow completed(7)
REJECT missing-control: required controls are missing batch data: control-nonseparable-negative
exit code 0

uv run python -c "import latent_anything as la; ..."
9 ['CaptureSelection', 'ComparisonRequest', 'ControlSelection', 'DiagnosticRequest', 'DiagnosticRequestError', 'DiagnosticResult', 'DiagnosticSelection', 'InterventionRequest', 'OutputSelection']
9
False False False False
```

The final check confirms the 80.6 nine-name surface is intact, `ResultStatus`/`DiagnosticWorkflow` stay out of the top level, and no redundancy/separability detector symbol was added.

## Affected Claims

- Separability positives promote only with leakage-safe splits, capacity, randomization gap, non-separable negative, and all required controls passed; the frozen transformer manifest thresholds gate the claim.
- Overlap/leakage, duplicate sample identities, identical split identities, capacity violations, noise memorization, and separable negatives all block promotion fail-closed.
- Redundancy positives report both lenses with a counterexample and a randomized-feature null; correlation alone cannot promote; missing predeclared metrics return honest `unsupported`.
- The detect executor integrates through the workflow (`completed(7)`) with no method logic in the coordinator.
- No 80.11+ family, localization, centralized control execution (80.15), explanation/intervention, or real benchmark proof claim is made.

## Negative Results and Limitations

- Detector inputs in tests/smoke are synthetic NumPy batches (separable/noise/collinear/healthy), not real GPT-2/ConvVAE captures; real core benchmark proof remains 80.23/80.24.
- The frozen transformer manifest wires both metrics to `separability_probe_leakage` and predeclares no redundancy metrics, so redundancy tests use a directly constructed `DetectionConfig` with the same frozen threshold/control semantics; the transformer manifest itself was not mutated.
- Sharing is dictionary-atom coherence (max inter-atom cosine), not reconstruction gain: reconstruction gain saturates near 1.0 for healthy independent features, while coherence separates collinear redundancy (~0.76–0.79) from healthy independence (~0.58–0.69) under the frozen decision rules.
- Bootstrap: 200 probe-prediction resamples and 200 coherence resamples reuse fitted outputs without refitting; the 200 correlation draws refit `fit_covariance` once per draw (the only bootstrap refit). Dictionary fits dominate detector cost (`max_iter=100`, `n_components=2*dim`).
- Redundancy requires at least two feature dimensions with strictly positive per-feature variance; separability requires at least two classes overall and per split.
- No formatter, linter, type checker, packaging, docs build, or project-wide suites were run per task constraints.

## Files Modified

- `src/latent_anything/_redundancy_separability_detection.py` — private detector plus single-evaluation seam; correction round: probe fits route through shared `probes._fast_probe`, separability threshold gate precedes the control gate, bootstrap refit documented honestly; zero top-level export changes.
- `tests/test_redundancy_separability_detection.py` — 14 consumer-observable tests; correction round: real high-accuracy-alone case (acc passes, gap fails, `threshold:probe-leakage-gap`) replaces the misleading claim; memorization test updated to threshold-first semantics.
- `scripts/sprint80_task80_10_smoke.py` — committed direct smoke; correction round: real high-accuracy-alone case (acc=0.783 gap=0.100) replaces the stale claim.
- `docs/sprint-plans/sprint-80.md` — 80.10 marked `[x]`; 80.11+ pending.

## Evidence-Review Readiness

Ready: focused detector tests prove probe accuracy alone cannot promote a diagnosis (leakage/capacity/randomization/non-separable controls gate the claim), redundancy evidence is bounded with an explicit counterexample (correlation alone cannot promote; missing declarations return honest unsupported), results are deterministic, the detect executor completes through the method-agnostic workflow, fail-closed inputs reject; the committed smoke reproduces every acceptance shape; the nine-name 80.6 surface is verified unchanged.

## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, status, and decision log:

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction: 100/252 uncached files (39%) [8 workers]
  AST extraction: 200/252 uncached files (79%) [8 workers]
  AST extraction: 252/252 uncached files (100%) [8 workers]
  warning: 248 source file(s) produced zero nodes and are absent from the graph (...). A re-run will retry them (empties are no longer cached); if it persists, please report the file(s) (#1666).
[graphify watch] No code-graph topology changes detected; outputs left untouched.
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit code 0
```

Pre-final mtimes (before writing this record): `graphify-out/graph.json` 14:18:57 older than this summary (14:23:27) because the tool left outputs untouched (artifact-only change, zero graph nodes); newer than all code/test/script task files (`_redundancy_separability_detection.py` 14:14:52, `test_redundancy_separability_detection.py` 14:16:07, `sprint80_task80_10_smoke.py` 14:14:00) and `docs/sprint-plans/sprint-80.md` (13:55:31). Prior rebuilt graph: 14277 nodes, 30106 edges, 1042 communities.
Final no-write confirming run (executed after this section was written; no file writes followed it): `graphify update .` exited 0 with `No code-graph topology changes detected; outputs left untouched`. Root `graph.json` therefore retained 14:18:57 — newer than all indexed code/test/script/status inputs (detector 14:14:52, test file 14:16:07, smoke 14:14:00, sprint-80.md 13:55:31) but older than this Markdown summary, which the AST code-graph update does not index. No mtime freshness beyond what the tool produces is claimed; the exact final-run output was reported to Main via hub.
