# Task 80.16 — Probe, TCAV, and Integrated-Gradients Explanation Evidence

## Status

**Complete (provisional, pending independent evidence review).** Only Sprint 80 task 80.16 is claimed here; tasks 80.1–80.15 are independently evidence-gated and remain intact. Task 80.17 is separately complete/provisional and outside 80.16's implementation scope; 80.18+ are untouched/unchecked. Historical evidence unchanged. Frozen taxonomy/manifests/report schema/validator not mutated; no public API growth.
## Summary of Work

Added `src/latent_anything/_probe_tcav_ig_explanation.py`, one private architecture-neutral explain-stage seam. One immutable `ExplanationHypothesis` (with `manifest_id` plus a private method-specific role→declared-control-ID binding: probe `capacity`/`randomized`/`negative`, tcav/IG `fidelity`/`stability`/`selectivity`, each declared as `"<role>:<id>"`) binds symptom/family, target, representation/layer/slice, method, expected direction, dataset/split identities, seeds, required controls, thresholds, and baseline/concept definitions. Every executed control runs under its exact declared identity; a declared required control not executed/passed blocks support. Both `make_explain_executor` paths bind the expected manifest/run identity (plus declared representation/dataset/split/target identity against the request capture and prior detect/localize outputs) and fail with `StageContractError` on mismatch. `evaluate_explanations` rejects non-`SUPPORTED_METHODS` hypotheses before any callback fires. Methods execute only for declared hypotheses via caller-supplied `MethodInputs` bundles; undeclared pairs yield honest `omitted` records with callbacks uncalled. Each executed method emits one canonical non-causal evidence record gating fidelity, stability, selectivity, leakage, and uncertainty; any missing/failed dimension or required control blocks promotion. `explanation_report_items` renders shape-compatible fragments (`validate_report_shape` only; `validate_diagnostic_report` requires 80.21/80.22 registration and metric-bearing rows; refs provisional).

- Probe: reuses `probes._fast_probe` (training-only StandardScaler, LogReg C=1.0/lbfgs/balanced); held-out fidelity only (train accuracy never promoted); sign-aware coefficient stability across declared seeds; central label-permutation gap for selectivity; disjoint-identity + distinct-split + capacity (`linear-logreg-C1.0-standardized`) leakage gates; central bootstrap uncertainty over fitted predictions.
- TCAV: reuses `tcav.learn_mean_diff_direction`/`learn_linear_separator_direction` for CAV fits; caller-fitted gradients; concept-classifier fidelity, sign-aware CAV stability, ten-permutation averaged random-concept selectivity margin plus declared negative concepts, identity/target leakage gates, sign-resample uncertainty.
- IG: consumes one declared target/baseline run (completeness error read from the caller's `IntegratedGradients` result); completeness fidelity, baseline/seed cosine stability, off-target + random-baseline selectivity margin, baseline/input/target identity-drift leakage gates, coordinate-resample uncertainty. Disconnected/non-differentiable targets yield `unsupported`.
- All resampling/controls route through 80.15 `ControlPlan`/`execute_plan`/`run_bootstrap`/`failed_required` with executor-owned RNG; `ControlOutcome` payloads are non-empty and wired into records. One shared `_ExplainContext` per `evaluate_explanations` call: no duplicate fits/gradients. `make_explain_executor` integrates with `DiagnosticWorkflow` carrying no algorithm; `explanation_report_items` renders `kind="explanation"`, `causal=False` claims.

## Validation Evidence

```text
uv run pytest tests/test_probe_tcav_ig_explanation.py -q
22 passed

uv run python scripts/sprint80_task80_16_smoke.py
PASS positive path: probe/TCAV/IG support under declared hypotheses
PASS blocked path: weak probe inconclusive (failed-fidelity:heldout_accuracy)
PASS unsupported path: disconnected IG target is unsupported, never promoted
PASS undeclared path: method code never invoked, honest omission
PASS workflow path: explain executor binds manifest/run identity and completed
PASS report path: explanation fragments shape-validate (validate_report_shape only), causal=False throughout
PASS determinism: repeated evaluation reproduces the record exactly
public surface: 9/9 80.6 names intact, no explanation symbols leaked
exit code 0
```

## Binding Correction (Final Deep Review)

- Replaced the generic `_prior_field_set` arbitrary-string harvester with dedicated structured extractors: `_detect_family_ids` (explicit `family_id`/`family_ids`, known-list string entries, identity-keyed `family_evidence` keys only), `_detect_metric_ids` (explicit `metric_id`/`metric_ids`, per-family `observed_metrics`/`threshold_pass` keys, declared `config` metric identities including thresholds only), and `_localize_declared_ids` (exact `family_id`/`metric_id` when the localize payload declares them). The recursive "add every nested string" path is removed.
- Verified against the real detect payload: families == `{'separability_probe_leakage'}`, metrics == `{'heldout-probe-accuracy', 'probe-leakage-gap'}`; status word `"observed"` and control-kind word `"null"` no longer bind.
- New real-payload regressions in both executor suites: actual family + actual metric pass; `family_id='observed'` and `metric_ids=('null',)` fail `StageContractError` before callbacks; fabricated IDs still fail. Axial regressions use a relabeled axial-family detect composition so every completed hypothesis links to the exact real detect family/metric plus the exact declared localize family/metric.
- Both explain-executor docstrings now name non-empty `localization_bindings` as the only upstream localization contract (every binding exact-matched); `layer_id`/`slice_id` are hypothesis-local method context, never upstream-localized by presence; family binds against the detect payload (and the localize family only when the payload declares it).

## Behavioral Claims

- Undeclared hypothesis/method never invokes method code; record reads `omitted` with `claim_allowed=False`. Feature methods raise `ExplanationError` before any IG callback.
- Leakage-safe selective stable probe supports; train-only, leaky, random-label, non-separable-negative-separating, declaration-mismatch cases block.
- Synthetic known concept TCAV supports; unstable/random concept cannot.
- Differentiable IG run with completeness/stability/selectivity supports; bad baseline fails fidelity, disconnected target is `unsupported`.
- Missing evidence/control/declaration in any method blocks; payload control keys equal the declaration; supported records prove all declared required controls ran/passed; streams/uncertainty deterministic; fragments shape-validate with `causal=False`; explain executors bind manifest/run identity (run manifest, capture representation, manifest split, detect family/metric, explicit localization bindings) and verify prior stage order; nine-name surface and frozen artifacts intact.
## Negative Results and Limitations

- Single-permutation random-concept TCAV baseline was too high-variance to gate (observed 1.0 on the positive fixture); replaced with a ten-permutation mean inside one central callback, same executor-owned stream.
- Existing `assemble_tcav_result` not reused for scoring (its internal loops own local RNG); same statistics replayed under central streams.
- Synthetic NumPy fixtures only, not real core-model captures; real proof remains 80.23/80.24. No causal claims (80.18+). SAE/lens/geometry explanations are owned by separately-complete task 80.17, outside this task's scope.
- No formatter/linter/type-check/package/docs-build/project-wide suites per task constraints.

## Files Modified

- `src/latent_anything/_probe_tcav_ig_explanation.py` — private adapter (hypothesis with `manifest_id` + role→declared-control binding, evidence, three evaluators, method guard, structured identity-bound executor, fragment report items); zero top-level exports.
- `tests/test_probe_tcav_ig_explanation.py` — 22 consumer-observable tests (structured real-payload identity binding, declaration-driven controls, TCAV alias removal, method guard).
- `scripts/sprint80_task80_16_smoke.py` — committed direct smoke (positive, blocked, unsupported, undeclared, workflow, report, determinism, surface paths).
- `docs/sprint-plans/sprint-80.md` — only task 80.16 marked [x] provisionally here (task 80.17 is separately complete/provisional; 80.18+ untouched/unchecked).

## Scope and Public-Surface Evidence

- Graphify query navigation used before edits (probe/TCAV/IG/control/workflow seam mapped).
- Nine-name 80.6 surface verified; no `ExplanationHypothesis`/`evaluate_explanations`/`ControlPlan` top-level leak.
- Frozen artifacts untouched: `benchmark_manifest_schema_v1.json`, `diagnostic_report_schema_v1.json`, `representation_problem_taxonomy_v1.json`, both core benchmark manifests.

## Graph Refresh Record

Final task-level step, run after implementation, focused validation, artifact, and status writes:

```text
graphify update .
[graphify watch] community set changed since labeling (1075 saved labels, 1046 communities now; renamed 169 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-22/
Graph has 14950 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1046 community nodes, 1181 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 14950 nodes, 32730 edges, 1046 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
[graphify watch] community set changed since labeling (1059 saved labels, 1047 communities now; renamed 196 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) -> 2026-09-22/
Graph has 14957 nodes (above 5000 limit). Building aggregated community view...
graph.html written (aggregated: 1047 community nodes, 1376 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt: 14957 nodes, 32743 edges, 1047 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Wall time: 77.90 seconds
```

Ordering: the rebuild recorded above preceded this final confirmation; the final exact result is reported to Main with no writes afterward.
