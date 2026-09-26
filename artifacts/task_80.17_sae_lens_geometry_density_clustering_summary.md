# Task 80.17 — SAE, Lens, Geometry, Density, and Clustering Explanation Evidence

## Status

**Complete (provisional, pending independent evidence review).** Sprint 80 tasks 80.1–80.17 are done; 80.18+ untouched/unchecked. Historical evidence unchanged. Frozen taxonomy/manifests/report schema/validator not mutated; no public API growth. 80.16 behavior preserved (22/22 regression green).

## Summary of Work

Added `src/latent_anything/_sae_lens_geometry_density_clustering.py`, one private feature explain-stage adapter that reuses — never duplicates — the 80.16 common convention (`ExplanationHypothesis` with `manifest_id` plus the shared private role→declared-control binding, `ExplanationEvidence`, `MethodInputs`, central 80.15 executor, canonical payload, `explanation_report_items` fragment contract, identity-bound explain executor). The 80.16 carriers accept the five feature method names through a shared private registry (`EXTENDED_METHODS`/`_ALL_METHODS`); `SUPPORTED_METHODS` itself is frozen and both 80.16 executors reject feature methods before any callback fires. Every feature control runs under its exact declared `fidelity`/`stability`/`selectivity` identity; the missing symptom relationship blocks before method work; semantic labels/projections promote only on full-gate support with redaction otherwise.

- SAE/sparse: one `DictionaryLearning` fit per declared seed; `match_by_decoder_cosine` permutation-invariant alignment; reconstruction-quality fidelity; cross-seed fraction stability; headline-feature activation selectivity versus negative/structured-off-target/random-feature slices; central bootstrap uncertainty over fitted cosines/activations; central shuffled control.
- Lens: `apply_logit_lens` + `softmax` readout; declared target/off-target indices and token/sample/preprocessing identity; readout fidelity (row cosine vs declared recomputation); distribution-level target-mass stability across seed logit bundles; symptom-vs-benign readout-margin selectivity; central resampling + shuffled control.
- Geometry: covariance-eigen fit + `OrthonormalSubspace` identity binding; fit fidelity; split-half principal-angle (`subspace_alignment`) stability plus declared seed matrices; coverage-margin selectivity versus benign/negative slices; column-shuffle null; coverage resampling.
- Density: single `GaussianMixtureDensity` fit + held-out calibration (one fit, one calibration; score-only passes; cross-identity scoring rejects); AUROC fidelity; seed-score spread stability; symptom flag-margin selectivity versus negative; shuffled null; AUROC resampling. No "outlier means cause" claim.
- Clustering: `KMeans` fits per declared seed plus one benign fit; label-agreement fidelity; permutation-invariant ARI stability (no raw-ID comparison); shuffled-label selectivity margin; assignment resampling. Semantic cluster label only on full support.

## Validation Evidence

```text
uv run pytest tests/test_sae_lens_geometry_density_clustering.py -q
17 passed in 267.85s

uv run pytest tests/test_probe_tcav_ig_explanation.py -q
22 passed

uv run python scripts/sprint80_task80_17_smoke.py
PASS five supported synthetic shapes: sae/lens/geometry/density/clustering
PASS redaction: failed gates drop the semantic label, anonymous evidence retained
PASS undeclared: method code never invoked, honest omission
PASS unsupported: identity drift blocks promotion
PASS workflow: feature explain executor binds manifest/run identity and completed
PASS report: feature fragments shape-validate (validate_report_shape only; validate_diagnostic_report needs 80.21/80.22 registration), causal=False throughout
PASS determinism: repeated evaluation reproduces the record exactly
public surface: 9/9 80.6 names intact, no feature symbols leaked
exit code 0
```

## Behavioral Claims

- Valid positive and blocked/negative for each of the five methods.
- Raw index/cluster permutation leaves stability unchanged (decoder-cosine matching, subspace alignment, ARI).
- Unstable/random/nonselective evidence blocks with semantic label/projection absent/redacted.
- Missing relationship blocks before method invocation; undeclared methods uncalled.
- All five dimensions + control records present; central deterministic streams with exact reps; density one-fit/calibration; no duplicate expensive work (instrumented fit/calibrate/score counts).
- Report-schema/canonical/workflow explain integration (kind="explanation", causal=False); 80.16 regressions green (22/22); nine-name surface and frozen artifacts intact.
- Binding semantics: every feature hypothesis declares explicit non-empty `localization_bindings` (no permissive default); the feature executor reuses the shared `_check_explain_identities` gate, so matching checkpoint/token/time bindings from a real `evaluate_axial_localization` payload complete while mismatched/absent/not_applicable bindings raise `StageContractError` before any method callback.
- Structured linkage correction (final deep review, shared `_probe_tcav_ig_explanation` seam): generic `_prior_field_set` harvesting replaced with `_detect_family_ids` / `_detect_metric_ids` / `_localize_declared_ids`; status words (`"observed"`) and control-kind words (`"null"`) no longer bind; new real-payload regressions (actual family+metric pass; `observed` / `('null',)` / fabricated IDs fail closed) added to this suite's workflow integration test; axial regression uses a relabeled axial-family detect composition. No promotion/control/report-fragment contract changed.

## Negative Results and Limitations

- Single-permutation intuition failed twice during development: (1) lens vector-cosine stability collapsed to 0.0 on near-tied readout rows — replaced with distribution-level target-mass drift; (2) SAE random-input activation comparison could not separate specific from diffuse features (noise projects onto every atom) — replaced with a random-*feature* margin plus structured off-target slices, matching the task's "random features" wording.
- Raw-QR split-half geometry stability undervalued the fitted subspace — replaced with centered-covariance halves.
- Synthetic NumPy fixtures only; real core proof remains 80.23/80.24. No interventions/causal work. No formatter/linter/type-check/project-wide suites per task constraints.

## Files Modified

- `src/latent_anything/_sae_lens_geometry_density_clustering.py` — private feature adapter (five evaluators with declaration-bound controls, relationship gate, identity-bound executor); zero top-level exports.
- `src/latent_anything/_probe_tcav_ig_explanation.py` — shared carriers accept the five feature names (`EXTENDED_METHODS`/`_ALL_METHODS`); `SUPPORTED_METHODS`, method guard, role binding (`METHOD_CONTROL_ROLES`/`_bound_control_ids`), identity checker, and fragment report contract added.
- `src/latent_anything/_layer_slice_localization.py` — stale `unsupported_axes_80_14` marker removed; confidence=1.0 documented as deterministic selection certainty.
- `src/latent_anything/_redundancy_separability_detection.py` — redundancy null consumes the already-evaluated result (no second transform/evaluation).
- `tests/test_sae_lens_geometry_density_clustering.py` — 17 tests (identity binding, declaration-driven controls, redaction, axial checkpoint/token/time binding regression).
- `tests/test_probe_tcav_ig_explanation.py` — 22 tests (identity binding, declaration-driven controls, method guard, axial regression).
- `tests/test_layer_slice_localization.py` — marker removal + confidence semantics assertions.
- `tests/test_statistical_controls.py` — redundancy single-evaluation instrumented test.
- `scripts/sprint80_task80_16_smoke.py`, `scripts/sprint80_task80_17_smoke.py`, `scripts/sprint80_task80_10_smoke.py` — identity binding, fragment wording, retuned weak fixture.
- Binding correction (this review): explicit `localization_bindings` on every 80.17 hypothesis fixture/caller plus `verdict=localized` on workflow localize payloads; the shared empty-bindings fail-closed check and axial-status gating apply to both executors; no permissive default added.
- `artifacts/task_80.13*`, `task_80.14*`, `task_80.15*`, `task_80.16*` summaries — scope/fragment/duplicate corrections.

## Scope and Public-Surface Evidence

- Graphify query navigation used before edits (SAE/lens/geometry/density/clustering primitives mapped).
- Nine-name 80.6 surface verified; no feature symbols leaked top-level.
- Frozen artifacts untouched: `benchmark_manifest_schema_v1.json`, `diagnostic_report_schema_v1.json`, `representation_problem_taxonomy_v1.json`, both core benchmark manifests (synthetic fixtures only; no manifest edits).

## Graph Refresh Record

Recorded rebuild, run after every source/test/script/artifact/status write for the final deep re-review corrections:

```text
graphify update .
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

Ordering: the rebuild recorded above preceded the final confirmation; the final exact result is reported to Main with no writes afterward.
