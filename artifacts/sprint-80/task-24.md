# Sprint 80 Task 80.24 — Prospective transformer stability correction

**Plan status:** `[~]` — task remains in progress and is not marked `[x]`.

**Current outcome:** The prospectively frozen grouped stability supplement passed independent validation and deterministic report rendering on the pinned GPT-2 replay: held-out accuracy `1.0000`, `coef_stability=0.974878 >= 0.8`, and leakage gap `0.425490 >= 0.15`; all declared controls passed. The immutable v2 result still has `threshold:null` and is not counted as gated stability evidence. Main's review is pending; plan row stays `[~]`.

Historical proof record: [`artifacts/task_80.24_transformer_hidden_state_proof_summary.md`](../task_80.24_transformer_hidden_state_proof_summary.md). Plan: [`docs/sprint-plans/sprint-80.md`](../../docs/sprint-plans/sprint-80.md); the 80.24 row remains `[~]`.

## Historical v1 diagnosis and exact blocker

The proof used manifest `sprint80-core-transformer-hidden-state-probe-v1`, canonical commitment `c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a`. The pinned GPT-2 revision and Wikitext validation selection reproduced against the pinned L04 manifest: 3,760 official rows, 2,461 nonblank rows, and 2,048 selected indices/text hashes. Grouped original-index//8 split (seed 79) produced 1,549 train and 499 evaluation rows, with disjoint groups, both classes per partition, and the declared capacity gate satisfied.

The predeclared section-header attribute (`^ = .+ = $`) yielded 516 positives and 1,532 negatives. Native hidden-state capture and full re-extraction were bit-identical (pooled-state SHA-256 `b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b`). The bound final-layer matrix was `(2048, 768)`, capture identity `28f4fbd2c47231fec8908044d335d06a87b2fda2f18bde1e3f0567dcb8c05d89`, and truthful capture axes were `slice` and `feature`.

The actual incompatibility is:

- Frozen taxonomy `separability_probe_leakage` requires every capture to declare `{sample, feature, label}`.
- Frozen manifest v1 declares representation axes `{layer, token, slice}`. The binder synthesizes the `feature` dimension but cannot bind `sample` unless it is manifest-declared; `label` is not in binder `SUPPORTED_AXES` at all.
- The labels are a separate supervised target in `LabeledBatch`, not an activation-array axis. The report validator checks taxonomy-required axes against each real capture's axes, and persistence rebuilds that provenance from the real `BoundCapture`. A report-side alias or extra axis would not be truthful and cannot survive registration.
- The validator raised `taxonomy applicability mismatch for separability_probe_leakage`; `persist_diagnostic_artifact` refused before writes. Consequently, there is no fresh artifact to reload independently and no persisted report to render or re-hash.

No taxonomy, manifest, report schema, binder, scientific outcome, threshold, label rule, or historical output was changed to force a pass.

## Historical seven-stage results

A fresh run completed the exact shared stage chain: `capture`, `detect`, `localize`, `explain`, `intervene`, `compare`, `report`. The stage results were all `completed`; explanation evidence was `supported` with `claim_allowed: true`.

- Detection: held-out probe accuracy `1.0000 >= 0.7`; leakage gap `0.4489 >= 0.15`. Capacity, label-randomization, nonseparable-negative, and split-swap controls behaved as declared; randomized accuracy was `0.5511`, negative-control accuracy `0.5832`.
- Localization: `localized`, earliest affected layer `transformer.h.0`; layer accuracies were `[1.0, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`.
- Intervention: removal at `transformer.h.0` was supported; accuracy changed from `1.0000` to `0.7315` (effect `-0.2685`). Zero-strength, random, shuffled, and off-target controls passed.
- Comparison: `neither`; both held-out accuracy and leakage-gap deltas were `0.000000`.
- Report stage: structured report assembly completed and passed report-shape validation. Independent validation then blocked persistence on the axes mismatch; this is not evidence of a persisted/rendered report.

The section-header result establishes separability under the frozen protocol, not that the model causally uses section-header features. Row length and last-token identity remain confounds. The `neither` comparison is the truthful identical-replay result. These limits and the earlier fail-closed record remain historical evidence, not an accepted end-to-end artifact.

## Historical v1 verification

- `uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python scripts/sprint80_task80_24_proof.py` — exit 0 for truthful proof execution; `ACCEPTANCE: NOT PASSED`. All six frozen artifacts were byte-identical after the run. Runtime: proof body `498.91s`, wall including imports `505.22s`, extraction/replay `465.64s`, capture `8.75s`, detect `1.80s`, localize `0.21s`, determinism `1.65s`, workflow `3.19s`, peak traced memory `542.4 MiB`; Python 3.13.3, NumPy 2.4.6, Transformers 4.57.6, CPU PyTorch, scikit-learn 1.9.0, datasets 3.6.0.
- `uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" pytest tests/test_transformer_end_to_end_proof.py -q` — **6 passed in 66.60s**. This covers frozen inputs/grouping, controls, localization, the completed workflow and fail-closed persistence, and tamper/leakage/resume/control failures.
- `uv run pytest -q tests/test_capture_binding.py tests/test_diagnostic_validator.py` — **22 passed in 31.90s**.
- `uv run pytest -q tests/test_report_renderer.py tests/test_intervention_trials.py::test_comparison_only_task_metric_can_drive_a_causal_intervention` — **16 passed in 30.13s**, preserving the generic 80.23 report-rendering and comparison-only task-metric contracts.
- Workflow stage-output inspection: `uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python -c "import scripts.sprint80_task80_24_proof as p; w, r = p.build_workflow(); result, checkpoint = w.run(r, p.load_manifest()); print('workflow:', result.status, checkpoint.status, list(checkpoint.completed_stages)); print('stage outcomes:', {stage.stage: stage.outcome for stage in checkpoint.outputs}); print('explanation evidence:', checkpoint.outputs[3].payload['family_evidence'])"` — both workflow/checkpoint `completed`; all seven stage outcomes `completed`; explanation `supported`, `claim_allowed: True`, method `probe`.

No project-wide validation, formatter, linter, build, clean-environment reproduction, or unrelated task was run. Since persistence rejected before writing, fresh-context artifact reload and deterministic report rendering were not reachable and are not claimed.

## Authorized contract resolution

The user authorized a generic, versioned evidence correction. Real activation axes (`slice`, `feature`) stay bound to the registered capture. Labels, sample identities, exact train/evaluation membership, dataset split, target rule, representation, and capacity are stored in a distinct content-addressed target record. The independent v2 validator reconciles that record with detect-stage provenance and the actual capture artifact; tampering, sample-order misalignment, or forged capture axes reject.

The frozen v1 manifest, taxonomy, report-schema file, historical v1 failure, and scientific thresholds remain unchanged. Reports without target evidence continue through v1 and retain the original fail-closed taxonomy mismatch.

## Historical files from initial v1 blocker run

- `scripts/sprint80_task80_24_proof.py` — corrected the module documentation so its persistence/rendering success path is conditional on independent validation; the current frozen blocker is stated accurately.
- `tests/test_transformer_end_to_end_proof.py` — corrected the module documentation to describe the tested fail-closed result rather than claim a persisted artifact/report.
- `artifacts/task_80.24_transformer_hidden_state_proof_summary.md` — retained historical evidence and added this reconciliation.
- `artifacts/sprint-80/task-24.md` — this required handoff record.
- `graphify-out/graph.json`, `graphify-out/graph.html`, and `graphify-out/GRAPH_REPORT.md` — refreshed by the required graph update; six semantic/curated files were archived under `graphify-out/2026-09-23/`.

No library source or frozen input was changed. `docs/sprint-plans/sprint-80.md` was left unchanged; its task row remains `[~]` and the earlier Notes blocker remains historical evidence.

## Historical graph update

- `graphify update .` — exit 0 after the proof/test documentation updates; code graph refreshed without LLM. AST extraction processed 284 uncached code files. Rebuilt graph: 16,186 nodes, 37,386 edges, 1,107 communities; aggregated `graph.html`: 1,107 community nodes and 1,681 cross-community edges.
- Six prior semantic/curated graph files were backed up to `graphify-out/2026-09-23/`.
- Graphify warned that 274 source files yielded zero nodes and saved community labels were stale (1,081 saved labels vs. 1,107 communities; 212 communities renamed); it suggested `graphify label`, which was not run.

## Final rule-bound v2 proof — 2026-09-24

The current acceptance evidence is the final frozen-transformer run `421578bb1c06fa15`, persisted at `artifacts/diagnostics/proof-80-24-v2-rule-bound/`. The shared `capture → detect → localize → explain → intervene → compare → report` workflow completed all seven stages. Fresh artifact reload and independent validation passed; the deterministic rendered report was regenerated and matched; all 13 content-addressed evidence links were re-hashed. All six frozen inputs, including the manifest, remained byte-identical after the proof.

- Persisted run: `artifacts/diagnostics/proof-80-24-v2-rule-bound/runs/421578bb1c06fa15.json` (status `completed`).
- Artifact SHA-256: `e2ebd582ac9fa0defe03f1b221290067d931a9b99b606b661da75fccab7eee7a`.
- Diagnostic report SHA-256: `c670acea7b27665581844916f17fb03aa38b5aa6d7740583dea34a5a6f467bdb`.
- Deterministic rendered report: [`diagnostic-report`](../diagnostics/proof-80-24-v2-rule-bound/diagnostic-report), SHA-256 `788b9a316ff3ec14922b7a1515fd8aee2b0d2b4b4d0dc884f321eb0e68b8693a`. The report records validator `passed` and evidence contract `diagnostic-evidence-v2`.

### Target rule and sample/split provenance

Real activation axes remain `slice` and `feature`; target labels are not represented as an activation axis. The separately content-addressed target record binds the predeclared section-header rule and its sample/split provenance:

- Target record `target-section-header-attribute-record`, SHA-256 `3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19`.
- Target `section-header-attribute`; rule `section-header attribute '^ = .+ = $'`, SHA-256 `9b12485635decf738c4904702f75d7d5ae8cb6663555e2af5ee66476269f025b`.
- The record binds 2,048 labels and sample identities: labels SHA-256 `196284d27fbeca8c3d4ffe2e17310deacc3e2aa07b558c9fe743a31529974bee`; sample IDs SHA-256 `23fd5605b8c632e6b7823ab79a9d36061947fe5c916ea50bd5d10310e07ce181`.
- Exhaustive grouped split: 1,549 train rows (`grouped-original-index-div8-train79-1549rows`) and 499 evaluation rows (`grouped-original-index-div8-eval79-499rows`), with exact identities and memberships in the target record.
- Frozen manifest `sprint80-core-transformer-hidden-state-probe-v1`, commitment SHA-256 `c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a`; detect payload SHA-256 `663e80b0f655981a6e304f4e4de18c1888986981bf4d59a981bfa8176d65c87b`; localization payload SHA-256 `1129ac27a96fdbf79fff47e985779a7dcbd9c591e70319bedb0d2caceb00c2bc`.

### Seven-stage outcomes and scientific limits

- Detection observed held-out probe accuracy `1.0000` (threshold `0.7`) and leakage gap `0.4489` (threshold `0.15`); capacity, label-randomization, nonseparable-negative, and split-swap controls passed.
- Localization was supported at `transformer.h.0`; probe explanation was supported.
- Removal intervention at `transformer.h.0` was supported (accuracy `1.0000 → 0.7315`, effect `-0.2685`); zero-strength, random, shuffled, and off-target controls passed.
- Aligned replay comparison was `neither`, with both metric deltas `0.000000`; report assembly completed and independent validation passed.
- The frozen target is separable under this protocol, not evidence of causal feature use: section-header rows correlate with row length and last-token identity. The report also records the declared non-applicable token/time localization scope and per-sample localization evidence in the content-addressed stage record.

### Verification and chronology

- Final proof command: `uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python scripts/sprint80_task80_24_proof.py` — **ACCEPTANCE: PASSED**; seven stages complete; fresh independent validation, deterministic report, and 13 link checks passed; all six frozen files byte-identical.
- Final focused regression command: `uv run pytest -q tests/test_target_evidence.py tests/test_diagnostic_validator.py tests/test_redundancy_separability_detection.py tests/test_transformer_end_to_end_proof.py` — **37 passed in 44.40s**. Covers target/rule/sample/split integrity, tamper and misalignment rejection, forged-axis rejection, and preservation of the legacy v1 fail-closed/no-write behavior.
- Preliminary v2 pass remains at `artifacts/diagnostics/proof-80-24-v2/` (run `6e65beddb1dfeadf`; artifact SHA-256 `14e60dfb1d06843af92670e0bd4f789664d4039589da7fa8a199380205179b1d`; rendered-report SHA-256 `e6cac432eca469880314a6d44ff69f23779f8e84c020e00cf96b9ac305ac2716`; earlier focused battery 23 passed). It predates final rule-bound verification and is retained as preliminary chronology, not the final artifact.
- Historical frozen-v1 persistence refusal and `ACCEPTANCE: NOT PASSED` are retained above and in the linked proof summary; they are not relabeled as a successful v1 run. The preliminary root is also retained unchanged.
- 80.23 compatibility: current `load_diagnostic_artifact` reloaded run `6bca9afb72af32c4` from `artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-complete-20260923/` (artifact SHA-256 `e5c6db0ab5080807138e091bc332a5d3e788889a382917e93775b15f562aef0f`). Two current renders matched each other and persisted `encoder-diagnosis-v3.md` (5,690 bytes; SHA-256 `b8d713de3181ef39990d994d4f3c708e584eb3c559f762376e2b9f2de65de6e3`).

### Current changed files

- Library: `src/latent_anything/_target_evidence.py`, `_redundancy_separability_detection.py`, `_diagnostic_report.py`, `_diagnostic_validator.py`, `_diagnostic_artifact.py`, `_report_renderer.py`.
- Proof/tests/docs: `scripts/sprint80_task80_24_proof.py`, `tests/test_target_evidence.py`, `tests/test_transformer_end_to_end_proof.py`, `docs/AI_ENGINEER_GUIDE.md`.
- Records: `CHANGELOG.md`, `.agents/memory/decisions.md`, `.agents/memory/lessons-learned.md`, this task handoff, and [`the historical proof summary`](../task_80.24_transformer_hidden_state_proof_summary.md).
- Graph outputs: `graphify-out/graph.json`, `graphify-out/graph.html`, `graphify-out/GRAPH_REPORT.md`, `graphify-out/manifest.json`, `graphify-out/.graphify_labels.json`, and its signature; six prior semantic/curated files were backed up under `graphify-out/2026-09-24/`.

No project-wide tests, lint, formatter, or build were run. Main's evidence review is pending and the sprint plan remains `[~]`.

### Earlier graph refresh (before the stability correction)

`graphify update .` for the earlier v2 handoff processed 287 uncached code files; that graph had 16,245 nodes, 37,536 edges, and 1,111 communities. Its 2026-09-24 snapshot is historical and is superseded by the stability-correction refresh below.

- Warnings: 276 source files produced zero nodes; saved community labels were stale (1,085 saved labels versus 1,111 communities; 204 renamed). Graphify suggested `graphify label`, which was not run.
- Graphify's output says document/paper/image changes require the AI-assistant `/graphify --update` path. That path was not run, so this records a refreshed code graph, not semantic extraction of the edited Markdown artifacts.
- This earlier refresh predates the stability-correction source/test changes. The final graph refresh for task 80.24 is recorded in the supplemental correction section below.

The graph refresh does not change the task's pending evidence-review status; the plan row remains `[~]`.

## Prospective transformer stability correction — 2026-09-24

The immutable v2 run `421578bb1c06fa15` remains unchanged and retains its historical `coef_stability` record with a null threshold. The prospective supplemental protocol—not that historical observation—owns the stability gate. The proof uses the shared explanation evaluator, five group-subset refits, real cached GPT-2 features, an independent record validator, and deterministic content-addressed report rendering.

### Frozen protocol and provenance

- Current protocol: `artifacts/benchmark_manifest_sprint80_transformer_explanation_stability_v1.json`, protocol ID `sprint80-core-transformer-explanation-stability-v1`, SHA-256 `a69b7152a800ac543ac76fec4d699c16681517708a95b353e5d241241c8acf0a`.
- `coef_stability >= 0.8` was frozen before the first independent outcome, borrowed from the accepted 80.23 encoder-v3 run `6bca9afb72af32c4`, whose persisted artifact is `e5c6db0ab5080807138e091bc332a5d3e788889a382917e93775b15f562aef0f`; its accepted `proof-run.json` SHA-256 is `deb38278b6442db366d09a2b8b5c0271eada8162f3913efd65f37648b47ba7f1`. Threshold-source code SHA-256 is `87891b0c75d4ddcfe890792e421ac4108801d08aa49a9d3a234da764481cb53f`.
- A pre-run protocol snapshot with SHA-256 `4c0b58ab20a78156459a9c2a03fd55af4d41d35ed69a12c9522e26f3ab876c60` was reconstructed from the recorded edits and its bytes hash-verified. Before its output was inspected, only threshold-precedent provenance was corrected in the current manifest; the thresholds, model/data identity, split rule/seed, and probe seeds did not change. The final run records the provenance-complete current protocol hash above.
- Frozen inputs remain the original GPT-2 revision `e7da7f221d5bf496a48136c0cd264e630fe9fcc8` and WikiText revision `f776294184f13b8ff2337b3841cf9269a6216d1e`, with the same 2,048 selected validation rows and predeclared section-header rule. The new grouped split uses seed `83`: 1,538 train and 510 heldout rows, with zero sample/group overlap. It is a new split over the same source rows, not an independent corpus.
- The 2,048 ordered sample IDs and labels in the supplemental record were compared with the immutable v2 target record; both arrays match exactly. New-record digests are sample IDs `156c6517670fd4298324056e3581dc5064092287bfdbaf2fb9bf51182920333a` and labels `d3eb6ab1052f09a26a927ded5f75232e069546e9faa9c973372a2741b4df0e5b`; the different digest encoding does not alter the identical ordered arrays.
- The historical v2 artifact, explain-stage record, report, rendered report, run record, and null stability threshold were verified unchanged before and after the supplemental proof. Frozen capture axes remain `slice` and `feature`; labels/sample IDs remain separate target provenance.

### Measurements and content-addressed result

- Full GPT-2 re-extraction was bit-identical to the cached representation (pooled hidden-state SHA-256 `b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b`); real capture identity remains `28f4fbd2c47231fec8908044d335d06a87b2fda2f18bde1e3f0567dcb8c05d89`, shape `(2048, 768)`, dtype `float64`.
- Held-out accuracy `1.000000 >= 0.700` passed. Sign-aware minimum coefficient cosine `0.974878 >= 0.800` passed. Leakage gap `0.425490 >= 0.150` passed. Capacity, randomized-label, and nonseparable-negative controls passed; negative-control accuracy was `0.619608`.
- The five predeclared grouped subset refits (sampling/fit seeds) were `109/193` cosine `0.978693` (1,243 rows), `113/197` cosine `0.975197` (1,228), `127/199` cosine `0.974878` (1,211), `131/211` cosine `0.976767` (1,238), and `137/223` cosine `0.988987` (1,223).
- Independently validated content-addressed run record: [`a95ccce32461b78ddd0197188fc1f4b51a1b349d434d3a9812dbcc8251dbeb2d.json`](../diagnostics/proof-80-24-stability-v1/runs/a95ccce32461b78ddd0197188fc1f4b51a1b349d434d3a9812dbcc8251dbeb2d.json), SHA-256 `a95ccce32461b78ddd0197188fc1f4b51a1b349d434d3a9812dbcc8251dbeb2d`.
- Deterministic report: [`5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0.md`](../diagnostics/proof-80-24-stability-v1/reports/5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0.md), SHA-256 `5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0`.
- The initial content-addressed candidate (`runs/593545a20b136ede1400786c15d4665c444591d50f86569a00f0da566777812a.json`, report `reports/572588e3de89a3976c31f3a580cff3b24f8a3366be42415a01a877d609b6cb46.md`) was not accepted: independent validation caught that the shared `ExplanationHypothesis.to_dict()` omitted `target_id`. Its computed metrics were not treated as a validated artifact. The shared serializer now preserves `target_id`, with a regression test; the corrected replay reproduced the same metrics and passed validation. This replay validates the already-predeclared seed-83 outcome; it is not a second independent corpus or split, and no threshold or scientific setting was selected from the rejected candidate.

### Verification and boundaries

- `uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python scripts/sprint80_task80_24_stability_proof.py` — **ACCEPTANCE: PASSED**; protocol hash, old-evidence immutability, real replay, split, three gates, controls, independent validator, and report rendering all passed. Runtime `632.97s`, explanation `1.64s`, Python `3.13.3`.
- `uv run pytest -q tests/test_probe_tcav_ig_explanation.py` — **25 passed in 17.35s**. This includes a fail-able stability-gate regression, heldout-identity leakage rejection, and serialization of target identity.
- `uv run pytest -q tests/test_sae_lens_geometry_density_clustering.py::test_each_method_executes_only_for_declared_hypothesis` — **1 passed in 17.35s**.
- `uv run python -m py_compile scripts/sprint80_task80_24_stability_proof.py src/latent_anything/_probe_tcav_ig_explanation.py tests/test_probe_tcav_ig_explanation.py` — exit 0. Provenance preflight verified the current protocol/source hashes, accepted encoder precedent, prior v2 hashes, and historical null stability threshold.
- The broader `uv run pytest -q tests/test_sae_lens_geometry_density_clustering.py` attempt timed out at 300 seconds after 12 progress dots without a summary; it is not reported as passing. No project-wide suite, linter, formatter, or build was run. Main's evidence review remains pending and row 80.24 remains `[~]`.

The result supports only the declared probe separability under this protocol. It does not establish GPT-2 causal feature use; row length and last-token identity remain confounds. All original v1/v2 artifacts, the frozen transformer manifest/taxonomy, and their historical outcomes remain unchanged.

### Files changed or generated for this correction

- Shared code and behavior: `src/latent_anything/_probe_tcav_ig_explanation.py`; regression coverage: `tests/test_probe_tcav_ig_explanation.py`.
- Proof and predeclared protocol: `scripts/sprint80_task80_24_stability_proof.py`; `artifacts/benchmark_manifest_sprint80_transformer_explanation_stability_v1.json`.
- New content-addressed proof root: `artifacts/diagnostics/proof-80-24-stability-v1/runs/` and `reports/` (including the rejected pre-validation candidate, which remains clearly unaccepted).
- Related documentation/records: `docs/SPRINT_80_DEPTH_EVIDENCE.md`; `docs/AI_ENGINEER_GUIDE.md`; `CHANGELOG.md`; `.agents/memory/lessons-learned.md`; this handoff and the historical proof summary.

### Stability-correction graph refresh

`graphify update .` completed after the stability-correction source, tests, proof/protocol, and handoff edits. Graphify ran code-only AST extraction without LLM over 310 uncached code files. The current graph has 16,369 nodes, 37,712 edges, and 1,098 communities; aggregated `graph.html` has 1,098 community nodes and 1,707 cross-community edges. Six semantic/curated files were backed up under `graphify-out/2026-09-24/`.

- Graphify warned that 295 source files produced zero nodes and saved labels were stale (1,118 saved labels versus 1,098 communities; 217 communities renamed). `graphify label` was suggested but not run.
- The code graph is updated; the output states Markdown/document changes require the assistant `/graphify --update` path. That semantic update was not run, so task handoff and docs are not semantically indexed.
- Updated graph outputs include `graphify-out/graph.json`, `graphify-out/graph.html`, and `graphify-out/GRAPH_REPORT.md`; this refresh does not change plan status `[~]`.
