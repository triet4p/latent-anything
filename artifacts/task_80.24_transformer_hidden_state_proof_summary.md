# Task Summary: 80.24 — Historical v1/v2 evidence and prospective stability correction

**Sprint:** Sprint 80 — Stable Depth
**Task:** 80.24
**Status:** The historical rule-bound v2 artifact passed its original workflow and validator, but F2 found its `coef_stability` record had `threshold: null`; that run is not evidence of a gated stability result. The prospective supplement passed its independent validator and report checks. The sprint plan remains `[~]` pending Main's evidence review.

## Historical initial v1 proof summary

Added the committed executable proof `scripts/sprint80_task80_24_proof.py`,
which executes the frozen transformer manifest
(`sprint80-core-transformer-hidden-state-probe-v1`, commitment digest
`c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a`) through the
single high-level `DiagnosticRequest`/`DiagnosticWorkflow` using only the
accepted architecture-neutral seams (80.7 capture binding, 80.10 separability
detector, 80.13 layer/slice localizer, 80.16 probe explanation, 80.18
concept-removal intervention, 80.20 aligned comparison, 80.21 persistence, 80.4
independent validator, 80.22 renderer — all consumed read-only; **zero `src/`
changes**). Before any evidence generation it verifies the immutable inputs:
the manifest commitment digest, the pinned `openai-community/gpt2` commit
snapshot `e7da7f221d5bf496a48136c0cd264e630fe9fcc8`, the pinned
`Salesforce/wikitext` `wikitext-2-raw-v1` validation split at revision
`f776294184f13b8ff2337b3841cf9269a6216d1e` reproduced **byte-exactly** against
the pinned L04 manifest (official rows 3760, nonblank 2461, all 2048 selected
indices and text hashes exact), and a predeclared grouped leakage-safe split
(group = original split-local row index // 8, seed 79; train 1549 / eval 499
rows; groups disjoint, both classes in both partitions, `n_train ≥ 770`
capacity gate satisfied).

The declared target attribute was **predeclared before any evidence**: wikitext
raw section-header rows (`^ = .+ = $`; 516 positives / 1532 negatives), with
last-non-padding-token pooling per layer under native
`output_hidden_states` capture (hooks reserved for intervention, as the
manifest declares). Hidden states for all 2048 rows × 12 layers × 768 dims
replay **bit-identically** on a full re-extraction. The real seven-stage
workflow then completed end to end with real measurements only — the declared
positive defect (separability) was **observed**, every manifest control
behaved exactly as predeclared, localization named the manifest-expected layer
axis selection, the probe explanation was **supported**, the intervention was
**supported** with all four controls passing, and the aligned comparison
classified truthfully. Persistence then failed closed on a **frozen-contract
triangle** (below); the proof records the precise blocker and prints
`ACCEPTANCE: NOT PASSED`.

## Historical v1 blocker

The independent 80.4 validator requires every capture cited by a truthful
supported `separability_probe_leakage` claim to declare capture axes ⊇
`{sample, feature, label}` (frozen taxonomy
`representation_problem_taxonomy_v1.json`, `applicability.required_axes`), but:

1. the frozen transformer manifest (80.2) declares only
   `representation.axes = {layer, token, slice}` (`feature` is synthesized by
   the binder fallback; `sample` and `label` are **not declared**), and
2. the accepted 80.7 binder rejects any requested axis that is neither
   manifest-declared nor `feature`
   (`CaptureBindingError: provenance mismatch: requested axis … is not
   declared by manifest`), and its `SUPPORTED_AXES` contains **no `label`
   axis at all** — a `label` axis can never be bound or carried in real
   provenance.

Therefore the truthful report capture row (rebuilt by 80.21 registration from
the real bound capture) can carry at most `{slice, feature}`, the validator
raises `ReportValidationError: taxonomy applicability mismatch for
separability_probe_leakage`, and `persist_diagnostic_artifact` refuses with
`DiagnosticArtifactError: report failed independent validation; nothing was
persisted: taxonomy applicability mismatch for separability_probe_leakage` —
**with no files written**. Any way past this requires either relabeling the
bound `slice` axis as `sample`/`label` (a temporary role alias, explicitly
forbidden by this task), forging capture provenance (forbidden), or changing
one of the frozen 80.1/80.2/80.7 contracts (outside this task's authority;
80.1–80.22 are accepted and evidence-gated). Consequently the 80.21/80.22
lanes (content-addressed artifact, fresh-context revalidation, deterministic
rendered report, evidence-link re-hashing, artifact-level tamper refusal) are
**unreachable for this manifest**, and 80.24's frozen acceptance cannot pass.

## Historical v1 recorded results

```text
PASS frozen-inputs: manifest c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a validated; gpt2@e7da7f221d5b… snapshot present; wikitext f776294184f1… validation verified against the pinned L04 manifest (official 3760, nonblank 2461, selected 2048 indices+hashes exact); 6 frozen artifacts hashed
PASS leakage-safe grouping: grouped original-index//8 split seed79 -> train 1549 rows (grouped-original-index-div8-train79-1549rows) / eval 499 rows (grouped-original-index-div8-eval79-499rows); groups disjoint, both classes in both partitions, n_train>=770 capacity gate satisfied
RECORD declared attribute: is-section-header (pattern '^ = .+ = $') -> 516 positives / 1532 negatives (predeclared before evidence; never changed)
PASS deterministic-replay: pooled hidden states sha b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b reproduced bit-identically on a full re-extraction (12 layers x 2048 rows x768)
PASS capture: capture_identity 28f4fbd2c47231fec8908044d335d06a87b2fda2f18bde1e3f0567dcb8c05d89 shape (2048, 768) dtype float64 via bind_selection/resolve_captures (native output_hidden_states, last-nonpadding-token-pooling-v1)
RECORD positive-case (declared separability): OBSERVED — heldout-probe-accuracy 1.0000 ≥0.7, probe-leakage-gap 0.4489 ≥0.15; threshold_pass={'heldout-probe-accuracy': True, 'probe-leakage-gap': True}
RECORD controls: capacity passed (declared linear-logreg, n_params770<=n_train), label-randomization passed (randomized acc 0.5511), nonseparable-negative passed (noise acc 0.5832 <0.7), split-swap null recorded
RECORD headline probe: accuracy 1.0000 at the localized-layer probe (cached predictions shared by detect/compare/intervene — no stage refits)
RECORD per-layer probe accuracies h.0..h.11: [1.0, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; localize verdict 'localized', earliest 'transformer.h.0', affected layers ['transformer.h.0', 'transformer.h.1', 'transformer.h.2', 'transformer.h.3', 'transformer.h.4', 'transformer.h.5', 'transformer.h.6', 'transformer.h.7', 'transformer.h.8', 'transformer.h.9', 'transformer.h.10', 'transformer.h.11']
PASS determinism: detect payload sha 23863ec19820f0ee555da8dda345b7971e141036e01b24d0c8cf1c4edddcc2ce, localize payload sha 1129ac27a96fdbf79fff47e985779a7dcbd9c591e70319bedb0d2caceb00c2bc, capture identity stable (split digests train/eval identities fixed)
RECORD intervene: remove remove-section-header-direction target transformer.h.0 -> conclusion supported (baseline 1.0000, intervened 0.7315, effect -0.2685); controls {'remove-layer-off': 'passed', 'remove-layer-random': 'passed', 'remove-layer-shuffled': 'passed', 'remove-layer-zero': 'passed'}
RECORD compare: neither — heldout-probe-accuracy delta 0.000000, probe-leakage-gap delta 0.000000
BLOCKED persistence: the taxonomy/manifest/binder axes triangle makes a validator-clean artifact unreachable — DiagnosticArtifactError: report failed independent validation; nothing was persisted: taxonomy applicability mismatch for separability_probe_leakage
PASS fail-closed manifest tamper: BenchmarkManifestValidationError: manifest digest does not match canonical content
PASS fail-closed capture provenance: CaptureBindingError: provenance mismatch: selection representation_identity 'openai-community-gpt2:hidden-states:layers0-10:dim768' != manifest 'openai-community-gpt2:hidden-states:layers0-11:dim768'
PASS fail-closed leakage: DetectionError: train/eval split leaks: a sample position appears on both sides
PASS fail-closed resume identity: WorkflowError: checkpoint request_digest does not match current inputs
PASS fail-closed failed control: using the target itself as the non-separable negative -> outcome inconclusive, claim_allowed=False, missing ['failed-control:control-nonseparable-negative']
PASS frozen invariance: all six frozen artifacts byte-identical after the run
ACCEPTANCE: NOT PASSED — see recorded outcomes above; nothing was tuned to force a pass
  blocker: frozen-contract triangle blocks every validator-clean artifact for this family: the frozen taxonomy requires capture axes {sample, feature, label} for separability_probe_leakage, the frozen transformer manifest declares only {layer, token, slice}, and the accepted80.7 binder rejects undeclared axes and has no label axis at all (SUPPORTED_AXES), so the independent80.4 validator rejects any truthful supported claim — persist refused: report failed independent validation; nothing was persisted: taxo…
  blocker record: artifacts/task_80.24_transformer_hidden_state_proof_summary.md
resources: proof body 379.96s (frozen+data=0.00s, extraction+replay=355.49s, capture=6.22s, detect=0.77s, localize=0.13s, determinism=0.84s, workflow=2.39s, report-assembly=0.00s); wall including imports 383.16s; tracemalloc peak 535.3 MiB; environment python 3.13.3, numpy 2.4.6, transformers 4.57.6, torch-cpu, scikit-learn 1.9.0, datasets 3.6.0
commands: uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python scripts/sprint80_task80_24_proof.py; uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" pytest tests/test_transformer_end_to_end_proof.py -q
```

Workflow stage outcomes (verified in a follow-up execution of the same proof
composition): `capture/detect/localize/explain/intervene/compare/report` all
`completed`; explain evidence `{"claim_allowed": true, "method": "probe",
"outcome": "supported"}`; trial reason `material decrease effect
-0.26853707414829664 beyond tolerance 0.02 with identity, random, shuffled, and
off-target controls passing and uncertainty excluding the baseline`;
comparison `{'cmp-hiddenstate-eval-replay': 'neither'}`.

## Historical v1 hashes recorded

- Manifest commitment (canonical): `c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a`.
- Frozen artifact file SHA-256 (verified byte-identical after the run; six files):
  - `benchmark_manifest_schema_v1.json` — `de24e80e267484df99f880927885fef836f2d70e4bd750e4a7a61d0c979364fc`
  - `benchmark_manifest_sprint80_encoder_autoencoder_v1.json` — `e2935cd821a1605c719b63c748a9f93c2375cd6087c51739862d480a18c56959`
  - `benchmark_manifest_sprint80_transformer_hidden_state_v1.json` — `339c5c302def9d8b859e1914101803c84905347eec3caa1d596a5eadc015d0a2`
  - `diagnostic_report_schema_v1.json` — `a76580e58a672ac283fa0a313b34f6148375dc5181dcb395ae1061ce42928d07`
  - `representation_problem_taxonomy_v1.json` — `e1742b7833e587708133b9d927e39636df93b64b3e348b759353e1be46373c3b`
  - `m14/l04-wikitext-2-manifest.json` — `0908f843efd72ce93c628e34cdb27f56e37e764d196ef91be7eff6d7757b78f3`
- Pinned selection digests re-verified: L04 `content_sha256`
  `bd235bad5a7643c860bca04a98ba545214f25702cd7625dd4ff591f0ea32cf7b`,
  `split_sha256` `bb2dab8721bb8e244bf38f9add6af9e5c2fc70291ce4de6cf2263f7e0f970703`.
- Pooled hidden states (12×2048×768 float64, canonical bytes):
  `b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b`
  (reproduced bit-identically on full re-extraction).
- Real capture identity: `28f4fbd2c47231fec8908044d335d06a87b2fda2f18bde1e3f0567dcb8c05d89`.
- Detect payload SHA-256 (byte-deterministic): `23863ec19820f0ee555da8dda345b7971e141036e01b24d0c8cf1c4edddcc2ce`.
- Localize payload SHA-256 (byte-deterministic): `1129ac27a96fdbf79fff47e985779a7dcbd9c591e70319bedb0d2caceb00c2bc`.

## Historical v1 output locations

**No diagnostic artifact or rendered report exists.** The declared output
location `artifacts/diagnostics/proof-80-24` was intentionally not
materialized (persistence refused before any write; the proof cleans up any
partial output). Evidence lives in this artifact, the proof output quoted
above, and the focused tests (which assert the refusal plus
`not (root / "runs").exists()`).

## Historical v1 files modified

* [scripts/sprint80_task80_24_proof.py](../scripts/sprint80_task80_24_proof.py) - The revision-backed executable core proof: immutable-input verification, grouped leakage-safe split, real seven-stage workflow, truthful persistence blocker for the original v1 contract, and bounded execution evidence.
* Historical focused test source `tests/test_transformer_end_to_end_proof.py` — prior worktree results are reported below; the test source is not included in this evidence-only publication.
* [docs/sprint-plans/sprint-80.md](../docs/sprint-plans/sprint-80.md) - Explicit blocker and amended acceptance recorded in the sprint plan.
* No `src/` file was modified (zero library changes; all accepted seams behave as frozen).

The focused test source above remains historical worktree evidence and is intentionally excluded from this documentation/evidence publication. Its recorded result is not represented as a clean-clone test run; the accepted contract and measurement lineage are independently recorded in the revision-backed task handoff and 80.25 replay.

## Historical v1 testing

* **Test File (historical worktree source, not included in this publication):** `tests/test_transformer_end_to_end_proof.py`
* **Status:** Passed
* **Execution Command:** `uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" pytest tests/test_transformer_end_to_end_proof.py -q`
* **Result:** `6 passed in 24.37s`

Proof (exact command, exit 0 on truthful execution with
`ACCEPTANCE: NOT PASSED` + the axes-triangle blocker printed):

```text
uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python scripts/sprint80_task80_24_proof.py
(exit code0; full output quoted above)
```

Focused regression battery over every consumed seam (no formatters, linters,
or project-wide suites were run, per task constraints):

```text
uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" pytest tests/test_transformer_end_to_end_proof.py tests/test_redundancy_separability_detection.py tests/test_layer_slice_localization.py tests/test_capture_binding.py tests/test_diagnostic_workflow.py tests/test_probe_tcav_ig_explanation.py tests/test_intervention_trials.py tests/test_run_comparison.py tests/test_sprint80_core_manifests.py -q
97 passed in50.33s
```

Runtime/resources (bounded, far under the permanent 16 GiB ceiling): proof
body 379.96s (extraction+replay 355.49s, capture 6.22s, detect 0.77s,
localize 0.13s, determinism 0.84s, workflow 2.39s), wall including imports
383.16s, tracemalloc peak 535.3 MiB, CPU-only torch, python 3.13.3, numpy
2.4.6, transformers 4.57.6, scikit-learn 1.9.0, datasets 3.6.0.

## Historical v1 graph refresh record

First graph refresh after all graph-visible writes (proof script, tests):

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction:100/253 uncached files (39%) [8 workers]
  AST extraction:200/253 uncached files (79%) [8 workers]
  AST extraction:253/253 uncached files (100%) [8 workers]
  warning:248 source file(s) produced zero nodes and are absent from the graph (...).
[graphify watch] community set changed since labeling (1074 saved labels,1080 communities now; renamed197 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) ->2026-09-22/
Graph has15589 nodes (above5000 limit). Building aggregated community view...
graph.html written (aggregated:1080 community nodes,1481 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt:15589 nodes,35299 edges,1080 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Wall time:64.15 seconds
```

A final `graphify update .` is run after the final source, test, guide, and evidence-artifact updates; the current result is recorded in the final rule-bound addendum below. Its graph-result note is appended after the refresh, so this documentation-only addition is not indexed by that run; code graph currency and this limitation are stated with the result.

## Negative results, limitations, and resolution paths

* **Blocker (precise):** the axes triangle — taxonomy required axes
  `{sample, feature, label}` vs frozen manifest axes `{layer, token, slice}`
  vs binder `SUPPORTED_AXES` without `label` and without undeclared-axis
  acceptance — makes every validator-clean artifact for
  `separability_probe_leakage` unreachable. Resolution requires an
  evidence-review decision touching a frozen contract (extend the taxonomy's
  required axes, re-declare the manifest axes, or extend the 80.7 binder with
  its accepted tests) — explicitly outside 80.24's authority.
* The scientific claim itself passed honestly: positive observed at the frozen
  thresholds, all four manifest controls as declared, localization at
  `transformer.h.0` exactly per the manifest's expected localization,
  supported probe explanation, supported removal intervention with four
  passing controls, truthful `neither` comparison. These outcomes are recorded
  here and asserted by the focused tests but are **not** promoted to a passed
  acceptance because persistence/rendering/re-hash evidence does not exist.
* Attribute confound (declared limitation in the assembled report as well):
  section-header rows correlate with row length and last-token identity; the
  claim is separability under the frozen protocol, not causal feature use.
  Per-layer accuracy ≈ 1.0 makes the probe task easy but is the truthful
  measurement at these layers; no threshold or attribute was adjusted.
* The compare stage's task role carries the second manifest-declared metric
  (`probe-leakage-gap`) because the frozen manifest declares no other task
  metric; both aligned evaluation replays of the pinned revision are identical
  (`neither`, |delta| = 0), which is the truthful reproducibility result.
* Token and time localization axes are explicit non-applicable (no time axis
  in the manifest; no token-level claim); token scope is bound by the capture
  identity and split identity instead. Per-sample localization rows for all
  499 heldout rows live in the localize stage payload (recorded limitation),
  with the report listing only layer/slice rows for concision.
* Pooled features are cached content-addressed in the OS temp directory;
  a cold run re-extracts (~6 minutes CPU) and needs the pinned dataset
  (network or HF cache) plus the local gpt2 snapshot — both available here.
* At this earlier stage only the proof and focused regressions were run (no formatters, linters, or project-wide suites). The graph refreshes recorded in this historical section were superseded by the final post-edit update documented below. Main's evidence review remains pending.

## Reconciliation update — 2026-09-23

The task is reopened at `[~]` in the sprint plan. This update supplements,
but does not rewrite, the original frozen-input proof and fail-closed result.
The required current handoff is [`artifacts/sprint-80/task-24.md`](sprint-80/task-24.md).

A fresh execution of the frozen proof command completed the same truthful
scientific chain. Manifest commitment `c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a`
validated; GPT-2 and the Wikitext validation selection were present and
byte-exact against L04 (3,760 official rows, 2,461 nonblank, 2,048 selected).
The grouped original-index//8 split at seed 79 again yielded 1,549 train and
499 evaluation rows. The predeclared section-header target remained 516/1,532.
Full hidden-state replay remained bit-identical with SHA-256
`b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b`; the
real bound capture remained `(2048, 768)`, identity
`28f4fbd2c47231fec8908044d335d06a87b2fda2f18bde1e3f0567dcb8c05d89`, axes
`{slice, feature}`.

The fresh measurements reproduced: held-out accuracy `1.0000` and leakage
gap `0.4489`; all four declared manifest controls passed/recorded; earliest
localized layer `transformer.h.0`; supported probe explanation; supported
intervention effect `-0.2685` with zero-strength, random, shuffled, and
off-target controls passing; and comparison `neither` with both metric deltas
`0.000000`. A separate focused `build_workflow()` execution confirmed all
seven stage outputs (`capture`, `detect`, `localize`, `explain`, `intervene`,
`compare`, `report`) were `completed`, with explanation evidence
`supported` and `claim_allowed: true`.

Persistence again stopped before writes with
`DiagnosticArtifactError: report failed independent validation; nothing was
persisted: taxonomy applicability mismatch for separability_probe_leakage`.
Thus this run produced no content-addressed artifact or rendered report; no
fresh-context reload, independent artifact revalidation, report rendering,
or report/evidence re-hash is claimed. The script exited 0 to record its
truthful result, but printed `ACCEPTANCE: NOT PASSED`. All six frozen
artifacts were byte-identical after the run. Manifest tamper, capture
provenance mismatch, train/eval leakage, resume identity mismatch, and failed
control cases continued to reject/block as declared.

Exact fresh verification:

```text
uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python scripts/sprint80_task80_24_proof.py
exit 0; ACCEPTANCE: NOT PASSED
proof body 498.91s; wall including imports 505.22s; extraction+replay 465.64s;
capture 8.75s; detect 1.80s; localize 0.21s; determinism 1.65s;
workflow 3.19s; peak traced memory 542.4 MiB;
Python 3.13.3, NumPy 2.4.6, Transformers 4.57.6, CPU PyTorch,
scikit-learn 1.9.0, datasets 3.6.0

uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" pytest tests/test_transformer_end_to_end_proof.py -q
6 passed in 66.60s

uv run pytest -q tests/test_capture_binding.py tests/test_diagnostic_validator.py
22 passed in 31.90s

uv run pytest -q tests/test_report_renderer.py tests/test_intervention_trials.py::test_comparison_only_task_metric_can_drive_a_causal_intervention
16 passed in 30.13s
```

Focused stage-output command and result:

```text
uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python -c "import scripts.sprint80_task80_24_proof as p; w, r = p.build_workflow(); result, checkpoint = w.run(r, p.load_manifest()); print('workflow:', result.status, checkpoint.status, list(checkpoint.completed_stages)); print('stage outcomes:', {stage.stage: stage.outcome for stage in checkpoint.outputs}); print('explanation evidence:', checkpoint.outputs[3].payload['family_evidence'])"
workflow: completed completed ['capture', 'detect', 'localize', 'explain', 'intervene', 'compare', 'report']
stage outcomes: {'capture': 'completed', 'detect': 'completed', 'localize': 'completed', 'explain': 'completed', 'intervene': 'completed', 'compare': 'completed', 'report': 'completed'}
explanation evidence: {'h-section-header-separability': {'outcome': 'supported', 'claim_allowed': True, 'method': 'probe'}}
```

## Graph refresh — 2026-09-23

`graphify update .` completed with exit 0 after the proof/test documentation
updates; no LLM was used. It processed 284 uncached code files and rebuilt
the graph to 16,186 nodes, 37,386 edges, and 1,107 communities. Aggregated
`graph.html` contains 1,107 community nodes and 1,681 cross-community edges.
Six semantic/curated graph files were backed up under
`graphify-out/2026-09-23/`. Graphify warned that 274 source files produced
zero nodes and that saved labels were stale (1,081 saved labels versus 1,107
communities; 212 communities renamed); `graphify label` was suggested but
not run. The output stated that the code graph was updated and that document
changes require the assistant `/graphify --update` path.

The minimum truthful general-purpose resolution is a versioned shared contract
that distinguishes model-capture axes from a separately content-addressed,
sample/split-aligned target-label evidence record and makes the independent
validator enforce both. That requires explicit versioned taxonomy,
validator, and report-evidence semantics; the frozen v1 artifacts and this
v1 result must remain unchanged. A new transformer manifest/report contract
could instead predeclare target identity and separate target provenance, but
that would be a prospective case and cannot silently satisfy acceptance for
the frozen v1 manifest. No contract change was authorized in this task, so
the blocker remains unresolved and the plan status remains `[~]`.

The proof composition predeclares the concrete section-header rule
(`HEADER_PATTERN`), while manifest v1 describes only a generic “declared
target attribute” and has no separate target-label identity field. A
versioned correction must bind the rule and label-to-sample alignment as
target evidence, not infer a `label` activation axis.

This reconciliation updated the proof/test module docstrings, task artifacts,
and graphify code-graph outputs; no library behavior, frozen input, historical
outcome, 80.23 generic contract, or sprint checkbox was changed.

## Historical rule-bound v2 proof — 2026-09-24 (original scope only)

### Contract and evidence chronology

The frozen-v1 diagnosis remains a truthful historical negative: its taxonomy requires an activation `label` axis that the frozen manifest and capture binder cannot truthfully provide, so the independent validator rejected persistence and the v1 run recorded `ACCEPTANCE: NOT PASSED`. That result, the frozen taxonomy/manifest, and all six frozen inputs remain unchanged.

An earlier generic v2 proof was a successful **preliminary** run at `artifacts/diagnostics/proof-80-24-v2/` (run `6e65beddb1dfeadf`; artifact SHA-256 `14e60dfb1d06843af92670e0bd4f789664d4039589da7fa8a199380205179b1d`; report SHA-256 `c670acea7b27665581844916f17fb03aa38b5aa6d7740583dea34a5a6f467bdb`; rendered-report SHA-256 `e6cac432eca469880314a6d44ff69f23779f8e84c020e00cf96b9ac305ac2716`). It had an earlier 23-test focused battery. This root is deliberately retained and is not the final rule-bound evidence.

The current contract, `diagnostic-evidence-v2`, keeps truthful activation axes (`slice`, `feature`) separate from a content-addressed target record binding the declared target rule, labels, sample IDs, exhaustive train/evaluation membership, split identities, dataset split, representation, and detector capacity. The independent validator reconciles that target record with detect-stage provenance and the registered capture artifact. The final follow-up verified the target rule in detect provenance; it did not add or alias a `label` capture axis.

### Final seven-stage proof and persisted output

The immutable `diagnostic-evidence-v2` run completed the seven-stage workflow; fresh validation, deterministic rendering, and evidence-link checks passed for its original contract. End-of-sprint F2 later found the run's `coef_stability` observation (approximately `1.0`) had `threshold: null`, so the v2 validator pass does not establish a gated stability result. Keep this run as historical evidence for its original scope only; the prospective correction is recorded below.

- Artifact SHA-256: `e2ebd582ac9fa0defe03f1b221290067d931a9b99b606b661da75fccab7eee7a`.
- Diagnostic report SHA-256: `c670acea7b27665581844916f17fb03aa38b5aa6d7740583dea34a5a6f467bdb`.
- Rendered report: [`diagnostic-report`](diagnostics/proof-80-24-v2-rule-bound/diagnostic-report), SHA-256 `788b9a316ff3ec14922b7a1515fd8aee2b0d2b4b4d0dc884f321eb0e68b8693a`; it records validator `passed` and contract `diagnostic-evidence-v2`.
- Native hidden-state replay remained bit-identical (pooled-state SHA-256 `b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b`; capture identity `28f4fbd2c47231fec8908044d335d06a87b2fda2f18bde1e3f0567dcb8c05d89`).

### Target rule, alignment, and scientific outcomes

- Target reference `target-section-header-attribute-record`, SHA-256 `3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19`.
- Target `section-header-attribute`; predeclared rule `section-header attribute '^ = .+ = $'`, SHA-256 `9b12485635decf738c4904702f75d7d5ae8cb6663555e2af5ee66476269f025b`.
- Bound 2,048 target labels and sample identities: label SHA-256 `196284d27fbeca8c3d4ffe2e17310deacc3e2aa07b558c9fe743a31529974bee`; sample-ID SHA-256 `23fd5605b8c632e6b7823ab79a9d36061947fe5c916ea50bd5d10310e07ce181`.
- Exhaustive grouped split: 1,549 train (`grouped-original-index-div8-train79-1549rows`) and 499 eval (`grouped-original-index-div8-eval79-499rows`); full split identities/memberships are in the content-addressed record.
- Frozen manifest commitment `c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a`; detect payload SHA-256 `663e80b0f655981a6e304f4e4de18c1888986981bf4d59a981bfa8176d65c87b`; localization payload SHA-256 `1129ac27a96fdbf79fff47e985779a7dcbd9c591e70319bedb0d2caceb00c2bc`.
- All seven stages completed. Detection: held-out accuracy `1.0000` and leakage gap `0.4489`; capacity, label-randomization, nonseparable-negative, and split-swap controls passed. Localization and probe explanation were supported at `transformer.h.0`. Removal intervention there was supported (accuracy `1.0000 → 0.7315`; effect `-0.2685`) with zero-strength, random, shuffled, and off-target controls passing. Aligned replay comparison was `neither` with both metric deltas `0.000000`.
- Limit: the frozen section-header target is separable under this protocol, not evidence of causal feature use; section headers correlate with row length and last-token identity. The report keeps token/time localization non-applicability explicit.

### Final verification and 80.23 compatibility

- Final proof command: `uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python scripts/sprint80_task80_24_proof.py` — **v2 contract acceptance passed; no gated `coef_stability` threshold was recorded**. Seven stages, fresh independent validation, deterministic report, 13 content-addressed link re-hashes, and unchanged frozen inputs.
- `uv run pytest -q tests/test_target_evidence.py tests/test_diagnostic_validator.py tests/test_redundancy_separability_detection.py tests/test_transformer_end_to_end_proof.py` — **37 passed in 44.40s**. Coverage includes target/rule/sample/split tamper and misalignment rejection, forged-axis rejection, and legacy v1 fail-closed/no-write behavior.
- Accepted 80.23 v3 run `6bca9afb72af32c4` reloaded read-only through current `load_diagnostic_artifact` (artifact SHA-256 `e5c6db0ab5080807138e091bc332a5d3e788889a382917e93775b15f562aef0f`). Two fresh renders matched each other and retained `encoder-diagnosis-v3.md` (5,690 bytes; SHA-256 `b8d713de3181ef39990d994d4f3c708e584eb3c559f762376e2b9f2de65de6e3`); the accepted 80.23 artifact was not changed.
- No project-wide test suite, lint, formatter, or build was run. The evidence review remains pending; task 80.24 remains `[~]`.

### Prior graph refresh (before the stability correction)

`graphify update .` for the earlier v2 handoff processed 287 uncached code files; that graph had 16,245 nodes, 37,536 edges, and 1,111 communities. It is retained as historical chronology and superseded by the stability-correction refresh below.

- Warnings: 276 source files produced zero nodes; saved community labels were stale (1,085 saved labels versus 1,111 communities; 204 renamed). Graphify suggested `graphify label`, which was not run.
- Graphify's output says document/paper/image changes require the AI-assistant `/graphify --update` path. That path was not run, so this records a refreshed code graph, not semantic extraction of the edited Markdown artifacts.
- The statements above describe that earlier code-graph update only. The stability-correction refresh below includes the new source/test changes; its Markdown handoff/doc edits are not semantically indexed by the code-only graph command.

### Current changed files

- Library: `src/latent_anything/_target_evidence.py`, `_redundancy_separability_detection.py`, `_diagnostic_report.py`, `_diagnostic_validator.py`, `_diagnostic_artifact.py`, `_report_renderer.py`.
- Proof/tests/docs: `scripts/sprint80_task80_24_proof.py`, `tests/test_target_evidence.py`, `tests/test_transformer_end_to_end_proof.py`, `docs/AI_ENGINEER_GUIDE.md`.
- Records: `CHANGELOG.md`, `.agents/memory/decisions.md`, `.agents/memory/lessons-learned.md`, `artifacts/sprint-80/task-24.md`, and this summary.
- Graph outputs: `graphify-out/graph.json`, `graphify-out/graph.html`, `graphify-out/GRAPH_REPORT.md`, `graphify-out/manifest.json`, `graphify-out/.graphify_labels.json`, and its signature; six prior semantic/curated files were backed up under `graphify-out/2026-09-24/`.

## Prospective transformer stability correction — 2026-09-24

### F2 correction and protocol provenance

Deep-review F2 identified that the immutable rule-bound v2 run `421578bb1c06fa15` recorded `coef_stability≈1.0` with `threshold:null`; that run is not stability-gated even though the v2 artifact and report passed their original validation contract. Its artifact, report, rendered report, target record, run record, and the frozen v1 manifest/taxonomy remain unchanged.

The supplemental protocol `artifacts/benchmark_manifest_sprint80_transformer_explanation_stability_v1.json` declares `coef_stability >= 0.8`, fidelity `heldout_accuracy >= 0.7`, and selectivity `leakage_gap >= 0.15`. Protocol ID: `sprint80-core-transformer-explanation-stability-v1`; final provenance-complete SHA-256: `a69b7152a800ac543ac76fec4d699c16681517708a95b353e5d241241c8acf0a`. The stability threshold was borrowed from accepted 80.23 encoder-v3 run `6bca9afb72af32c4`, artifact SHA-256 `e5c6db0ab5080807138e091bc332a5d3e788889a382917e93775b15f562aef0f`, accepted proof-record SHA-256 `deb38278b6442db366d09a2b8b5c0271eada8162f3913efd65f37648b47ba7f1`; source proof SHA-256 `87891b0c75d4ddcfe890792e421ac4108801d08aa49a9d3a234da764481cb53f`.

Before the first replay result was inspected, the only protocol edits were a correction to the precedent manifest ID and addition of content-addressed precedent file/run hashes; thresholds, model/data identity, grouped split algorithm/seed, and probe seeds did not change. The pre-edit bytes were reconstructed and hash-verified against the initial frozen protocol digest `4c0b58ab20a78156459a9c2a03fd55af4d41d35ed69a12c9522e26f3ab876c60`. The final run records the corrected protocol hash above.

### Prospective measurements

- Real `openai-community/gpt2` commit `e7da7f221d5bf496a48136c0cd264e630fe9fcc8` and WikiText revision `f776294184f13b8ff2337b3841cf9269a6216d1e` were replayed on the same selected 2,048 validation rows; full extraction matched cached pooled features bit-for-bit (SHA-256 `b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b`). This is a new grouped split over the same corpus, not an independent source corpus.
- Grouped split seed `83`: 1,538 train rows, 510 heldout rows, zero sample/group overlap. Ordered 2,048 sample IDs and labels were compared directly with the immutable v2 target record and matched exactly. The new record's identity/label digests are `156c6517670fd4298324056e3581dc5064092287bfdbaf2fb9bf51182920333a` and `d3eb6ab1052f09a26a927ded5f75232e069546e9faa9c973372a2741b4df0e5b`.
- Results: held-out accuracy `1.000000 >= 0.700`; sign-aware minimum coefficient cosine `0.974878 >= 0.800`; leakage gap `0.425490 >= 0.150`. Capacity, randomized-label, and nonseparable-negative controls passed; negative-control accuracy `0.619608`.
- Five grouped subset refits: sampling/fit seed `109/193`, cosine `0.978693` (1,243 rows); `113/197`, `0.975197` (1,228); `127/199`, `0.974878` (1,211); `131/211`, `0.976767` (1,238); `137/223`, `0.988987` (1,223).
- Shared explanation payload now includes its declared `target_id`. The run uses real capture axes `slice` and `feature`; sample IDs and labels remain separate target provenance. No causal feature-use inference is claimed; row length and last-token identity remain confounds.

### Content-addressed result and verification

- Independently validated run record: `artifacts/diagnostics/proof-80-24-stability-v1/runs/a95ccce32461b78ddd0197188fc1f4b51a1b349d434d3a9812dbcc8251dbeb2d.json`, SHA-256 `a95ccce32461b78ddd0197188fc1f4b51a1b349d434d3a9812dbcc8251dbeb2d`.
- Deterministic report: `artifacts/diagnostics/proof-80-24-stability-v1/reports/5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0.md`, SHA-256 `5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0`. Independent validation, report regeneration, the real replay, all three gates, controls, and old-v2 immutability passed.
- Initial candidate files remain content-addressed under the same root: run `593545a20b136ede1400786c15d4665c444591d50f86569a00f0da566777812a.json` and report `572588e3de89a3976c31f3a580cff3b24f8a3366be42415a01a877d609b6cb46.md`. The candidate was not accepted: validator rejection exposed that the generic `ExplanationHypothesis.to_dict()` omitted `target_id`. The serializer was fixed and covered by a regression test; the corrected replay reproduced the same metrics and passed. This is a validation rerun of the predeclared seed-83 outcome, not a second independent corpus/split; no threshold shopping occurred.
- `uv run --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python scripts/sprint80_task80_24_stability_proof.py` — `ACCEPTANCE: PASSED`; runtime `632.97s`, explanation `1.64s`, Python `3.13.3`.
- `uv run pytest -q tests/test_probe_tcav_ig_explanation.py` — 25 passed in `17.35s`; includes fail-able stability behavior, heldout identity leakage rejection, and target-identity serialization.
- `uv run pytest -q tests/test_sae_lens_geometry_density_clustering.py::test_each_method_executes_only_for_declared_hypothesis` — 1 passed in `17.35s`.
- Focused `py_compile` over the supplemental proof, shared explanation module, and its regression test exited 0. Protocol provenance preflight passed and verified the historical v2 threshold remains null.
- The broader `uv run pytest -q tests/test_sae_lens_geometry_density_clustering.py` attempt timed out at 300 seconds after 12 progress dots without a summary; it is not claimed as passing. No project-wide tests, lint, formatter, or build were run.

### Latest graph refresh

The stability-correction graph refresh is recorded after the final code/docs updates below. It supersedes the earlier graph counts above. The task remains `[~]` pending Main's evidence review; no plan checkbox was changed.

### Files changed or generated for the stability correction

- `src/latent_anything/_probe_tcav_ig_explanation.py` — retain target identity in shared hypothesis serialization; `tests/test_probe_tcav_ig_explanation.py` — serialization and fail-able stability behavior.
- `scripts/sprint80_task80_24_stability_proof.py` and `artifacts/benchmark_manifest_sprint80_transformer_explanation_stability_v1.json` — prospective frozen proof and validator.
- `artifacts/diagnostics/proof-80-24-stability-v1/runs/` and `reports/` — content-addressed final record/report and retained rejected candidate.
- `docs/SPRINT_80_DEPTH_EVIDENCE.md`, `docs/AI_ENGINEER_GUIDE.md`, `CHANGELOG.md`, `.agents/memory/lessons-learned.md`, and the Sprint 80 task handoff.

### Stability-correction graph refresh

`graphify update .` completed after the stability-correction source, tests, proof/protocol, and handoff edits. Graphify ran code-only AST extraction without LLM over 310 uncached code files. The current graph has 16,369 nodes, 37,712 edges, and 1,098 communities; aggregated `graph.html` has 1,098 community nodes and 1,707 cross-community edges. Six semantic/curated files were backed up under `graphify-out/2026-09-24/`.

- Warnings: 295 source files produced zero nodes; saved labels were stale (1,118 saved labels versus 1,098 communities; 217 communities renamed). `graphify label` was suggested but not run.
- Graphify's output requires the assistant `/graphify --update` path for document/paper/image semantics. It was not run; the new docs and handoffs are not semantically indexed.
- The code graph was refreshed for the corrected explanation serializer and all stability-proof code/tests. The current task remains `[~]` until Main completes evidence review.
