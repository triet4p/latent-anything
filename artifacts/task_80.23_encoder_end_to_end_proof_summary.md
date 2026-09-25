# Task Summary: 80.23 — frozen v1/v2 evidence preserved; prospective v3 proof passed (evidence review pending)

**Sprint:** Sprint 80 — Stable Depth
**Task:** 80.23
**Historical v1 disposition:** **PASS_BLOCKED — NOT checked.** Its frozen
positive was not observed; this v3 prospective replacement does not rewrite or
promote the v1 outcome. Historical v2 remains inconclusive and capture-level.
The user-amended v3 evidence below is a separate prospective proof; current
plan status remains `[~]` pending Main's evidence gate, not `[x]`.

## Summary of Work

Added the committed executable proof `scripts/sprint80_task80_23_proof.py`,
which executes the frozen encoder manifest (`sprint80-core-encoder-autoencoder-collapse-v1`,
commitment digest `b4c49c890cf34cc0530d499c2b82a09057f1f1ebf5b9d9a3843200760b137dd3`)
through the single high-level `DiagnosticRequest`/`DiagnosticWorkflow` using the
existing architecture-neutral seams only (capture binding, 80.9 detect, 80.13
localize, 80.16 explain, 80.18 intervene, 80.20 compare, 80.21 persistence, all
consumed read-only): zero `src/` files were changed and no encoder-only
orchestration/reporting branch exists anywhere in the library — composition
lives in the proof script, the same pattern the accepted 80.18–80.22 smokes use.
The proof runs the real ConvVAE (`latent4/seed0/epochs5`, refit bit-identical at
the declared seeds) on the pinned `default_rng(42)` 1437/360 digits split
(index digests match the pinned benchmark artifact), binds a real capture
identity over the real heldout bottleneck-mu array through
`bind_selection`/`resolve_captures`, and executes every manifest-declared
control case with real data: the healthy full-rank counterexample (PCA-4 of the
same heldout pixels) scores above both frozen thresholds, the benign
low-variance negative (the real bottleneck at 1e-3 global gain, min-variance
2.168e-09) does not trigger, and the null-shuffle is recorded.

The declared known positive defect, however, is **not observed**: on the
declared capture both frozen metrics pass
(`bottleneck-effective-rank` 3.1149 ≥ 3.0, `bottleneck-singular-spread`
0.4735 ≥ 0.25; bootstrap CI for the rank [2.9134, 3.2664]), and every other
legitimate mu capture of the pinned artifacts passes as well (train
3.0126/0.4644, all-digits 3.0346/0.4662, untrained-init 3.1469/0.4777).
Truthful localization is therefore negative — no frozen rule marks any
direction — and the manifest's expected feature-axis selection
("bottleneck-mu direction with the smallest heldout variance", dim2 by the
frozen estimator) cannot be named without fabricating a location. The 80.16
contract requires every hypothesis to bind a localized/supported prior
selection (empty bindings and unresolvable bindings both raise
`StageContractError`), so the seven-stage workflow **fails closed at the
explain stage**: no completed workflow exists, therefore no validator-clean
content-addressed artifact and no rendered report can be produced without
fabrication. The proof script records all of this, exits 0 on the truthful
execution, and prints `ACCEPTANCE: NOT PASSED` with the blockers.

## The blocker (three independent, recorded reasons)

1. **The declared positive case does not exist on the pinned inputs.** The
   manifest asserts a known defect ("rank-deficient or near-constant
   bottleneck directions"), but under the frozen estimator
   (`compute_latent_health` participation ratio; centered-batch
   min-singular/max-singular spread — the accepted 80.9 implementation) the
   pinned ConvVAE bottleneck is structurally healthy on every legitimate
   capture. The manifest's own bootstrap CI for the rank straddles the
   threshold ([2.9134, 3.2664] around 3.0), but the frozen detector gates on
   the point estimate (accepted 80.9 semantics), which passes. 80.5 already
   pre-registered this failure mode: "Threshold values are predeclared
   judgment choices … 80.23/80.24 must execute them as written or fail
   explicitly."
2. **The80.16 localization binding makes a truthful negative localization
   fatal to the chain.** After a negative localize verdict the hypothesis can
   declare neither a resolvable binding
   (`declares localization ('slice', 'slice-all') with no localized/supported
   prior slice selection carrying that identity`) nor empty bindings
   (`declares no localization bindings but the run carries a prior localize
   payload`); both were demonstrated empirically. The workflow result is
   `failed` at `explain` (`StageContractError`), and 80.21 persistence
   refuses the incomplete workflow with no files written
   (`diagnostic result must be completed`).
3. **No localization seam can express the manifest's expected feature-axis
   direction selection under the frozen effective-rank metric.** The encoder
   manifest declares no layer axis and the axial seam (80.14) supports only
   checkpoint/token/time, so only the layer/slice seam is available; but
   per-feature cells must carry `bottleneck-effective-rank` observations
   against the ≥ 3.0 threshold, and a single direction carries rank ≤ 1, any
   ≤ 3-dim subspace rank ≤ 3.0 (inside the ambiguity band), while
   full-batch-derived quantities do not vary per feature. The in-script
   arithmetic records the concrete failure: leave-one-feature-out ranks
   [2.6354, 2.4429, 2.6009, 2.2737] all sit below the 2.9 affected boundary,
   so every direction would be flagged and the earliest-in-order selection
   would be `bottleneck-mu-dim0` ≠ the manifest-expected `bottleneck-mu-dim2`.
   This architectural gap is independent of reason 1 — even an injected
   defect could not name the expected selection through the existing seams.

## Actual recorded results (exact proof output)

```text
PASS frozen-inputs: manifest b4c49c890cf34cc0530d499c2b82a09057f1f1ebf5b9d9a3843200760b137dd3 validated; scikit-learn 1.9.0; 5 frozen artifacts hashed
PASS split: train1437/heldout360 disjoint; index digests 0b03643758a0/48e9e0ed2c89 match the pinned benchmark
PASS deterministic-replay: refit (latent4/seed0/epochs5) re-encodes heldout bit-identically; heldout latents sha a9bb7895b9a0db1fd2c2791f79da95d7cb0ac099c0ac0cf56422d1ae2e80c31a
PASS capture: capture_identity 16a5c11ae062d83ec489635f05c68df0b3c19acbd0bad65a8a11da3021bc2acd shape (360, 4) dtype float64 device cpu via bind_selection/resolve_captures
RECORD positive-case (declared known defect): NOT OBSERVED — bottleneck-effective-rank 3.1149 ≥3.0, bottleneck-singular-spread 0.4735 ≥0.25; threshold_pass={'bottleneck-effective-rank': True, 'bottleneck-singular-spread': True}; bootstrap CI bottleneck-effective-rank [2.9134, 3.2664]
RECORD counterexample: control-healthy-counterexample passed — ER 3.8490 ≥3.0, spread 0.7491 ≥0.25 (PCA-4 of the same heldout pixels)
RECORD negative: control-benign-low-variance passed, not triggered — ER 3.1149, spread 0.4735, min-variance 2.168e-09 (real bottleneck ×1e-3)
RECORD null: control-null-shuffle recorded — shuffled ER 3.7654, spread 0.6929
RECORD capture-variant diligence (trained-mu-on-train): ER 3.0126, spread 0.4644 — passes both frozen thresholds
RECORD capture-variant diligence (trained-mu-on-all-digits): ER 3.0346, spread 0.4662 — passes both frozen thresholds
RECORD capture-variant diligence (untrained-init-mu-on-heldout): ER 3.1469, spread 0.4777 — passes both frozen thresholds
RECORD feature-axis localization arithmetic: leave-one-feature-out effective ranks [2.6354, 2.4429, 2.6009, 2.2737] — all below the2.9 affected boundary, so every direction would be flagged and the earliest-in-order selection (bottleneck-mu-dim0) != the manifest-expected smallest-variance direction (bottleneck-mu-dim2); a single direction carries rank ≤1, so no per-feature effective-rank observation ≥3.0 exists to separate it
RECORD localize: truthful negative — no layer or sample meets the declared affected criterion (no layer or sample meets the declared affected criterion; benign negative produces no location); the manifest's expected feature-axis selection cannot be named without fabricating a location
PASS determinism: detect payload sha faeff74e9ad7aa9185814ef94cbfa8a21afcda72e7fd8cb97709aa18f83bfc8a, localize payload sha 35c17b76f805453b36087776849c2a9131d92dce4a4f384bcfe388a9341ac8df, capture identity stable across reruns
BLOCKED explain: StageContractError — explain hypothesis 'h-collapse-bottleneck-mu' declares localization ('slice', 'slice-all') with no localized/supported prior slice selection carrying that identity; workflow failed at explain (7-stage chain cannot complete, no validator-clean artifact or rendered report is producible without fabricating a localization)
PASS fail-closed resume identity: WorkflowError: checkpoint request_digest does not match current inputs
PASS fail-closed persistence: DiagnosticArtifactError: diagnostic result must be completed
PASS fail-closed manifest tamper: BenchmarkManifestValidationError: manifest digest does not match canonical content
PASS fail-closed capture provenance: CaptureBindingError: provenance mismatch: selection representation_identity 'conv_vae_8x8:bottleneck-mu:latent_dim=99' != manifest 'conv_vae_8x8:bottleneck-mu:latent_dim=4'
PASS fail-closed failed control: real rank-2 counterexample fails the frozen threshold → outcome inconclusive, claim_allowed=False, missing ['failed-control:control-healthy-counterexample']
RECORD ablation dim0: bottleneck-effective-rank 2.6354 (delta -0.4795), heldout reconstruction mse 0.177033 (delta +0.000334), heldout variance 3.259790e-03
RECORD ablation dim1: bottleneck-effective-rank 2.4429 (delta -0.6720), heldout reconstruction mse 0.176497 (delta -0.000202), heldout variance 3.788530e-03
RECORD ablation dim2: bottleneck-effective-rank 2.6009 (delta -0.5140), heldout reconstruction mse 0.176671 (delta -0.000028), heldout variance 2.167675e-03
RECORD ablation dim3: bottleneck-effective-rank 2.2737 (delta -0.8412), heldout reconstruction mse 0.176699 (delta +0.000000), heldout variance 4.336445e-03
RECORD downstream (direct; stage chain blocked at explain): per-direction ablation effects on bottleneck-effective-rank + heldout reconstruction MSE recorded for4 directions; declared trial target bottleneck-mu-dim2 (smallest heldout variance per frozen estimator), off-target bottleneck-mu-dim3; on-target heldout-reconstruction delta -0.000028 against baseline 0.176699 (negligible — the manifest falsification rule1 arithmetic would fire); off-target specificity rule violated by arithmetic=True (the manifest falsification rule2 would fire if the trial executed)
RECORD comparison preview (direct; stage chain blocked): aligned evaluation replays of the pinned revision give identical points — bottleneck-effective-rank 3.1149, bottleneck-singular-spread 0.4735 on both sides, |delta|=0; frozen manifest declares no downstream task metric (the compare task role is bound to a second manifest-declared metric by contract)
PASS frozen invariance: all five frozen artifacts byte-identical after the run
ACCEPTANCE: NOT PASSED —80.23 frozen acceptance cannot be satisfied truthfully
  blocker: declared known defect not observed on the pinned revision: both frozen metrics pass on the declared capture (bottleneck-effective-rank 3.1149 ≥3.0, bottleneck-singular-spread 0.4735 ≥0.25) and on every other legitimate mu capture of the pinned artifacts (train, all-digits, untrained-init variants recorded above)
  blocker: truthful negative localization leaves the80.16 localization binding unresolvable, so the seven-stage chain fails closed at explain and no validator-clean artifact/rendered report exists
  blocker: no localization seam can express the manifest's expected feature-axis direction selection under the frozen effective-rank metric (per-feature rank observations cannot separate a direction at the ≥3.0 threshold)
  blocker record: artifacts/task_80.23_encoder_end_to_end_proof_summary.md (positive/counterexample/negative results, hashes, runtime, limitations, commands)
resources: proof body 10.18s (frozen+data+model=7.66s, capture=0.02s, detect=0.43s, localize=0.54s, workflow=0.93s, guards+previews=0.61s); wall including imports 13.77s; tracemalloc peak 54.5 MiB; environment python 3.13.3, numpy 2.4.6, torch-cpu, scikit-learn 1.9.0; declared output location 'artifacts/diagnostics/proof-80-23' intentionally not materialized (persistence refused)
commands: uv run python scripts/sprint80_task80_23_proof.py; uv run pytest tests/test_encoder_end_to_end_proof.py -q
```

## Hashes recorded

- Manifest commitment (canonical, self-referential field excluded):
  `b4c49c890cf34cc0530d499c2b82a09057f1f1ebf5b9d9a3843200760b137dd3` (matches
  the 80.5 sprint-pinned digest).
- Frozen artifact file SHA-256 (verified byte-identical after the run):
  - `benchmark_manifest_schema_v1.json` — `de24e80e267484df99f880927885fef836f2d70e4bd750e4a7a61d0c979364fc`
  - `benchmark_manifest_sprint80_encoder_autoencoder_v1.json` — `e2935cd821a1605c719b63c748a9f93c2375cd6087c51739862d480a18c56959`
  - `benchmark_manifest_sprint80_transformer_hidden_state_v1.json` — `339c5c302def9d8b859e1914101803c84905347eec3caa1d596a5eadc015d0a2`
  - `diagnostic_report_schema_v1.json` — `a76580e58a672ac283fa0a313b34f6148375dc5181dcb395ae1061ce42928d07`
  - `representation_problem_taxonomy_v1.json` — `e1742b7833e587708133b9d927e39636df93b64b3e348b759353e1be46373c3b`
- Split index digests (match the pinned benchmark artifact):
  train `0b03643758a0dd1999442234f213e51b20f87948dfc7d2871ca05d0ff7f39b90`,
  heldout `48e9e0ed2c89e42ee0fdc3592e8c995f8b95d37184b4ae791e137765c32444c6`.
- Heldout latents SHA-256 (canonical JSON): `a9bb7895b9a0db1fd2c2791f79da95d7cb0ac099c0ac0cf56422d1ae2e80c31a`
- Real capture identity: `16a5c11ae062d83ec489635f05c68df0b3c19acbd0bad65a8a11da3021bc2acd`
- Detect payload SHA-256 (canonical JSON, byte-identical across reruns):
  `faeff74e9ad7aa9185814ef94cbfa8a21afcda72e7fd8cb97709aa18f83bfc8a`
- Localize payload SHA-256 (canonical JSON, byte-identical across reruns):
  `35c17b76f805453b36087776849c2a9131d92dce4a4f384bcfe388a9341ac8df`

## Output locations

**No diagnostic artifact or rendered report was produced** — 80.21 persistence
of the failed workflow is refused with no files written, and the declared
`OutputSelection` (`artifacts/diagnostics/proof-80-23`) is intentionally not
materialized. Evidence lives in this artifact, in the proof script's output
(quoted above), and in the focused tests. A deliberately produced artifact
would require fabricating a localization, which this task forbids.

## Files Modified

* [scripts/sprint80_task80_23_proof.py](../scripts/sprint80_task80_23_proof.py) - New committed executable core proof: frozen-input verification, pinned training with bit-identical replay, real capture binding, real detect/localize execution with every declared control case, truthful-negative localization, workflow stop/resume with fail-closed explain gate, persistence/tamper/failed-control fail-closed guards, payload determinism, direct downstream measurement previews, bounded runtime/resource record, and the computed `ACCEPTANCE: NOT PASSED` verdict.
* Historical task-local test source `tests/test_encoder_end_to_end_proof.py` — its worktree result is retained below, but that source file is not included in this evidence-only publication.
* [docs/sprint-plans/sprint-80.md](../docs/sprint-plans/sprint-80.md) - Explicit truthful blocker recorded under Notes/Blockers; **80.23 checkbox deliberately left `[ ]`** at the original v1 report.
* No `src/` file was modified (zero library changes; all existing seams behave as accepted).

The task-local focused-test source and capture-injection v2 proof script are historical worktree-only files and are intentionally excluded from this documentation/evidence publication. Their observed outcomes remain historical; no clean-clone rerun or source-reproducibility claim is made for those files. The accepted v3 proof and revision-backed replay are separately recorded in the current task handoff and depth-evidence report.

## Testing

* **Test File (historical worktree source, not included in this publication):** `tests/test_encoder_end_to_end_proof.py`
* **Status:** Passed
* **Execution Command:** `uv run pytest tests/test_encoder_end_to_end_proof.py -q`
* **Result:** `7 passed in 8.16s`

Proof (exact command, exit 0 on truthful execution with
`ACCEPTANCE: NOT PASSED` printed):

```text
uv run python scripts/sprint80_task80_23_proof.py
(exit code0; full output quoted above)
```

Focused regression battery over every consumed seam (no formatters, linters,
or project-wide suites were run, per task constraints):

```text
uv run pytest tests/test_encoder_end_to_end_proof.py tests/test_collapse_detection.py tests/test_layer_slice_localization.py tests/test_capture_binding.py tests/test_diagnostic_workflow.py tests/test_probe_tcav_ig_explanation.py tests/test_intervention_trials.py tests/test_run_comparison.py tests/test_sprint80_core_manifests.py -q
93 passed in8.48s
```

Runtime/resources (bounded, well under the16 GiB ceiling): proof body
10.18s (phases: frozen+data+model7.66s, capture0.02s, detect0.43s,
localize0.54s, workflow0.93s, guards+previews0.61s), wall including imports
13.77s, tracemalloc peak54.5 MiB, CPU-only torch, python3.13.3, numpy2.4.6,
scikit-learn1.9.0.

## Graph Refresh Record

First graph refresh after all graph-visible writes (code, tests, proof script):

```text
graphify update .
Re-extracting code files in . (no LLM needed)...
  AST extraction:100/253 uncached files (39%) [8 workers]
  AST extraction:200/253 uncached files (79%) [8 workers]
  AST extraction:253/253 uncached files (100%) [8 workers]
  warning:248 source file(s) produced zero nodes and are absent from the graph (...).
[graphify watch] community set changed since labeling (1056 saved labels, 1036 communities now; renamed167 community(ies) by their hub). Run `graphify label` to refresh names with the LLM.
[graphify] backed up semantic+curated graph (6 files) ->2026-09-22/
Graph has15508 nodes (above5000 limit). Building aggregated community view...
graph.html written (aggregated:1036 community nodes,1471 cross-community edges)
Tip: run with --obsidian for full node-level detail.
[graphify watch] Rebuilt:15508 nodes,34980 edges,1036 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
Wall time:76.93 seconds
```

A final confirming `graphify update .` runs after this artifact and the plan
edit; its output is reported in the task handoff, with no writes afterwards.

## Additional Notes / Limitations

* The80.21/80.22 lanes (content-addressed artifact, independent-validator
  pass, deterministic rendered report, evidence-link re-hash, artifact-level
  tamper refusal) are **unreachable for the encoder case** while the chain
  cannot complete; the persistence refusal itself is exercised and recorded.
  Blob-level tamper/re-hash behavior remains covered only by the accepted
 80.21/80.22 suites and is unexercised for this manifest.
* The real probe bundle (bottleneck → high/low reconstruction error,
  train-median-thresholded labels, leakage-safe identities, declared
  capacity, noise negative control) is constructed and predeclared but
  **never evaluated** — the identity gate fails first, as designed.
* Intervention and comparison stage executors are constructed from real
  declarations (real trial targeting the manifest-expected `bottleneck-mu-dim2`,
  real measure callbacks) but **never invoked**; their real measurement
  values are recorded directly and clearly labeled as direct measurements,
  not executed stage outcomes. The manifest's own falsification-rule
  arithmetic on those values would fire (both rule1 and rule2), i.e. even a
  reachable trial would falsify rather than support on the frozen inputs.
* The frozen manifest declares no downstream task metric: the compare
  stage's task role is contractually bound to a second manifest-declared
  representation metric (`bottleneck-singular-spread`); real heldout
  reconstruction deltas are recorded as supplementary downstream evidence
  outside the manifest metric set. This constraint is relevant to 80.24.
* Candidate resolutions all lie outside this task's authority and would each
  require their own reviewed decision: (a) an evidence-review ruling that
  reclassifies/replaces the encoder manifest's defect declaration as a new
  frozen revision, (b) a predeclared injection protocol for the positive
  case, and/or (c) a feature-axis localization seam plus a
  negative-localization-tolerant explain contract (changes to the accepted
 80.13/80.16 behavior). No post-hoc threshold, data, or manifest tuning was
  performed.
* Scope: no formatters, linters, type checkers, packaging, docs builds, or
  project-wide suites were run, per task constraints.

## Addendum — Prospective injected v2 evidence (separate from frozen v1)

Task handoff: [artifacts/sprint-80/task-23.md](sprint-80/task-23.md).

This addendum records a new prospective qualification; it does not rewrite or
replace the frozen v1 manifest, thresholds, model result, or original
`PASS_BLOCKED` evidence. The v1 manifest digest remains
`b4c49c890cf34cc0530d499c2b82a09057f1f1ebf5b9d9a3843200760b137dd3`.
The prospective v2 manifest is
[`artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v2_injected.json`](benchmark_manifest_sprint80_encoder_autoencoder_v2_injected.json),
SHA-256 `118a1f1380f7bb462657b5c33972e49c7ee15c3f7b67af75903090f52b432f0a`.
Its declared capture-level intervention clamps bottleneck feature 0 to the
training-split mean. It is an injected capture defect, not model-weight damage
or a claim that the original trained encoder naturally collapsed.

### Historical v2 evidence path

The v2 proof used a historical task-local script (`scripts/sprint80_task80_23_v2_proof.py`) that is not included in this evidence-only publication. It used a predeclared per-feature variance-ratio detector and feature-axis localization through the generic axial seam; no encoder-specific product path was added. The detector is opt-in, so the v1 manifest's metric set and decisions remain unchanged. The v2 feature-positive threshold was a variance ratio of at least `0.1`, with the manifest-declared `0.01` tolerance; the manifest was committed before the injected proof data was produced.

### End-to-end injected run

**Command:** `uv run python scripts/sprint80_task80_23_v2_proof.py`  
**Observed result:** exit 0. The run completed capture, detect, localize,
explain, intervene, compare, and report stages, then persisted to a temporary
artifact root, reloaded through the independent validator, rendered
deterministically, and cleaned the temporary root.

- Detector metric `bottleneck-feature-variance-ratio` was
  `2.5867879120338035e-29` against the declared `>= 0.1` criterion. Per-feature
  ratios were approximately `[2.59e-29, 1.2721, 0.7279, 1.4561]`; the injected
  feature is below criterion while the other three remain noncollapsed. Both
  declared detector controls passed.
- Feature-axis localization returned `localized` and selected exactly
  `bottleneck-mu-dim0`; the feature-axis declaration matched the manifest's
  identity, selection, representation, and complete ordered coverage.
- The probe explanation is **inconclusive**, not a positive: it stopped on
  `failed-fidelity:heldout_accuracy`. In particular, the reconstruction-error
  hypothesis is not evidence attributing the injected feature.
- The paired healthy-value patch intervention was supported with effect
  `0.6150898081662849`; zero, random, shuffled, and off-target controls all
  passed. The observed metric values were approximately `2.59e-29` for zero,
  shuffled, and off-target controls, `0.0001094` for random, and `0.6151` for
  the paired healthy-value patch. This is evidence about the predeclared
  capture-level injection, not an intervention on model weights.
- Comparison classified the two identical injected replays as `neither` (no
  metric change). It is a replay/alignment check, not evidence of utility or
  improvement.
- Temporary persisted artifact run ID was `093a6fb0bb3eb317`; artifact digest
  `484fbf23747cd8bfe54f8455348067c684ef51c4fcabc92d4217a2d0132995fb`;
  report digest
  `0c0bea45858ed718971d66ed5c47b923840e0af48cf60bc20f468b8fa669440d`;
  rendered report SHA-256
  `78b443ddac8db5606fe2bf838a70af6c645a6ae0cfc1748dbbc930215400d475`
  (5,800 bytes). The run artifact root was temporary and has been cleaned;
  the committed script reproduces the run.
- The run verified the frozen v1 artifacts and v2 manifest remained byte
  unchanged. It does not turn the original v1 case into a validator-clean,
  user-reproducible positive case.

### Focused verification and disposition

The changed feature-axis and detector behavior passed:

```text
uv run pytest -q tests/test_checkpoint_token_time_localization.py tests/test_probe_tcav_ig_explanation.py::test_explain_executor_binds_real_axial_location tests/test_collapse_detection.py::test_predeclared_feature_variance_ratio_detects_injected_axis_collapse
16 passed in 5.23s
```

The original frozen v1 remains **PASS_BLOCKED**: its declared positive was not
observed, and the v1 workflow still stops at the explain fidelity gate after
truthful negative localization. The separate v2 result remains capture-level
and inconclusive. The user authorized amended prospective v3 acceptance on
2026-09-23; the v3 outcome and exact validator/render evidence are recorded in
the next addendum. Its success does not relabel v1/v2 or resolve
clean-environment or production-model breadth. Plan remains `[~]` pending
Main's evidence review, not `[x]`. No v1/v2 manifest or evidence was modified.

**V2 limitations:** the v2 explanation is inconclusive; the injection is
capture-level and cannot establish weight-damage behavior; identical-replay
comparison says nothing about utility. The historical v1 blockers and 80.28
release gates remain unchanged.

## Addendum — Evidence-review correction (2026-09-23)

The type-check finding was corrected in `src/latent_anything/_layer_slice_localization.py`: feature-index validation now runs through a helper accepting `object`, retaining runtime rejection of booleans, non-integers, and negatives without pyright's unnecessary `isinstance(int, int)` diagnostic. Regression coverage is in `tests/test_checkpoint_token_time_localization.py`.

Focused verification on the correction:

- `uv run pyright src/latent_anything/_layer_slice_localization.py` — **0 errors, 0 warnings, 0 information diagnostics**.
- `uv run pytest -q tests/test_checkpoint_token_time_localization.py` — **17 passed in 6.06s**.
- `uv run python scripts/sprint80_task80_23_v2_proof.py` — **exit 0**; all seven stages, temporary persistence reload/independent validation, deterministic rendering, and cleanup succeeded. The repeat returned the same temporary run ID `093a6fb0bb3eb317`, artifact digest `484fbf23747cd8bfe54f8455348067c684ef51c4fcabc92d4217a2d0132995fb`, report digest `0c0bea45858ed718971d66ed5c47b923840e0af48cf60bc20f468b8fa669440d`, and render digest `78b443ddac8db5606fe2bf838a70af6c645a6ae0cfc1748dbbc930215400d475`. Frozen v1 artifacts and the v2 manifest were unchanged.

The v2 explanation remains **inconclusive** (`failed-fidelity:heldout_accuracy`); the existing reconstruction-error probe does not attribute signal to dim0. The comparison remains `neither`: it compares two identical injected replays, with zero point delta on both metrics, so it establishes neither task utility nor improvement. Its task-role metric is `bottleneck-effective-rank`, also a representation metric; the frozen v2 manifest declares no downstream task-utility metric. Although the manifest predeclares a paired healthy-value intervention, it does not predeclare an explanation target/label protocol or a distinct comparison pair. Replacing these after observing outcomes and rerunning would not provide predeclared proof. No scientifically admissible path to a positive explanation or task-utility comparison was found within the frozen v2 evidence specification.

The v1 declared positive remains unobserved and **PASS_BLOCKED**; v2 remains
inconclusive. The user-amended v3 prospective run passes separately, as
recorded in the addendum below. This does not reclassify the historical
outcomes. The plan remains `[~]` pending Main's evidence gate. No frozen
manifest, threshold, or prior output was modified. See the
[task handoff](sprint-80/task-23.md).

## Addendum — Frozen v3 model-weight-lesion proof (2026-09-23)

The final frozen v3 run satisfies the user-amended prospective 80.23 acceptance; it does not alter the v1 `PASS_BLOCKED` or v2 inconclusive results. The plan remains `[~]` until Main's evidence review. Full attempt chronology and control results are recorded in the [task handoff](sprint-80/task-23.md).

### Frozen inputs and final run

- Manifest `sprint80-core-encoder-autoencoder-collapse-model-lesion-v3`: commitment digest `f823efc1b016ca9ad62c0f207e17d138947040a39f5d1c1f42ab6d0d65f5954f`; raw-file SHA-256 `e5bcd341b1b746f05e4ac02f908bcd27f23bace39737c86c8ac38888e7451b82`.
- Train-only baseline checkpoint SHA-256 `92829e57aaaebf52d5446d1a9c9db5de4e12c1657fad5ecc986329ed8752bb2a`.
- Final run command:

  ```text
  uv run python -c "import scripts.sprint80_task80_23_v3_proof as proof; proof.OUTPUT_LOCATION = 'artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-complete-20260923'; proof.OUTPUT_ROOT = proof.REPO / proof.OUTPUT_LOCATION; raise SystemExit(proof.main())"
  ```

  Result: exit 0, `acceptance: passed`; capture, detect, localize, explain, intervene, compare, and report completed. The proof confirmed the manifest remained byte-identical during the run and recorded the unchanged baseline checkpoint hash.


### Verified contract changes and files

`src/latent_anything/_intervention_trials.py` allows a manifest-declared causal target outside detector selection only through a requested comparison metric, while retaining manifest threshold, family, representation, request, and evaluated-hypothesis checks. Its comparison-only regression in `tests/test_intervention_trials.py` passed. `src/latent_anything/_report_renderer.py` now verifies that such a metric is the request/report-declared comparison task metric and renders positive `supported` location rows. `scripts/sprint80_task80_23_v3_proof.py` maps the detector's supported result to the report's valid observation status. The architecture decision and user-facing changelog entry are in `.agents/memory/decisions.md` and `CHANGELOG.md`.

`tests/test_report_renderer.py` adds a regression asserting that `supported` localization status is preserved in rendered output. Verification: `uv run pytest -q tests/test_report_renderer.py::test_supported_location_status_is_rendered` — **1 passed in 4.25s**.

### Scientific result

The model-boundary defect zeros encoder row 0 and its bias in a copy before heldout encoding. Detection supported the defect: effective rank `2.9576990034956805` (`>=3.0` frozen threshold), singular spread `0.0` (`>=0.25`), and feature-variance ratio `0.0` (`>=0.1`). The healthy counterexample, benign low-gain negative, and null-shuffle controls passed.

Localization selected feature `brightness-axis-dim0` with confidence `1.0`. The leakage-safe explanation was supported: heldout accuracy `1.0` against `0.7`, coefficient stability `1.0` against `0.8`, and leakage gap `0.9916666666666667` against `0.15`; probe capacity, randomized, and negative controls passed, and train/heldout identities were disjoint.

Restoration of the actual lesioned model row and bias was causally supported with heldout brightness-bin-accuracy effect `+0.4638888888888889` against the frozen `0.02` tolerance. Zero-strength, random-direction, shuffled-pairing, and off-target controls passed. The aligned comparison classified `both`: task accuracy was `1.0` for the healthy run and `0.5361111111111111` for the lesioned run (candidate-minus-baseline `-0.4638888888888889`, tolerance `0.05`); representation feature-variance ratio changed by `-0.8494925859561244` (tolerance `0.05`).

### Validated artifact and limits

- Output: [`artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-complete-20260923/`](diagnostics/proof-80-23-v3-model-weight-lesion-complete-20260923/), run ID `6bca9afb72af32c4`.
- Artifact digest `e5c6db0ab5080807138e091bc332a5d3e788889a382917e93775b15f562aef0f`; report digest `8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194`.
- Independent validator passed (`diagnostic-report-validator`; inputs digest `c92eb128bc43c8bcbf9d673abf17ff9c4153c3abe9c12d70b64eaa0dfc635318`).
- The deterministic persisted rendered report is [`encoder-diagnosis-v3.md`](diagnostics/proof-80-23-v3-model-weight-lesion-complete-20260923/encoder-diagnosis-v3.md); rendered and persisted SHA-256 match at `b8d713de3181ef39990d994d4f3c708e584eb3c559f762376e2b9f2de65de6e3`. It explicitly renders `feature brightness-axis-dim0` as supported.

This proves the frozen controlled four-dimensional linear autoencoder and brightness-bin task only. It does not turn the historical ConvVAE v1 blocker into a pass, resolve v2's explanation/comparison limitations, demonstrate a production encoder failure, or reproduce the result in a clean environment; those historical and broader findings remain separate.

### Graph refresh after v3 source and test changes

`graphify update .` completed successfully after the renderer regression was added: **16,175 nodes, 37,374 edges, 1,081 communities**; aggregated `graph.html` has 1,081 community nodes and 1,594 cross-community edges. It processed 278 uncached code files and rebuilt `graph.json`, `graph.html`, and `GRAPH_REPORT.md`. Graphify warned that 274 source files produced zero nodes and that labels were stale (1,083 labels versus 1,081 communities; 207 communities renamed); it noted that document/paper/image changes require the assistant's `/graphify --update` path, while this CLI run refreshed the code graph.

Graphify also backed up six semantic/curated graph files under `graphify-out/2026-09-23/`.
