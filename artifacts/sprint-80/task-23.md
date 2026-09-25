# Sprint 80 Task 80.23 — Prospective injected encoder proof

**Plan status:** `[~]` — awaiting Main's evidence review. This task has not been marked `[x]`.

## Outcome

The frozen v2 capture-injection proof remains separate historical evidence: its probe was inconclusive and its identical-replay comparison did not establish task utility. A later user-authorized prospective v3 model-weight lesion now passes the amended 80.23 acceptance through all seven stages with a supported explanation, causal restoration controls, downstream task comparison, independent validation, and deterministic report rendering. It does not relabel v1 or v2; this task remains `[~]` pending Main's evidence review and is not `[x]`.

The detailed historical v1 evidence plus the v2 evidence addendum are in [`artifacts/task_80.23_encoder_end_to_end_proof_summary.md`](../task_80.23_encoder_end_to_end_proof_summary.md). The frozen v2 manifest is [`artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v2_injected.json`](../benchmark_manifest_sprint80_encoder_autoencoder_v2_injected.json), SHA-256 `118a1f1380f7bb462657b5c33972e49c7ee15c3f7b67af75903090f52b432f0a`; its capture-level dim0 clamp and criteria were committed before producing the new proof result. The immutable v1 files and v2 manifest remained byte-identical during the smoke run.

## Changed files

- `src/latent_anything/_collapse_detection.py` — opt-in feature variance-ratio metric; not added to the v1 manifest or decision path.
- `src/latent_anything/_layer_slice_localization.py` — generic feature-axis declaration/cell validation and localization via the existing axial seam; old no-feature serialization remains unchanged.
- `src/latent_anything/_probe_tcav_ig_explanation.py` — permits feature-bound hypotheses through the generic explanation contract.
- `scripts/sprint80_task80_23_v2_proof.py` — reproducible seven-stage v2 smoke path, temporary artifact persistence, independent reload validation, deterministic render, and cleanup.
- `tests/test_collapse_detection.py` — predeclared feature-metric injected-axis behavior.
- `tests/test_checkpoint_token_time_localization.py` — feature-axis localization, order, negative, tolerance, manifest alignment and coverage behavior.
- `tests/test_probe_tcav_ig_explanation.py` — composed feature-axis binding and fabricated-identity rejection.
- `artifacts/task_80.23_encoder_end_to_end_proof_summary.md` — preserves the original v1 blocker and adds separately labeled v2 evidence.
- `.agents/memory/decisions.md` — records the v1/v2 evidence boundary and generic feature-axis seam decision.
- `artifacts/sprint-80/task-23.md` — this required task handoff record.

No sprint-plan row or checkbox was changed. No frozen v1/v2 manifest, threshold, or prior proof output was modified. User files `head_init_dump.py` and `head_init_tmp.py` were untouched.

## Verification

- `uv run pytest -q tests/test_checkpoint_token_time_localization.py tests/test_probe_tcav_ig_explanation.py::test_explain_executor_binds_real_axial_location tests/test_collapse_detection.py::test_predeclared_feature_variance_ratio_detects_injected_axis_collapse` — **16 passed in 5.23s**.
- `uv run python scripts/sprint80_task80_23_v2_proof.py` — **exit 0** on final source state. All seven stages (`capture`, `detect`, `localize`, `explain`, `intervene`, `compare`, `report`) executed; temporary persistence reloaded successfully through the independent validator and deterministic rendering succeeded. The temporary root was removed.
  - Detector metric: `2.5867879120338035e-29` against predeclared ratio `>= 0.1`; ratios approximately `[2.59e-29, 1.2721, 0.7279, 1.4561]`; both detector controls passed.
  - Localization: `localized`, exactly `bottleneck-mu-dim0`.
  - Explanation: **inconclusive**, blocked by `failed-fidelity:heldout_accuracy`; not claimed as a positive.
  - Paired healthy-value capture patch: supported effect `0.6150898081662849`; zero/random/shuffled/off-target controls passed.
  - Identical injected replay comparison: `neither` (no metric change); not utility evidence.
  - Temporary run ID `093a6fb0bb3eb317`; artifact digest `484fbf23747cd8bfe54f8455348067c684ef51c4fcabc92d4217a2d0132995fb`; report digest `0c0bea45858ed718971d66ed5c47b923840e0af48cf60bc20f468b8fa669440d`; deterministic render SHA-256 `78b443ddac8db5606fe2bf838a70af6c645a6ae0cfc1748dbbc930215400d475` (5,800 bytes). These checksums identify the temporary validated run; its artifact directory was cleaned, and the committed script is the reproducible source.

No project-wide suite, formatter, linter, remote/CUDA run, or post-hoc threshold change was performed.

## Historical v1/v2 findings

- Original v1 acceptance remains `PASS_BLOCKED`; its original positive is still absent and the frozen workflow fails its explain gate.
- The injected proof is capture-level and does not establish model-weight damage, naturally occurring encoder collapse, or user-reproducibility of either frozen core case.
- Probe explanation is inconclusive; identical-replay comparison does not establish utility.
- The v2-only acceptance gap was superseded by the user's 2026-09-23 amendment and the separate v3 proof recorded below; Main's evidence review remains pending.

## Graph update

The `graphify update .` refresh after the final source/test/script and Markdown link edits and the task record completed successfully (exit 0). Observed graph: 16,098 nodes, 37,119 edges, 1,076 communities; aggregated `graph.html` has 1,076 community nodes and 1,545 cross-community edges. Graphify warned that 262 source files produced zero nodes and suggested retrying; it also suggested `graphify label` after community names changed. A confirming refresh was invoked after recording this result; graph absence was not observed.

## Review correction follow-up — 80.23

### Changed files in this correction

- `src/latent_anything/_layer_slice_localization.py` — centralized non-negative integer validation in an `object`-accepting helper. `FeatureCell` still rejects booleans, non-integers, and negative indices at runtime without an unnecessary type check on its annotated `int`.
- `tests/test_checkpoint_token_time_localization.py` — added a regression check for boolean, floating-point, and negative feature indices.
- `artifacts/task_80.23_encoder_end_to_end_proof_summary.md` and this handoff — recorded the reviewer correction and bounded scientific findings; historical v1/v2 evidence remains intact.
- `graphify-out/graph.json`, `graphify-out/graph.html`, and `graphify-out/GRAPH_REPORT.md` — refreshed by the code graph update; six prior semantic/curated graph files were archived under `graphify-out/2026-09-23/`.

### Focused verification

- `uv run pyright src/latent_anything/_layer_slice_localization.py` — **0 errors, 0 warnings, 0 information diagnostics**.
- `uv run pytest -q tests/test_checkpoint_token_time_localization.py` — **17 passed in 6.06s**, including the three invalid runtime feature-index cases.
- `uv run python scripts/sprint80_task80_23_v2_proof.py` — **exit 0**; all seven workflow stages ran, the temporary artifact reloaded through the independent validator, deterministic rendering succeeded, and temporary output was removed. The manifest digest remained `118a1f1380f7bb462657b5c33972e49c7ee15c3f7b67af75903090f52b432f0a`; detector/localization/intervention remained as previously recorded. Explanation is still **inconclusive** (`failed-fidelity:heldout_accuracy`). The comparison remains **`neither`**, with zero point delta for both compared metrics on the identical injected replay. Frozen v1 artifacts and the v2 manifest were unchanged.

### Bounded scientific finding

No new explanation target, label construction, probe capacity, comparison pair, or metric was introduced after observing the run. The frozen v2 manifest predeclares the capture-level injection, detector thresholds/controls, and paired healthy-value intervention expectation, but not an explanation target/label protocol or an independent run-comparison pair and downstream task-utility metric. The current probe predicts train-thresholded reconstruction-error category; its feature binding does not establish dim0 attribution, and its heldout-fidelity gate fails. The comparison replays the same injected capture on both sides; its `neither` result is an alignment/replay check, not utility evidence. The task-metric slot uses `bottleneck-effective-rank`, itself a representation metric; no downstream task metric is declared. Changing these inputs or the comparison contrast and rerunning now would not be a predeclared positive proof. I found no admissible path to promote explanation fidelity or task utility within the frozen v2 evidence specification.

The original v1 declared positive remains unobserved and **PASS_BLOCKED**; the separate v2 capture injection remains inconclusive. The user-amended prospective v3 result is recorded below and does not reclassify either historical result. Task status remains `[~]` pending Main's evidence gate; no frozen v1/v2 manifest, threshold, proof output, or prior result was modified.

The required `graphify update .` completed after the source, test, summary, and handoff correction edits (exit 0). It rebuilt the code graph to **16,107 nodes, 37,131 edges, and 1,072 communities**; aggregated `graph.html` contains **1,072 community nodes and 1,535 cross-community edges**. Graphify warned that 262 source files produced zero nodes and that labels were stale (1,076 saved labels versus 1,072 communities; 210 communities renamed). This invocation performed AST code extraction only and advised `/graphify --update` for document changes; no semantic documentation refresh was run. No source code changed after this successful code-graph update.


The preceding graph note records the earlier v2 correction snapshot; v3 source and evidence edits followed, and their graph refresh is recorded below.

## Addendum — Frozen v3 model-weight-lesion proof (2026-09-23)

The user-selected prospective acceptance amendment is recorded at [`docs/sprint-plans/sprint-80.md`](../../docs/sprint-plans/sprint-80.md). This addendum preserves the frozen v1 `PASS_BLOCKED` and v2 inconclusive outcomes and records the v3 proof separately. The final v3 evidence passes the amended scope, but the plan row remains `[~]` until Main's evidence review.


### V3 changes in this follow-up

- `src/latent_anything/_intervention_trials.py` and `tests/test_intervention_trials.py` — retained and exercised the comparison-only task-metric causal binding and focused regression.
- `src/latent_anything/_report_renderer.py` — cross-checks comparison-task metric evidence against request/report declarations and renders positive `supported` localization rows.
- `tests/test_report_renderer.py` — regression verifies a `supported` localization status is preserved in the rendered report.
- `scripts/sprint80_task80_23_v3_proof.py` — serializes supported detection as the report's observation status instead of passing an invalid `supported` observation status to the report validator.
- `.agents/memory/decisions.md` and `CHANGELOG.md` — record the generic task-metric and report-binding contract.
- `graphify-out/graph.json`, `graphify-out/graph.html`, and `graphify-out/GRAPH_REPORT.md` — refreshed after v3 source and regression-test changes.
- This handoff and [`artifacts/task_80.23_encoder_end_to_end_proof_summary.md`](../task_80.23_encoder_end_to_end_proof_summary.md) — preserve chronological v3 evidence and limitations.

The frozen v3 manifest and train-only checkpoint were not changed. `docs/sprint-plans/sprint-80.md` remains `[~]`; no plan status was changed.

### Frozen inputs and execution chronology

- Frozen manifest: [`artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v3_model_weight_lesion.json`](../benchmark_manifest_sprint80_encoder_autoencoder_v3_model_weight_lesion.json), commitment digest `f823efc1b016ca9ad62c0f207e17d138947040a39f5d1c1f42ab6d0d65f5954f`, raw-file SHA-256 `e5bcd341b1b746f05e4ac02f908bcd27f23bace39737c86c8ac38888e7451b82`.
- Train-only baseline checkpoint: [`artifacts/benchmark_model_sprint80_encoder_autoencoder_v3_baseline.npz`](../benchmark_model_sprint80_encoder_autoencoder_v3_baseline.npz), SHA-256 `92829e57aaaebf52d5446d1a9c9db5de4e12c1657fad5ecc986329ed8752bb2a`.
- Every v3 attempt used those same frozen inputs. No v1/v2 manifest, threshold, target, control, checkpoint, or proof output was edited or overwritten.

Chronological attempts:

1. The previous worker's first `--run` is preserved at [`artifacts/diagnostics/proof-80-23-v3-model-weight-lesion/proof-run.json`](../diagnostics/proof-80-23-v3-model-weight-lesion/proof-run.json). It failed before intervention measurement with `StageContractError` because `heldout-brightness-bin-accuracy` was not considered request-declared by the intervention executor. That record remains unchanged.

   ```text
   uv run python scripts/sprint80_task80_23_v3_proof.py --run
   ```

2. The comparison-only binding regression was exercised:

   ```text
   uv run pytest -q tests/test_intervention_trials.py::test_comparison_only_task_metric_can_drive_a_causal_intervention
   1 passed in 7.57s
   ```


   First fresh retry command:

   ```text
   uv run python -c "import scripts.sprint80_task80_23_v3_proof as proof; proof.OUTPUT_LOCATION = 'artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-rerun'; proof.OUTPUT_ROOT = proof.REPO / proof.OUTPUT_LOCATION; raise SystemExit(proof.main())"
   ```
   The first fresh retry, output root `artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-rerun`, reached report construction but failed with `DiagnosticReportValidationError: observation cannot use status supported`. It produced no persisted diagnostic artifact.
3. The next fresh retry, `artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-final-20260923`, completed all workflow stages and persisted a validator-clean artifact (`run_id` `eefec71de29280e9`), but rendering stopped because the generic renderer required an intervention metric to appear in detection evidence. After adding request/report-bound comparison-task evidence, direct rendering of that persisted record succeeded in memory (`5,690` bytes, SHA-256 `a5c89b38dd9196de32469ec53584b26cddccd57fbf18f7b3fa81daa90eba9dce`).

   Retry command:

   ```text
   uv run python -c "import scripts.sprint80_task80_23_v3_proof as proof; proof.OUTPUT_LOCATION = 'artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-final-20260923'; proof.OUTPUT_ROOT = proof.REPO / proof.OUTPUT_LOCATION; raise SystemExit(proof.main())"
   ```

   Direct renderer verification after its task-metric fix:

   ```text
   uv run python -c "import hashlib,json; from pathlib import Path; from latent_anything._report_renderer import render_diagnostic_report; root=Path('artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-final-20260923'); manifest=json.loads(Path('artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v3_model_weight_lesion.json').read_text(encoding='utf-8')); rendered=render_diagnostic_report(root,run_id='eefec71de29280e9',manifest=manifest); text=rendered.decode('utf-8'); print(next(line for line in text.splitlines() if line.startswith('location:'))); print(json.dumps({'bytes':len(rendered),'render_sha256':hashlib.sha256(rendered).hexdigest()},sort_keys=True))"
   ```
4. A subsequent complete run at `artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-verified-20260923` reported `acceptance: passed` for run ID `106b2bad3185ac77`, artifact digest `4e1e63108637f85f5926186c7e6948824686007118ff0d50450234bee0d6da82`, and rendered digest `fb212d7da9d8ff2b6de2093cd2d2698ff054c29113a0ded00fa1c3d709a29332`. Its rendered report incorrectly said no supported location existed despite the localized dim0 evidence, so this intermediate artifact is preserved but not used as final evidence. The renderer was corrected to render `supported` feature locations.

   Intermediate complete-run command:

   ```text
   uv run python -c "import scripts.sprint80_task80_23_v3_proof as proof; proof.OUTPUT_LOCATION = 'artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-verified-20260923'; proof.OUTPUT_ROOT = proof.REPO / proof.OUTPUT_LOCATION; raise SystemExit(proof.main())"
   ```
5. The final, fresh proof used this exact command and output root:

   ```text
   uv run python -c "import scripts.sprint80_task80_23_v3_proof as proof; proof.OUTPUT_LOCATION = 'artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-complete-20260923'; proof.OUTPUT_ROOT = proof.REPO / proof.OUTPUT_LOCATION; raise SystemExit(proof.main())"
   ```

   It exited 0 with `acceptance: passed`, workflow status `completed`, and all seven stages: `capture`, `detect`, `localize`, `explain`, `intervene`, `compare`, `report`.

### Frozen v3 scientific results

- The declared model-boundary lesion zeroed `encoder_weight[0, :]` and `encoder_bias[0]` in a copy of the train-only checkpoint before heldout encoding. The frozen healthy checkpoint and heldout inputs were unchanged. The resulting lesion checkpoint SHA-256 is `81a29cc8d92b38164b108958764a6f1ffe07e5b822701077bd45faeffea91008`; lesion record SHA-256 is `d1b0763ee791b0069dc94e13b698edb51f4fcd7711ff169ce22264c3130d0570`.
- Detection supported the declared collapse: feature-variance ratio `0.0` against `>= 0.1`, singular spread `0.0` against `>= 0.25`, and effective rank `2.9576990034956805` against `>= 3.0`. The healthy counterexample, benign low-gain negative, and null-shuffle controls all passed.
- Localization selected exactly `brightness-axis-dim0` on the feature axis (confidence `1.0`). The persisted rendered report states `location: feature brightness-axis-dim0 (confidence 1.0, status supported)`.
- Explanation was **supported**. The leakage-safe probe had heldout accuracy `1.0` (threshold `>= 0.7`), coefficient stability `1.0` (threshold `>= 0.8`), and leakage gap `0.9916666666666667` (threshold `>= 0.15`); its capacity, randomized, and negative controls passed, and train/heldout sample identities were disjoint.
- Restoring the actual lesioned encoder row and bias was causally supported for heldout brightness-bin accuracy: intervention effect `+0.4638888888888889`, greater than the frozen `0.02` tolerance. Zero-strength, random-direction, shuffled-pairing, and off-target-row controls all passed.
- The aligned healthy-versus-lesion comparison classified `both`: representation feature-variance-ratio delta `-0.8494925859561244` against tolerance `0.05`; task accuracy was `1.0` on healthy versus `0.5361111111111111` on lesion, with candidate-minus-baseline delta `-0.4638888888888889` against tolerance `0.05`. The task metric and causal tolerance were predeclared in the unchanged manifest.

### Persisted artifact and independent verification

- Final output root: [`artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-complete-20260923/`](../diagnostics/proof-80-23-v3-model-weight-lesion-complete-20260923/).
- Run ID `6bca9afb72af32c4`; content-addressed artifact digest `e5c6db0ab5080807138e091bc332a5d3e788889a382917e93775b15f562aef0f`.
- Independent validator: `diagnostic-report-validator`, status `passed`, inputs digest `c92eb128bc43c8bcbf9d673abf17ff9c4153c3abe9c12d70b64eaa0dfc635318`.
- Report digest `8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194`. Persisted report: [`encoder-diagnosis-v3.md`](../diagnostics/proof-80-23-v3-model-weight-lesion-complete-20260923/encoder-diagnosis-v3.md).
- Rendering was deterministic; rendered and persisted bytes both hash to `b8d713de3181ef39990d994d4f3c708e584eb3c559f762376e2b9f2de65de6e3`.
- The proof record confirms manifest commitment `f823efc1b016ca9ad62c0f207e17d138947040a39f5d1c1f42ab6d0d65f5954f`, raw manifest SHA-256 `e5bcd341b1b746f05e4ac02f908bcd27f23bace39737c86c8ac38888e7451b82`, baseline checkpoint SHA-256 `92829e57aaaebf52d5446d1a9c9db5de4e12c1657fad5ecc986329ed8752bb2a`, and `manifest_raw_unchanged_during_run: true`.

Focused verification after the contract changes:

```text
uv run pytest -q tests/test_report_renderer.py tests/test_intervention_trials.py::test_comparison_only_task_metric_can_drive_a_causal_intervention
15 passed in 24.22s
```

The focused status-rendering regression added for this v3 contract passed:

```text
uv run pytest -q tests/test_report_renderer.py::test_supported_location_status_is_rendered
1 passed in 4.25s
```

No project-wide suite, formatter, linter, remote/CUDA run, or clean-environment reproduction was run. This is prospective controlled evidence for the frozen four-dimensional linear autoencoder and one digits brightness-bin task. It does not establish a naturally occurring failure in the historical ConvVAE, production encoders, general digit recognition utility, or clean-environment reproduction. Historical v1 remains **PASS_BLOCKED**, v2 remains inconclusive, and task status remains `[~]` pending Main's evidence gate.

### Graph refresh after v3 source and test changes

```text
graphify update .
Graph: 16,175 nodes, 37,374 edges, 1,081 communities.
Aggregated graph.html: 1,081 community nodes, 1,594 cross-community edges.
AST extraction: 278 uncached code files processed.
```


Graphify also backed up six semantic/curated graph files under `graphify-out/2026-09-23/`.
The graph refresh completed successfully after the renderer regression was added. Graphify warned that 274 source files produced zero nodes and that saved community labels were stale (1,083 labels versus 1,081 communities; 207 communities renamed). It rebuilt `graph.json`, `graph.html`, and `GRAPH_REPORT.md`. The command explicitly noted that document/paper/image changes require the assistant's `/graphify --update` path; this CLI run refreshed the code graph.
