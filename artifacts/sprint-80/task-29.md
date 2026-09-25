# Sprint 80 Task 80.29 — Revision-backed depth-evidence publication

**Plan state:** `[~]`; this handoff does not mark task 80.29 `[x]` or Sprint 80 complete. Main owns the evidence gate and final status. The third sprint-wide deep review, `agent://DeepReviewSprint80Third`, returned **FAIL** on H1–H3; those prior corrections remain recorded below. The fourth deep review, `agent://DeepReviewSprint80Fourth`, returned **FAIL** on D1/D2; this correction addresses both in commit `4a5e8f8`, and a fresh final deep review remains pending.

## Outcome

Published the missing evidence/report closure without changing proof code, thresholds, frozen inputs, or scientific outcomes. The 23-path integrated content package is at `9d86407a3b56ab634eb7228078a51d6df828bdc7`, following the initial evidence publication `cf96f703f31ed2366b1359adcb054e464fec63f1`. The `9d86407` follow-up removed one stale link to an unpublished, out-of-scope task-80.19 summary. This task record and its sprint-row link are included in the subsequent handoff-record update.

The depth report, `docs/PLAN.md`, `docs/INDEX.md`, Sprint 80 plan, task evidence, small blocker inputs, and published quality summary now agree on revision lineage and the bounded handoff. Sprint 80 is Active; Sprint 81 READY remains limited to the accepted ordinary-DL core and conditional on final deep review.

## Fourth deep review — D1/D2 correction and scope

**D1 — current versus historical API inventory:** `docs/PLAN.md` now identifies the published `v0.9.0` release-time 205-runtime / 202-canonical-stable / 7-exception inventory and digest `048ac553adabb11c24d3e1f4d86e0c6d469df6590064014c915d6a08c7d53c26` as historical. The current checked-in source snapshot is identified separately as 214 runtime exports / 211 canonical-stable names / 8 public exceptions, digest `3fd8c73e6fd9fa9bae107b4be6727cc3e9c3c907c9b9d757fdae4c267d8691ec`. Published release assets and other genuine historical records were not rewritten.

**D2 — memory scope (LOW):** The pre-existing `.agents/memory/decisions.md` diff consisted of 82 appended lines forming 10 Sprint 80 ADRs (dated 2026-09-21 through 2026-09-24), tied to the 80.6–80.24 design/API and evidence decisions. The `.agents/memory/lessons-learned.md` diff consisted of 139 appended lines forming 12 Sprint 80 bug or environment lessons (dated 2026-09-22 through 2026-09-24), with concrete symptoms, causes, fixes, and triggers. The additions are task-scoped project memory, not unrelated user work; they were published without rewriting earlier entries. No unrelated memory changes were found in either diff.

**Plan ownership:** The shared worktree's Sprint 80 row 80.29 reads `[~]`; its existing `docs/sprint-plans/sprint-80.md` change was not staged or committed by this task. The clean clone at the correction revision therefore still has the prior committed `[x]` row. Main owns publishing the `[~]` plan update; this handoff does not change it.


## H2 — Revision-backed evidence and link closure

- The report and plan distinguish the accepted 80.25 replay measured from source revision `5241553` (24/24 checks), replay evidence commit `938edd9`, and task-25 handoff commit `f15859b`. `f15859b` contains the 80.25 handoff, not the 80.28 handoff.
- The 80.28 broad gates are attributed to the shared-worktree code basis `f15859b`, not to a clean-clone suite. The final selection recorded **2,549 passed, 3 skipped, 42 deselected, 39 warnings** in 2,562.53 seconds. The earlier corrected same-count shared-worktree run at 1,603.61 seconds is separately identified in the quality summary. Neither is a full-suite clean-clone run. The optional-extras matrix remains prior evidence, not an 80.28 rerun.
- H1 overlays at `cef267e` remain scoped to the clean-clone API check, 12 focused tests, file-scoped Ruff checks, and 9/9 driver dry-run; the full suite was not rerun in that clone.
- The revision-backed [Task 80.24 accepted stability record](../diagnostics/proof-80-24-stability-v1/runs/a95ccce32461b78ddd0197188fc1f4b51a1b349d434d3a9812dbcc8251dbeb2d.json) and [deterministic report](../diagnostics/proof-80-24-stability-v1/reports/5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0.md) are included. Run SHA-256 matches `a95ccce32461b78ddd0197188fc1f4b51a1b349d434d3a9812dbcc8251dbeb2d` (332,209 bytes); report SHA-256 matches `5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0` (2,380 bytes). Both the accepted run and report are content-addressed by their raw bytes.
- The separate pre-validation stability candidate is preserved but remains explicitly **unaccepted**: its run (`593545a20b136ede1400786c15d4665c444591d50f86569a00f0da566777812a`, 331,837 bytes) and report (`572588e3de89a3976c31f3a580cff3b24f8a3366be42415a01a877d609b6cb46`, 2,380 bytes) are also raw-hash matched to their filenames. The accepted supplement has held-out accuracy `1.0`, `coef_stability=0.9748782295192386` at threshold `>=0.8`, leakage gap `0.4254901960784314` at threshold `>=0.15`, and all three gates/controls passed. The immutable target-evidence-v2 observation remains `coef_stability=1` with `threshold:null`; it is not counted as gated evidence.
- Task 80.23 and 80.24 records/summaries preserve old negatives and distinguish historical worktree-only focused-test sources from revision-backed evidence. Those excluded test sources are not presented as clean-clone runnable. The first v3 proof-failure record is published at [`proof-run.json`](../diagnostics/proof-80-23-v3-model-weight-lesion/proof-run.json), retaining `acceptance:not_passed` and its pre-measurement `StageContractError`; it is not the accepted v3 result.
- The [80.27 remote blocker log](../diagnostics/80-27-smolvla-v1-remote-capture.log) is 2,339 bytes, SHA-256 `50b97cd9a027bf7c371e3d5fe692de5810d18dae386d74a52d9de9dc15537b55`; the [768-dimension probe](../diagnostics/80-27-smolvla-v1-dim768-probe.json) is 78 bytes, SHA-256 `89862b4051c94b1b34386041f36d8556438a8b557098540e10e01fed0a3a12bb`. Both copies retain their previously recorded source bytes. They document the v1 `(64,768)` versus frozen `(64,1024)` mismatch and detector undersampling; no SmolVLA diagnosis, new remote run, peak-memory result, or GPU feasibility claim is made.
- The frozen encoder-v2 manifest and prospective SmolVLA-v2 manifest are revision-backed and byte-protected. A fresh-clone `validate_manifest` check passed for both. Encoder v2 raw SHA-256 is `0fee3b6e03ba1289a34cdaaefed0563674d1079a853f2d238b7abf3cad94e947`, canonical digest `118a1f1380f7bb462657b5c33972e49c7ee15c3f7b67af75903090f52b432f0a`; SmolVLA v2 raw SHA-256 is `af5fa26713531d0dafa4a98a33c9c7392b36eaa742ff7f5cb551a50b5c5239f3`, canonical digest `ec70d423fcf33a6ebbaabfb2dccbcac6ee8de167f53a125bc377386f6cd1cf1`. SmolVLA v2 remains an unexecuted, non-gating specification.
- `docs/INDEX.md` exposes guide entry 29 and report entry 30. The stale unpublished dose-response-summary citation was removed; the report now states only that dose-response is not applicable to the single-strength core interventions and that no core dose-response measurement is claimed.

## Published scope and byte audit

The 23-path package at `9d86407` comprises:

- `.gitattributes`; `docs/INDEX.md`, `docs/PLAN.md`, `docs/sprint-plans/sprint-80.md`, and `docs/SPRINT_80_DEPTH_EVIDENCE.md`.
- `artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v2_injected.json` and `artifacts/benchmark_manifest_sprint80_smolvla_secondary_v2.json`.
- `artifacts/diagnostics/proof-80-23-v3-model-weight-lesion/{proof-run.json,model-lesion.json,model/lesioned_encoder.npz}` and all four accepted/rejected 80.24 stability run/report files under `artifacts/diagnostics/proof-80-24-stability-v1/`.
- `artifacts/diagnostics/80-27-smolvla-v1-{remote-capture.log,dim768-probe.json}`.
- Task summaries for 80.23, 80.24, 80.27, and 80.28; task handoffs for 80.23, 80.24, and 80.27.

The clean Windows checkout contained those 23 paths totaling **941,794 bytes** (including normal Markdown CRLF conversion). Eleven raw/content-addressed files totaling **693,441 bytes** were protected by narrow `-text` rules; all 11 reported `text: unset` under Git attributes and matched the recorded raw hashes or CAS filenames. The four 80.24 content-addressed run/report files matched their filename digests exactly. No large model, dataset, cache, or unrelated build output was added.

- The other protected 80.23 first-attempt bytes also matched in the clean clone: `proof-run.json` SHA-256 `4c49e6fcfd02d4a61bae34b79f739b1a09facd68977a19e53d6f2ae3e395f1b7`, `model-lesion.json` SHA-256 `d1b0763ee791b0069dc94e13b698edb51f4fcd7711ff169ce22264c3130d0570`, and the 9,198-byte `lesioned_encoder.npz` SHA-256 `81a29cc8d92b38164b108958764a6f1ffe07e5b822701077bd45faeffea91008`.

## Verification

- **Fresh Windows clone:** `git -c core.autocrlf=true clone --local --no-hardlinks F:/ai-ml/latent-anything F:/ai-ml/s80-task29-verified-p1mPWz`, then `git -C F:/ai-ml/s80-task29-verified-p1mPWz config core.autocrlf true`. The checked revision was `9d86407a3b56ab634eb7228078a51d6df828bdc7`; `core.autocrlf` read back `true`, and Git status remained clean after validation/build.
- **Strict docs build:** `uv run --locked --extra docs mkdocs build --strict --site-dir F:/ai-ml/s80-task29-verified-p1mPWz-docs` — **PASS**, documentation built in 72.18 seconds. Only the upstream Material for MkDocs 2.0 advisory was emitted.
- **Clean-clone relative-link audit:** 16 published docs/summaries/task records including the guide; 248 local Markdown links across 138 unique targets, **0 missing**. Both committed AI-engineer example scripts were present in the clone.
- **Manifest validation:** in the clone, `uv run --locked --extra docs python -c "import json; from pathlib import Path; from latent_anything._benchmark_manifest import manifest_digest, validate_manifest; p=Path('artifacts'); names=('benchmark_manifest_sprint80_encoder_autoencoder_v2_injected.json','benchmark_manifest_sprint80_smolvla_secondary_v2.json'); [(lambda x,n: (validate_manifest(x), print(n, 'validated', manifest_digest(x), 'declared='+x['commitment']['manifest_sha256'])))(json.loads((p/n).read_text(encoding='utf-8')),n) for n in names]` — **PASS**; both schemas validated and canonical digests equaled their declarations.
- **Evidence-byte audit:** all 4 stability CAS filenames matched raw SHA-256; both remote blocker files and both v2 manifests matched their committed raw hashes; the 80.23 failed-attempt files were byte-preserved; the original v2 `threshold:null` observation and separate passing stability thresholds were inspected in committed artifacts.
- No project-wide tests, lint, formatting, type-check, replay, model acquisition, graph refresh, or remote/CUDA run was performed for this docs/evidence-only task.

### Fourth-review correction audit

- **Correction revision:** `4a5e8f8cc891abfd6ec671e3c9f3444b871149f5` (`docs(sprint-80): reconcile API inventory and memory records`). `git show --stat` reported 4 files changed, 233 insertions, 6 deletions: `.agents/memory/decisions.md`, `.agents/memory/lessons-learned.md`, `docs/PLAN.md`, and `docs/SPRINT_80_DEPTH_EVIDENCE.md`. Only these exact paths were staged.
- **Clean Windows revision:** `git -c core.autocrlf=true clone --local --no-hardlinks F:/ai-ml/latent-anything F:/ai-ml/s80-29-doc-audit-4a5e8f8`, then `git -C F:/ai-ml/s80-29-doc-audit-4a5e8f8 config core.autocrlf true`. `git -C ... rev-parse HEAD` returned `4a5e8f8cc891abfd6ec671e3c9f3444b871149f5`; `git status --short` produced no output both before and after the docs build.
- **Snapshot/report consistency:** In that clean revision, `docs/PLAN.md` preserves the historical 205/202/7 values and `048ac553...` digest while naming 214/211/8 and `3fd8c73e...` as current. `artifacts/api_freeze_snapshot_0.9.0.json` independently records `current_count: 214`, `canonical_stable_count: 211`, and the same current digest. The depth-evidence report identifies `agent://DeepReviewSprint80Fourth`, D1/D2, and the task-29 handoff without asserting final signoff.
- **Strict docs and links:** In the clean clone, `uv run --locked --extra docs mkdocs build --strict --site-dir F:/ai-ml/s80-29-doc-audit-4a5e8f8-site` — **PASS**, documentation built in 50.34 seconds. The strict build emitted no project documentation/link errors; it emitted the upstream Material for MkDocs 2.0 advisory and the uv cache hardlink fallback warning only. This validates the published docs at the clean correction revision; the earlier 248-link audit remains prior evidence and was not rerun as a separate parser here.
- No source code changed. No source-graph refresh or project-wide tests, lint, formatting, type-check, replay, or remote/CUDA run was performed.


## Graph status

`graphify-out/graph.json` exists. The latest code-graph update recorded by task 80.28 has 16,403 nodes, 37,788 edges, and 1,117 communities. This task changes Markdown and small evidence files only; the graph rule does not require a source-graph refresh or semantic Markdown reindex for this scope.

## Remaining gate and exclusions

- The fourth review's D1/D2 corrections are revision-backed as recorded above. Main must still obtain a fresh sprint-wide deep review; neither this record nor prior task-level PASS results assert final deep-review PASS or Sprint 80 signoff.
- Sprint 80 remains **Active** and task 80.29 remains `[~]`. The Sprint 81 handoff is **READY only for the bounded accepted ordinary-DL core**.
- The full 2,549-test selection was measured in the shared worktree, not rerun in a clean clone. Optional-extras results are prior evidence; `network`, `large_download`, `viz`, `integration`, remote CUDA/GPU, broad stable-depth, and SmolVLA diagnosis/feasibility remain unverified, blocked, or excluded as stated in the report.
