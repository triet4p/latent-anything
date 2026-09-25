# Task 80.27 — Bounded SmolVLA Secondary Lane: truthful BLOCKED result

Status: **BLOCKED** — one bounded real CUDA proof executed at the exact pushed
revision; it fails closed on an actionable representation-geometry defect, so no
validator-clean artifact exists and the sprint checkbox stays `[ ]`.

## Verdict

- Acceptance: **NOT PASSED** — no bounded secondary artifact persisted.
- Sprint checkbox: `80.27` remains `[ ]` with the explicit blocker below.
- Secondary lane does not gate the ordinary-DL core (per sprint plan).

## Exact blocker

The committed proof asserts a hardcoded vision-context width of 1024, but the
real pinned policy (`lerobot/smolvla_libero@31d453f7`) emits per-camera
`vision_context` captures of shape `(1024 tokens, 768 features)` on CUDA, so the
64-row mean-pooled capture matrix is `(64, 768)`. `run_proof` aborts at
`adapter_case` with:

```text
AssertionError: captured matrix shape (64, 768) != (64, 1024)
```

This is an actionable code defect in the proof script (wrong hardcoded width),
compounded by a deeper representation-geometry blocker: the frozen collapse
detector requires `n_samples > dim` (`_batch_matrix` raises `DetectionError`
for `64 samples / 768 dims`), proven on the same box with synthetic 64x768
input: `undersampled batch: 64 samples cannot support rank estimation over 768
dims`. Fixing only the width constant would walk straight into that detector
refusal. A correct fix must re-predeclare the representation (width, axis
selection, and sample budget) — it cannot be a silent in-place constant edit.

## Source identity (all three equal)

- Local HEAD: `24850717ca3f55afbd8d19dc28aeca587744f996`
- GitHub `origin/main`: `24850717ca3f55afbd8d19dc28aeca587744f996`
  (`git ls-remote origin main` matches local `git rev-parse HEAD`)
- Disposable remote clone HEAD: `24850717ca3f55afbd8d19dc28aeca587744f996`
  (`REMOTE_CLONE_HEAD`, asserted equal before install/proof)
- Persistent remote checkout `/home/trietlm/latent-anything`: HEAD
  `24850717ca3f55afbd8d19dc28aeca587744f996`, `git status --porcelain` clean
  (0 entries) after the run. It was used only by the prior setup-only job
  (`git pull --ff-only` + `uv sync` + verification); the proof never executed
  from it.

## Remote environment

- Host: `trietlm@192.168.30.244` (`di-server`), home `/home/trietlm`
- GPU: NVIDIA GeForce RTX 4060 Ti, 16380 MiB total, ~170 MiB used at idle,
  driver 580.126.20
- Disposable venv (fresh clone): CPython 3.13.12, torch 2.10.0+cu128
  (CUDA build 12.8, `cuda_ok True`), lerobot 0.6.1, transformers 5.5.4,
  numpy 2.2.6
- Disk: 96% used, 39 GB free after cleanup (27–28 GB free during the run)

## Proof command, exit, timing

- Command (detached, disposable clone only):

```bash
cd /home/trietlm/remote-80-27-proof/repo
export UV_CACHE_DIR=$TMPD/uv-cache HF_HOME=$TMPD/hf-cache \
  HUGGINGFACE_HUB_CACHE=$TMPD/hf-cache/hub \
  TORCH_EXTENSIONS_DIR=$TMPD/torch-extensions TMPDIR=$TMPD/tmp
nohup .venv/bin/python scripts/sprint80_task80_27_smolvla_proof.py \
  --device cuda > $TMPD/run1.log 2>&1 &
```

- Launch: 2026-09-22T16:52:52Z; failure recorded 2026-09-22T16:58:01Z
  (~5–6 min wall: dataset fetch + ~900 MB policy download + one 64-query
  capture pass).
- Exit: nonzero (traceback, no `EXIT=`/`WALL=` trailer — the script raises
  before its summary lines).
- Peak memory: no `resources:` line was emitted (abort precedes it); GPU was
  idle at 170 MiB after the failure; the 16 GiB ceiling was never approached.
  No peak-memory claim is made.

## Scientific outcome (grounded, operational vs scientific)

- Operational: data stage **PASS** — 64 real frames from episodes 0–3
  (scanned 874 items; selection digest `0ee63f53aa9e6fba…` matches the
  predeclared commitment). Policy load reached the real pinned checkpoint.
- Scientific: **no diagnosis was produced** — the run never reached
  detect/localize/explain/intervene/compare/report. Nothing about collapse,
  rank, or spread on this checkpoint is established by this run.
- Dim-agnostic follow-up probe on the same box (same 64 frames, same policy,
  no manifest changes): capture matrix `(64, 768)`, naive SVD effective rank
  17.90 (informational only — not the frozen estimator), frozen-estimator
  spread ≈ 5.86e-17, frozen health effective rank 3.17. These numbers are
  diagnostic context only, not a workflow result.
- Detector guard probe (synthetic 64x768, same frozen config): `DetectionError:
  undersampled batch: 64 samples cannot support rank estimation over 768 dims`.
  The frozen detector cannot adjudicate 64 samples at width 768 by contract.
- Pins verified: model blob sha256
  `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`
  (906,712,520 bytes, matches manifest); dataset selection digest
  `0ee63f53aa9e6fba…` (runtime log line). Licenses: Apache-2.0 for checkpoint,
  dataset, and LeRobot (per manifest/model card; no license files copied).

## Limits

64 frames / four episodes (0–3; train 0–1, eval 2–3) / single seed pair
(42 eval, 17 controls) / single device (CUDA, RTX 4060 Ti) / no simulator /
single run. The secondary lane never gates the ordinary-DL core.

## Evidence paths, sizes, checksums

Copied back (review-sized only; verified byte-identical after `scp`):

- [Remote capture log](diagnostics/80-27-smolvla-v1-remote-capture.log) — 2,339 bytes, SHA-256 `50b97cd9a027bf7c371e3d5fe692de5810d18dae386d74a52d9de9dc15537b55`; the repository copy is byte-identical to the previously cited `F:/ai-ml/sprint80_27_remote_run1.log` (remote `/home/trietlm/remote-80-27-proof/run1.log`, removed with the disposable clone).
- [64×768 detector probe](diagnostics/80-27-smolvla-v1-dim768-probe.json) — 78 bytes, SHA-256 `89862b4051c94b1b34386041f36d8556438a8b557098540e10e01fed0a3a12bb`; the repository copy is byte-identical to the previously cited `F:/ai-ml/sprint80_27_dim768_probe.json` (remote `/home/trietlm/remote-80-27-proof/dim768_probe.json`, removed).
- Pre-existing local references (not completion evidence):
  `F:/ai-ml/sprint80_27_run1.log` (interrupted local CPU run),
  `F:/ai-ml/sprint80_27_remote_setup.log` (setup-only job tee)

Models, datasets, caches, and bulk outputs were kept remote and uncommitted;
no checkpoint binaries, datasets, or bulk data are committed.

## Cleanup status

- Disposable remote checkout + caches `/home/trietlm/remote-80-27-proof`
  (repo, `.venv`, uv-cache, hf-cache, torch-extensions, tmp, logs): **removed**
  (`rm -rf`, verified absent). No leftover path.
- Persistent remote checkout `/home/trietlm/latent-anything`: untouched by the
  proof, clean (0 entries) at the same SHA.
- Retained remote data (explicitly documented, not proof outputs): prior
  setup-only caches in `/home/trietlm/remote-80-27/uv-cache` (1.8 GB);
  `~/.cache/huggingface` (845 MB, pre-existing).
- Local: no 80.27 proof/model process remains (only two `pyright-langserver`
  `python.exe` PIDs, unrelated). Local CPU runs remain stopped; nothing was
  staged, committed, reset, or cleaned in the main worktree (88 unrelated dirty
  entries preserved). Clean local worktree `F:/ai-ml/latent-anything-80-27-clean`
  at the exact SHA retained for inspection (detached HEAD, 0 dirty).
- No unresolved local/remote proof process remains.

## Changed files (this task, local repo)

- NEW `artifacts/task_80.27_smolvla_secondary_lane_summary.md` (this file)
- Sprint plan `docs/sprint-plans/sprint-80.md`: 80.27 checkbox left `[ ]`;
  blocker note appended under Notes / Blockers.
- No code, manifest, or test changes (no fix-commit cycle was run: the defect
  spans the proof constant, the predeclared manifest width, and the frozen
  detector sample budget, so it needs a re-predeclaration decision, not a
  silent patch).

## Sprint status

- `80.27` — `[ ]` BLOCKED (explicit blocker recorded in-plan).
- `80.28` — not started (out of scope for this worker).

## Continuation — 2026-09-24

This dated addendum preserves the original CUDA attempt and its failed outcome above; it does not revise the v1 manifest, capture, metrics, or historical source identity. The current task record is [`artifacts/sprint-80/task-27.md`](sprint-80/task-27.md).

- Re-checked the frozen v1 manifest and proof geometry. The manifest remains valid with canonical digest `cb45b0e90c8a30def0a34294a9e19665bf9204f262a346d608a70712549df8bc` and declares 64 frame rows × 1,024 features. The real capture remains `(64, 768)`; the detector guard rejects 64 samples for 768 dimensions. Width-only correction remains insufficient.
- Re-checked upstream pinned-source license declarations: the checkpoint model card at revision `31d453f7edd78c839a8bbc39744a292686daf0de`, dataset card at revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`, and LeRobot 0.6.1 PyPI metadata each declare Apache-2.0. This is not a legal/redistribution review.
- The 16-GiB RTX 4060 Ti completed the historical 64 serial real policy captures before the shape assertion, but no peak-memory line was recorded and no diagnostic workflow stage ran. This does not prove a corrected proof's resource feasibility.
- A new prospective v2 manifest (`artifacts/benchmark_manifest_sprint80_smolvla_secondary_v2.json`) is digest-locked at `2026-09-24T05:48:12Z` (SHA-256 `ec70d423fcf33a6ebbaabfb2dccbcac6ee8de167f53a125bc377386f6cd1cf1f`) with a fixed NumPy 2.2.6 PCG64/QR 768→32 orthonormal projection, applied to the same 64 frame samples, pins, thresholds, and episode split. It was predeclared before any fresh capture. The claim is limited to the projected representation, not the full 768-feature spectrum; these 64 frame samples remain nested in four episodes, and bootstrap is still at the frame/sample unit, so no episode-level or population uncertainty is claimed. This v2 has not been wired to the proof script or measured; the original v1 and its failed outcome remain immutable.
- At the pre-run clean-worktree check, Git reported 30 modified files and 97 untracked entries, including the proof script, plan, and core sources. `skill://remote-cuda-test` requires stopping on a dirty worktree; no push, SSH, disposable clone, or new CUDA run was attempted. No files were staged, committed, reset, or cleaned.
- The focused local check passed: v1 and v2 both validate, their computed digests match locked commitments, and model/data pins and thresholds match. The preserved remote log/probe SHA-256 values match the checksums recorded above. Upstream license cards and PyPI metadata were read at the pinned revisions/version.
- Task 80.27 remains **BLOCKED**, with the sprint row left `[~]` for Main's review. The secondary SmolVLA lane does not gate the ordinary-DL core. No product code, v1 manifest, tests, or plan row changed in this continuation; the prospective v2 manifest was added as a distinct input.
