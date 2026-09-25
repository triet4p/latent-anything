# Sprint 80 Task 80.27 — SmolVLA secondary lane

**Status:** BLOCKED; leave the sprint row `[~]` for Main's evidence review. This outcome does not gate ordinary-DL core work (80.23–80.26).

## Result

No new CUDA proof or validator-clean diagnostic run artifact was produced. The preserved v1 proof attempt remains a real, bounded CUDA capture that failed closed before diagnosis; no claim about SmolVLA collapse is supported. A second run was not attempted because the current checkout is dirty and the required remote-CUDA workflow requires stopping on a dirty local worktree. No files were staged or committed, and no local CPU substitute was run.

The v1 manifest and historical measurements remain unchanged. A separate prospective v2 manifest was digest-locked at `2026-09-24T05:48:12Z`, before any new measurement, with canonical SHA-256 `ec70d423fcf33a6ebbaabfb2dccbcac6ee8de167f53a125bc377386f6cd1cf1f`. It freezes the same 64 pinned frame samples as a 32-dimensional orthonormal Gaussian sketch of the observed 768-feature capture: NumPy 2.2.6 PCG64 seed `20260924`, reduced QR, canonical column signs. Thus the detector receives 64 rows × 32 features (`64 > 32`) without increasing the frame count; model/data pins, sample selection, metrics, and threshold values remain unchanged. The predeclared result is explicitly limited to this transformed representation, not the raw 768-dimensional spectrum. The v2 file is not wired to the current proof script and has not been executed; temporal/episode limitations remain explicit.

## Pinned inputs, license, and resource evidence

- **Checkpoint:** `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`; v1 manifest model artifact SHA-256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`. The historical real run recorded 906,712,520 bytes and loaded the pinned policy. The pinned model card declares Apache-2.0: <https://huggingface.co/lerobot/smolvla_libero/resolve/31d453f7edd78c839a8bbc39744a292686daf0de/README.md>.
- **Dataset:** `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`; frozen v1 selection digest `0ee63f53aa9e6fba04b79b382297bfa235a2df6d6544e3402862fe69689d4b51`, covering 64 even-indexed frames from episodes 0–3, with episodes 0–1 train and 2–3 eval. The previous runtime log records a successful real selection after scanning 874 items. The pinned dataset card declares Apache-2.0: <https://huggingface.co/datasets/lerobot/libero/resolve/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4/README.md>.
- **Runtime license:** the prior remote environment used LeRobot 0.6.1. PyPI metadata for 0.6.1 declares Apache-2.0: <https://pypi.org/pypi/lerobot/0.6.1/json>. These are the upstream declared licenses, not a redistribution/legal review.
- **16-GiB CUDA evidence:** the prior isolated run used an NVIDIA RTX 4060 Ti (16,380 MiB) with CUDA-enabled torch and reached the end of all 64 serial policy queries/captures, then failed at the post-capture shape assertion. This demonstrates that that specific capture workload loaded and ran on the device. Its log emitted no peak-memory/resource summary, and the seven-stage workflow never ran; therefore it does **not** establish a peak-memory bound or feasibility for a corrected, rank-valid workflow.

## Frozen failure and current blockers

- Immutable v1 manifest: `artifacts/benchmark_manifest_sprint80_smolvla_secondary_v1.json`; manifest id `sprint80-secondary-smolvla-vision-collapse-v1`; declared/canonical SHA-256 `cb45b0e90c8a30def0a34294a9e19665bf9204f262a346d608a70712549df8bc`. Its declared geometry is 64 frame rows × 1,024 features.
- Prospective `artifacts/benchmark_manifest_sprint80_smolvla_secondary_v2.json` is schema-valid with canonical SHA-256 `ec70d423fcf33a6ebbaabfb2dccbcac6ee8de167f53a125bc377386f6cd1cf1f`. It versions the representation and interpretation to a fixed NumPy 2.2.6 768→32 projection while preserving the v1 model/data/sample pins and threshold values. This declaration is not evidence that the projection or full workflow has run.
- The previous attempt's operational scope was dataset fetch, pinned policy load, and capture; it stopped before detect/localize/explain/intervene/compare/report. Historical elapsed time was about 5–6 minutes including fetch and model download. Peak memory was not recorded.
- At the pre-run clean-worktree check, Git reported 30 modified files and 97 untracked entries (no staged entries), including the proof script, sprint plan, and core source files. The proof script was itself modified. Per `skill://remote-cuda-test`, a dirty local worktree is a stop condition. No push, SSH run, disposable clone, or cache operation was attempted. This preserves concurrent/user changes and prevents claiming a new source identity.

## Changed files

- `artifacts/sprint-80/task-27.md` — this task evidence record.
- `artifacts/benchmark_manifest_sprint80_smolvla_secondary_v2.json` — prospective locked geometry/sample declaration; the v1 manifest is unchanged.
- `artifacts/task_80.27_smolvla_secondary_lane_summary.md` — appended a dated continuation; all historical run details above the addendum are preserved.
- `graphify-out/` — incremental graph update; exact completed result recorded below.
- No product code, frozen v1 manifest, tests, or sprint-plan row was changed.

## Focused verification

- `uv run --no-sync python -c "import json; from pathlib import Path; from latent_anything._benchmark_manifest import manifest_digest, validate_manifest; root=Path('artifacts'); a=json.loads((root/'benchmark_manifest_sprint80_smolvla_secondary_v1.json').read_text(encoding='utf-8')); b=json.loads((root/'benchmark_manifest_sprint80_smolvla_secondary_v2.json').read_text(encoding='utf-8')); validate_manifest(a); validate_manifest(b); assert a['model']==b['model']; assert a['dataset']==b['dataset']; assert a['thresholds']==b['thresholds']; assert b['representation']['axes'][1]['selection']=='768-to-32-numpy2.2.6-PCG64-20260924-standard-normal-reduced-QR-canonical-sign'; assert 64>32; print('v1_valid=True v2_valid=True'); print('v2_digest='+manifest_digest(b)); print('samples=64 features=32 n_gt_dim=True'); print('thresholds_unchanged=True model_and_dataset_pins_unchanged=True')"` — **passed**; v1/v2 validate, v2 sample budget is 64 over 32 features, with model/data pins and thresholds unchanged.
- The small remote blocker inputs are now revision-backed: [capture log](../diagnostics/80-27-smolvla-v1-remote-capture.log) — 2,339 bytes, SHA-256 `50b97cd9a027bf7c371e3d5fe692de5810d18dae386d74a52d9de9dc15537b55`; [dimension probe](../diagnostics/80-27-smolvla-v1-dim768-probe.json) — 78 bytes, SHA-256 `89862b4051c94b1b34386041f36d8556438a8b557098540e10e01fed0a3a12bb`. Their bytes were copied unchanged from the previously recorded `F:/ai-ml/` files. No new remote proof was run.
- Read the model and dataset cards at their pinned revisions and LeRobot 0.6.1 PyPI metadata; all declare Apache-2.0.
- No remote proof, full suite, lint, formatter, or build was run. The prior historical remote run is not represented as a new verification.

## Graph update

- The first incremental attempt failed before merge with HTTP 429 quota errors. After recovery, the final `graphify . --update --no-viz` completed after the v2 evidence edits: 0 code / 231 documentation files detected changed, 1,514 unchanged; 218 semantic files were dispatched in 3 completed chunks (185,816 input / 5,736 output tokens; estimated $0.1101). It wrote `graph.json` with 16,335 nodes / 36,654 edges before clustering. `graphify cluster-only . --no-viz --no-label` completed at 1,111 communities and refreshed `GRAPH_REPORT.md`/`graph.json`; 824 communities were deterministically relabeled by hub, without LLM label refresh. Extraction warnings: 194/218 dispatched files produced no nodes, 24 out-of-scope nodes were dropped, and one edge lacked `source_file`, so semantic coverage remains partial. Visualization was skipped; `graphify` removed the old `graph.html` rather than generating a visualization for the >5,000-node graph.

## Unresolved findings

1. No validator-clean run artifact exists; the secondary lane remains explicitly blocked and non-gating for ordinary-DL core.
2. The v2 geometry is only predeclared. The current proof script is not wired to it; no corrected diagnostic, inference, or output artifact exists.
3. The current dirty worktree blocks the mandatory clean-commit → push → exact-SHA disposable-clone remote-CUDA sequence. Main must review/resolve the worktree state before any such run.
4. Full corrected-workflow GPU peak memory and runtime on the 16-GiB device remain unmeasured.
