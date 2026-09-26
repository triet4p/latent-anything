# Task Summary: Sprint 80 Task 80.25 — committed accepted-workflow clean replay

**Sprint:** Sprint 80 — Stable Depth
**Task:** 80.25
**Execution status:** The corrected committed-driver replay **PASSED** on 2026-09-24: 24/24 checks, zero failures, and three byte-exact output roots. The sprint row remains `[~]` pending Main's evidence review.

This correction replaces the failure-only committed entrypoint with the accepted encoder-v3 and transformer target-evidence-v2 workflows. It also runs the independently frozen stability supplement in a separate proof root. Prior temporary-driver outputs, the initial failed finalization, and historical negative runs remain distinct and unchanged.

The detailed [task handoff](sprint-80/task-25.md) links the full evidence bundle, output reports, exact pins, commands, and resource record. No frozen manifest, checkpoint, target rule, threshold, lockfile, or scientific input was changed.

## Current committed-driver replay (2026-09-24)

Command: `uv run python scripts/sprint80_task80_25_clean_repro.py --hf-root F:/llms/hf/models`. Run `20260924-160321-176370-23440`; driver SHA-256 `89fb65243baab0b7a2c2f25dc2920e82bb5ad195dca2255299017ffafc210908`. Evidence files are in [`task_80.25_evidence_committed_20260924-160321-176370-23440/`](task_80.25_evidence_committed_20260924-160321-176370-23440/).

| Case | Run / artifact SHA-256 | Report SHA-256 | Render SHA-256 | Wall / peak process-tree RSS |
|---|---|---|---|---|
| Encoder v3 | `ade7b3144207bb88` / `fedfe97792d8f08fb5a343e2d259109000de4a79255efcb34442098d257e1f93` | `8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194` | `9e4d4127e3815b55d959cfc8bcb5f69f0ff4e369d9bfc82175b51b3c8454532b` | 7.498 s / 354,619,392 B |
| Transformer v1 + target evidence v2 | `2765d61287933ecb` / `f80c4ab78ee6ede0b760299bb3239832ddbc81bf69b4dfc9abe90b200d954157` | `c670acea7b27665581844916f17fb03aa38b5aa6d7740583dea34a5a6f467bdb` | `505002cd45d4a53646c29f70fd1414cef4e927262b123753628b3388857baa81` | 1,013.704 s / 1,837,969,408 B |
| Independent stability supplement | run-record SHA-256 `a6428146d414129e9a34a2ebb9b93f1556a18b6d32fe339a18e51816a6d574c5` | `5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0` | — | 1,054.484 s / 1,700,847,616 B |

The fresh root copied 1,631 files / 28,869,588 bytes, installed non-editably via `uv sync --frozen --no-dev --extra transformers --no-editable`, applied `datasets==3.6.0` and `huggingface-hub==0.35.3`, and passed `uv pip check` (64 packages). Python 3.13.3 / uv 0.9.7 / CPU-only PyTorch 2.10.0; no HF auth token was passed. Each case had 1,800 s / 16 GiB process-tree caps; fresh runtime caches/root were removed after output copying.

**Exact provenance:** v3 manifest `e5bcd341b1b746f05e4ac02f908bcd27f23bace39737c86c8ac38888e7451b82`, baseline checkpoint `92829e57aaaebf52d5446d1a9c9db5de4e12c1657fad5ecc986329ed8752bb2a`; transformer manifest `339c5c302def9d8b859e1914101803c84905347eec3caa1d596a5eadc015d0a2`, target record `3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19`, target rule `9b12485635decf738c4904702f75d7d5ae8cb6663555e2af5ee66476269f025b`. GPT-2 snapshot revision `e7da7f221d5bf496a48136c0cd264e630fe9fcc8` (6 files), WikiText revision `f776294184f13b8ff2337b3841cf9269a6216d1e` (4 files); exact hashes are in `pins.json`, verified pre/post. The package source tree pin covers 162 Python files, SHA-256 `5dc8a1922fe39893b218c9a80aa78c61e4f09a2e09e552febd53dbb0cccb1a89`.

The seven-stage transformer replay returned pooled-state SHA-256 `b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b`, held-out accuracy `1.0`, and leakage gap `0.44889779559118237`, matching the accepted reference. The separate stability replay retained its frozen protocol SHA-256 `a69b7152a800ac543ac76fec4d699c16681517708a95b353e5d241241c8acf0a`; held-out accuracy `1.0`, coefficient stability `0.9748782295192386` (threshold `>=0.8`), leakage gap `0.4254901960784314` (threshold `>=0.15`), all controls passed. It remains independent of the transformer v2 seven-stage run and does not overwrite Task 80.24 outputs.

**F4 sign:** the restoration intervention effect is `+0.4638888888888889`; the separate candidate-minus-baseline task comparison delta is `-0.4638888888888889`. Current committed proof sources were v3 `87891b0c75d4ddcfe890792e421ac4108801d08aa49a9d3a234da764481cb53f`, transformer target-v2 `dee2d016bf9a22dfd5ef9089ca7bea74bc28bf86d6f36f78ae28db505f8d6f7a`, stability `22cd2b4f8b9db1bb0d41600ee8c761251dd317ac9d654940bfd21f84242954b8`; prior temporary-driver hashes and historical variance remain below.

**Failed first committed-driver attempt retained:** run `20260924-151256-484003-10236` recorded `NOT PASSED` because finalization referenced `fresh_post_pins` before assignment. All three case validators passed, but no outputs were copied before cleanup. The full failed record remains at [`task_80.25_evidence_committed_20260924-151256-484003-10236/summary.json`](task_80.25_evidence_committed_20260924-151256-484003-10236/summary.json). Moving post-run pin verification before that check fixed the exception; only the subsequent 24/24 replay is accepted evidence.

## Historical temporary-driver clean replay (2026-09-24)

**Historical paired result:** `PASSED`, 19 checks, zero failures; `cleanup.removed=true`; original negative-evidence directories preserved. The temporary root copied 1,404 files (24,725,021 bytes), installed non-editably, and produced the accepted v3/v2 outputs listed below. This replay used uncommitted inline proof code and is not evidence for the current committed-driver source.

| Case | Fresh run | Artifact SHA-256 | Report SHA-256 | Render SHA-256 | Wall / peak process-tree RSS |
|---|---|---|---|---|---|
| Encoder v3 model-weight lesion | `ae320a82c7db5f8b` | `ed4b0bbb1ac91a92bd6e4f388f11f85cc0e3cf105edbc97fbe1773e8b4e4f681` | `8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194` | `fe3e1444db1e5888ae3eb415f2c1e99ebe4e113ef19da7a5af748ec8fbe8ded5` | 15.563 s / 337.2 MiB |
| Transformer v1 + target evidence v2 | `c4012406e89254ea` | `972babae550002d74fa68f2cfae107caa21bf4350d38c99ac2ad97684a6879ba` | `eac5228f809d4127ba2cbf9c216be4e643ac4c37a26bb77f4a5adb70b0609c7b` | `930a2082d9fa297c1b4f90f5c7858dbea65d4f8f6085fc6048be5f4294360aad` | 983.945 s / 1,702.1 MiB |

Each proof had a 1,800-second timeout and 16-GiB process-tree RSS ceiling. Ten repository pins and all ten immutable GPT-2/Wikitext snapshot files matched before and after; transformer caches were fresh, and both output trees were verified byte-exact. Full pin and inventory evidence is in `summary.json`, `checks.json`, and `runs.json`.

Encoder v3 used manifest SHA `e5bcd341b1b746f05e4ac02f908bcd27f23bace39737c86c8ac38888e7451b82` / commitment `f823efc1b016ca9ad62c0f207e17d138947040a39f5d1c1f42ab6d0d65f5954f`; lesion checkpoint SHA `81a29cc8d92b38164b108958764a6f1ffe07e5b822701077bd45faeffea91008` and lesion record SHA `d1b0763ee791b0069dc94e13b698edb51f4fcd7711ff169ce22264c3130d0570` match the accepted reference. The independent validator passed and the seven-stage report reproduced the reference report digest. Output: `diagnostics/proof-80-25-clean-encoder-v3-20260924-035357-6996/`.

Transformer v1 used manifest SHA `339c5c302def9d8b859e1914101803c84905347eec3caa1d596a5eadc015d0a2` / commitment `c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a`; the report confirms `diagnostic-evidence-v2`, target record SHA `3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19`, and target rule SHA `9b12485635decf738c4904702f75d7d5ae8cb6663555e2af5ee66476269f025b`. The validator passed, seven stages completed, 13 evidence links were rehashed, and the content-addressed output contains 15 blobs. Output: `diagnostics/proof-80-25-clean-transformer-v1-target-v2-20260924-035357-6996/`.

**Historical cross-run variance:** the temporary-driver leakage gap `0.45090180360721444` differed from accepted `0.44889779559118237` by `0.002004008016` (one evaluation row of 499). The corrected committed replay returned the accepted reference value exactly. Both results remain recorded; no threshold was tuned.

## Legacy failure-only committed driver and original blocked attempts (historical)

At the earlier checkpoint, the committed executable `scripts/sprint80_task80_25_clean_repro.py` ran only the historical negative cases and expected `ACCEPTANCE: NOT PASSED`. The behavior and outputs documented below are historical; the current committed entrypoint has been cut over to accepted v3/v2 cases with the separate stability supplement. Original negative evidence remains unchanged.

1. **Pinned-input preflight (before any execution):** SHA-256 verification of
   12 repository inputs (5 frozen Sprint 80 artifacts + L04 wikitext manifest +
   `conv_vae_heldout_benchmark.json` + both proof entrypoints + `pyproject.toml`
   + `uv.lock` + `.python-version`) against embedded pins re-derived from the
   80.23/80.24 recorded hashes, plus hash-verification of the declared
   immutable external snapshots: gpt2 `e7da7f221d5bf496a48136c0cd264e630fe9fcc8`
   (6 files) and wikitext `f776294184f13b8ff2337b3841cf9269a6216d1e` (4 files)
   under the declared `HF_HOME` root `F:\llms\hf\models` — re-verified again
   after all runs.
2. **Fresh root:** copies exactly `pyproject.toml, uv.lock, .python-version,
   README.md, LICENSE, src/, scripts/, artifacts/, tests/` into a fresh temp
   root (1395 files, 23.4 MiB), excluding `.git`, `.venv`, `__pycache__`, and
   `*.egg-info`; a baseline SHA-256 tree inventory is diffed against a final
   inventory (only setuptools `build/` + `*.egg-info` build metadata may
   appear; any other change fails the run).
3. **Fresh environment from declared lock metadata:** `uv sync --frozen
   --no-dev --extra transformers --no-editable` (non-editable wheel install
   into `<fresh-root>/.venv`), then the exact 80.24-declared overlay
   (`datasets>=2.19,<4` + `huggingface-hub>=0.34,<1`) pinned to the recorded
   resolutions `datasets==3.6.0`, `huggingface-hub==0.35.3`, followed by
   `uv pip check`. Isolation probes assert: interpreter and
   `latent_anything.__file__` inside the fresh root only, no developer
   `src` on `sys.path`, non-editable install, and exact pinned versions.
4. **Fresh runtime state per run:** each of the four proof executions gets its
   own empty cwd (proving no cwd-relative output), its own temp directory
   (so the pooled hidden-state `.npz` cache is always cold and re-extracted),
   and its own fresh HF datasets/modules caches (arrow cache rebuilt from the
   pinned parquet snapshot each run); `PYTHONPATH`, `HF_TOKEN`, `HF_HOME`,
   user-site, and developer temp state are stripped; `HF_HUB_CACHE` points at
   the declared immutable snapshot; telemetry disabled; the developer's warm
   pooled cache in the OS temp dir is recorded before/after and proven
   untouched.
5. **Verification:** exact recorded stage-payload hashes, capture identities,
   blocker identity, `ACCEPTANCE: NOT PASSED`, no artifact directories
   materialized, peak-RSS bounds (16 GiB ceiling plus a 32 MiB sanity floor),
   cross-run identity of detect/localize/capture per case, and loud failure on
   any unexpected validator-clean claim. Bounded evidence (logs,
   `environment.json`, `runs.json`, `checks.json`, inventories, `summary.json`)
   is written to `artifacts/task_80.25_evidence/`; the temporary environment
   and caches are then deleted (deletion verified), and cleanup is recorded.

## Historical exact results (attempt 2, original blocked clean run)


```text
80_23_run0: exit 0; proof body 72.01s (cold model fit 70.40s), wall incl. imports 79.89s, tracemalloc peak 81.1 MiB
  PASS capture: capture_identity 16a5c11ae062d83ec489635f05c68df0b3c19acbd0bad65a8a11da3021bc2acd
  PASS determinism: detect payload sha faeff74e9ad7aa9185814ef94cbfa8a21afcda72e7fd8cb97709aa18f83bfc8a, localize payload sha 35c17b76f805453b36087776849c2a9131d92dce4a4f384bcfe388a9341ac8df
  BLOCKED explain: StageContractError — explain hypothesis 'h-collapse-bottleneck-mu' declares localization ('slice', 'slice-all') with no localized/supported prior slice selection carrying that identity; workflow failed at explain ...
  ACCEPTANCE: NOT PASSED —80.23 frozen acceptance cannot be satisfied truthfully
80_23_run1: exit 0; proof body 6.05s, wall incl. imports 10.61s, tracemalloc peak 62.1 MiB; detect faeff74e9ad7aa918…, localize 35c17b76f805453b…, identical BLOCKED explain message, ACCEPTANCE: NOT PASSED
80_24_run2: exit 0; proof body 948.87s (cold extraction+replay 911.86s), wall incl. imports 953.03s, tracemalloc peak 538.8 MiB
  PASS deterministic-replay: pooled hidden states sha b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b reproduced bit-identically (12 layers x 2048 rows x 768)
  PASS capture: capture_identity 28f4fbd2c47231fec8908044d335d06a87b2fda2f18bde1e3f0567dcb8c05d89
  RECORD positive-case: OBSERVED — heldout-probe-accuracy 1.0000 ≥0.7, probe-leakage-gap 0.4489 ≥0.15
  PASS determinism: detect payload sha 23863ec19820f0ee555da8dda345b7971e141036e01b24d0c8cf1c4edddcc2ce, localize payload sha 1129ac27a96fdbf79fff47e985779a7dcbd9c591e70319bedb0d2caceb00c2bc
  RECORD intervene: remove remove-section-header-direction target transformer.h.0 -> conclusion supported (baseline 1.0000, intervened 0.7315, effect -0.2685); all four controls passed
  RECORD compare: neither — heldout-probe-accuracy delta 0.000000, probe-leakage-gap delta 0.000000
  BLOCKED persistence: DiagnosticArtifactError: report failed independent validation; nothing was persisted: taxonomy applicability mismatch for separability_probe_leakage
  ACCEPTANCE: NOT PASSED — see recorded outcomes above; nothing was tuned to force a pass
  resources: python 3.13.3, numpy 2.4.6, transformers 4.57.6, torch-cpu, scikit-learn 1.9.0, datasets 3.6.0
80_24_run3 (second cold transformer run): completed; byte-identical `BLOCKED persistence` result and final harness verdict are recorded below.
```

Every stage payload hash, capture identity, threshold value, and blocker string
above is **byte-identical to the values recorded in the 80.23/80.24 evidence
artifacts**, which is the truthful reproduction result: the same pinned inputs
regenerate the same fail-closed outcomes from a cold, isolated install.

## Historical attempt 1 (preserved): environment-coupling defect found and fixed


The first full attempt (`artifacts/task_80.25_evidence_attempt1_failed/`,
136.15 s, 92/133 checks failed) showed all four proofs crashing **before any
stage ran**: `_benchmark_manifest.load_schema()` resolved
`<fresh-root>/.venv/Lib/artifacts/benchmark_manifest_schema_v1.json` — the
library located frozen artifacts via `Path(__file__).parents[2]`, which only
works in the editable source layout. This is an implementation defect on the
80.25 execution path, not a frozen contract (no manifest, taxonomy, schema
content, threshold, or control was touched). Fixed with the smallest coherent
change: new `src/latent_anything/_artifact_path.py::resolve_artifact_path`
walks upward from the installed package to the directory containing
`artifacts/<name>` (content still fail-closed validated by the loaders), with
the historical developer location as the error fallback; rewired the three
constants (`_benchmark_manifest.SCHEMA_PATH`, `_diagnostic_report.SCHEMA_PATH`,
`_representation_taxonomy.TAXONOMY_PATH`). In the developer layout the resolved
path is byte-identical to before (pinned by test).

## Historical findings from the original harness attempts


* **Peak-RSS measurement:** uv's virtualenv `python.exe` (254,696 B vs the base
  interpreter's 105,832 B) is a launcher that spawns the real interpreter as a
  child; sampling only `Popen.pid` recorded 4.1 MiB. The harness now samples the
  whole process tree (and kills the tree on timeout) and asserts a 32 MiB
  sanity floor plus the 16 GiB ceiling.
* **Declared overlay adjustment:** `datasets==3.6.0` itself requires
  `fsspec[http]<=2025.3.0,>=2023.1.0`, so the recorded 80.24 `uv run --with`
  overlay ran on fsspec 2025.3.0 (lock: 2026.2.0). The harness allows exactly
  this one adjustment (verified version) and fails on any other drift.
* **Dataset cache path:** offline load with a cold datasets cache fails in
  datasets 3.6 (module factory needs the hub), so the clean runs resolve the
  pinned revision online while `HF_HUB_CACHE` reuses only the hash-verified
  immutable snapshot (blob mtimes proven unchanged); arrow packing regenerates
  in the fresh per-run `HF_DATASETS_CACHE`.

## Hashes recorded for the historical blocked attempt


* Frozen inputs: `de24e80e…` (manifest schema), `e2935cd8…` (encoder
  manifest), `339c5c30…` (transformer manifest), `a76580e5…` (report schema),
  `e1742b78…` (taxonomy), `0908f843…` (L04), `3a2c867e…`
  (conv_vae_heldout_benchmark), entrypoints `91c8d8b7…` (80.23) and
  `629c2c59…` (80.24), `pyproject.toml` `106da07c…`, `uv.lock` `27e5e881…`,
  `.python-version` `a7b9da5e…`.
* Snapshot files: gpt2 blobs incl. `model.safetensors`
  `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`
  (= LFS blob name), wikitext parquet blobs likewise name-matched; full
  per-file table in `artifacts/task_80.25_evidence/environment.json`.
* Stage payloads/captures: as quoted above (identical to 80.23/80.24 evidence).

## Historical environment and commands


* Host: Windows 11, uv 0.9.7, Python 3.13.3; clean env versions:
  numpy 2.4.6, scikit-learn 1.9.0, torch 2.10.0 (module `2.10.0+cpu`),
  transformers 4.57.6, tokenizers 0.22.2, huggingface-hub 0.35.3,
  datasets 3.6.0, fsspec 2025.3.0; `uv pip check` clean; full package list,
  child-environment overrides, git identity, and every executed command with
  wall times are in `artifacts/task_80.25_evidence/environment.json`.
* Commands:
  `uv run python scripts/sprint80_task80_25_clean_repro.py --dry-run`
  (7/7 checks passed),
  `uv run python scripts/sprint80_task80_25_clean_repro.py` (full run; writes
  `artifacts/task_80.25_evidence/`),
  `uv run pytest tests/test_sprint80_25_clean_repro.py tests/test_artifact_path_resolution.py -q`.

## Output locations for historical attempts and clean replays

**The original attempts documented below did not produce validator-clean records.** Their declared `proof-80-23` and `proof-80-24` destinations were not materialized because those earlier frozen contracts failed closed. The separate outputs under `diagnostics/proof-80-25-clean-encoder-v3-20260924-035357-6996/` and `diagnostics/proof-80-25-clean-transformer-v1-target-v2-20260924-035357-6996/` are the reviewed temporary-driver replay only; their evidence is in `task_80.25_evidence_clean_20260924/`.
The corrected committed-driver replay copied validated output roots to `diagnostics/proof-80-25-committed-encoder-v3-20260924-160321-176370-23440/`, `diagnostics/proof-80-25-committed-transformer-v1-target-v2-20260924-160321-176370-23440/`, and the independent stability root `diagnostics/proof-80-25-committed-transformer-stability-supplement-20260924-160321-176370-23440/`; its complete evidence is in [`task_80.25_evidence_committed_20260924-160321-176370-23440/summary.json`](task_80.25_evidence_committed_20260924-160321-176370-23440/summary.json).


## Files modified during the historical harness implementation


* [scripts/sprint80_task80_25_clean_repro.py](../scripts/sprint80_task80_25_clean_repro.py) — committed clean-environment reproduction harness (pin verification, fresh root/env, four cold proof executions, deterministic-hash and blocker-identity checks, inventory diff, evidence, cleanup).
* [src/latent_anything/_artifact_path.py](../src/latent_anything/_artifact_path.py) — new installation-layout artifact resolver (no frozen content changed).
* [src/latent_anything/_benchmark_manifest.py](../src/latent_anything/_benchmark_manifest.py), [src/latent_anything/_diagnostic_report.py](../src/latent_anything/_diagnostic_report.py), [src/latent_anything/_representation_taxonomy.py](../src/latent_anything/_representation_taxonomy.py) — constants rewired to the resolver (developer path byte-identical).
* [tests/test_sprint80_25_clean_repro.py](../tests/test_sprint80_25_clean_repro.py) — 6 focused harness tests (copy excludes, hash drift naming, verdict/blocker parsing incl. loud unexpected-success detection, child-env isolation, snapshot drift).
* [tests/test_artifact_path_resolution.py](../tests/test_artifact_path_resolution.py) — 3 tests pinning dev-identical, installed-layout, and fail-closed-fallback resolution.
* [docs/sprint-plans/sprint-80.md](../docs/sprint-plans/sprint-80.md) — 80.25 blocker recorded at that historical checkpoint; the current row remains `[~]` pending Main's evidence review.

* [.agents/memory/lessons-learned.md](../.agents/memory/lessons-learned.md) — three dated, bottom-only lessons appended per the 80.25 evidence review (verdict `PASS_BLOCKED`): installed-layout artifact loading, the `torchinductor_<user>` temp-directory quirk, and the uv launcher RSS-sampling quirk; no prior entries or frozen contracts altered.

## Historical testing (prior harness implementation)


* `uv run pytest tests/test_sprint80_25_clean_repro.py tests/test_artifact_path_resolution.py tests/test_benchmark_manifest.py tests/test_representation_taxonomy.py tests/test_diagnostic_report.py tests/test_sprint80_core_manifests.py -q` → **35 passed**.
* `uv run pytest tests/test_encoder_end_to_end_proof.py -q` → **7 passed**;
  `uv run --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" pytest tests/test_transformer_end_to_end_proof.py -q` → **6 passed**.
* Dry-run rehearsal: **7/7 checks passed** (fresh copy+sync+overlay+isolation).
* The earlier implementation checkpoint omitted formatters, linters, and project-wide suites. This correction's scoped verification was `uv run pytest tests/test_sprint80_25_clean_repro.py -q` (**7 passed**), `uv run ruff check scripts/sprint80_task80_25_clean_repro.py tests/test_sprint80_25_clean_repro.py` (**all checks passed**), `uv run ruff format --check scripts/sprint80_task80_25_clean_repro.py tests/test_sprint80_25_clean_repro.py` (**two files already formatted**), and the final `--dry-run` (**9 checks passed**). Project-wide validation remains assigned to Task 80.28.

## Historical negative results and limitations (original blocked attempts)


* **Blocker (precise):** validator-clean run records/content-addressed artifacts
  cannot be regenerated because the upstream frozen contracts make them
  unreachable — encoder: truthful negative localization leaves the 80.16
  binding unresolvable (fails closed at explain); transformer: the frozen
  taxonomy requires capture axes `{sample, feature, label}` while the frozen
  manifest declares `{layer, token, slice}` and the accepted 80.7 binder has no
  `label` axis (persistence refuses). Resolution requires an evidence-review
  decision on a frozen contract — outside 80.25's authority.
* **Final harness verdict (recorded as it ran): `CHECK_FAILURES`, 136 checks,
  4 failed — root-caused, not a reproduction failure.** All four proof runs
  exited 0 with byte-identical recorded hashes/blockers, `ACCEPTANCE: NOT PASSED`,
  `validator_clean=False`, no artifact directories, unchanged frozen inputs and
  snapshots, and `cleanup.removed=true` (total wall 2197.26 s). The 4 failures
  are the harness's own over-strict temp-directory assertion demanding exact
  listing equality: importing torch creates its empty
  `torchinductor_<user>` cache dir in whatever `TEMP` is set (the same
  `torchinductor_admin` exists in the developer OS temp, 0 children) — freshly
  created inside the isolated per-run temp, which proves isolation rather than
  violating it. The pooled `.npz` itself was correctly recomputed in the
  isolated temp with the same content key as the developer cache
  (`latent-anything-80-24-66594a8c…npz`), the developer warm cache was proven
  untouched, and every substantive check (hash/identity/blocker/inventory/
  snapshot/peak/cleanup) passed. The assertion now ignores `torchinductor_*`
  entries; the recorded `summary.json` keeps the as-run verdict.
* Second cold transformer run **`80_24_run3` completed byte-exact**: pooled
  `b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b` again on a
  full re-extraction, same `BLOCKED persistence … taxonomy applicability
  mismatch for separability_probe_leakage`, `ACCEPTANCE: NOT PASSED`; proof
  body 1040.18 s (extraction+replay 1003.40 s), wall incl. imports 1045.80 s,
  tracemalloc peak 535.8 MiB, python 3.13.3 / numpy 2.4.6 / transformers 4.57.6
  / torch-cpu / scikit-learn 1.9.0 / datasets 3.6.0. Tree-sampled peak RSS per
  run (launcher+worker): 488.5 / 457.4 / 1725.5 / 1694.3 MiB — all far below
  the permanent 16 GiB ceiling; cross-run detect/localize/capture identity is
  asserted by the harness (`*_identical_across_runs` checks passed).
* The clean runs access the HuggingFace hub for revision resolution and file
  listing (datasets 3.6 cannot enumerate a parquet repo offline); all content
  comes from the hash-verified immutable snapshot and all selections are
  byte-verified against the pinned L04 manifest by the proof itself.
* Warm first-run walls (79.89 s encoder / 953.03 s transformer) reflect cold
  code/AV scanning of a fresh install, not measurement error; every run stays
  far below the permanent 16 GiB ceiling (tracemalloc ≤ 538.8 MiB).

## Historical graph refresh record

`graphify update .` ran after the initial script/test writes, again after the
resolver fix, and as a confirming update after the first artifact/plan draft
(EXIT=0: 15676 nodes, 35446 edges, 1063 aggregated communities, wall 216.50 s);
a further confirming update followed the harness `torchinductor_*` assertion
correction (EXIT=0: 15677 nodes, 35447 edges, 1061 communities, wall 143.34 s);
and the update after the evidence-review lesson writes recorded above returned
EXIT=0 (15677 nodes, 35448 edges, 1072 communities, wall 110.89 s). A final
confirmation was pending at that historical checkpoint.

The post-correction `graphify update .` reported `Code graph updated`; `graphify-out/graph.json` contains 16,403 nodes, 37,788 edges, 1,112 communities, and the HTML view was rebuilt (wall 148.75 s). Graphify warned that 314 source inputs produced zero nodes and that 1,098 saved community labels do not cover all 1,112 current communities (206 were renamed), recommending a label refresh. This command updated the code graph only; semantic re-extraction of the changed evidence reports was not performed under the assignment's no-subagent scope. See the task handoff for this graph-refresh caveat.

## Sprint 81 correction replay (2026-09-26)

The 2026-09-24 records above remain historical. After adding the `ci-proofs`
test dependency group and regenerating `uv.lock`, the committed clean-replay
driver was run against the exact current project bytes. A one-off Python wrapper
set the in-memory pins for `pyproject.toml`, `uv.lock`, and the package source
tree to their current SHA-256 values before calling the driver's normal
`main()`; no wrapper file was created. The run recorded driver SHA-256
`371775c2e1f8f1d009386178346895f29b51eb6dfb4f533869da7ba3f677a27e` and
the observed input pins. Those exact observed values are now embedded in
`scripts/sprint80_task80_25_clean_repro.py`.

- Run `20260926-115522-629609-4488` finished **`PASSED`**: 24 checks, 0 failed,
  2,429.965 seconds wall time. The isolated working tree was removed
  (`cleanup.removed=true`); historical negative evidence remained byte-exact.
- Exact candidate input digests: `pyproject.toml`
  `7050e09260e2f11dcd12b46faf65ddd7f526563d3a78846187b2b1898653913d`,
  `uv.lock`
  `cb6a23d95b73954f212a29e3e0360fc20773ead147c7709c460e4c6e79eb1379`,
  and the 162-file `src/latent_anything/**/*.py` tree
  `8df33742de6c3072969aff823b514500c4b067592fc4d0ae303638f994fed3b6`.
- The clean non-editable install and package-isolation check passed. `uv pip
  check` reported all 64 installed packages compatible. On Python 3.13.3 with
  `torch 2.10.0+cpu`, the encoder-v3 proof, transformer target-v2 proof, and
  transformer stability supplement all completed successfully; their exact
  run records, source and snapshot pins, logs, and content-addressed outputs
  are retained under
  `artifacts/task_80.25_evidence_committed_20260926-115522-629609-4488/`.
- The committed-pin regression file now passes **7 tests**. This appendix
  supplements rather than rewrites the earlier historical records.
