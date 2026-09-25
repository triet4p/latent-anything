# Sprint 80 Task 80.28 — Quality and compatibility gate evidence

**Task:** 80.28 — Run proportionate quality and compatibility gates  
**Plan status:** `[~]` — submitted to Main for evidence review; not marked `[x]`.  
**Candidate:** `F:/ai-ml/latent-anything` (the current shared project tree)  
**Environment:** Windows 11 10.0.26200, x64; CPython 3.13.3 for the host gates.

This report replaces the stale evidence that ran from `F:/ai-ml/sprint80-candidate`. The corrected current candidate was checked directly after DeepReview F1 and the committed task-80.25 replay correction. Existing scientific manifests, thresholds, baseline checkpoints, and accepted proof-output artifacts were not changed. The current 80.25 helper's source-byte pins were refreshed only for the type/style cleanup recorded here; the accepted cold replay is separately linked below and was not repeated.

## Direct 80.28 changes

- `src/latent_anything/_diagnostic_artifact.py`, `src/latent_anything/_redundancy_separability_detection.py`, and `src/latent_anything/_target_evidence.py`: strict-type narrowing, validated evaluation split identity, and target-evidence mapping validation; `tests/test_target_evidence.py` covers the observable binding and rejection behavior.
- `src/latent_anything/_probe_tcav_ig_explanation.py`: type the stability evidence mapping as `dict[str, object]` for strict Pyright and apply Ruff-only line formatting. No stability thresholds, evidence fields, or measurements changed.
- `scripts/sprint80_task80_24_stability_proof.py`: sort imports, remove two unused `indices` locals, and wrap long source/report strings. A post-edit report render matched the accepted 2,380-byte report byte-for-byte.
- `scripts/_m14_l04_transport_seam.psm1` and `tests/test_m14_l04_remote_transport.py`: prevent native Windows PowerShell 5.1 from prepending a BOM to raw stdin, with a focused regression.
- `tests/test_transformer_end_to_end_proof.py`: mark only the pretrained GPT-2 replay as `large_download`; the default offline lane excludes it. `tests/test_cli.py` isolates absent LeRobot distribution metadata as well as a blocked import.
- Ruff cleanup in `scripts/ai_engineer_example_encoder.py`, `scripts/ai_engineer_example_transformer.py`, `scripts/sprint80_task80_23_v2_proof.py`, `scripts/sprint80_task80_25_clean_repro.py`, `scripts/sprint80_task80_23_proof.py`, and `scripts/sprint80_task80_24_proof.py`; timed imports remain below `_MODULE_START` with local E402 exemptions only.
- `.gitattributes` and `artifacts/diagnostic_report_schema_v1.json`: prevent Git line-ending conversion of raw-byte-pinned inputs; the schema bytes still match the original `a76580e5…` pin and parsed JSON semantics.
- `scripts/sprint80_task80_25_clean_repro.py`: refresh its pins for the current stability proof source, shared probe source, and 162-file package-source digest after type/style-only cleanup. The accepted v3/target-v2/stability manifests, checkpoints, target record/rule, schema, and historical output JSON were not repinned or rewritten.
- `.agents/memory/lessons-learned.md`, `artifacts/sprint-80/task-28.md`, and `artifacts/task_80.28_quality_compatibility_gates_summary.md`: record the Windows raw-hash pitfall and current-candidate gate evidence.

## Verification

### Strict type, lint, and formatting

- `uv run --locked --python 3.13 --extra viz --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 pyright` — **PASS**, 0 errors, 0 warnings, 0 information diagnostics. The locked version advisory was informational only.
- `uv run --locked ruff check src tests scripts --output-format concise` — **PASS**, `All checks passed!`.
- `uv run --locked ruff format --check src tests scripts` — **PASS**, **509 files already formatted**.
- On this current refresh, the first Ruff/format/Pyright passes found 26 lint diagnostics (24 in the stability proof, two E501 in the shared probe), two files needing format, and one strict typing error on the stability metadata assignment. The narrow source fixes resolved all gates. An initial report-render smoke caught an omitted target-provenance bullet (2,247 bytes versus the accepted 2,380); after restoring it, the current renderer matched the accepted report byte-for-byte.

### Offline tests and regressions

The first current-tree full offline command was:

```text
uv run --locked --python 3.13 --extra viz --with datasets==3.6.0 --with huggingface-hub==0.35.3 pytest -q -p no:cacheprovider -m "not network and not large_download and not viz and not integration"
```

Observed result: **3 failed, 2,543 passed, 3 skipped, 41 deselected, 39 warnings** in 1,780.89 seconds. The three failures were diagnosed as follows:

1. `tests/test_m14_l04_remote_transport.py::test_native_windows_powershell_fake_seam_captures_raw_and_exit` exposed a real Windows PowerShell 5.1 byte-stream defect. A direct reproduction showed `Process.StandardInput` adding `EF BB BF` before the supplied raw bootstrap. The seam now temporarily uses UTF-8 without a preamble only around `Process.Start()` and restores the original setting. The exact regression test was rerun after the fix with the optional gate dependencies: **1 passed**.
2. `tests/test_sprint80_25_clean_repro.py::test_verify_pinned_inputs_passes_on_real_repo_and_names_drift` first reported raw SHA-256 `e2bbdccdb5999e53b13bfcbdb497282059818ab693a95258e5bf0fb02a16c71a` versus the unchanged pin `a76580e58a672ac283fa0a313b34f6148375dc5181dcb395ae1061ce42928d07`.
   The 4,837-byte checkout had 226 CRLF pairs and no bare line endings; replacing only CRLF with LF yields that existing pin and identical parsed JSON. `.gitattributes -text` plus canonical LF normalization resolved the transport mismatch; the focused verifier passes without repinning or changing schema semantics.
3. `tests/test_transformer_end_to_end_proof.py::test_deterministic_replay_of_hidden_state_extraction` initially lacked the declared `transformers` extra. In this first default-cache offline attempt, installing the extra exposed `LocalEntryNotFoundError` because that cache lacked the pinned GPT-2 revision and outbound access was disabled. The test remains marked `large_download`; the later accepted task-80.25 cold replay used the explicit pinned snapshot root and passed 24/24 checks.

The previously accepted task-80.25 cold committed replay was not repeated. Its **24/24 checks passed** result and output evidence are recorded in [`task-25.md`](task-25.md); this refresh instead exercised the corrected helper against those committed outputs and updated source pins.

```text
uv run --locked --python 3.13 --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 pytest tests/test_sprint80_25_clean_repro.py -q -p no:cacheprovider
```

Result: **7 passed** in 3.04 seconds; the current driver pin verifier checks all **55 repository inputs** (54 explicit file pins plus the 162-file package-source digest).

```text
uv run --locked --python 3.13 --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 pytest tests/test_probe_tcav_ig_explanation.py -q -p no:cacheprovider
```

Result: **25 passed** in 8.27 seconds on CPython 3.13.3; the same 25-probe suite also passed on 3.12.12 and 3.14.0 (see the compatibility section).

```text
uv run --locked python scripts/sprint80_task80_25_clean_repro.py --dry-run --hf-root F:/llms/hf/models
```

The refreshed driver SHA-256 is `371775c2e1f8f1d009386178346895f29b51eb6dfb4f533869da7ba3f677a27e`. The dry-run passed **9 checks**, including the 55 current repository pins, immutable external snapshot preflight, fresh-root non-editable package installation, package overlay, `uv pip check`, and isolation probe.

The current helper's artifact validators accepted the persisted encoder v3 and transformer v1/target-v2 result trees plus the separate stability supplement. The transformer/stability validators were called with accepted-contract stdout fixtures and read the actual persisted report/run roots. Encoder run `ade7b3144207bb88` retained artifact/report hashes `fedfe977…`/`8fb1528f…`; transformer run `2765d61287933ecb` retained `diagnostic-evidence-v2`, artifact/report hashes `f80c4ab7…`/`c670acea…`; stability run/report hashes remained `a6428146…`/`5f88ac98…` with all three gates passing. A separate stability renderer smoke matched the persisted report byte-for-byte: **2,380 bytes**, SHA-256 `5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0`.
The current helper pins the source bytes and all scientific/replay inputs independently. Current source-map entries: 80.23 v3 proof `87891b0c75d4ddcfe890792e421ac4108801d08aa49a9d3a234da764481cb53f`, 80.24 target-v2 proof `dee2d016bf9a22dfd5ef9089ca7bea74bc28bf86d6f36f78ae28db505f8d6f7a`, stability proof `a1c08fff6cb45f8a768ae57d2f727a4a89392b04f0b3bd26f4807e0972dc8ac5`, shared probe `0d114dfeb884c5ddbbabbee9f077ac284e547936a249423b65f517b456c75f93`, and 162 package sources `3226dc26ea19e15889e29f70c33c83ac29a0cf66115a81bd229a3b8eace75be4`. The helper has 54 explicit file hashes plus that package-tree digest and verified 55 repository inputs. These are current source bytes, not altered science pins: the accepted encoder v3 manifest/checkpoint, transformer v1 manifest/target record/target rule, stability protocol, and schema retain the values documented below and in task 80.25. Earlier immutable replay output JSON still records its original source-tree hash `5dc8a1922fe39893b218c9a80aa78c61e4f09a2e09e552febd53dbb0cccb1a89`.
An intermediate full offline attempt with the transformer extra reported **1 failed, 2,544 passed, 3 skipped, 42 deselected, 39 warnings** in 1,904.95 seconds. The failure was `tests/test_cli.py::test_inspect_dataset_missing_lerobot_has_actionable_extra`: this test blocked `lerobot` imports but did not isolate installed distribution metadata, so the compatibility guard ran first and reported an unrelated NumPy range. The same focused test failed without the transformer extra too; its fixture now stubs only the LeRobot metadata lookup to represent an absent optional distribution while retaining the import block. The exact focused retest passed: **1 passed** in 6.98 seconds.

The corrected current candidate's final offline suite **passed**: **2,549 passed, 3 skipped, 42 deselected, 39 warnings** in **1,603.61 seconds (26:43)**. This run used the non-network, non-large-download, non-viz, non-integration selection:

```text
uv run --locked --python 3.13 --extra viz --with datasets==3.6.0 --with huggingface-hub==0.35.3 pytest -q -p no:cacheprovider -m "not network and not large_download and not viz and not integration"
```

### API, documentation, and evidence ledger

- `uv run --locked --python 3.13 --extra viz --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 python scripts/api_freeze_snapshot.py --check` — **PASS** on the current candidate; snapshot clean, SHA-256 `3fd8c73e6fd9fa9bae107b4be6727cc3e9c3c907c9b9d757fdae4c267d8691ec`.
- API snapshot and surface tests were included in the final offline suite and passed.
- `uv run --locked --python 3.13 --extra viz --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 python scripts/validate_evidence_ledger.py` — **PASS**, 107 capabilities; core 41/63 (65.1%); overall 41/64 (64.1%).
- `uv run --locked --python 3.13 --extra docs mkdocs build --strict --site-dir <temporary-directory>` — **PASS**, strict documentation build completed in 143.10 seconds. The configured `.gh-pages-build` output was not overwritten; MkDocs Material emitted only its upstream 2.0 advisory.

### Python and optional-extra compatibility matrix

The complete matrix below was produced during the earlier 80.28 compatibility pass. This source-only refresh rechecked clean base imports and the changed probe behavior across the supported Python boundaries; `pyproject.toml` and `uv.lock` dependency pins remained unchanged.

- Clean-base import and optional-module isolation: **3/3 passed** on CPython 3.12.12, 3.13.3, and 3.14.0. Each base import reported version 0.9.0 and confirmed the optional libraries were absent from `sys.modules`.
- Changed probe regressions: `tests/test_probe_tcav_ig_explanation.py` — **25 passed** on CPython 3.12.12, 3.13.3, and 3.14.0.
- Optional-extra resolution/import: **36/36 passed** previously, the 12 declared extras on each of those three Python versions: `docs`, `diffusers`, `transformers`, `diffusers-full`, `3d`, `lerobot`, `lerobot-diffusion`, `lerobot-smolvla`, `viz`, `tracking-mlflow`, `tracking-wandb`, and `tracking`. This matrix was not rerun for the source-only refresh.
- Previously verified offline extras lanes: VQ-VAE 11/11 on Python 3.12 and 3.13; LeRobot CPU 22 passed/1 skipped each; LeRobot diffusion 8 passed/1 skipped each. These lanes were not rerun in this refresh.
- `lerobot-smolvla` was only resolved/import-smoked. **No SmolVLA diagnosis, checkpoint, GPU, or proof test was run.**

### Wheel build and clean installation

- `uv build --out-dir <temporary-build-directory>` — **PASS**, built `latent_anything-0.9.0-py3-none-any.whl` and the source archive.
- `uv venv --python 3.13 <temporary-wheel-venv>` and `uv pip install --python <venv>/Scripts/python.exe <wheel>` — **PASS**, installed the wheel non-editably with 40 runtime packages.
- The first throwaway smoke assertion expected `py.typed`, which is not in this package wheel; that probe exited 1 and did not indicate a package import failure. The corrected smoke checked an actually packaged module instead and **passed**: version 0.9.0, 214 top-level exports, origin under the fresh venv's `site-packages`, `latent_value.py` present, and `LatentValue`/`LatentSpace` NumPy round-trip shape `(1, 3)`.

## Deliberate omissions and preserved provenance

- This task-80.28 refresh did not execute `network`, `large_download`, `viz`, or `integration` tests, acquire models, or run remote CUDA. The default-cache GPT-2 attempt in the earlier offline run failed with `LocalEntryNotFoundError`; task 80.25 later used the explicit snapshot root `F:/llms/hf/models`, which the current helper's dry-run preflight verified.
- The latest accepted Task 80.25 replay is run `20260925-065327-115093-14640` at source revision `5241553` (`task-25.md`): **24/24 checks passed**. The earlier `20260924-160321-176370-23440` replay remains separately preserved as prior evidence.
- This final 80.28 pass did not repeat the cold replay. It independently compared current working-tree and clean-checkout pins, the accepted replay's recorded pre/post pin maps, all three accepted output-tree hashes, and content-addressed blobs. The revision and exact results are below.
- No SmolVLA diagnosis or scientific outcome was inferred. The optional `lerobot-smolvla` extra was only resolved/import-smoked in the prior compatibility matrix.
- Only the recorded current source hashes changed due to scoped type/style cleanup. Frozen scientific manifests, thresholds, checkpoints, protocol, schema, and accepted proof outputs were left intact; historical replay JSON retains the older source-tree hash.
- The task-specific matrix, docs, wheel, PowerShell diagnostic, and task temp roots were removed after verification, including the `uv build`-generated `src/latent_anything.egg-info`. No project-local `build/` or `dist/` output was created; existing `.gh-pages-build` was left untouched.

## Remaining findings

1. The full cold Task 80.25 replay was deliberately not re-executed. Its **24/24** outcome remains the accepted run above; the current and autocrlf checkout byte maps match that run's pre/post pins and persisted outputs. This task performed compatibility/G2 verification, not a new scientific measurement.
2. Network-, model-backed-, and remote-CUDA tests were not executed in this refresh. No SmolVLA diagnosis/checkpoint/GPU proof was run.

The final current-candidate offline suite passed, as did Ruff, formatting, strict Pyright, API, docs, evidence-ledger, wheel, and compatibility gates recorded above. The plan remains `[~]` pending Main's evidence review; no evidence here marks the task `[x]`.

## Graph update

- `graphify update .` completed after all source changes: **16,403 nodes, 37,788 edges, 1,117 communities**. The aggregated `graph.html` has 1,117 community nodes and 1,528 cross-community edges; six semantic/curated graph files were backed up under `graphify-out/2026-09-25/`.
- Graphify reported 314 source files with zero nodes and stale labels (1,114 saved versus 1,117 current communities; 30 renamed). No `graphify label` ran. This was the code-graph refresh; the later task-report-only edits were not semantically reindexed.

## Post-commit final gates and G2 checkout recheck (2026-09-25)

### Revision and source boundary

- Final repository `HEAD`: `f15859b4b7fe5d770254b570be4b6405547744e8` (`docs(sprint80): record current-revision 80.25 handoff for review`). The accepted replay source commit `5241553` is an ancestor; the later `938edd9` and `f15859b` commits add the replay evidence and task handoff. The accepted run remains `20260925-065327-115093-14640`, recorded as run at source revision `5241553`.
- Project-wide gates ran against the shared worktree at `f15859b`, **not** the clean clone. Initial `git status` reported 15 modified and 95 untracked paths.
- `git diff --name-only HEAD -- src tests scripts pyproject.toml uv.lock .gitattributes` listed five source/test deltas: `scripts/_m14_l04_transport_seam.psm1`, `scripts/sprint80_task80_27_smolvla_proof.py`, `tests/test_api_freeze_snapshot.py`, `tests/test_api_surface.py`, and `tests/test_cli.py`.
- `src/latent_anything`, `pyproject.toml`, `uv.lock`, and `.gitattributes` had no worktree diff. The 80.28 Windows transport and CLI fixes were included in the worktree gates; other existing docs/evidence changes were preserved and not attributed to this task.
- The final test/docs/API gates cover the shared worktree and its named overlays; they are not represented as clean-commit gates for those overlays. The separate clean clone verifies the committed byte-pinned core and evidence. The only file edited in this final-gate pass was `artifacts/sprint-80/task-28.md`; no product code, sprint plan, `docs/PLAN.md`, or 80.29 report was edited. The task row remains `[~]`.

### G2: revision-backed Windows checkout and raw bytes

- Created a disposable clone with `git clone --no-hardlinks -c core.autocrlf=true --no-checkout F:/ai-ml/latent-anything F:/ai-ml/s80-clone-task28-20260925`, then checked out `f15859b4b7fe5d770254b570be4b6405547744e8` detached. `git config --get core.autocrlf` returned `true`; `git status --short --untracked-files=all` was empty after checkout and again after the audit.
- `git check-attr -z text eol --stdin` audited 208 unique paths: 54 explicit pinned inputs, 162 package sources, 15 accepted transformer blobs, and the accepted run summary. All 416 attribute records reported `text: unset`, `eol: unspecified`; the scoped `-text` rules disable conversion.
- `git ls-files --eol` confirmed matching index/worktree bytes with `attr/-text`: CRLF for `.python-version`, `pyproject.toml`, `uv.lock`, and the accepted summary; LF for the clean driver, pinned package sources, tests, and representative CAS blob.
- `uv run --locked --python 3.13 python scripts/sprint80_task80_25_clean_repro.py --dry-run --hf-root F:/llms/hf/models` passed **9/9** checks in both the shared worktree (run `20260925-080317-104874-21928`, 285.15 s) and the clean autocrlf clone (run `20260925-080428-747692-14012`, 286.60 s). Both checked the non-editable fresh-root setup, package overlay, `uv pip check`, isolation probe, and immutable external snapshot preflight; neither invoked the cold proof.
- The driver's `verify_pinned_inputs` matched **54/54** explicit file pins plus the `162`-file source digest `3226dc26ea19e15889e29f70c33c83ac29a0cf66115a81bd229a3b8eace75be4` in the current worktree and clone. Both maps exactly equal the accepted run's `pinned_repository_inputs_pre` and `pinned_repository_inputs_post`.
- The accepted summary bytes were identical in the worktree and clone (SHA-256 `2d1696817a027e2b060c26994cd1b2f5ac0ee171340edd5dd5ca00cdda6f55fd`). It records run `20260925-065327-115093-14640`, `acceptance: PASSED`, 24 checks, zero failures, with every check marked OK.
- All three output trees match their summary file counts and tree SHA-256 values and are byte-identical between clone and worktree: encoder v3 19 files, transformer target-v2 17 files, stability supplement 2 files (38 total).
- The encoder's **14/14** and transformer target-v2's **15/15** content-addressed blobs each match their filename SHA-256 in the clone and current worktree.
- The disposable clone, docs output, fresh wheel environment, generated wheel/sdist files, and `src/latent_anything.egg-info` were removed after recording results. The wheel-output directory's 1-byte `.gitignore` was left untouched; no project-local `build/` or `dist/` directory remained, and `.gh-pages-build` was not used.

### Final release-candidate gates

The following commands ran against the current shared worktree on CPython 3.13.3:

- `uv run --locked ruff check src tests scripts --output-format concise` — **PASS**, all checks passed.
- `uv run --locked ruff format --check src tests scripts` — **PASS**, 509 files already formatted.
- `uv run --locked --python 3.13 --extra viz --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 pyright` — **PASS**, 0 errors, 0 warnings, 0 information diagnostics. Pyright's newer-version notice was informational.
- `uv run --locked --python 3.13 --extra viz --with datasets==3.6.0 --with huggingface-hub==0.35.3 pytest -q -p no:cacheprovider -m "not network and not large_download and not viz and not integration"` — **PASS**, 2,549 passed, 3 skipped, 42 deselected, 39 warnings in 2,562.53 s (42:42). Warnings were the reported sklearn convergence, project deprecation, and UMAP parallelism warnings.
- `uv run --locked --python 3.13 --extra viz --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 python scripts/api_freeze_snapshot.py --check` — **PASS**, snapshot SHA-256 `3fd8c73e6fd9fa9bae107b4be6727cc3e9c3c907c9b9d757fdae4c267d8691ec`.
- `uv run --locked --python 3.13 --extra viz --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 python scripts/validate_evidence_ledger.py` — **PASS**, 107 capabilities; core 41/63 (65.1%), overall 41/64 (64.1%).
- `uv run --locked --python 3.13 --extra docs mkdocs build --strict --site-dir F:/ai-ml/s80-task28-mkdocs-20260925` — **PASS** in 278.56 s to the isolated output path. MkDocs Material emitted its upstream MkDocs 2.0 advisory; `.gh-pages-build` remained untouched.
- `uv build --out-dir F:/ai-ml/s80-task28-wheel-20260925` — **PASS**, built `latent_anything-0.9.0-py3-none-any.whl` and the source archive. Setuptools emitted its upstream `project.license` table deprecation notice.
- `uv venv --python 3.13 F:/ai-ml/s80-task28-venv-20260925` and `uv pip install --python F:/ai-ml/s80-task28-venv-20260925/Scripts/python.exe F:/ai-ml/s80-task28-wheel-20260925/latent_anything-0.9.0-py3-none-any.whl` — **PASS**, fresh non-editable install.
- `uv pip check --python F:/ai-ml/s80-task28-venv-20260925/Scripts/python.exe` — **PASS**, all 40 installed packages compatible.
- Fresh-venv smoke — **PASS**: version `0.9.0`, declared `__all__` has 214 exports, `latent_value.py` is packaged, import origin is under the venv's `site-packages`, and `LatentValue`/`LatentSpace` NumPy round-trip shape is `(1, 3)`.

### Compatibility scope and remaining limits

- The unchanged optional-extra matrix remains **prior evidence**, not a rerun in this final gate: 36/36 resolution/import checks across 12 extras and CPython 3.12.12, 3.13.3, and 3.14.0; prior clean-base import and changed-probe checks were also across those three interpreters. The 80.28 final gates above ran on CPython 3.13.3; dependency pins were unchanged.
- The pytest marker selection explicitly excludes `network`, `large_download`, `viz`, and `integration`. No remote CUDA/GPU test or SmolVLA diagnosis/checkpoint/proof was run. The earlier default-cache GPT-2 attempt's `LocalEntryNotFoundError` remains a failure/history, not a pass; the accepted cold replay's pinned snapshot was only preflight-checked here. Historical negative runs remain unchanged.
- The accepted 24/24 replay was not repeated because its pinned current source bytes and all committed output-tree digests match the accepted run exactly. No new scientific result or hardware/network lane is inferred.
- Graph update for this final gate: **not run** because the task made no source/code changes. The existing project graph is present; its last source update is recorded above. The plan remains `[~]` pending Main's evidence review.

## Third deep-review H1 — revision-backed closure (2026-09-25)

This section records the correction for H1 from `agent://DeepReviewSprint80Third`. Task 80.28 remains `[~]` pending Main's evidence review. No guide/examples, depth report, global plan, sprint plan, or H2/H3 findings were changed.

At `f15859b4b7fe5d770254b570be4b6405547744e8`, the committed package already exported the nine diagnostic names `CaptureSelection`, `ComparisonRequest`, `ControlSelection`, `DiagnosticRequest`, `DiagnosticRequestError`, `DiagnosticResult`, `DiagnosticSelection`, `InterventionRequest`, and `OutputSelection`; no package source changed in this correction. The live 0.9.0 surface therefore measured 214 current names, 211 stable names, and 8 exceptions, while the committed snapshot still recorded 205, 202, and 7 respectively (old snapshot SHA-256 `048ac553adabb11c24d3e1f4d86e0c6d469df6590064014c915d6a08c7d53c26`). The already-regenerated worktree snapshot records the live surface and SHA-256 `3fd8c73e6fd9fa9bae107b4be6727cc3e9c3c907c9b9d757fdae4c267d8691ec`.

The six audited target-file deltas were committed without staging unrelated work:

- `artifacts/api_freeze_snapshot_0.9.0.json`: commit `7a8fb82c24d31e850da8c0d119166e364f600e16`; records the current runtime surface, including `DiagnosticRequestError`.
- `tests/test_api_freeze_snapshot.py` and `tests/test_api_surface.py`: same API commit; update the three observed counts and list the nine already-public names.
- `tests/test_cli.py`: commit `773c5a47ee9db3a81c016cfd72c3a5e96eb10231`; the missing-LeRobot test isolates distribution metadata inside its subprocess so installed host metadata cannot mask the missing-import behavior.
- `scripts/_m14_l04_transport_seam.psm1`: commit `bf8dcadc2ae821c5b299d3a15f4c04bdd8581ce8`; temporarily disables the Windows PowerShell 5.1 UTF-8 preamble only around `Process.Start()`, then restores the original console input encoding so raw transport stdin remains byte-exact. The existing native-Windows fake-seam regression covers the behavior.
- `scripts/sprint80_task80_27_smolvla_proof.py`: commit `45fa60266a2ab79f3d4803b53aede29358673ff5`; formatting/import-order cleanup only, retained because the previous whole-repository Ruff/format gates include scripts. No proof behavior changed.

The resulting code candidate is `45fa60266a2ab79f3d4803b53aede29358673ff5` (four conventional commits total). No source package, lockfile, frozen scientific input, or accepted replay output was edited.

### Clean-checkout evidence

Verification used a fresh detached clone at the code candidate with `core.autocrlf=true`: `F:/ai-ml/s80-task28-h1-clean-20260925`. Its initial and post-gate Git status was clean. `uv run` created an isolated `.venv` and installed 84 packages.

```text
uv run --locked --python 3.13 --extra viz --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 python scripts/api_freeze_snapshot.py --check
```

**PASS** — `snapshot clean (3fd8c73e6fd9fa9bae107b4be6727cc3e9c3c907c9b9d757fdae4c267d8691ec)`.

```text
uv run --locked --python 3.13 --extra viz --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 pytest -q -p no:cacheprovider tests/test_api_freeze_snapshot.py tests/test_api_surface.py tests/test_cli.py::test_inspect_dataset_missing_lerobot_has_actionable_extra tests/test_m14_l04_remote_transport.py::test_native_windows_powershell_fake_seam_captures_raw_and_exit
```

**PASS** — 12 passed in 22.64 seconds, including the native Windows PowerShell 5.1 fake-seam test.

```text
uv run --locked --python 3.13 ruff check scripts/sprint80_task80_27_smolvla_proof.py
uv run --locked --python 3.13 ruff format --check scripts/sprint80_task80_27_smolvla_proof.py
```

**PASS** — Ruff checks passed; the file was already formatted.

```text
uv run --locked --python 3.13 python scripts/sprint80_task80_25_clean_repro.py --dry-run --hf-root F:/llms/hf/models
```

**PASS** — the clean-revision preflight passed all 9 dry-run checks, including the accepted repository-input hash audit, package-source-tree digest, immutable external snapshot preflight, isolated non-editable environment, package overlay, `uv pip check`, and isolation probe. No cold scientific replay was run. The dry-run temporary root was removed by the script.

The earlier full Ruff, formatting, strict Pyright, and 2,549-pass offline pytest results remain evidence for the same six target-file contents in the shared worktree; those contents were committed without further source/test edits. They were **not** rerun as whole-project gates in the clean clone. The new clean-clone proof is the API snapshot check, 12 focused tests, file-scoped Ruff/format checks, and pinned-core dry-run recorded above; it is not represented as a clean-clone 2,549-test run.

### Graph update

After the code commits, `graphify update .` completed: 16,428 nodes, 37,813 edges, and 1,089 communities; the aggregated graph has 1,089 community nodes and 1,592 cross-community edges. Six semantic/curated graph files were backed up under `graphify-out/2026-09-25/`. Graphify reported 326 source files with zero nodes and stale labels (1,096 saved labels versus 1,089 current communities; 199 renamed); no `graphify label` ran.
