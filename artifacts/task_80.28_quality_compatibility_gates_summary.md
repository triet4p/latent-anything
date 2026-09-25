# Task Summary: 80.28 — Quality and compatibility gates

**Sprint:** Sprint 80  
**Task:** 80.28 — Run proportionate quality and compatibility gates  
**Status:** `[~]` — submitted to Main for evidence review; not marked `[x]`.  
**Candidate:** `F:/ai-ml/latent-anything`, not the old `F:/ai-ml/sprint80-candidate` checkout.

## Current gate results

| Gate | Result | Observed evidence |
|---|---|---|
| Strict Pyright | PASS | `uv run --locked --python 3.13 --extra viz --extra transformers --with datasets==3.6.0 --with huggingface-hub==0.35.3 pyright` — 0 errors, 0 warnings, 0 information diagnostics; only the locked-tool version advisory. |
| Full release Ruff/format | PASS | `ruff check src tests scripts --output-format concise`: all checks passed. `ruff format --check src tests scripts`: 509 files already formatted. Current refresh fixed 26 diagnostics, two formatted files, and one strict typing error. |
| Focused current-candidate regressions | PASS | Clean-repro driver tests: 7 passed; probe tests: 25 passed on CPython 3.12.12, 3.13.3, and 3.14.0; current helper's 9-check dry-run and committed-output validators passed. |
| Full offline pytest | PASS | The earlier corrected current-candidate run recorded **2,549 passed, 3 skipped, 42 deselected, 39 warnings** in 1,603.61s (26:43). The later final release-candidate shared-worktree gate recorded the same counts in 2,562.53s (42:42) in [`artifacts/sprint-80/task-28.md`](sprint-80/task-28.md). Neither is a clean-clone full-suite run. Both used the `viz` extra plus pinned `datasets`/`huggingface-hub` and excluded `network`, `large_download`, `viz`, and `integration`. |
| Schema and proof-source pins | PASS | Current helper verifies 54 explicit file pins plus the 162-file source tree (55 repository inputs). Current source pins: v3 `87891b0c…`, target-v2 `dee2d016…`, stability proof `a1c08fff…`, shared probe `0d114dfe…`, source tree `3226dc26…`; driver SHA `371775c2…`. Frozen manifest/checkpoint/target/protocol/schema pins and historical 80.25 outputs remain unchanged. |
| API snapshot | PASS | `snapshot clean`; SHA-256 `3fd8c73e6fd9fa9bae107b4be6727cc3e9c3c907c9b9d757fdae4c267d8691ec`. |
| Docs and evidence ledger | PASS | Strict MkDocs build completed in 143.10s in a temporary directory. Ledger: 107 capabilities; core 41/63 (65.1%), overall 41/64 (64.1%). |
| Python/extras and offline compatibility lanes | PASS | Current refresh: clean-base imports 3/3 and changed probe tests 25/25 on each of CPython 3.12.12, 3.13.3, 3.14.0. Earlier unchanged-dependency matrix: optional extras 36/36; VQ 11/11 on 3.12/3.13; LeRobot CPU 22 passed/1 skipped each; diffusion 8 passed/1 skipped each. Those broad extras lanes were not rerun for this source-only refresh. |
| Wheel and clean installation | PASS | `uv build` produced the 0.9.0 wheel; fresh CPython 3.13 non-editable install passed `uv pip check`; smoke confirmed 214 exports, packaged `latent_value.py`, and `(1,3)` NumPy round-trip. |
| Graphify | PASS with warnings | Updated graph: 16,403 nodes, 37,788 edges, 1,117 communities; `graph.html`: 1,117 community nodes, 1,528 cross-community edges. Warnings: 314 zero-node source files; labels stale (1,114 saved vs. 1,117 current; 30 renamed), no `graphify label`. |

## Findings and scope

- The initial Windows raw-byte schema mismatch was CRLF-only checkout conversion: 226 CRLF pairs in the 4,837-byte JSON file. Canonical LF bytes match the pre-existing pin and parse to identical JSON; `.gitattributes -text` prevents future conversion. No schema pin was changed.
- The 80.23/80.24 proof entrypoints and shared probe were cleaned/formatted without changing timed-import scope or scientific thresholds. Current helper source pins now match the current candidate; only these source-byte hashes and the aggregate package-source digest were refreshed. Frozen manifests, thresholds, baseline checkpoint, target record/rule, stability protocol, schema, and accepted 80.25 proof outputs remain unchanged.
- Historical red runs remain documented in the detailed report: the first current-tree run had three failures (PowerShell BOM, raw CRLF schema pin, and the default-cache GPT-2 replay); a later run had one fixture failure on missing-LeRobot metadata. The corrected current offline suite passed with 2,549/3/42/39 counts above.
- The accepted task-80.25 committed replay remains prior evidence: run `20260924-160321-176370-23440`, **24/24 passed**, with encoder v3, transformer target-v2, and a separate stability supplement ([task artifact](sprint-80/task-25.md)). This task did not repeat the long cold run; the refreshed 80.25 driver dry-run, source pins, focused regressions, direct persisted-output validation, and byte-identical stability render all passed. Its historical output JSON/manifests were not rewritten.
- This refresh excluded `network`, `large_download`, `viz`, and `integration` tests; no model acquisition, current cold replay, remote CUDA, or SmolVLA diagnosis/checkpoint/GPU proof was performed. `F:/llms/hf/models` external snapshot preflight passed in the current dry-run. The broad optional-extra and LeRobot/VQ/diffusion lanes are earlier unchanged-dependency results, not reruns.
- The wheel smoke initially asserted `py.typed`, absent from this wheel; the corrected smoke of a packaged module and API round-trip passed. Upstream Pyright/MkDocs/setuptools advisories were non-blocking. Temporary matrix/docs/wheel/probe outputs and the build-generated `src/latent_anything.egg-info` were removed; `.gh-pages-build` was left untouched.

See [`artifacts/sprint-80/task-28.md`](sprint-80/task-28.md) for commands, detailed counts, direct code changes, source-hash chronology, and full gate evidence. The sprint plan remains `[~]` pending Main's evidence review; this summary is not a review verdict.
