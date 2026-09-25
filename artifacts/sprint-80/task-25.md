# Sprint 80 Task 80.25 — committed accepted-workflow clean replay

**Current replay:** `PASSED` — 24/24 driver checks on 2026-09-25 at revision `5241553` (run `20260925-065327-115093-14640`). This is a worker evidence handoff, not a review verdict; the sprint-plan row remains `[~]` pending Main's review.

**Scope:** The committed entrypoint reproduces the accepted encoder v3 and transformer v1 `diagnostic-evidence-v2` workflows, plus the separately frozen Task 80.24 stability supplement. Historical negative outcomes and prior temporary-driver outputs remain unchanged and are identified below as history.

## Second deep-review correction (G1/G2/G3) — what changed

- **G1 (accepted core in Git revisions):** five commits now sit on top of `2485071`:
  - `7972091` — accepted 80.25 clean-replay basis (driver, 4 proof scripts, 24 package sources, manifests, reference + committed proof outputs, historical failed-run evidence, task-25 handoff) with byte-preserving attributes (209 files).
  - `cb0384c` — `--renormalize` `pyproject.toml`/`uv.lock` plus force-add of gitignored `.python-version` so stored blobs equal the driver's raw-byte pins (3 files, no content change).
  - `8ce083f` — extend `-text` cover to `src/latent_anything/**/*.py` and commit the 4 sibling sources carrying working-tree content changes (5 files).
  - `5241553` — renormalize the 25 tracked package sources whose LF blobs predated the `-text` rules so committed bytes equal pinned CRLF bytes (25 files, no source-content change).
  - `938edd9` — new current-revision replay evidence and byte-exact proof outputs (56 files).
- **G2 (EOL protection):** `.gitattributes` carries narrow `-text` rules for `.python-version`, `pyproject.toml`, `uv.lock`, all 54 driver-pinned inputs, the full 162-file package tree, the pinned tests, the reference/clean/committed proof-output trees, and all five evidence directories (historical failures included). Disposable-clone audit at `5241553` with `core.autocrlf=true`: **54/54** pinned files clone==worktree==pin, **162-file tree digest `3226dc26…` matches**, **15/15** content-addressed transformer blobs name==bytes, zero source diffs.
- **G3 (current-revision measurement):** the new replay ran from the committed revision's bytes (driver `371775c2…`, stability proof `a1c08fff…`, probe `0d114dfe…`, tree `3226dc26…`) — no drift. All three measured values below are re-measured in this run, not re-validated stored records.
- Out of scope (left dirty, uncommitted): 80.26 guide prose, 80.28/80.29 records, unrelated src/tests/scripts/docs deltas, and stray `head_init_*.py` dumps. No push was made.

## Final committed-driver replay

- Command: `uv run python scripts/sprint80_task80_25_clean_repro.py --hf-root F:/llms/hf/models`
- Run `20260925-065327-115093-14640`; committed driver SHA-256 `371775c2e1f8f1d009386178346895f29b51eb6dfb4f533869da7ba3f677a27e`. Full evidence: [summary](../task_80.25_evidence_committed_20260925-065327-115093-14640/summary.json), [runs](../task_80.25_evidence_committed_20260925-065327-115093-14640/runs.json), [checks](../task_80.25_evidence_committed_20260925-065327-115093-14640/checks.json), [pins](../task_80.25_evidence_committed_20260925-065327-115093-14640/pins.json), and [environment](../task_80.25_evidence_committed_20260925-065327-115093-14640/environment.json).
- Fresh root: `uv sync --frozen --no-dev --extra transformers --no-editable`; declared `datasets==3.6.0` / `huggingface-hub==0.35.3` overlay; `uv pip check` clean; non-editable install isolated to the clean venv's `site-packages` (Python 3.13.3); no auth token was passed.
- Runtime versions: uv 0.9.7, Python 3.13.3, NumPy 2.4.6, scikit-learn 1.9.0, torch 2.10.0+cpu, transformers 4.57.6, tokenizers 0.22.2, datasets 3.6.0, huggingface-hub 0.35.3, fsspec 2025.3.0.
- Every proof was bounded to 1,800 seconds and 16 GiB process-tree RSS. The completed root and per-proof temp/dataset/modules/HF caches were removed after byte-exact output copies; summary records `PASSED`, 24 checks, zero failures, unchanged pre/post pins and external snapshots, historical negative evidence preserved, and `cleanup.removed=true`. Total wall `2042.677 s`.

| Case | Run / artifact SHA-256 | Report SHA-256 | Render SHA-256 | Wall / peak process-tree RSS |
|---|---|---|---|---|
| Encoder v3 | `fa9bfd738a7dbc01` / `b2db02d6b9f39dc2331d503018b0f964a93e3e29414303c112652ac08e9ec0cb` | `8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194` | `e43f27391a8af063e3971b10bd94adbb00470de35e9e72e8c6f98a777c013b32` | 7.272 s / 353,763,328 B |
| Transformer v1 + target evidence v2 | `c64d382b1560e40e` / `b8a4afdc17108e14ebf028b469b0e32fd519b63acdfb5aea8cd91b26de3f488d` | `c670acea7b27665581844916f17fb03aa38b5aa6d7740583dea34a5a6f467bdb` | `8ee798f01d730df6e991a0fa4d64db5b16cf5fcd76029d53ff3bffaad3d2fde5` | 944.193 s / 1,840,766,976 B |
| Transformer stability supplement (separate run) | run-record SHA-256 `111fb5726b0f338eadd15d250d70470bb8e7922355e8192acb70f9294fc844fe` | `5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0` | — | 1,005.243 s / 1,692,811,264 B |

- Encoder v3 used manifest `e5bcd341b1b746f05e4ac02f908bcd27f23bace39737c86c8ac38888e7451b82` (commitment `f823efc1b016ca9ad62c0f207e17d138947040a39f5d1c1f42ab6d0d65f5954f`) and baseline checkpoint `92829e57aaaebf52d5446d1a9c9db5de4e12c1657fad5ecc986329ed8752bb2a`. The generated lesioned checkpoint SHA-256 is `81a29cc8d92b38164b108958764a6f1ffe07e5b822701077bd45faeffea91008`; lesion-record SHA-256 is `d1b0763ee791b0069dc94e13b698edb51f4fcd7711ff169ce22264c3130d0570`. Seven stages and independent validator `passed`.
- **F4 sign correction:** restoration intervention effect is `+0.4638888888888889` (increase). Separately, the aligned task comparison's candidate-minus-baseline delta is `-0.4638888888888889`. They have opposite signs by definition; neither is substituted for the other.
- Transformer v1 used manifest `339c5c302def9d8b859e1914101803c84905347eec3caa1d596a5eadc015d0a2` (commitment `c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a`), target record `3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19`, and frozen target rule `9b12485635decf738c4904702f75d7d5ae8cb6663555e2af5ee66476269f025b`. All seven workflow stages, the independent validator, 13 evidence links and 15 content-addressed blobs passed. Held-out accuracy was `1.0`; leakage gap `0.44889779559118237` (randomized `0.5511022044088176`); pooled states digest `b0e60bc5…`.
- The independent stability supplement used frozen protocol SHA-256 `a69b7152a800ac543ac76fec4d699c16681517708a95b353e5d241241c8acf0a`, separate output root [`proof-80-25-committed-transformer-stability-supplement-20260925-065327-115093-14640`](../diagnostics/proof-80-25-committed-transformer-stability-supplement-20260925-065327-115093-14640), and reported held-out accuracy `1.0`, coefficient stability `0.9748782295192386` (threshold `>=0.8`), leakage gap `0.4254901960784314` (threshold `>=0.15`), and all gates passed (`supplemental_not_part_of_transformer_v2_seven_stage_run=true`).
- Persisted reports and their containing output trees: [encoder v3](../diagnostics/proof-80-25-committed-encoder-v3-20260925-065327-115093-14640/encoder-diagnosis-v3.md), [transformer v1 target-v2](../diagnostics/proof-80-25-committed-transformer-v1-target-v2-20260925-065327-115093-14640/diagnostic-report), and [separate stability supplement](../diagnostics/proof-80-25-committed-transformer-stability-supplement-20260925-065327-115093-14640/reports/5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0.md).

## Exact pins, historical source drift, and failed first finalization

- The run pins the v3 and transformer manifests, v3 baseline checkpoint, report schema/taxonomy, proof entrypoints, critical implementation files, all 162 `src/latent_anything/**/*.py` files (tree SHA-256 `3226dc26ea19e15889e29f70c33c83ac29a0cf66115a81bd229a3b8eace75be4`), and all ten immutable external snapshot files. GPT-2 is pinned to `e7da7f221d5bf496a48136c0cd264e630fe9fcc8` (six files); WikiText is pinned to `f776294184f13b8ff2337b3841cf9269a6216d1e` (four files). `pins.json` and `environment.json` confirm pre/post equality and unchanged external snapshots.
- Current proof-source hashes at replay: encoder v3 `87891b0c75d4ddcfe890792e421ac4108801d08aa49a9d3a234da764481cb53f`; transformer target-v2 `dee2d016bf9a22dfd5ef9089ca7bea74bc28bf86d6f36f78ae28db505f8d6f7a`; stability supplement `a1c08fff6cb45f8a768ae57d2f727a4a89392b04f0b3bd26f4807e0972dc8ac5`. These are the committed bytes the replay executed (G3 resolved by re-measurement).
- The previously reviewed temporary-driver replay is historical, not proof of this committed run. It was executed through uncommitted inline `python -c`; its proof-source hashes were encoder `f9f25df0e746c1423b116ea5c3e481ab8e10837a856d4fad0618297689c888be` and transformer `b9e036eba0fb6831dbfa3ba6ba960f51843c8cce0e24f887b635146c2488a31a`. Its earlier leakage gap `0.45090180360721444` (one row of 499; delta `0.002004008016`) is retained as historical variance; the committed replays return the accepted reference `0.44889779559118237` with pooled digest `b0e60bc5…`.
- The prior accepted committed replay `20260924-160321-176370-23440` (24/24, driver `89fb6524…`) remains committed evidence alongside the first failed-finalization attempt `20260924-151256-484003-10236` (**NOT PASSED**, `UnboundLocalError` before output copy). Neither was rewritten by this run.
- Historical negative runs/directories remain distinct and hash-pinned; neither their failure outcomes nor the temporary-driver source history was rewritten as a pass.

## Verification and open review

- `uv run python scripts/sprint80_task80_25_clean_repro.py --dry-run --hf-root F:/llms/hf/models` — 9 checks passed (113.45 s, this revision).
- `uv run pytest tests/test_sprint80_25_clean_repro.py tests/test_target_evidence.py -q -p no:cacheprovider` — 11 passed (27.91 s, post-evidence commit).
- `uv run ruff check scripts/sprint80_task80_25_clean_repro.py tests/test_sprint80_25_clean_repro.py` — all checks passed. `uv run ruff format --check` on those same two files — both already formatted.
- Disposable `core.autocrlf=true` clone of `5241553` (`F:/ai-ml/s80-clone4`): 54/54 pins, 162-file digest, 15/15 blobs verified (see G2 above).
- Full project gates were not run; Task 80.28 owns those gates. No project-wide formatter/linter/test suite was run here.
- Remaining items: Main's evidence review. The plan status intentionally remains `[~]`; this worker artifact does not mark the sprint task complete.

## Changed files and graph

- `.gitattributes` — narrow `-text` protection for all raw-byte-pinned inputs and content-addressed evidence (commits `7972091`/`8ce083f`).
- `scripts/sprint80_task80_25_clean_repro.py` + `scripts/sprint80_task80_23_v3_proof.py` + `scripts/sprint80_task80_24_proof.py` + `scripts/sprint80_task80_24_stability_proof.py` — committed accepted replay basis (commit `7972091`).
- `src/latent_anything/**` (24 explicitly pinned + 4 content-changed siblings + 25 renormalized + 8 byte-identical HEAD files) and `tests/test_sprint80_25_clean_repro.py` + `tests/test_target_evidence.py` — committed replay sources and focused tests.
- Manifests, baseline checkpoint, reference/clean/committed proof outputs, all five evidence directories — committed inputs and byte-exact outputs.
- `artifacts/sprint-80/task-25.md` — this evidence handoff.
- Graph update: `graphify update .` reported `Code graph updated`; `graphify-out/graph.json` contains 16,419 nodes, 37,809 edges, 1,096 communities, and `graph.html` was rebuilt (wall 174.88 s). Graphify warned 326 inputs produced zero nodes and that 1,117 saved community labels no longer cover all 1,096 communities (188 renamed), recommending a `graphify label` refresh. This command updated the code graph only; semantic re-extraction of the changed reports was not performed.
