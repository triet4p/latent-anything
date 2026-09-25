# Sprint 80 Task 80.26 — AI-engineer guide and examples

**Plan status:** `[~]`, pending Main's evidence review. This artifact records implementation and observed evidence; it is not a review verdict and does not mark the sprint task complete.

## Outcome

Updated the user-facing guide and its two executable examples to demonstrate the accepted encoder v3 model-weight lesion and frozen transformer v1 manifest with versioned target-evidence-v2. Each example delegates the frozen case composition to its committed proof script, then independently reloads and validates the persisted diagnosis and deterministic rendered report. The user does not need to assemble the low-level capture, statistics, intervention, comparison, persistence, validation, or rendering pipeline.

The old fail-closed v1 guide/example results remain distinguishable historical evidence in the [initial task summary](../task_80.26_ai_engineer_guide_examples_summary.md); they have not been rewritten as earlier successes. No frozen input/manifest, source API, proof composition, or sprint-plan row was changed. `docs/INDEX.md` and the sprint-wide evidence packaging remain outside this H2 publication; task80.29 owns those updates. `mkdocs.yml` was not changed because its `docs_dir` is the separate `latent-anything-theory` site.

## Published files and generated evidence

- `scripts/ai_engineer_example_encoder.py` — runnable encoder v3 entry point, safe output selection, persisted-artifact reload, independent validation and deterministic render check.
- `scripts/ai_engineer_example_transformer.py` — corresponding transformer v1 / target-evidence-v2 entry point.
- `docs/AI_ENGINEER_GUIDE.md` — prerequisites, pinned model/data/manifest/target record, exact commands, expected diagnoses, output behavior, interpretation limits and observed resource envelope.
- `artifacts/diagnostics/ai-engineer-task80-26-encoder/` and `artifacts/diagnostics/ai-engineer-task80-26-transformer/` — prior shared-worktree smoke outputs, not included in the publication commit. Clean-clone outputs are generated in the disposable verification checkout and recorded below; they are not checked in.
- `artifacts/task_80.26_ai_engineer_guide_examples_summary.md` — retained historical negative evidence with a current-results follow-up.
- `graphify-out/graph.json`, `graphify-out/graph.html`, and `graphify-out/GRAPH_REPORT.md` — local ignored code-graph outputs refreshed during this task; they are not part of the scoped publication commit.
- `artifacts/sprint-80/task-26.md` — this evidence handoff.

The row in `docs/sprint-plans/sprint-80.md` remains `[~]`. No library source, test, lockfile, or frozen proof file was changed. No project-wide suite, formatter, linter, or build was run.

## Focused verification

### Encoder v3

Command (from repository root):

```bash
uv run python scripts/ai_engineer_example_encoder.py --output artifacts/diagnostics/ai-engineer-task80-26-encoder
```

Observed exit 0, wall 15.51 s. The accepted v3 proof completed all seven stages and required controls; the wrapper independently reloaded the persisted artifact, passed validation, rerendered deterministically, and checked the content-addressed render. Run `b52ab1cde4d818c8`; artifact SHA-256 `9c4b62262404c5c30789c0604bd34d3e13d96ddce3991d94c950b9dea4255c56`; report SHA-256 `8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194`; rendered report SHA-256 `05f7ef165f2779e6e6b7bb0890aa9ad2003b1469356710373af063ae8265209a`. Twelve content-addressed blobs were independently reloaded. The frozen manifest raw SHA-256 is `e5bcd341b1b746f05e4ac02f908bcd27f23bace39737c86c8ac38888e7451b82`, commitment `f823efc1b016ca9ad62c0f207e17d138947040a39f5d1c1f42ab6d0d65f5954f`, baseline checkpoint SHA-256 `92829e57aaaebf52d5446d1a9c9db5de4e12c1657fad5ecc986329ed8752bb2a`. The output reports zero feature-variance ratio and singular spread, effective rank `2.9576990034956805` (95% interval `[2.875274672590536, 2.9839066845293694]`), and supported restoration effect `+0.4638888888888889`.

The encoder output-collision check was exercised by rerunning the same command against the existing output path. It refused before proof execution with exit 2 and `output root already exists`; no overwrite occurred.

### Transformer v1 with target-evidence-v2

Command (from repository root):

```bash
uv run --extra transformers --with "datasets==3.6.0" --with "huggingface-hub==0.35.3" python scripts/ai_engineer_example_transformer.py --output artifacts/diagnostics/ai-engineer-task80-26-transformer
```

Observed exit 0, total wall 513.85 s (proof body 498.77 s). The frozen model and dataset checks passed; the run completed all seven stages, passed required controls, independently validated the report and 13 content-addressed blobs, rehashed all 13 evidence links, and checked deterministic rendering. Run `93d70ac7aa4cdfea`; artifact SHA-256 `da7ea7f0eafb544abcf68bacba116972fd46f7645812bd01a6ed9e9b0c972cfd`; report SHA-256 `c670acea7b27665581844916f17fb03aa38b5aa6d7740583dea34a5a6f467bdb`; rendered report SHA-256 `c3d1e2e03ce5065f859161932d488f3a537d14916bdbaa1fc25398c8bc4219d3`. The finding localized at `transformer.h.0`; the declared section-header attribute was supported, removal changed accuracy from `1.0000` to `0.7315` (effect `-0.2685`), and the aligned comparison was truthfully `neither` (both deltas zero). The frozen inputs remained byte-identical.

The transformer smoke also exercised fail-closed controls for manifest tampering, capture-provenance mismatch, train/evaluation leakage, resume identity mismatch, and a failed non-separable control; each produced the expected failure/inconclusive behavior. The frozen transformer manifest raw SHA-256 is `339c5c302def9d8b859e1914101803c84905347eec3caa1d596a5eadc015d0a2`, commitment `c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a`. Target record `target-section-header-attribute-record` SHA-256 `3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19`; predeclared rule digest `9b12485635decf738c4904702f75d7d5ae8cb6663555e2af5ee66476269f025b`. It binds the pinned 2,048 selected Wikitext rows, their labels, and split provenance separately from activation axes.

The resource figures in the guide now use the accepted revision-backed 80.25 replay at source commit `5241553`, run `20260925-065327-115093-14640`: encoder 7.272 s / 353,763,328 bytes; cold transformer 944.193 s / 1,840,766,976 bytes (below the 16-GiB process-tree ceiling). These are case-specific measurements, not a performance guarantee. The accepted transformer's leakage gap is `0.44889779559118237` (95% interval `[0.40881763527054105, 0.49699398797595196]`). Historical temporary-driver run `c4012406e89254ea` instead measured `0.45090180360721444` (95% interval `[0.4108216432865731, 0.49699398797595196]`); it is not attributed to the accepted run.

A second run of the transformer command against the populated output path was refused before proof execution with exit 2 and `output root already exists`; no overwrite occurred.

## Documentation checks

The guide gives a one-time pinned GPT-2 snapshot-download command, exact runtime overlays, run commands, expected artifacts, uncertainty and task-specific limitations. It distinguishes the current accepted cases from the old fail-closed v1 history and does not claim a general-purpose auto-diagnoser. The guide follows the project’s English documentation convention; the index update is outside this publication and remains with task80.29.

Earlier worktree link verification after the G4/G5 edits found 55 local Markdown links across the guide, index, historical summary, and task handoff; all resolved. The clean-clone link check appears under H2 publication evidence below.

## Graph refresh

`graphify update .` exited 0 in 111.40 s after the example and documentation changes. It re-extracted the code graph and reported 16,285 nodes, 37,591 edges, and 1,079 communities; `graphify-out/graph.json`, `graph.html`, and `GRAPH_REPORT.md` were updated. The aggregate visualization has 1,079 community nodes and 1,566 cross-community edges.

Graphify reported 291 source files with zero nodes and absent from the graph (mostly JSON); it also noted 1,111 saved community labels for 1,079 current communities, renaming 200 communities by hub. The CLI explicitly identifies this command as code/AST-only and says semantic document changes require `/graphify --update` in the assistant. The repository rule in `AGENTS.md` requires `graphify update .` after code changes; that required code-graph refresh completed. Semantic re-extraction of the guide and evidence Markdown was not performed.

## Second deep-review correction — guide G4/G5

The guide now states bounded-core readiness for the accepted, revision-backed
80.25 replay at source commit `5241553`, run
`20260925-065327-115093-14640`, conditional on completion of the final review
and plan gate. It explicitly does not claim final Sprint 80 signoff or a
sprint-wide gate pass. The sprint-plan row remains `[~]` pending Main's review;
`docs/PLAN.md`, the sprint plan, the depth report, and frozen evidence were not
changed.

The transformer metric attribution is per-run. The accepted report
`proof-80-25-committed-transformer-v1-target-v2-20260925-065327-115093-14640/diagnostic-report`
records held-out accuracy `1.0` with interval `[1.0, 1.0]` and leakage gap
`0.44889779559118237` with interval
`[0.40881763527054105, 0.49699398797595196]`. The historical temporary-driver
report
`proof-80-25-clean-transformer-v1-target-v2-20260924-035357-6996/diagnostic-report`
records `0.45090180360721444` with interval
`[0.4108216432865731, 0.49699398797595196]`; that inline-driver result remains
historical and was not rewritten. The estimates differ by one of 499 evaluation
rows (approximately `0.002004008016`). The accepted intervention record confirms
baseline `1.0`, intervened `0.7314629258517034`, and effect
`-0.26853707414829664` (the guide rounds these to `0.7315` / `-0.2685`; see
[record](../diagnostics/proof-80-25-committed-transformer-v1-target-v2-20260925-065327-115093-14640/artifacts/791bf36ad48fe4fa01960a7d98ff5afe8497ae7e5529a77212262c31a5d78838)).
The accepted report confirms the aligned `neither` comparison and its two zero
deltas.

## Changed files and focused verification

- `docs/AI_ENGINEER_GUIDE.md` — bounded-core readiness, accepted-replay resource measurements, and per-run transformer evidence.
- `artifacts/sprint-80/task-26.md` — corrected provenance, focused evidence, and review status.
- Focused evidence comparison used the accepted `runs.json`, `diagnostic-report`,
  persisted intervention record, and `task-25.md`, plus the historical
  temporary-driver `diagnostic-report`. The accepted run records transformer
  `wall_s` `944.193`, peak RSS `1840766976`, held-out accuracy `1.0`, and leakage
  gap `0.44889779559118237`; both per-run intervals and run attribution match
  their respective reports.
- Light executable smoke: `uv run python scripts/ai_engineer_example_transformer.py --help` — exit 0.
  The help listed the documented options; the proof itself was not executed.
  No full transformer proof was rerun for this documentation-only
  correction.
- No final deep-review or plan-gate PASS is asserted. No example script or code
  changed.
- Graph update: not run because this correction changes Markdown only; existing
  `graphify-out/graph.json` is present and was not refreshed.

## H2 publication and clean-clone evidence

The five-file publication commit is `85a4c77c38ab8ec0a0905602dfac53924126b067` (`feat(sprint80): publish ai engineer guide examples`). The scoped files were exactly `docs/AI_ENGINEER_GUIDE.md`, both `scripts/ai_engineer_example_*.py` entry points, this task handoff, and the historical task summary. Before staging, the shared worktree had nine modified and 94 untracked paths; the staged-name check contained only those five files. `docs/INDEX.md`, `docs/PLAN.md`, `docs/SPRINT_80_DEPTH_EVIDENCE.md`, `docs/sprint-plans/sprint-80.md`, task80.29, frozen proof sources, and generated output directories were not staged.

A disposable local Git clone at that revision had an empty `git status --short` before setup and execution, `core.autocrlf=true`, and HEAD `85a4c77c38ab8ec0a0905602dfac53924126b067`. `uv sync --frozen` succeeded with CPython 3.13.3 / uv 0.9.7 and installed 48 locked packages. The two examples and both committed case-composition modules were present in the clone. A relative-link check found 22 local Markdown links across the guide (13), handoff (2), and historical summary (7); all 22 resolved in the clone. The same links still resolve after the evidence-record edits below.

### Encoder v3

Command from the clean clone root:

```bash
uv run python scripts/ai_engineer_example_encoder.py --output artifacts/diagnostics/ai-engineer-task80-26-clean-encoder
```

Observed exit 0, wall 16.42 s. The frozen model lesion and digits data completed the full seven-stage workflow; required controls and independent validation passed, the example reloaded the persisted artifact, and deterministic rendering/content-address checks passed. Run `6889aada172832a9`; artifact SHA-256 `4a731e6e586a9bcca7fc39eda8db0071cfab38407157b2664d423237314a7d1d`; report SHA-256 `8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194`; rendered report SHA-256 `ceb15e47ece2d939154da2cce83524fb1cdf3f22372d5c0c719bf2fac6904baa`; 12 content-addressed blobs reloaded.

### Transformer v1 with target-evidence-v2

The pinned GPT-2 snapshot and Wikitext cache were external to the clone under `F:/llms/hf/models`. Executed with `HF_HOME=F:/llms/hf/models`:

```bash
env HF_HOME=F:/llms/hf/models uv run --extra transformers --with "datasets==3.6.0" --with "huggingface-hub==0.35.3" python scripts/ai_engineer_example_transformer.py --output artifacts/diagnostics/ai-engineer-task80-26-clean-transformer
```

Observed exit 0, wall 498.15 s (proof body 490.59 s). Frozen model/data checks passed, including all 3,760 official Wikitext rows, 2,461 nonblank rows, and 2,048 selected index/text hashes; the 1,549/499 grouped split was disjoint. The full seven-stage diagnosis and required controls passed, including deterministic replay, frozen-input invariance, independent report validation, all 13 content-addressed evidence links, and deterministic rendering. The probe found held-out accuracy `1.0000` and leakage gap `0.4489`; the removal intervention changed accuracy to `0.7315` (effect `-0.2685`), and the aligned comparison remained `neither` with zero deltas. Run `375b5a873b731e31`; artifact SHA-256 `22e194cb37f9303454f2d020084a8379c66633350c63982da2439fe5e90e3988`; report SHA-256 `c670acea7b27665581844916f17fb03aa38b5aa6d7740583dea34a5a6f467bdb`; rendered report SHA-256 `883f4e58e040776ead885fb0f5433f3ddeb3afb1271150aaaff9f36cb3e1fa35`; 13 content-addressed blobs independently reloaded. The entry point printed `ACCEPTANCE PASSED`.

### Output-collision, scope, and graph checks

Re-running each entry point against its populated `--output` directory refused before proof execution with exit 2. The encoder output-tree SHA-256 remained `1f03598a773a0543eac3f90d2ce7549f255c5a454bee38c7418b0eaff4c21ab2`; the transformer output-tree SHA-256 remained `789b6154e2fcd22eccd9b87eaa33192fe664dfc0b1cf4a88b14a8888a4aadb29`. A first transformer collision invocation transiently reported `can't open file` for the tracked entry point; file/status inspection showed the source present and unchanged, and an immediate identical retry produced the expected exit-2 refusal with the output-tree digest unchanged. The transient's cause is unknown; the full transformer run and collision behavior both passed on retry.

The clean-clone run outputs totalled 114,423 bytes (19 encoder files) and 322,099 bytes (17 transformer files); no large model/checkpoint files or credential-shaped assignments were found. They are run-specific outputs and were not committed. The external Hugging Face cache stayed outside the repository. No tracked example code changed after commit `85a4c77`.

After publishing the example entry points, `graphify update .` exited 0 in 83.72 s. The local ignored graph now reports 16,431 nodes, 37,814 edges, and 1,108 communities; the aggregate HTML contains 1,108 community nodes and 1,614 cross-community edges. Graphify noted 326 source files with zero nodes (mostly JSON) and 1,089 saved labels for 1,108 communities, renaming 211 by hub; no label refresh or semantic-document extraction was run. `graphify-out/` remains uncommitted.

The guide retains the bounded-core READY statement only as conditional on the final review and plan gate. Its accepted revision-backed leakage gap remains `0.44889779559118237`; the distinct historical inline-driver value `0.45090180360721444` remains attributed to run `c4012406e89254ea`, a one-row difference across 499 evaluation rows. The historical summary now labels the earlier resource envelope separately from the newer `5241553` replay used by the guide. No final deep-review or plan-gate PASS is asserted; the sprint row remains `[~]` for Main's review.
