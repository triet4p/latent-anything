# Historical Task Summary: 80.26 initial guide/example iteration (fail-closed outcomes preserved)

> **Historical scope:** The initial iteration below recorded the first v1 fail-closed example results and its then-current `[ ]` plan status. It is intentionally retained as historical negative evidence, not relabeled as success or presented as current acceptance. The current examples target encoder v3 and transformer v1 with target-evidence-v2; see the [current task handoff](sprint-80/task-26.md). The sprint row remains `[~]` pending Main's evidence review.


**Sprint:** Sprint 80 — Stable Depth
**Task:** 80.26
**Status:** **BLOCKED — NOT checked.** The checkbox for 80.26 remains `[ ]` in
`docs/sprint-plans/sprint-80.md`. The guide and both executable examples are
published, indexed, and smoke-exercised with observed evidence, but the
literal acceptance — a user reproduces **both diagnoses** end to end — is
unreachable under the frozen contracts: no validator-clean content-addressed
artifact or rendered report can exist for either frozen manifest (the encoder
chain fails closed at `explain` so `report` never runs; the transformer report
cannot pass the independent validator because of the frozen
taxonomy/manifest/binder axes triangle). Nothing was promoted, no frozen
contract was reopened, and no success is claimed.

## Summary of Work

* **Guide:** `docs/AI_ENGINEER_GUIDE.md` (English, per `docs/LANGUAGE.md`) —
  workflow overview, exact prerequisites table (Python 3.13 / uv 0.9.x, lock
  commands, transformers extra, datasets/hub overlays with recorded
  resolutions, pinned gpt2/wikitext snapshots, memory/time bounds), the two
  examples with **observed** expected outputs and interpretations, a
  "diagnose your own model and data" section built only on the public request
  types plus the taxonomy-axes persistence requirement, a command/bounds
  summary table, exit-code contract (exit 2 = drift, never success), and
  navigation links (INDEX, ARCHITECTURE, API_REFERENCE, PORTABLE_ARTIFACTS,
  sprint plan, frozen inputs, evidence artifacts).
* **Navigation:** `docs/INDEX.md` gains entry **29. [AI_ENGINEER_GUIDE]** so
  the guide is discoverable per documentation convention (mkdocs nav covers
  only the theory site; doc guides are indexed in INDEX.md).
* **Examples:** two minimal drivers that compose **nothing themselves** —
  they load the committed frozen-case compositions
  (`scripts/sprint80_task80_23_proof.py` / `sprint80_task80_24_proof.py`, the
  same importlib pattern the accepted tests use), build the high-level
  workflow, run it, and make exactly one `persist_diagnostic_artifact` call in
  a temporary directory:
  * `scripts/ai_engineer_example_encoder.py` — capture → detect → localize
    with real measurements, then the expected fail-closed `StageContractError`
    at explain and persistence refusal with zero files written.
  * `scripts/ai_engineer_example_transformer.py` — all seven stages, report
    assembly (`intervention_report_items`/`comparison_report_items`/
    `build_report`), then the expected axes-triangle persistence refusal with
    zero files written.
  Both exit `0` only on the expected truthful outcome, `2` on any drift
  (unexpected completion, unexpected persistence success, or files written by
  a refused call), `1` on execution failure.

## Exact commands and observed results (smoke evidence)

```text
uv run python scripts/ai_engineer_example_encoder.py
  -> EXIT=0, wall 5.00–16.32s (warm fit)
  stages completed: capture, detect, localize
  detect  : family collapse_rank_loss claim_allowed=True controls={"control-benign-low-variance": "passed", "control-healthy-counterexample": "passed", "control-null-shuffle": "passed"} observed={"bottleneck-effective-rank": 3.114875679428139, "bottleneck-singular-spread": 0.4735270836652963}
  localize: verdict 'negative'; score 3.114875679428139; reason 'no layer or sample meets the declared affected criterion; benign negative produces no location'
  status  : result=failed checkpoint=failed
  failure : stage=explain error=StageContractError: explain hypothesis 'h-collapse-bottleneck-mu' declares localization ('slice', 'slice-all') with no localized/supported prior slice selection carrying that identity
  EXAMPLE OUTCOME: expected fail-closed reproduction — ... (tasks 80.23/80.25)

uv run --no-sync --extra transformers --with "datasets>=2.19,<4" --with "huggingface-hub>=0.34,<1" python scripts/ai_engineer_example_transformer.py
  -> EXIT=0, wall 26.86s (warm pooled cache)
  stages completed: capture, detect, localize, explain, intervene, compare, report
  localize: verdict 'localized'; earliest 'transformer.h.0'
  report  : assembled 1 symptom(s), 1 intervention claim(s), 1 comparison row(s)
  intervene: remove remove-section-header-direction -> supported (effect -0.2685); comparison 'neither'
  EXAMPLE OUTCOME: expected fail-closed reproduction — ...
  persistence refusal: report failed independent validation; nothing was persisted: taxonomy applicability mismatch for separability_probe_leakage

uv run python -m py_compile scripts/ai_engineer_example_encoder.py scripts/ai_engineer_example_transformer.py
  -> COMPILE_OK

link check over docs/AI_ENGINEER_GUIDE.md (relative markdown links, resolved from docs/)
  -> 17 links;0 broken (re-run after this artifact's creation; the 80.26 artifact link itself was the only pending target before this file existed)
```

A first transformer smoke before the fix printed a degraded payload line
(stage payloads are `mappingproxy`, not `dict`); the examples were corrected
to the observed payload shapes and re-smoked (outputs above are from the
corrected, final versions). An unused `json` import flagged by LSP was
removed. No `src/` file was modified for 80.26.

## Why the literal acceptance remains blocked

1. **Encoder case (80.23, PASS_BLOCKED):** the frozen known-positive does not
   exist on the pinned revision, localization is truthfully negative, and the
   80.16 binding contract fails the chain at `explain` — the user obtains the
   truthful diagnosis *state* but no report/artifact, by design.
2. **Transformer case (80.24, PASS_BLOCKED):** the seven-stage diagnosis and
   report assembly complete, but the frozen taxonomy requires capture axes
   `{sample, feature, label}` while the frozen manifest declares
   `{layer, token, slice}` and the accepted 80.7 binder has no `label` axis —
   the independent 80.4 validator refuses the truthful report, so nothing is
   persisted or rendered.
3. **80.25 (PASS_BLOCKED):** both blockers reproduced byte-exactly from a
   clean non-editable environment, proving they are properties of the frozen
   contracts, not environment or harness artifacts.

Resolution requires an evidence-review decision on a frozen contract
(80.1/80.2/80.7, or the encoder manifest thresholds/expected defect). Until
then 80.26 stays unchecked; the guide documents this status prominently in its
"Honest status" callout so no reader can mistake the fail-closed outcomes for
success.

## Files Modified

* [docs/AI_ENGINEER_GUIDE.md](../docs/AI_ENGINEER_GUIDE.md) — new English AI-engineer guide (prerequisites, commands, observed outcomes, own-model section, navigation).
* [docs/INDEX.md](../docs/INDEX.md) — new entry 29 linking the guide.
* [scripts/ai_engineer_example_encoder.py](../scripts/ai_engineer_example_encoder.py) — minimal executable encoder-case example (high-level workflow only + one persistence attempt).
* [scripts/ai_engineer_example_transformer.py](../scripts/ai_engineer_example_transformer.py) — minimal executable transformer-case example (seven stages, report assembly, persistence attempt).
* [docs/sprint-plans/sprint-80.md](../docs/sprint-plans/sprint-80.md) — 80.26 blocker recorded; **checkbox deliberately left `[ ]`**.
* No `src/` file and no frozen contract was modified.

## Testing

Focused smoke and link checks only (no formatters, linters, or project-wide
suites, per task constraints):

* Encoder example: exit 0 with the expected fail-closed reproduction (twice:
  initial run + re-smoke after the payload-print fix).
* Transformer example: exit 0 with the seven-stage completion and axes-triangle
  refusal.
* `py_compile` on both examples: OK.
* Guide link check: all relative links resolve (including this artifact).
* Prior focused suites remain green from 80.25 (35 + 7 + 6 passing tests);
  nothing they cover was changed by 80.26.

## Graph Refresh Record

`graphify update .` after all 80.26 writes (two examples, guide, INDEX,
sprint plan): EXIT=0 — 15696 nodes, 35476 edges, 1070 aggregated communities,
wall 82.72 s. A final confirming `graphify update .` runs after this artifact,
with no writes afterwards.

## Follow-up — current accepted example results

This follow-up records the updated guide and executable examples. The fail-closed v1 results and explanations above remain historical evidence, unchanged and distinguishable from these accepted v3/v2 runs. The current evidence handoff is [task 80.26](sprint-80/task-26.md); the sprint row remains `[~]` pending Main's review.

### Encoder v3

Command from the repository root:

```bash
uv run python scripts/ai_engineer_example_encoder.py --output artifacts/diagnostics/ai-engineer-task80-26-encoder
```

Observed exit 0 (15.51 s). The seven-stage proof and required controls passed; the example independently reloaded the persisted artifact, validated it, rerendered deterministically, and checked the content-addressed render. Run `b52ab1cde4d818c8`; artifact SHA-256 `9c4b62262404c5c30789c0604bd34d3e13d96ddce3991d94c950b9dea4255c56`; report SHA-256 `8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194`; rendered report SHA-256 `05f7ef165f2779e6e6b7bb0890aa9ad2003b1469356710373af063ae8265209a`. Twelve content-addressed blobs were independently reloaded. The supported intervention restored the model, with effect `+0.4638888888888889`.

Rerunning against the same populated `--output` path refused before proof execution, exit 2; no overwrite occurred.

### Transformer v1 with target-evidence-v2

Command from the repository root:

```bash
uv run --extra transformers --with "datasets==3.6.0" --with "huggingface-hub==0.35.3" python scripts/ai_engineer_example_transformer.py --output artifacts/diagnostics/ai-engineer-task80-26-transformer
```

Observed exit 0, 513.85 s total wall (498.77 s proof body). Pinned GPT-2/Wikitext checks passed; all seven stages, required controls, independent report validation, 13 evidence-link hash checks and deterministic rendering passed. Run `93d70ac7aa4cdfea`; artifact SHA-256 `da7ea7f0eafb544abcf68bacba116972fd46f7645812bd01a6ed9e9b0c972cfd`; report SHA-256 `c670acea7b27665581844916f17fb03aa38b5aa6d7740583dea34a5a6f467bdb`; rendered report SHA-256 `c3d1e2e03ce5065f859161932d488f3a537d14916bdbaa1fc25398c8bc4219d3`. Thirteen content-addressed blobs were independently reloaded. The declared attribute was supported at `transformer.h.0`; removal changed held-out accuracy from `1.0000` to `0.7315` (effect `-0.2685`), and the aligned comparison was `neither` with zero deltas.

Rerunning against the populated `--output` path refused before proof execution, exit 2; no overwrite occurred. The smoke also exercised fail-closed manifest-tamper, capture-provenance, leakage and resume-identity checks plus the failed-control inconclusive outcome.

The guide's resource envelope and cross-run variability are sourced from the independent non-editable 80.25 reproduction: encoder 15.563 s / 353,533,952 bytes (337.2 MiB); cold transformer 983.945 s / 1,784,811,520 bytes (1,702.1 MiB), under the 16-GiB ceiling; randomized-control leakage-gap variation `0.002004008016` across accepted runs, with pinned inputs and thresholds unchanged.

Relative-link check: 52 local Markdown links across the guide, index, historical summary, and this task artifact resolved; zero failures. The sprint row remains `[~]`; these results are evidence for Main's review, not a task-completion verdict.

## Current Graph Refresh

After the updated examples and guide/index, `graphify update .` exited 0 in 111.40 s and refreshed the code graph to 16,285 nodes, 37,591 edges, and 1,079 communities. It updated `graphify-out/graph.json`, `graph.html`, and `GRAPH_REPORT.md`; the aggregate visualization contains 1,079 community nodes and 1,566 cross-community edges. Graphify reported 291 source files with zero nodes (mostly JSON), and 1,111 saved community labels for 1,079 communities, renaming 200 by hub.

The CLI identifies `graphify update .` as code/AST-only and notes that semantic document changes require `/graphify --update` in the assistant. The repository `AGENTS.md` rule requires this code-graph refresh after code changes; semantic re-extraction of the guide and evidence Markdown was not run.
