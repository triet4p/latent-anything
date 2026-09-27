# Sprint 82 Task 7 — Compose, persist, and extend

**Task status:** `[~]` (Main owns evidence review and status change)

## Summary

Added a task-oriented guide to the published `latent-anything==1.0.0` pipeline stories. It distinguishes `AnalysisPipeline`, `InterventionPipeline`/`ManipulationPipeline`, and `RolloutPipeline` by input, lifecycle, and result; explains canonical registry kinds, config construction, external plugin discovery and its untrusted-code boundary; and sets the explicit boundaries for batching, cache, profiling, async calls, and rollout streaming. A bounded CPU example builds an analysis pipeline from `ObjectSpec`, observes its result/profile, round-trips a typed result envelope, and writes/completes a filesystem run record. The guide distinguishes serialization integrity from model/run reproducibility and links to existing specialist references rather than duplicating them.

## Files changed

- `docs/public/compose-persist-extend.md` — new released-surface guide, runnable config/persistence example, specialist links, and limits.
- `docs/public/index.md` — added the composition/persistence route and links to pipeline, plugin, and storage specialists.
- `mkdocs-api.yml` — added the new page to API-site navigation.
- `graphify-out/graph.json` — scoped deterministic AST merge for the guide and updated landing page.
- `graphify-out/manifest.json` — refreshed AST stamps for those two Markdown sources; unrelated entries preserved.
- `artifacts/sprint-82/task-7.md` — this evidence record.

No package API, immutable release/tag, theory-site source/configuration, sprint-plan status, or public deployment was changed. The existing Task 7 `[~]` status is unchanged.

## Verification

- **Runnable bounded CPU/config/persistence example — PASS.** Extracted and executed the Python block in `docs/public/compose-persist-extend.md` with:

  ```text
  uv run --locked python -c "from pathlib import Path; p=Path('docs/public/compose-persist-extend.md').read_text(encoding='utf-8'); block=p.split(chr(96)*3+'python',1)[1].split(chr(96)*3,1)[0]; exec(block, {'__name__':'__main__'})"
  ```

  Exit status 0. Observed:

  ```text
  PipelineResult: latents=(12, 3), transformed=(12, 2)
  Profile stages: ['encode', 'method']
  Run record: completed; artifacts=1
  ```

  The example asserts the latent/transformed shapes, exact result type and latent-array round-trip, and reads the stored portable payload back before completing the run record.

- **Strict API-site build — PASS.** Ran after the final guide/table edits:

  ```text
  uv run --locked --extra docs mkdocs build --strict --config-file mkdocs-api.yml
  ```

  MkDocs built `.api-pages-build/` successfully with exit status 0. The Material for MkDocs 2.0 announcement appeared; there were no strict-build errors.

- **Local HTTP and Chromium visual/navigation smoke — PASS.** Served the final `.api-pages-build/` locally at `127.0.0.1:8772` with the configured `/latent-anything/api/` prefix. Fetching `http://127.0.0.1:8772/latent-anything/api/compose-persist-extend/` returned HTTP `200` and the guide title. Chromium rendered the page with the API navigation, table of contents, comparison table, and page title `Compose, persist, and extend - Latent Anything API Guide`; H1 was `Compose, persist, and extend a pipeline`. From the landing page, clicking the new visible route reached `/latent-anything/api/compose-persist-extend/` and the same title. This is a local surface check only, not a public deployment check.

- **Scoped deterministic AST graph update — PASS.** Used the interpreter recorded in `graphify-out/.graphify_python`. Only `docs/public/compose-persist-extend.md` and `docs/public/index.md` were extracted, with `extract(..., root=root, parallel=False)` and an isolated temporary AST cache. A temporary-copy dry run passed before writing the graph. The merge used `build_merge([extraction], graph_path=..., root=root, dedup=False)`, retained the existing community assignments/labels and `built_at_commit`, exported without reclustering, and stamped the manifest with `kind='ast'` and `scan_corpus=None`. No semantic extraction, full-corpus graph rebuild, community reclustering, report regeneration, or HTML regeneration ran.

  - Extraction: **14 nodes / 18 edges** from the two Markdown inputs.
  - Graph: **16,866 nodes / 39,478 links / 38 hyperedges** → **16,874 / 39,486 / 38**.
  - Preserved unrelated records: **16,860 nodes and 39,469 links** unchanged attribute-for-attribute; graph metadata and hyperedges unchanged.
  - Manifest: **1,829 → 1,830 entries**; every unrelated entry unchanged.
  - `docs/public/compose-persist-extend.md` AST hash: `ec3de39f9098b2a21db2efeb1eccce78` (semantic hash empty).
  - `docs/public/index.md` AST hash: `d2499e0c3befc11f76f7d4983b8eb34e` (semantic hash empty).

## Evidence limits and open findings

- The CPU example exercises a synthetic fixed-random adapter, PCA, profiling, and local serialization/run-record operations only. It does not validate a trained or pretrained model, task performance, plugin distribution loading, optional provider, cloud service, or accelerator.
- A local build/HTTP/browser smoke does not establish that the intended public `/api/` URL is deployed or available. Publication remains outside Task 7 scope.
- Project-wide validation remains Main's responsibility; only the example, strict API-site build, local site surface, and scoped graph merge above were exercised.
- No task-level blocker remains. This artifact records evidence for Main's review; Task 7 remains `[~]` pending that review.
