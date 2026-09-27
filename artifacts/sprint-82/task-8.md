# Sprint 82 Task 8 — Reference and specialist navigation

**Task status:** `[~]` (Main owns evidence review and status change)

## Summary

Added a public 1.0.0 method-selection index and snapshot-derived classification map. Every canonical root export is classified exactly once as a worked-example API, explained supporting API, or specialist/reference API, with routes into the public guide or the frozen API reference and existing specialist documentation. The index also accounts for the explicit submodule re-exports, configuration/result schema counts, CLI commands and aliases, optional profiles, all nine sync/async pairs, and eight exception records without copying signatures. It distinguishes the blocked SmolVLA lane and excluded arbitrary-model/GPU/CUDA diagnostic claims.

## Files changed

- `docs/public/api-reference-index.md` — new method-selection index, advanced routes, release-contract inventory, and canonical 211-entry coverage map.
- `docs/public/index.md` — added the release-wide reference/specialist route to the landing page.
- `mkdocs-api.yml` — added the API index to site navigation.
- `graphify-out/graph.json` — scoped deterministic AST merge for the API index and landing page; existing graph metadata and unrelated records preserved.
- `graphify-out/manifest.json` — stamped the two Markdown sources as AST-extracted; unrelated manifest entries preserved.
- `artifacts/sprint-82/task-8.md` — this evidence record.

No package code, release tag, theory-site configuration, sprint-plan status, public deployment, or permanent test was changed.

## Verification

- **Frozen snapshot coverage check — PASS.** A scoped throwaway data check compared the map with `artifacts/api_freeze_snapshot_1.0.0.json`; no checker or permanent source-text test was added. Observed: **211** snapshot entries, **211** mapped names, **211** unique names, **0** missing, **0** duplicates, and **0** unexpected. Classification counts were **22 worked-example**, **123 explained supporting**, and **66 specialist/reference** APIs. All **8/8** submodule re-export modules and export lists matched section C. The same check found all **7** CLI command/alias names, all **12** optional profiles, all **9** sync/async pairs, all **8** exception names, and recorded the snapshot's **28** configuration and **89** dataclass/result schema counts.

- **Strict API-site build — PASS.** Final documentation build completed with exit status 0:

  ```text
  uv run --locked --extra docs mkdocs build --strict --config-file mkdocs-api.yml
  ```

  MkDocs reported no broken local anchors or strict-build errors. The Material for MkDocs 2.0 announcement was the only warning output.

- **Local HTTP and Chromium smoke — PASS.** The local MkDocs preview served the final index at `http://127.0.0.1:8772/latent-anything/api/api-reference-index/`. Browser `fetch()` returned **HTTP 200**; the rendered title was `API coverage and method-selection index - Latent Anything API Guide`, and the H1 was `Choose a method and find a released API`. Chromium showed the API navigation, method routes, classification table, and table of contents. A rendered-site audit checked **30** local links and anchors; **0** were broken. The 43 rendered table rows include the grouped canonical map and supplementary route tables.

- **Scoped deterministic graph update — PASS.** Used the interpreter recorded in `graphify-out/.graphify_python`. Only `docs/public/api-reference-index.md` and `docs/public/index.md` were AST-extracted with `extract(..., root=root, parallel=False)` and a temporary isolated AST cache. A temporary-copy dry run passed before writing; `build_merge(..., root=root, dedup=False)` and the existing assignments/labels were used for export. Final graph size remained **16,887 nodes / 39,505 links / 38 hyperedges**; the second synchronization after a prose-only refinement produced **19 AST nodes / 30 edges** and no structural graph delta. Unrelated node/link records, graph metadata, community assignments/labels, and hyperedges were preserved. The manifest remained at **1,831 entries**, with the two target pages stamped and all **1,829** unrelated entries unchanged. No semantic extraction, full-corpus rebuild, reclustering, report regeneration, or HTML graph regeneration ran.

## Evidence limits and open findings

- The local build and browser preview do not establish that the intended public `/api/` URL is deployed or publicly available. Publication remains Task 9 scope; the landing page continues to state this boundary.
- SmolVLA remains **BLOCKED** and non-gating; arbitrary-model and GPU/CUDA diagnostic claims are excluded. `DiagnosticWorkflow` is not a stable public import.
- Project-wide validation remains Main's responsibility; only the scoped snapshot/data check, strict documentation build, local HTTP/browser smoke, and scoped graph update above were exercised.
- No task-level blocker remains. This artifact records evidence for Main's review; Task 8 remains `[~]` pending that review.