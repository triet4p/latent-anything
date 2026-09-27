# Sprint 82 Task 1 — Public-guide foundation

**Task status:** `[~]` (Main owns the evidence review and status change)

## Summary

Added an independent English MkDocs source/configuration for the intended `/api/` path and a task-oriented 1.0.0 landing page. The page links to existing API, support, integration, evidence, and persistence references; states the release/evidence/publication boundaries; gives page-authoring conventions; and documents locked local build/preview commands. The theory MkDocs configuration and deployment workflow were not changed.

## Files changed

- `.gitignore` — ignored the separate `.api-pages-build/` output.
- `mkdocs-api.yml` — added the English Material site configuration, intended `/api/` `site_url`, isolated docs/output directories, and only the existing `index.md` navigation entry.
- `docs/public/index.md` — added the landing page, task routes, 1.0.0 contract/evidence caveats, page-writing conventions, and local build instructions.
- `artifacts/sprint-82/task-1.md` — this evidence record.
- `graphify-out/` — refreshed by `graphify update .`, `graphify label .`, and `graphify export html`; results are recorded below.

## Verification

- **Strict local build — PASS.** Ran from the repository root:

  ```text
  uv run --locked --extra docs mkdocs build --strict --config-file mkdocs-api.yml
  ```

  MkDocs 1.6.1 / Material 9.7.7 built to `.api-pages-build/` in 3.34 seconds with exit status 0. The Material MkDocs 2.0 announcement banner appeared; no MkDocs build errors or strict-mode warnings were reported.
- **Output inspection — PASS.** `.api-pages-build/sitemap.xml` and the generated page canonical URL both contain `https://triet4p.github.io/latent-anything/api/`. Generated output contains the English Material landing page, a single `Start here` navigation entry, task-route list, release/evidence caveats, page-writing section, and build commands.
- **Internal-link check — PASS.** A temporary `HTMLParser` check over the generated page found 12 internal links; every path and fragment resolved to the built page (9 IDs available). No future guide page is linked from navigation.
- **Repository-reference check — PASS.** All 13 GitHub `blob` link occurrences in the landing-page source map to files present in the repository, including the `v1.0.0` freeze snapshot path. This checks local source targets, not live HTTP reachability.
- **Visual smoke — PASS.** Opened `.api-pages-build/index.html` in Chromium via `file://`; confirmed title `Latent Anything API Guide`, rendered route/evidence/build sections, and the single-page navigation, then closed the tab. No HTTP preview server was started.
- **Navigation interaction — PASS.** In Chromium, clicked the visible table-of-contents link for “Page-writing conventions”; the page navigated to `#page-writing-conventions`.
- **Theory boundary — preserved.** The new build writes to `.api-pages-build/`, separate from the existing `.gh-pages-build/`; `mkdocs.yml` and `.github/workflows/deploy-latent-anything-theory.yml` were not edited.

## Publication boundary and open findings

The local build is not publication. `https://triet4p.github.io/latent-anything/api/` is the intended location and has not been claimed as live or HTTP-verified. Public availability still requires the authorized combined deployment that retains the theory site at `/`, followed by an HTTP/browser check (Sprint 82 Task 9). No other Task 1 findings remain from the local build and output inspection.

## Graph update

`graphify update .` completed and rebuilt `graphify-out/graph.json`, `graphify-out/graph.html`, and `graphify-out/GRAPH_REPORT.md`. The result had 16,810 nodes, 39,418 edges, and 1,055 communities; six nodes are sourced from `docs/public/index.md`. It warned that 342 source files produced zero nodes and would be retried later.

`graphify label .` re-clustered to 1,059 communities. One 100-community batch (8/11) hit the Gemini free-tier HTTP 429 quota, but the graph has a saved, non-placeholder label for each of its 1,059 community IDs (checked). The label command skipped HTML because the graph exceeds its 5,000-node limit, so `graphify export html` was run afterward and succeeded, writing an aggregate view of 1,059 community nodes and 1,881 cross-community edges. The zero-node warning remains unrelated to the guide files.
