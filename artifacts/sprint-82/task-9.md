# Sprint 82 Task 9 — Publish and verify a safe combined theory + API Pages site

**Task status:** `[~]` (Main owns the evidence review and status change)

## Summary

Wired a boring single-deploy mechanism that publishes the existing theory site at `/` and the new API guide at `/api/` from one merged tree, without touching release tags/assets or Pages settings. No remote publication was performed: the live `/api/` URL still returns 404 and no tag was created or pushed. All entry-point links to the guide are marked pending truthfully.

## Decision (recorded here, not in memory files)

**Decision:** single combined deploy — both `mkdocs.yml` (theory, `site_dir: .gh-pages-build`) and `mkdocs-api.yml` (guide, `site_dir: .api-pages-build`) build with `--strict`, then `scripts/merge_pages_trees.py` copies the API tree under `.gh-pages-build/api/`; the existing `deploy-latent-anything-theory.yml` workflow (now triggers `theory-v*` + `guide-v*` + `workflow_dispatch`, `concurrency: gh-pages-deploy`, `force_orphan: true`) publishes the merged tree atomically.
**Alternatives considered:** (a) separate API-only workflow targeting `gh-pages` — rejected, it would erase the theory root on `force_orphan`; (b) one MkDocs config with both trees — rejected, it would force theory notebook execution/image gates onto guide edits and vice versa.
**Reason:** separate `site_dir` values mean MkDocs `--clean` (default) only wipes its own output dir, so neither strict build erases the other; the merge is the only shared step and is used identically locally and in CI.
**Consequences:** `guide-v*` tags are new (no `guide-v*` tag created/pushed in this task); any future second writer to `gh-pages` must go through the same merged tree.
Memory files (`.agents/memory/decisions.md`, `lessons-learned.md`) were intentionally NOT appended: both already carry unrelated uncommitted changes, and mixing Task 9 entries into them would contaminate the user's working tree.

## Files changed (Task 9 scope only)

- `scripts/merge_pages_trees.py` — new; single merge implementation (stale `api/` removed first, both `index.html` required, reports file counts). Shared by local proof and deploy workflow.
- `scripts/check_pages_links.py` — new; merged-tree internal link/anchor gate. Excludes three pre-existing theory patterns byte-identical on the live site (notebook raw `.md`/`.ipynb` hrefs, duplicate/truncated notebook TOC anchors, 404 absolute base-path links); fails on any other broken `(page, link, reason)`.
- `.github/workflows/deploy-latent-anything-theory.yml` — renamed to combined deploy; triggers `theory-v*` + `guide-v*` + `workflow_dispatch`; added `concurrency: gh-pages-deploy, cancel-in-progress: false`; steps: theory sync → notebook execute → image-output gate (unchanged) → theory strict build → API env sync → API strict build → merge → link audit → single `gh-pages` publish with `force_orphan: true`.
- `.github/workflows/docs-check.yml` — new; strict gate for docs PRs/main pushes touching `docs/public/**`, `latent-anything-theory/**`, both MkDocs configs, merge/link scripts: theory strict build (renders committed notebooks as-is, `execute: false`) + API strict build + merge + link audit. Never publishes. (CI on main ignores `docs/**`, so this workflow owns the docs contract.)
- `README.md` — added `1.0.0 API guide` entry pointing at `docs/public/index.md`, marked pending public site explicitly.
- `docs/INDEX.md` — added `1.0.0 API guide` entry, marked pending.
- `docs/VERSIONED_DOCS_1.0.0.md` — publication-status caveat now describes the combined `/` + `/api/` deployment, `theory-v*`/`guide-v*` triggers, and states `/api/` remains pending until landed + HTTP-verified.
- `artifacts/sprint-82/task-9.md` — this record.
- `graphify-out/graph.json` (+ `manifest.json` stamps) — scoped AST merge of the 7 Task 9 files; ignored by git, see graph section.

Not touched: `v1.0.0` / `docs-v1.0.0` / `docs-v1.0.0-r1` tags, release assets, Pages settings, theory notebook sources, `mkdocs-api.yml` nav, guide page substance (Task 8 owns unlanded `docs/public/`, `mkdocs-api.yml` edits — coordinated via `agent://Sprint82Task8`).

## Verification (exact commands, observed results)

- **Theory strict build — PASS.** `uv run --locked --extra docs mkdocs build --strict` → exit 0, `Documentation built in 125.11 seconds`, output `.gh-pages-build/` (~107M). Committed notebooks render as-is (96/96 lack executed `image/png` outputs locally — that gate runs deploy-time notebook execution, unchanged).
- **API strict build — PASS.** `uv run --locked --extra docs mkdocs build --strict --config-file mkdocs-api.yml` → exit 0, `built in 0.96 seconds`, `.api-pages-build/` (8 sitemap URLs, canonical `.../latent-anything/api/`).
- **Merge — PASS.** `uv run --locked --no-sync python scripts/merge_pages_trees.py` → `merged 55 API files under .gh-pages-build/api; deploy tree now holds 407 files`.
- **Link audit — PASS.** `uv run --locked --no-sync python scripts/check_pages_links.py` → `checked 54391 internal links (8449 anchors); unexpected broken: 0`; known pre-existing patterns: `404-absolute-base-path: 231`, `notebook-duplicate-heading-anchor: 724`, `notebook-raw-md-link: 169`, `notebook-raw-ipynb-link: 1`. Every excluded pattern was confirmed byte-identical on the live site (fetched `https://triet4p.github.io/latent-anything/...` pages contain the same `.md`/`.ipynb` hrefs and `_1`/`experiment-*` anchors). Zero broken among `api/*` and `*/research/*` pages.
- **Workflow YAML — PASS.** Both workflows parse (`yaml.safe_load`); deploy steps (12): Checkout, Set up Python, Install uv, Sync theory docs environment, Execute theory notebooks, Verify notebook image outputs, Build theory site, Sync API guide env, Build API guide, Merge, Audit, Deploy combined tree. `permissions: contents: write`, `concurrency: gh-pages-deploy`.
- **Local HTTP smoke — PASS** (server: `python -m http.server 8897 --bind 127.0.0.1 --directory .gh-pages-build`; note: dual-stack `http.server` on port 8899 silently dropped connections in this environment — IPv4 `--bind 127.0.0.1` fixed it): `200` + correct `<title>` for `/` (Latent-Anything Theory), `/api/` (Latent Anything API Guide), `/api/latent-primitives/`, `/api/api-reference-index/`, and a theory research page.
- **Browser smoke — PASS** (headless Chromium, HTTP): `/api/` renders H1 `Latent Anything 1.0.0 API Guide` with all 8 nav entries and route content; `/api/api-reference-index/` renders H1 `Choose a method and find a released API`; `/` renders theory root H1 with section content. (`file://` directory URLs show a listing — expected; HTTP is the valid proof.)
- **Live baseline (pre-deploy) — theory intact, API absent as expected.** `GET https://triet4p.github.io/latent-anything/` → 200 `Latent-Anything Theory`; `GET .../latent-anything/api/` → HTTP 404. Remote `gh-pages` tree top-level has no `api/` dir; Pages source is `gh-pages:/`, `force_orphan` deploy preserved by design.
- **Lint/format — PASS.** `ruff check` + `ruff format --check` clean on both new scripts. Pyright: project `include` enumerates scripts individually and does not cover the two new files, so no config change was made (verified direct `pyright` on them is out of scope, not a failure).
- **Release-asset/tag safety — PASS.** `git tag --list` shows `v1.0.0`, `docs-v1.0.0`, `docs-v1.0.0-r1` untouched; no tag created, no push, no Pages-settings change. Working-tree diff for Task 9 scope is limited to the files above; unrelated uncommitted changes (`.agents/memory/*`, `docs/PLAN.md`, `artifacts/m14/*`, sprint-80 scripts, `build/`, Task 1–8 artifacts) were inspected and left alone.

## Deployment gate / trigger / rollback (no destructive action taken)

- Gate: `guide-v*`/`theory-v*` tag or `workflow_dispatch` → combined workflow runs notebook execution + image gate + both strict builds + merge + link audit, then one atomic `gh-pages` publish. Docs PRs get the same builds + merge + audit via `docs-check.yml` without publishing.
- Rollback: re-run the workflow from the previous good tag/commit via `workflow_dispatch` (single `force_orphan` commit); no history surgery. `concurrency: gh-pages-deploy` serializes overlapping deploys.
- Blocker (external, explicit): **public deployment NOT performed — no authorization was established in this task.** `gh auth status` shows a logged-in user, but per the sprint contract Main owns closure and no reviewed push authorization was granted; no `guide-v*` tag was created or pushed and no workflow was dispatched. The intended `https://triet4p.github.io/latent-anything/api/` is therefore **not live** (verified HTTP 404) and must not be claimed. Next step after Main's review: create/push `guide-v0.1.0` (new `guide-v*` namespace; does not touch `theory-v*`, `v1.0.0`, or `docs-v1.0.0-r1`), then HTTP + browser verify `/` and `/api/`.

## Graph update

Scoped deterministic merge with the interpreter in `graphify-out/.graphify_python`: `extract([merge_pages_trees.py, check_pages_links.py, README.md, docs/INDEX.md, docs/VERSIONED_DOCS_1.0.0.md], root='.', parallel=False)` after a temp-copy dry run, then `build_merge(..., dedup=False)` + persist. Result: graph `16919 nodes / 39541 links / 38 hyperedges` (was `16887/39505/38`); new-file nodes present (6 merge-script, 21 checker). Per-node `community` fields and 1059 labels preserved; `graphify-out/` is git-ignored so nothing commits. `graphify export html` was not rerun (graph > 5000-node aggregate limit; HTML remains the prior aggregate view).

## Open findings

- The three excluded link patterns are pre-existing theory-notebook rendering behavior, identical on the live site; fixing them means editing ~96 notebooks — out of Task 9 scope, flagged for a future theory-cleanup task, not a deploy blocker.
- `docs/public/*` + `mkdocs-api.yml` substance edits belong to unlanded Task 8; this task's builds consumed them from the working tree and the merge/audit passed, but Task 9 claims no ownership of that content.
