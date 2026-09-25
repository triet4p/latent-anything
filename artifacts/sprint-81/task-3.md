# Sprint 81 Task 3 — 1.0.0 candidate metadata and release documentation

**Task:** Finalize version metadata, changelog, release notes, API reference, migration guide, plugin SDK, model/LeRobot guides, and theory coverage report.

**Plan status:** Task 3 remains `[~]`; Main owns the plan checkbox and evidence review. This handoff does not mark it complete or authorize a release.

## Outcome

The working-tree package and runtime version are aligned at `1.0.0`; `uv.lock` was refreshed and a versioned API-freeze snapshot was generated at `artifacts/api_freeze_snapshot_1.0.0.json` (SHA-256 `46ef4edd2bcb5d6235a02633caba10408b6e2000d5614928bef0794b20b005f4`). The historical `0.9.0` release-time API inventory and `0.1.0-beta.1` snapshot remain intact. The root changelog keeps its candidate changes under `[Unreleased]`; `docs/RELEASE_NOTES_1.0.0.md` is explicitly a draft, not a release announcement.

User-facing docs consistently distinguish the published `0.9.0` baseline from the unpublished `1.0.0` candidate. Candidate evidence claims remain limited to the signed-off ordinary-DL encoder v3 and transformer target-evidence-v2 cases plus the separate stability supplement. SmolVLA remains **BLOCKED** and non-gating. No broad VLA, GPU/CUDA, arbitrary-model, or dataset-support claim is made. The compatibility docs and runtime metadata retain aliases through `1.x`; removal requires a separately reviewed major-version migration. The release workflow is pinned to the `1.0.0` candidate while preserving its dispatch-only, fail-closed gate path.

No external prerequisite status was changed. No tag, push, build, upload, workflow run, or publication was performed.

## Changed files

- Version/API metadata: `pyproject.toml`, `uv.lock`, `src/latent_anything/__init__.py`, `src/latent_anything/_api_freeze_runtime.py`, `src/latent_anything/registry_aliases.py`, `scripts/api_freeze_snapshot.py`, and `artifacts/api_freeze_snapshot_1.0.0.json`.
- Release contract: `.github/workflows/release.yml`, `CHANGELOG.md`, `README.md`, and new draft `docs/RELEASE_NOTES_1.0.0.md`.
- Guidance: `docs/API_REFERENCE.md`, `docs/API_COMPATIBILITY.md`, `docs/MIGRATION.md`, `docs/PLUGIN_AUTHOR_GUIDE.md`, `docs/PLUGIN_TEMPLATE.md`, `docs/AI_ENGINEER_GUIDE.md`, `docs/LEROBOT_INTEGRATION.md`, `docs/EVIDENCE_GAP_PLAN.md`, `docs/PLAN.md`, and `docs/INDEX.md`.
- Focused contract tests: `tests/test_api_freeze_snapshot.py`, `tests/test_api_compatibility.py`, `tests/test_latent_anything/test_package.py`, and `tests/test_release_workflow_contract.py`.
- Project graph: `graphify-out/graph.json` and `graphify-out/manifest.json`.
- This handoff: `artifacts/sprint-81/task-3.md`.

`docs/release-gates.json`, the sprint-plan checkbox, the historical API snapshots, and the Sprint 80 evidence artifacts were intentionally left unchanged.

## Focused verification

- `uv lock --offline` resolved the lock and updated `latent-anything` from `0.9.0` to `1.0.0`.
- `uv run --offline --no-sync python scripts/api_freeze_snapshot.py --check` passed: `snapshot clean (46ef4edd2bcb5d6235a02633caba10408b6e2000d5614928bef0794b20b005f4)`.
- Focused compatibility/package/workflow checks passed:

  ```text
  uv run --offline --no-sync pytest -q tests/test_api_freeze_snapshot.py tests/test_api_compatibility.py tests/test_registry_migration.py tests/test_latent_anything/test_package.py tests/test_release_workflow_contract.py
  23 passed in 7.49s
  ```

- A preliminary run exposed one test that asserted obsolete exact Markdown wording rather than consumer-visible behavior. That text-only assertion was removed, not repinned; the final focused run above passes. Snapshot, runtime alias policy, package-version metadata, and release-workflow contracts remain behaviorally checked.
- A focused metadata/document smoke check confirmed `pyproject.toml`, `uv.lock`, and the candidate snapshot all report `1.0.0`; the snapshot digest is current; alias policy is `retain-through-1.x`; release-tag-ruleset and PyPI Trusted Publisher (`pypi`) prerequisites remain `pending`; Task 3 remains `[~]` and Task 4 remains pending. A local Markdown-link audit across 13 changed Markdown documents checked 258 targets with 0 missing links.
- The actual release-notes extraction command was exercised without publishing: `uv run --offline --no-sync python scripts/extract_release_notes.py v1.0.0 --body-file graphify-out/.task3_release_body.md`. It exits with `ValueError: No CHANGELOG.md section found for release tag 'v1.0.0' (normalized to '1.0.0')`. This is intentionally unresolved here: a dated release section must not be fabricated before an actual release date exists. See open findings below.
- No project-wide tests, formatter, linter, MkDocs build, wheel/sdist build, clean install, optional-extra install, remote/GPU run, or release workflow were run. Task 4 owns clean build/install validation.

## Graph update

- Followed the project's incremental Graphify procedure for 24 changed/added target paths. Deterministic AST extraction covered 22 eligible code/Markdown files and returned 203 nodes, 672 candidate edges, and no errors. Five candidate edges whose undirected endpoint pairs already had provenance from unchanged sources were omitted; the merge used the remaining 667 AST edges to retain the pre-existing relationships.
- Added 15 focused inline semantic nodes and 22 candidate semantic edges for the current candidate, bounded evidence scope, alias/plugin policy, blocked SmolVLA lane, pending release prerequisites, candidate snapshot, and draft release notes. Preserved the 22 prior semantic nodes, 22 prior semantic edges, and the existing target-source hyperedge for refreshed files. The task handoff itself was not indexed, following the prior Sprint 81 Task 1 graph-update scope.
- Graph changed from **16,516 nodes / 37,984 links / 37 hyperedges** to **16,569 / 37,867 / 37**. Existing community assignments were preserved without reclustering. All **16,333 unrelated nodes**, **37,424 unrelated edges**, and **36 unrelated hyperedges** were preserved; unrelated node/edge/hyperedge mismatches were 0.
- The manifest changed from **1,792** to **1,794** rows for the 24 targeted paths; non-target manifest changes were 0. Post-write inspection found 37 target-source semantic nodes and 44 target-source semantic edges; all 15 new semantic nodes have repo-relative provenance in the graph.
- Graphify's read-only merged-graph diagnostic reported zero missing/dangling endpoints, self-loops, exact duplicate edges, or same-endpoint collapses. It reported one unverified code node, `parametrize`, with no owning `source_file`; this was emitted as a source-less external symbol by deterministic extraction from the changed tests and remains a diagnostic note, not a release gate.
- Focused query `graphify query "Draft unpublished 1.0.0 notes for bounded ordinary-DL scope" --budget 500` exited 0 and retrieved the candidate notes, README scope, migration policy, API snapshot, and Sprint 80 depth report among its first 13 of 104 nodes (the query output was budget-truncated). A broader candidate query also surfaced the historical “evidence review pending” title from untouched `artifacts/task_80.23_encoder_end_to_end_proof_summary.md`; that old, differently sourced node was deliberately preserved and is not attributed to any refreshed Task 3 file.

## Open findings / blockers

1. The external stable-tag ruleset and PyPI Trusted Publisher (`pypi` environment) prerequisites remain **pending** in `docs/release-gates.json`; no owner-verified evidence was supplied or changed.
2. Sprint 81 Task 4 clean wheel/sdist and base/optional-extra installation checks remain pending and were not run here.
3. Before Task 5 dispatches the actual `v1.0.0` release workflow, the changelog needs a dated `## [1.0.0] - <actual release date>` section for `scripts/extract_release_notes.py`. The current `[Unreleased]` candidate entry and draft release-notes page intentionally do not invent that date; the extraction failure above is observed and must be resolved at the actual publication boundary.
4. The graph's one source-less `parametrize` code node and the historical stale review-title retrieval are noted above; neither was rewritten as unrelated source history.
