# Task Summary: Sprint 81 Task 8 — User-Facing Support and Release Policies

**Sprint:** 81  
**Task:** 8 — State semantic-versioning, deprecation, security-reporting, upstream-compatibility, and artifact-migration policies (`[~]`)

## Summary of Work

Added a canonical user-facing `docs/SUPPORT_POLICY.md` covering the conditional post-release 1.x support boundary; SemVer behavior; deprecation and major-version removal rules; bounded upstream compatibility and only documented/evidence-backed optional profiles; and explicit artifact writer, reader, and migration behavior. Added `SECURITY.md` without inventing a contact or claiming non-maintainer access to GitHub private reporting is verified. Linked both policies from the README and document index, and cross-referenced the support policy from compatibility, migration, artifact, plugin, integration, and candidate release-note docs. The policy remains explicitly conditional: `0.9.0` is the latest published package and the `1.0.0` candidate is not a release or publication authorization.

The canonical policies and their cross-references were previously added to the graph for the ten changed Markdown sources. After the owner-managed Task5 graph refresh, this Task8-only pass added this handoff’s deterministic Markdown structure (six AST nodes and five AST edges) plus one explicitly semantic rationale node with references to `docs/SUPPORT_POLICY.md` and `SECURITY.md`. The merge preserved all prior graph nodes and unrelated edges; only this artifact’s manifest row changed.

## Files Modified

- `docs/SUPPORT_POLICY.md` — canonical support, SemVer, deprecation, security reporting, upstream, and versioned-artifact policy.
- `SECURITY.md` — current reporting-route status, safe reporting guidance, and owner action without a fabricated mailbox or alternate channel.
- `README.md`, `docs/INDEX.md` — discoverable links to the policy documents.
- `docs/API_COMPATIBILITY.md`, `docs/MIGRATION.md`, `docs/PORTABLE_ARTIFACTS.md`, `docs/PLUGIN_AUTHOR_GUIDE.md`, `docs/OPTIONAL_INTEGRATIONS.md`, `docs/RELEASE_NOTES_1.0.0.md` — cross-references and alignment with the central policy.
- `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`, `graphify-out/graph.html`, `graphify-out/.graphify_analysis.json`, `graphify-out/.graphify_labels.json`, `graphify-out/.graphify_labels.json.sig`, `graphify-out/manifest.json` — refreshed project graph outputs and manifest.
- `graphify-out/2026-09-26/` — Graphify's automatic dated safety backup; retained.
- `artifacts/sprint-81/task-8.md` — this task's evidence and handoff record.

No sprint-plan checkbox, release rule, tag, package, upload, or publication was changed.

## Verification

- **Focused documentation/policy smoke:** the final `uv run --locked --no-sync python -c` check passed over 11 Markdown files; 103 relative links resolved (`missing=[]`), and policy/migration markers, concrete Python/LeRobot ranges, project version, current enabled-setting/unverified-reporter/no-contact status, and pending release gates passed. Several initial literal marker assertions expected wording not used by the docs; after checking the actual upstream, artifact-schema, and reporter-access wording (`Upstream and optional-integration compatibility`, `Run-record schema-v1`, “has not been verified”), the corrected check passed. An earlier pre-setting-change exact-phrase assertion also failed before the factual wording was updated; corrected current-status checks passed. No `mailto:` or email address was found in `SECURITY.md` or `docs/SUPPORT_POLICY.md`.
- **Policy grounding:** reviewed `pyproject.toml` and workflow declarations: CI matrices cover Python 3.12–3.14, import-check each declared extra, and use Python 3.12–3.13 for LeRobot-specific lanes; the LeRobot guide bounds support to `>=0.6.0,<0.7.0` and records the 0.6.1 pin. The API compatibility ledger lists 18 retained aliases; source handlers/fixtures identify the supported artifact migrations. The policy excludes untested dependency combinations, moving upstream branches, broad VLA/GPU claims, and unsupported format conversions.
- **Security-source check:** Task5HardRuleset verified the GitHub private vulnerability-reporting setting via the admin API (`PUT` returned 204 and `GET` returned `{"enabled":true}` on 2026-09-26). The non-maintainer `Security → Report a vulnerability` route remains untested; no confidential intake is claimed.
- **Release-gate source check:** Read-only `uv run --locked --no-sync python -c` parsed `docs/release-gates.json` and `pyproject.toml`: `release-tag-ruleset=pending`, `pypi-trusted-publisher=pending` with matching `release.yml`/`pypi` configuration verified for OIDC bootstrap, and working-tree version `1.0.0`. No release authorization was inferred.
- **Scoped Graphify merge:** starting from the completed Task5 graph (16,630 nodes / 37,996 edges / 37 hyperedges), the first Task8 handoff merge added six AST nodes/five AST edges and one semantic rationale node/two `references` edges. After the final artifact text was written, the same source was re-extracted with the same 6/5 structure and graph totals remained stable. Final graph: 16,637 / 38,003 / 37; no prior nodes or unrelated edges were lost.
- **Graphify cluster refresh:** `C:/Users/admin/AppData/Local/uv/cache/archive-v0/fSUTpEZ94kzDWKoBJUceQ/Scripts/graphify.exe cluster-only .` completed; the final graph, report, and aggregated HTML view were regenerated (1,105 communities, 1,764 cross-community edges). The first cluster pass renamed 990 community labels by their hub; the final pass reused them. No explicit LLM labeling command was run.
- **Graph integrity:** `C:/Users/admin/AppData/Local/uv/cache/archive-v0/fSUTpEZ94kzDWKoBJUceQ/Scripts/graphify.exe diagnose multigraph --json` — 16,637 nodes and 38,003 edges; zero missing/dangling endpoints, self-loops, exact duplicate edges, or same-endpoint collapse risks. One unrelated unverified node remains.
- **Graph queries:** `graphify.exe query "How does the Task 8 handoff connect the support policy with current security-reporting status and release gates?" --budget 1000` returned the rationale plus support/security policy nodes; the broader traversal was truncated at 26 of 182 results. Focused `graphify.exe path` checks from the rationale to `"SUPPORT_POLICY.md"` and `"SECURITY.md"` each returned the direct one-hop `references` edge.
- **Manifest scope:** this final scoped refresh stamped only `artifacts/sprint-81/task-8.md`; non-Task8 manifest rows changed: 0. At the post-stamp snapshot, `detect_incremental` reported 298 pending changes (65 code, 233 documents); all ten Task8 policy pages and this handoff were not pending. The concurrent Task5 `.github/workflows/release.yml` and `docs/M14_REAL_SYSTEM_VALIDATION.md` edits remained unstamped for that owner’s graph refresh.
- **Task status:** `docs/sprint-plans/sprint-81.md:15` currently reads `[x]`, and its note records `agent://Review81Task8` passing Task 8. This task did not modify the plan checkbox; Main owns any reset or re-review decision after the later reporting-setting and graph updates. No project-wide tests, formatters, linters, or builds were run.

## Findings and Release Status

- **External owner action remains:** the private vulnerability-reporting setting is enabled, but non-maintainer access remains unverified. The repository owner must test the end-to-end reporting route before the project can claim confidential intake. No SLA or substitute email/form is promised.
- **Release remains gated:** `docs/release-gates.json` still lists `release-tag-ruleset` and `pypi-trusted-publisher` as `pending`. The matching PyPI publisher configuration is verified for a first OIDC upload, but no tag, upload, or publication was attempted; the tag ruleset remains absent. The support policy takes effect only after audited gates pass and a `1.0.0` distribution is published; `0.9.0` remains the latest published package.
- **Graphify environment note:** the installed package was 0.9.68 while the local Graphify skill was 0.9.32; Graphify warned about this mismatch. The cluster and query commands completed and their exact outcomes are recorded above.
