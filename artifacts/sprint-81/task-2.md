# Sprint 81 — Task 2 handoff evidence

**Status:** `[~]` — implementation is ready for Main's evidence review; this artifact is not a review verdict and the sprint plan remains unchanged.

## Scope and result

The release workflow is now dispatch-only from `main`; a separate job creates/pushes the stable tag only after release-readiness, quality, documentation, and package-build checks pass, and GitHub publication depends on that job. Direct tag pushes no longer trigger publication. The preflight fails closed on incomplete required sprint gates, incomplete/failed bounded-core evidence, malformed or missing external-prerequisite data, an unverified stable-tag ruleset, or an unverified PyPI Trusted Publisher.

The accepted claim boundary remains limited to the Sprint 80 bounded ordinary-DL core. The report's non-1.0-approval statement and blocked/non-gating SmolVLA state are required; neither GPU/VLA behavior nor a general 1.0 approval is claimed. The prerequisite manifest intentionally remains `pending` for both `release-tag-ruleset` and `pypi-trusted-publisher` (environment `pypi`), so this checkout is not release-authorized. The manifest can be changed to `ready` only with actual owner-controlled verification evidence; the guard is not a permanent blocker.

No tag was created or pushed, and no GitHub/PyPI release or upload was performed.

## Changed files

- `.github/workflows/release.yml` — dispatch-only, gate/build before tag, publish after tag.
- `scripts/check_release_readiness.py` — fail-closed evidence and external-prerequisite preflight.
- `docs/release-gates.json` — pending tag-ruleset and PyPI Trusted Publisher evidence.
- `tests/test_release_readiness.py` — readiness, missing evidence, and pending external-control behavior.
- `tests/test_release_workflow_contract.py` — trigger/order/publish contract assertions.
- `docs/M14_REAL_SYSTEM_VALIDATION.md` — documented audited workflow, bounded claim boundary, and external control blockers.
- `artifacts/sprint-81/task-2.md` — this evidence handoff; the task stays `[~]`.
- `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`, `graphify-out/graph.html`, `graphify-out/manifest.json`, and Graphify's dated backup under `graphify-out/2026-09-26/` — project graph refresh.

## Verification

- `uv run pytest tests/test_release_readiness.py tests/test_release_workflow_contract.py -q` — **10 passed in 0.31s**.
- `uv run python scripts/check_release_readiness.py` — **exit 1 as expected**. It reports the two external prerequisites as `pending` and Sprint 81 tasks 2, 3, 4, and 8 as incomplete. This is the current fail-closed behavior, not a release failure to suppress.
- Graph update: `graphify update .` completed its code extraction (345 files; 328 produced no AST nodes), followed by a focused merge for the changed release guard/tests, manifest, M14 document, and workflow semantics. The merge preserved 16,456 unrelated nodes, 37,895 unrelated edges, and 36 unrelated hyperedges; the existing M14 validation-lanes hyperedge was retained.
  The handoff-artifact merge then preserved all 16,509 existing nodes, 37,976 edges, and 37 hyperedges. Targeted graph manifest checks showed no pending AST or semantic entries for the updated sources.
- `graphify cluster-only .` after adding the handoff refreshed `graph.json` and `GRAPH_REPORT.md`: **16,516 nodes, 37,984 edges, 37 hyperedges, 1,080 communities**. The aggregate HTML view was regenerated with **1,080 community nodes and 1,483 cross-community edges**. A graph-integrity check found **0 dangling edges and 0 dangling hyperedges**.

## Open findings / blockers

- The repository stable-tag ruleset is not verified; `release-tag-ruleset` remains `pending`. Task 5 must verify a ruleset that blocks direct bypass writes while permitting the audited workflow.
- PyPI Trusted Publisher activation is still reported pending; task 5 must verify actual activation in the `pypi` environment. No upload was attempted.
- The release workflow still has hard-coded `0.9.0` asset paths and metadata. This was intentionally deferred to task 3's version-metadata cutover; task 3 must update and verify those references before task 5.
- Sprint 81 task 2 remains `[~]`; tasks 3, 4, and 8 are also still open. The guard correctly blocks until the prerequisite tasks and evidence are complete.
- `graphify label` was attempted; all 12 community-label batches failed on the Gemini free-tier quota (HTTP 429). A subsequent handoff-driven re-cluster changed the community set to 1,080 without another label attempt, so labels may be stale and should be refreshed when quota is available.
