# Sprint 78.40 — API-freeze closure checkpoint

## Verdict

**PASS-WITH-WARNINGS for the Sprint 78 closure checkpoint; release remains
blocked.** The owner-level API-freeze decision is appended to
[`decisions.md`](../.agents/memory/decisions.md). It treats the checked-in
snapshot and compatibility documentation as the reviewed contract boundary,
but authorizes no alias removal, version bump, tag, publication, or
release-readiness claim.

## Decision and status reconciliation

The decision is grounded in snapshot digest
`48d64721b73a9d0c9e73da4a41940008c70dfa7841e500bc11bc8dcd22ddf7f6`:

- Current runtime surface: **205** top-level exports.
- Canonical-stable projection: **202** entries.
- Human compatibility ledger: **18 alias rows**; the **2 schema/path data
  migrations are separate** and excluded from that alias count.
- Built-in registry: **32** rows; plugin groups: **5** with API version 1;
  optional profiles: **12**.

The current package metadata remains `0.1.0b1`. The `0.9.0` pre-stable epoch
is planned, not released; RFC0001's `0.2.0` planning window was never
published. Sprint 80 remains responsible for the stable `1.0.0` decision.

Status is synchronized in:

- [`CHANGELOG.md`](../CHANGELOG.md), with an Unreleased checkpoint entry;
- [`sprint-78.md`](../docs/sprint-plans/sprint-78.md), where task 78.40 and the
  API-freeze decision/full-gate item are complete while package publication
  remains pending;
- [`PLAN.md`](../docs/PLAN.md), which records the compatibility-controlled
  checkpoint without treating it as release readiness; and
- [`M14_REAL_SYSTEM_VALIDATION.md`](../docs/M14_REAL_SYSTEM_VALIDATION.md),
  which records the same stop-before-release boundary in the Vietnamese
  project-level contract.

## Authoritative closure gates

| Gate | Result |
|---|---|
| `uv sync --locked --extra viz` | PASS; CI-equivalent test profile resolved |
| Fresh `uv run pytest -q` | **1563 passed, 36 skipped, 39 warnings** in 211.90s |
| `uv run ruff check src tests scripts` | PASS |
| `uv run ruff format --check src tests scripts` | PASS; 320 files already formatted |
| Strict Pyright | PASS; 0 errors, 0 warnings, 0 informations |
| API snapshot `--check` | PASS; digest unchanged |
| API drift tests | PASS; 14 passed |
| Evidence validator | PASS integrity / honest gate failure: 25/63 core (39.7%), 25/65 overall (38.5%) |
| Strict MkDocs | PASS with only upstream Material warning; temporary site/index cleaned |
| Scoped Markdown links | PASS; 231 links checked, 0 broken |
| `git diff --check` | PASS; line-ending notices only |
| Package build/smoke | Deferred to Sprint 80 clean-checkout release gate; Sprint 78 plan does not authorize publication |

The full pytest run was executed fresh after the owner correction, not reused.
The only warnings are existing sklearn/UMAP diagnostics and expected registry
deprecation warnings for the retained beta aliases.

## Release and evidence blockers

The validator remains honest at 25/63 core and 25/65 overall. Named 3DGS and
corrected SmolVLA reruns, model/checkpoint/license/access prerequisites, and
the external GitHub Actions/account blocker remain unresolved. These block
release-readiness and publication; they do not invalidate the API snapshot or
the compatibility checkpoint. No public API, runtime behavior, serialization
bytes, version metadata, tag, or release artifact was changed.

## Graph

After the final artifact, plan, changelog, and status edits, Graphify was
refreshed. The clustered report contains **11,111 nodes / 21,232 edges / 940
communities**; the unclustered AST graph contains 11,111 nodes and 23,299
links. Graphify reported 53 JSON sidecars with zero extracted nodes; these are
known extraction warnings and do not affect the closure gates.

No commit or push was performed. This task stops for owner review.
