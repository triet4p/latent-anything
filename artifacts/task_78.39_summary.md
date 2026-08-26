# Sprint 78.39 — Snapshot-grounded migration guide and API reference

## Verdict

**PASS-WITH-WARNINGS for the documentation/API-entry-point task.** The
user-facing migration guide and API reference are now present in English and
link to the checked-in API-freeze snapshot and compatibility ledger rather
than duplicating generated symbol truth. No source, runtime behavior, public
signature, serialized byte, alias-removal, version, tag, or release change was
made.

## Files and authoritative sources

- [`docs/MIGRATION.md`](../docs/MIGRATION.md) explains migration from the
  published metadata state `0.1.0b1` (`0.1.0-beta.1`) to the planned pre-stable
  `0.9.0` compatibility epoch. It explicitly states that `0.2.0` was an RFC
  planning window that was never published and that `0.9.0` is not released,
  tagged, or publishable while the external Actions/account blocker remains.
- [`docs/API_REFERENCE.md`](../docs/API_REFERENCE.md) is the human-readable
  contract index. The machine-readable source remains
  [`artifacts/api_freeze_snapshot_0.1.0b1.json`](api_freeze_snapshot_0.1.0b1.json),
  with alias/deprecation policy in
  [`docs/API_COMPATIBILITY.md`](../docs/API_COMPATIBILITY.md).
- [`docs/INDEX.md`](../docs/INDEX.md) and [`README.md`](../README.md) link the
  two entry points. M14 and Sprint 79 also link them as navigation aids.

## Snapshot-derived coverage

The checked-in snapshot digest is
`48d64721b73a9d0c9e73da4a41940008c70dfa7841e500bc11bc8dcd22ddf7f6`.

| Contract surface | Snapshot value | Guide coverage |
|---|---:|---|
| Runtime top-level exports | 205 | Runtime surface and stability boundary |
| Canonical stable projection | 202 | Three canonical additions identified |
| Human-ledger alias rows | 18 | Expanded from snapshot section B; `lambda` and `lambda_` are separate rows |
| Built-in registry rows | 32 | Registry naming and factory contract |
| Plugin groups / API version | 5 / 1 | Entry-point compatibility |
| Optional profiles | 12 | Profile names and lazy-boundary policy |
| Config schemas | 28 | Config/alias compatibility |
| Dataclass/result schemas | 81 | Result/property compatibility |
| CLI commands | 5 | Aliases, success/error exit behavior |
| Serialization families | 5 | Versions and two data migrations |
| Data migrations | 2 | Separate schema/path migrations; not included in the 18 alias rows |
| Sync/async pairs | 9 | Pairing contract |
| Custom exception types | 7 | Public error taxonomy |

The human-facing ledger contains **18 alias rows**, expanded from snapshot
section B: 3 canonical symbol rows + 2 registry-kind rows + 2 CLI rows + 2
config rows (`lambda` and `lambda_`) + 3 result-property rows + 6 transition
property rows = 18. The **2 schema/path data migrations are separate** and are
not included in that alias-row count. The source ledger remains authoritative
for exact spelling and policy.

## Gates and reuse rationale

- API snapshot check: **PASS**; the digest above is unchanged.
- Focused API/compatibility and serialization/CLI/registry checks: **PASS**;
  the selected contract suite reported **47 passed**.
- Ruff check and format check: **PASS** (`320 files already formatted` in the
  latest authoritative run).
- Strict Pyright: **PASS**, zero errors in the latest authoritative run.
- Markdown/link and strict MkDocs checks: **PASS**; no broken local links or
  strict-build errors.
- `git diff --check`: **PASS**.
- Full locked-viz pytest evidence reused from the immediately preceding clean
  source/test tree: **1563 passed, 36 skipped, 39 warnings**. Task 78.39 only
  changed English docs, navigation, plan text, and this artifact, so a second
  full behavioral run would not add coverage.

## Remaining blockers

This task does not promote evidence or release the package. The theory ledger
remains at **25/63 core (39.7%)** and **25/65 overall (38.5%)**; named 3DGS and
corrected SmolVLA reruns, model/checkpoint/license access, and the external
GitHub Actions/account blocker remain open. Metadata stays `0.1.0b1`; Sprint 80
targets `1.0.0`; no `v0.9.0` tag or publication is implied by these documents.

## Graph

After the artifact and plan/navigation updates, Graphify refresh completed with
**11,103 nodes / 21,221 edges / 972 communities**. The refresh reported 53
JSON sidecars with zero extracted nodes; these are extraction warnings and do
not affect the API snapshot or documentation claims.

The task stops here for owner review; no commit or push was performed.
