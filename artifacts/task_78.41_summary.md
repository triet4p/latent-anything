# Sprint 78.41 — M14 contract-path remediation

## Scope and verdict

This bounded remediation reconciles the final Sprint 78 status conflict and
keeps the M14 24-lane execution table executable against the current checkout.
It changes no implementation behavior and does not promote any evidence level.

## Corrections

- Marked the completed API-freeze snapshot and quality-audit tasks in
  `docs/sprint-plans/sprint-78.md` as complete, while leaving publication
  pending and preserving all release blockers.
- Fixed the owner-level API-freeze paragraph typo in the Sprint 78 notes.
- Replaced stale M14 module and test paths for L03, L04, L05, L06, L07, L08,
  L09, L11, L12, L13, L14, L15, L16, L17, L18, L19, L20, L22, and L23.
- Replaced placeholder commands in L02, L05, L07, L12, L15, L16, L17, L19,
  and L20 with exact repository-relative pytest or inspection commands.
- Kept L17 explicitly blocked until a named 3DGS checkpoint is provisioned;
  kept real-model/license/access blockers and planned statuses unchanged.
- Added `tests/test_m14_validation_contract.py`, a focused deterministic check
  that extracts every supported repository-relative `.py`, `.md`, `.toml`,
  `.yml`, and `.yaml` path from every M14 lane code span—including multiple
  workflow paths—then validates existence and pytest selectors (normal or
  async). Future `artifacts/m14/*` outputs and model/checkpoint identifiers are
  intentionally excluded. A dedicated regression test proves a missing
  non-Python path is rejected by the same guard.

## Verification

- `uv run pytest tests/test_scripts/test_validate_evidence_ledger.py tests/test_m14_validation_contract.py -q`: **12 passed**.
- `uv run ruff check src tests scripts`: **PASS**.
- `uv run ruff format --check src tests scripts`: **PASS; 321 files already formatted**.
- `uv run pyright`: **0 errors, 0 warnings, 0 informations**.
- `uv run mkdocs build --strict --site-dir .mkdocs-review-site`: **PASS**;
  only the known upstream Material warning; temporary output removed.
- Focused Markdown scan over the changed contract/artifact docs: **54 links,
  0 broken** (planned `artifacts/m14/*` outputs excluded).
- `uv run python scripts/validate_evidence_ledger.py --json`: **integrity exit
  0**; coverage remains honestly 25/63 core and 25/65 overall.
- `git diff --check`: **PASS**; line-ending notices only.
- `graphify update .`: completed AST refresh; the existing 53 JSON-sidecar
  zero-node warnings remain non-blocking.

No commit, stage, push, evidence promotion, or release authorization is part
of this remediation.
