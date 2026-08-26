# Sprint 78 Atomic Task 78.21 — SmolVLA Ruff B009 Remediation

Status: complete (test-only lint remediation; no product behavior changed).

## Change

Replaced the constant-string `getattr` in
`tests/test_lerobot_smolvla.py::test_smolvla_public_api_and_result_schema_snapshot`
with direct attribute access. The assertion still checks that the facade's
private compatibility alias is exactly the expected hook-session class; it was
not weakened or removed.

## Validation

- Focused SmolVLA compatibility test: **passed**.
- Repository Ruff check: **pass**.
- Ruff format check: **pass**.
- Strict Pyright on `src` and `tests`: **0 errors, 0 warnings, 0 informations**.
- `git diff --check`: **pass** (normal LF/CRLF conversion warnings only).
- Immediate final graphify: **10,659 nodes / 20,688 edges / 923 communities**;
  graphify reported 50 JSON files producing zero graph nodes.
- No full suite was run because this is a test-only syntax-equivalent change.
- No commit, push, network, model, remote, or product-source changes.
