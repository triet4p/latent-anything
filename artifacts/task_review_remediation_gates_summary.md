# Review remediation: gates and documentation

## Scope

Fixed repository formatting failures, synchronized Milestone 12 plan statuses, declared MkDocs dependencies, and narrowed evidence claims.

## Verification

- `ruff check src tests scripts`: pass.
- `ruff format --check src tests scripts`: pass.
- `pyright`: pass with 0 errors.
- Focused remediation tests: pass.
- `uv run --extra docs mkdocs build --strict`: pass.
