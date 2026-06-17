# Task Summary: Sprint 3 — Task 10 — Python Version Range Adjustment

**Sprint:** Sprint 3
**Task:** Task 10 — Điều chỉnh Python version range

## Summary of Work
Updated the Python version constraints across the project:
- **Minimum:** raised from 3.11 → **3.12**
- **Maximum:** set to **3.14** (`<3.15`)
- **Local development:** pinned to **Python 3.13** via `.python-version`
- Recreated `.venv` with Python 3.13 (was 3.12)
- Updated all tooling configurations to reflect the new target version

## Files Modified

| File | Change |
|---|---|
| `pyproject.toml` | `requires-python` → `>=3.12,<3.15`; `ruff.target-version` → `py312`; `pyright.pythonVersion` → `3.12` |
| `.github/workflows/ci.yml` | CI matrix → `["3.12", "3.13", "3.14"]` |
| `.claude/rules/python.md` | Minimum Python version → 3.12; PEP 723 example → `>=3.12` |
| `docs/sprint-plans/sprint-3.md` | Tasks 1-9 marked [x]; Task 10 added with description |
| `CHANGELOG.md` | Added entry for version range adjustment |
| `.python-version` | **Created** — `3.13` (gitignored, local-only) |

## Testing
- **Python version:** 3.13.3 (local)
- **Tooling gate:** `ruff check` → pass, `ruff format --check` → pass, `pyright` → 0 errors, `pytest` → 2 passed
