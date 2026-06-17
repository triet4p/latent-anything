# Task Summary: Sprint 3 — Task 11 — Restrict Deploy Workflow to Tags + Manual

**Sprint:** Sprint 3
**Task:** Task 11 — Sửa `deploy-latent-anything-theory.yml` trigger

## Summary of Work
Changed the deploy workflow trigger from `push: branches: [main]` (with path filters) to `push: tags: ["theory-v*"]` + `workflow_dispatch`. This prevents automatic deployment on every commit to `main` that changes theory files, and instead requires an explicit tag push (e.g., `theory-v0.1.0`) or manual trigger.

## Files Modified

| File | Change |
|---|---|
| `.github/workflows/deploy-latent-anything-theory.yml` | Removed `push: branches: [main]` with `paths`; added `push: tags: ["theory-v*"]` |
| `docs/sprint-plans/sprint-3.md` | Task 11 added and marked [x] |
| `CHANGELOG.md` | Added entry for deploy workflow restriction |

## Testing
- **Validation:** Manual inspection of YAML syntax — no code change to test.
- CI trigger logic verified: push to `main` without a matching tag **will not** trigger deploy; only `theory-v*` tags or `workflow_dispatch` will.
