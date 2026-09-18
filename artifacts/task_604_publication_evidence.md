# Sprint 79 task 604 publication evidence

## Outcome

Task 604 remains **open**. The reviewed candidate was pushed and tagged exactly, but the audited GitHub release workflow failed in its remote pytest gate. No GitHub Release or PyPI publication is claimed, and Sprint 79 remains open. No product code or Sprint 80/81 scope was changed.

## Candidate, branch, and tag

- Reviewed candidate: `47f4486bc0e8cd429c452f36a0847d60b70a099f`
- Candidate subject: `docs(plan): mark release gate complete`
- Remote branch: `sprint79-local-gate-remediation`
- Remote branch URL: https://github.com/triet4p/latent-anything/tree/sprint79-local-gate-remediation
- Candidate branch push result: `07ac285..47f4486 sprint79-local-gate-remediation -> sprint79-local-gate-remediation`
- Tag: `v0.9.0`
- Tag kind: lightweight commit ref
- Remote tag verification (`git ls-remote origin refs/tags/v0.9.0`):
  `47f4486bc0e8cd429c452f36a0847d60b70a099f refs/tags/v0.9.0`
- Tag target therefore equals the reviewed candidate exactly.

## Audited release workflow

- Workflow name: `Release`
- Workflow file: https://github.com/triet4p/latent-anything/blob/47f4486bc0e8cd429c452f36a0847d60b70a099f/.github/workflows/release.yml
- Run ID: `35331167216`
- Run URL: https://github.com/triet4p/latent-anything/actions/runs/35331167216
- Event/ref: tag push `v0.9.0`
- Job: `Release gate and GitHub Release` (job ID `105555596600`)
- Conclusion: **failure**
- Failed step: `Pytest`, exit code `1`
- Remote result: `37 failed, 2157 passed, 67 skipped, 39 warnings in 450.17s`
- Exact failure evidence: `gh run view 35331167216 --repo triet4p/latent-anything --log-failed`
- Representative failures include API-freeze CLI defaults drift, canonical Stage-B policy unavailable, and Linux execution of Windows-only `ssh.exe` transport tests. The full immutable failed-run log is at the run URL above. Because the workflow failed before `Extract release notes` and `Create GitHub Release`, no GitHub Release was created.

## GitHub Release and assets

- Expected release URL: https://github.com/triet4p/latent-anything/releases/tag/v0.9.0
- Verification command: `gh release view v0.9.0 --repo triet4p/latent-anything`
- Result: `release not found` (exit code `1`)
- GitHub Release metadata/assets: **not published**. No asset URLs or GitHub-hosted digests exist to verify.

## Package hashes and PyPI state

Candidate-gate expected files and SHA-256 values:

| File | Expected SHA-256 |
|---|---|
| `latent_anything-0.9.0-py3-none-any.whl` | `18b82bed4520c094c2de41c1f1ab78c1c9a993635d6082371ed2020deac9c117` |
| `latent_anything-0.9.0.tar.gz` | `6e43f91cff8e2d4f8f73f82044e31e0282ac26dc5813bb42168e31019a0d36ad` |

- PyPI project/version URL: https://pypi.org/project/latent-anything/0.9.0/
- JSON verification URL: https://pypi.org/pypi/latent-anything/0.9.0/json
- Verification result: HTTP `404`; version is not published.
- A disposable rebuild was intentionally **not uploaded** because archive bytes did not match the reviewed immutable hashes: rebuilt wheel `5eb811f28c4bdef7918e82062b8a7e8dd8c2ac09b8da2aa92f506a784c3574da`, rebuilt sdist `00294d70da8c205f188f12cfa6757b1a51ee8c3f4eb5a5dc55610ad840876809`. The candidate binaries were removed during the line-603 cleanup, so publishing these different bytes would violate provenance.
- No `UV_PUBLISH_*`, `TWINE_*`, or `PYPI_*` credentials were present in the publication environment. No PyPI upload command was run.

## Install smoke evidence

The candidate-gate artifact records clean isolated local wheel and sdist installs from the candidate build, each importing `latent_anything` and asserting `latent_anything.__version__ == "0.9.0"`. These are **candidate-local** smoke results, not published-distribution evidence. A published clean-install smoke cannot be performed while PyPI returns HTTP 404 and no GitHub Release assets exist.

## Plan state and worktree

- `docs/sprint-plans/sprint-79.md` task 604 remains `[ ]` and Sprint 79 remains open.
- `docs/PLAN.md` Milestone 14 remains `[ ]`; Sprint 79 is not moved to completed.
- This artifact records the concrete external/remote blocker without fabricating publication or closure.
- No local project-wide lint, type, test, documentation, or build suite was run. The only release-gate suite observed was the audited GitHub Actions run above; the disposable packaging rebuild was used only to prevent uploading a hash-mismatched artifact.
- Evidence commit: `8e2e595ed6d035977995713cb9699a7189487111`
- Final branch push: `sprint79-local-gate-remediation` remote head
  `8e2e595ed6d035977995713cb9699a7189487111`
- Final worktree verification: `git status --short --branch` reported only
  `## sprint79-local-gate-remediation...origin/sprint79-local-gate-remediation`
  (no file changes).
