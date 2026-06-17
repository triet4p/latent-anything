# Changelog

## [Unreleased]

### Added

- Package scaffold: `src/latent_anything/` (src-layout) with root-level `pyproject.toml` and PEP 621 metadata.
- Development tooling: `ruff` (lint + format), `pyright` (strict type-checking), `pytest`, and `hypothesis` configured in `pyproject.toml`.
- Smoke test suite: `tests/test_latent_anything/test_package.py` verifying import and version.
- CI workflow (`.github/workflows/ci.yml`) running `ruff check`, `ruff format --check`, `pyright`, and `pytest` on Python 3.12, 3.13, and 3.14.
- README updated with package installation via `uv`, Quick Start example, and current project structure.
- Python version range adjusted: `requires-python` set to `>=3.12,<3.15`, local development targets Python 3.13.
- Deploy workflow restricted: `deploy-latent-anything-theory.yml` now triggers only on `theory-v*` tag pushes or manual `workflow_dispatch`.
