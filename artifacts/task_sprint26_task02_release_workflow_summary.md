# Task Summary: Sprint 26 Task 2 - Tag-Driven Release Workflow

**Sprint:** Sprint 26
**Task:** Task 2

## Summary of Work

Added `.github/workflows/release.yml`, a tag-driven GitHub Release workflow for package release tags. It triggers on `v*.*.*` and plain `*.*.*` numeric tags, runs the release gate first, extracts release notes, and creates a GitHub Release without binary artifacts. Beta and rc prerelease status is delegated to the tested release-note extraction script.

## Files Modified

* [.github/workflows/release.yml](.github/workflows/release.yml) - Adds the release gate and GitHub Release creation workflow.
* [scripts/extract_release_notes.py](scripts/extract_release_notes.py) - Provides release metadata consumed by the workflow.
* [tests/test_latent_anything/test_release_notes.py](tests/test_latent_anything/test_release_notes.py) - Verifies tag normalization, prerelease detection, and missing-section failures.
* [docs/sprint-plans/sprint-26.md](docs/sprint-plans/sprint-26.md) - Marks Task 2 complete.

## Testing

* **Test File:** [tests/test_latent_anything/test_release_notes.py](tests/test_latent_anything/test_release_notes.py)
* **Status:** Passed focused release metadata tests; full gate remains Task 11.
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_release_notes.py -v`

## Additional Notes

The workflow intentionally does not publish wheels, sdists, PyPI packages, or other binary artifacts in this sprint.
