# Task Summary: Sprint 26 Task 3 - Release Title And Body Contract

**Sprint:** Sprint 26
**Task:** Task 3

## Summary of Work

Defined a tested release metadata contract: pushed tags normalize an optional leading `v`, the matching `CHANGELOG.md` section is required, the extracted section becomes the release body, and the title is explicit as `Latent Anything <version> - Core latent-space framework beta` for prerelease tags.

## Files Modified

* [scripts/extract_release_notes.py](scripts/extract_release_notes.py) - Implements changelog section extraction, title generation, prerelease detection, and GitHub Actions outputs.
* [tests/test_latent_anything/test_release_notes.py](tests/test_latent_anything/test_release_notes.py) - Covers the release metadata contract.
* [.github/workflows/release.yml](.github/workflows/release.yml) - Consumes extracted title/body/prerelease metadata when creating the GitHub Release.
* [docs/sprint-plans/sprint-26.md](docs/sprint-plans/sprint-26.md) - Marks Task 3 complete.

## Testing

* **Test File:** [tests/test_latent_anything/test_release_notes.py](tests/test_latent_anything/test_release_notes.py)
* **Status:** Passed focused release metadata tests; full gate remains Task 11.
* **Execution Command:** `uv run pytest tests/test_latent_anything/test_release_notes.py -v`

## Additional Notes

The workflow will fail before GitHub Release creation when a pushed package release tag has no matching changelog section.
