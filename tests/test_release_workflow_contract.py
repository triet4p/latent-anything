"""Contract tests for the fail-closed stable-release workflow."""

from __future__ import annotations

from pathlib import Path

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"


_REQUIRED_ASSETS = (
    "dist/latent_anything-1.0.0-py3-none-any.whl",
    "dist/latent_anything-1.0.0.tar.gz",
    "dist/SHA256SUMS",
    "dist/PROVENANCE.json",
    "dist/release-body.md",
)


def _workflow() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def test_release_tag_is_created_only_after_a_dispatch_gate_and_build() -> None:
    workflow = _workflow()
    assert "workflow_dispatch:" in workflow
    assert "\n  push:\n" not in workflow
    assert "description: New release tag to create after all release gates pass" in workflow
    assert 'if [[ "${RELEASE_TAG#v}" != "1.0.0" ]]; then' in workflow

    gate_and_build = workflow.split("\n  gate-and-build:\n", 1)[1].split("\n  create-release-tag:\n", 1)[0]
    tag_job = workflow.split("\n  create-release-tag:\n", 1)[1].split("\n  publish:\n", 1)[0]
    assert "uv run python scripts/check_release_readiness.py" in gate_and_build
    assert "uv run mkdocs build --strict" in gate_and_build
    assert "needs: gate-and-build" in tag_job
    assert 'git tag --annotate --message "Release $RELEASE_TAG"' in tag_job
    assert 'git push origin "$tag_ref"' in tag_job


def test_release_workflow_has_no_pypi_or_oidc_publication_path() -> None:
    workflow = _workflow()
    assert "pypa/gh-action-pypi-publish" not in workflow
    assert "PyPI" not in workflow
    assert "pypi.org" not in workflow
    assert "id-token:" not in workflow
    assert "id-token: write" not in workflow


def test_gate_builds_exact_distributions_and_records_integrity_metadata() -> None:
    workflow = _workflow()
    assert "uv sync --locked --extra docs" in workflow
    assert "uv run ruff check src tests scripts" in workflow
    assert "uv run ruff format --check src tests scripts" in workflow
    assert "uv run pyright" in workflow
    assert "uv run pytest -v" in workflow
    assert "uv build --wheel --sdist --out-dir dist" in workflow
    assert '"schema": "latent-anything-release-provenance-v1"' in workflow
    assert '(dist / "SHA256SUMS").write_text(' in workflow
    assert '(dist / "PROVENANCE.json").write_text(' in workflow


def test_publish_job_is_github_release_only_and_verifies_exact_assets() -> None:
    workflow = _workflow()
    publish = workflow.split("\n  publish:\n", 1)[1]
    assert "needs:" in publish
    assert "- gate-and-build" in publish
    assert "- create-release-tag" in publish
    assert "contents: write" in publish
    assert "softprops/action-gh-release@v2" in publish
    assert "overwrite_files: true" in publish
    assert "fail_on_unmatched_files: true" in publish
    assert "gh release download" in publish
    assert "Verified exact GitHub Release assets and SHA-256 hashes" in publish
    for asset in _REQUIRED_ASSETS:
        assert asset in publish
