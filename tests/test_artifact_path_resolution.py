"""Tests for frozen-artifact path resolution across installation layouts.

The clean-environment reproduction (Sprint 80.25) proved that resolving the
frozen schema/taxonomy documents from ``Path(__file__).parents[2]`` only works
in the developer source layout: an installed package inside a project-local
virtual environment resolved to ``<venv>/Lib/artifacts/...`` and failed before
any proof stage could run. These tests pin both behaviors of the replacement
resolver: byte-identical developer-layout resolution, upward resolution from an
installed layout, and the fail-closed legacy fallback when no artifacts
directory exists anywhere above the package.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from latent_anything import _artifact_path
from latent_anything._benchmark_manifest import SCHEMA_PATH as MANIFEST_SCHEMA_PATH
from latent_anything._benchmark_manifest import load_schema
from latent_anything._diagnostic_report import SCHEMA_PATH as REPORT_SCHEMA_PATH
from latent_anything._diagnostic_report import load_schema as load_report_schema
from latent_anything._representation_taxonomy import TAXONOMY_PATH, load_taxonomy

REPO = Path(__file__).resolve().parents[1]
PINNED_ARTIFACTS = (
    ("benchmark_manifest_schema_v1.json", MANIFEST_SCHEMA_PATH),
    ("diagnostic_report_schema_v1.json", REPORT_SCHEMA_PATH),
    ("representation_problem_taxonomy_v1.json", TAXONOMY_PATH),
)


def test_dev_layout_resolution_is_byte_identical_to_the_legacy_location() -> None:
    for filename, resolved in PINNED_ARTIFACTS:
        legacy = REPO / "artifacts" / filename
        assert resolved == legacy
        assert resolved.is_file()
    # The frozen documents still load through the resolved paths.
    assert load_schema()["schema_id"] == "benchmark-manifest"
    assert load_report_schema()["schema_id"] == "diagnostic-report"
    assert load_taxonomy()["schema_version"] == "representation-problem-taxonomy-v1"


def test_installed_layout_resolves_upward_to_the_project_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_package = tmp_path / "root" / ".venv" / "Lib" / "site-packages" / "latent_anything"
    fake_package.mkdir(parents=True)
    artifacts = tmp_path / "root" / "artifacts"
    artifacts.mkdir()
    target = artifacts / "benchmark_manifest_schema_v1.json"
    target.write_text('{"schema_id": "benchmark-manifest"}', encoding="utf-8")

    monkeypatch.setattr(_artifact_path, "_PACKAGE_DIR", fake_package)
    resolved = _artifact_path.resolve_artifact_path("benchmark_manifest_schema_v1.json")
    assert resolved == target
    # The process working directory plays no role in resolution.
    monkeypatch.chdir(tmp_path)
    assert _artifact_path.resolve_artifact_path("benchmark_manifest_schema_v1.json") == target


def test_missing_artifact_falls_back_to_the_legacy_fail_closed_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_package = tmp_path / "root" / ".venv" / "Lib" / "site-packages" / "latent_anything"
    fake_package.mkdir(parents=True)

    monkeypatch.setattr(_artifact_path, "_PACKAGE_DIR", fake_package)
    resolved = _artifact_path.resolve_artifact_path("no_such_artifact.json")
    assert resolved == fake_package.parents[1] / "artifacts" / "no_such_artifact.json"
    assert not resolved.exists()
