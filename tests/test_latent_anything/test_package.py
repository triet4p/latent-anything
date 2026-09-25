"""Smoke tests for the latent-anything package."""

import tomllib
from pathlib import Path

import latent_anything


def test_package_imports() -> None:
    """Verify the package imports and exposes expected attributes."""
    assert hasattr(latent_anything, "__version__")
    assert isinstance(latent_anything.__version__, str)
    assert latent_anything.__version__ == "1.0.0"


def test_package_docstring() -> None:
    """Verify the package has a non-empty docstring."""
    assert latent_anything.__doc__
    assert len(latent_anything.__doc__) > 50


def test_project_metadata_matches_runtime_version() -> None:
    project_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))
    assert project["project"]["version"] == latent_anything.__version__
