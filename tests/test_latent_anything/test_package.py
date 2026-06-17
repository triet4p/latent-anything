"""Smoke tests for the latent-anything package."""

import latent_anything


def test_package_imports() -> None:
    """Verify the package imports and exposes expected attributes."""
    assert hasattr(latent_anything, "__version__")
    assert isinstance(latent_anything.__version__, str)
    assert latent_anything.__version__ == "0.1.0"


def test_package_docstring() -> None:
    """Verify the package has a non-empty docstring."""
    assert latent_anything.__doc__
    assert len(latent_anything.__doc__) > 50
