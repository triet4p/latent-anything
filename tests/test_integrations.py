"""Import-isolation tests for optional integration boundaries."""

from __future__ import annotations

import builtins

import pytest

import latent_anything
from latent_anything.integrations import require_optional


def test_base_package_import_does_not_import_optional_backends() -> None:
    assert latent_anything.__version__


def test_missing_optional_backend_has_actionable_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fail_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "imaginary_backend":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    with pytest.raises(ImportError, match="uv sync --extra diffusers"):
        require_optional("imaginary_backend", extra="diffusers")
