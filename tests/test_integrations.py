"""Import-isolation tests for optional integration boundaries."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from types import ModuleType

import pytest

import latent_anything
from latent_anything.integrations import require_optional


def test_base_package_import_does_not_import_optional_backends() -> None:
    assert latent_anything.__version__


def test_missing_optional_backend_has_actionable_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fail_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "imaginary_backend":
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    with pytest.raises(ImportError, match="uv sync --extra diffusers"):
        require_optional("imaginary_backend", extra="diffusers")


def test_broken_optional_backend_preserves_its_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def broken_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "diffusers":
            error = ModuleNotFoundError("missing nested dependency")
            error.name = "huggingface_hub"
            raise error
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("latent_anything.integrations._optional.import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="nested dependency"):
        require_optional("diffusers", extra="diffusers")
