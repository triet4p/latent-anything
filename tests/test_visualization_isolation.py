"""Prove the visualization subpackage stays optional and base-package-safe.

The base package and the pure renderer-input module must never import
plotly/kaleido/ipywidgets, and the frontend functions must fail with an
actionable message when the ``viz`` extra is missing.
"""

from __future__ import annotations

import builtins
import sys
from collections.abc import Generator, Mapping, Sequence
from types import ModuleType

import pytest

from latent_anything.visualization import build_projection
from latent_anything.visualization import explorer as explorer_module
from latent_anything.visualization import figures as figures_module
from latent_anything.visualization.figures import projection_explorer

_FRONTEND_MODULES = ("plotly", "kaleido", "ipywidgets", "anywidget")


@pytest.fixture(autouse=True)
def _purge_frontend_modules() -> Generator[None, None, None]:  # pyright: ignore[reportUnusedFunction]
    saved: dict[str, ModuleType] = {}
    for module_name in _FRONTEND_MODULES:
        for key in list(sys.modules):
            if key == module_name or key.startswith(f"{module_name}."):
                saved[key] = sys.modules[key]
                del sys.modules[key]
    yield
    sys.modules.update(saved)


def _failing_import() -> object:
    original_import = builtins.__import__

    def fail_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name in _FRONTEND_MODULES or name.startswith(tuple(f"{module}." for module in _FRONTEND_MODULES)):
            raise ImportError(name)
        return original_import(name, globals, locals, fromlist, level)

    return fail_import


def test_base_package_import_does_not_import_plotly() -> None:

    assert all(module not in sys.modules for module in _FRONTEND_MODULES)


def test_visualization_package_import_does_not_import_plotly() -> None:
    assert all(module not in sys.modules for module in _FRONTEND_MODULES)


def test_pure_data_module_imports_under_blocked_frontends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "__import__", _failing_import())
    import latent_anything.visualization.data as data

    assert data.ProjectionView is not None
    view = build_projection(__import__("numpy").zeros((4, 2)))
    assert view.n_points == 4


def test_base_package_import_succeeds_when_frontends_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "__import__", _failing_import())
    module = __import__("latent_anything", fromlist=["__version__"])
    assert module.__version__


def test_explorer_imports_without_plotly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "__import__", _failing_import())
    assert explorer_module.ProjectionExplorer is not None
    assert figures_module.prepare_view is not None


def test_missing_plotly_has_actionable_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    view = build_projection(__import__("numpy").zeros((4, 2)))

    def fail_plotly(module_name: str) -> ModuleType:
        if module_name == "plotly":
            error = ModuleNotFoundError("plotly missing")
            error.name = "plotly"
            raise error
        return __import__(module_name, fromlist=["*"])

    monkeypatch.setattr("latent_anything.integrations._optional.import_module", fail_plotly)
    with pytest.raises(ImportError, match="uv sync --extra viz"):
        projection_explorer(view)
