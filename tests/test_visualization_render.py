"""Widget-path and static-export tests for the projection explorer.

Verifies that a single ``ProjectionExplorer`` serves the notebook widget path
and degrades cleanly to static HTML/PNG export outside a notebook, and that
the metadata-inspection helpers format selected/hovered points.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

plotly = pytest.importorskip("plotly")
ipywidgets = pytest.importorskip("ipywidgets")
pytest.importorskip("kaleido")

from latent_anything.visualization import ProjectionExplorer, build_projection, render  # noqa: E402
from latent_anything.visualization.explorer import (  # noqa: E402
    _in_notebook,  # noqa: E402  # pyright: ignore[reportPrivateUsage]
    _make_hover_handler,  # noqa: E402  # pyright: ignore[reportPrivateUsage]
    format_point_metadata,
    format_view_summary,
)


def _view(n: int = 40) -> Any:
    coords = np.asarray(np.random.default_rng(0).random((n, 2)), dtype=np.float64)
    return build_projection(
        coords,
        categories=[f"c{i % 2}" for i in range(n)],
        metadata=[{"score": float(i % 5) / 5, "token": f"tok{i}"} for i in range(n)],
        title="demo",
        extra_metadata={"metrics": {"silhouette_score": 0.42}},
    )


class TestStaticExport:
    def test_to_html_is_self_contained_plotly(self) -> None:
        html = ProjectionExplorer(_view()).to_html()
        assert "<div" in html
        assert "plotly" in html
        assert "<html" in html and "<body>" in html

    def test_to_image_returns_png(self) -> None:
        png = ProjectionExplorer(_view()).to_image(format="png", width=200, height=150)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_save_html(self, tmp_path: Path) -> None:
        target = ProjectionExplorer(_view()).save(tmp_path / "out" / "chart.html")
        assert target.is_file()
        assert "<div" in target.read_text(encoding="utf-8")

    def test_save_png(self, tmp_path: Path) -> None:
        target = ProjectionExplorer(_view()).save(tmp_path / "chart.png")
        assert target.suffix == ".png"
        assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_save_unsupported_format_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported export format"):
            ProjectionExplorer(_view()).save(tmp_path / "chart.json")

    def test_render_returns_explorer(self) -> None:
        explorer = render(_view(), title="wrapped")
        assert isinstance(explorer, ProjectionExplorer)
        assert explorer.title == "wrapped"

    def test_repr_html_is_self_contained(self) -> None:
        html = ProjectionExplorer(_view())._repr_html_()  # pyright: ignore[reportPrivateUsage]
        assert "<div" in html
        assert "plotly" in html
        assert "<body>" in html


class TestWidgetPath:
    def test_widget_is_vbox_with_figure_and_inspector(self) -> None:
        widget = ProjectionExplorer(_view()).widget()
        assert type(widget).__name__ == "VBox"
        child_names = [type(child).__name__ for child in widget.children]
        assert child_names == ["FigureWidget", "Output"]

    def test_widget_figure_is_interactive_scatter(self) -> None:
        widget = ProjectionExplorer(_view()).widget()
        figure_widget = widget.children[0]
        assert figure_widget.data[0].type == "scatter"

    def test_widget_has_inspection_output_panel(self) -> None:
        widget = ProjectionExplorer(_view()).widget()
        assert type(widget.children[1]).__name__ == "Output"


class TestInspectionHelpers:
    def test_format_point_metadata(self) -> None:
        view = _view()
        text = format_point_metadata(view, [0])
        assert "index 0" in text
        assert "token" in text
        assert "coords" in text

    def test_format_point_metadata_many(self) -> None:
        text = format_point_metadata(_view(), [0, 1])
        assert text.startswith("2 point(s) selected")

    def test_format_point_metadata_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            format_point_metadata(_view(), [999])

    def test_format_view_summary_includes_metrics(self) -> None:
        assert "silhouette_score=0.42" in format_view_summary(_view())

    def test_hover_handler_updates_output(self) -> None:
        output = _CapturingOutput()
        view = _view()
        handle = _make_hover_handler(view, output)

        class FakePoints:
            point_inds = [0, 2]

        handle(figure=object(), trace=object(), points=FakePoints())
        assert "2 point(s) selected" in output.latest


class TestDegradation:
    def test_show_outside_notebook_returns_html(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("latent_anything.visualization.explorer._in_notebook", lambda: False)
        result = ProjectionExplorer(_view()).show()
        assert isinstance(result, str)
        assert "<div" in result

    def test_show_in_notebook_displays_widget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        displayed: list[object] = []

        def fake_display(widget: object) -> None:
            displayed.append(widget)

        monkeypatch.setattr("latent_anything.visualization.explorer._in_notebook", lambda: True)
        monkeypatch.setattr(
            "latent_anything.visualization.explorer.require_optional", _fake_display_module(fake_display)
        )
        result = ProjectionExplorer(_view()).show()
        assert result is None
        assert len(displayed) == 1

    def test_widget_requires_ipywidgets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_optional(module_name: str, *, extra: str) -> Any:
            del extra
            if module_name == "ipywidgets":
                raise ImportError("uv sync --extra viz")
            return __import__(module_name, fromlist=["*"])

        monkeypatch.setattr("latent_anything.visualization.explorer.require_optional", failing_optional)
        with pytest.raises(ImportError, match="uv sync --extra viz"):
            ProjectionExplorer(_view()).widget()

    def test_notebook_detection_false_in_test_env(self) -> None:
        assert _in_notebook() is False


def _fake_display_module(display_fn: Any) -> Any:
    class FakeDisplayModule:
        display: Any

        def __call__(self, widget: object) -> None:
            display_fn(widget)

    fake = FakeDisplayModule()
    fake.display = display_fn

    def require(module_name: str, *, extra: str) -> Any:
        del extra
        if module_name == "ipywidgets":
            return ipywidgets
        if module_name == "IPython.display":
            return fake
        return __import__(module_name, fromlist=["*"])

    return require


class _CapturingOutput:
    """ipywidgets.Output stand-in that captures ``print`` inside its context."""

    def __init__(self) -> None:
        self.latest = ""

    def clear_output(self, *, wait: bool = False) -> None:
        del wait
        self.latest = ""

    def __enter__(self) -> _CapturingOutput:
        self._redirect: Any = contextlib.redirect_stdout(_Stream(self))
        self._redirect.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self._redirect.__exit__(None, None, None)


class _Stream(io.StringIO):
    def __init__(self, owner: _CapturingOutput) -> None:
        super().__init__()
        self._owner = owner

    def write(self, message: str) -> int:
        self._owner.latest += message
        return len(message)
