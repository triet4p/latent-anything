"""Notebook widget path and static HTML/PNG export for projection views.

A single :class:`ProjectionExplorer` serves both environments: inside a
Jupyter notebook it renders an interactive widget (Plotly ``FigureWidget`` in
an ``ipywidgets`` container with a metadata-inspection panel), and everywhere
else it degrades cleanly to static HTML or PNG/SVG export through
:meth:`ProjectionExplorer.to_html`, :meth:`ProjectionExplorer.to_image`, and
:meth:`ProjectionExplorer.save`.

Both ipywidgets and plotly are imported lazily (never at module import time),
preserving base-package import isolation. When they are missing,
:func:`require_optional` raises an actionable error pointing at ``uv sync
--extra viz``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from latent_anything.integrations._optional import require_optional
from latent_anything.visualization.data import ProjectionView
from latent_anything.visualization.figures import prepare_view, projection_explorer

_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"}


def _in_notebook() -> bool:
    """Return ``True`` when running inside an interactive Jupyter kernel."""
    try:
        from IPython import get_ipython  # type: ignore[reportMissingImports]
    except Exception:  # pragma: no cover - depends on the runtime environment
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return getattr(shell, "kernel", None) is not None


def format_view_summary(view: ProjectionView) -> str:
    """One-line summary of a projection view for the inspection panel."""
    parts = [f"points={view.n_points}"]
    metrics_value = view.metadata.get("metrics")
    if isinstance(metrics_value, Mapping):
        metrics = cast(Mapping[str, object], metrics_value)
        parts.extend(f"{key}={value}" for key, value in metrics.items())
    if view.metadata.get("downsampled") is True:
        parts.append(f"dropped={view.metadata.get('n_dropped', '?')}")
    return f"{view.title or 'Projection'} — " + " · ".join(parts)


def format_point_metadata(view: ProjectionView, indices: Sequence[int]) -> str:
    """Format a readable inspection panel for the selected/hovered points.

    ``indices`` refer to positions in ``view.points`` (the downsampled view
    actually rendered by the explorer).
    """
    lines = [f"{len(indices)} point(s) selected"]
    for index in indices:
        if index < 0 or index >= len(view.points):
            msg = f"index {index} out of range for a view with {len(view.points)} points"
            raise ValueError(msg)
        point = view.points[index]
        block = [f"- index {index}"]
        if point.label is not None:
            block.append(f"  label: {point.label}")
        if point.category is not None:
            block.append(f"  category: {point.category}")
        coordinates = ", ".join(f"{coordinate:.4g}" for coordinate in point.coordinates)
        block.append(f"  coords: ({coordinates})")
        for key, value in point.metadata.items():
            block.append(f"  {key}: {value}")
        lines.append("\n".join(block))
    return "\n".join(lines)


class ProjectionExplorer:
    """Interactive/static explorer that degrades cleanly outside notebooks.

    Parameters
    ----------
    view :
        The renderer input produced by the builders in
        :mod:`latent_anything.visualization.data`.
    point_limit :
        Maximum background points to render (``None`` uses the declared
        responsiveness target).
    color_by :
        Optional continuous metadata field to color points by.
    title :
        Overrides ``view.title`` when given.
    """

    def __init__(
        self,
        view: ProjectionView,
        *,
        point_limit: int | None = None,
        color_by: str | None = None,
        title: str | None = None,
    ) -> None:
        self.view = view
        self.point_limit = point_limit
        self.color_by = color_by
        self.title = title

    @property
    def prepared_view(self) -> ProjectionView:
        """The view actually rendered (downsampled when over the limit)."""
        return prepare_view(self.view, point_limit=self.point_limit)

    def figure(self) -> Any:
        """Return the interactive Plotly figure for this view."""
        return projection_explorer(
            self.view,
            point_limit=self.point_limit,
            color_by=self.color_by,
            title=self.title,
        )

    def widget(self) -> Any:
        """Return an ``ipywidgets`` container with the figure and an inspector.

        The container holds a Plotly ``FigureWidget`` (interactive selection,
        zoom, hover) plus an inspection panel that shows the view summary and
        the metadata of hovered points.
        """
        ipywidgets = require_optional("ipywidgets", extra="viz")
        plotly = require_optional("plotly", extra="viz")
        view = self.prepared_view
        figure_widget = plotly.graph_objs.FigureWidget(self.figure())
        output = ipywidgets.Output()
        with output:
            print(format_view_summary(view))
        for trace in figure_widget.data:
            trace.on_hover(_make_hover_handler(view, output))
        return ipywidgets.VBox([figure_widget, output])

    def to_html(self, *, include_plotlyjs: str = "inline") -> str:
        """Return a self-contained HTML document embedding the figure.

        ``include_plotlyjs`` accepts plotly's ``to_html`` values: ``"inline"``
        (default; fully offline but large) or ``"cdn"`` (small, needs
        internet).
        """
        return str(self.figure().to_html(full_html=True, include_plotlyjs=include_plotlyjs))

    def to_image(self, *, format: str = "png", width: int | None = None, height: int | None = None) -> bytes:
        """Render the figure to image bytes (requires the ``kaleido`` backend)."""
        require_optional("kaleido", extra="viz")
        kwargs: dict[str, Any] = {"format": format}
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        return bytes(self.figure().to_image(**kwargs))

    def save(
        self,
        path: str | Path,
        *,
        width: int | None = None,
        height: int | None = None,
        include_plotlyjs: str = "inline",
    ) -> Path:
        """Write the figure to ``.html`` or an image file (``.png``/``.svg``/...).

        Returns the resolved output path. ``.html`` export needs only plotly;
        image export needs the kaleido backend. ``width``/``height`` only
        affect image export; ``include_plotlyjs`` only affects HTML export
        (``"inline"`` = offline, ``"cdn"`` = small but needs internet).
        """
        target = Path(path)
        suffix = target.suffix.lower()
        if suffix == ".html":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.to_html(include_plotlyjs=include_plotlyjs), encoding="utf-8")
        elif suffix in _IMAGE_FORMATS:
            require_optional("kaleido", extra="viz")
            target.parent.mkdir(parents=True, exist_ok=True)
            image_kwargs: dict[str, Any] = {}
            if width is not None:
                image_kwargs["width"] = width
            if height is not None:
                image_kwargs["height"] = height
            self.figure().write_image(str(target), **image_kwargs)
        else:
            msg = f"Unsupported export format {suffix!r}; use .html or one of {sorted(_IMAGE_FORMATS)}"
            raise ValueError(msg)
        return target

    def show(self) -> Any:
        """Render for the current environment.

        In a notebook this displays the interactive widget and returns
        ``None``; outside a notebook it degrades to a static HTML string
        (write it with :meth:`save`).
        """
        if _in_notebook():
            widget = self.widget()
            display_module = require_optional("IPython.display", extra="viz")
            display_module.display(widget)
            return None
        return self.to_html()

    def _repr_html_(self) -> str:
        """Auto-display the interactive figure in Jupyter."""
        return self.to_html()


def _update_inspector(view: ProjectionView, output: Any, points: Any) -> None:
    """Update an ipywidgets Output with the metadata of hovered points."""
    indices = list(getattr(points, "point_inds", []))
    output.clear_output(wait=True)
    with output:
        if indices:
            print(format_point_metadata(view, indices))
        else:
            print(format_view_summary(view))


def _make_hover_handler(view: ProjectionView, output: Any) -> Any:
    """Return a plotly ``on_hover`` callback bound to a view and output."""

    def handle(figure: Any, trace: Any, points: Any) -> None:  # noqa: ARG001
        """Render metadata for the points reported by Plotly hover."""
        _update_inspector(view, output, points)

    return handle


def render(
    view: ProjectionView,
    *,
    point_limit: int | None = None,
    color_by: str | None = None,
    title: str | None = None,
) -> ProjectionExplorer:
    """Wrap a projection view in a :class:`ProjectionExplorer`.

    In a notebook the returned object auto-displays its interactive figure;
    call :meth:`ProjectionExplorer.show` or :meth:`ProjectionExplorer.save`
    for the widget or static export respectively.
    """
    return ProjectionExplorer(view, point_limit=point_limit, color_by=color_by, title=title)
