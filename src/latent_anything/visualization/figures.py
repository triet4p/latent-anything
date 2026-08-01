"""Plotly figure builders for projection views.

plotly is an **optional** dependency: it is imported lazily inside each
function (never at module import time), so the visualization subpackage and
the base package stay import-clean without the ``viz`` extra. When plotly is
missing, :func:`require_optional` raises an actionable error pointing at
``uv sync --extra viz``.

The frontends only render :class:`~latent_anything.visualization.data.ProjectionView`
objects; they never compute metrics or model logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from latent_anything.integrations._optional import require_optional
from latent_anything.visualization.data import ProjectionView, TrajectoryView, downsample_view

_QUALITATIVE_COLORS = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)

_TRAJECTORY_COLORS = ("#FF0000", "#0000FF", "#00AA00", "#FF8800", "#AA00AA")


def prepare_view(view: ProjectionView, *, point_limit: int | None = None) -> ProjectionView:
    """Apply the declared responsiveness downsampling to a view.

    When ``point_limit`` is ``None`` the dimension-specific declared target is
    used (:data:`DEFAULT_POINT_LIMIT_2D` / :data:`DEFAULT_POINT_LIMIT_3D`).
    Views already under the limit are returned unchanged.
    """
    from latent_anything.visualization.data import default_point_limit

    limit = default_point_limit(view.ndim) if point_limit is None else point_limit
    return downsample_view(view, limit=limit)


def _point_hover_text(point: Any, *, category: str | None) -> str:
    """Format one point's hover block from its label, category, and metadata."""
    from latent_anything.visualization.data import PointView

    assert isinstance(point, PointView)
    lines = [f"<b>{category}</b>" if category is not None else "<b>point</b>"]
    if point.label is not None:
        lines.append(f"label: {point.label}")
    for key, value in point.metadata.items():
        lines.append(f"{key}: {value}")
    return "<br>".join(lines)


def _coordinates_for(points: Sequence[Any], axis: int) -> list[float]:
    return [float(point.coordinates[axis]) for point in points]


def _axis_titles(view: ProjectionView) -> tuple[str, str, str]:
    metadata = view.metadata
    return (
        str(metadata.get("xlabel", "component 1")),
        str(metadata.get("ylabel", "component 2")),
        str(metadata.get("zlabel", "component 3")),
    )


def _metrics_annotation_text(view: ProjectionView) -> str:
    """Build the subtitle annotation from the view's embedded metrics."""
    pieces: list[str] = []
    metrics_value = view.metadata.get("metrics")
    if isinstance(metrics_value, Mapping):
        metrics = cast(Mapping[str, object], metrics_value)
        for key, value in metrics.items():
            pieces.append(f"{key}={_format_metric_value(value)}")
    if view.metadata.get("downsampled") is True:
        n_dropped = view.metadata.get("n_dropped", "?")
        pieces.append(f"downsampled (dropped {n_dropped})")
    return " &nbsp;·&nbsp; ".join(pieces)


def _format_metric_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def projection_explorer(
    view: ProjectionView,
    *,
    point_limit: int | None = None,
    color_by: str | None = None,
    title: str | None = None,
) -> Any:
    """Build an interactive 2D/3D Plotly scatter explorer for a projection view.

    Parameters
    ----------
    view :
        The renderer input produced by the builders in
        :mod:`latent_anything.visualization.data`.
    point_limit :
        Maximum number of background points to render; ``None`` uses the
        declared responsiveness target for the projection dimension.
    color_by :
        When set, points are colored by the continuous ``view.metadata``
        field of this name (e.g. ``"calibrated_ood_score"``) with a
        sequential colorscale instead of by category.
    title :
        Overrides ``view.title`` when given.

    Returns
    -------
    plotly.graph_objects.Figure
        An interactive figure with box/lasso selection, hover metadata, and
        trajectory overlays. The concrete figure type is available only when
        the ``viz`` extra is installed, so the return is typed loosely.

    Raises
    ------
    ImportError
        If plotly is not installed (install with ``uv sync --extra viz``).
    """
    plotly = require_optional("plotly", extra="viz")
    go = plotly.graph_objects

    display_view = prepare_view(view, point_limit=point_limit)
    ndim = display_view.ndim
    if ndim not in (2, 3):
        msg = f"ProjectionView must be 2D or 3D, got ndim={ndim}"
        raise ValueError(msg)

    figure_title = title if title is not None else (display_view.title or "Latent projection")

    figure: Any = go.Figure()

    if color_by is not None:
        _add_continuous_trace(figure, display_view, ndim, color_by)
    else:
        _add_category_traces(figure, display_view, ndim)
    _add_trajectory_overlays(figure, display_view, ndim)

    xlabel, ylabel, zlabel = _axis_titles(display_view)
    layout_kwargs: dict[str, Any] = {
        "title": {"text": figure_title},
        "template": "plotly_white",
        "showlegend": True,
        "height": 600,
    }
    if ndim == 2:
        layout_kwargs["xaxis"] = {"title": xlabel}
        layout_kwargs["yaxis"] = {"title": ylabel}
    else:
        layout_kwargs["scene"] = {"xaxis": {"title": xlabel}, "yaxis": {"title": ylabel}, "zaxis": {"title": zlabel}}
    annotation_text = _metrics_annotation_text(display_view)
    if annotation_text:
        layout_kwargs["annotations"] = [
            {
                "text": annotation_text,
                "x": 0.0,
                "y": 1.02,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 11},
            }
        ]
    figure.update_layout(**layout_kwargs)
    return figure


def _add_category_traces(figure: Any, view: ProjectionView, ndim: int) -> None:
    categories: list[str] = []
    for point in view.points:
        category = point.category if point.category is not None else "unlabeled"
        if category not in categories:
            categories.append(category)
    categories = sorted(categories)
    for color_index, category in enumerate(categories):
        points = [
            point
            for point in view.points
            if (point.category if point.category is not None else "unlabeled") == category
        ]
        color = _QUALITATIVE_COLORS[color_index % len(_QUALITATIVE_COLORS)]
        _add_scatter(figure, ndim, points, name=category, color=color, mode="markers")


def _add_continuous_trace(figure: Any, view: ProjectionView, ndim: int, color_by: str) -> None:
    values: list[float] = []
    for point in view.points:
        raw = point.metadata.get(color_by)
        if not isinstance(raw, (int, float)):
            msg = f"color_by field {color_by!r} must be numeric on every point"
            raise ValueError(msg)
        values.append(float(raw))
    trace_kwargs: dict[str, Any] = {
        "x": _coordinates_for(view.points, 0),
        "y": _coordinates_for(view.points, 1),
        "mode": "markers",
        "name": color_by,
        "marker": {"color": values, "colorscale": "Viridis", "showscale": True, "colorbar": {"title": color_by}},
        "text": [_point_hover_text(point, category=point.category) for point in view.points],
        "hoverinfo": "text",
    }
    if ndim == 3:
        trace_kwargs["z"] = _coordinates_for(view.points, 2)
        figure.add_scatter3d(**trace_kwargs)
    else:
        figure.add_scatter(**trace_kwargs)


def _add_scatter(
    figure: Any,
    ndim: int,
    points: Sequence[Any],
    *,
    name: str,
    color: str,
    mode: str,
    line_width: int = 1,
) -> None:
    from latent_anything.visualization.data import PointView

    for point in points:
        assert isinstance(point, PointView)
    trace_kwargs: dict[str, Any] = {
        "x": _coordinates_for(points, 0),
        "y": _coordinates_for(points, 1),
        "mode": mode,
        "name": name,
        "text": [_point_hover_text(point, category=point.category) for point in points],
        "hoverinfo": "text",
        "marker": {"size": 6, "color": color},
    }
    if ndim == 3:
        trace_kwargs["z"] = _coordinates_for(points, 2)
        trace_kwargs["line"] = {"width": line_width, "color": color}
        figure.add_scatter3d(**trace_kwargs)
    else:
        trace_kwargs["line"] = {"width": line_width, "color": color}
        figure.add_scatter(**trace_kwargs)


def _add_trajectory_overlays(figure: Any, view: ProjectionView, ndim: int) -> None:
    for index, trajectory in enumerate(view.trajectories):
        _add_trajectory_overlay(figure, ndim, trajectory, index)


def _add_trajectory_overlay(figure: Any, ndim: int, trajectory: TrajectoryView, index: int) -> None:
    if not trajectory.points:
        return
    name = trajectory.name if trajectory.name is not None else f"trajectory {index + 1}"
    color = _TRAJECTORY_COLORS[index % len(_TRAJECTORY_COLORS)]
    text: list[str] = []
    for step, point in enumerate(trajectory.points):
        block = [f"<b>step {step}</b>"]
        if point.label is not None:
            block.append(f"label: {point.label}")
        for key, value in point.metadata.items():
            block.append(f"{key}: {value}")
        text.append("<br>".join(block))
    trace_kwargs: dict[str, Any] = {
        "x": _coordinates_for(trajectory.points, 0),
        "y": _coordinates_for(trajectory.points, 1),
        "mode": "lines+markers",
        "name": name,
        "text": text,
        "hoverinfo": "text",
        "marker": {"size": 5, "color": color},
        "line": {"width": 2, "color": color},
    }
    if ndim == 3:
        trace_kwargs["z"] = _coordinates_for(trajectory.points, 2)
        figure.add_scatter3d(**trace_kwargs)
    else:
        figure.add_scatter(**trace_kwargs)
