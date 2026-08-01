"""Structural snapshot tests for the Plotly projection explorer.

These assert the stable shape of the figures the explorer produces — trace
types, modes, hover payloads, trajectory overlays, and the metrics annotation
— without diffing full plotly JSON (which churns across plotly versions).
"""

from __future__ import annotations

import numpy as np
import pytest

plotly = pytest.importorskip("plotly")

from latent_anything import Trajectory  # noqa: E402
from latent_anything.visualization import (  # noqa: E402
    ProjectionView,
    build_projection,
    downsample_view,
    prepare_view,
    projection_explorer,
    projection_from_trajectory,
)


def _view(n: int = 50, ndim: int = 2, seed: int = 0, title: str = "demo") -> ProjectionView:
    coords = np.asarray(np.random.default_rng(seed).random((n, ndim)), dtype=np.float64)
    return build_projection(
        coords,
        categories=[f"c{i % 3}" for i in range(n)],
        title=title,
        extra_metadata={"metrics": {"silhouette_score": 0.42, "n_clusters": 3}},
    )


class TestExplorerStructure:
    def test_2d_explorer_has_per_category_scatter_traces(self) -> None:
        figure = projection_explorer(_view())
        assert len(figure.data) == 3
        assert all(trace.type == "scatter" for trace in figure.data)
        assert {trace.name for trace in figure.data} == {"c0", "c1", "c2"}
        assert all(trace.mode == "markers" for trace in figure.data)

    def test_3d_explorer_uses_scatter3d(self) -> None:
        figure = projection_explorer(_view(ndim=3, seed=1))
        assert all(trace.type == "scatter3d" for trace in figure.data)
        for trace in figure.data:
            assert trace.z is not None

    def test_hover_text_includes_metadata(self) -> None:
        figure = projection_explorer(_view())
        first = figure.data[0]
        assert any("c0" in text for text in first.text)
        assert any("silhouette" not in text for text in first.text)

    def test_metrics_annotation_present(self) -> None:
        figure = projection_explorer(_view())
        annotation = figure.layout.annotations[0]
        assert "silhouette_score=0.42" in annotation.text
        assert "n_clusters=3" in annotation.text

    def test_title_override(self) -> None:
        figure = projection_explorer(_view(), title="override")
        assert figure.layout.title.text == "override"

    def test_continuous_color_by_single_trace(self) -> None:
        coords = np.asarray(np.random.default_rng(2).random((40, 2)), dtype=np.float64)
        view = build_projection(
            coords,
            metadata=[{"calibrated_ood_score": float(i % 4) / 4} for i in range(40)],
            title="density",
        )
        figure = projection_explorer(view, color_by="calibrated_ood_score")
        assert len(figure.data) == 1
        marker = figure.data[0].marker
        assert marker.colorscale is not None
        assert marker.showscale is True

    def test_color_by_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            projection_explorer(_view(), color_by="nope")

    def test_rejects_non_2d3d_view(self) -> None:
        from latent_anything.visualization import PointView

        view = ProjectionView(points=(PointView((0.0, 1.0, 2.0, 3.0)),))
        with pytest.raises(ValueError, match="2D or 3D"):
            projection_explorer(view)


class TestTrajectoryOverlays:
    def test_trajectory_overlay_is_line_plus_markers(self) -> None:
        trajectory = Trajectory(np.asarray(np.random.default_rng(3).random((8, 4)), dtype=np.float64))
        view = projection_from_trajectory(
            trajectory,
            np.asarray(np.random.default_rng(4).random((8, 2)), dtype=np.float64),
            step_labels=[str(i) for i in range(8)],
        )
        figure = projection_explorer(view)
        marker_trace = figure.data[0]
        overlay = figure.data[1]
        assert marker_trace.type == "scatter"
        assert overlay.mode == "lines+markers"
        assert overlay.name == "trajectory"
        assert len(overlay.x) == 8

    def test_overlay_points_hover_with_step_index(self) -> None:
        trajectory = Trajectory(np.asarray(np.random.default_rng(5).random((5, 4)), dtype=np.float64))
        view = projection_from_trajectory(
            trajectory,
            np.asarray(np.random.default_rng(6).random((5, 2)), dtype=np.float64),
            step_labels=[f"s{i}" for i in range(5)],
        )
        figure = projection_explorer(view)
        overlay = figure.data[1]
        assert any("step" in text for text in overlay.text)
        assert any("s0" in text for text in overlay.text)


class TestResponsiveness:
    def test_prepare_view_uses_declared_2d_limit(self) -> None:
        view = _view(n=60_000, seed=7)
        prepared = prepare_view(view)
        assert prepared.n_points <= 50_000
        assert prepared.metadata["downsampled"] is True

    def test_prepare_view_uses_declared_3d_limit(self) -> None:
        view = _view(n=25_000, ndim=3, seed=8)
        prepared = prepare_view(view)
        assert prepared.n_points <= 20_000

    def test_prepare_view_with_explicit_limit(self) -> None:
        view = _view(n=1_000, seed=9)
        prepared = prepare_view(view, point_limit=100)
        assert prepared.n_points <= 100

    def test_explorer_annotates_downsampling(self) -> None:
        view = _view(n=60_000, seed=10)
        figure = projection_explorer(view)
        annotation = figure.layout.annotations[0]
        assert "downsampled" in annotation.text

    def test_explorer_trace_count_capped(self) -> None:
        view = _view(n=60_000, seed=11)
        figure = projection_explorer(view)
        rendered = sum(len(trace.x) for trace in figure.data if trace.type == "scatter")
        assert rendered <= 50_000

    def test_downsample_identical_to_data_module(self) -> None:
        view = _view(n=5_000, seed=12)
        assert prepare_view(view, point_limit=400).to_dict() == downsample_view(view, limit=400).to_dict()
