"""Optional interactive visualization frontends for analysis results.

This subpackage is **optional**: importing ``latent_anything`` never imports
it, and importing this package does not require plotly/kaleido/ipywidgets
(the frontend modules import those lazily). Install with ``uv sync --extra
viz``.

The data/view models in :mod:`latent_anything.visualization.data` are the
only renderer inputs; analysis results are converted into them by the
builders there, and the frontends (Plotly explorer, notebook widget, static
HTML/PNG export) only render them.
"""

from latent_anything.visualization.data import (
    DEFAULT_POINT_LIMIT_2D as DEFAULT_POINT_LIMIT_2D,
)
from latent_anything.visualization.data import (
    DEFAULT_POINT_LIMIT_3D as DEFAULT_POINT_LIMIT_3D,
)
from latent_anything.visualization.data import (
    DOWNSAMPLE_SEED as DOWNSAMPLE_SEED,
)
from latent_anything.visualization.data import MetricSummary as MetricSummary
from latent_anything.visualization.data import PointView as PointView
from latent_anything.visualization.data import ProjectionView as ProjectionView
from latent_anything.visualization.data import TrajectoryView as TrajectoryView
from latent_anything.visualization.data import build_projection as build_projection
from latent_anything.visualization.data import (
    default_point_limit as default_point_limit,
)
from latent_anything.visualization.data import downsample_view as downsample_view
from latent_anything.visualization.data import (
    metric_summary_from_atlas as metric_summary_from_atlas,
)
from latent_anything.visualization.data import (
    metric_summary_from_cross_seed as metric_summary_from_cross_seed,
)
from latent_anything.visualization.data import (
    metric_summary_from_density as metric_summary_from_density,
)
from latent_anything.visualization.data import (
    metric_summary_from_kmeans as metric_summary_from_kmeans,
)
from latent_anything.visualization.data import (
    metric_summary_from_probe as metric_summary_from_probe,
)
from latent_anything.visualization.data import (
    metric_summary_from_sae as metric_summary_from_sae,
)
from latent_anything.visualization.data import (
    metric_summary_from_stability as metric_summary_from_stability,
)
from latent_anything.visualization.data import points_view as points_view
from latent_anything.visualization.data import (
    projection_from_atlas as projection_from_atlas,
)
from latent_anything.visualization.data import (
    projection_from_density as projection_from_density,
)
from latent_anything.visualization.data import projection_from_dtw as projection_from_dtw
from latent_anything.visualization.data import (
    projection_from_kmeans as projection_from_kmeans,
)
from latent_anything.visualization.data import (
    projection_from_probe as projection_from_probe,
)
from latent_anything.visualization.data import (
    projection_from_trajectory as projection_from_trajectory,
)
from latent_anything.visualization.data import trajectory_view as trajectory_view
from latent_anything.visualization.explorer import ProjectionExplorer as ProjectionExplorer
from latent_anything.visualization.explorer import format_point_metadata as format_point_metadata
from latent_anything.visualization.explorer import format_view_summary as format_view_summary
from latent_anything.visualization.explorer import render as render
from latent_anything.visualization.figures import prepare_view as prepare_view
from latent_anything.visualization.figures import projection_explorer as projection_explorer

__all__ = [
    "DEFAULT_POINT_LIMIT_2D",
    "DEFAULT_POINT_LIMIT_3D",
    "DOWNSAMPLE_SEED",
    "MetricSummary",
    "PointView",
    "ProjectionExplorer",
    "ProjectionView",
    "TrajectoryView",
    "build_projection",
    "default_point_limit",
    "downsample_view",
    "format_point_metadata",
    "format_view_summary",
    "metric_summary_from_atlas",
    "metric_summary_from_cross_seed",
    "metric_summary_from_density",
    "metric_summary_from_kmeans",
    "metric_summary_from_probe",
    "metric_summary_from_sae",
    "metric_summary_from_stability",
    "points_view",
    "prepare_view",
    "projection_explorer",
    "projection_from_atlas",
    "projection_from_dtw",
    "projection_from_density",
    "projection_from_kmeans",
    "projection_from_probe",
    "projection_from_trajectory",
    "render",
    "trajectory_view",
]
