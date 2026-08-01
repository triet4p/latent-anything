"""Typed renderer inputs for the optional visualization frontends.

The renderer inputs (``ProjectionView``, ``TrajectoryView``, ``MetricSummary``)
are the stable contract between analysis results and every visualization
frontend. They carry **only data** — no plotly, no matplotlib, no widget
types — so the base package stays import-clean and the 2D/3D Plotly
explorer, the notebook widget, and the static HTML/PNG exporters all
consume the same view model.

The builders in this module are the only place analysis result objects are
turned into renderer inputs. Analysis methods (probes, clustering, density,
SAE feature atlas) never know that a visualization frontend exists, and the
frontends never recompute metrics or model logic — they only render what
these builders produce.

The point-count limits declared here are the responsiveness contract for the
interactive frontends: above ``DEFAULT_POINT_LIMIT_2D`` (or the 3D limit) a
view is deterministically downsampled before rendering (see
:func:`downsample_view`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from latent_anything.clustering import ClusterStabilityReport, KMeansResult
from latent_anything.density import DensityEvaluationReport, DensityResult, DensityStabilityReport
from latent_anything.probes import CrossSeedReport, LinearProbeResult
from latent_anything.sae_evaluation import FeatureAtlas, SAEEvaluationResult
from latent_anything.trajectory import Trajectory

# Declared responsiveness targets for interactive rendering. Above these the
# frontends deterministically downsample (see ``downsample_view``).
DEFAULT_POINT_LIMIT_2D = 50_000
DEFAULT_POINT_LIMIT_3D = 20_000
DOWNSAMPLE_SEED = 0


@dataclass(frozen=True)
class PointView:
    """One projected point in a :class:`ProjectionView`.

    Attributes
    ----------
    coordinates :
        Two or three projected coordinates.
    label :
        Optional hover label (e.g. a sample id or decoded text).
    category :
        Optional discrete group used for coloring and the legend
        (e.g. a predicted class, a cluster id, "id"/"ood").
    metadata :
        Arbitrary per-point fields shown during metadata inspection.
    """

    coordinates: tuple[float, ...]
    label: str | None = None
    category: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    @property
    def ndim(self) -> int:
        """Number of projected coordinates (2 or 3)."""
        return len(self.coordinates)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict for schema/snapshot testing."""
        return {
            "coordinates": list(self.coordinates),
            "label": self.label,
            "category": self.category,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrajectoryView:
    """Ordered overlay of points on top of a projection.

    A trajectory overlay is rendered as a connected line with markers,
    visually independent from the scatter of background points.

    Attributes
    ----------
    points :
        Ordered projected points of the trajectory.
    name :
        Optional legend name for the overlay.
    """

    points: tuple[PointView, ...]
    name: str | None = None

    @property
    def ndim(self) -> int:
        """Number of projected coordinates of the first point (0 when empty)."""
        if not self.points:
            return 0
        return self.points[0].ndim

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict for schema/snapshot testing."""
        return {"name": self.name, "points": [point.to_dict() for point in self.points]}


@dataclass(frozen=True)
class ProjectionView:
    """The renderer input for the 2D/3D projection explorer.

    This single stable view model is consumed by the interactive explorer,
    the notebook widget, and the static HTML/PNG exporters. It is produced
    exclusively by the builders in this module.

    Attributes
    ----------
    points :
        The projected background points.
    trajectories :
        Optional trajectory overlays sharing the same projection.
    title :
        Optional chart title.
    metadata :
        View-level metadata; the ``"metrics"`` key (when present) carries the
        scalar diagnostics that the frontends annotate on the chart and show
        in the inspection panel.
    """

    points: tuple[PointView, ...]
    trajectories: tuple[TrajectoryView, ...] = ()
    title: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    @property
    def ndim(self) -> int:
        """Number of projected coordinates (2 or 3, 0 when empty)."""
        if not self.points:
            return 0
        return self.points[0].ndim

    @property
    def n_points(self) -> int:
        """Number of background points in this view."""
        return len(self.points)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict for schema/snapshot testing."""
        return {
            "title": self.title,
            "ndim": self.ndim,
            "n_points": self.n_points,
            "points": [point.to_dict() for point in self.points],
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MetricSummary:
    """Labeled scalar metrics shown with a projection view.

    Metrics are always produced from analysis results by the builders in this
    module — the visualization frontends never compute them.
    """

    title: str
    metrics: Mapping[str, float | int | str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict for schema/snapshot testing."""
        return {"title": self.title, "metrics": dict(self.metrics)}


# ── Coordinate and metadata validation ────────────────────────────────────


def _validate_coordinates(coordinates: np.ndarray, *, name: str = "coordinates") -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (2, 3):
        msg = f"{name} must be a 2D array with 2 or 3 columns, got shape {values.shape}"
        raise ValueError(msg)
    if values.shape[0] == 0:
        msg = f"{name} must contain at least one point"
        raise ValueError(msg)
    if not np.isfinite(values).all():
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return values


def _as_strings(values: Sequence[Any] | None, *, n: int, name: str) -> tuple[str | None, ...] | None:
    if values is None:
        return None
    if len(values) != n:
        msg = f"{name} must have one entry per point ({n}), got {len(values)}"
        raise ValueError(msg)
    return tuple(None if value is None else str(value) for value in values)


def _as_metadata(items: Sequence[Mapping[str, Any]] | None, *, n: int) -> tuple[Mapping[str, Any], ...] | None:
    if items is None:
        return None
    if len(items) != n:
        msg = f"metadata must have one entry per point ({n}), got {len(items)}"
        raise ValueError(msg)
    return tuple(dict(item) for item in items)


# ── Points and trajectories ───────────────────────────────────────────────


def points_view(
    coordinates: np.ndarray,
    *,
    labels: Sequence[str | None] | None = None,
    categories: Sequence[Any | None] | None = None,
    metadata: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[PointView, ...]:
    """Build the point list of a projection view from plain arrays.

    Parameters
    ----------
    coordinates :
        ``(n_points, 2)`` or ``(n_points, 3)`` projected coordinates.
    labels :
        Optional per-point hover labels.
    categories :
        Optional per-point discrete groups (coerced to ``str``) for coloring.
    metadata :
        Optional per-point metadata dicts for inspection.
    """
    values = _validate_coordinates(coordinates)
    n = values.shape[0]
    label_values = _as_strings(labels, n=n, name="labels")
    category_values = _as_strings(categories, n=n, name="categories")
    metadata_values = _as_metadata(metadata, n=n)
    return tuple(
        PointView(
            coordinates=tuple(float(entry) for entry in values[i]),
            label=label_values[i] if label_values is not None else None,
            category=category_values[i] if category_values is not None else None,
            metadata=metadata_values[i] if metadata_values is not None else {},
        )
        for i in range(n)
    )


def trajectory_view(
    trajectory: Trajectory,
    coordinates: np.ndarray,
    *,
    name: str | None = None,
    labels: Sequence[str | None] | None = None,
) -> TrajectoryView:
    """Project a :class:`Trajectory` into a renderer overlay.

    ``coordinates`` must be the trajectory's points projected with the same
    projection model used for the surrounding view (e.g. the same fitted PCA),
    with one row per trajectory point.
    """
    values = _validate_coordinates(coordinates, name="trajectory coordinates")
    if values.shape[0] != len(trajectory):
        msg = f"trajectory has {len(trajectory)} points but only {values.shape[0]} coordinates were given"
        raise ValueError(msg)
    return TrajectoryView(points=points_view(values, labels=labels), name=name)


def build_projection(
    coordinates: np.ndarray,
    *,
    labels: Sequence[str | None] | None = None,
    categories: Sequence[Any | None] | None = None,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    trajectories: Sequence[TrajectoryView] = (),
    title: str = "",
    extra_metadata: Mapping[str, Any] | None = None,
) -> ProjectionView:
    """Build a :class:`ProjectionView` from projected coordinates and optics."""
    return ProjectionView(
        points=points_view(coordinates, labels=labels, categories=categories, metadata=metadata),
        trajectories=tuple(trajectories),
        title=title,
        metadata=dict(extra_metadata or {}),
    )


# ── Builders from stable analysis results ─────────────────────────────────


def projection_from_kmeans(
    result: KMeansResult,
    coordinates: np.ndarray,
    *,
    title: str = "K-means projection",
    extra_metadata: Mapping[str, Any] | None = None,
) -> ProjectionView:
    """Build a projection view colored by cluster assignment.

    Per-point metadata carries the assignment confidence and per-sample
    silhouette (when computed). The view metadata embeds the scalar
    diagnostics from :func:`metric_summary_from_kmeans`.
    """
    values = _validate_coordinates(coordinates)
    if values.shape[0] != len(result.assignments):
        msg = f"coordinates must have one row per clustered sample ({len(result.assignments)}), got {values.shape[0]}"
        raise ValueError(msg)
    confidence = np.asarray(result.confidence)
    per_point: list[dict[str, Any]] = [{"confidence": float(confidence[i])} for i in range(values.shape[0])]
    if result.per_sample_silhouette is not None:
        silhouette = np.asarray(result.per_sample_silhouette)
        for i in range(values.shape[0]):
            per_point[i]["silhouette"] = float(silhouette[i])
    summary = metric_summary_from_kmeans(result)
    metadata = {"metrics": dict(summary.metrics)}
    metadata.update(dict(extra_metadata or {}))
    return build_projection(
        values,
        categories=[f"cluster {int(label)}" for label in result.assignments],
        metadata=per_point,
        title=title,
        extra_metadata=metadata,
    )


def projection_from_probe(
    result: LinearProbeResult,
    coordinates: np.ndarray,
    *,
    predictions: Sequence[Any] | np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    title: str = "Probe predictions projection",
    extra_metadata: Mapping[str, Any] | None = None,
) -> ProjectionView:
    """Build a projection view colored by the probe's predicted class.

    Per-point metadata carries the predicted label and the top predicted
    probability. ``coordinates`` must have one row per point; by default the
    points are the probed test samples (``len(result.predictions)``). Pass
    ``predictions``/``probabilities`` to color a different (e.g. full) set of
    samples with the same fitted probe; the scalar metrics still come from
    ``result``.
    """
    values = _validate_coordinates(coordinates)
    if predictions is None:
        predictions = list(result.predictions)
    if len(predictions) != values.shape[0]:
        msg = f"predictions must have one entry per point ({values.shape[0]}), got {len(predictions)}"
        raise ValueError(msg)
    probabilities = np.asarray(result.probabilities) if probabilities is None else np.asarray(probabilities)
    if probabilities.ndim == 2 and probabilities.shape[0] != values.shape[0]:
        msg = f"probabilities must have one row per point ({values.shape[0]}), got {probabilities.shape[0]}"
        raise ValueError(msg)
    per_point = [
        {
            "predicted_label": str(predictions[i]),
            "top_probability": float(probabilities[i].max()) if probabilities.ndim == 2 else float(probabilities[i]),
        }
        for i in range(values.shape[0])
    ]
    summary = metric_summary_from_probe(result)
    metadata = {"metrics": dict(summary.metrics)}
    metadata.update(dict(extra_metadata or {}))
    return build_projection(
        values,
        categories=[str(label) for label in predictions],
        metadata=per_point,
        title=title,
        extra_metadata=metadata,
    )


def projection_from_density(
    result: DensityResult,
    coordinates: np.ndarray,
    *,
    ood_threshold: float = 0.5,
    title: str = "Calibrated OOD score projection",
    extra_metadata: Mapping[str, Any] | None = None,
) -> ProjectionView:
    """Build a projection view colored by calibrated OOD membership.

    Points are split into ``id`` / ``ood`` categories at ``ood_threshold`` on
    the calibrated OOD score, and per-point metadata carries the raw score and
    log density. ``coordinates`` must have one row per scored sample.
    """
    values = _validate_coordinates(coordinates)
    scores = np.asarray(result.calibrated_ood_score)
    if values.shape[0] != len(scores):
        msg = f"coordinates must have one row per scored sample ({len(scores)}), got {values.shape[0]}"
        raise ValueError(msg)
    densities = np.asarray(result.log_density)
    per_point = [
        {"calibrated_ood_score": float(scores[i]), "log_density": float(densities[i])} for i in range(values.shape[0])
    ]
    categories = ["ood" if scores[i] >= ood_threshold else "id" for i in range(values.shape[0])]
    metadata = {"ood_threshold": ood_threshold, "source_representation_identity": result.source_representation_identity}
    metadata.update(dict(extra_metadata or {}))
    return build_projection(
        values,
        categories=categories,
        metadata=per_point,
        title=title,
        extra_metadata=metadata,
    )


def projection_from_trajectory(
    trajectory: Trajectory,
    coordinates: np.ndarray,
    *,
    title: str = "Trajectory projection",
    step_labels: Sequence[str | None] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> ProjectionView:
    """Build a projection view whose points are one trajectory's steps.

    The trajectory is drawn as both markers (with optional step labels) and a
    connecting line overlay, making the ordering explicit.
    """
    values = _validate_coordinates(coordinates)
    if values.shape[0] != len(trajectory):
        msg = f"trajectory has {len(trajectory)} points but only {values.shape[0]} coordinates were given"
        raise ValueError(msg)
    overlay = trajectory_view(trajectory, values, name="trajectory", labels=step_labels)
    metadata = {"n_trajectory_points": len(trajectory)}
    metadata.update(dict(extra_metadata or {}))
    return build_projection(
        values,
        labels=step_labels,
        metadata=[{"step": i} for i in range(values.shape[0])],
        trajectories=[overlay],
        title=title,
        extra_metadata=metadata,
    )


def projection_from_atlas(
    atlas: FeatureAtlas,
    *,
    title: str = "Feature atlas: activation frequency vs decoder norm",
    extra_metadata: Mapping[str, Any] | None = None,
) -> ProjectionView:
    """Build a feature-scatter view from a portable :class:`FeatureAtlas`.

    Each point is one SAE feature located by its activation frequency and
    decoder norm (logged), colored by dead/alive, with per-point metadata
    carrying the feature index, mean activation, and encoder norm.
    """
    if not atlas.entries:
        msg = "atlas must contain at least one entry"
        raise ValueError(msg)
    coordinates_list: list[list[float]] = []
    categories: list[str] = []
    per_point: list[dict[str, Any]] = []
    for entry in atlas.entries:
        coordinates_list.append([entry.activation_frequency, float(np.log1p(entry.decoder_norm))])
        categories.append("dead" if entry.is_dead else "alive")
        per_point.append(
            {
                "feature_index": entry.feature_index,
                "mean_activation": entry.mean_activation,
                "encoder_norm": entry.encoder_norm,
                "decoder_norm": entry.decoder_norm,
                "n_top_examples": len(entry.top_examples),
            }
        )
    values = np.asarray(coordinates_list, dtype=np.float64)
    summary = metric_summary_from_atlas(atlas)
    metadata = {"metrics": dict(summary.metrics)}
    metadata.update(dict(extra_metadata or {}))
    return build_projection(
        values,
        categories=categories,
        metadata=per_point,
        title=title,
        extra_metadata=metadata,
    )


# ── Metric summaries from stable analysis results ─────────────────────────


def metric_summary_from_probe(result: LinearProbeResult) -> MetricSummary:
    """Scalar diagnostics from a :class:`LinearProbeResult`."""
    return MetricSummary(
        title="Linear probe",
        metrics={
            "accuracy": result.accuracy,
            "val_accuracy": result.val_accuracy,
            "n_classes": int(len(result.classes)),
            "n_iter": result.n_iter,
        },
    )


def metric_summary_from_cross_seed(report: CrossSeedReport) -> MetricSummary:
    """Scalar diagnostics from a :class:`CrossSeedReport`."""
    return MetricSummary(
        title="Cross-seed probe stability",
        metrics={
            "mean_accuracy": report.mean_accuracy,
            "ci95": report.ci95,
            "std_accuracy": report.std_accuracy,
            "min_accuracy": report.min_accuracy,
            "max_accuracy": report.max_accuracy,
            "n_seeds": report.n_seeds,
        },
    )


def metric_summary_from_kmeans(result: KMeansResult) -> MetricSummary:
    """Scalar diagnostics from a :class:`KMeansResult`."""
    return MetricSummary(
        title="K-means",
        metrics={
            "n_clusters": result.n_clusters,
            "inertia": result.inertia,
            "silhouette_score": result.silhouette_score,
            "n_iter": result.n_iter,
        },
    )


def metric_summary_from_stability(report: ClusterStabilityReport) -> MetricSummary:
    """Scalar diagnostics from a :class:`ClusterStabilityReport`."""
    return MetricSummary(
        title="Cluster stability",
        metrics={
            "adjusted_rand_index": report.adjusted_rand_index,
            "mean_stability": report.mean_stability,
            "n_seeds": report.n_seeds,
        },
    )


def metric_summary_from_density(
    report: DensityEvaluationReport | DensityStabilityReport,
) -> MetricSummary:
    """Scalar diagnostics from a density evaluation or stability report."""
    if isinstance(report, DensityStabilityReport):
        return MetricSummary(
            title="Density stability",
            metrics={
                "mean_auroc": report.mean_auroc,
                "auroc_ci95": report.auroc_ci95,
                "mean_auprc": report.mean_auprc,
                "auprc_ci95": report.auprc_ci95,
                "n_seeds": len(report.seeds),
            },
        )
    return MetricSummary(
        title="Density evaluation",
        metrics={
            "auroc": report.metrics.auroc,
            "auprc": report.metrics.auprc,
            "brier_score": report.metrics.brier_score,
            "n_id": report.metrics.n_id,
            "n_ood": report.metrics.n_ood,
        },
    )


def metric_summary_from_sae(result: SAEEvaluationResult) -> MetricSummary:
    """Scalar diagnostics from an :class:`SAEEvaluationResult`."""
    return MetricSummary(
        title="SAE evaluation",
        metrics={
            "reconstruction_mse": result.reconstruction_mse,
            "mean_l0": result.mean_l0,
            "mean_l1": result.mean_l1,
            "dead_fraction": result.dead_fraction,
            "n_dead_features": result.n_dead_features,
        },
    )


def metric_summary_from_atlas(atlas: FeatureAtlas) -> MetricSummary:
    """Scalar diagnostics from a portable :class:`FeatureAtlas`."""
    n_dead = sum(1 for entry in atlas.entries if entry.is_dead)
    return MetricSummary(
        title="Feature atlas",
        metrics={
            "n_features": len(atlas.entries),
            "n_dead": n_dead,
            "dead_fraction": n_dead / len(atlas.entries) if atlas.entries else 0.0,
            "n_examples": atlas.n_examples,
        },
    )


# ── Responsiveness / downsampling ─────────────────────────────────────────


def downsample_view(view: ProjectionView, *, limit: int, seed: int = DOWNSAMPLE_SEED) -> ProjectionView:
    """Deterministically reduce a view to at most ``limit`` background points.

    Downsampling is stratified per category (points without a category share
    one stratum) so class balance is roughly preserved, and it is seeded so
    the same view downsamples to the same subset. Trajectory overlays are
    never downsampled — they are the structure the user is inspecting.

    When no downsampling is needed the view is returned unchanged.
    """
    if limit < 1:
        msg = f"limit must be >= 1, got {limit}"
        raise ValueError(msg)
    if view.n_points <= limit:
        return view
    rng = np.random.default_rng(seed)
    strata: dict[str, list[int]] = {}
    for index, point in enumerate(view.points):
        strata.setdefault(point.category or "", []).append(index)
    selected: list[int] = []
    remaining = limit
    for category in sorted(strata):
        indices = list(strata[category])
        target = max(1, round(limit * len(indices) / view.n_points))
        target = min(target, len(indices), remaining)
        if target < 1:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:target])
        remaining -= target
    if len(selected) > limit:
        selected = sorted(selected)[:limit]
    metadata = dict(view.metadata)
    metadata.update({"n_dropped": view.n_points - len(selected), "n_kept": len(selected), "downsampled": True})
    return ProjectionView(
        points=tuple(view.points[index] for index in sorted(selected)),
        trajectories=view.trajectories,
        title=view.title,
        metadata=metadata,
    )


def default_point_limit(ndim: int) -> int:
    """Return the declared responsiveness target for a projection dimension."""
    return DEFAULT_POINT_LIMIT_3D if ndim == 3 else DEFAULT_POINT_LIMIT_2D
