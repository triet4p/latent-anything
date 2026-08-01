"""Interactive real-model walkthrough: every chart tied to quantitative metrics.

Runs a compact, CPU-reproducible pipeline on a real model (ConvVAE trained on
sklearn digits) and renders each analysis with the interactive Plotly explorer,
exporting self-contained HTML plus PNG thumbnails and a metrics JSON:

- K-means projection           → silhouette score, inertia, cluster sizes
- Linear probe projection      → accuracy, validation accuracy, n_classes
- GMM density ID/OOD projection→ AUROC, AUPRC, Brier score
- Interpolation trajectory     → geodesic path between digit-class centroids
- SAE feature atlas            → dead-fraction, activation frequency, decoder norms
- 60k-point responsiveness     → declared downsampling target and render time

Usage::

    uv run python scripts/interactive_viz_walkthrough.py            # full run
    uv run python scripts/interactive_viz_walkthrough.py --quick     # faster run

Artifacts are written to ``artifacts/interactive-viz-walkthrough/``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]

from latent_anything import Trajectory
from latent_anything.adapters.conv_vae import ConvVAE
from latent_anything.clustering import KMeans, KMeansConfig
from latent_anything.density import GaussianMixtureDensity, GMMConfig
from latent_anything.methods import PCA, Lerp
from latent_anything.probes import LinearProbe, LinearProbeConfig
from latent_anything.sae_evaluation import SAEConfig, SAEFeatureEvaluation, build_feature_atlas
from latent_anything.visualization import (
    ProjectionExplorer,
    ProjectionView,
    build_projection,
    projection_explorer,
    projection_from_atlas,
    projection_from_density,
    projection_from_kmeans,
    projection_from_probe,
    projection_from_trajectory,
)

ARTIFACT_DIR = Path("artifacts/interactive-viz-walkthrough")
SOURCE_IDENTITY = "conv_vae_digits_8x8_latent_dim_4"


class _DigitsDataset(Protocol):
    images: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class WalkthroughResult:
    """Views, metrics, and responsiveness numbers from one walkthrough run."""

    views: dict[str, ProjectionView]
    metrics: dict[str, dict[str, float | int | str]]
    responsiveness: dict[str, float | int]
    artifact_dir: Path


@dataclass(frozen=True)
class _Fit:
    adapter: ConvVAE
    latents: np.ndarray
    labels: np.ndarray


def _load_digits(n: int) -> tuple[np.ndarray, np.ndarray]:
    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images[:n] / 16.0).astype(np.float64)[:, None, :, :]
    labels = np.asarray(digits.target[:n], dtype=np.int64)
    return images, labels


def _fit_encoder(images: np.ndarray, labels: np.ndarray, *, n_epochs: int, latent_dim: int) -> _Fit:
    adapter = ConvVAE(latent_dim=latent_dim, random_state=42, n_epochs=n_epochs)
    adapter.fit(images)
    latents = adapter.encode_value(images).to_numpy()
    return _Fit(adapter=adapter, latents=latents, labels=labels)


def _project(latents: np.ndarray, n_components: int) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components)
    pca.fit(latents)
    return pca, np.asarray(pca.transform(latents), dtype=np.float64)


def _kmeans_view(
    latents: np.ndarray, coordinates: np.ndarray, *, n_clusters: int
) -> tuple[ProjectionView, dict[str, float | int | str]]:
    result = KMeans(KMeansConfig(n_clusters=n_clusters, random_state=42)).fit_predict(
        latents, provenance={"source": SOURCE_IDENTITY}
    )
    view = projection_from_kmeans(result, coordinates, title="Digits latent space — K-means clusters (2D PCA)")
    metrics: dict[str, float | int | str] = {
        "n_clusters": result.n_clusters,
        "silhouette_score": result.silhouette_score,
        "inertia": result.inertia,
        "cluster_sizes": ",".join(str(int(size)) for size in result.cluster_sizes),
    }
    return view, metrics


def _probe_view(
    latents: np.ndarray, labels: np.ndarray, coordinates: np.ndarray
) -> tuple[ProjectionView, dict[str, float | int | str]]:
    probe = LinearProbe(LinearProbeConfig())
    result = probe.fit(latents, labels, provenance={"source": SOURCE_IDENTITY})
    all_predictions = np.asarray(probe.predict(latents))
    all_probabilities = np.asarray(probe.predict_proba(latents))
    view = projection_from_probe(
        result,
        coordinates,
        predictions=all_predictions,
        probabilities=all_probabilities,
        title="Digits latent space — linear-probe predictions (2D PCA)",
    )
    metrics: dict[str, float | int | str] = {
        "accuracy": result.accuracy,
        "val_accuracy": result.val_accuracy,
        "n_classes": int(len(result.classes)),
        "n_iter": result.n_iter,
    }
    return view, metrics


def _density_view(
    latents: np.ndarray,
    labels: np.ndarray,
    coordinates: np.ndarray,
    *,
    n_components: int,
) -> tuple[ProjectionView, dict[str, float | int | str]]:
    id_mask = labels < 5
    ood_mask = ~id_mask
    estimator = GaussianMixtureDensity(GMMConfig(n_components=n_components, random_state=42))
    estimator.fit(latents[id_mask], source_representation_identity=SOURCE_IDENTITY, geometry="euclidean")
    estimator.calibrate(latents[id_mask])
    result = estimator.score(latents, source_representation_identity=SOURCE_IDENTITY)
    view = projection_from_density(
        result,
        coordinates,
        title="Digits latent space — calibrated OOD score (2D PCA, ID=0-4)",
    )
    report = estimator.evaluate(
        latents[id_mask],
        latents[ood_mask],
        source_representation_identity=SOURCE_IDENTITY,
        split_provenance={"in_distribution": "digits 0-4", "out_of_distribution": "digits 5-9"},
    )
    metrics: dict[str, float | int | str] = {
        "auroc": report.metrics.auroc,
        "auprc": report.metrics.auprc,
        "brier_score": report.metrics.brier_score,
        "n_id": report.metrics.n_id,
        "n_ood": report.metrics.n_ood,
    }
    return view, metrics


def _trajectory_view(
    latents: np.ndarray,
    labels: np.ndarray,
    pca: PCA,
    adapter: ConvVAE,
) -> tuple[ProjectionView, dict[str, float | int | str]]:
    even_center = latents[labels % 2 == 0].mean(axis=0)
    odd_center = latents[labels % 2 == 1].mean(axis=0)
    lerp = Lerp(space=adapter.latent_space)
    steps = np.linspace(0.0, 1.0, 11)
    points = np.stack([np.asarray(lerp(even_center, odd_center, float(step)), dtype=np.float64) for step in steps])
    trajectory = Trajectory(points)
    trajectory_coordinates = np.asarray(pca.transform(points), dtype=np.float64)
    view = projection_from_trajectory(
        trajectory,
        trajectory_coordinates,
        title="Interpolation path between even/odd digit-class centroids (2D PCA)",
        step_labels=[f"t={step:.2f}" for step in steps],
    )
    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    metrics: dict[str, float | int | str] = {
        "n_points": len(trajectory),
        "total_latent_path_length": float(step_lengths.sum()),
        "mean_step_length": float(step_lengths.mean()),
    }
    return view, metrics


def _atlas_view(
    latents: np.ndarray, labels: np.ndarray, *, n_components: int, n_epochs: int
) -> tuple[ProjectionView, dict[str, float | int | str]]:
    rng = np.random.default_rng(42)
    permutation = rng.permutation(len(latents))
    n_val = max(16, int(0.2 * len(latents)))
    val_indices = permutation[:n_val]
    train_indices = permutation[n_val:]
    evaluation = SAEFeatureEvaluation(
        SAEConfig(n_components=n_components, n_epochs=n_epochs, random_state=42, l1_coef=0.1)
    ).fit(
        latents[train_indices],
        val_data=latents[val_indices],
        source_representation_identity=SOURCE_IDENTITY,
        provenance={"dataset": "sklearn-digits", "n_train": int(len(train_indices)), "n_val": int(n_val)},
    )
    atlas = build_feature_atlas(
        evaluation,
        example_labels=[str(int(labels[index])) for index in val_indices],
    )
    view = projection_from_atlas(atlas, title="Feature atlas — activation frequency vs decoder norm (log)")
    metrics: dict[str, float | int | str] = {
        "reconstruction_mse": evaluation.reconstruction_mse,
        "mean_l0": evaluation.mean_l0,
        "dead_fraction": evaluation.dead_fraction,
        "n_features": len(atlas.entries),
    }
    return view, metrics


def measure_responsiveness() -> dict[str, float | int]:
    """Render a 60k-point view and report the declared downsampling behavior."""
    rng = np.random.default_rng(0)
    big_coordinates = rng.random((60_000, 2))
    view = build_projection(
        big_coordinates,
        categories=[f"c{i % 8}" for i in range(60_000)],
        title="Responsiveness check — 60k points",
    )
    start = time.perf_counter()
    figure = projection_explorer(view)
    elapsed = time.perf_counter() - start
    rendered = sum(len(trace.x) for trace in figure.data if trace.type == "scatter")
    return {
        "n_input": 60_000,
        "n_rendered": rendered,
        "render_seconds": float(round(elapsed, 3)),
    }


def build_digits_views(
    *,
    n_samples: int = 300,
    vae_epochs: int = 8,
    latent_dim: int = 4,
    n_clusters: int = 8,
    density_components: int = 3,
    sae_components: int = 16,
    sae_epochs: int = 120,
) -> WalkthroughResult:
    """Run the full walkthrough and return views, metrics, and responsiveness."""
    images, labels = _load_digits(n_samples)
    fitted = _fit_encoder(images, labels, n_epochs=vae_epochs, latent_dim=latent_dim)
    pca2, coords2 = _project(fitted.latents, 2)

    kmeans_view, kmeans_metrics = _kmeans_view(fitted.latents, coords2, n_clusters=n_clusters)
    probe_view, probe_metrics = _probe_view(fitted.latents, labels, coords2)
    density_view, density_metrics = _density_view(fitted.latents, labels, coords2, n_components=density_components)
    trajectory_view, trajectory_metrics = _trajectory_view(fitted.latents, labels, pca2, fitted.adapter)
    atlas_view, atlas_metrics = _atlas_view(fitted.latents, labels, n_components=sae_components, n_epochs=sae_epochs)
    responsiveness = measure_responsiveness()

    vae_metrics: dict[str, float | int | str] = {
        "reconstruction_mse": fitted.adapter.metrics_.get("reconstruction_mse", 0.0),
        "posterior_kl": fitted.adapter.metrics_.get("posterior_kl", 0.0),
        "latent_utilization": fitted.adapter.metrics_.get("latent_utilization", 0.0),
        "n_samples": len(fitted.latents),
    }
    return WalkthroughResult(
        views={
            "kmeans": kmeans_view,
            "probe": probe_view,
            "density": density_view,
            "trajectory": trajectory_view,
            "atlas": atlas_view,
        },
        metrics={
            "adapter": vae_metrics,
            "kmeans": kmeans_metrics,
            "probe": probe_metrics,
            "density": density_metrics,
            "trajectory": trajectory_metrics,
            "atlas": atlas_metrics,
        },
        responsiveness=responsiveness,
        artifact_dir=ARTIFACT_DIR,
    )


def _export(result: WalkthroughResult) -> list[Path]:
    output = result.artifact_dir
    output.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for name, view in result.views.items():
        explorer = ProjectionExplorer(view)
        html_path = explorer.save(output / f"{name}.html", include_plotlyjs="cdn")
        exported.append(html_path)
        png_path = explorer.save(output / f"{name}.png", width=1200, height=800)
        exported.append(png_path)
    return exported


def _print_summary(result: WalkthroughResult, exported: list[Path]) -> None:
    print("Interactive visualization walkthrough — quantitative summary")
    print("=" * 72)
    for group, metrics in result.metrics.items():
        formatted = ", ".join(
            f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items()
        )
        print(f"{group:12s}: {formatted}")
    print(f"{'responsive':12s}: {result.responsiveness}")
    print("=" * 72)
    print(f"Artifacts written to {result.artifact_dir.resolve()}")
    for path in exported:
        print(f"  - {path.name}")


def main() -> None:
    """Run the full walkthrough and write all interactive/static artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a smaller, faster configuration")
    parser.add_argument("--no-export", action="store_true", help="Build views/metrics only (no files)")
    args = parser.parse_args()

    if args.quick:
        result = build_digits_views(n_samples=96, vae_epochs=3, sae_epochs=40, sae_components=8)
    else:
        result = build_digits_views()

    if args.no_export:
        print(json.dumps(result.metrics, indent=2))
        return

    exported = _export(result)
    result.artifact_dir.joinpath("metrics.json").write_text(
        json.dumps(
            {
                "metrics": result.metrics,
                "responsiveness": result.responsiveness,
                "artifacts": [path.name for path in exported],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _print_summary(result, exported)


if __name__ == "__main__":
    main()
