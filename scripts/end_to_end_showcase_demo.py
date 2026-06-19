#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0,<3.0",
#     "scikit-learn>=1.6,<2.0",
#     "matplotlib>=3.9,<4.0",
#     "torch>=2.5,<3.0",
# ]
# ///

"""Sprint 13 showcase — end-to-end latent edit story with the VAE adapter.

This is the **first composition showcase** of the latent-anything framework.
It composes existing, validated primitives — NOT new abstractions — into a
coherent end-to-end narrative:

    1. Generate structured synthetic cluster data (scaled to [0, 1]).
    2. Train a VAE adapter → encode into latent space.
    3. **Layer A introspection**: PCA projection reveals source/target regions
       and a held-out failure slice.
    4. **Baseline metrics**: compute reconstruction error and distance-to-target
       before any edit.
    5. **Layer B edit**: fit ``ActivationPatch`` from source→target, apply to
       held-out failure samples, decode back to data space.
    6. **Post-edit metrics**: measure how edited outputs moved toward target.
    7. **Trajectory panel**: reuse ``Lerp`` to create latent trajectory, then
       ``ActivationPatch.apply_trajectory()`` for trajectory-level decode.
    8. **Composite visualization**: 2×2 grid + console summary + artifact files.

Usage:
    uv run scripts/end_to_end_showcase_demo.py

Config:
    All parameters are in ``scripts/showcase_config.py``. Edit that file to
    change seed, data shape, VAE params, or split.

Artifacts produced (in ``artifacts/``):
    - ``showcase_demo_plot.png`` — composite 2×2 figure
    - ``showcase_demo_summary.txt`` — console-style metric report
    - ``showcase_config_snapshot.txt`` — frozen copy of the config used
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Ellipse

# Ensure we can import from src and the config file
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from showcase_config import SHOWCASE_CONFIG, OutputConfig, ShowcaseConfig  # noqa: E402

from latent_anything import Trajectory  # noqa: E402
from latent_anything.adapters import VAE  # noqa: E402
from latent_anything.methods import PCA, ActivationPatch, Lerp  # noqa: E402

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class ClusterInfo(TypedDict):
    cluster_centers: FloatArray
    indices_by_label: dict[int, IntArray]


class SplitResult(TypedDict):
    source_data: FloatArray
    target_data: FloatArray
    failure_data: FloatArray
    test_source_idx: IntArray
    test_target_idx: IntArray
    test_target_ref: FloatArray


class BaselineMetrics(TypedDict):
    recon_mse_failure: float
    recon_mse_source: float
    recon_mse_target: float
    dist_to_target_before: float
    centroid_source_to_target: float


class PostMetrics(TypedDict):
    dist_to_target_after: float
    improvement_ratio: float
    dist_delta: float


TrajectoryPanelResult = tuple[FloatArray, FloatArray, FloatArray, FloatArray, Trajectory]
PCAProjectionResult = tuple[PCA, FloatArray, FloatArray, FloatArray]

# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------


def _generate_data(cfg: ShowcaseConfig) -> tuple[FloatArray, IntArray, ClusterInfo]:
    """Generate structured synthetic cluster data.

    Returns
    -------
    points : np.ndarray  shape (total, input_dim)
    labels : np.ndarray  shape (total,)
    cluster_info : dict with keys ``cluster_centers``, ``indices_by_label``
    """
    dc = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    n_clusters = dc["n_clusters"]
    n_per = dc["n_per_cluster"]
    input_dim = dc["input_dim"]
    noise_scale = dc["noise_scale"]

    # Four well-separated cluster centers in 8D
    centers_raw: list[FloatArray] = [
        np.array([0.9, 0.1, 0.9, 0.1, 0.5, 0.1, 0.1, 0.1], dtype=np.float64),
        np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.5, 0.1, 0.9], dtype=np.float64),
        np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.5, 0.1], dtype=np.float64),
        np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.9, 0.9], dtype=np.float64),
    ]
    # Truncate or tile centers to match n_clusters
    if n_clusters < len(centers_raw):
        centers = centers_raw[:n_clusters]
    elif n_clusters > len(centers_raw):
        # Tile to fill
        repeats = (n_clusters + len(centers_raw) - 1) // len(centers_raw)
        centers = (centers_raw * repeats)[:n_clusters]
    else:
        centers = centers_raw

    points_list: list[FloatArray] = []
    label_list: list[IntArray] = []
    for idx, center in enumerate(centers):
        pts = center + rng.normal(scale=noise_scale, size=(n_per, input_dim))
        pts = np.clip(pts, 0.0, 1.0)
        points_list.append(pts)
        label_list.append(np.full(n_per, idx, dtype=np.int64))

    points = np.vstack(points_list)
    labels = np.concatenate(label_list)

    cluster_info: ClusterInfo = {
        "cluster_centers": np.array([c[:input_dim] for c in centers], dtype=np.float64),
        "indices_by_label": {i: np.where(labels == i)[0] for i in range(n_clusters)},
    }
    return points, labels, cluster_info


# ---------------------------------------------------------------------------
# 2. Config-driven split: source clusters, target clusters, held-out failure
# ---------------------------------------------------------------------------


def _split_data(
    points: FloatArray,
    labels: IntArray,
    cfg: ShowcaseConfig,
) -> SplitResult:
    """Split data into source, target, test (held-out failure), and rest.

    Returns dict with keys:
        source_data, target_data, failure_data, test_source_idx, test_target_idx
    """
    sc = cfg["split"]
    source_clusters = sc["source_clusters"]
    target_clusters = sc["target_clusters"]
    n_held = sc["n_held_out"]

    source_mask = np.isin(labels, source_clusters)
    target_mask = np.isin(labels, target_clusters)

    source_all = points[source_mask]
    target_all = points[target_mask]

    rng = np.random.default_rng(cfg["seed"])

    # Shuffle source indices and split
    source_idx = np.arange(len(source_all), dtype=np.int64)
    rng.shuffle(source_idx)
    failure_idx = source_idx[:n_held]
    train_source_idx = source_idx[n_held:]

    # Also shuffle target and hold out some for reference
    target_idx = np.arange(len(target_all), dtype=np.int64)
    rng.shuffle(target_idx)
    test_target_idx = target_idx[:n_held]
    train_target_idx = target_idx[n_held:]

    return {
        "source_data": source_all[train_source_idx],
        "target_data": target_all[train_target_idx],
        "failure_data": source_all[failure_idx],
        "test_source_idx": failure_idx,
        "test_target_idx": test_target_idx,
        "test_target_ref": target_all[test_target_idx],
    }


# ---------------------------------------------------------------------------
# 3. Baseline metrics
# ---------------------------------------------------------------------------


def _compute_baseline_metrics(
    source_data: FloatArray,
    target_data: FloatArray,
    failure_data: FloatArray,
    target_centroid_data: FloatArray,
    vae: VAE,
) -> BaselineMetrics:
    """Compute baseline (pre-edit) metrics for failure samples.

    Metrics computed:
    - ``recon_mse_failure``: reconstruction MSE of failure samples
    - ``recon_mse_source``: reconstruction MSE of source training data
    - ``recon_mse_target``: reconstruction MSE of target training data
    - ``dist_to_target_before``: mean data-space distance from failure samples
      to target centroid (before edit)
    - ``centroid_source_to_target``: data-space distance between source and
      target centroids (reference magnitude)

    Returns
    -------
    dict of metric name → float value
    """
    combined_train = np.vstack([source_data, target_data])
    recon_all = vae.decode(vae.encode(combined_train))
    recon_source = recon_all[: len(source_data)]
    recon_target = recon_all[len(source_data) :]

    # Failure reconstruction
    recon_failure = cast(FloatArray, vae.decode(vae.encode(failure_data)))
    recon_mse_failure = float(np.mean((failure_data - recon_failure) ** 2))
    recon_mse_source = float(np.mean((source_data - recon_source) ** 2))
    recon_mse_target = float(np.mean((target_data - recon_target) ** 2))

    # Data-space distances
    failure_centroid_data = failure_data.mean(axis=0)
    target_centroid_data_val = target_centroid_data.mean(axis=0)

    failure_distances = cast(FloatArray, np.linalg.norm(failure_data - target_centroid_data_val, axis=1))
    dist_to_target = float(np.mean(failure_distances))
    centroid_dist = float(np.linalg.norm(failure_centroid_data - target_centroid_data_val))

    return {
        "recon_mse_failure": recon_mse_failure,
        "recon_mse_source": recon_mse_source,
        "recon_mse_target": recon_mse_target,
        "dist_to_target_before": dist_to_target,
        "centroid_source_to_target": centroid_dist,
    }


# ---------------------------------------------------------------------------
# 4. Layer A: PCA introspection of latent space
# ---------------------------------------------------------------------------


def _project_latent_pca(
    encoded_source: FloatArray,
    encoded_target: FloatArray,
    encoded_failure: FloatArray,
    n_components: int = 2,
) -> PCAProjectionResult:
    """Fit PCA on source+target latents and project all three sets.

    Returns
    -------
    pca : PCA
        Fitted PCA method.
    proj_source, proj_target, proj_failure : np.ndarray
        2D projections.
    """
    all_latent = np.vstack([encoded_source, encoded_target])
    pca = PCA(n_components=n_components)
    pca.fit(all_latent)

    proj_source = cast(FloatArray, pca.transform(encoded_source))
    proj_target = cast(FloatArray, pca.transform(encoded_target))
    proj_failure = cast(FloatArray, pca.transform(encoded_failure))
    return pca, proj_source, proj_target, proj_failure


# ---------------------------------------------------------------------------
# 5. Layer B: ActivationPatch edit
# ---------------------------------------------------------------------------


def _apply_activation_patch(
    vae: VAE,
    source_data: FloatArray,
    target_data: FloatArray,
    failure_data: FloatArray,
) -> tuple[ActivationPatch, FloatArray]:
    """Fit an ``ActivationPatch`` and apply to failure samples.

    Returns
    -------
    patch : ActivationPatch
        Fitted patch.
    edited : np.ndarray
        Edited failure samples in data space.
    """
    patch = ActivationPatch(adapter=vae)
    patch.fit(source_data=source_data, target_data=target_data)
    edited = cast(FloatArray, patch(failure_data))
    return patch, edited


def _compute_post_metrics(
    edited_data: FloatArray,
    target_centroid: FloatArray,
    baseline: BaselineMetrics,
) -> PostMetrics:
    """Compute post-edit metrics and compare to baseline.

    Returns
    -------
    dict with additional keys:
        dist_to_target_after, improvement_ratio
    """
    edited_distances = cast(FloatArray, np.linalg.norm(edited_data - target_centroid.mean(axis=0), axis=1))
    dist_after = float(np.mean(edited_distances))
    dist_before = baseline["dist_to_target_before"]
    improvement = (dist_before - dist_after) / max(dist_before, 1e-12)

    return {
        "dist_to_target_after": dist_after,
        "improvement_ratio": float(improvement),
        "dist_delta": float(dist_before - dist_after),
    }


# ---------------------------------------------------------------------------
# 6. Trajectory panel (Lerp + ActivationPatch trajectory)
# ---------------------------------------------------------------------------


def _build_trajectory_panel(
    vae: VAE,
    patch: ActivationPatch,
    source_data: FloatArray,
    target_data: FloatArray,
    n_steps: int = 6,
) -> TrajectoryPanelResult:
    """Build latent trajectory with Lerp and compare with ActivationPatch.

    1. Compute mean latent centroids for source and target.
    2. Create a Lerp trajectory between them.
    3. Decode the trajectory directly (no patch) → data space.
    4. Apply ``ActivationPatch.apply_trajectory()`` → patched data space.

    Returns
    -------
    latent_a, latent_b : np.ndarray  (latent_dim,) — endpoints
    orig_decoded : np.ndarray  (n_points, input_dim) — decoded trajectory (no patch)
    patched_decoded : np.ndarray  (n_points, input_dim) — patched decoded trajectory
    traj_lerp : Trajectory — the latent-space Lerp trajectory
    """
    latent_source = cast(FloatArray, vae.encode(source_data).mean(axis=0))
    latent_target = cast(FloatArray, vae.encode(target_data).mean(axis=0))

    # Lerp trajectory in latent space
    lerp = Lerp()
    endpoints = Trajectory(data=np.vstack([latent_source, latent_target]))
    traj_lerp = lerp.blend_sequence(endpoints, n_steps=n_steps)

    # Decode trajectory directly (no patch)
    orig_decoded = cast(FloatArray, vae.decode(traj_lerp.to_numpy()))

    # Apply ActivationPatch to trajectory
    patched_decoded = cast(FloatArray, patch.apply_trajectory(traj_lerp))

    return latent_source, latent_target, orig_decoded, patched_decoded, traj_lerp


# ---------------------------------------------------------------------------
# 7. Composite visualization
# ---------------------------------------------------------------------------


def _plot_composite(
    pca: PCA,
    proj_source: FloatArray,
    proj_target: FloatArray,
    proj_failure: FloatArray,
    failure_data: FloatArray,
    edited_data: FloatArray,
    target_centroid_data: FloatArray,
    orig_decoded: FloatArray,
    patched_decoded: FloatArray,
    baseline: BaselineMetrics,
    post: PostMetrics,
    cfg: ShowcaseConfig,
    output_path: str,
) -> None:
    """Build a 2×2 composite figure.

    Panel (1): PCA of latent space — source, target, failure regions.
    Panel (2): Before/after in data space — failure reconstruction vs edited.
    Panel (3): Trajectory morphing — decoded lerp trajectory vs patched trajectory.
    Panel (4): Metric summary panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ---- Panel 1: PCA introspection of latent space ----
    ax1 = axes[0, 0]
    ax1.scatter(proj_source[:, 0], proj_source[:, 1], c="blue", alpha=0.4, s=25, label="source (train)")
    ax1.scatter(proj_target[:, 0], proj_target[:, 1], c="green", alpha=0.4, s=25, label="target (train)")
    ax1.scatter(
        proj_failure[:, 0],
        proj_failure[:, 1],
        c="red",
        marker="x",
        s=80,
        linewidths=2,
        label="failure (held-out)",
    )

    # Highlight failure region ellipse
    if len(proj_failure) >= 3:
        cov = np.cov(proj_failure.T)
        if cov.size == 4:
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ellipse = Ellipse(
                xy=proj_failure.mean(axis=0),
                width=4 * np.sqrt(eigvals[0]),
                height=4 * np.sqrt(eigvals[1]),
                angle=angle,
                facecolor="red",
                alpha=0.08,
                edgecolor="red",
                linewidth=1,
                linestyle="--",
            )
            ax1.add_patch(ellipse)

    ev_ratio = pca.explained_variance_ratio_
    ax1.set_title(
        f"(1) Latent Space (PCA)\nsource→target direction visible\nvar explained: {ev_ratio[0]:.2f}, {ev_ratio[1]:.2f}",
        fontsize=10,
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(True, alpha=0.2)

    # ---- Panel 2: Before/after in data space ----
    ax2 = axes[0, 1]
    # Pick 2D for display: PCA of data space
    from sklearn.decomposition import PCA as _SKPCA  # pyright: ignore[reportMissingTypeStubs]

    data_pca = _SKPCA(n_components=2)
    all_data_2d: FloatArray = data_pca.fit_transform(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        np.vstack([failure_data, edited_data, target_centroid_data])
    )
    n_f = len(failure_data)
    fail_2d = cast(FloatArray, all_data_2d[:n_f])
    edit_2d = cast(FloatArray, all_data_2d[n_f : 2 * n_f])
    tgt_2d = cast(FloatArray, all_data_2d[2 * n_f :])

    ax2.scatter(fail_2d[:, 0], fail_2d[:, 1], c="blue", marker="o", s=40, alpha=0.6, label="before (failure)")
    ax2.scatter(edit_2d[:, 0], edit_2d[:, 1], c="red", marker="^", s=40, alpha=0.8, label="after (edited)")
    ax2.scatter(tgt_2d[:, 0], tgt_2d[:, 1], c="green", marker="s", s=40, alpha=0.5, label="target centroid")

    # Arrows from before → after
    for i in range(min(n_f, 5)):
        ax2.annotate(
            "",
            xy=(edit_2d[i, 0], edit_2d[i, 1]),
            xytext=(fail_2d[i, 0], fail_2d[i, 1]),
            arrowprops=dict(arrowstyle="->", color="purple", lw=1.2, alpha=0.6),
        )

    ax2.set_title(
        f"(2) Before vs After (data-space PCA)\n"
        f"dist to target: {baseline['dist_to_target_before']:.4f} → {post['dist_to_target_after']:.4f}\n"
        f"improvement: {post['improvement_ratio'] * 100:.1f}%",
        fontsize=10,
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, alpha=0.2)

    # ---- Panel 3: Trajectory morphing ----
    ax3 = axes[1, 0]
    # Project decoded trajectories via shared PCA
    all_traj = np.vstack([orig_decoded, patched_decoded])
    traj_pca = _SKPCA(n_components=2)
    traj_pca.fit(all_traj)  # pyright: ignore[reportUnknownMemberType]
    orig_2d = cast(FloatArray, traj_pca.transform(orig_decoded))  # pyright: ignore[reportUnknownMemberType]
    patch_2d = cast(FloatArray, traj_pca.transform(patched_decoded))  # pyright: ignore[reportUnknownMemberType]

    ax3.plot(orig_2d[:, 0], orig_2d[:, 1], "c-o", linewidth=2, markersize=6, label="lerp trajectory (decode)")
    ax3.plot(patch_2d[:, 0], patch_2d[:, 1], "r-^", linewidth=2, markersize=6, label="patched trajectory")
    for i in range(len(orig_decoded)):
        ax3.annotate(
            "",
            xy=(patch_2d[i, 0], patch_2d[i, 1]),
            xytext=(orig_2d[i, 0], orig_2d[i, 1]),
            arrowprops=dict(arrowstyle="->", color="purple", lw=0.8, alpha=0.4),
        )

    ax3.set_title(
        f"(3) Trajectory Morphing\n"
        f"Lerp latent trajectory → decode → patched decode\n"
        f"(n_steps={cfg['lerp']['n_steps']})",
        fontsize=10,
    )
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.legend(fontsize=7, loc="best")
    ax3.grid(True, alpha=0.2)

    # ---- Panel 4: Metric summary ----
    ax4 = axes[1, 1]
    ax4.axis("off")
    metric_lines = [
        "─── Showcase Metrics ───",
        f"Seed: {cfg['seed']}",
        "",
        "Reconstruction MSE:",
        f"  Source:    {baseline['recon_mse_source']:.6f}",
        f"  Target:    {baseline['recon_mse_target']:.6f}",
        f"  Failure:   {baseline['recon_mse_failure']:.6f}",
        "",
        "Data-space distance to target centroid:",
        f"  Before:    {baseline['dist_to_target_before']:.4f}",
        f"  After:     {post['dist_to_target_after']:.4f}",
        f"  Delta:     {post['dist_delta']:+.4f}",
        f"  Improve:   {post['improvement_ratio'] * 100:.1f}%",
        "",
        "Source→target centroid dist:",
        f"  {baseline['centroid_source_to_target']:.4f}",
        "",
        "Artifacts:",
        f"  Figure:   {output_path}",
    ]
    ax4.text(
        0.05,
        0.95,
        "\n".join(metric_lines),
        transform=ax4.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    plt.suptitle(  # pyright: ignore[reportUnknownMemberType]
        "Sprint 13 Showcase — VAE → PCA → ActivationPatch → Decode\n"
        "Composition of existing primitives (no new abstraction)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
    print(f"\n  ✓ Composite figure saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Console summary
# ---------------------------------------------------------------------------


def _print_summary(
    baseline: BaselineMetrics,
    post: PostMetrics,
    cfg: ShowcaseConfig,
    output_paths: OutputConfig,
) -> None:
    """Print a structured console summary with seed, metrics, and artifact paths."""
    sep = "=" * 62
    print(f"\n{sep}")
    print("  Sprint 13 Showcase — Summary")
    print(f"{sep}")
    print(f"  Seed:                {cfg['seed']}")
    print(
        f"  Data:                {cfg['data']['n_clusters']} clusters × "
        f"{cfg['data']['n_per_cluster']} pts, "
        f"{cfg['data']['input_dim']}D → latent {cfg['data']['latent_dim']}D"
    )
    print(f"  VAE epochs:          {cfg['vae']['n_epochs']}")
    print(f"  Held-out failure:    {cfg['split']['n_held_out']} samples")
    print(f"{sep}")
    print("  Reconstruction MSE:")
    print(f"    Source:            {baseline['recon_mse_source']:.6f}")
    print(f"    Target:            {baseline['recon_mse_target']:.6f}")
    print(f"    Failure:           {baseline['recon_mse_failure']:.6f}")
    print(f"{sep}")
    print("  Distance to target (data-space):")
    print(f"    Before edit:       {baseline['dist_to_target_before']:.4f}")
    print(f"    After edit:        {post['dist_to_target_after']:.4f}")
    print(f"    Improvement:       {post['improvement_ratio'] * 100:.1f}%")
    print(f"{sep}")
    print("  Artifacts:")
    for key, path in output_paths.items():
        print(f"    {key}: {path}")
    print(f"{sep}")

    # Also write to summary file
    summary_path = _REPO_ROOT / output_paths["summary"]
    lines = [
        f"{sep}",
        "  Sprint 13 Showcase — Summary",
        f"{sep}",
        f"  Seed:                {cfg['seed']}",
        f"  Data:                {cfg['data']['n_clusters']} clusters × "
        f"{cfg['data']['n_per_cluster']} pts, "
        f"{cfg['data']['input_dim']}D → latent {cfg['data']['latent_dim']}D",
        f"  VAE epochs:          {cfg['vae']['n_epochs']}",
        f"  Held-out failure:    {cfg['split']['n_held_out']} samples",
        f"{sep}",
        "  Reconstruction MSE:",
        f"    Source:            {baseline['recon_mse_source']:.6f}",
        f"    Target:            {baseline['recon_mse_target']:.6f}",
        f"    Failure:           {baseline['recon_mse_failure']:.6f}",
        f"{sep}",
        "  Distance to target (data-space):",
        f"    Before edit:       {baseline['dist_to_target_before']:.4f}",
        f"    After edit:        {post['dist_to_target_after']:.4f}",
        f"    Improvement:       {post['improvement_ratio'] * 100:.1f}%",
        f"{sep}",
        "  Artifacts:",
    ]
    for key, path in output_paths.items():
        lines.append(f"    {key}: {path}")
    lines.append(f"{sep}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Summary saved to {summary_path}")

    # Save config snapshot
    config_path = _REPO_ROOT / output_paths["config_snapshot"]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(str(cfg), encoding="utf-8")
    print(f"  ✓ Config snapshot saved to {config_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = SHOWCASE_CONFIG
    output_paths = cfg["output"]

    print("=" * 62)
    print("  Sprint 13 Showcase — end-to-end latent edit story")
    print("  VAE → PCA (Layer A) → ActivationPatch (Layer B) → Decode")
    print(f"  Seed: {cfg['seed']}")
    print("=" * 62)

    # 1. Generate data
    print("\n[1/7] Generating synthetic cluster data...")
    points, labels, _cluster_info = _generate_data(cfg)
    total_pts = len(points)
    print(
        f"       {total_pts} points, {cfg['data']['n_clusters']} clusters, "
        f"{cfg['data']['input_dim']}D → latent {cfg['data']['latent_dim']}D"
    )

    # 2. Split
    print("\n[2/7] Splitting data into source / target / failure...")
    split = _split_data(points, labels, cfg)
    print(f"       Source: {len(split['source_data'])} train samples")
    print(f"       Target: {len(split['target_data'])} train samples")
    print(f"       Failure (held-out): {len(split['failure_data'])} samples")

    # 3. Train VAE
    print("\n[3/7] Training VAE adapter...")
    vae_params = cfg["vae"]
    vae = VAE(
        input_dim=cfg["data"]["input_dim"],
        latent_dim=cfg["data"]["latent_dim"],
        hidden_dim=vae_params["hidden_dim"],
        n_epochs=vae_params["n_epochs"],
        learning_rate=vae_params["learning_rate"],
        beta=vae_params["beta"],
        random_state=cfg["seed"],
    )
    # Train on combined source + target
    combined_train = np.vstack([split["source_data"], split["target_data"]])
    vae.fit(combined_train)
    print(f"       Initial loss: {vae.loss_history_[0]:.6f}")
    print(f"       Final loss:   {vae.loss_history_[-1]:.6f}")

    # 4. Encode → baseline metrics
    print("\n[4/7] Computing baseline metrics...")
    baseline = _compute_baseline_metrics(
        split["source_data"],
        split["target_data"],
        split["failure_data"],
        split["target_data"],
        vae,
    )
    print(f"       Recon MSE (failure): {baseline['recon_mse_failure']:.6f}")
    print(f"       Dist to target (before): {baseline['dist_to_target_before']:.4f}")

    # 5. Layer A: PCA introspection
    print("\n[5/7] PCA introspection of latent space...")
    encoded_source = vae.encode(split["source_data"])
    encoded_target = vae.encode(split["target_data"])
    encoded_failure = vae.encode(split["failure_data"])
    pca, proj_source, proj_target, proj_failure = _project_latent_pca(
        encoded_source,
        encoded_target,
        encoded_failure,
        n_components=cfg["pca"]["n_components"],
    )
    print(f"       PCA explained variance: {pca.explained_variance_ratio_}")

    # 6. Layer B: ActivationPatch edit
    print("\n[6/7] Applying ActivationPatch edit...")
    patch, edited_data = _apply_activation_patch(
        vae,
        split["source_data"],
        split["target_data"],
        split["failure_data"],
    )
    print(f"       Patch delta norm: {np.linalg.norm(patch.delta):.6f}")
    post = _compute_post_metrics(
        edited_data,
        split["target_data"],
        baseline,
    )
    print(f"       Dist to target (after):  {post['dist_to_target_after']:.4f}")
    print(f"       Improvement:             {post['improvement_ratio'] * 100:.1f}%")

    # 7. Trajectory panel
    print("\n[7/7] Building trajectory panel...")
    _latent_a, _latent_b, orig_decoded, patched_decoded, traj_lerp = _build_trajectory_panel(
        vae,
        patch,
        split["source_data"],
        split["target_data"],
        n_steps=cfg["lerp"]["n_steps"],
    )
    print(f"       Lerp trajectory: {len(traj_lerp)} latent points")

    # Composite figure
    output_fig = str(_REPO_ROOT / output_paths["figure"])
    _plot_composite(
        pca,
        proj_source,
        proj_target,
        proj_failure,
        split["failure_data"],
        edited_data,
        split["target_data"],
        orig_decoded,
        patched_decoded,
        baseline,
        post,
        cfg,
        output_fig,
    )

    # Summary
    _print_summary(baseline, post, cfg, output_paths)

    print("\n  ✓ Sprint 13 showcase complete.")


if __name__ == "__main__":
    main()
