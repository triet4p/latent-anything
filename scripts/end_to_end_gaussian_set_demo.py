"""End-to-end demo: Gaussian-set latent geometry.

Shows the three key operations on a Gaussian-set latent space:
1. Construction and parameter layout
2. Permutation-aware distance
3. Interpolation with constrained fields (log-scale, clamped opacity/color)

Output: ``artifacts/gaussian_set_demo_plot.png``

Run: uv run python scripts/end_to_end_gaussian_set_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from latent_anything import LatentSpace


def make_gaussian_set(
    n_gaussians: int, seed: int, shift: float = 0.0
) -> np.ndarray:
    """Create a synthetic Gaussian-set point.

    Returns shape ``(n_gaussians, 10)`` with columns:
    position(3) + scale(3) + opacity(1) + color(3).
    """
    rng = np.random.default_rng(seed)
    point = np.zeros((n_gaussians, 10))
    # Position: normal, optionally shifted
    point[:, :3] = rng.normal(loc=shift, size=(n_gaussians, 3))
    # Scale: log-normal (always > 0)
    point[:, 3:6] = np.exp(rng.normal(size=(n_gaussians, 3)))
    # Opacity: uniform in [0, 1]
    point[:, 6] = rng.uniform(0.1, 0.9, size=n_gaussians)
    # Color: uniform in [0, 1]
    point[:, 7:10] = rng.uniform(0.0, 1.0, size=(n_gaussians, 3))
    return point


def main() -> None:
    n_gaussians = 20

    # ── 1. Construction ──────────────────────────────────────────────
    space = LatentSpace(
        dim=10,
        geometry="gaussian_set",
        n_gaussians=n_gaussians,
        source_model="gaussian_synthetic",
    )
    print(f"Gaussian-set space: {space}")
    print(f"  shape:        {space.shape}")
    print(f"  param_dim:    {space.param_dim}")
    print(f"  n_gaussians:  {space.n_gaussians}")
    print(f"  param layout: {space.metadata['gaussian_set_param_layout']}")

    # ── 2. Create two points (source cluster, target cluster) ────────
    a = make_gaussian_set(n_gaussians, seed=42, shift=0.0)
    b = make_gaussian_set(n_gaussians, seed=99, shift=5.0)

    # Validate both
    space.validate_point(a)
    space.validate_point(b)
    print(f"\nPoint A shape: {a.shape}, Point B shape: {b.shape}")
    print(f"  A position range: [{a[:, :3].min():.2f}, {a[:, :3].max():.2f}]")
    print(f"  B position range: [{b[:, :3].min():.2f}, {b[:, :3].max():.2f}]")

    # ── 3. Permutation-aware distance ────────────────────────────────
    d_ab = space.distance(a, b)
    print(f"\nDistance(A, B) = {d_ab:.4f}")

    # Permutation invariance: shuffle A rows
    rng = np.random.default_rng(123)
    perm = rng.permutation(n_gaussians)
    a_shuffled = a[perm].copy()
    d_shuffled = space.distance(a_shuffled, b)
    print(f"Distance(shuffled(A), B) = {d_shuffled:.4f}  (should match)")

    # Self distance
    d_aa = space.distance(a, a)
    print(f"Distance(A, A) = {d_aa:.4f}  (should be 0)")

    # ── 4. Interpolation at multiple steps ───────────────────────────
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {t: space.interpolate(a, b, t) for t in ts}

    # Validate all interpolated points
    for t, pt in results.items():
        space.validate_point(pt)
        pos_mean = pt[:, :3].mean(axis=0)
        print(f"  t={t:.2f}: pos_mean=({pos_mean[0]:.2f}, {pos_mean[1]:.2f}, {pos_mean[2]:.2f})"
              f"  opacity_mean={pt[:, 6].mean():.3f}")

    # ── 5. Visualize ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Gaussian-Set Latent Geometry — Interpolation Paths", fontsize=14)

    # Row 0: Position (x, y), scale (mean), opacity, color (R)
    ax_pos = axes[0, 0]
    ax_scale = axes[0, 1]
    ax_opacity = axes[0, 2]

    # Row 1: Color (G, B), distance profile, validation info
    ax_color = axes[1, 0]
    ax_dist = axes[1, 1]
    ax_info = axes[1, 2]

    colors_ts = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, t in enumerate(ts):
        pt = results[t]
        c = colors_ts[i]
        ax_pos.scatter(pt[:, 0], pt[:, 1], c=c, alpha=0.7, s=30, label=f"t={t:.2f}")
        ax_scale.scatter(pt[:, 3], pt[:, 4], c=c, alpha=0.7, s=30, label=f"t={t:.2f}")
        ax_opacity.scatter(
            np.full(n_gaussians, t), pt[:, 6], c=c, alpha=0.5, s=20
        )
        ax_color.scatter(pt[:, 7], pt[:, 8], c=c, alpha=0.7, s=30, label=f"t={t:.2f}")

    ax_pos.set_title("Position (x, y)")
    ax_pos.set_xlabel("x"); ax_pos.set_ylabel("y")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    ax_scale.set_title("Scale (sx, sy)")
    ax_scale.set_xlabel("sx"); ax_scale.set_ylabel("sy")
    ax_scale.legend(fontsize=8)
    ax_scale.grid(True, alpha=0.3)

    ax_opacity.set_title("Opacity (all Gaussians)")
    ax_opacity.set_xlabel("interpolation t"); ax_opacity.set_ylabel("opacity")
    ax_opacity.set_ylim(-0.1, 1.1)
    ax_opacity.grid(True, alpha=0.3)

    ax_color.set_title("Color (R, G)")
    ax_color.set_xlabel("R"); ax_color.set_ylabel("G")
    ax_color.legend(fontsize=8)
    ax_color.grid(True, alpha=0.3)

    # Distance profile
    dists = [space.distance(a, results[t]) for t in ts]
    ax_dist.plot(ts, dists, "o-", color="#2ca02c")
    ax_dist.set_title("Distance from A Along Path")
    ax_dist.set_xlabel("interpolation t"); ax_dist.set_ylabel("distance(A, interp(t))")
    ax_dist.grid(True, alpha=0.3)

    ax_info.axis("off")
    info_text = (
        f"Space: {space}\n\n"
        f"n_gaussians: {n_gaussians}\n"
        f"param_dim: {space.param_dim}\n"
        f"param_layout:\n  position(3) + scale(3)\n"
        f"  + opacity(1) + color(3)\n\n"
        f"Distance(A, B): {d_ab:.2f}\n"
        f"Permutation invariant: "
        f"{'YES' if abs(d_ab - d_shuffled) < 1e-10 else 'NO'}\n\n"
        f"Interpolation:\n"
        f"  position: lerp\n"
        f"  scale: log-space lerp\n"
        f"  opacity: lerp + clamp [0,1]\n"
        f"  color: lerp + clamp [0,1]"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "gaussian_set_demo_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
