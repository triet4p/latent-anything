#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///

"""End-to-end Lerp demo — B-Method #1, stateless interpolation.

Demonstrates the Lerp class, the first Layer B (Manipulation) method,
across two scenarios:

  **Scenario A (Euclidean)**: Two random latent vectors in 8D → Lerp()
  with no LatentSpace → interpolate at t values → PCA to 2D →
  visualize the straight-line interpolation path.

  **Scenario B (Spherical)**: Unit-norm vectors on sphere → Lerp with
  LatentSpace(geometry="unit_norm") → slerp interpolation → PCA
  projection → compare Euclidean lerp path (straight line leaving
  sphere) vs spherical slerp path (arc on sphere surface). Also
  demonstrates trajectory blending with ``blend_sequence``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Ensure we can import from the src directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latent_anything import LatentSpace, Trajectory
from latent_anything.methods import Lerp  # noqa: F811

# ---------------------------------------------------------------------------
# Scenario A: Euclidean lerp
# ---------------------------------------------------------------------------


def scenario_a_euclidean() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scenario A: generate two random 8D vectors and interpolate.

    Returns
    -------
    a, b : np.ndarray
        The two source vectors.
    t_values : np.ndarray
        The interpolation parameters (0, 0.25, 0.5, 0.75, 1.0).
    path : np.ndarray
        Array of shape ``(len(t_values), 8)`` — interpolated points.
    """
    print("--- Scenario A: Euclidean Lerp ---")
    rng = np.random.default_rng(42)
    a = rng.normal(size=8)
    b = rng.normal(size=8)

    lerp = Lerp()  # No space → default Euclidean

    t_values = np.linspace(0, 1, 5)
    path = np.array([lerp(a, b, t) for t in t_values])

    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  t=0.0 → {path[0]}")
    print(f"  t=0.5 → {path[2]}")
    print(f"  t=1.0 → {path[4]}")

    # Verify endpoints
    np.testing.assert_array_almost_equal(path[0], a)
    np.testing.assert_array_almost_equal(path[-1], b)
    print("  ✓ Endpoints correct")

    return a, b, t_values, path


# ---------------------------------------------------------------------------
# Scenario B: Spherical slerp with trajectory blending
# ---------------------------------------------------------------------------


def _make_unit_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random unit vectors on the sphere."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def scenario_b_spherical() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Trajectory,
    Trajectory,
]:
    """Scenario B: unit-norm vectors with spherical slerp.

    Returns
    -------
    a, b : np.ndarray
        The two source unit vectors.
    t_values : np.ndarray
        Interpolation parameters.
    lerp_path : np.ndarray
        Euclidean lerp path (leaves sphere).
    slerp_path : np.ndarray
        Spherical slerp path (stays on sphere).
    traj_a, traj_b : Trajectory
        Two trajectories for between/blend_sequence demo.
    traj_blended : Trajectory
        Result of ``between(traj_a, traj_b, 0.5)``.
    traj_dense : Trajectory
        Result of ``blend_sequence(traj_a, n_steps=3)``.
    """
    print("\n--- Scenario B: Spherical Slerp ---")

    a = _make_unit_vectors(1, 8, seed=42)[0]
    b = _make_unit_vectors(1, 8, seed=99)[0]

    spherical_ls = LatentSpace(dim=8, geometry="unit_norm")
    spherical_ls = LatentSpace(dim=8, geometry="unit_norm")

    lerp_euc = Lerp()  # No space → Euclidean
    lerp_sph = Lerp(space=spherical_ls)  # With space → slerp

    t_values = np.linspace(0, 1, 7)
    lerp_path = np.array([lerp_euc(a, b, t) for t in t_values])
    slerp_path = np.array([lerp_sph(a, b, t) for t in t_values])

    print(f"  {'t':>5s}  {'lerp norm':>10s}  {'slerp norm':>10s}  {'on sphere?':>10s}")
    for i, t in enumerate(t_values):
        lerp_norm = np.linalg.norm(lerp_path[i])
        slerp_norm = np.linalg.norm(slerp_path[i])
        on_sphere = "✓" if abs(slerp_norm - 1.0) < 1e-10 else "✗"
        print(f"  {t:.2f}  {lerp_norm:10.6f}  {slerp_norm:10.6f}  {on_sphere:>10s}")

    # --- Trajectory blending demo ---
    print("\n  --- trajectory blending ---")

    # Build two random trajectories
    n_points = 5
    points_a = _make_unit_vectors(n_points, 8, seed=42)
    points_b = _make_unit_vectors(n_points, 8, seed=77)

    traj_a = Trajectory(points_a)
    traj_b = Trajectory(points_b)

    # between(): pointwise interpolation at t=0.5
    traj_mid = lerp_sph.between(traj_a, traj_b, 0.5)
    print(f"  between(traj_a, traj_b, 0.5): {traj_mid}")

    # blend_sequence(): densify traj_a
    traj_dense = lerp_sph.blend_sequence(traj_a, n_steps=3)
    print(f"  blend_sequence(traj_a, n_steps=3): {traj_dense}")
    print(f"    original points: {len(traj_a)}, dense points: {len(traj_dense)}")

    return a, b, t_values, lerp_path, slerp_path, traj_a, traj_b, traj_mid, traj_dense


# ---------------------------------------------------------------------------
# Visualization: 1×2 matplotlib
# ---------------------------------------------------------------------------


def plot_lerp_vs_slerp(
    lerp_path: np.ndarray,
    slerp_path: np.ndarray,
    t_values: np.ndarray,
    traj_a: Trajectory | None = None,
    traj_dense: Trajectory | None = None,
) -> None:
    """Side-by-side visualization: Euclidean lerp vs spherical slerp.

    Left: Euclidean lerp path in PCA-projected 2D.
    Right: Spherical slerp path in PCA-projected 2D with sphere outline.
    Both annotated with ``t`` values.
    """
    print("\n--- Visualization ---")

    # Stack everything together for a common PCA fit
    all_points = np.vstack([lerp_path, slerp_path])
    pca = PCA(n_components=2)
    pca.fit(all_points)

    lerp_2d = pca.transform(lerp_path)
    slerp_2d = pca.transform(slerp_path)

    # Also project trajectory points if provided
    traj_a_2d = None
    traj_dense_2d = None
    if traj_a is not None:
        traj_a_2d = pca.transform(traj_a.to_numpy())
    if traj_dense is not None:
        traj_dense_2d = pca.transform(traj_dense.to_numpy())

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- Left: Euclidean lerp path ---
    ax_left.plot(
        lerp_2d[:, 0],
        lerp_2d[:, 1],
        "ro-",
        linewidth=2,
        markersize=8,
        label="lerp path",
    )
    # Annotate with t values
    for i, t in enumerate(t_values):
        ax_left.annotate(
            f"t={t:.2f}",
            (lerp_2d[i, 0], lerp_2d[i, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax_left.scatter(lerp_2d[0, 0], lerp_2d[0, 1], c="blue", s=80, zorder=5, label="a (t=0)")
    ax_left.scatter(lerp_2d[-1, 0], lerp_2d[-1, 1], c="green", s=80, zorder=5, label="b (t=1)")
    ax_left.set_title("Euclidean Lerp Path\n(PCA projection)")
    ax_left.set_xlabel("PC1")
    ax_left.set_ylabel("PC2")
    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.3)

    # --- Right: Spherical slerp path with trajectory blending ---
    ax_right.plot(
        slerp_2d[:, 0],
        slerp_2d[:, 1],
        "go-",
        linewidth=2,
        markersize=8,
        label="slerp path",
    )
    for i, t in enumerate(t_values):
        ax_right.annotate(
            f"t={t:.2f}",
            (slerp_2d[i, 0], slerp_2d[i, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax_right.scatter(slerp_2d[0, 0], slerp_2d[0, 1], c="blue", s=80, zorder=5, label="a (t=0)")
    ax_right.scatter(slerp_2d[-1, 0], slerp_2d[-1, 1], c="green", s=80, zorder=5, label="b (t=1)")

    # Trajectory blending overlay
    if traj_a_2d is not None and traj_dense_2d is not None:
        ax_right.plot(
            traj_a_2d[:, 0],
            traj_a_2d[:, 1],
            "bs--",
            linewidth=1.5,
            markersize=6,
            alpha=0.6,
            label="original traj",
        )
        ax_right.plot(
            traj_dense_2d[:, 0],
            traj_dense_2d[:, 1],
            "m^-",
            linewidth=1,
            markersize=5,
            alpha=0.8,
            label="blend_sequence (×3)",
        )

    # Draw reference unit circle (PCA projection of sphere)
    theta = np.linspace(0, 2 * np.pi, 100)
    # Approximate sphere outline in PCA space
    circle_x = np.cos(theta) * np.max(np.abs(slerp_2d)) * 1.05
    circle_y = np.sin(theta) * np.max(np.abs(slerp_2d)) * 1.05
    ax_right.plot(circle_x, circle_y, "k--", alpha=0.2, linewidth=1, label="sphere (ref)")

    ax_right.set_title("Spherical Slerp Path + Trajectory Blending\n(PCA projection)")
    ax_right.set_xlabel("PC1")
    ax_right.set_ylabel("PC2")
    ax_right.legend(fontsize=7)
    ax_right.grid(True, alpha=0.3)

    fig.suptitle(
        "Lerp — B-Method #1: Stateless Interpolation (Layer B Foundation)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    # Save to file
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "lerp_demo_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Visualization saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  End-to-End Lerp Demo — B-Method #1")
    print("  Stateless Interpolation (Layer B Foundation)")
    print("=" * 60)

    print("\n[Scenario A] Euclidean Lerp")
    a, b, t_values, euc_path = scenario_a_euclidean()

    print("\n[Scenario B] Spherical Slerp + Trajectory Blending")
    (
        sph_a,
        sph_b,
        sph_t_values,
        lerp_path,
        slerp_path,
        traj_a,
        traj_b,
        traj_mid,
        traj_dense,
    ) = scenario_b_spherical()

    plot_lerp_vs_slerp(lerp_path, slerp_path, sph_t_values, traj_a, traj_dense)

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("  ✓ Lerp is B-Method #1 — stateless, pure transform")
    print("  ✓ Euclidean: default (1-t)*a + t*b")
    print("  ✓ Spherical: delegates to LatentSpace.interpolate() (slerp)")
    print("  ✓ Trajectory blending: between() + blend_sequence()")
    print("  ✓ Protocol unchanged — expansion at B-Method #3")
    print("=" * 60)


if __name__ == "__main__":
    main()
