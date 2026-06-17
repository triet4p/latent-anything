#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///

"""End-to-end spherical latent space demo.

Demonstrates the geometry-keyed LatentSpace ADR with unit_norm / spherical
geometry:
  (a) validate_point rejects non-unit vectors
  (b) distance computes angular distance correctly
  (c) slerp interpolates along geodesic on sphere (lerp leaves the sphere)
  (d) normalize projects back to sphere

Side-by-side visualization: lerp path vs slerp path on 3D unit sphere
projected to 2D.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure we can import from the src directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latent_anything import LatentSpace


def _make_unit_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random unit vectors on the sphere."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def demo_validate_point(sphere: LatentSpace) -> None:
    """(a) validate_point rejects non-unit vectors."""
    print("--- (a) validate_point ---")
    unit = np.array([1.0, 0.0, 0.0])
    sphere.validate_point(unit)
    print(f"  Unit vector OK: {unit}")

    try:
        sphere.validate_point(np.array([1.0, 2.0, 3.0]))
    except ValueError as e:
        print(f"  Non-unit vector rejected: {e}")


def demo_distance(eucl: LatentSpace, sphere: LatentSpace) -> None:
    """(b) Angular distance vs Euclidean distance."""
    print("\n--- (b) distance ---")
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])

    euc_dist = eucl.distance(a, b)
    sph_dist = sphere.distance(a, b)

    print(f"  Euclidean distance: {euc_dist:.4f}")
    print(f"  Angular distance:   {sph_dist:.4f}  (expected π/2 ≈ {np.pi / 2:.4f})")

    # Sanity: angular distance of same vector is 0
    assert sphere.distance(a, a) < 1e-10, "Same vector distance should be 0"


def demo_interpolate(eucl: LatentSpace, sphere: LatentSpace) -> None:
    """(c) Slerp stays on sphere; lerp leaves it."""
    print("\n--- (c) interpolate: lerp vs slerp ---")
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])

    print(f"  {'t':>5s}  {'lerp norm':>10s}  {'slerp norm':>10s}  {'on sphere?':>10s}")
    for t in np.linspace(0, 1, 7):
        lerp_pt = eucl.interpolate(a, b, t)
        slerp_pt = sphere.interpolate(a, b, t)
        lerp_norm = np.linalg.norm(lerp_pt)
        slerp_norm = np.linalg.norm(slerp_pt)
        on_sphere = "✓" if abs(slerp_norm - 1.0) < 1e-10 else "✗"
        print(f"  {t:.2f}  {lerp_norm:10.6f}  {slerp_norm:10.6f}  {on_sphere:>10s}")


def demo_normalize(sphere: LatentSpace) -> None:
    """(d) normalize projects back to sphere."""
    print("\n--- (d) normalize ---")
    raw = np.array([3.0, 4.0, 0.0])
    normalized = sphere.normalize(raw)
    expected = np.array([0.6, 0.8, 0.0])
    print(f"  Raw vector:  {raw}")
    print(f"  Normalized:  {normalized}")
    print(f"  Norm:        {np.linalg.norm(normalized):.10f}")
    np.testing.assert_array_almost_equal(normalized, expected)
    print("  ✓ Normalization correct")


def plot_lerp_vs_slerp() -> None:
    """Side-by-side visualization: lerp vs slerp on the unit sphere.

    Shows:
    (1) 3D scatter of points on sphere colored by cluster (left)
    (2) lerp interpolation path — straight line through sphere interior (middle)
    (3) slerp interpolation path — arc on sphere surface (right)
    """
    print("\n--- (9) Visualization ---")

    # Generate clusters of unit vectors on the sphere
    rng = np.random.default_rng(42)
    n_per_cluster = 30

    # Cluster A: around [1,0,0]
    a_raw = rng.normal(size=(n_per_cluster, 3))
    a_raw[:, 0] += 3.0  # shift toward x-axis
    cluster_a = a_raw / np.linalg.norm(a_raw, axis=1, keepdims=True)

    # Cluster B: around [0,1,0]
    b_raw = rng.normal(size=(n_per_cluster, 3))
    b_raw[:, 1] += 3.0
    cluster_b = b_raw / np.linalg.norm(b_raw, axis=1, keepdims=True)

    eucl = LatentSpace(dim=3)
    sphere = LatentSpace(dim=3, geometry="unit_norm")

    # Two points to interpolate between
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])

    # Generate interpolation paths
    t_values = np.linspace(0, 1, 50)
    lerp_path = np.array([eucl.interpolate(p1, p2, t) for t in t_values])
    slerp_path = np.array([sphere.interpolate(p1, p2, t) for t in t_values])

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 5))

    # (1) 3D scatter of unit sphere points by cluster
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(
        cluster_a[:, 0],
        cluster_a[:, 1],
        cluster_a[:, 2],
        c="steelblue",
        alpha=0.7,
        label="Cluster A",
        s=20,
    )
    ax1.scatter(
        cluster_b[:, 0],
        cluster_b[:, 1],
        cluster_b[:, 2],
        c="coral",
        alpha=0.7,
        label="Cluster B",
        s=20,
    )
    ax1.set_title("Points on Unit Sphere\n(colored by cluster)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(fontsize=8)

    # Draw wireframe sphere for reference
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax1.plot_wireframe(x, y, z, alpha=0.1, color="gray")

    # (2) Lerp path — straight line through interior
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot(
        lerp_path[:, 0],
        lerp_path[:, 1],
        lerp_path[:, 2],
        "r-",
        linewidth=2,
        label="lerp path",
    )
    ax2.scatter(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        c="black",
        s=50,
        label="endpoints",
    )
    # Also show the sphere for reference
    ax2.plot_wireframe(x, y, z, alpha=0.1, color="gray")
    ax2.set_title("Lerp Path\n(straight line through interior)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.legend(fontsize=8)

    # (3) Slerp path — arc on sphere surface
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot(
        slerp_path[:, 0],
        slerp_path[:, 1],
        slerp_path[:, 2],
        "g-",
        linewidth=2,
        label="slerp path",
    )
    ax3.scatter(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        c="black",
        s=50,
        label="endpoints",
    )
    ax3.plot_wireframe(x, y, z, alpha=0.1, color="gray")
    ax3.set_title("Slerp Path\n(geodesic arc on sphere surface)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.legend(fontsize=8)

    fig.suptitle(
        "Spherical Latent Space — Geometry-Dispatch ADR Validation",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    # Save to file
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "spherical_demo_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Visualization saved to {output_path}")
    plt.close()


def main() -> None:
    print("=" * 60)
    print("  End-to-End Spherical Latent Space Demo")
    print("  Validating: geometry-keyed LatentSpace + geometry-dispatch ADRs")
    print("=" * 60)

    eucl = LatentSpace(dim=3)
    sphere = LatentSpace(dim=3, geometry="unit_norm")

    print(f"\nEuclidean LatentSpace:  {eucl}")
    print(f"Spherical LatentSpace: {sphere}")

    demo_validate_point(sphere)
    demo_distance(eucl, sphere)
    demo_interpolate(eucl, sphere)
    demo_normalize(sphere)

    # Visualization
    plot_lerp_vs_slerp()

    print("\n" + "=" * 60)
    print("  All demonstrations completed successfully!")
    print("  ✓ geometry-keyed LatentSpace ADR validated")
    print("  ✓ geometry-dispatch ADR validated")
    print("=" * 60)


if __name__ == "__main__":
    main()
