#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///

"""End-to-end SteeringVector demo — B-Method #2, stateful steering.

Demonstrates the ``SteeringVector`` class across two scenarios:

  **Scenario A (Euclidean, simple mean-difference)**:
  Generate synthetic 8D contrast dataset — "positive" cluster centered at
  ``[+1, +1, 0, ..., 0]`` with noise σ=0.3, "negative" cluster at
  ``[-1, -1, 0, ..., 0]``. Fit ``SteeringVector()``. Apply steering at
  strengths [0, 0.5, 1.0, 2.0] to test points from both classes.
  Project to 2D via PCA. Visualise steering path with matplotlib:
  original points and steered points connected by arrows, coloured by
  class.

  **Scenario B (Spherical, geometry-aware)**:
  Generate unit-norm contrast dataset on 3-sphere — "positive" direction
  ``[1, 0, 0]`` with small angular noise, "negative" direction
  ``[-1, 0, 0]`` with noise. Fit ``SteeringVector(space=...)`` with
  ``LatentSpace(dim=3, geometry='unit_norm')``. Steer with strength=1.0,
  normalise back to sphere. Show that steered vectors stay on sphere
  (norm ≈ 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Ensure we can import from the src directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latent_anything import LatentSpace
from latent_anything.methods import SteeringVector  # noqa: F811

# ---------------------------------------------------------------------------
# Scenario A: Euclidean steering
# ---------------------------------------------------------------------------


def _make_contrast_clusters_8d(
    n_pos: int = 50,
    n_neg: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic 8D contrast clusters.

    Returns
    -------
    positives : np.ndarray  shape (n_pos, 8)
    negatives : np.ndarray  shape (n_neg, 8)
    test_pos : np.ndarray   shape (3, 8)
    test_neg : np.ndarray   shape (3, 8)
    """
    rng = np.random.default_rng(seed)
    center_pos = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    center_neg = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    positives = center_pos + rng.normal(scale=0.3, size=(n_pos, 8))
    negatives = center_neg + rng.normal(scale=0.3, size=(n_neg, 8))

    # Separate held-out test points
    test_pos = center_pos + rng.normal(scale=0.3, size=(3, 8))
    test_neg = center_neg + rng.normal(scale=0.3, size=(3, 8))

    return positives, negatives, test_pos, test_neg


def scenario_a_euclidean() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, SteeringVector]:
    """Scenario A: Euclidean steering with mean-difference direction.

    Returns
    -------
    strengths : np.ndarray
        Steering strengths tested.
    steer_results_pos : np.ndarray  shape (n_test, n_strengths, 8)
    steer_results_neg : np.ndarray  shape (n_test, n_strengths, 8)
    all_points_2d : np.ndarray  shape (total, 2) — PCA of all points
    pos_indices, neg_indices : index masks for the PCA array
    sv : SteeringVector — fitted instance
    """
    print("--- Scenario A: Euclidean Steering ---")

    positives, negatives, test_pos, test_neg = _make_contrast_clusters_8d()

    sv = SteeringVector()
    sv.fit(positives, negatives)

    learned_dir = sv.direction
    print(f"  Learned direction (first 4 dims): {learned_dir[:4]}")
    print(f"  Direction norm: {np.linalg.norm(learned_dir):.6f}  (expected ≈ 1)")

    strengths = np.array([0.0, 0.5, 1.0, 2.0])

    steer_results_pos = np.array([[sv(pt, s) for s in strengths] for pt in test_pos])
    steer_results_neg = np.array([[sv(pt, s) for s in strengths] for pt in test_neg])

    print("\n  Test positive point #0 at strength=1.0:")
    print(f"    Before: {test_pos[0]}")
    print(f"    After:  {steer_results_pos[0, 2]}")
    print(f"    Delta:  {steer_results_pos[0, 2] - test_pos[0]}")

    # Combine everything for PCA projection
    all_points = np.vstack(
        [positives, negatives]
        + [steer_results_pos[:, i, :] for i in range(len(strengths))]
        + [steer_results_neg[:, i, :] for i in range(len(strengths))]
    )
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_points)

    n_train_pos = len(positives)
    n_train_neg = len(negatives)
    n_test = 3
    offset = n_train_pos + n_train_neg

    return strengths, steer_results_pos, steer_results_neg, all_2d, offset, n_test, sv


# ---------------------------------------------------------------------------
# Scenario B: Spherical steering (geometry-aware)
# ---------------------------------------------------------------------------


def _make_spherical_contrast(
    n_pos: int = 30,
    n_neg: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate unit-norm contrast clusters on the 3-sphere.

    "Positive" direction is ``[1, 0, 0]`` with small angular noise.
    "Negative" direction is ``[-1, 0, 0]`` with small angular noise.

    Returns
    -------
    positives, negatives : np.ndarray  shape (n, 3)
    test_pos, test_neg : np.ndarray    shape (3, 3)
    All unit-norm.
    """
    rng = np.random.default_rng(seed)

    def _points_around(center: np.ndarray, n: int) -> np.ndarray:
        pts = []
        for _ in range(n):
            # Start at center, add small Gaussian noise, renormalize
            noisy = center + rng.normal(scale=0.3, size=3)
            pts.append(noisy / np.linalg.norm(noisy))
        return np.array(pts)

    pos_center = np.array([1.0, 0.0, 0.0])
    neg_center = np.array([-1.0, 0.0, 0.0])

    positives = _points_around(pos_center, n_pos)
    negatives = _points_around(neg_center, n_neg)
    test_pos = _points_around(pos_center, 3)
    test_neg = _points_around(neg_center, 3)

    return positives, negatives, test_pos, test_neg


def scenario_b_spherical() -> tuple[np.ndarray, np.ndarray, np.ndarray, SteeringVector]:
    """Scenario B: spherical, geometry-aware steering.

    Returns
    -------
    test_pos, test_neg : np.ndarray  shape (3, 3)  — test points
    before_pos, before_neg : np.ndarray  shape (3, 3) — copies before steer
    sv : SteeringVector — fitted instance with unit_norm space
    """
    print("\n--- Scenario B: Spherical Steering (geometry-aware) ---")

    positives, negatives, test_pos, test_neg = _make_spherical_contrast()

    space = LatentSpace(dim=3, geometry="unit_norm")
    sv = SteeringVector(space=space)
    sv.fit(positives, negatives)

    learned_dir = sv.direction
    print(f"  Learned direction: {learned_dir}")
    print(f"  Direction norm: {np.linalg.norm(learned_dir):.6f}")

    # Steer at strength=1.0
    steered_pos = np.array([sv(pt, 1.0) for pt in test_pos])
    steered_neg = np.array([sv(pt, 1.0) for pt in test_neg])

    print(f"\n  {'Point':>10s}  {'Before norm':>12s}  {'After norm':>12s}  {'On sphere?':>12s}")
    for i, pt in enumerate(test_pos):
        after = steered_pos[i]
        on = "✓" if abs(np.linalg.norm(after) - 1.0) < 1e-10 else "✗"
        print(f"  {'pos#' + str(i):>10s}  {np.linalg.norm(pt):12.6f}  {np.linalg.norm(after):12.6f}  {on:>12s}")
    for i, pt in enumerate(test_neg):
        after = steered_neg[i]
        on = "✓" if abs(np.linalg.norm(after) - 1.0) < 1e-10 else "✗"
        print(f"  {'neg#' + str(i):>10s}  {np.linalg.norm(pt):12.6f}  {np.linalg.norm(after):12.6f}  {on:>12s}")

    # Also show steering at multiple strengths for the first test point
    print("\n  Spherical steering at varying strengths (test pos #0):")
    for s in [0.0, 0.5, 1.0, 2.0]:
        pt = sv(test_pos[0], s)
        norm = np.linalg.norm(pt)
        ok = "✓" if abs(norm - 1.0) < 1e-10 else "✗"
        print(f"    strength={s:.1f}  norm={norm:.6f}  {ok}")

    return test_pos, test_neg, steered_pos, steered_neg, sv


# ---------------------------------------------------------------------------
# Visualization: 1×2 matplotlib
# ---------------------------------------------------------------------------


def plot_steering(
    strengths: np.ndarray,
    _steer_results_pos: np.ndarray,
    _steer_results_neg: np.ndarray,
    all_2d: np.ndarray,
    offset: int,
    n_test: int,
    test_pos_sph: np.ndarray,
    test_neg_sph: np.ndarray,
    steered_pos_sph: np.ndarray,
    steered_neg_sph: np.ndarray,
) -> None:
    """Side-by-side visualisation: Euclidean steering vs spherical steering.

    Left: Euclidean — training clusters + test points steered at multiple
    strengths, connected by arrows coloured by class.

    Right: Spherical — test points before/after steering at strength=1.0,
    connected by arrows showing movement along great circle while staying
    on sphere. 2D PCA projection with unit-circle reference.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: Euclidean steering ----
    # Training clusters
    n_train_pos = offset // 2  # approximate
    ax_left.scatter(
        all_2d[:n_train_pos, 0],
        all_2d[:n_train_pos, 1],
        c="blue",
        alpha=0.3,
        s=20,
        label="positives (train)",
    )
    ax_left.scatter(
        all_2d[n_train_pos:offset, 0],
        all_2d[n_train_pos:offset, 1],
        c="red",
        alpha=0.3,
        s=20,
        label="negatives (train)",
    )

    # Test points at different strengths
    colors_pos = plt.cm.Blues(np.linspace(0.4, 0.9, len(strengths)))
    colors_neg = plt.cm.Reds(np.linspace(0.4, 0.9, len(strengths)))

    for i in range(n_test):
        for s_idx, s in enumerate(strengths):
            idx = offset + s_idx * n_test + i
            pt_2d = all_2d[idx]
            marker = "o" if s == 0.0 else "^"
            size = 40 if s == 0.0 else 60
            ax_left.scatter(
                pt_2d[0],
                pt_2d[1],
                c=[colors_pos[s_idx]],
                marker=marker,
                s=size,
                edgecolors="blue" if s == 0.0 else "none",
                linewidths=0.5,
            )

    for i in range(n_test):
        for s_idx, s in enumerate(strengths):
            idx = offset + s_idx * n_test + i
            # Negatives start after positives for all strength levels
            neg_offset_extra = len(strengths) * n_test
            neg_idx = offset + neg_offset_extra + s_idx * n_test + i
            pt_2d = all_2d[neg_idx]
            marker = "o" if s == 0.0 else "^"
            size = 40 if s == 0.0 else 60
            ax_left.scatter(
                pt_2d[0],
                pt_2d[1],
                c=[colors_neg[s_idx]],
                marker=marker,
                s=size,
                edgecolors="red" if s == 0.0 else "none",
                linewidths=0.5,
            )

    # Draw arrows from original to steered for test pos #0
    for s_idx in range(1, len(strengths)):
        orig_idx = offset + 0  # strength=0, test_pos[0]
        steer_idx = offset + s_idx * n_test + 0
        ax_left.annotate(
            "",
            xy=(all_2d[steer_idx, 0], all_2d[steer_idx, 1]),
            xytext=(all_2d[orig_idx, 0], all_2d[orig_idx, 1]),
            arrowprops=dict(
                arrowstyle="->",
                color=colors_pos[s_idx],
                lw=1.5,
                connectionstyle="arc3,rad=0",
            ),
        )

    ax_left.set_title("Euclidean Steering (PCA 2D)")
    ax_left.set_xlabel("PC1")
    ax_left.set_ylabel("PC2")
    ax_left.legend(loc="best", fontsize=8)
    ax_left.grid(True, alpha=0.3)

    # Annotate strength values on arrows
    for s_idx, s in enumerate(strengths):
        if s == 0.0:
            continue
        orig_idx = offset + 0
        steer_idx = offset + s_idx * n_test + 0
        mid = (all_2d[orig_idx] + all_2d[steer_idx]) / 2
        ax_left.text(mid[0], mid[1], f"s={s:.1f}", fontsize=7, ha="center", va="bottom")

    # ---- Right: Spherical steering ----
    # PCA projection of spherical test points + steered results
    sph_all = np.vstack([test_pos_sph, test_neg_sph, steered_pos_sph, steered_neg_sph])
    pca_sph = PCA(n_components=2)
    sph_2d = pca_sph.fit_transform(sph_all)

    n_before = 3
    pos_before_2d = sph_2d[:n_before]
    neg_before_2d = sph_2d[n_before : 2 * n_before]
    pos_after_2d = sph_2d[2 * n_before : 3 * n_before]
    neg_after_2d = sph_2d[3 * n_before :]

    # Unit circle reference
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax_right.plot(circle_x, circle_y, "k--", alpha=0.2, label="unit circle (approx)")

    # Positive points: before (blue circles) → after (blue triangles)
    ax_right.scatter(
        pos_before_2d[:, 0],
        pos_before_2d[:, 1],
        c="blue",
        marker="o",
        s=60,
        label="positive (before)",
    )
    ax_right.scatter(
        pos_after_2d[:, 0],
        pos_after_2d[:, 1],
        c="blue",
        marker="^",
        s=60,
        label="positive (after, s=1.0)",
    )
    for i in range(n_before):
        ax_right.annotate(
            "",
            xy=(pos_after_2d[i, 0], pos_after_2d[i, 1]),
            xytext=(pos_before_2d[i, 0], pos_before_2d[i, 1]),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.2, alpha=0.6),
        )

    # Negative points: before (red circles) → after (red triangles)
    ax_right.scatter(
        neg_before_2d[:, 0],
        neg_before_2d[:, 1],
        c="red",
        marker="o",
        s=60,
        label="negative (before)",
    )
    ax_right.scatter(
        neg_after_2d[:, 0],
        neg_after_2d[:, 1],
        c="red",
        marker="^",
        s=60,
        label="negative (after, s=1.0)",
    )
    for i in range(n_before):
        ax_right.annotate(
            "",
            xy=(neg_after_2d[i, 0], neg_after_2d[i, 1]),
            xytext=(neg_before_2d[i, 0], neg_before_2d[i, 1]),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2, alpha=0.6),
        )

    ax_right.set_title("Spherical Steering (PCA 2D)")
    ax_right.set_xlabel("PC1")
    ax_right.set_ylabel("PC2")
    ax_right.legend(loc="best", fontsize=8)
    ax_right.grid(True, alpha=0.3)
    ax_right.set_aspect("equal")

    plt.suptitle("SteeringVector — B-Method #2 (stateful, fit from contrast)", fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent.parent / "artifacts" / "steering_demo_plot.png", dpi=150)
    plt.show()
    print("\n✓ Plot saved to artifacts/steering_demo_plot.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    strengths, steer_results_pos, steer_results_neg, all_2d, offset, n_test, sv_euc = scenario_a_euclidean()
    test_pos_sph, test_neg_sph, steered_pos_sph, steered_neg_sph, sv_sph = scenario_b_spherical()

    # Verify geometry-awareness
    for pt in steered_pos_sph:
        assert abs(np.linalg.norm(pt) - 1.0) < 1e-10, "Spherical steering broke unit norm!"
    for pt in steered_neg_sph:
        assert abs(np.linalg.norm(pt) - 1.0) < 1e-10, "Spherical steering broke unit norm!"
    print("\n  ✓ All spherical steered points remain on unit sphere")

    plot_steering(
        strengths,
        steer_results_pos,
        steer_results_neg,
        all_2d,
        offset,
        n_test,
        test_pos_sph,
        test_neg_sph,
        steered_pos_sph,
        steered_neg_sph,
    )


if __name__ == "__main__":
    main()
