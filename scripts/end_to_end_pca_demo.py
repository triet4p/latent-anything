#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "latent-anything",
#     "numpy>=2.0,<3.0",
#     "scikit-learn>=1.6,<2.0",
#     "matplotlib>=3.9,<4.0",
# ]
# ///
"""End-to-end demo: synthetic latent → Trajectory → PCA fit → 2D projection → visualize.

Usage:
    uv run scripts/end_to_end_pca_demo.py

This script demonstrates the full Round-1 pipeline:
    1. Generate synthetic 8D latent points with cluster structure.
    2. Pack them into a Trajectory (immutable sequence).
    3. Fit PCA and project to 2D.
    4. Visualize the 2D projection with matplotlib.
"""

from __future__ import annotations

import numpy as np

from latent_anything import LatentSpace, Trajectory
from latent_anything.methods import PCA

# ---------------------------------------------------------------------------
# 1. Generate synthetic 8D latent data with 3 cluster-like groups
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_points = 300
dim = 8

# Three cluster centers in 8D
centers = [
    np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
    np.array([-2.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
]
points = np.vstack([center + rng.normal(scale=0.3, size=(n_points // 3, dim)) for center in centers])

# ---------------------------------------------------------------------------
# 2. Declare a LatentSpace and wrap data into a Trajectory
# ---------------------------------------------------------------------------
space = LatentSpace(dim=dim, source_model="synthetic")
trajectory = Trajectory(data=points)

print(f"LatentSpace: {space}")
print(f"Trajectory:  {trajectory}")
print(f"Trajectory[0]: single-point {trajectory[0]}")

# ---------------------------------------------------------------------------
# 3. Fit PCA (n_components=2) and project to 2D
# ---------------------------------------------------------------------------
pca = PCA(n_components=2)
projected = pca.fit_transform(trajectory.to_numpy())

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Projected shape: {projected.shape}")

# ---------------------------------------------------------------------------
# 4. Visualize the 2D projection
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

# Colour each of the 3 groups
labels = np.repeat([0, 1, 2], n_points // 3)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    projected[:, 0],
    projected[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.7,
    edgecolors="k",
    linewidth=0.3,
)
ax.set_title("PCA 2D projection of synthetic 8D latent space")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
fig.colorbar(scatter, ax=ax, label="Cluster")
fig.tight_layout()

# Save to file (also show if interactive)
output_path = "pca_projection_demo.png"
fig.savefig(output_path, dpi=150)
print(f"Visualization saved to {output_path}")
plt.show()
