#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "latent-anything",
#     "numpy>=2.0,<3.0",
#     "scikit-learn>=1.6,<2.0",
#     "umap-learn>=0.5,<1.0",
#     "matplotlib>=3.9,<4.0",
# ]
# ///
"""End-to-end demo: synthetic latent → Trajectory → PCA / UMAP → 2D projection → visualize.

Usage:
    uv run scripts/end_to_end_umap_demo.py

This script demonstrates the Round-2 pipeline:
    1. Generate synthetic 8D latent points with cluster structure.
    2. Pack them into a Trajectory (immutable sequence).
    3. Fit PCA and UMAP, project to 2D.
    4. Visualise the 2D projections side-by-side with matplotlib.
"""

from __future__ import annotations

import numpy as np

from latent_anything import LatentSpace, Trajectory
from latent_anything.methods import PCA, UMAP

# ---------------------------------------------------------------------------
# 1. Generate synthetic 8D latent data with 3 cluster-like groups
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_points = 300
dim = 8

# Three cluster centres in 8D
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

# ---------------------------------------------------------------------------
# 3. Fit PCA (n_components=2) and project to 2D
# ---------------------------------------------------------------------------
pca = PCA(n_components=2)
pca_projected = pca.fit_transform(trajectory.to_numpy())

print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"PCA projected shape:          {pca_projected.shape}")

# ---------------------------------------------------------------------------
# 4. Fit UMAP (n_components=2) and project to 2D
# ---------------------------------------------------------------------------
umap = UMAP(n_components=2, random_state=42)
umap_projected = umap.fit_transform(trajectory.to_numpy())

print(f"UMAP projected shape:         {umap_projected.shape}")

# ---------------------------------------------------------------------------
# 5. Visualise side-by-side
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

labels = np.repeat([0, 1, 2], n_points // 3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# PCA panel
sc1 = ax1.scatter(
    pca_projected[:, 0],
    pca_projected[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.7,
    edgecolors="k",
    linewidth=0.3,
)
ax1.set_title("PCA 2D projection")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
fig.colorbar(sc1, ax=ax1, label="Cluster")

# UMAP panel
sc2 = ax2.scatter(
    umap_projected[:, 0],
    umap_projected[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.7,
    edgecolors="k",
    linewidth=0.3,
)
ax2.set_title("UMAP 2D projection")
ax2.set_xlabel("UMAP-1")
ax2.set_ylabel("UMAP-2")
fig.colorbar(sc2, ax=ax2, label="Cluster")

fig.suptitle("Dimensionality reduction comparison: PCA vs UMAP on synthetic 8D latent space")
fig.tight_layout()

# Save to file
output_path = "umap_vs_pca_demo.png"
fig.savefig(output_path, dpi=150)
print(f"\nVisualisation saved to {output_path}")
plt.show()
