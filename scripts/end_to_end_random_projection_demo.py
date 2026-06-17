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

"""End-to-end demo: RandomProjection adapter → LatentSpace → Trajectory → PCA/UMAP → visualize.

This script demonstrates the Round-5 pipeline — RandomProjection as
ModelAdapter #2 (stateless/fixed-weight, pretrained pattern):

    1. Generate synthetic cluster data in high-dimensional space.
    2. Create a RandomProjection adapter (no training — weights are fixed).
    3. Encode to latent → wrap as LatentSpace + Trajectory.
    4. Decode to verify approximate reconstruction.
    5. Project original data via PCA, and projected latents via PCA + UMAP.
    6. Visualize a 1×3 matplotlib grid:
       (1) PCA of original data
       (2) PCA of random-projected latents
       (3) UMAP of random-projected latents

    Demonstrates that random projection approximately preserves cluster
    structure (Johnson-Lindenstrauss lemma in action).

Usage:
    uv run scripts/end_to_end_random_projection_demo.py
"""

from __future__ import annotations

import numpy as np

from latent_anything import Trajectory
from latent_anything.adapters import RandomProjection
from latent_anything.methods import PCA, UMAP

# ---------------------------------------------------------------------------
# 1. Generate synthetic cluster data in 50D
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_points = 400
input_dim = 50
latent_dim = 10
n_clusters = 5

# Five cluster centers in 50D — random but fixed for reproducibility
centers_raw = rng.random((n_clusters, input_dim)) * 2.0 - 1.0  # range [-1, 1]

points = np.vstack([center + rng.normal(scale=0.2, size=(n_points // n_clusters, input_dim)) for center in centers_raw])
labels = np.repeat(np.arange(n_clusters), n_points // n_clusters)

print(f"Generated {n_points} points in {input_dim}D with {n_clusters} clusters")
print(f"Data range: [{points.min():.3f}, {points.max():.3f}]")

# ---------------------------------------------------------------------------
# 2. Create RandomProjection adapter (no fit — weights fixed at construction)
# ---------------------------------------------------------------------------
print(f"\nCreating RandomProjection (input={input_dim}, latent={latent_dim}, random_state=42)...")
rp = RandomProjection(input_dim=input_dim, latent_dim=latent_dim, random_state=42)

print(f"  Projection matrix shape: {rp.projection_matrix_.shape}")
print(f"  Projection matrix range: [{rp.projection_matrix_.min():.4f}, {rp.projection_matrix_.max():.4f}]")

# ---------------------------------------------------------------------------
# 3. Encode to latent → LatentSpace → Trajectory
# ---------------------------------------------------------------------------
encoded = rp.encode(points)
space = rp.latent_space
trajectory = Trajectory(data=encoded)

print(f"\nLatentSpace:  {space}")
print(f"Encoded shape: {encoded.shape}")
print(f"Trajectory:   {trajectory}")

# Verify latent space property
assert space.dim == latent_dim
assert space.source_model == "random_projection"
assert space.geometry == "euclidean"

# ---------------------------------------------------------------------------
# 4. Decode and check reconstruction quality (approximate)
# ---------------------------------------------------------------------------
reconstructed = rp.decode(encoded)
mse = np.mean((points - reconstructed) ** 2)
print(f"\nReconstruction MSE (transpose approx): {mse:.6f}")

# ---------------------------------------------------------------------------
# 5. PCA + UMAP projection
# ---------------------------------------------------------------------------
pca_orig = PCA(n_components=2)
orig_2d = pca_orig.fit_transform(points)

pca_latent = PCA(n_components=2)
latent_pca_2d = pca_latent.fit_transform(trajectory.to_numpy())

umap_latent = UMAP(n_components=2, random_state=42)
latent_umap_2d = umap_latent.fit_transform(trajectory.to_numpy())

print(f"\nOriginal data PCA explained variance: {pca_orig.explained_variance_ratio_}")
print(f"Latent PCA explained variance:        {pca_latent.explained_variance_ratio_}")
print(f"Latent PCA projected shape:  {latent_pca_2d.shape}")
print(f"Latent UMAP projected shape: {latent_umap_2d.shape}")

# ---------------------------------------------------------------------------
# 6. Visualize: 1×3 grid
#      (1) PCA of original data
#      (2) PCA of random-projected latents
#      (3) UMAP of random-projected latents
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

scatter_kw = dict(c=labels, cmap="tab10", alpha=0.7, edgecolors="k", linewidth=0.3, s=40)

# Panel 1: PCA of original data
ax = axes[0]
sc1 = ax.scatter(orig_2d[:, 0], orig_2d[:, 1], **scatter_kw)
ev0_orig = pca_orig.explained_variance_ratio_[0]
ev1_orig = pca_orig.explained_variance_ratio_[1]
ax.set_title(f"(1) PCA of original {input_dim}D data\n(var: {ev0_orig:.2f}, {ev1_orig:.2f})", fontsize=12)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_aspect("equal")

# Panel 2: PCA of random-projected latents
ax = axes[1]
sc2 = ax.scatter(latent_pca_2d[:, 0], latent_pca_2d[:, 1], **scatter_kw)
ev0_lat = pca_latent.explained_variance_ratio_[0]
ev1_lat = pca_latent.explained_variance_ratio_[1]
ax.set_title(f"(2) PCA of random-projected latents\n(var: {ev0_lat:.2f}, {ev1_lat:.2f})", fontsize=12)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_aspect("equal")

# Panel 3: UMAP of random-projected latents
ax = axes[2]
sc3 = ax.scatter(latent_umap_2d[:, 0], latent_umap_2d[:, 1], **scatter_kw)
ax.set_title("(3) UMAP of random-projected latents", fontsize=12)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_aspect("equal")

fig.colorbar(sc1, ax=axes, label="Cluster", shrink=0.8, aspect=40)
fig.suptitle(
    f"RandomProjection Adapter — {input_dim}D → {latent_dim}D (JL-style), {n_clusters} clusters — MSE = {mse:.4f}",
    fontsize=14,
    y=1.02,
)
fig.tight_layout()

output_path = "random_projection_end_to_end.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nVisualization saved to {output_path}")
plt.show()

# ---------------------------------------------------------------------------
# 7. Distance preservation check (JL lemma sanity)
# ---------------------------------------------------------------------------
# Compute pairwise distances before and after projection
from scipy.spatial.distance import correlation, pdist  # noqa: E402

orig_dists = pdist(points, metric="euclidean")
proj_dists = pdist(encoded, metric="euclidean")

# Correlation between original and projected distances
corr = correlation(orig_dists, proj_dists)
ratio_mean = float(np.mean(proj_dists / orig_dists))
ratio_std = float(np.std(proj_dists / orig_dists))

print("\nDistance preservation (JL check):")
print(f"  Original vs projected distance correlation: {1.0 - corr:.4f}")
print(f"  Distance ratio (proj/orig): mean={ratio_mean:.4f}, std={ratio_std:.4f}")

# Projection should preserve distance ratios approximately
# (for a random projection, mean ratio should be ~1.0)
assert abs(ratio_mean - 1.0) < 0.3, f"Distance ratio too far from 1.0: {ratio_mean:.4f}"

print("\nDone — RandomProjection adapter successfully demonstrated the adapter→method pipeline.")
