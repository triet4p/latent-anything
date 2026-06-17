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

"""End-to-end demo: VAE adapter → LatentSpace → Trajectory → PCA/UMAP → visualize.

This script demonstrates the full Round-4 pipeline — the first time ALL
Layer A primitives appear together in one end-to-end flow:

    1. Generate synthetic structured cluster data (normalised to [0, 1]).
    2. Train a VAE adapter on the data (ModelAdapter #1, mode i).
    3. Encode to latent mean → wrap as LatentSpace + Trajectory.
    4. Decode to verify reconstruction quality.
    5. Project encoded latents via PCA and UMAP (Methods #1 and #2).
    6. Visualize a 2×2 matplotlib grid: original, reconstruction, PCA, UMAP.

Usage:
    uv run scripts/end_to_end_vae_demo.py
"""

from __future__ import annotations

import numpy as np

from latent_anything import Trajectory
from latent_anything.adapters import VAE
from latent_anything.methods import PCA, UMAP

# ---------------------------------------------------------------------------
# 1. Generate synthetic 8D cluster data, scaled to [0, 1]
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_points = 300
input_dim = 8
latent_dim = 3
n_clusters = 4

# Four cluster centers in 8D
centers_raw = [
    np.array([0.9, 0.1, 0.9, 0.1, 0.5, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.5, 0.1, 0.9]),
    np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.5, 0.1]),
    np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.9, 0.9]),
]
points = np.vstack(
    [center + rng.normal(scale=0.08, size=(n_points // n_clusters, input_dim)) for center in centers_raw]
)
# Clamp to [0, 1] as required by VAE sigmoid decoder
points = np.clip(points, 0.0, 1.0)
labels = np.repeat(np.arange(n_clusters), n_points // n_clusters)

print(f"Generated {n_points} points in {input_dim}D with {n_clusters} clusters")
print(f"Data range: [{points.min():.3f}, {points.max():.3f}]")

# ---------------------------------------------------------------------------
# 2. Train VAE adapter
# ---------------------------------------------------------------------------
print(f"\nTraining VAE (input={input_dim}, latent={latent_dim}, hidden={max(latent_dim * 4, input_dim)})...")
vae = VAE(
    input_dim=input_dim,
    latent_dim=latent_dim,
    n_epochs=300,
    learning_rate=0.005,
    beta=1.0,
    random_state=42,
)
vae.fit(points)

print(f"  Initial loss: {vae.loss_history_[0]:.6f}")
print(f"  Final loss:   {vae.loss_history_[-1]:.6f}")
print(f"  Loss reduction: {vae.loss_history_[0] / vae.loss_history_[-1]:.1f}x")

# ---------------------------------------------------------------------------
# 3. Encode to latent → LatentSpace → Trajectory
# ---------------------------------------------------------------------------
encoded = vae.encode(points)
space = vae.latent_space
trajectory = Trajectory(data=encoded)

print(f"\nLatentSpace:  {space}")
print(f"Encoded shape: {encoded.shape}")
print(f"Trajectory:   {trajectory}")

# Verify latent space property
assert space.dim == latent_dim
assert space.source_model == "vae"
assert space.geometry == "euclidean"

# ---------------------------------------------------------------------------
# 4. Decode and check reconstruction quality
# ---------------------------------------------------------------------------
reconstructed = vae.decode(encoded)
mse = np.mean((points - reconstructed) ** 2)
print(f"\nReconstruction MSE: {mse:.6f}")

# ---------------------------------------------------------------------------
# 5. PCA + UMAP projection of the encoded latents
# ---------------------------------------------------------------------------
pca = PCA(n_components=2)
pca_projected = pca.fit_transform(trajectory.to_numpy())

umap = UMAP(n_components=2, random_state=42)
umap_projected = umap.fit_transform(trajectory.to_numpy())

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"PCA projected shape:  {pca_projected.shape}")
print(f"UMAP projected shape: {umap_projected.shape}")

# ---------------------------------------------------------------------------
# 6. Visualize: 2×2 grid
#      (1) Original data (PCA 2D projection for visual reference)
#      (2) VAE reconstruction (PCA 2D projection)
#      (3) PCA of encoded latents colored by class
#      (4) UMAP of encoded latents colored by class
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

# For panels 1 & 2: project original and reconstructed data via PCA
pca_orig = PCA(n_components=2)
orig_2d = pca_orig.fit_transform(points)
recon_2d = pca_orig.transform(reconstructed)

fig, axes = plt.subplots(2, 2, figsize=(12, 11))

scatter_kw = dict(c=labels, cmap="viridis", alpha=0.7, edgecolors="k", linewidth=0.3, s=35)

# Panel 1: Original data (PCA)
ax = axes[0, 0]
sc1 = ax.scatter(orig_2d[:, 0], orig_2d[:, 1], **scatter_kw)
ax.set_title("(1) Original data (PCA 2D)", fontsize=12)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_aspect("equal")

# Panel 2: VAE reconstruction (PCA)
ax = axes[0, 1]
sc2 = ax.scatter(recon_2d[:, 0], recon_2d[:, 1], **scatter_kw)
ax.set_title(f"(2) VAE reconstruction (PCA 2D)\nMSE = {mse:.4f}", fontsize=12)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_aspect("equal")

# Panel 3: PCA of encoded latents
ax = axes[1, 0]
sc3 = ax.scatter(pca_projected[:, 0], pca_projected[:, 1], **scatter_kw)
ev0 = pca.explained_variance_ratio_[0]
ev1 = pca.explained_variance_ratio_[1]
ax.set_title(f"(3) PCA of VAE latent\n(explained var: {ev0:.2f}, {ev1:.2f})", fontsize=12)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_aspect("equal")

# Panel 4: UMAP of encoded latents
ax = axes[1, 1]
sc4 = ax.scatter(umap_projected[:, 0], umap_projected[:, 1], **scatter_kw)
ax.set_title("(4) UMAP of VAE latent", fontsize=12)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_aspect("equal")

fig.colorbar(sc1, ax=axes, label="Cluster", shrink=0.8)
fig.suptitle(
    f"VAE Adapter End-to-End — input={input_dim}D, latent={latent_dim}D, {n_clusters} clusters",
    fontsize=14,
    y=1.01,
)
fig.tight_layout()

output_path = "vae_adapter_end_to_end.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nVisualization saved to {output_path}")
plt.show()

# ---------------------------------------------------------------------------
# 7. Training loss curve
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(vae.loss_history_, linewidth=0.8)
ax2.set_title("VAE training loss (MSE + KL divergence)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

loss_path = "vae_training_loss.png"
fig2.savefig(loss_path, dpi=150)
print(f"Loss curve saved to {loss_path}")
plt.show()

print("\nDone — VAE adapter successfully trained and demonstrated the full adapter→method pipeline.")
