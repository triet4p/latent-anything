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

"""End-to-end demo: SAE sparse coding of synthetic latent space.

Compares three dimensionality-reduction methods side-by-side:
1. PCA — linear, matrix decomposition (Method #1)
2. UMAP — nonlinear, manifold-learning (Method #2)
3. SAE  — neural, trained with L1 sparsity (Method #3)

Usage:
    uv run scripts/end_to_end_sae_demo.py
"""

from __future__ import annotations

import numpy as np

from latent_anything import LatentSpace, Method, Trajectory
from latent_anything.methods import PCA, SAE, UMAP

# ---------------------------------------------------------------------------
# 1. Generate synthetic 8D latent data with 4 cluster-like groups
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_points = 400
dim = 8

# Four cluster centers in 8D — more structure for SAE to learn
centers = [
    np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
    np.array([-3.0, 0.0, -3.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0]),
]
points = np.vstack([center + rng.normal(scale=0.3, size=(n_points // 4, dim)) for center in centers])

# ---------------------------------------------------------------------------
# 2. Declare a LatentSpace and wrap data into a Trajectory
# ---------------------------------------------------------------------------
space = LatentSpace(dim=dim, source_model="synthetic")
trajectory = Trajectory(data=points)

print(f"LatentSpace: {space}")
print(f"Trajectory:  {trajectory}")
print(f"Data shape:  {trajectory.to_numpy().shape}")

# ---------------------------------------------------------------------------
# 3. Fit all three methods and project to 2D
# ---------------------------------------------------------------------------

# PCA
pca = PCA(n_components=2)
pca_projected = pca.fit_transform(trajectory.to_numpy())
print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")

# UMAP
umap = UMAP(n_components=2, random_state=42)
umap_projected = umap.fit_transform(trajectory.to_numpy())

# SAE (train on the same data)
sae = SAE(n_components=2, l1_coef=0.01, learning_rate=0.01, n_epochs=500, random_state=42)
sae.fit(trajectory.to_numpy())
sae_projected = sae.transform(trajectory.to_numpy())

print(f"SAE final loss: {sae.loss_history_[-1]:.6f}  (initial: {sae.loss_history_[0]:.6f})")
print(f"  L1 coef: {sae.l1_coef} — latent activations encouraged to be sparse")

# Sanity check: all produce (n_points, 2) output
assert pca_projected.shape == (n_points, 2)
assert umap_projected.shape == (n_points, 2)
assert sae_projected.shape == (n_points, 2)

# ---------------------------------------------------------------------------
# 4. Protocol conformance checks (structural duck-typing)
# ---------------------------------------------------------------------------
assert isinstance(pca, Method), "PCA must conform to Method Protocol"
assert isinstance(umap, Method), "UMAP must conform to Method Protocol"
assert isinstance(sae, Method), "SAE must conform to Method Protocol"
print("\n✓ All three methods conform to the Method Protocol (structural duck-typing)")

# ---------------------------------------------------------------------------
# 5. Visualize 3 projections side-by-side
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

labels = np.repeat([0, 1, 2, 3], n_points // 4)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

projections = [
    ("PCA 2D projection", pca_projected, "PC1", "PC2"),
    ("UMAP 2D embedding", umap_projected, "UMAP-1", "UMAP-2"),
    ("SAE sparse latent (L1)", sae_projected, "Latent-1", "Latent-2"),
]

for ax, (title, proj, xlabel, ylabel) in zip(axes, projections, strict=True):
    scatter = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
        edgecolors="k",
        linewidth=0.3,
        s=30,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

fig.colorbar(scatter, ax=axes, label="Cluster", shrink=0.8)
fig.suptitle(
    "Method comparison on synthetic 8D latent space — PCA (linear) vs UMAP (nonlinear) vs SAE (sparse neural)",
    fontsize=14,
    y=1.02,
)
fig.tight_layout()

output_path = "sae_method_comparison.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nVisualization saved to {output_path}")
plt.show()

# ---------------------------------------------------------------------------
# 6. Train loss curve (SAE convergence diagnostic)
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(sae.loss_history_, linewidth=0.8)
ax2.set_title("SAE training loss (MSE + L1 sparsity)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

loss_path = "sae_loss_curve.png"
fig2.savefig(loss_path, dpi=150)
print(f"Loss curve saved to {loss_path}")
plt.show()

print("\nDone — SAE successfully learned a sparse decomposition of synthetic latent space.")
