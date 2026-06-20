#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0,<3.0",
#     "scikit-learn>=1.6,<2.0",
#     "umap-learn>=0.5,<1.0",
#     "matplotlib>=3.9,<4.0",
# ]
# ///

"""End-to-end demo: HiddenStateAdapter → encode → PCA/UMAP visualization.

This script demonstrates ModelAdapter #3 (mode ii: no-explicit-latent) —
the ``HiddenStateAdapter``. Unlike VAE (mode i: explicit learned latent
with decoder) and RandomProjection (mode i-like: fixed projection with
pseudo-inverse decode), the HiddenStateAdapter has **no decoder**: the
hidden-state activations ARE the latent representation.

Pipeline:
    1. Generate synthetic structured cluster data.
    2. Create a HiddenStateAdapter with fixed random weights.
    3. Encode data → hidden activations (n_samples, hidden_dim).
    4. Project encoded latents via PCA and UMAP (Methods #1 and #2).
    5. Visualize a 1×2 matplotlib grid: PCA 2D + UMAP 2D.
    6. No decode — this adapter cannot reconstruct input from latent.

Usage:
    uv run scripts/end_to_end_hidden_state_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from latent_anything.adapters import HiddenStateAdapter
from latent_anything.methods import PCA, UMAP

# ---------------------------------------------------------------------------
# 1. Generate synthetic 8D cluster data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_points = 300
input_dim = 8
hidden_dim = 5
n_clusters = 4

# Four cluster centers in 8D
centers_raw = [
    np.array([0.9, 0.1, 0.9, 0.1, 0.5, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.5, 0.1, 0.9]),
    np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.5, 0.1]),
    np.array([0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]),
]

data_list: list[np.ndarray] = []
label_list: list[int] = []
for i, center in enumerate(centers_raw):
    cluster = rng.normal(loc=center, scale=0.15, size=(n_points // n_clusters, input_dim))
    data_list.append(cluster)
    label_list.extend([i] * (n_points // n_clusters))

data = np.vstack(data_list)
labels = np.array(label_list)

print(f"Data shape: {data.shape}")
print(f"Clusters: {n_clusters}")

# ---------------------------------------------------------------------------
# 2. Create HiddenStateAdapter and encode
# ---------------------------------------------------------------------------
adapter = HiddenStateAdapter(input_dim=input_dim, hidden_dim=hidden_dim, random_state=42)

print(f"\nAdapter: HiddenStateAdapter(input_dim={input_dim}, hidden_dim={hidden_dim})")
print(f"  latent_space: {adapter.latent_space}")
print(f"  latent_space.metadata: {adapter.latent_space.metadata}")

hidden = adapter.encode(data)
print(f"\nHidden activations shape: {hidden.shape}")
print(f"  min={hidden.min():.4f}, max={hidden.max():.4f}, mean={hidden.mean():.4f}")
print(f"  % zero (ReLU dead): {(hidden == 0).mean() * 100:.1f}%")

# Verify ModelAdapter protocol conformance
from latent_anything.adapters import DecodableAdapter, ModelAdapter  # noqa: E402

print(f"\n  Conforms to ModelAdapter: {isinstance(adapter, ModelAdapter)}")
print(f"  Conforms to DecodableAdapter: {isinstance(adapter, DecodableAdapter)}")
print(f"  Has decode: {hasattr(adapter, 'decode')}")

# ---------------------------------------------------------------------------
# 3. PCA and UMAP projection
# ---------------------------------------------------------------------------
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(hidden)

umap_model = UMAP(n_components=2, random_state=42)
umap_2d = umap_model.fit_transform(hidden)

explained_var = pca.explained_variance_ratio_  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]
print(f"\nPCA explained variance ratio: {explained_var}")

# ---------------------------------------------------------------------------
# 4. Visualize 1×2 grid
# ---------------------------------------------------------------------------
colors = ["#4285F4", "#EA4335", "#34A853", "#FBBC04"]
_, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, proj, name in zip(axes, [pca_2d, umap_2d], ["PCA", "UMAP"]):
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            c=colors[cluster_id],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
            s=10,
        )
    ax.set_title(f"Hidden activations — {name} 2D")
    ax.set_xlabel(f"{name} 1")
    ax.set_ylabel(f"{name} 2")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

plt.suptitle(
    f"HiddenStateAdapter (input_dim={input_dim}, hidden_dim={hidden_dim})\nMode (ii): no-explicit-latent — no decoder",
    fontsize=11,
)
plt.tight_layout()

# Save
output_dir = Path(__file__).resolve().parent.parent / "artifacts"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "hidden_state_demo_plot.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n✓ Plot saved to {output_path}")

plt.show()
