#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "latent-anything",
#     "numpy>=2.0,<3.0",
#     "scikit-learn>=1.6,<2.0",
#     "matplotlib>=3.9,<4.0",
# ]
# """
"""End-to-end demo: AnalysisPipeline #1 with VAE → encode → PCA → result.

Usage:
    uv run scripts/end_to_end_pipeline_demo.py

This script reproduces the analysis portion of the Sprint 13 showcase
(VAE → encode → PCA → project) through Pipeline #1's unified interface.
It demonstrates:

1. Direct construction: ``AnalysisPipeline(adapter=..., method=...)``
2. Config-backed construction: ``build_pipeline_from_config(PipelineSpec(...))``
3. Running the pipeline and inspecting the typed result.
4. Visualising the 2D PCA projection from the pipeline result.
"""

from __future__ import annotations

import numpy as np

from latent_anything.adapters import VAE
from latent_anything.config import ObjectSpec
from latent_anything.methods import PCA
from latent_anything.pipeline import AnalysisPipeline, PipelineSpec, build_pipeline_from_config

# ---------------------------------------------------------------------------
# 1. Generate synthetic 8D cluster data (same pattern as Sprint 13 showcase)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_points = 300
input_dim = 8
latent_dim = 3

# Three cluster centres in 8D
centers = [
    np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
    np.array([-2.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
]
points = np.vstack([center + rng.normal(scale=0.3, size=(n_points // 3, input_dim)) for center in centers])
# Scale to [0, 1] for VAE
points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0) + 1e-10)

labels = np.repeat([0, 1, 2], n_points // 3)

print(f"Data shape: {points.shape}")
print(f"Labels:     {np.bincount(labels)}")

# ---------------------------------------------------------------------------
# 2. Method A: Direct construction
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Method A: Direct construction")
print("=" * 60)

vae = VAE(input_dim=input_dim, latent_dim=latent_dim, n_epochs=50, beta=0.5, random_state=42)
pca = PCA(n_components=2)
pipeline_direct = AnalysisPipeline(adapter=vae, method=pca)

print(f"  Adapter: VAE(input_dim={input_dim}, latent_dim={latent_dim})")
print("  Method:  PCA(n_components=2)")
print(f"  Latent space: {pipeline_direct.latent_space}")

# Train VAE (separate from pipeline.run — VAE needs fit called before encode)
vae.fit(points)
result_direct = pipeline_direct.run(points)

print(f"\n  Result type: {type(result_direct).__name__}")
print(f"  Latents shape:     {result_direct.latents.shape}")
print(f"  Transformed shape: {result_direct.transformed.shape}")
print(f"  PCA explained variance: {pca.explained_variance_ratio_}")

# ---------------------------------------------------------------------------
# 3. Method B: Config-backed construction (via PipelineSpec)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Method B: Config-backed construction (PipelineSpec)")
print("=" * 60)

spec = PipelineSpec(
    adapter=ObjectSpec(
        kind="adapter",
        name="vae",
        params={"input_dim": input_dim, "latent_dim": latent_dim, "n_epochs": 50, "beta": 0.5, "random_state": 42},
    ),
    method=ObjectSpec(kind="method_a", name="pca", params={"n_components": 2}),
)
pipeline_config = build_pipeline_from_config(spec)

print(f"  Spec adapter: {spec.adapter.kind}/{spec.adapter.name}")
print(f"  Spec method:  {spec.method.kind}/{spec.method.name}")

# Train VAE (separately for the config-built pipeline too)
config_adapter = pipeline_config.adapter
if not isinstance(config_adapter, VAE):
    msg = f"Expected config-built VAE, got {type(config_adapter).__name__}"
    raise TypeError(msg)
config_adapter.fit(points)
result_config = pipeline_config.run(points)

print(f"\n  Result type: {type(result_config).__name__}")
print(f"  Latents shape:     {result_config.latents.shape}")
print(f"  Transformed shape: {result_config.transformed.shape}")

# Verify results match across construction methods
diff = np.abs(result_direct.transformed - result_config.transformed).max()
print(f"\n  Max diff between direct and config: {diff:.2e}")
print(f"  Results match: {diff < 1e-10}")

# ---------------------------------------------------------------------------
# 4. Visualise the 2D PCA projection from the pipeline result
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Direct-construction result
scatter0 = axes[0].scatter(
    result_direct.transformed[:, 0],
    result_direct.transformed[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.7,
    edgecolors="k",
    linewidth=0.3,
)
axes[0].set_title("Pipeline #1 (direct): VAE encode → PCA 2D")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
fig.colorbar(scatter0, ax=axes[0], label="Cluster")  # pyright: ignore[reportUnknownMemberType]

# Config-construction result
scatter1 = axes[1].scatter(
    result_config.transformed[:, 0],
    result_config.transformed[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.7,
    edgecolors="k",
    linewidth=0.3,
)
axes[1].set_title("Pipeline #1 (config): build_from_config → run")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
fig.colorbar(scatter1, ax=axes[1], label="Cluster")  # pyright: ignore[reportUnknownMemberType]

fig.suptitle(  # pyright: ignore[reportUnknownMemberType]
    "Sprint 20 — AnalysisPipeline #1 Demonstration", fontsize=14
)
fig.tight_layout()

# Save to file
output_path = "pipeline_demo.png"
fig.savefig(output_path, dpi=150)  # pyright: ignore[reportUnknownMemberType]
print(f"\nVisualization saved to {output_path}")
plt.show()  # pyright: ignore[reportUnknownMemberType]
