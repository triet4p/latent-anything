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

"""End-to-end demo: ManipulationPipeline #2 — two stories on the Sprint 13 path.

Usage:
    uv run scripts/end_to_end_manipulation_demo.py

Demonstrates Pipeline #2 (``ManipulationPipeline``) through two stories
that reproduce and extend the Sprint 13 showcase path:

1. **Adapter-mediated story (data-space output)**:
   VAE → ActivationPatch → fit patch → apply to held-out data → return
   decoded data-space arrays. Metric-ready for before/after comparison.

2. **Latent-only story (trajectory output)**:
   SteeringVector → fit from contrast pairs → steer a trajectory of latent
   points → return a new ``Trajectory``.
"""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

# Ensure we can import from the src directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latent_anything import Trajectory
from latent_anything.adapters import VAE
from latent_anything.methods import ActivationPatch, Lerp, SteeringVector
from latent_anything.pipeline import ManipulationPipeline

# ---------------------------------------------------------------------------
# 1. Generate synthetic cluster data (same pattern as Sprint 13 showcase)
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

# Split into train/held-out
train_data = points[:200]
held_out = points[200:]
held_labels = labels[200:]

# ---------------------------------------------------------------------------
# 2. Train VAE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training VAE")
print("=" * 60)
vae = VAE(input_dim=input_dim, latent_dim=latent_dim, n_epochs=50, beta=0.5, random_state=42)
vae.fit(train_data)
train_latents = vae.encode(train_data)
print(f"  Train latents shape: {train_latents.shape}")

# ---------------------------------------------------------------------------
# 3. Story A: Adapter-mediated (ActivationPatch → data-space output)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Story A: Adapter-mediated — ActivationPatch through Pipeline #2")
print("=" * 60)

# Create source (cluster 0) and target (cluster 1) groups in data space
source_mask = held_labels == 0
target_mask = held_labels == 1
source_data = held_out[source_mask][:15]  # 15 samples from cluster 0
target_data = held_out[target_mask][:15]  # 15 samples from cluster 1

# Build pipeline with ActivationPatch
patch = ActivationPatch(adapter=vae)
pipeline_patch = ManipulationPipeline(method=patch, adapter=vae)

# Fit patch: source → target direction
pipeline_patch.fit(source_data, target_data)
print(f"  Patch delta norm: {np.linalg.norm(patch.delta):.4f}")

# Apply to held-out source samples → data-space output
held_source = held_out[source_mask][15:25]  # 10 new source samples
patched_output = pipeline_patch.run_data(held_source)

# Compare original vs patched (before/after metric)
original_recon = vae.decode(vae.encode(held_source))
reconstruction_error: np.ndarray = original_recon - held_source
patch_error: np.ndarray = patched_output - held_source
before_rmse = sqrt(cast(float, np.mean(reconstruction_error**2)))
after_rmse = sqrt(cast(float, np.mean(patch_error**2)))
print(f"  Before (recon) RMSE: {before_rmse:.4f}")
print(f"  After  (patched) RMSE: {after_rmse:.4f}")
print(f"  Change: {(after_rmse - before_rmse) / before_rmse * 100:+.1f}%")

# Also show distance to target cluster
target_centroid = train_data[labels[:200] == 1].mean(axis=0)
original_distances: np.ndarray = np.linalg.norm(original_recon - target_centroid, axis=1)
patch_distances: np.ndarray = np.linalg.norm(patched_output - target_centroid, axis=1)
orig_dist = cast(float, np.mean(original_distances))
patch_dist = cast(float, np.mean(patch_distances))
print(f"  Mean dist to target centroid (original): {orig_dist:.4f}")
print(f"  Mean dist to target centroid (patched):  {patch_dist:.4f}")

# ---------------------------------------------------------------------------
# 4. Story B: Latent-only (SteeringVector → trajectory output)
# ---------------------------------------------------------------------------
print("\n" + "\n" + "=" * 60)
print("Story B: Latent-only — SteeringVector through Pipeline #2")
print("=" * 60)

# Fit SteeringVector from contrast latent pairs
pos_latents = train_latents[labels[:200] == 1]  # cluster 1 — desired
neg_latents = train_latents[labels[:200] == 0]  # cluster 0 — undesired

steer = SteeringVector()
pipeline_steer = ManipulationPipeline(method=steer)

pipeline_steer.fit(pos_latents, neg_latents)
print(f"  Steering direction norm: {np.linalg.norm(steer.direction):.4f}")
print(f"  Steering direction dim: {steer.direction.shape[0]}")

# Create a trajectory from cluster 0 to cluster 2 (via Lerp blending)
cluster0_point = train_latents[labels[:200] == 0][:1]
cluster2_point = train_latents[labels[:200] == 2][:1]
lerp = Lerp()
traj_in = lerp.blend_sequence(
    Trajectory(data=np.vstack([cluster0_point, cluster2_point])),
    n_steps=5,
)
print(f"  Input trajectory: n_points={len(traj_in)}, dim={traj_in.dim}")

# Steer the trajectory
traj_out = pipeline_steer.run_trajectory(traj_in, strength=1.5)
if not isinstance(traj_out, Trajectory):
    msg = f"SteeringVector must return Trajectory, got {type(traj_out).__name__}"
    raise TypeError(msg)
print(f"  Output trajectory: n_points={len(traj_out)}, dim={traj_out.dim}")

# Steer at multiple strengths for comparison
traj_mild = pipeline_steer.run_trajectory(traj_in, strength=0.5)
traj_strong = pipeline_steer.run_trajectory(traj_in, strength=2.0)
if not isinstance(traj_mild, Trajectory) or not isinstance(traj_strong, Trajectory):
    msg = "SteeringVector trajectory runs must return Trajectory"
    raise TypeError(msg)

# Measure trajectory spread (standard deviation across points)
spread_in = float(np.std(traj_in.to_numpy()))
spread_mild = float(np.std(traj_mild.to_numpy()))
spread_strong = float(np.std(traj_strong.to_numpy()))
print(f"  Trajectory spread (in):      {spread_in:.4f}")
print(f"  Trajectory spread (strength=0.5):  {spread_mild:.4f}")
print(f"  Trajectory spread (strength=2.0):  {spread_strong:.4f}")

# ---------------------------------------------------------------------------
# 5. Visualise
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Before/after comparison (first 5 feature pairs of held-out sample)
ax_a = axes[0, 0]
sample_idx = 0
ax_a.plot(held_source[sample_idx], "b-o", label="Original", alpha=0.7, markersize=4)
ax_a.plot(original_recon[sample_idx], "g--s", label="Reconstructed", alpha=0.7, markersize=4)
ax_a.plot(patched_output[sample_idx], "r--^", label="Patched", alpha=0.7, markersize=4)
ax_a.set_title(f"Story A: ActivationPatch — sample {sample_idx}")
ax_a.set_xlabel("Feature index")
ax_a.set_ylabel("Value")
ax_a.legend(fontsize=8)
ax_a.grid(True, alpha=0.3)

# Panel B: Before/after RMSE bar chart
ax_b = axes[0, 1]
ax_b.bar(["Reconstruction", "Patched"], [before_rmse, after_rmse], color=["steelblue", "coral"])
ax_b.set_title("Story A: RMSE vs original input")
ax_b.set_ylabel("RMSE")
for i, v in enumerate([before_rmse, after_rmse]):
    ax_b.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

# Panel C: Trajectory comparison (first 2 latent dims)
ax_c = axes[1, 0]
data_in = traj_in.to_numpy()
data_mild = traj_mild.to_numpy()
data_strong = traj_strong.to_numpy()
ax_c.plot(data_in[:, 0], data_in[:, 1], "b-o", label="Input trajectory", alpha=0.8, markersize=6)
ax_c.plot(data_mild[:, 0], data_mild[:, 1], "g--s", label="Steered (strength=0.5)", alpha=0.8, markersize=6)
ax_c.plot(data_strong[:, 0], data_strong[:, 1], "r--^", label="Steered (strength=2.0)", alpha=0.8, markersize=6)
ax_c.set_title("Story B: SteeringVector — trajectory (dim 0 vs dim 1)")
ax_c.set_xlabel("Latent dim 0")
ax_c.set_ylabel("Latent dim 1")
ax_c.legend(fontsize=8)
ax_c.grid(True, alpha=0.3)

# Panel D: Trajectory spread comparison
ax_d = axes[1, 1]
strengths = ["Input", "Strength=0.5", "Strength=2.0"]
spreads = [spread_in, spread_mild, spread_strong]
ax_d.bar(strengths, spreads, color=["steelblue", "lightgreen", "coral"])
ax_d.set_title("Story B: Trajectory spread (std)")
ax_d.set_ylabel("Standard deviation")
for i, v in enumerate(spreads):
    ax_d.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

plt.suptitle(  # pyright: ignore[reportUnknownMemberType]
    "Pipeline #2 (ManipulationPipeline) — Sprint 21 Demo", fontsize=14, y=0.98
)
plt.tight_layout()
plt.savefig(  # pyright: ignore[reportUnknownMemberType]
    "artifacts/manipulation_demo_plot.png", dpi=150, bbox_inches="tight"
)
print("\nPlot saved to artifacts/manipulation_demo_plot.png")
plt.show()  # pyright: ignore[reportUnknownMemberType]
