#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
#     "torch",
# ]
# ///

"""End-to-end ActivationPatch demo — B-Method #3, model-mediated data→data.

Demonstrates the ``ActivationPatch`` class across two scenarios:

  **Scenario A (VAE latent arithmetic)**:
  Train a tiny VAE on synthetic 2D blob data. Fit ``ActivationPatch``
  with source=cluster_A, target=cluster_B. Apply patch to test samples
  from cluster_A → decode → visualize patched reconstruction vs original
  reconstruction side-by-side. Show that patched outputs morph toward
  cluster_B characteristics.

  **Scenario B (Trajectory patching)**:
  Create a trajectory of latent points from cluster_A to cluster_B (via
  Lerp). Apply ``ActivationPatch.apply_trajectory(trajectory)`` →
  decode each point → create a grid visualization showing the morphing
  sequence. Compare with direct latent interpolation (Lerp → Trajectory)
  to highlight data-space vs latent-space perspective.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# Ensure we can import from the src directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latent_anything import Trajectory
from latent_anything.adapters import VAE
from latent_anything.methods import ActivationPatch, Lerp  # noqa: F811

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_2d_blob_clusters(
    n_per_cluster: int = 60,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate two separated 2D blob clusters in [0, 1]^2.

    Returns
    -------
    data_all : np.ndarray  shape (total, 2)
    labels : np.ndarray  shape (total,)
    cluster_a : np.ndarray  shape (n_per_cluster, 2)  — centered near (0.3, 0.3)
    cluster_b : np.ndarray  shape (n_per_cluster, 2)  — centered near (0.7, 0.7)
    """
    centers = np.array([[0.3, 0.3], [0.7, 0.7]])
    data_all, labels = make_blobs(
        n_samples=n_per_cluster * 2,
        centers=centers,
        cluster_std=0.08,
        random_state=seed,
    )
    # Clip to [0, 1] for VAE sigmoid assumption
    data_all = np.clip(data_all, 0.0, 1.0)

    cluster_a = data_all[labels == 0]
    cluster_b = data_all[labels == 1]
    return data_all, labels, cluster_a, cluster_b


# ---------------------------------------------------------------------------
# Scenario A: VAE latent arithmetic
# ---------------------------------------------------------------------------


def scenario_a_vae_latent_arithmetic() -> tuple[
    VAE,
    ActivationPatch,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Scenario A: train VAE → fit ActivationPatch → patch test data.

    Returns
    -------
    vae : VAE
        Trained VAE adapter.
    patch : ActivationPatch
        Fitted ActivationPatch instance.
    test_a : np.ndarray  shape (5, 2)
        Held-out test points from cluster A.
    recon_a : np.ndarray  shape (5, 2)
        Reconstruction of test_a (no patch).
    patched : np.ndarray  shape (5, 2)
        Patched reconstruction of test_a (with patch).
    test_b_ref : np.ndarray  shape (5, 2)
        Reference test points from cluster B for comparison.
    """
    print("--- Scenario A: VAE Latent Arithmetic ---")

    _, _, cluster_a, cluster_b = _make_2d_blob_clusters(n_per_cluster=60, seed=42)

    # Keep 5 samples from each for testing
    test_a = cluster_a[:5].copy()
    test_b_ref = cluster_b[:5].copy()
    train_a = cluster_a[5:]
    train_b = cluster_b[5:]

    # Train a small VAE
    vae = VAE(input_dim=2, latent_dim=4, hidden_dim=16, n_epochs=100, beta=0.5, random_state=42)
    # Train on combined data so VAE learns both clusters
    combined = np.vstack([train_a, train_b])
    vae.fit(combined)
    final_loss = vae.loss_history_[-1] if vae.loss_history_ else 0
    print(f"  VAE trained: final loss = {final_loss:.4f}")

    # Fit ActivationPatch: source = cluster_a, target = cluster_b
    patch = ActivationPatch(adapter=vae)
    patch.fit(source_data=train_a, target_data=train_b)
    delta_norm = np.linalg.norm(patch.delta)
    print(f"  Patch delta norm: {delta_norm:.6f}")

    # Reconstruct test_a (no patch)
    recon_a = vae.decode(vae.encode(test_a))

    # Patch test_a
    patched = patch(test_a)

    print(f"  Test point #0 (cluster A):        {test_a[0]}")
    print(f"  Reconstructed (no patch):         {recon_a[0]}")
    print(f"  Patched reconstruction:           {patched[0]}")
    print(f"  Reference cluster B point:        {test_b_ref[0]}")

    return vae, patch, test_a, recon_a, patched, test_b_ref


# ---------------------------------------------------------------------------
# Scenario B: Trajectory patching
# ---------------------------------------------------------------------------


def scenario_b_trajectory_patching(
    vae: VAE,
    patch: ActivationPatch,
    train_a: np.ndarray,
    train_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Trajectory, np.ndarray]:
    """Scenario B: create latent trajectory → apply trajectory patching.

    Encodes a point from cluster A and a point from cluster B, creates a
    Lerp trajectory between them, then applies ``ActivationPatch.apply_trajectory``
    to decode each point into data space.

    Returns
    -------
    latent_a, latent_b : np.ndarray  shape (latent_dim,)
        Endpoints of the trajectory in latent space.
    decoded_grid : np.ndarray  shape (5, 2)
        Decoded outputs of the patched trajectory (data space).
    """
    print("\n--- Scenario B: Trajectory Patching ---")

    # Take one mean point from each cluster in latent space
    latent_a = vae.encode(train_a).mean(axis=0)
    latent_b = vae.encode(train_b).mean(axis=0)

    # Create a Lerp trajectory from latent_a to latent_b
    lerp = Lerp()
    traj = Trajectory(data=np.array([latent_a, latent_b]))
    blended = lerp.blend_sequence(traj, n_steps=4)  # 5 points total
    print(f"  Trajectory: {len(blended)} points, dim={blended.dim}")

    # Apply ActivationPatch to the trajectory
    decoded = patch.apply_trajectory(blended)
    print(f"  Decoded trajectory shape: {decoded.shape}")

    for i in range(len(decoded)):
        print(f"    Point {i}: decoded = ({decoded[i, 0]:.4f}, {decoded[i, 1]:.4f})")

    return latent_a, latent_b, blended, decoded


# ---------------------------------------------------------------------------
# Visualization: 2×2 grid
# ---------------------------------------------------------------------------


def plot_results(
    test_a: np.ndarray,
    recon_a: np.ndarray,
    patched: np.ndarray,
    test_b_ref: np.ndarray,
    vae: VAE,
    patch: ActivationPatch,
    train_a: np.ndarray,
    train_b: np.ndarray,
    orig_traj: Trajectory,
    decoded_traj: np.ndarray,
) -> None:
    """2×2 grid visualization.

    (1) Original reconstruction — test_a points before patch
    (2) Patched reconstruction — test_a points after patch, compared to cluster B
    (3) Latent space PCA with patch direction arrow
    (4) Trajectory morphing grid — decoded trajectory points
    """
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # ---- (1) Original reconstruction ----
    ax1.scatter(test_a[:, 0], test_a[:, 1], c="blue", marker="o", s=40, label="test_a (input)")
    ax1.scatter(recon_a[:, 0], recon_a[:, 1], c="cyan", marker="^", s=40, label="recon (no patch)")
    for i in range(len(test_a)):
        ax1.annotate(
            "",
            xy=(recon_a[i, 0], recon_a[i, 1]),
            xytext=(test_a[i, 0], test_a[i, 1]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8, alpha=0.5),
        )
    ax1.set_title("(1) Original Reconstruction (no patch)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # ---- (2) Patched reconstruction vs target cluster ----
    ax2.scatter(test_a[:, 0], test_a[:, 1], c="blue", marker="o", s=40, alpha=0.3, label="test_a (input)")
    ax2.scatter(patched[:, 0], patched[:, 1], c="red", marker="^", s=40, label="patched (after patch)")
    ax2.scatter(test_b_ref[:, 0], test_b_ref[:, 1], c="green", marker="s", s=40, alpha=0.5, label="cluster B (ref)")
    for i in range(len(test_a)):
        ax2.annotate(
            "",
            xy=(patched[i, 0], patched[i, 1]),
            xytext=(test_a[i, 0], test_a[i, 1]),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.0, alpha=0.6),
        )
    ax2.set_title("(2) Patched Reconstruction → cluster B")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # ---- (3) Latent space PCA with patch direction ----
    latent_a = vae.encode(train_a)
    latent_b = vae.encode(train_b)
    latent_all = np.vstack([latent_a, latent_b])

    from sklearn.decomposition import PCA  # noqa: F811

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_all)

    n_a = len(latent_a)
    ax3.scatter(latent_2d[:n_a, 0], latent_2d[:n_a, 1], c="blue", alpha=0.5, s=20, label="cluster A (latent)")
    ax3.scatter(latent_2d[n_a:, 0], latent_2d[n_a:, 1], c="green", alpha=0.5, s=20, label="cluster B (latent)")

    # Patch direction arrow in latent space
    delta_2d = pca.transform(patch.delta.reshape(1, -1))[0]
    mean_a_2d = pca.transform(latent_a.mean(axis=0, keepdims=True))[0]
    ax3.arrow(
        mean_a_2d[0],
        mean_a_2d[1],
        delta_2d[0],
        delta_2d[1],
        head_width=0.3,
        head_length=0.3,
        fc="red",
        ec="red",
        label="patch direction",
    )

    ax3.set_title("(3) Latent Space PCA + Patch Direction")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ---- (4) Trajectory morphing grid ----
    # Original trajectory latent → decode directly (no patch)
    orig_decoded = vae.decode(orig_traj.to_numpy())
    # Patched trajectory (already decoded via apply_trajectory)
    patched_decoded = decoded_traj

    ax4.scatter(orig_decoded[:, 0], orig_decoded[:, 1], c="cyan", marker="o", s=60, label="lerp traj (decode)")
    ax4.scatter(patched_decoded[:, 0], patched_decoded[:, 1], c="red", marker="^", s=60, label="patched traj")
    for i in range(len(orig_decoded)):
        ax4.annotate(
            "",
            xy=(patched_decoded[i, 0], patched_decoded[i, 1]),
            xytext=(orig_decoded[i, 0], orig_decoded[i, 1]),
            arrowprops=dict(arrowstyle="->", color="purple", lw=1.0, alpha=0.5),
        )
    # Connect trajectory points in order
    ax4.plot(orig_decoded[:, 0], orig_decoded[:, 1], "c--", alpha=0.4)
    ax4.plot(patched_decoded[:, 0], patched_decoded[:, 1], "r--", alpha=0.4)

    ax4.set_title("(4) Trajectory Morphing: latent→data")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect("equal")

    plt.suptitle(
        "ActivationPatch — B-Method #3 (model-mediated encode→patch→decode)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent.parent / "artifacts" / "activation_patch_demo_plot.png",
        dpi=150,
    )
    plt.show()
    print("\n✓ Plot saved to artifacts/activation_patch_demo_plot.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _, _, cluster_a, cluster_b = _make_2d_blob_clusters(n_per_cluster=60, seed=42)
    train_a = cluster_a[5:]
    train_b = cluster_b[5:]

    vae, patch, test_a, recon_a, patched, test_b_ref = scenario_a_vae_latent_arithmetic()

    # Verify patch moves toward target
    dist_before = float(np.linalg.norm(recon_a.mean(axis=0) - test_b_ref.mean(axis=0)))
    dist_after = float(np.linalg.norm(patched.mean(axis=0) - test_b_ref.mean(axis=0)))
    print(f"\n  Mean dist to cluster B (before patch): {dist_before:.4f}")
    print(f"  Mean dist to cluster B (after patch):  {dist_after:.4f}")
    if dist_after < dist_before:
        print("  ✓ Patch moves reconstructions toward target cluster B")
    else:
        print("  ⚠ Patch did not reduce distance to cluster B (expected for small VAE)")

    _latent_a, _latent_b, orig_traj, decoded_traj = scenario_b_trajectory_patching(vae, patch, train_a, train_b)

    plot_results(
        test_a,
        recon_a,
        patched,
        test_b_ref,
        vae,
        patch,
        train_a,
        train_b,
        orig_traj,
        decoded_traj,
    )


if __name__ == "__main__":
    main()
