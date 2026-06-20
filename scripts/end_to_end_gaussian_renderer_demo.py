#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///

"""End-to-end GaussianRendererAdapter demo.

Demonstrates:
1. Constructing a GaussianRendererAdapter with synthetic Gaussian parameters
2. Decoding into an RGB image (2D Gaussian splat rendering)
3. Encoding an image back to Gaussian parameters (heuristic grid-based)
4. Roundtrip encode → decode
5. Interpolation between two latent states using LatentSpace.gaussian_set
6. An interpolation sequence visualisation

Usage:
    uv run python scripts/end_to_end_gaussian_renderer_demo.py
"""

from __future__ import annotations

import numpy as np

from latent_anything.adapters import GaussianRendererAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_GAUSSIANS = 16  # 4 × 4 grid
IMG_H, IMG_W = 64, 80
PARAM_DIM = 8  # pos(2) + scale(2) + opacity(1) + color(3)
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helper: build a Gaussian parameter array
# ---------------------------------------------------------------------------


def _make_colour_block(
    n_g: int,
    h: int,
    w: int,
    r_val: float,
    g_val: float,
    b_val: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale_factor: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Create a grid of Gaussian parameters with a single colour.

    Returns array of shape (n_g, 8).
    """
    rng = np.random.default_rng(seed)
    latent = np.zeros((n_g, PARAM_DIM), dtype=np.float64)

    # Compute grid dimensions
    n_cols = max(1, int(np.round(np.sqrt(n_g * w / h))))
    n_rows = max(1, n_g // n_cols)
    while n_rows * n_cols < n_g:
        n_cols += 1
    while n_rows * n_cols > n_g and n_rows > 1 and (n_rows - 1) * n_cols >= n_g:
        n_rows -= 1

    cell_h = h / n_rows
    cell_w = w / n_cols

    row_centres = (np.arange(n_rows, dtype=np.float64) + 0.5) * cell_h + offset_y
    col_centres = (np.arange(n_cols, dtype=np.float64) + 0.5) * cell_w + offset_x
    grid_y, grid_x = np.meshgrid(row_centres, col_centres, indexing="ij")

    latent[:, 0] = grid_x.ravel()[:n_g]
    latent[:, 1] = grid_y.ravel()[:n_g]
    latent[:, 2] = cell_w * 0.35 * scale_factor  # scale x
    latent[:, 3] = cell_h * 0.35 * scale_factor  # scale y
    latent[:, 4] = 1.0  # opacity

    # Colour
    latent[:, 5] = r_val
    latent[:, 6] = g_val
    latent[:, 7] = b_val

    # Add small jitter
    latent[:, 0] += rng.uniform(-2, 2, size=n_g)
    latent[:, 1] += rng.uniform(-2, 2, size=n_g)
    latent[:, 2] *= rng.uniform(0.7, 1.3, size=n_g)
    latent[:, 3] *= rng.uniform(0.7, 1.3, size=n_g)

    return np.clip(latent, 0, None)


# ===================================================================
# Main demo
# ===================================================================


def main() -> None:
    print("=" * 60)
    print("GaussianRendererAdapter — End-to-End Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Construct adapter
    # ------------------------------------------------------------------
    print("\n[1] Constructing GaussianRendererAdapter ...")
    adapter = GaussianRendererAdapter(
        n_gaussians=N_GAUSSIANS,
        img_height=IMG_H,
        img_width=IMG_W,
        random_state=RANDOM_STATE,
    )
    print(f"    Latent space: {adapter.latent_space}")
    print(f"    Latent shape per point: {adapter.latent_space.shape}")

    # ------------------------------------------------------------------
    # 2. Build two distinct latent states
    # ------------------------------------------------------------------
    print("\n[2] Building two Gaussian-set latent states ...")

    latent_a = _make_colour_block(
        N_GAUSSIANS,
        IMG_H,
        IMG_W,
        r_val=1.0,
        g_val=0.2,
        b_val=0.2,  # red
        offset_x=-8,
        offset_y=-4,
        seed=10,
    )
    latent_b = _make_colour_block(
        N_GAUSSIANS,
        IMG_H,
        IMG_W,
        r_val=0.2,
        g_val=0.3,
        b_val=1.0,  # blue
        offset_x=8,
        offset_y=4,
        seed=20,
    )

    print(f"    State A: {latent_a.shape}, mean R={latent_a[:, 5].mean():.2f}")
    print(f"    State B: {latent_b.shape}, mean R={latent_b[:, 5].mean():.2f}")

    # ------------------------------------------------------------------
    # 3. Decode both states to images
    # ------------------------------------------------------------------
    print("\n[3] Decoding to RGB images ...")
    img_a = adapter.decode(latent_a)
    img_b = adapter.decode(latent_b)
    print(f"    Image A shape: {img_a.shape}, range [{img_a.min():.3f}, {img_a.max():.3f}]")
    print(f"    Image B shape: {img_b.shape}, range [{img_b.min():.3f}, {img_b.max():.3f}]")

    # ------------------------------------------------------------------
    # 4. Interpolation sequence using LatentSpace
    # ------------------------------------------------------------------
    print("\n[4] Interpolation sequence (gaussian_set geometry) ...")
    space = adapter.latent_space
    assert space.geometry == "gaussian_set"

    n_steps = 5
    print(f"    {n_steps} steps from A → B:")
    interpolated = []
    for _i, t in enumerate(np.linspace(0, 1, n_steps)):
        latent_t = space.interpolate(latent_a, latent_b, float(t))
        img_t = adapter.decode(latent_t)
        interpolated.append(img_t)
        print(f"      t={t:.2f}: image range [{img_t.min():.3f}, {img_t.max():.3f}]")

    # ------------------------------------------------------------------
    # 5. Distance
    # ------------------------------------------------------------------
    print("\n[5] Gaussian-set distance:")
    dist = space.distance(latent_a, latent_b)
    print(f"    distance(A, B) = {dist:.4f}")

    # ------------------------------------------------------------------
    # 6. Encode — heuristic
    # ------------------------------------------------------------------
    print("\n[6] Heuristic encode (grid-based) ...")
    # Use image A as the source for encoding
    encoded = adapter.encode(img_a)
    print(f"    Encoded shape: {encoded.shape}")
    print(
        f"    Encoded positions: x∈[{encoded[:, 0].min():.1f}, {encoded[:, 0].max():.1f}], "
        f"y∈[{encoded[:, 1].min():.1f}, {encoded[:, 1].max():.1f}]"
    )
    print(
        f"    Encoded mean colour: RGB=({encoded[:, 5].mean():.2f}, "
        f"{encoded[:, 6].mean():.2f}, {encoded[:, 7].mean():.2f})"
    )

    # Roundtrip
    reconstructed = adapter.decode(encoded)
    print(f"    Roundtrip image shape: {reconstructed.shape}")
    print(f"    Roundtrip range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

    # ------------------------------------------------------------------
    # 7. Determinism verification
    # ------------------------------------------------------------------
    print("\n[7] Determinism check ...")
    img_a_again = adapter.decode(latent_a)
    assert np.allclose(img_a, img_a_again), "Decode is NOT deterministic!"
    print("    ✓ Decode is deterministic (same latent → same image)")

    # ------------------------------------------------------------------
    # 8. Matplotlib visualisation
    # ------------------------------------------------------------------
    print("\n[8] Generating matplotlib figure ...")
    _plot_results(img_a, img_b, interpolated, dist)
    print("    Figure saved to artifacts/gaussian_renderer_demo.png")


def _plot_results(
    img_a: np.ndarray,
    img_b: np.ndarray,
    interpolated: list[np.ndarray],
    dist: float,
) -> None:
    """Create a 2×3 matplotlib figure showing the interpolation sequence."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(
        "GaussianRendererAdapter — Deterministic 2D Splat Decode\n"
        f"gaussian_set interpolation: {N_GAUSSIANS} Gaussians, {IMG_H}×{IMG_W} px",
        fontsize=12,
    )

    n_steps = len(interpolated)

    # Row 0: states A, B, interpolation frames
    axes[0, 0].imshow(img_a)
    axes[0, 0].set_title("State A (red)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(interpolated[n_steps // 2])
    axes[0, 1].set_title("Midpoint (t=0.5)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_b)
    axes[0, 2].set_title("State B (blue)")
    axes[0, 2].axis("off")

    # Row 1: interpolation overview and info
    for i in range(3):
        axes[1, i].imshow(interpolated[i])
        axes[1, i].set_title(f"t={i / (n_steps - 1):.2f}")
        axes[1, i].axis("off")

    # Info panel
    info_text = (
        f"Distance(A,B): {dist:.2f}\n"
        f"N Gaussians: {N_GAUSSIANS}\n"
        f"Image: {IMG_H}×{IMG_W}×3\n"
        f"Latent dim: {PARAM_DIM}\n"
        f"Decode: deterministic\n"
        f"Encode: heuristic grid"
    )
    axes[1, 2].text(
        0.1,
        0.5,
        info_text,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("artifacts/gaussian_renderer_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Matplotlib figure created.")


if __name__ == "__main__":
    main()
