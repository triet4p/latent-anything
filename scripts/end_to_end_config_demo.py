#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0,<3.0",
# ]
# ///

"""Sprint 18 config demo — build objects from registry-backed config specs.

This script demonstrates how to create the showcase demo's object stack
entirely from pydantic config specs, without writing manual constructor
calls. It builds:

    1. A VAE adapter (with training params)   — from ObjectSpec
    2. A PCA introspection method             — from ObjectSpec
    3. A Lerp interpolation method            — from ObjectSpec
    4. An ActivationPatch edit method         — from nested ObjectSpec (adapter inside)

No Pipeline abstraction is introduced — this is config instantiation
instance #1, registry-local and deliberately narrow.

Usage:
    uv run scripts/end_to_end_config_demo.py
"""

from __future__ import annotations

from latent_anything.config import ObjectSpec, build_from_config

SEPARATOR = "─" * 72


def print_header(text: str) -> None:
    """Print a section header."""
    print()
    print(SEPARATOR)
    print(f"  {text}")
    print(SEPARATOR)


def main() -> None:
    """Build the showcase object stack from config specs."""
    print("=" * 72)
    print("  Latent Anything — Config-Driven Object Instantiation")
    print("  Sprint 18 — registry-backed pydantic config specs")
    print("=" * 72)

    # ── 1. VAE adapter ──────────────────────────────────────────
    print_header("1. Adapter: VAE (Variational Autoencoder)")
    vae_spec = ObjectSpec(
        kind="adapter",
        name="vae",
        params={
            "input_dim": 8,
            "latent_dim": 3,
            "hidden_dim": None,
            "learning_rate": 0.005,
            "n_epochs": 300,
            "beta": 1.0,
            "random_state": 42,
        },
    )
    vae = build_from_config(vae_spec)
    print(f"    Built: {type(vae).__name__}")
    print(f"    input_dim={vae.input_dim}, latent_dim={vae.latent_dim}")
    print(f"    hidden_dim={vae.hidden_dim}, n_epochs={vae.n_epochs}")
    print(f"    latent_space: dim={vae.latent_space.dim}, geometry={vae.latent_space.geometry}")

    # ── 2. PCA Layer A method ───────────────────────────────────
    print_header("2. Layer A Method: PCA (dimensionality reduction)")
    pca_spec = ObjectSpec(kind="method_a", name="pca", params={"n_components": 2})
    pca = build_from_config(pca_spec)
    print(f"    Built: {type(pca).__name__}")
    print(f"    n_components={pca.n_components}")

    # ── 3. Lerp Layer B method ──────────────────────────────────
    print_header("3. Layer B Method: Lerp (stateless interpolation)")
    lerp_spec = ObjectSpec(kind="method_b", name="lerp")
    lerp = build_from_config(lerp_spec)
    print(f"    Built: {type(lerp).__name__}")
    print(f"    space={lerp.space} (None = Euclidean)")
    print(f"    is_fitted={lerp.is_fitted} (stateless, always ready)")

    # ── 4. ActivationPatch with nested adapter ──────────────────
    print_header("4. Layer B Method: ActivationPatch (model-mediated edit)")
    # Build with nested ObjectSpec for the adapter
    patch_spec = ObjectSpec(
        kind="method_b",
        name="activation_patch",
        params={
            "adapter": ObjectSpec(
                kind="adapter",
                name="vae",
                params={"input_dim": 8, "latent_dim": 3},
            ),
        },
    )
    patch = build_from_config(patch_spec)
    print(f"    Built: {type(patch).__name__}")
    print(f"    adapter type: {type(patch._adapter).__name__}")  # type: ignore[attr-defined]
    print(f"    space dim: {patch.space.dim}")
    print(f"    is_fitted={patch.is_fitted} (needs fit() call before use)")

    # ── 5. Build from plain dict (alternative API) ──────────────
    print_header("5. Alternative: build_from_dict (plain dict input)")
    from latent_anything.config import build_from_dict

    steering = build_from_dict({"kind": "method_b", "name": "steering"})
    print(f"    Built: {type(steering).__name__} from plain dict")
    print(f"    space={steering.space}")

    # ── 6. Summary ──────────────────────────────────────────────
    print()
    print(SEPARATOR)
    print("  Summary — config-built object stack")
    print(SEPARATOR)
    objects = [
        ("vae", type(vae).__name__, "adapter"),
        ("pca", type(pca).__name__, "method_a"),
        ("lerp", type(lerp).__name__, "method_b"),
        ("activation_patch", type(patch).__name__, "method_b (nested adapter)"),
        ("steering", type(steering).__name__, "method_b"),
    ]
    for name, cls, kind in objects:
        print(f"    {name:<20s}  {cls:<30s}  {kind}")

    print()
    print(SEPARATOR)
    print("  No Pipeline abstraction introduced.")
    print("  Config instantiation stays narrow and registry-local.")
    print(SEPARATOR)
    print()
    print("=" * 72)
    print("  Config demo complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
