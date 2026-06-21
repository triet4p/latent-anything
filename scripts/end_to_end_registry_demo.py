#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Registry demo — list all built-in adapters and methods.

Usage:
    uv run scripts/end_to_end_registry_demo.py

This demo shows how the in-process registry works: it lists all registered
entries grouped by kind, demonstrates lookup by name, and verifies factory
callables are retrievable. This is Sprint 17's end-to-end verification.

.. note::

    This script is intentionally standalone (PEP 723) — it does not depend
    on the ``latent_anything`` package's uv environment. It serves as a
    lightweight demonstration of the registry API.
"""

from latent_anything.registry import GLOBAL_REGISTRY, KIND_ADAPTER, KIND_METHOD_A, KIND_METHOD_B


def print_separator(char: str = "─", width: int = 72) -> None:
    """Print a horizontal separator line."""
    print(char * width)


def main() -> None:
    """List all registry entries and demonstrate lookup."""
    print("=" * 72)
    print("  Latent Anything — Built-in Registry Summary")
    print("=" * 72)
    print()

    # ── Overview ────────────────────────────────────────────────
    total = len(GLOBAL_REGISTRY)
    n_adapters = len(GLOBAL_REGISTRY.list(KIND_ADAPTER))
    n_method_a = len(GLOBAL_REGISTRY.list(KIND_METHOD_A))
    n_method_b = len(GLOBAL_REGISTRY.list(KIND_METHOD_B))
    print(f"  Total entries:  {total}")
    print(f"  Adapters:       {n_adapters}")
    print(f"  Layer A methods:{n_method_a}")
    print(f"  Layer B methods:{n_method_b}")
    print()

    # ── Adapters ────────────────────────────────────────────────
    print_separator()
    print("  ADAPTERS (Layer 0)")
    print_separator()
    for entry in GLOBAL_REGISTRY.list(KIND_ADAPTER):
        proto = entry.metadata.get("protocol", "—")
        desc = entry.metadata.get("description", "—")
        print(f"    {entry.name:<20s}  {proto:<45s}")
        print(f"    {'':20s}  {desc}")
        print()

    # ── Layer A methods ─────────────────────────────────────────
    print_separator()
    print("  LAYER A METHODS (Introspection / Dimensionality Reduction)")
    print_separator()
    for entry in GLOBAL_REGISTRY.list(KIND_METHOD_A):
        proto = entry.metadata.get("protocol", "—")
        desc = entry.metadata.get("description", "—")
        print(f"    {entry.name:<20s}  {proto:<45s}")
        print(f"    {'':20s}  {desc}")
        print()

    # ── Layer B methods ─────────────────────────────────────────
    print_separator()
    print("  LAYER B METHODS (Manipulation)")
    print_separator()
    for entry in GLOBAL_REGISTRY.list(KIND_METHOD_B):
        proto = entry.metadata.get("protocol", "—")
        desc = entry.metadata.get("description", "—")
        print(f"    {entry.name:<20s}  {proto:<45s}")
        print(f"    {'':20s}  {desc}")
        print()

    # ── Lookup demo ─────────────────────────────────────────────
    print_separator()
    print("  LOOKUP DEMO")
    print_separator()

    # Look up VAE by name
    vae_entry = GLOBAL_REGISTRY.lookup("vae")
    print(f"  lookup('vae') → {vae_entry.kind} / {vae_entry.name}")
    print(f"    factory callable: {callable(vae_entry.factory)}")
    print(f"    metadata keys: {list(vae_entry.metadata.keys())}")

    # Look up PCA by name
    pca_entry = GLOBAL_REGISTRY.lookup("pca")
    print(f"  lookup('pca') → {pca_entry.kind} / {pca_entry.name}")
    print(f"    factory callable: {callable(pca_entry.factory)}")

    # Look up Lerp by name
    lerp_entry = GLOBAL_REGISTRY.lookup("lerp")
    print(f"  lookup('lerp') → {lerp_entry.kind} / {lerp_entry.name}")
    print(f"    factory callable: {callable(lerp_entry.factory)}")

    # ── Duplicate registration guard demo ───────────────────────
    print()
    print_separator()
    print("  DUPLICATE GUARD DEMO")
    print_separator()
    try:
        from latent_anything.registry import Registry

        demo_reg = Registry(name="demo")
        demo_reg.register(KIND_ADAPTER, "test_item", lambda: None)
        print("  First registration: OK")
        demo_reg.register(KIND_ADAPTER, "test_item", lambda: None)
    except ValueError as e:
        print(f"  Duplicate registration: caught ValueError — {e}")

    # ── Missing name guard demo ─────────────────────────────────
    print()
    print_separator()
    print("  MISSING NAME GUARD DEMO")
    print_separator()
    try:
        GLOBAL_REGISTRY.lookup("nonexistent_adapter")
    except KeyError as e:
        print(f"  Missing lookup: caught KeyError — {e}")

    print()
    print("=" * 72)
    print("  Registry demo complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
