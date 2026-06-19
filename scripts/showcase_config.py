"""Showcase configuration — lightweight local config for Sprint 13 end-to-end demo.

This config is **not** a framework-wide config system. It exists solely to
guarantee reproducibility for the Sprint 13 showcase. Do not promote to
``src/`` or treat as a ``Pipeline`` API — it is a local artifact.

All parameters are plain Python values (no serialisation dependency).
To reproduce a run, import this module or copy the ``SHOWCASE_CONFIG`` dict.
"""

from __future__ import annotations

from typing import TypedDict


class DataConfig(TypedDict):
    n_clusters: int
    n_per_cluster: int
    input_dim: int
    latent_dim: int
    noise_scale: float


class VAEConfig(TypedDict):
    hidden_dim: int | None
    n_epochs: int
    learning_rate: float
    beta: float


class SplitConfig(TypedDict):
    source_clusters: list[int]
    target_clusters: list[int]
    n_held_out: int


class PCAConfig(TypedDict):
    n_components: int


class PatchConfig(TypedDict):
    pass


class LerpConfig(TypedDict):
    n_steps: int


class OutputConfig(TypedDict):
    figure: str
    summary: str
    config_snapshot: str


class ShowcaseConfig(TypedDict):
    seed: int
    data: DataConfig
    vae: VAEConfig
    split: SplitConfig
    pca: PCAConfig
    patch: PatchConfig
    lerp: LerpConfig
    output: OutputConfig


SHOWCASE_CONFIG: ShowcaseConfig = {
    # ── Seed ──────────────────────────────────────────────────────────
    "seed": 42,
    # ── Data generation ───────────────────────────────────────────────
    "data": {
        "n_clusters": 4,
        "n_per_cluster": 80,
        "input_dim": 8,
        "latent_dim": 3,
        "noise_scale": 0.08,
    },
    # ── VAE adapter ───────────────────────────────────────────────────
    "vae": {
        "hidden_dim": None,  # None → max(latent_dim * 4, input_dim)
        "n_epochs": 300,
        "learning_rate": 0.005,
        "beta": 1.0,
    },
    # ── Source / target split ─────────────────────────────────────────
    # Two clusters are designated "source", two "target".
    # Held-out failure samples are drawn from source clusters.
    "split": {
        "source_clusters": [0, 1],
        "target_clusters": [2, 3],
        "n_held_out": 10,
    },
    # ── PCA visualisation ─────────────────────────────────────────────
    "pca": {
        "n_components": 2,
    },
    # ── ActivationPatch ───────────────────────────────────────────────
    "patch": {
        # No extra params — ActivationPatch infers dim from adapter.
    },
    # ── Lerp trajectory ───────────────────────────────────────────────
    "lerp": {
        "n_steps": 6,  # 6 interpolations → 7 points per trajectory
    },
    # ── Output paths (relative to repo root) ──────────────────────────
    "output": {
        "figure": "artifacts/showcase_demo_plot.png",
        "summary": "artifacts/showcase_demo_summary.txt",
        "config_snapshot": "artifacts/showcase_config_snapshot.txt",
    },
}

__all__ = [
    "DataConfig",
    "LerpConfig",
    "OutputConfig",
    "PCAConfig",
    "PatchConfig",
    "SHOWCASE_CONFIG",
    "ShowcaseConfig",
    "SplitConfig",
    "VAEConfig",
]
