"""Latent Understanding, Manipulation & Execution Network.

A Python framework that treats latent space as a first-class object:
load latent representations from any model, inspect them, manipulate
them, and execute pipelines efficiently.

Plugin-first architecture with three pillars:
- **Introspection (A)** — Visualization, probing, clustering, sparse
  decomposition, trajectory analysis.
- **Manipulation (B)** — Interpolation, arithmetic, steering, activation
  patching, composition, constrained editing.
- **Runtime (C)** — Batching, caching, async execution, streaming,
  profiling.
"""

from latent_anything.latent_space import LatentSpace as LatentSpace
from latent_anything.methods import Method as Method
from latent_anything.trajectory import Trajectory as Trajectory

__version__ = "0.1.0"

__all__ = ["LatentSpace", "Method", "Trajectory"]
