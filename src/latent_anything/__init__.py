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
from latent_anything.registry import GLOBAL_REGISTRY as GLOBAL_REGISTRY
from latent_anything.registry import Registry as Registry
from latent_anything.registry import RegistryEntry as RegistryEntry
from latent_anything.registry import list_entries as list_entries
from latent_anything.registry import lookup as lookup_entry
from latent_anything.registry import register as register_entry
from latent_anything.trajectory import Trajectory as Trajectory

__version__ = "0.1.0"

__all__ = [
    "GLOBAL_REGISTRY",
    "LatentSpace",
    "Method",
    "Registry",
    "RegistryEntry",
    "Trajectory",
    "list_entries",
    "lookup_entry",
    "register_entry",
]
