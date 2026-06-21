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

# Trigger built-in registration into GLOBAL_REGISTRY before any
# registry-dependent imports (like config).
from latent_anything import _plugin_builtins as _plugin_builtins  # noqa: F401  # trigger registration
from latent_anything.config import ObjectSpec as ObjectSpec
from latent_anything.config import build_from_config as build_from_config
from latent_anything.config import build_from_dict as build_from_dict
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
    "ObjectSpec",
    "Registry",
    "RegistryEntry",
    "Trajectory",
    "build_from_config",
    "build_from_dict",
    "list_entries",
    "lookup_entry",
    "register_entry",
]
