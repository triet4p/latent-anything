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
from latent_anything.latent_value import LatentValue as LatentValue
from latent_anything.methods import Method as Method
from latent_anything.pipeline import AnalysisPipeline as AnalysisPipeline
from latent_anything.pipeline import ManipulationPipeline as ManipulationPipeline
from latent_anything.pipeline import ManipulationPipelineSpec as ManipulationPipelineSpec
from latent_anything.pipeline import PipelineResult as PipelineResult
from latent_anything.pipeline import PipelineSpec as PipelineSpec
from latent_anything.pipeline import build_manipulation_pipeline_from_config as build_manipulation_pipeline_from_config
from latent_anything.pipeline import build_pipeline_from_config as build_pipeline_from_config
from latent_anything.probes import ControlBaselines as ControlBaselines
from latent_anything.probes import CrossSeedReport as CrossSeedReport
from latent_anything.probes import LinearProbe as LinearProbe
from latent_anything.probes import LinearProbeConfig as LinearProbeConfig
from latent_anything.probes import LinearProbeResult as LinearProbeResult
from latent_anything.probes import cross_seed_evaluation as cross_seed_evaluation
from latent_anything.probes import evaluate_layers as evaluate_layers
from latent_anything.registry import GLOBAL_REGISTRY as GLOBAL_REGISTRY
from latent_anything.registry import Registry as Registry
from latent_anything.registry import RegistryEntry as RegistryEntry
from latent_anything.registry import list_entries as list_entries
from latent_anything.registry import lookup as lookup_entry
from latent_anything.registry import register as register_entry
from latent_anything.runtime import BatchExecutor as BatchExecutor
from latent_anything.runtime import CacheKey as CacheKey
from latent_anything.runtime import CacheStats as CacheStats
from latent_anything.runtime import InMemoryCache as InMemoryCache
from latent_anything.runtime import ProfileEvent as ProfileEvent
from latent_anything.runtime import RuntimeProfile as RuntimeProfile
from latent_anything.runtime import RuntimeProfiler as RuntimeProfiler
from latent_anything.trajectory import Trajectory as Trajectory

__version__ = "0.1.0b1"

__all__ = [
    "AnalysisPipeline",
    "BatchExecutor",
    "CacheKey",
    "CacheStats",
    "ControlBaselines",
    "CrossSeedReport",
    "GLOBAL_REGISTRY",
    "InMemoryCache",
    "LatentSpace",
    "LatentValue",
    "LinearProbe",
    "LinearProbeConfig",
    "LinearProbeResult",
    "ManipulationPipeline",
    "ManipulationPipelineSpec",
    "Method",
    "ObjectSpec",
    "PipelineResult",
    "PipelineSpec",
    "ProfileEvent",
    "Registry",
    "RegistryEntry",
    "RuntimeProfile",
    "RuntimeProfiler",
    "Trajectory",
    "build_from_config",
    "build_from_dict",
    "build_manipulation_pipeline_from_config",
    "build_pipeline_from_config",
    "cross_seed_evaluation",
    "evaluate_layers",
    "list_entries",
    "lookup_entry",
    "register_entry",
]
