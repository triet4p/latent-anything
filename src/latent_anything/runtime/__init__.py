"""Layer C runtime helpers.

Sprint 22 starts Layer C with a single concrete runtime primitive:
``BatchExecutor``. It is intentionally eager and synchronous. Cache,
async execution, streaming, and profiling remain future runtime
increments.
"""

from latent_anything.runtime.batch_executor import BatchExecutor as BatchExecutor
from latent_anything.runtime.cache import CacheKey as CacheKey
from latent_anything.runtime.cache import CacheStats as CacheStats
from latent_anything.runtime.cache import InMemoryCache as InMemoryCache
from latent_anything.runtime.cache import hash_array as hash_array
from latent_anything.runtime.cache import hash_component_config as hash_component_config
from latent_anything.runtime.cache import make_cache_key as make_cache_key

__all__ = [
    "BatchExecutor",
    "CacheKey",
    "CacheStats",
    "InMemoryCache",
    "hash_array",
    "hash_component_config",
    "make_cache_key",
]
