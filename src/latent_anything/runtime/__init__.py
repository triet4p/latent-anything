"""Layer C runtime helpers.

Sprint 22 starts Layer C with a single concrete runtime primitive:
``BatchExecutor``. It is intentionally eager and synchronous. Cache,
streaming remain future runtime increments. Sprint 24 adds thin async
wrappers and profiling helpers around the existing concrete runtime
paths.
"""

from latent_anything.runtime.batch_executor import BatchExecutor as BatchExecutor
from latent_anything.runtime.cache import CacheKey as CacheKey
from latent_anything.runtime.cache import CacheStats as CacheStats
from latent_anything.runtime.cache import InMemoryCache as InMemoryCache
from latent_anything.runtime.cache import hash_array as hash_array
from latent_anything.runtime.cache import hash_component_config as hash_component_config
from latent_anything.runtime.cache import make_cache_key as make_cache_key
from latent_anything.runtime.profiling import ProfileEvent as ProfileEvent
from latent_anything.runtime.profiling import RuntimeProfile as RuntimeProfile
from latent_anything.runtime.profiling import RuntimeProfiler as RuntimeProfiler

__all__ = [
    "BatchExecutor",
    "CacheKey",
    "CacheStats",
    "InMemoryCache",
    "ProfileEvent",
    "RuntimeProfile",
    "RuntimeProfiler",
    "hash_array",
    "hash_component_config",
    "make_cache_key",
]
