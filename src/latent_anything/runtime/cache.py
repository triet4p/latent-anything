"""In-memory cache #1 for runtime reuse.

Sprint 23 adds a small memory-only cache and a stable key structure.
The key records the operation namespace, component identity, component
configuration hash, input data hash, and framework version when
available.

No disk backend, pickle format, eviction policy, or async/cache protocol is
introduced in this sprint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from typing import cast

import numpy as np


@dataclass(frozen=True)
class CacheKey:
    """Stable cache key for one numpy-array runtime operation."""

    namespace: str
    operation: str
    component_name: str
    config_hash: str
    data_hash: str
    framework_version: str | None


@dataclass(frozen=True)
class CacheStats:
    """Point-in-time in-memory cache statistics."""

    hits: int
    misses: int
    sets: int
    size: int


class InMemoryCache:
    """Small numpy-array cache backed by a process-local dictionary."""

    def __init__(self) -> None:
        self._store: dict[CacheKey, np.ndarray] = {}
        self._hits = 0
        self._misses = 0
        self._sets = 0

    def get(self, key: CacheKey) -> np.ndarray | None:
        """Return a defensive copy for ``key`` or ``None`` on miss."""
        value = self._store.get(key)
        if value is None:
            self._misses += 1
            return None
        self._hits += 1
        return value.copy()

    def set(self, key: CacheKey, value: np.ndarray) -> None:
        """Store a defensive copy of ``value`` under ``key``."""
        self._store[key] = value.copy()
        self._sets += 1

    def clear(self) -> None:
        """Remove all entries and reset stats."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._sets = 0

    @property
    def stats(self) -> CacheStats:
        """Return point-in-time cache statistics."""
        return CacheStats(hits=self._hits, misses=self._misses, sets=self._sets, size=len(self._store))

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._store)


def make_cache_key(
    *,
    namespace: str,
    operation: str,
    component: object,
    data: np.ndarray,
    framework_version: str | None = None,
) -> CacheKey:
    """Build a cache key for a component operation over numpy input data."""
    return CacheKey(
        namespace=namespace,
        operation=operation,
        component_name=_component_name(component),
        config_hash=hash_component_config(component),
        data_hash=hash_array(data),
        framework_version=framework_version if framework_version is not None else _framework_version(),
    )


def hash_array(data: np.ndarray) -> str:
    """Return a stable SHA-256 hash for a numpy array's content and shape."""
    contiguous = np.ascontiguousarray(data)
    digest = sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def hash_component_config(component: object) -> str:
    """Return a stable hash for public component configuration fields.

    Public fitted artifacts conventionally end in ``_`` and private runtime
    state starts with ``_``. Both are excluded so the hash captures
    construction/config values rather than learned arrays.
    """
    payload = {name: _jsonable(value) for name, value in vars(component).items() if _is_config_field(name)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def _component_name(component: object) -> str:
    component_type = type(component)
    return f"{component_type.__module__}.{component_type.__qualname__}"


def _is_config_field(name: str) -> bool:
    return not name.startswith("_") and not name.endswith("_") and not name.endswith("_calls")


def _framework_version() -> str | None:
    try:
        return metadata.version("latent-anything")
    except metadata.PackageNotFoundError:
        return None


def _jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        return cast(object, value.item())
    if isinstance(value, np.ndarray):
        array = cast(np.ndarray, value)
        return {
            "array_dtype": str(array.dtype),
            "array_hash": hash_array(array),
            "array_shape": array.shape,
        }
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return {str(key): _jsonable(item) for key, item in mapping.items()}
    if isinstance(value, list | tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        return [_jsonable(item) for item in sequence]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)
