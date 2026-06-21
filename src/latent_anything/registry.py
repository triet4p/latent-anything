"""In-process registry infrastructure (no built-in class dependencies).

This module defines the ``Registry`` class, kind constants, and
convenience helpers. It has **no knowledge** of concrete adapter or
method classes — those are registered separately in
``_plugin_builtins.py``.

This is registry instance #1 — the first step of **Plugin Extraction**
(Milestone 4). It is intentionally in-process with no Python entry points
yet. The registry is a simple ``OrderedDict``-backed store with
deterministic insertion-order iteration.

Sprint 17 design decisions (following project ADRs):

- **No entry points.** The registry uses local class references only.
  Python ``importlib.metadata`` entry points will be considered when a
  second instance (external plugin) demands them (Rule of Three).
- **In-process singleton.** A global ``GLOBAL_REGISTRY`` instance is
  provided for convenience, but callers may create standalone
  ``Registry`` instances for testing or scoped sub-registries.
- **No behavior change.** Existing adapters and methods are unchanged —
  the registry only adds a discovery layer on top.
- **Factory = class.** For built-in classes, the "factory" is the class
  itself (or a callable that returns an instance). Callers do
  ``registry.lookup("vae").factory(...)`` to construct.

Kind constants
--------------
- ``KIND_ADAPTER`` — ModelAdapter / DecodableAdapter (Layer 0: models)
- ``KIND_METHOD_A`` — Layer A introspection methods (dimensionality
  reduction: PCA, UMAP, SAE)
- ``KIND_METHOD_B`` — Layer B manipulation methods (Lerp, SteeringVector,
  ActivationPatch)
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ── Kind constants ──────────────────────────────────────────────────

KIND_ADAPTER = "adapter"
"""ModelAdapter / DecodableAdapter — Layer 0 model adapters."""

KIND_METHOD_A = "method_a"
"""Layer A introspection methods (dimensionality reduction)."""

KIND_METHOD_B = "method_b"
"""Layer B manipulation methods."""

# ── Registry records ────────────────────────────────────────────────


@dataclass(frozen=True)
class RegistryEntry:
    """A single entry in the registry.

    Parameters
    ----------
    kind : str
        One of ``KIND_ADAPTER``, ``KIND_METHOD_A``, ``KIND_METHOD_B``.
    name : str
        Canonical lowercase name for lookup (e.g. ``"vae"``, ``"pca"``).
    factory : Callable[..., Any]
        Callable that returns an instance — typically the class itself
        for built-in entries.
    metadata : dict[str, Any]
        Arbitrary key-value metadata (description, version, notes, etc.).
    """

    kind: str
    name: str
    factory: Callable[..., Any]
    metadata: dict[str, Any] = field(default_factory=lambda: {})


# ── Registry class ──────────────────────────────────────────────────


class Registry:
    """In-process registry for adapters and methods.

    Provides ``register``, ``lookup``, and ``list`` operations with
    deterministic insertion-order iteration (backed by ``OrderedDict``).

    Parameters
    ----------
    name : str | None
        Optional human-readable name for this registry instance
        (e.g. ``"global"``, ``"test-scope"``).
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = name
        self._entries: OrderedDict[str, RegistryEntry] = OrderedDict()

    # ── Properties ──────────────────────────────────────────────

    @property
    def name(self) -> str | None:
        """Return this registry's optional human-readable name."""
        return self._name

    # ── Registration ────────────────────────────────────────────

    def register(
        self,
        kind: str,
        name: str,
        factory: Callable[..., Any],
        **metadata: Any,
    ) -> None:
        """Register a new entry.

        Parameters
        ----------
        kind : str
            One of ``KIND_ADAPTER``, ``KIND_METHOD_A``, ``KIND_METHOD_B``.
        name : str
            Canonical lowercase name. Must not already be registered.
        factory : Callable[..., Any]
            Callable that returns an instance.
        **metadata : Any
            Extra metadata attached to the entry.

        Raises
        ------
        ValueError
            If ``name`` is already registered in this instance.
        """
        if name in self._entries:
            msg = f"Duplicate registry entry {name!r}" + (f" in registry {self._name!r}" if self._name else "")
            raise ValueError(msg)
        meta: dict[str, Any] = dict(metadata)
        self._entries[name] = RegistryEntry(
            kind=kind,
            name=name,
            factory=factory,
            metadata=meta,
        )

    # ── Lookup ──────────────────────────────────────────────────

    def lookup(self, name: str) -> RegistryEntry:
        """Look up a registered entry by name.

        Parameters
        ----------
        name : str
            Canonical lowercase name.

        Returns
        -------
        RegistryEntry
            The matching entry.

        Raises
        ------
        KeyError
            If no entry with that name exists.
        """
        if name not in self._entries:
            msg = f"No registry entry for {name!r}" + (f" in registry {self._name!r}" if self._name else "")
            raise KeyError(msg)
        return self._entries[name]

    # ── Listing ─────────────────────────────────────────────────

    def list(self, kind: str | None = None) -> list[RegistryEntry]:
        """List registered entries, optionally filtered by kind.

        Parameters
        ----------
        kind : str | None
            If provided, only entries of this kind are returned.
            ``None`` returns all entries.

        Returns
        -------
        list[RegistryEntry]
            Entries in insertion order.
        """
        if kind is None:
            return list(self._entries.values())
        return [e for e in self._entries.values() if e.kind == kind]

    # ── Convenience ─────────────────────────────────────────────

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered (``name in registry``)."""
        return name in self._entries

    def __len__(self) -> int:
        """Return the number of registered entries (``len(registry)``)."""
        return len(self._entries)

    def __bool__(self) -> bool:
        """A ``Registry`` is always truthy, even when empty.

        Without this override, Python falls back to ``__len__`` for
        truthiness, so ``registry or GLOBAL_REGISTRY`` would silently
        fall through to the global singleton when the registry is empty.
        """
        return True

    def __repr__(self) -> str:
        info = f"Registry(len={len(self)})"
        if self._name:
            info = f"Registry({self._name!r}, len={len(self)})"
        return info


# ── Global singleton ────────────────────────────────────────────────

GLOBAL_REGISTRY = Registry(name="global")
"""The module-level global registry instance.

Built-in adapters and methods are registered here by
``_plugin_builtins.py``, not by this module directly. Importing
``latent_anything`` (or ``_plugin_builtins``) triggers registration.
"""


# ── Convenience helpers ─────────────────────────────────────────────


def register(
    kind: str,
    name: str,
    factory: Callable[..., Any],
    *,
    registry: Registry | None = None,
    **metadata: Any,
) -> None:
    """Register an entry in a registry (defaults to ``GLOBAL_REGISTRY``).

    This is a convenience function so callers can write::

        from latent_anything.registry import register, KIND_ADAPTER

        register(KIND_ADAPTER, "vae", VAE, description="Variational Autoencoder")

    Parameters
    ----------
    kind : str
        One of ``KIND_ADAPTER``, ``KIND_METHOD_A``, ``KIND_METHOD_B``.
    name : str
        Canonical lowercase name.
    factory : Callable[..., Any]
        Callable that returns an instance.
    registry : Registry | None
        Target registry. Defaults to ``GLOBAL_REGISTRY``.
    **metadata : Any
        Extra metadata attached to the entry.
    """
    target = registry if registry is not None else GLOBAL_REGISTRY
    target.register(kind, name, factory, **metadata)


def list_entries(kind: str | None = None, *, registry: Registry | None = None) -> list[RegistryEntry]:
    """List entries from a registry (defaults to ``GLOBAL_REGISTRY``).

    Parameters
    ----------
    kind : str | None
        Optional kind filter.
    registry : Registry | None
        Source registry. Defaults to ``GLOBAL_REGISTRY``.

    Returns
    -------
    list[RegistryEntry]
        Entries in insertion order.
    """
    target = registry or GLOBAL_REGISTRY
    return target.list(kind)


def lookup(name: str, *, registry: Registry | None = None) -> RegistryEntry:
    """Look up an entry by name from a registry (defaults to ``GLOBAL_REGISTRY``).

    Parameters
    ----------
    name : str
        Canonical lowercase name.
    registry : Registry | None
        Source registry. Defaults to ``GLOBAL_REGISTRY``.

    Returns
    -------
    RegistryEntry
        The matching entry.

    Raises
    ------
    KeyError
        If no entry with that name exists.
    """
    target = registry or GLOBAL_REGISTRY
    return target.lookup(name)
