"""Registry-backed config instantiation — pydantic specs for object construction.

This module provides a thin pydantic v2 layer on top of the in-process
registry (Sprint 17). It lets callers describe registry objects via typed
config specs and instantiate them in one call::

    from latent_anything.config import ObjectSpec, build_from_config

    spec = ObjectSpec(kind="method_a", name="pca", params={"n_components": 3})
    pca = build_from_config(spec)

This is **config instantiation instance #1** — registry-local and deliberately
narrow. It is not a workflow language, a Pipeline abstraction, or a Hydra
replacement.

Supports nested specs for adapter references in B-Methods::

    spec = ObjectSpec(
        kind="method_b",
        name="activation_patch",
        params={
            "adapter": ObjectSpec(
                kind="adapter",
                name="vae",
                params={"input_dim": 8, "latent_dim": 3},
            ),
        },
    )
    patch = build_from_config(spec)

Sprint 18 design constraints:
- No Hydra dependency.
- No workflow/Pipeline language.
- Pydantic v2 only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from latent_anything.registry import GLOBAL_REGISTRY, Registry

# ── Public type alias ───────────────────────────────────────────────

ParamsDict = dict[str, Any]
"""Free-form parameter dictionary forwarded to a registry entry's factory."""

# ── Pydantic config model ──────────────────────────────────────────


class ObjectSpec(BaseModel):
    """A config spec that describes a single registry object to instantiate.

    Parameters
    ----------
    kind : str
        One of ``"adapter"``, ``"method_a"``, ``"method_b"``. Must match
        the kind the entry was registered under.
    name : str
        Canonical lowercase name of a registered entry (e.g. ``"vae"``,
        ``"pca"``, ``"lerp"``).
    params : dict[str, Any], optional
        Parameters forwarded to the entry's factory as ``factory(**params)``.
        Values may be nested ``ObjectSpec`` instances (or plain dicts
        compatible with ``ObjectSpec``), which are resolved recursively
        by ``build_from_config``.

    Examples
    --------
    >>> spec = ObjectSpec(kind="method_a", name="pca", params={"n_components": 3})
    >>> spec.kind
    'method_a'
    >>> spec.name
    'pca'
    >>> spec.params
    {'n_components': 3}
    """

    kind: str = Field(..., description="One of: adapter, method_a, method_b")
    name: str = Field(..., min_length=1, description="Registered entry name")
    params: dict[str, Any] = Field(default_factory=dict, description="Factory keyword arguments")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ObjectSpec:
        """Construct an ``ObjectSpec`` from a plain dict, coercing nested specs.

        This allows callers to pass nested ``ObjectSpec``-compatible dicts
        inside ``params`` without constructing ``ObjectSpec`` instances manually::

            spec = ObjectSpec.from_dict({
                "kind": "method_b",
                "name": "activation_patch",
                "params": {
                    "adapter": {
                        "kind": "adapter",
                        "name": "vae",
                        "params": {"input_dim": 8, "latent_dim": 3},
                    },
                },
            })
        """
        return cls.model_validate(data)  # pyright: ignore[reportUnknownArgumentType]


# ── Build from config ──────────────────────────────────────────────


def _resolve_params(
    params: dict[str, Any],
    *,
    registry: Registry,
) -> dict[str, Any]:
    """Recursively resolve nested ``ObjectSpec`` values inside a params dict.

    Any value that is an ``ObjectSpec`` instance or a dict with ``"kind"``
    and ``"name"`` keys is resolved via ``build_from_config``. Other values
    are passed through unchanged.
    """
    resolved: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, ObjectSpec):
            resolved[key] = build_from_config(value, registry=registry)
        elif isinstance(value, dict) and "kind" in value and "name" in value:
            resolved[key] = build_from_config(ObjectSpec.from_dict(value), registry=registry)  # pyright: ignore[reportUnknownArgumentType]
        else:
            resolved[key] = value
    return resolved


def build_from_config(
    spec: ObjectSpec,
    *,
    registry: Registry | None = None,
) -> Any:
    """Resolve a registry entry from *spec* and instantiate it.

    This is the primary entry point for config-driven object construction::

        pca = build_from_config(ObjectSpec(kind="method_a", name="pca", params={"n_components": 3}))
        vae = build_from_config(ObjectSpec(kind="adapter", name="vae", params={"input_dim": 8, "latent_dim": 3}))

    Nested specs inside ``params`` (for B-Methods that take adapters) are
    resolved recursively before instantiation.

    Parameters
    ----------
    spec : ObjectSpec
        The config spec describing what to build.
    registry : Registry | None
        Registry to resolve names from. Defaults to ``GLOBAL_REGISTRY``.

    Returns
    -------
    Any
        The instantiated object.

    Raises
    ------
    KeyError
        If ``name`` is not found in the registry.
    ValueError
        If ``kind`` does not match the registered entry's kind.
    TypeError
        If the factory cannot be called with the resolved parameters
        (e.g. missing required args, wrong types).
    ValidationError
        If the spec itself fails pydantic validation (e.g. empty name,
        invalid kind).
    """
    target = registry if registry is not None else GLOBAL_REGISTRY

    # ── 1. Lookup ───────────────────────────────────────────────
    try:
        entry = target.lookup(spec.name)
    except KeyError:
        msg = (
            f"build_from_config: unknown name {spec.name!r}"
            + (f" in registry {target.name!r}" if target.name else "")
            + f". Available names: {sorted(e.name for e in target.list())}"
        )
        raise KeyError(msg) from None

    # ── 2. Kind check ───────────────────────────────────────────
    if entry.kind != spec.kind:
        msg = (
            f"build_from_config: kind mismatch for {spec.name!r} — "
            f"spec says {spec.kind!r} but registry entry has kind {entry.kind!r}"
        )
        raise ValueError(msg)

    # ── 3. Resolve nested specs ─────────────────────────────────
    resolved_params = _resolve_params(spec.params, registry=target)

    # ── 4. Instantiate ──────────────────────────────────────────
    try:
        return entry.factory(**resolved_params)
    except TypeError as e:
        msg = (
            f"build_from_config: failed to instantiate {spec.name!r} "
            f"(kind={spec.kind!r}) with params {resolved_params!r}: {e}"
        )
        raise TypeError(msg) from None


# ── Convenience helpers ─────────────────────────────────────────────


def build_from_dict(
    data: dict[str, Any],
    *,
    registry: Registry | None = None,
) -> Any:
    """Convenience wrapper: parse a plain dict as an ``ObjectSpec`` and build.

    Examples
    --------
    >>> pca = build_from_dict({"kind": "method_a", "name": "pca", "params": {"n_components": 3}})
    >>> patch = build_from_dict({
    ...     "kind": "method_b",
    ...     "name": "activation_patch",
    ...     "params": {
    ...         "adapter": {"kind": "adapter", "name": "vae", "params": {"input_dim": 8, "latent_dim": 3}},
    ...     },
    ... })
    """
    spec = ObjectSpec.from_dict(data)
    return build_from_config(spec, registry=registry)
