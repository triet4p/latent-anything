"""Semantic registry-kind aliases and migration diagnostics."""

from __future__ import annotations

import warnings

KIND_ADAPTER = "adapter"
KIND_ANALYSIS = "analysis"
KIND_INTERVENTION = "intervention"
KIND_PLANNER = "planner"
KIND_RUNTIME = "runtime"

LEGACY_KIND_ALIASES: dict[str, str] = {
    "method_a": KIND_ANALYSIS,
    "method_b": KIND_INTERVENTION,
}


def canonical_kind(kind: str, *, warn: bool = False) -> str:
    """Return the canonical kind, warning only at a user construction boundary."""

    canonical = LEGACY_KIND_ALIASES.get(kind, kind)
    if warn and canonical != kind:
        warnings.warn(
            f"Registry kind {kind!r} is deprecated; use {canonical!r} before 0.9.0.",
            DeprecationWarning,
            stacklevel=3,
        )
    return canonical


def migration_record(kind: str) -> dict[str, str | bool]:
    """Return a machine-readable migration record for one config kind."""

    canonical = canonical_kind(kind)
    return {"input_kind": kind, "canonical_kind": canonical, "migrated": canonical != kind}
