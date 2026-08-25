"""Canonical Python entry-point groups for external plugins.

The groups describe the concrete capability families that have working
built-ins by Sprint 73.  They are deliberately a small vocabulary rather
than a general dependency-injection or workflow contract.  The existing
registry keeps ``transition`` and ``planner`` implementations under its
``runtime`` kind; their separate entry-point groups preserve the public
capability distinction without widening that registry kind prematurely.

``intervention`` is the canonical name for transformation-like operations.
``transformation`` is intentionally not a second group: RFC 0001 rejected
that ambiguous term in favour of the behavior-oriented ``intervention``.
"""

from __future__ import annotations

ENTRY_POINT_GROUP_ADAPTER = "latent_anything.adapter"
"""Entry-point group for model/representation adapters."""

ENTRY_POINT_GROUP_ANALYSIS = "latent_anything.analysis"
"""Entry-point group for representation analysis capabilities."""

ENTRY_POINT_GROUP_INTERVENTION = "latent_anything.intervention"
"""Entry-point group for representation interventions/transformations."""

ENTRY_POINT_GROUP_TRANSITION = "latent_anything.transition"
"""Entry-point group for latent-state transition capabilities."""

ENTRY_POINT_GROUP_PLANNER = "latent_anything.planner"
"""Entry-point group for action-sequence planner capabilities."""

# A tuple, rather than a set, makes the discovery order part of the stable
# contract.  Individual entry points are sorted by the loader in a later
# task; this tuple only defines the canonical group vocabulary and order.
CANONICAL_ENTRY_POINT_GROUPS = (
    ENTRY_POINT_GROUP_ADAPTER,
    ENTRY_POINT_GROUP_ANALYSIS,
    ENTRY_POINT_GROUP_INTERVENTION,
    ENTRY_POINT_GROUP_TRANSITION,
    ENTRY_POINT_GROUP_PLANNER,
)

__all__ = [
    "CANONICAL_ENTRY_POINT_GROUPS",
    "ENTRY_POINT_GROUP_ADAPTER",
    "ENTRY_POINT_GROUP_ANALYSIS",
    "ENTRY_POINT_GROUP_INTERVENTION",
    "ENTRY_POINT_GROUP_PLANNER",
    "ENTRY_POINT_GROUP_TRANSITION",
]
