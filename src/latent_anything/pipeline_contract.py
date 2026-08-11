"""Shared metadata contract for the three concrete pipeline stories."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from latent_anything.latent_space import LatentSpace

PipelineKind = Literal["analysis", "manipulation", "rollout"]


@runtime_checkable
class PipelineContract(Protocol):
    """The small surface proven by Analysis, Manipulation, and Rollout.

    The stories intentionally do not share a generic ``run`` method: their
    inputs and result semantics differ.  They do share stable introspection
    metadata, which is useful to registries, experiment records, and callers
    that need to describe a composed pipeline without executing it.
    """

    pipeline_kind: PipelineKind

    @property
    def latent_space(self) -> LatentSpace | None:
        """Return the associated latent space when one is available."""

        ...
