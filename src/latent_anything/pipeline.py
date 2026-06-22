"""Pipeline #1 — adapter → encode → Layer A method → typed result.

This module provides the first concrete ``Pipeline`` instance in the
latent-anything framework: ``AnalysisPipeline``, which chains a model
adapter's ``encode`` step with a Layer A dimensionality-reduction
method's ``fit_transform``.

This is **Pipeline instance #1** — deliberately concrete and narrow:

- It does **not** support Layer B manipulation yet.
- It is **not** a generic DAG/executor abstraction.
- It returns a typed ``PipelineResult`` dataclass, not a raw dict.
- It provides a config-backed construction path via ``PipelineSpec``
  that uses the Sprint 18 config machinery (``ObjectSpec`` resolution).

Design constraints (from Sprint 20 plan):
- Only adapter + Layer A method composition.
- ``encode`` then ``fit_transform`` / ``transform`` with numpy-only data.
- Config-backed path uses ``ObjectSpec`` for component specs.
- No broad workflow engine — generalisation waits for Pipeline #2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from latent_anything.adapters.protocols import ModelAdapter
from latent_anything.config import ObjectSpec, build_from_config
from latent_anything.latent_space import LatentSpace
from latent_anything.methods.protocols import Method
from latent_anything.registry import Registry

# ── Public type alias ───────────────────────────────────────────────


# ── Typed result ────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineResult:
    """Typed result from an ``AnalysisPipeline.run()`` call.

    Attributes
    ----------
    latents : np.ndarray
        Encoded latent representation of shape ``(n_samples, latent_dim)``.
    transformed : np.ndarray
        Method-transformed data of shape ``(n_samples, n_components)``.
    latent_space : LatentSpace
        The adapter's latent space descriptor (shape, geometry, metadata).
    """

    latents: np.ndarray
    transformed: np.ndarray
    latent_space: LatentSpace


# ── AnalysisPipeline ────────────────────────────────────────────────


class AnalysisPipeline:
    """Concrete analysis pipeline: adapter → encode → fit → transform → result.

    Chains a model adapter's ``encode`` step with a Layer A
    dimensionality-reduction method's ``fit`` + ``transform``. This
    is the first ``Pipeline`` instance — deliberately narrow, no
    Layer B or DAG support.

    Parameters
    ----------
    adapter : ModelAdapter
        A model adapter conforming to the ``ModelAdapter`` Protocol.
        Provides ``encode(data)`` and ``latent_space``.
    method : Method
        A Layer A introspection method conforming to the ``Method``
        Protocol. Provides ``fit(data)`` and ``transform(data)``.

    Examples
    --------
    >>> from latent_anything.adapters.vae import VAE
    >>> from latent_anything.methods.pca import PCA
    >>> import numpy as np

    >>> vae = VAE(input_dim=8, latent_dim=3)
    >>> pca = PCA(n_components=2)
    >>> pipeline = AnalysisPipeline(adapter=vae, method=pca)

    >>> data = np.random.rand(50, 8)
    >>> result = pipeline.run(data)
    >>> result.latents.shape
    (50, 3)
    >>> result.transformed.shape
    (50, 2)
    """

    def __init__(self, adapter: ModelAdapter, method: Method) -> None:
        self.adapter = adapter
        self.method = method
        self._latent_space: LatentSpace = adapter.latent_space

    # ── Properties ──────────────────────────────────────────────

    @property
    def latent_space(self) -> LatentSpace:
        """The adapter's latent space descriptor."""
        return self._latent_space

    # ── Run ─────────────────────────────────────────────────────

    def run(self, data: np.ndarray) -> PipelineResult:
        """Execute the pipeline: encode → fit → transform.

        Encodes *data* through the adapter, then fits the Layer A
        method on the latent representation and transforms it.

        Parameters
        ----------
        data : np.ndarray
            Input data compatible with the adapter's ``encode`` method.
            Typically ``(n_samples, input_dim)`` for flat-batch adapters.

        Returns
        -------
        PipelineResult
            A typed result containing the encoded latents, the
            method-transformed data, and the latent space descriptor.
        """
        latents = self.adapter.encode(data)
        self.method.fit(latents)
        transformed = self.method.transform(latents)
        return PipelineResult(
            latents=latents,
            transformed=transformed,
            latent_space=self._latent_space,
        )


# ── Config-backed construction ──────────────────────────────────────


class PipelineSpec(BaseModel):
    """Config spec for building an ``AnalysisPipeline`` from registry specs.

    Wraps two ``ObjectSpec`` instances (adapter + Layer A method) that
    are resolved through the registry by ``build_pipeline_from_config``.

    This is a composition layer on top of the Sprint 18 config machinery.
    It does **not** introduce a new registry kind — it composes existing
    registry objects.

    Parameters
    ----------
    adapter : ObjectSpec
        Config spec for the adapter (kind must be ``"adapter"``).
    method : ObjectSpec
        Config spec for the Layer A method (kind must be ``"method_a"``).

    Examples
    --------
    >>> spec = PipelineSpec(
    ...     adapter={"kind": "adapter", "name": "vae", "params": {"input_dim": 8, "latent_dim": 3}},
    ...     method={"kind": "method_a", "name": "pca", "params": {"n_components": 2}},
    ... )
    >>> pipeline = build_pipeline_from_config(spec)
    """

    adapter: ObjectSpec = Field(..., description="Config spec for the adapter")
    method: ObjectSpec = Field(..., description="Config spec for the Layer A method")


def build_pipeline_from_config(
    spec: PipelineSpec,
    *,
    registry: Registry | None = None,
) -> AnalysisPipeline:
    """Build an ``AnalysisPipeline`` from a ``PipelineSpec``.

    Uses the Sprint 18 config machinery (``build_from_config``) to
    resolve the adapter and method specs through the registry, then
    composes them into a pipeline.

    Parameters
    ----------
    spec : PipelineSpec
        Config spec containing ``adapter`` and ``method`` ``ObjectSpec``
        instances (or compatible dicts for auto-coercion).
    registry : Registry | None
        Registry for resolving ``ObjectSpec`` entries. Defaults to
        ``GLOBAL_REGISTRY``.

    Returns
    -------
    AnalysisPipeline
        The composed pipeline with resolved adapter and method.

    Raises
    ------
    KeyError
        If either spec name is not found in the registry.
    ValueError
        If a spec kind does not match the registered entry's kind.
    TypeError
        If a factory cannot be called with the resolved parameters.
    """
    adapter = build_from_config(spec.adapter, registry=registry)  # pyright: ignore[reportUnknownVariableType]
    method = build_from_config(spec.method, registry=registry)  # pyright: ignore[reportUnknownVariableType]
    return AnalysisPipeline(adapter=adapter, method=method)  # pyright: ignore[reportUnknownArgumentType]
