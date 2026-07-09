"""Pipeline #1 (Analysis) + Pipeline #2 (Manipulation) + shared shape sketch.

This module provides two concrete ``Pipeline`` instances and a minimal
shared base that records their common shape without freezing an
abstraction:

- **``AnalysisPipeline``** (Pipeline #1) — adapter → encode → Layer A
  method → typed result. Concrete, narrow, no Layer B support.
- **``ManipulationPipeline``** (Pipeline #2) — Layer B method on
  latent or data-space. Supports two stories:
    1. Adapter-mediated (data-space output): encode → BMethod → decode.
    2. Latent-only (trajectory output): BMethod.apply_trajectory.

- **``_PipelineBase``** — internal sketch of the shared surface between
  the two pipeline instances. Not frozen — a third pipeline instance
  (e.g. RuntimePipeline) must appear before generalisation.

Design constraints:
- ``__call__`` signatures differ across B-Methods (Lerp: ``(a,b,t)``,
  SteeringVector: ``(latent, strength)``, ActivationPatch: ``(input_data)``).
  The pipeline does **not** hide this behind a brittle generic call.
- Config-backed path uses ``ObjectSpec`` for component specs.
- No broad DAG/executor abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from latent_anything.adapters.protocols import (
    FlatBatchDecodableAdapter,
    ModelAdapter,
)
from latent_anything.config import ObjectSpec, build_from_config
from latent_anything.latent_space import LatentSpace
from latent_anything.methods.b_protocols import BMethod
from latent_anything.methods.protocols import Method
from latent_anything.registry import Registry
from latent_anything.runtime.cache import InMemoryCache, make_cache_key
from latent_anything.trajectory import Trajectory

# ── Shared base (sketch) ────────────────────────────────────────────


class _PipelineBase:
    """Minimal shared base for pipeline instances — sketch only.

    Records the common surface that ``AnalysisPipeline`` and
    ``ManipulationPipeline`` share: an adapter reference, a method
    reference, and a ``LatentSpace`` descriptor.

    This is deliberately thin — no abstract interface, no generic
    ``run()`` signature. A third pipeline instance (e.g. runtime /
    streaming) must appear before this is promoted from sketch to
    frozen Protocol or ABC.

    .. note::

        ``_PipelineBase`` is **internal** and not part of the public
        API. It may change or disappear without notice when Pipeline
        #3 arrives.
    """

    def __init__(self, adapter: object, method: object) -> None:
        self._adapter: object = adapter
        self._method: object = method
        self._latent_space: LatentSpace | None = None


# ── Typed results ───────────────────────────────────────────────────


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


# ── AnalysisPipeline (Pipeline #1) ──────────────────────────────────


class AnalysisPipeline(_PipelineBase):
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

    def __init__(self, adapter: ModelAdapter, method: Method, cache: InMemoryCache | None = None) -> None:
        super().__init__(adapter=adapter, method=method)
        self.adapter: ModelAdapter = adapter
        self.method: Method = method
        self.cache = cache
        self._latent_space: LatentSpace = adapter.latent_space  # pyright: ignore[reportIncompatibleVariableOverride]

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
        if self.cache is None:
            latents = self.adapter.encode(data)
            self.method.fit(latents)
            transformed = self.method.transform(latents)
        else:
            latents = self._cached_encode(data)
            transformed = self._cached_fit_transform(latents)
        return PipelineResult(
            latents=latents,
            transformed=transformed,
            latent_space=self._latent_space,
        )

    def _cached_encode(self, data: np.ndarray) -> np.ndarray:
        key = make_cache_key(
            namespace="analysis_pipeline", operation="adapter.encode", component=self.adapter, data=data
        )
        cached = self.cache.get(key) if self.cache is not None else None
        if cached is not None:
            return cached
        latents = self.adapter.encode(data)
        if self.cache is not None:
            self.cache.set(key, latents)
        return latents

    def _cached_fit_transform(self, latents: np.ndarray) -> np.ndarray:
        key = make_cache_key(
            namespace="analysis_pipeline",
            operation="method.fit_transform",
            component=self.method,
            data=latents,
        )
        cached = self.cache.get(key) if self.cache is not None else None
        if cached is not None:
            return cached
        self.method.fit(latents)
        transformed = self.method.transform(latents)
        if self.cache is not None:
            self.cache.set(key, transformed)
        return transformed


# ── ManipulationPipeline (Pipeline #2) ──────────────────────────────


class ManipulationPipeline(_PipelineBase):
    """Concrete manipulation pipeline: BMethod → latent / data-space output.

    Chains a Layer B (Manipulation) method with an optional
    ``FlatBatchDecodableAdapter`` for data-space workflows. This
    is **Pipeline instance #2** — deliberately concrete, supporting
    two stories without a generic DAG abstraction:

    1. **Adapter-mediated (data-space)**: The BMethod is model-mediated
       (e.g. ``ActivationPatch``). ``run_data(data)`` encodes, applies
       the method, decodes, and returns a ``np.ndarray`` in data space.
    2. **Latent-only (trajectory)**: The BMethod operates directly on
       latent points (e.g. ``Lerp``, ``SteeringVector``).
       ``run_trajectory(trajectory, **kwargs)`` delegates to the
       method's ``apply_trajectory`` and returns a ``Trajectory`` (or
       ``np.ndarray`` for data-space methods).

    **Why no generic ``run()``?** ``__call__`` signatures differ
    across B-Methods (Lerp: ``(a, b, t)``, SteeringVector:
    ``(latent, strength)``, ActivationPatch: ``(input_data)``).
    A generic ``run()`` would require brittle argument inspection
    or a complicated dispatch — the sprint explicitly avoids this.

    Parameters
    ----------
    method : BMethod
        A Layer B manipulation method conforming to the ``BMethod``
        Protocol (structural, duck-typed). Provides ``is_fitted``,
        ``space``, and ``apply_trajectory``.
    adapter : FlatBatchDecodableAdapter | None, optional
        Optional ``FlatBatchDecodableAdapter`` for data-space stories.
        Required when the BMethod is model-mediated (e.g.
        ``ActivationPatch``). Defaults to ``None``.

    Examples
    --------
    >>> from latent_anything.methods.steering import SteeringVector
    >>> import numpy as np

    >>> steer = SteeringVector()
    >>> pipeline = ManipulationPipeline(method=steer)
    >>> pos = np.random.rand(20, 5)
    >>> neg = np.random.rand(20, 5)
    >>> pipeline.fit(pos, neg)
    >>> traj_in = Trajectory(data=np.random.rand(10, 5))
    >>> traj_out = pipeline.run_trajectory(traj_in, strength=0.5)
    >>> isinstance(traj_out, Trajectory)
    True
    """

    def __init__(
        self,
        method: BMethod,
        adapter: FlatBatchDecodableAdapter | None = None,
    ) -> None:
        super().__init__(adapter=adapter, method=method)
        self._method: BMethod = method  # pyright: ignore[reportIncompatibleVariableOverride]
        self._adapter: FlatBatchDecodableAdapter | None = adapter  # pyright: ignore[reportIncompatibleVariableOverride]

        # Resolve latent_space from method.space or adapter, whichever
        # provides it.
        method_space = getattr(method, "space", None)
        if method_space is not None:
            self._latent_space = method_space  # pyright: ignore[reportIncompatibleVariableOverride]
        elif adapter is not None:
            self._latent_space = adapter.latent_space  # pyright: ignore[reportIncompatibleVariableOverride]
        else:
            self._latent_space = None

    # ── Properties ──────────────────────────────────────────────

    @property
    def method(self) -> BMethod:
        """The Layer B manipulation method."""
        return self._method

    @property
    def adapter(self) -> FlatBatchDecodableAdapter | None:
        """The optional flat-batch decodable adapter."""
        return self._adapter

    @property
    def latent_space(self) -> LatentSpace | None:
        """The latent space descriptor, if available."""
        return self._latent_space

    # ── Fit (delegate) ──────────────────────────────────────────

    def fit(self, *args: object, **kwargs: object) -> None:
        """Fit the BMethod if it has a ``fit`` method.

        Delegates directly to ``method.fit(*args, **kwargs)``.
        Stateless methods (``Lerp``) raise ``TypeError`` because
        they have no ``fit`` — use ``run_trajectory`` directly.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to the method's ``fit``.
        **kwargs : object
            Keyword arguments forwarded to the method's ``fit``.

        Raises
        ------
        TypeError
            If the method has no ``fit`` attribute.
        """
        fit_fn = getattr(self._method, "fit", None)
        if fit_fn is None:
            msg = f"{type(self._method).__name__} has no fit method (stateless methods like Lerp skip the fit phase)"
            raise TypeError(msg)
        fit_fn(*args, **kwargs)

    # ── Adapter-mediated story: data-space output ───────────────

    def run_data(self, data: np.ndarray) -> np.ndarray:
        """Encode → BMethod → decode → data-space output.

        For adapter-mediated BMethods (e.g. ``ActivationPatch``):
        encodes *data*, applies the learned manipulation, decodes
        back to data space, and returns metric-ready arrays.

        Parameters
        ----------
        data : np.ndarray
            Input data in data space, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Manipulated data in data space.

        Raises
        ------
        RuntimeError
            If no adapter was provided at construction.
        RuntimeError
            If the BMethod is not fitted (checked by the method
            itself).
        """
        if self._adapter is None:
            msg = (
                "No adapter provided — cannot run data-space pipeline. "
                "Provide a FlatBatchDecodableAdapter at construction."
            )
            raise RuntimeError(msg)
        # BMethod.__call__ handles encode → manipulate → decode internally
        # (e.g. ActivationPatch.__call__(input_data)).
        # __call__ is deliberately NOT part of the BMethod Protocol because
        # signatures differ across instances (see b_protocols.py docstring).
        return self._method(data)  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]

    # ── Latent-only story: trajectory output ────────────────────

    def run_trajectory(
        self,
        trajectory: Trajectory,
        **kwargs: object,
    ) -> np.ndarray | Trajectory:
        """Apply the BMethod to every point in a trajectory.

        For latent-only BMethods (e.g. ``SteeringVector``,
        ``Lerp``): delegates to ``method.apply_trajectory`` which
        returns a new ``Trajectory``.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory of latent points.
        **kwargs : object
            Method-specific keyword arguments forwarded to
            ``apply_trajectory`` (e.g. ``strength`` for
            ``SteeringVector``, ``other`` + ``t`` or ``n_steps``
            for ``Lerp``).

        Returns
        -------
        Trajectory
            A new ``Trajectory`` with the manipulation applied.

        Raises
        ------
        RuntimeError
            If the BMethod is not fitted (checked by the method
            itself).
        """
        # Latent-only BMethods (Lerp, SteeringVector) return Trajectory.
        # ActivationPatch returns np.ndarray — use run_data in that case.
        return self._method.apply_trajectory(trajectory, **kwargs)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    # ── Combined story: fit → apply_data → data-space output ───

    def fit_run_data(
        self,
        fit_args: tuple[object, ...] = (),
        fit_kwargs: dict[str, object] | None = None,
        data: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convenience: fit then run_data in one call.

        This is a convenience for interactive use. It supports the
        adapter-mediated story (fit a patch, apply to held-out data).

        Parameters
        ----------
        fit_args : tuple[object, ...]
            Positional args forwarded to ``fit``.
        fit_kwargs : dict[str, object] | None
            Keyword args forwarded to ``fit``.
        data : np.ndarray | None
            Input data for ``run_data``. If ``None``, only fit is
            performed (returns the empty array).

        Returns
        -------
        np.ndarray
            Manipulated data if *data* is provided, else empty array.
        """
        self.fit(*fit_args, **(fit_kwargs or {}))
        if data is not None:
            return self.run_data(data)
        return np.array([])

    # ── Combined story: fit → run_trajectory → trajectory output ─

    def fit_run_trajectory(
        self,
        fit_args: tuple[object, ...] = (),
        fit_kwargs: dict[str, object] | None = None,
        trajectory: Trajectory | None = None,
        **apply_kwargs: object,
    ) -> np.ndarray | Trajectory | None:
        """Convenience: fit then run_trajectory in one call.

        Supports the latent-only story.

        Parameters
        ----------
        fit_args : tuple[object, ...]
            Positional args forwarded to ``fit``.
        fit_kwargs : dict[str, object] | None
            Keyword args forwarded to ``fit``.
        trajectory : Trajectory | None
            Input trajectory. If ``None``, only fit is performed.
        **apply_kwargs : object
            Keyword args forwarded to ``apply_trajectory``.

        Returns
        -------
        Trajectory | None
            Result trajectory if *trajectory* is provided, else None.
        """
        self.fit(*fit_args, **(fit_kwargs or {}))
        if trajectory is not None:
            return self.run_trajectory(trajectory, **apply_kwargs)
        return None


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


# ── Config-backed construction (ManipulationPipeline) ────────────────


class ManipulationPipelineSpec(BaseModel):
    """Config spec for building a ``ManipulationPipeline`` from registry specs.

    Wraps a Layer B method ``ObjectSpec`` and an optional adapter
    ``ObjectSpec`` resolved through the registry by
    ``build_manipulation_pipeline_from_config``.

    This is a composition layer on top of the Sprint 18 config machinery.
    It does **not** introduce a new registry kind — it composes existing
    registry objects. The adapter spec is optional because latent-only
    stories (Lerp, SteeringVector) do not need one.

    Parameters
    ----------
    method : ObjectSpec
        Config spec for the Layer B method (kind must be
        ``"method_b"``). May contain nested ``ObjectSpec`` for
        adapter references (e.g. ``ActivationPatch`` with nested
        VAE spec).
    adapter : ObjectSpec | None, optional
        Optional config spec for the adapter (kind must be
        ``"adapter"``). Required for data-space stories when the
        method's nested adapter is not sufficient. Defaults to
        ``None``.

    Examples
    --------
    >>> spec = ManipulationPipelineSpec(
    ...     method={"kind": "method_b", "name": "steering_vector"},
    ... )
    >>> pipeline = build_manipulation_pipeline_from_config(spec)
    """

    method: ObjectSpec = Field(..., description="Config spec for the Layer B method")
    adapter: ObjectSpec | None = Field(
        default=None,
        description="Optional config spec for the adapter",
    )


def build_manipulation_pipeline_from_config(
    spec: ManipulationPipelineSpec,
    *,
    registry: Registry | None = None,
) -> ManipulationPipeline:
    """Build a ``ManipulationPipeline`` from a ``ManipulationPipelineSpec``.

    Uses the Sprint 18 config machinery (``build_from_config``) to
    resolve the method (and optional adapter) specs through the
    registry, then composes them into a pipeline.

    Parameters
    ----------
    spec : ManipulationPipelineSpec
        Config spec containing ``method`` and optional ``adapter``
        ``ObjectSpec`` instances (or compatible dicts).
    registry : Registry | None
        Registry for resolving ``ObjectSpec`` entries. Defaults to
        ``GLOBAL_REGISTRY``.

    Returns
    -------
    ManipulationPipeline
        The composed pipeline with resolved method and optional adapter.

    Raises
    ------
    KeyError
        If a spec name is not found in the registry.
    ValueError
        If a spec kind does not match the registered entry's kind.
    TypeError
        If a factory cannot be called with the resolved parameters.
    """
    method = build_from_config(spec.method, registry=registry)  # pyright: ignore[reportUnknownVariableType]
    adapter: object | None = None
    if spec.adapter is not None:
        adapter = build_from_config(spec.adapter, registry=registry)
    return ManipulationPipeline(
        method=method,  # pyright: ignore[reportUnknownArgumentType]
        adapter=adapter,  # pyright: ignore[reportArgumentType]
    )
