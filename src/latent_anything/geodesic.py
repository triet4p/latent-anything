"""Density-penalized geodesic path interpolation.

This module realizes one non-Euclidean interpolation method backed by a
**learned density oracle** (the tractable half of the Sprint 50 choice
between density-penalized paths and a decoder pullback metric). The latent
space is treated as a Riemannian manifold whose metric is the inverse of the
learned data density: moving through low-density regions is expensive, so the
geodesic bends toward on-manifold, high-density regions instead of cutting
across them.

When is this justified?
-----------------------
- Use the density geodesic when the data really is curved — the decoded
  straight line departs the manifold and passes through low-density regions
  (blurry / implausible interpolations).
- Prefer plain lerp when the space is flat or the endpoints are close enough
  that the chord stays on-manifold; optimization costs ``max_iter * n_points``
  density evaluations and buys nothing on flat data.
- Prefer slerp on ``unit_norm`` geometry (closed-form spherical geodesic) and
  the anisotropic metric geodesic under a constant covariance — both are
  cheaper than numeric path optimization because they are exact.

The path optimization is deterministic (lerp initialization), bounded
(``max_iter`` and fixed ``n_points``), reports its own convergence, and
returns the full path with length plus density/reconstruction diagnostics.

Pure math lives in :mod:`latent_anything.geometry` (energy, gradient, the
bounded gradient-descent loop); this module owns the configuration, the typed
result values, and the config-driven :class:`DensityGeodesic` entry point.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, Field

from latent_anything.density import GaussianMixtureDensity
from latent_anything.geometry import density_path_length as _density_path_length
from latent_anything.geometry import optimize_density_path as _optimize_density_path
from latent_anything.runtime.cache import CacheKey, InMemoryCache, hash_array
from latent_anything.runtime.profiling import RuntimeProfiler


def _immutable_array(array: np.ndarray) -> np.ndarray:
    """Copy an array onto a read-only bytes buffer that cannot be re-enabled."""
    immutable = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


def _freeze_provenance(value: Any) -> Any:
    """Return a recursively immutable defensive copy of a provenance value."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        frozen: dict[str, Any] = {str(key): _freeze_provenance(item) for key, item in mapping.items()}
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        return _immutable_array(np.array(cast(Any, value), copy=True))
    if isinstance(value, (list, tuple)):
        items = cast(list[Any] | tuple[Any, ...], value)
        return tuple(_freeze_provenance(item) for item in items)
    if isinstance(value, (set, frozenset)):
        items = cast(set[Any] | frozenset[Any], value)
        return frozenset(_freeze_provenance(item) for item in items)
    return deepcopy(value)


class GeodesicConfig(BaseModel):
    """Validated, deterministic configuration for density-penalized paths.

    Parameters
    ----------
    n_points : int
        Number of path points (including endpoints). Bounds compute and the
        discretization resolution.
    max_iter : int
        Maximum gradient-descent iterations. Bounds compute.
    step_size : float
        Initial gradient-descent step; a backtracking line search halves it
        until the energy decreases.
    tol : float
        Gradient-norm convergence tolerance for the interior points.
    density_exponent : float
        Exponent ``alpha`` in the metric ``g(z) = exp(-alpha * log p(z))``.
        ``0`` recovers the lerp path; larger values penalize crossing
        low-density regions more strongly.
    """

    n_points: int = Field(default=16, ge=3, le=256)
    max_iter: int = Field(default=200, ge=1, le=100_000)
    step_size: float = Field(default=0.1, gt=0)
    tol: float = Field(default=1e-6, gt=0)
    density_exponent: float = Field(default=1.0, ge=0)


@dataclass(frozen=True)
class PathOptimizationStatus:
    """Outcome of the bounded path optimization.

    Attributes
    ----------
    converged : bool
        Whether the gradient norm fell below ``tol`` (or the line search
        reached a flat-energy local minimum).
    n_iterations : int
        Number of gradient-descent iterations executed.
    initial_energy : float
        Energy of the deterministic lerp initialization.
    final_energy : float
        Energy of the returned path.
    message : str
        Human-readable convergence description.
    """

    converged: bool
    n_iterations: int
    initial_energy: float
    final_energy: float
    message: str


@dataclass(frozen=True)
class GeodesicPath:
    """A full density-penalized geodesic path with diagnostics.

    Attributes
    ----------
    path : np.ndarray
        Optimized path, shape ``(n_points, dim)``, rows are on-manifold latent
        points from ``endpoint_a`` to ``endpoint_b``.
    endpoint_a, endpoint_b : np.ndarray
        The fixed interpolation endpoints (copies of the inputs).
    length : float
        Density-penalized arc length ``sum_i sqrt(g(z_i)) * ||z_{i+1}-z_i||``.
    euclidean_length : float
        Plain summed segment length of the same path.
    log_density : np.ndarray
        Log-density of every path point, shape ``(n_points,)``.
    min_log_density : float
        Minimum log-density along the path (its most off-manifold point).
    mean_log_density : float
        Mean log-density along the path (the on-manifoldness diagnostic).
    decoded : np.ndarray | None
        ``decoder(path)`` when a decoder was attached, else ``None``.
    reconstruction_error : float | None
        Mean pairwise decoded distance between consecutive path points (decoded
        total variation / smoothness) when a decoder was attached, else
        ``None``. Lower means smoother, more coherent decoded transitions.
    status : PathOptimizationStatus
        Convergence and energy reporting for the optimization.
    source_representation_identity : str | None
        Representation identity the density oracle was fitted on.
    provenance : dict[str, Any]
        Free-form provenance (config, density, decoder, endpoints metadata).
    """

    path: np.ndarray
    endpoint_a: np.ndarray
    endpoint_b: np.ndarray
    length: float
    euclidean_length: float
    log_density: np.ndarray
    min_log_density: float
    mean_log_density: float
    decoded: np.ndarray | None
    reconstruction_error: float | None
    status: PathOptimizationStatus
    source_representation_identity: str | None
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate and defensively own every part of the result."""
        path = np.array(self.path, dtype=np.float64, copy=True)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 1:
            msg = f"GeodesicPath.path must be (n_points, dim) with n_points >= 2, got {path.shape}"
            raise ValueError(msg)
        if not np.isfinite(path).all():
            raise ValueError("GeodesicPath.path must contain only finite values")
        endpoint_a = np.array(self.endpoint_a, dtype=np.float64, copy=True)
        endpoint_b = np.array(self.endpoint_b, dtype=np.float64, copy=True)
        if endpoint_a.shape != (path.shape[1],) or endpoint_b.shape != (path.shape[1],):
            msg = f"endpoints must match path dim {path.shape[1]}, got {endpoint_a.shape} and {endpoint_b.shape}"
            raise ValueError(msg)
        if not np.allclose(path[0], endpoint_a) or not np.allclose(path[-1], endpoint_b):
            raise ValueError("GeodesicPath must keep its endpoints fixed")
        log_density = np.array(self.log_density, dtype=np.float64, copy=True)
        if log_density.shape != (path.shape[0],):
            msg = f"log_density must match path length {path.shape[0]}, got {log_density.shape}"
            raise ValueError(msg)
        if not np.isfinite(log_density).all():
            raise ValueError("GeodesicPath.log_density must contain only finite values")
        raw_length: Any = cast(Any, self.length)
        length_value = float(np.asarray(raw_length).item())  # type: ignore[arg-type]
        if (
            isinstance(raw_length, bool)
            or not isinstance(raw_length, (int, float, np.integer, np.floating))
            or not np.isfinite(length_value)
        ):
            raise ValueError(f"GeodesicPath.length must be finite, got {self.length!r}")
        if self.min_log_density > self.mean_log_density:
            raise ValueError("min_log_density cannot exceed mean_log_density")
        if self.reconstruction_error is not None and not np.isfinite(self.reconstruction_error):
            raise ValueError("reconstruction_error must be finite when provided")
        decoded = self.decoded
        if decoded is not None and np.asarray(decoded).shape[0] != path.shape[0]:
            msg = f"decoded must have {path.shape[0]} rows, got {np.asarray(decoded).shape}"
            raise ValueError(msg)
        provenance = cast(Any, self.provenance)
        if not isinstance(provenance, Mapping):
            raise ValueError("GeodesicPath.provenance must be a mapping")

        object.__setattr__(self, "path", _immutable_array(path))
        object.__setattr__(self, "endpoint_a", _immutable_array(endpoint_a))
        object.__setattr__(self, "endpoint_b", _immutable_array(endpoint_b))
        object.__setattr__(self, "log_density", _immutable_array(log_density))
        if decoded is not None:
            object.__setattr__(self, "decoded", _immutable_array(np.array(decoded, dtype=np.float64, copy=True)))
        object.__setattr__(self, "provenance", _freeze_provenance(provenance))

    @property
    def n_points(self) -> int:
        """Number of path points."""
        return int(self.path.shape[0])

    @property
    def dim(self) -> int:
        """Dimensionality of each latent point."""
        return int(self.path.shape[1])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly metadata payload (arrays as nested lists)."""
        return {
            "path": self.path.tolist(),
            "endpoint_a": self.endpoint_a.tolist(),
            "endpoint_b": self.endpoint_b.tolist(),
            "length": self.length,
            "euclidean_length": self.euclidean_length,
            "log_density": self.log_density.tolist(),
            "min_log_density": self.min_log_density,
            "mean_log_density": self.mean_log_density,
            "decoded": self.decoded.tolist() if self.decoded is not None else None,
            "reconstruction_error": self.reconstruction_error,
            "status": {
                "converged": self.status.converged,
                "n_iterations": self.status.n_iterations,
                "initial_energy": self.status.initial_energy,
                "final_energy": self.status.final_energy,
                "message": self.status.message,
            },
            "source_representation_identity": self.source_representation_identity,
            "provenance": _freeze_provenance(self.provenance),
        }


def _decode_rows(decoder: Callable[[np.ndarray], np.ndarray], path: np.ndarray) -> np.ndarray:
    """Decode every path point and return a ``(n_points, ...)`` array."""
    values = np.asarray(decoder(path), dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("decoder must return a numpy array")
    if values.shape[0] != path.shape[0]:
        msg = f"decoder must return one row per path point, got {values.shape}"
        raise ValueError(msg)
    return values


def _decoded_total_variation(decoded: np.ndarray) -> float:
    """Mean pairwise distance between consecutive decoded rows."""
    flat = decoded.reshape(decoded.shape[0], -1)
    if flat.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(flat[1:] - flat[:-1], axis=1)
    return float(np.asarray(distances).mean())  # type: ignore[arg-type]


class DensityGeodesic:
    """Config-driven density-penalized geodesic path optimization.

    Wrap a fitted density estimator (via :meth:`from_gmm_density`) or attach a
    ``log_density`` / ``log_density_gradient`` oracle directly, then compute the
    geodesic between two latent points::

        from latent_anything.geodesic import DensityGeodesic
        from latent_anything.density import GaussianMixtureDensity

        density = GaussianMixtureDensity().fit(train_latents, source_representation_identity="vae/digits")
        geodesic = DensityGeodesic.from_gmm_density(density)
        path = geodesic.optimize(a, b)          # GeodesicPath with diagnostics
        midpoint = geodesic.interpolate(a, b, 0.5)

    Construction from config (registry-driven) is supported::

        from latent_anything.config import build_from_dict

        geodesic = build_from_dict({"kind": "intervention", "name": "density_geodesic",
                                    "params": {"n_points": 24, "density_exponent": 1.0}})
        geodesic.from_gmm_density(density)      # attach the oracle afterwards

    Parameters
    ----------
    config : GeodesicConfig | None
        Optimization configuration; defaults to :class:`GeodesicConfig`.
    """

    def __init__(self, config: GeodesicConfig | None = None, **kwargs: Any) -> None:
        if kwargs:
            self._config = GeodesicConfig(**kwargs)
        else:
            self._config = config if config is not None else GeodesicConfig()
        self._log_density: Callable[[np.ndarray], float] | None = None
        self._log_density_gradient: Callable[[np.ndarray], np.ndarray] | None = None
        self._decoder: Callable[[np.ndarray], np.ndarray] | None = None
        self._source_identity: str | None = None
        self._state_digest = ""

    @classmethod
    def from_gmm_density(
        cls,
        estimator: GaussianMixtureDensity,
        *,
        config: GeodesicConfig | None = None,
        decoder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> DensityGeodesic:
        """Build a geodesic around a fitted :class:`GaussianMixtureDensity`.

        The estimator's identity, log-density oracle, analytic gradient, and a
        stable parameter digest (used in cache keys) are all captured.
        """
        if not estimator.is_fitted:
            raise RuntimeError("GaussianMixtureDensity has not been fitted")
        geodesic = cls(config=config)
        geodesic.attach_density(
            estimator.log_density,
            log_density_gradient=estimator.log_density_gradient,
            source_representation_identity=estimator.source_representation_identity,
            state_digest=estimator.state_digest(),
            decoder=decoder,
        )
        return geodesic

    def attach_density(
        self,
        log_density: Callable[[np.ndarray], float],
        *,
        log_density_gradient: Callable[[np.ndarray], np.ndarray] | None = None,
        decoder: Callable[[np.ndarray], np.ndarray] | None = None,
        source_representation_identity: str | None = None,
        state_digest: str = "",
    ) -> DensityGeodesic:
        """Attach a ``log_density`` oracle (and optional gradient/decoder).

        If no analytic gradient is given, a finite-difference gradient is used.
        ``state_digest`` is a stable hash of the oracle's fitted parameters and
        is incorporated into cache keys so a refitted density never reuses a
        stale cached path.
        """
        self._log_density = log_density
        self._log_density_gradient = (
            log_density_gradient if log_density_gradient is not None else self._finite_difference_gradient
        )
        self._decoder = decoder
        self._source_identity = source_representation_identity
        self._state_digest = state_digest
        return self

    def _finite_difference_gradient(self, point: np.ndarray) -> np.ndarray:
        """Central-difference gradient of the attached log-density oracle."""
        if self._log_density is None:
            raise RuntimeError("no log_density oracle attached")
        eps = 1e-5
        value = np.asarray(point, dtype=np.float64)
        gradient = np.zeros_like(value)
        for index in range(value.shape[0]):
            step = np.zeros_like(value)
            step[index] = eps
            gradient[index] = (float(self._log_density(value + step)) - float(self._log_density(value - step))) / (
                2.0 * eps
            )
        return gradient

    @property
    def config(self) -> GeodesicConfig:
        """Return the optimization configuration."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Whether a log-density oracle has been attached."""
        return self._log_density is not None

    @property
    def source_representation_identity(self) -> str | None:
        """Return the representation identity of the attached density."""
        return self._source_identity

    def _require_oracle(self) -> Callable[[np.ndarray], float]:
        if self._log_density is None or self._log_density_gradient is None:
            msg = "DensityGeodesic has no density oracle; call attach_density() or from_gmm_density()"
            raise RuntimeError(msg)
        return self._log_density

    def _validate_endpoints(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point_a = np.asarray(a, dtype=np.float64)
        point_b = np.asarray(b, dtype=np.float64)
        if point_a.ndim != 1 or point_b.ndim != 1:
            msg = f"endpoints must be flat (dim,) arrays, got {point_a.shape} and {point_b.shape}"
            raise ValueError(msg)
        if point_a.shape != point_b.shape:
            msg = f"endpoints must share shape, got {point_a.shape} and {point_b.shape}"
            raise ValueError(msg)
        if not np.isfinite(point_a).all() or not np.isfinite(point_b).all():
            raise ValueError("endpoints must contain only finite values")
        return point_a, point_b

    def _cache_key(self, a: np.ndarray, b: np.ndarray) -> CacheKey:
        """Build a stable cache key over endpoints, config, and oracle state."""
        config_hash = sha256(
            self._config.model_dump_json().encode("utf-8"),
        ).hexdigest()
        state_hash = sha256(self._state_digest.encode("utf-8")).hexdigest()
        data = np.stack([np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)])
        return CacheKey(
            namespace="geodesic",
            operation="density_path",
            component_name="latent_anything.geodesic.DensityGeodesic",
            config_hash=config_hash,
            state_hash=state_hash,
            data_hash=hash_array(data),
            framework_version=None,
        )

    def optimize(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        cache: InMemoryCache | None = None,
        profiler: RuntimeProfiler | None = None,
    ) -> GeodesicPath:
        """Optimize and return the density-penalized geodesic from ``a`` to ``b``.

        Parameters
        ----------
        a, b : np.ndarray
            Flat ``(dim,)`` endpoints in the fitted representation.
        cache : InMemoryCache | None
            Optional in-memory cache. Path optimization is expensive, so
            identical (endpoints, config, oracle state) calls skip re-optimizing
            and reuse the cached path array.
        profiler : RuntimeProfiler | None
            Optional profiler; records the ``cache`` lookup and ``method``
            optimization stages.

        Returns
        -------
        GeodesicPath
            The full path with length, density/reconstruction diagnostics, and
            optimization status.
        """
        self._require_oracle()
        point_a, point_b = self._validate_endpoints(a, b)

        key = self._cache_key(point_a, point_b)
        if cache is not None:
            cached_path: np.ndarray | None = (
                profiler.measure("cache", lambda: cache.get(key)) if profiler is not None else cache.get(key)
            )
        else:
            cached_path = None

        if cached_path is not None:
            path = cached_path
            status = PathOptimizationStatus(
                converged=True,
                n_iterations=0,
                initial_energy=0.0,
                final_energy=0.0,
                message="served from cache",
            )
        else:
            log_density = self._require_oracle()
            gradient_oracle = self._log_density_gradient
            if gradient_oracle is None:
                msg = "DensityGeodesic has no log_density_gradient oracle"
                raise RuntimeError(msg)
            cfg = self._config

            def _run() -> tuple[np.ndarray, float, float, int, bool, str]:
                return _optimize_density_path(
                    point_a,
                    point_b,
                    log_density=log_density,
                    log_density_gradient=gradient_oracle,
                    n_points=cfg.n_points,
                    max_iter=cfg.max_iter,
                    step_size=cfg.step_size,
                    tol=cfg.tol,
                    exponent=cfg.density_exponent,
                )

            if profiler is not None:
                path, initial_energy, final_energy, n_iterations, converged, message = profiler.measure("method", _run)
            else:
                path, initial_energy, final_energy, n_iterations, converged, message = _run()
            status = PathOptimizationStatus(
                converged=converged,
                n_iterations=n_iterations,
                initial_energy=initial_energy,
                final_energy=final_energy,
                message=message,
            )
            if cache is not None:
                cache.set(key, path)

        assert self._log_density is not None
        log_density_values = np.asarray([float(self._log_density(point)) for point in path], dtype=np.float64)
        log_ref = float(np.max(log_density_values))
        length, euclidean_length = _density_path_length(
            path,
            self._log_density,
            exponent=self._config.density_exponent,
            log_ref=log_ref,
        )

        decoded: np.ndarray | None = None
        reconstruction_error: float | None = None
        if self._decoder is not None:
            decoded = _decode_rows(self._decoder, path)
            reconstruction_error = _decoded_total_variation(decoded)

        provenance = {
            "config": self._config.model_dump(mode="json"),
            "density_state_digest": self._state_digest,
            "decoder": self._decoder is not None,
            "source_representation_identity": self._source_identity,
        }
        return GeodesicPath(
            path=path,
            endpoint_a=point_a,
            endpoint_b=point_b,
            length=length,
            euclidean_length=euclidean_length,
            log_density=log_density_values,
            min_log_density=float(np.min(log_density_values)),
            mean_log_density=float(np.mean(log_density_values)),
            decoded=decoded,
            reconstruction_error=reconstruction_error,
            status=status,
            source_representation_identity=self._source_identity,
            provenance=provenance,
        )

    def interpolate(
        self,
        a: np.ndarray,
        b: np.ndarray,
        t: float,
        *,
        cache: InMemoryCache | None = None,
        profiler: RuntimeProfiler | None = None,
    ) -> np.ndarray:
        """Return the geodesic point at parameter ``t`` in ``[0, 1]``.

        Equivalent to ``optimize(a, b).path`` sampled at the fractional index
        ``t * (n_points - 1)`` with linear interpolation between path rows.
        """
        if not 0.0 <= t <= 1.0:
            msg = f"t must be in [0, 1], got {t}"
            raise ValueError(msg)
        result = self.optimize(a, b, cache=cache, profiler=profiler)
        scaled = t * (result.n_points - 1)
        lower = int(np.floor(scaled))
        upper = min(int(np.ceil(scaled)), result.n_points - 1)
        if lower == upper:
            return result.path[lower].copy()
        fraction = scaled - lower
        return ((1.0 - fraction) * result.path[lower] + fraction * result.path[upper]).copy()
