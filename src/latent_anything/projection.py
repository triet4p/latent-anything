"""Fitted orthonormal subspace projection for latent manipulations.

Subspace projection decomposes a latent point ``z`` into two orthogonal parts,
``z = P z + (I - P) z``, where ``P = U U^T`` is the orthogonal projection onto
the span of an orthonormal basis ``U``. The concept component ``P z`` holds the
semantic directions in the subspace; the residual ``(I - P) z`` holds
everything else and is what "concept removal" keeps.

This module owns the **stateful** part of that geometry:

- :class:`OrthonormalSubspace` — an immutable fitted orthonormal basis bound to
  a single representation identity, carrying its *origin* (how the basis was
  derived) so different basis families are never silently interchangeable.
- :class:`SubspaceProjectionConfig` / :class:`SubspaceProjection` — the
  config-driven, registry-constructable transformation entry point with
  ``project``, ``remove``, ``coverage``, and ``transfer`` operations that
  consume immutable :class:`LatentValue` inputs and return new immutable
  values carrying operation/provenance metadata.

Pure math (basis validation, orthonormalization, ``P z`` / ``(I - P) z``,
coverage, subspace alignment) lives in ``geometry.py``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, Field

from latent_anything.geometry import (
    concept_coverage as _concept_coverage,
)
from latent_anything.geometry import (
    orthonormalize_directions as _orthonormalize_directions,
)
from latent_anything.geometry import (
    project_point as _project_point,
)
from latent_anything.geometry import (
    remove_point as _remove_point,
)
from latent_anything.geometry import (
    validate_orthonormal_basis as _validate_orthonormal_basis,
)
from latent_anything.latent_value import LatentValue, coordinate_identity

# Basis origins: how the orthonormal basis was derived. These families measure
# different things (variance, discriminability, concept alignment) and must not
# be treated as interchangeable.
_ORIGINS = frozenset({"explicit", "pca", "probe", "concept"})


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


def _thaw_provenance(value: Any) -> Any:
    """Convert recursively frozen provenance into JSON-friendly containers."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): _thaw_provenance(item) for key, item in mapping.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, frozenset)):
        items = cast(tuple[Any, ...] | frozenset[Any], value)
        return [_thaw_provenance(item) for item in items]
    return value


class SubspaceProjectionConfig(BaseModel):
    """Validated, deterministic configuration for a :class:`SubspaceProjection`.

    Parameters
    ----------
    n_basis : int | None
        Number of orthonormal basis vectors to keep. ``None`` keeps every
        fitted direction.
    """

    n_basis: int | None = Field(
        default=None,
        ge=1,
        description="Number of orthonormal basis vectors to keep (default: all)",
    )


@dataclass(frozen=True)
class OrthonormalSubspace:
    """Immutable fitted orthonormal subspace bound to one representation identity.

    Attributes
    ----------
    basis : np.ndarray
        Orthonormal basis, shape ``(dim, n_basis)``, columns spanning the
        subspace. The array is owned defensively and cannot be mutated.
    n_basis : int
        Number of orthonormal basis vectors (``1 <= n_basis < dim``).
    source_representation_identity : str
        Coordinate-system identity this subspace was fitted/derived for.
        Applying the subspace to a latent value with a different identity is a
        caller error and is rejected by :class:`SubspaceProjection`.
    origin : str
        How the basis was derived: ``"pca"``, ``"probe"``, ``"concept"``, or
        ``"explicit"``. Different origins measure different things and must not
        be treated as interchangeable.
    provenance : dict[str, Any]
        Free-form provenance (fitting data, seed, source component).
    """

    basis: np.ndarray
    source_representation_identity: str
    origin: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate and defensively own every part of the fitted subspace."""
        basis = np.array(self.basis, dtype=np.float64, copy=True)
        if basis.ndim != 2 or basis.shape[0] < 2 or basis.shape[1] < 1:
            msg = f"OrthonormalSubspace expects a 2D basis with at least 2 rows and 1 column, got {basis.shape}"
            raise ValueError(msg)
        _validate_orthonormal_basis(basis, dim=basis.shape[0])
        identity = cast(Any, self.source_representation_identity)
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("OrthonormalSubspace source_representation_identity must be non-empty")
        origin = cast(Any, self.origin)
        if not isinstance(origin, str) or origin not in _ORIGINS:
            msg = f"OrthonormalSubspace origin must be one of {sorted(_ORIGINS)}, got {origin!r}"
            raise ValueError(msg)
        provenance = cast(Any, self.provenance)
        if not isinstance(provenance, Mapping):
            raise ValueError("OrthonormalSubspace provenance must be a mapping")

        object.__setattr__(self, "basis", _immutable_array(basis))
        object.__setattr__(self, "provenance", _freeze_provenance(provenance))

    @property
    def dim(self) -> int:
        """Dimensionality of the ambient space the subspace lives in."""
        return int(self.basis.shape[0])

    @property
    def n_basis(self) -> int:
        """Number of orthonormal basis vectors spanning the subspace."""
        return int(self.basis.shape[1])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation (arrays as nested lists)."""
        return {
            "basis": self.basis.tolist(),
            "source_representation_identity": self.source_representation_identity,
            "origin": self.origin,
            "provenance": _thaw_provenance(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrthonormalSubspace:
        """Rebuild a :class:`OrthonormalSubspace` from a :meth:`to_dict` payload."""
        values = dict(data)
        try:
            raw_identity = values["source_representation_identity"]
            if not isinstance(raw_identity, str):
                raise TypeError("source_representation_identity must be a string")
            raw_origin = values["origin"]
            if not isinstance(raw_origin, str):
                raise TypeError("origin must be a string")
            basis = np.asarray(values["basis"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"OrthonormalSubspace.from_dict missing or malformed field: {exc}"
            raise ValueError(msg) from exc
        provenance = values.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("OrthonormalSubspace provenance must be a mapping")
        provenance_mapping = cast(Mapping[str, Any], provenance)
        return cls(basis, raw_identity, raw_origin, provenance_mapping)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Serialize to a portable ``.npz`` checkpoint with JSON provenance."""
        payload = self.to_dict()
        basis = payload.pop("basis")
        np.savez(
            path,
            basis=np.asarray(basis, dtype=np.float64),
            metadata_json=json.dumps(payload),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> OrthonormalSubspace:
        """Load a :class:`OrthonormalSubspace` from a checkpoint written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:  # pyright: ignore[reportUnknownMemberType]
            basis = np.asarray(data["basis"], dtype=np.float64)
            metadata_raw = data["metadata_json"].item()
            if not isinstance(metadata_raw, str):
                raise ValueError(f"checkpoint {path} has no metadata_json string")
            metadata = json.loads(metadata_raw)
        metadata["basis"] = basis.tolist()
        return cls.from_dict(metadata)

    # ── Derivation helpers (basis families must keep their origin) ────

    @classmethod
    def from_basis(
        cls,
        basis: np.ndarray,
        *,
        source_representation_identity: str,
        origin: str = "explicit",
        provenance: dict[str, Any] | None = None,
    ) -> OrthonormalSubspace:
        """Create a subspace from an already-orthonormal ``(dim, n_basis)`` basis."""
        return cls(
            np.asarray(basis, dtype=np.float64),
            source_representation_identity,
            origin,
            dict(provenance or {}),
        )

    @classmethod
    def from_directions(
        cls,
        directions: np.ndarray,
        *,
        source_representation_identity: str,
        origin: str = "explicit",
        provenance: dict[str, Any] | None = None,
    ) -> OrthonormalSubspace:
        """Create a subspace by orthonormalizing candidate directions (rows).

        ``directions`` is a 2D array whose rows are candidate directions (e.g.
        stacked probe coefficients or concept directions). The columns of the
        result are an orthonormal basis of their span.
        """
        rows = np.asarray(directions, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[np.newaxis, :]
        if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 2:
            msg = f"from_directions expects 1D or 2D directions in at least 2 dimensions, got {rows.shape}"
            raise ValueError(msg)
        basis = _orthonormalize_directions(rows.T)
        return cls(basis, source_representation_identity, origin, dict(provenance or {}))

    @classmethod
    def from_pca(
        cls,
        pca: Any,
        *,
        source_representation_identity: str,
        n_components: int | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> OrthonormalSubspace:
        """Create a ``"pca"``-origin subspace from a fitted PCA components matrix.

        ``pca`` must expose a ``components_`` attribute of shape
        ``(n_components, n_features)`` (as produced by ``PCA`` / sklearn).
        The top ``n_components`` principal directions form the basis.
        """
        components_raw = getattr(pca, "components_", None)
        if components_raw is None or not isinstance(components_raw, np.ndarray) or components_raw.ndim != 2:
            msg = "from_pca expects an object exposing a 2D components_ array"
            raise ValueError(msg)
        components = np.asarray(components_raw, dtype=np.float64)
        keep = n_components if n_components is not None else int(components.shape[0])
        if keep < 1 or keep > components.shape[0]:
            msg = f"n_components must be in [1, {components.shape[0]}], got {n_components}"
            raise ValueError(msg)
        basis = components[:keep, :].T
        return cls.from_basis(
            basis,
            source_representation_identity=source_representation_identity,
            origin="pca",
            provenance=provenance,
        )

    @classmethod
    def from_probe_coefficients(
        cls,
        coefficients: np.ndarray,
        *,
        source_representation_identity: str,
        provenance: dict[str, Any] | None = None,
    ) -> OrthonormalSubspace:
        """Create a ``"probe"``-origin subspace from linear-probe coefficients.

        ``coefficients`` is either a ``(n_features,)`` binary direction or a
        ``(n_classes, n_features)`` multiclass coefficient matrix. The rows are
        orthonormalized into the subspace basis.
        """
        return cls.from_directions(
            coefficients,
            source_representation_identity=source_representation_identity,
            origin="probe",
            provenance=provenance,
        )

    @classmethod
    def from_concept_direction(
        cls,
        direction: np.ndarray,
        *,
        source_representation_identity: str,
        provenance: dict[str, Any] | None = None,
    ) -> OrthonormalSubspace:
        """Create a ``"concept"``-origin one-dimensional subspace from a direction."""
        vector = np.asarray(direction, dtype=np.float64)
        if vector.ndim != 1 or vector.size < 2:
            msg = f"from_concept_direction expects a 1D direction, got shape {vector.shape}"
            raise ValueError(msg)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-15:
            raise ValueError("concept direction must be non-zero")
        basis = (vector / norm)[np.newaxis, :].T
        return cls.from_basis(
            basis,
            source_representation_identity=source_representation_identity,
            origin="concept",
            provenance=provenance,
        )

    @classmethod
    def fit_pca(
        cls,
        data: np.ndarray,
        n_components: int,
        *,
        source_representation_identity: str,
        provenance: dict[str, Any] | None = None,
    ) -> OrthonormalSubspace:
        """Fit a ``"pca"``-origin subspace by running PCA on *data*."""
        from sklearn.decomposition import PCA as _SKLearnPCA  # noqa: N811  # pyright: ignore[reportMissingTypeStubs]

        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
            msg = f"fit_pca expects 2D data with at least 2 samples and 2 features, got {values.shape}"
            raise ValueError(msg)
        if n_components < 1 or n_components >= values.shape[1]:
            msg = f"n_components must be in [1, {values.shape[1] - 1}], got {n_components}"
            raise ValueError(msg)
        pca = _SKLearnPCA(n_components=n_components)
        pca.fit(values)  # pyright: ignore[reportUnknownMemberType]
        return cls.from_pca(
            pca,  # pyright: ignore[reportUnknownArgumentType]
            source_representation_identity=source_representation_identity,
            n_components=n_components,
            provenance=provenance,
        )


class SubspaceProjection:
    """Config-driven orthonormal subspace projection transformation.

    Wraps :class:`OrthonormalSubspace` so the operation can be constructed via
    ``ObjectSpec`` / ``build_from_config`` and applied to immutable
    :class:`LatentValue` inputs::

        from latent_anything.config import build_from_dict

        proj = build_from_dict({"kind": "intervention", "name": "subspace_projection"})
        proj.fit_pca(latents, n_components=4, source_representation_identity="conv-vae/digits")
        removed = proj.remove(value)          # (I - P) z with provenance metadata
        projected = proj.project(value)       # P z with provenance metadata

    The subspace is bound to one representation identity; applying it to a
    value with a different coordinate-system identity raises ``ValueError``.
    Outputs are new immutable ``LatentValue`` instances whose metadata records
    the operation and a growing provenance chain.

    Parameters
    ----------
    config : SubspaceProjectionConfig | None
        Transformation configuration (e.g. ``n_basis``).
    """

    def __init__(self, config: SubspaceProjectionConfig | None = None, **kwargs: Any) -> None:
        if kwargs:
            self._config = SubspaceProjectionConfig(**kwargs)
        else:
            self._config = config if config is not None else SubspaceProjectionConfig()
        self._subspace: OrthonormalSubspace | None = None

    @classmethod
    def from_subspace(
        cls,
        subspace: OrthonormalSubspace,
        config: SubspaceProjectionConfig | None = None,
    ) -> SubspaceProjection:
        """Build a projection transformation around an already-fitted subspace."""
        projection = cls(config=config)
        projection._subspace = subspace
        return projection

    @property
    def config(self) -> SubspaceProjectionConfig:
        """Return the transformation configuration."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` once a subspace has been attached."""
        return self._subspace is not None

    @property
    def subspace(self) -> OrthonormalSubspace:
        """Return the fitted subspace, or raise a clear fitted-state error."""
        if self._subspace is None:
            msg = (
                "SubspaceProjection has no fitted subspace; call fit_basis()/fit_pca() "
                "with a source_representation_identity or construct via from_subspace()"
            )
            raise RuntimeError(msg)
        return self._subspace

    def fit_basis(
        self,
        basis: np.ndarray,
        *,
        source_representation_identity: str,
        origin: str = "explicit",
        provenance: dict[str, Any] | None = None,
    ) -> SubspaceProjection:
        """Attach an orthonormal basis, optionally trimming to ``config.n_basis``."""
        matrix = np.asarray(basis, dtype=np.float64)
        n_basis = self._config.n_basis if self._config.n_basis is not None else matrix.shape[1]
        if n_basis < 1 or n_basis > matrix.shape[1]:
            msg = f"config.n_basis must be in [1, {matrix.shape[1]}], got {n_basis}"
            raise ValueError(msg)
        self._subspace = OrthonormalSubspace(
            matrix[:, :n_basis],
            source_representation_identity,
            origin,
            dict(provenance or {}),
        )
        return self

    def fit_pca(
        self,
        data: np.ndarray,
        n_components: int,
        *,
        source_representation_identity: str,
        provenance: dict[str, Any] | None = None,
    ) -> SubspaceProjection:
        """Fit a PCA-derived subspace on *data* and attach it to this transformation."""
        keep = self._config.n_basis if self._config.n_basis is not None else n_components
        self._subspace = OrthonormalSubspace.fit_pca(
            data,
            keep,
            source_representation_identity=source_representation_identity,
            provenance=provenance,
        )
        return self

    # ── Validation of value compatibility ─────────────────────────────

    def _validate_value(self, value: LatentValue) -> None:
        """Reject values whose geometry, shape, or identity differs from the subspace."""
        subspace = self.subspace
        if value.space.geometry != "euclidean":
            msg = (
                f"subspace projection requires geometry='euclidean' (a flat vector space), got {value.space.geometry!r}"
            )
            raise ValueError(msg)
        if value.item_shape != (subspace.dim,):
            msg = f"subspace is defined for points of shape {(subspace.dim,)}, got value shape {value.item_shape}"
            raise ValueError(msg)
        identity = coordinate_identity(value.space, value.metadata)
        if identity != subspace.source_representation_identity:
            msg = (
                "subspace was fitted for coordinate system "
                f"{subspace.source_representation_identity!r} but the value declares {identity!r}; "
                "projecting across unrelated coordinate systems is not allowed"
            )
            raise ValueError(msg)

    def _transform(self, value: LatentValue, *, op: str) -> LatentValue:
        """Apply *op* pointwise to every latent point and return a new immutable value."""
        self._validate_value(value)
        subspace = self.subspace
        data = value.to_numpy()
        dim = subspace.dim
        points = data.reshape((-1, dim))
        if op == "project":
            transformed = _project_point(points, subspace.basis)
        else:
            transformed = _remove_point(points, subspace.basis)
        result = transformed.reshape(data.shape)
        provenance: list[Any] = list(cast(Any, value.metadata.get("provenance", ())))
        provenance.append(
            {
                "operation": "subspace_projection",
                "op": op,
                "basis_origin": subspace.origin,
                "n_basis": subspace.n_basis,
                "subspace_identity": subspace.source_representation_identity,
            }
        )
        metadata = {
            **dict(value.metadata),
            "operation": {
                "kind": "subspace_projection",
                "op": op,
                "basis_origin": subspace.origin,
                "n_basis": subspace.n_basis,
                "subspace_identity": subspace.source_representation_identity,
            },
            "provenance": provenance,
        }
        return LatentValue(result, value.space, metadata)

    def project(self, value: LatentValue) -> LatentValue:
        """Return ``P z`` — the component of *value* inside the fitted subspace."""
        return self._transform(value, op="project")

    def remove(self, value: LatentValue) -> LatentValue:
        """Return ``(I - P) z`` — *value* with every subspace component removed."""
        return self._transform(value, op="remove")

    def coverage(self, value: LatentValue) -> float | np.ndarray:
        """Return the fraction of each point's energy inside the subspace.

        Returns a scalar for a single-point value and an array matching the
        batch shape for a batched value.
        """
        self._validate_value(value)
        subspace = self.subspace
        data = value.to_numpy()
        dim = subspace.dim
        points = data.reshape((-1, dim))
        coverages = np.asarray(
            [_concept_coverage(point, subspace.basis) for point in points],
            dtype=np.float64,
        )
        if not value.is_batch:
            return float(coverages[0])
        return coverages.reshape(value.batch_shape)

    def transfer(self, source: LatentValue, target: LatentValue) -> LatentValue:
        """Return ``(I - P) z_target + P z_source`` (concept transfer).

        Keeps the target's content outside the subspace and carries the
        source's concept component into it. Both values must be compatible
        with the fitted subspace.
        """
        self._validate_value(source)
        self._validate_value(target)
        if source.shape != target.shape:
            msg = f"transfer requires equal value shapes, got {source.shape} and {target.shape}"
            raise ValueError(msg)
        subspace = self.subspace
        dim = subspace.dim
        source_data = source.to_numpy().reshape((-1, dim))
        target_data = target.to_numpy().reshape((-1, dim))
        transferred = _remove_point(target_data, subspace.basis) + _project_point(source_data, subspace.basis)
        result = transferred.reshape(target.shape)
        provenance: list[Any] = list(cast(Any, target.metadata.get("provenance", ())))
        provenance.append(
            {
                "operation": "subspace_projection",
                "op": "transfer",
                "basis_origin": subspace.origin,
                "n_basis": subspace.n_basis,
                "subspace_identity": subspace.source_representation_identity,
            }
        )
        metadata = {
            **dict(target.metadata),
            "operation": {
                "kind": "subspace_projection",
                "op": "transfer",
                "basis_origin": subspace.origin,
                "n_basis": subspace.n_basis,
                "subspace_identity": subspace.source_representation_identity,
            },
            "provenance": provenance,
        }
        return LatentValue(result, target.space, metadata)
