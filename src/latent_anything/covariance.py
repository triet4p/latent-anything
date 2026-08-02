"""Fitted covariance geometry contract for anisotropic latent spaces.

A constant anisotropic metric is a positive-definite covariance matrix
learned from data. This module owns the **stateful** part of that geometry:
the fitted ``CovarianceState`` value, its pydantic ``CovarianceConfig``, the
fitting entry point bound to a representation identity, and serialization.
Pure math (validation, Mahalanobis distance, whitening, interpolation) lives
in ``geometry.py``; ``LatentSpace`` is the public facade that dispatches on
``geometry == "anisotropic"``.
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

from latent_anything.geometry import fit_covariance as _fit_covariance
from latent_anything.geometry import validate_covariance


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


def _immutable_array(array: np.ndarray) -> np.ndarray:
    """Copy an array onto a read-only bytes buffer that cannot be re-enabled."""
    immutable = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


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


class CovarianceConfig(BaseModel):
    """Validated, deterministic configuration for fitting an anisotropic covariance."""

    reg_coef: float = Field(default=1e-6, gt=0, description="Diagonal loading added to the empirical covariance")
    min_samples_per_dimension: float = Field(
        default=2.0, gt=0, description="Minimum samples per dimension before fitting"
    )


@dataclass(frozen=True)
class CovarianceState:
    """Immutable fitted covariance geometry bound to one representation identity.

    Attributes
    ----------
    mean : np.ndarray
        Empirical mean vector, shape ``(dim,)``.
    covariance : np.ndarray
        Regularized positive-definite covariance matrix, shape ``(dim, dim)``.
    n_samples : int
        Number of samples the covariance was fitted from.
    source_representation_identity : str
        Identity of the representation the covariance was fitted on (dataset,
        model version, layer). Cross-space reuse of a fitted covariance is a
        caller error and is guarded where scoring is identity-aware.
    reg_coef : float
        Diagonal loading applied at fit time.
    provenance : dict[str, Any]
        Free-form provenance (dataset version, preprocessing, fit metadata).
    """

    mean: np.ndarray
    covariance: np.ndarray
    n_samples: int
    source_representation_identity: str
    reg_coef: float
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate and defensively own every part of the fitted state."""
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        covariance = np.array(self.covariance, dtype=np.float64, copy=True)
        if mean.ndim != 1 or covariance.ndim != 2 or covariance.shape != (mean.size, mean.size):
            msg = (
                "CovarianceState expects 1D mean and matching square covariance, "
                f"got mean {mean.shape}, covariance {covariance.shape}"
            )
            raise ValueError(msg)
        if not np.isfinite(mean).all():
            raise ValueError("CovarianceState mean must contain only finite values")
        validate_covariance(covariance, dim=mean.size)
        n_samples = cast(Any, self.n_samples)
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            raise ValueError(f"CovarianceState n_samples must be positive, got {self.n_samples!r}")
        reg_coef = cast(Any, self.reg_coef)
        reg_coef_value: float = float(reg_coef)
        if (
            isinstance(reg_coef, bool)
            or not isinstance(reg_coef, (int, float, np.integer, np.floating))
            or not np.isfinite(reg_coef_value)
            or reg_coef <= 0
        ):
            raise ValueError(f"CovarianceState reg_coef must be finite and > 0, got {self.reg_coef!r}")
        identity = cast(Any, self.source_representation_identity)
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("CovarianceState source_representation_identity must be non-empty")
        provenance = cast(Any, self.provenance)
        if not isinstance(provenance, Mapping):
            raise ValueError("CovarianceState provenance must be a mapping")

        object.__setattr__(self, "mean", _immutable_array(mean))
        object.__setattr__(self, "covariance", _immutable_array(covariance))
        object.__setattr__(self, "provenance", _freeze_provenance(provenance))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation (arrays as nested lists)."""
        return {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "n_samples": self.n_samples,
            "source_representation_identity": self.source_representation_identity,
            "reg_coef": self.reg_coef,
            "provenance": _thaw_provenance(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CovarianceState:
        """Rebuild a :class:`CovarianceState` from a :meth:`to_dict` payload."""
        values = dict(data)
        try:
            raw_n_samples = values["n_samples"]
            if isinstance(raw_n_samples, bool) or not isinstance(raw_n_samples, (int, np.integer)):
                raise TypeError("n_samples must be an integer, not bool or fractional")
            raw_identity = values["source_representation_identity"]
            if not isinstance(raw_identity, str):
                raise TypeError("source_representation_identity must be a string")
            raw_reg_coef = values["reg_coef"]
            if isinstance(raw_reg_coef, bool) or not isinstance(raw_reg_coef, (int, float, np.integer, np.floating)):
                raise TypeError("reg_coef must be numeric, not bool")
            mean = np.asarray(values["mean"], dtype=np.float64)
            covariance = np.asarray(values["covariance"], dtype=np.float64)
            n_samples = int(cast(Any, raw_n_samples))
            identity = raw_identity
            reg_coef = float(cast(Any, raw_reg_coef))
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"CovarianceState.from_dict missing or malformed field: {exc}"
            raise ValueError(msg) from exc
        provenance = values.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("CovarianceState provenance must be a mapping")
        provenance_mapping = cast(Mapping[str, Any], provenance)
        return cls(mean, covariance, n_samples, identity, reg_coef, provenance_mapping)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Serialize to a portable ``.npz`` checkpoint with JSON provenance."""
        payload = self.to_dict()
        mean = payload.pop("mean")
        covariance = payload.pop("covariance")
        np.savez(
            path,
            mean=np.asarray(mean, dtype=np.float64),
            covariance=np.asarray(covariance, dtype=np.float64),
            metadata_json=json.dumps(payload),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> CovarianceState:
        """Load a :class:`CovarianceState` from a checkpoint written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:  # pyright: ignore[reportUnknownMemberType]
            mean = np.asarray(data["mean"], dtype=np.float64)
            covariance = np.asarray(data["covariance"], dtype=np.float64)
            metadata_raw = data["metadata_json"].item()
            if not isinstance(metadata_raw, str):
                raise ValueError(f"checkpoint {path} has no metadata_json string")
            metadata = json.loads(metadata_raw)
        metadata["mean"] = mean.tolist()
        metadata["covariance"] = covariance.tolist()
        return cls.from_dict(metadata)


def fit_covariance_state(
    data: np.ndarray,
    *,
    source_representation_identity: str,
    config: CovarianceConfig | None = None,
    provenance: dict[str, Any] | None = None,
) -> CovarianceState:
    """Fit an empirical covariance geometry bound to one representation identity.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape ``(n_samples, dim)`` with ``n_samples > dim``.
    source_representation_identity : str
        Identity the fitted metric is valid for. A fitted covariance must not
        be silently reused for a different representation.
    config : CovarianceConfig | None
        Fitting configuration; defaults to ``CovarianceConfig()``.
    provenance : dict[str, Any] | None
        Free-form provenance attached to the fitted state.

    Returns
    -------
    CovarianceState
        The fitted, regularized, positive-definite covariance geometry.
    """
    cfg = config if config is not None else CovarianceConfig()
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        msg = f"fit_covariance_state expects 2D data, got {values.ndim}D"
        raise ValueError(msg)
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("fit_covariance_state needs at least 2 samples and 1 feature")
    min_samples = max(2, int(np.ceil(cfg.min_samples_per_dimension * values.shape[1])))
    if values.shape[0] < min_samples:
        raise ValueError(
            f"fit_covariance_state needs at least {min_samples} samples for {values.shape[1]} dimensions; "
            f"got {values.shape[0]}"
        )
    mean, covariance = _fit_covariance(values, reg_coef=cfg.reg_coef)
    return CovarianceState(
        mean=mean,
        covariance=covariance,
        n_samples=values.shape[0],
        source_representation_identity=source_representation_identity,
        reg_coef=cfg.reg_coef,
        provenance=dict(provenance or {}),
    )
