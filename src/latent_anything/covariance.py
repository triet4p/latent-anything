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
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, Field

from latent_anything.geometry import fit_covariance as _fit_covariance


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
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation (arrays as nested lists)."""
        return {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "n_samples": self.n_samples,
            "source_representation_identity": self.source_representation_identity,
            "reg_coef": self.reg_coef,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CovarianceState:
        """Rebuild a :class:`CovarianceState` from a :meth:`to_dict` payload."""
        values = dict(data)
        try:
            mean = np.asarray(values["mean"], dtype=np.float64)
            covariance = np.asarray(values["covariance"], dtype=np.float64)
            n_samples = int(values["n_samples"])
            identity = str(values["source_representation_identity"])
            reg_coef = float(values["reg_coef"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"CovarianceState.from_dict missing or malformed field: {exc}"
            raise ValueError(msg) from exc
        provenance = values.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("CovarianceState provenance must be a mapping")
        provenance_mapping = cast(Mapping[str, Any], provenance)
        provenance_items: dict[str, Any] = dict(provenance_mapping)
        if mean.ndim != 1 or covariance.ndim != 2 or covariance.shape != (mean.shape[0], mean.shape[0]):
            msg = (
                "CovarianceState.from_dict expects 1D mean and matching square covariance, "
                f"got mean {mean.shape}, covariance {covariance.shape}"
            )
            raise ValueError(msg)
        return cls(mean, covariance, n_samples, identity, reg_coef, provenance_items)

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
