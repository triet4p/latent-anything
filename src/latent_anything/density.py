"""Representation-bound Gaussian-mixture density and OOD evaluation.

The estimator deliberately owns covariance and Mahalanobis behavior.  A
``LatentSpace`` is used only as a compatibility declaration; this module does
not change the geometry abstraction.
"""

# scikit-learn's runtime estimator attributes are intentionally dynamic and
# are not fully represented by the installed type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportOptionalMemberAccess=false

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, Field
from sklearn.covariance import EmpiricalCovariance  # type: ignore[reportMissingTypeStubs]
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[reportMissingTypeStubs]
from sklearn.mixture import GaussianMixture  # type: ignore[reportMissingTypeStubs]

CovarianceType = Literal["full", "tied", "diag", "spherical"]


class GMMConfig(BaseModel):
    """Validated, deterministic configuration for :class:`GaussianMixtureDensity`."""

    n_components: int = Field(default=2, ge=1)
    covariance_type: CovarianceType = "full"
    reg_covar: float = Field(default=1e-6, gt=0)
    n_init: int = Field(default=1, ge=1)
    max_iter: int = Field(default=200, ge=1)
    tol: float = Field(default=1e-3, gt=0)
    random_state: int = Field(default=0, ge=0)
    min_samples_per_dimension: float = Field(default=2.0, gt=0)


@dataclass(frozen=True)
class DensityResult:
    """Density and calibrated OOD scores for one evaluation batch."""

    log_density: np.ndarray
    responsibilities: np.ndarray
    calibrated_ood_score: np.ndarray
    source_representation_identity: str
    fit_provenance: dict[str, Any]
    calibration_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly metadata while retaining arrays as arrays."""
        return asdict(self)


@dataclass(frozen=True)
class DensityMetrics:
    """OOD ranking and calibration diagnostics."""

    auroc: float
    auprc: float
    brier_score: float
    mean_id_score: float
    mean_ood_score: float
    n_id: int
    n_ood: int


@dataclass(frozen=True)
class DensityEvaluationReport:
    """Evaluation on predeclared ID and OOD data using one fitted estimator."""

    in_distribution: DensityResult
    out_of_distribution: DensityResult
    metrics: DensityMetrics
    split_provenance: dict[str, Any]


@dataclass(frozen=True)
class DensityStabilityReport:
    """Cross-seed uncertainty for the OOD metrics."""

    seeds: tuple[int, ...]
    aurocs: tuple[float, ...]
    auprcs: tuple[float, ...]
    mean_auroc: float
    auroc_ci95: float
    mean_auprc: float
    auprc_ci95: float
    reports: tuple[DensityEvaluationReport, ...]


def _validate_batch(data: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got {values.ndim}D")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and feature")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validate_point(
    point: np.ndarray,
    check_ready: Callable[[int, str | None], None],
    source_representation_identity: str | None,
) -> np.ndarray:
    """Validate a flat ``(dim,)`` point and check the estimator is ready."""
    value = np.asarray(point, dtype=np.float64)
    if value.ndim != 1:
        raise ValueError(f"point must be a flat 1D array, got {value.ndim}D")
    if value.shape[0] == 0:
        raise ValueError("point must have at least one feature")
    if not np.isfinite(value).all():
        raise ValueError("point must contain only finite values")
    check_ready(value.shape[0], source_representation_identity)
    return value


def _component_covariance(model: Any, component: int) -> np.ndarray:
    """Return the per-component covariance matrix for a fitted GMM.

    Normalizes every ``covariance_type`` to a ``(dim, dim)`` matrix so the
    gradient computation can solve a single linear system per component.
    """
    covariance_type = model.covariance_type if hasattr(model, "covariance_type") else "full"
    covariances = np.asarray(model.covariances_)
    if covariance_type == "tied":
        return np.asarray(covariances, dtype=np.float64)
    if covariance_type == "diag":
        return np.diag(np.asarray(covariances[component], dtype=np.float64))
    if covariance_type == "spherical":
        variance = float(covariances[component])
        dim = int(covariances.shape[1]) if covariances.ndim == 2 else int(model.means_.shape[1])
        return variance * np.eye(dim)
    return np.asarray(covariances[component], dtype=np.float64)


def validate_density_geometry(geometry: str) -> None:
    """Reject representations whose metric is not supported by this estimator."""
    if geometry not in {"euclidean", "unit_norm"}:
        raise ValueError(
            f"GaussianMixtureDensity supports only flat euclidean/unit_norm representations; got geometry {geometry!r}"
        )


class GaussianMixtureDensity:
    """Fit a deterministic Gaussian mixture bound to one representation identity."""

    def __init__(self, config: GMMConfig | None = None, **kwargs: Any) -> None:
        self._config = config if config is not None else GMMConfig(**kwargs)
        self._model: GaussianMixture | None = None
        self._source_identity: str | None = None
        self._fit_provenance: dict[str, Any] = {}
        self._calibration_scores: np.ndarray | None = None
        self._calibration_provenance: dict[str, Any] = {}

    @property
    def config(self) -> GMMConfig:
        """Return immutable-style pydantic configuration."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self._model is not None

    @property
    def source_representation_identity(self) -> str | None:
        """Return the representation identity the estimator was fitted on."""
        return self._source_identity

    def state_snapshot(self) -> dict[str, Any]:
        """Return a defensive, component-local state snapshot for reproducibility."""
        return {
            "config": self._config.model_dump(mode="json"),
            "is_fitted": self.is_fitted,
            "source_representation_identity": self._source_identity,
            "fit_provenance": dict(self._fit_provenance),
            "calibration_provenance": dict(self._calibration_provenance),
            "n_calibration_samples": int(len(self._calibration_scores)) if self._calibration_scores is not None else 0,
        }

    def fit(
        self,
        data: np.ndarray,
        *,
        source_representation_identity: str,
        geometry: str = "euclidean",
        provenance: dict[str, Any] | None = None,
    ) -> GaussianMixtureDensity:
        """Fit on training data only, recording source identity and provenance."""
        values = _validate_batch(data, name="fit data")
        validate_density_geometry(geometry)
        cfg = self._config
        min_samples = max(cfg.n_components * 2, int(np.ceil(cfg.min_samples_per_dimension * values.shape[1])))
        if values.shape[0] < min_samples:
            raise ValueError(
                f"fit data needs at least {min_samples} samples for {values.shape[1]} dimensions "
                f"and {cfg.n_components} components; got {values.shape[0]}"
            )
        if cfg.covariance_type == "full" and values.shape[0] <= values.shape[1]:
            raise ValueError("full covariance requires more samples than dimensions")
        model = GaussianMixture(
            n_components=cfg.n_components,
            covariance_type=cfg.covariance_type,
            reg_covar=cfg.reg_covar,
            n_init=cfg.n_init,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            random_state=cfg.random_state,
        )
        try:
            model.fit(values)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"GMM covariance fitting failed; increase reg_covar or add samples: {exc}") from exc
        self._model = model
        self._source_identity = source_representation_identity
        self._fit_provenance = dict(provenance or {})
        self._fit_provenance.update({"geometry": geometry, "n_samples": values.shape[0], "n_features": values.shape[1]})
        return self

    def calibrate(
        self,
        data: np.ndarray,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Calibrate scores using held-out in-distribution data only."""
        values = _validate_batch(data, name="calibration data")
        self._check_ready(values.shape[1], None)
        if values.shape[0] < 2:
            raise ValueError("calibration data needs at least 2 samples")
        self._calibration_scores = np.sort(self._raw_ood_score(values))
        self._calibration_provenance = dict(provenance or {})
        self._calibration_provenance["n_samples"] = values.shape[0]

    def score(self, data: np.ndarray, *, source_representation_identity: str | None = None) -> DensityResult:
        """Score one batch and return log density, responsibilities, and calibrated OOD score."""
        values = _validate_batch(data, name="evaluation data")
        self._check_ready(values.shape[1], source_representation_identity)
        assert self._model is not None and self._source_identity is not None
        raw = self._raw_ood_score(values)
        if self._calibration_scores is None:
            calibrated = raw.copy()
        else:
            calibrated = np.searchsorted(self._calibration_scores, raw, side="right") / len(self._calibration_scores)
        return DensityResult(
            log_density=np.asarray(self._model.score_samples(values)),
            responsibilities=np.asarray(self._model.predict_proba(values)),
            calibrated_ood_score=np.asarray(calibrated, dtype=np.float64),
            source_representation_identity=self._source_identity,
            fit_provenance=dict(self._fit_provenance),
            calibration_provenance=dict(self._calibration_provenance),
        )

    def state_digest(self) -> str:
        """Return a stable SHA-256 digest over the fitted GMM parameters.

        Two estimators with identical weights/means/covariances produce the
        same digest, which makes it usable as a deterministic component-state
        hash in cache keys (runtime counters and provenance are excluded).
        """
        if self._model is None:
            raise RuntimeError("GaussianMixtureDensity has not been fitted")
        import hashlib

        parts: list[bytes] = []
        for name in ("weights_", "means_", "covariances_", "precisions_cholesky_"):
            value = getattr(self._model, name, None)
            if value is not None:
                parts.append(np.asarray(value, dtype=np.float64).round(8).tobytes())
        return hashlib.sha256(b"".join(parts)).hexdigest()

    def log_density(self, point: np.ndarray, *, source_representation_identity: str | None = None) -> float:
        """Return the fitted GMM log-density at a single point.

        ``point`` is a flat ``(dim,)`` vector. The identity is checked the same
        way :meth:`score` checks batches, so cross-space scoring is rejected.
        """
        value = _validate_point(point, self._check_ready, source_representation_identity)
        assert self._model is not None
        return float(np.asarray(self._model.score_samples(value[None, :]))[0])  # pyright: ignore[reportUnknownMemberType]

    def log_density_gradient(
        self, point: np.ndarray, *, source_representation_identity: str | None = None
    ) -> np.ndarray:
        """Return the analytic gradient of the GMM log-density at a point.

        The gradient of a Gaussian-mixture log-density is

        .. math:: \\nabla \\log p(z) = \\sum_k \\gamma_k(z)\\, \\Sigma_k^{-1}(\\mu_k - z)

        where :math:`\\gamma_k` are the responsibilities. The per-component
        precision-times-vector products are solved rather than inverted for
        numerical stability. This is the pullback-type oracle that the
        density-penalized geodesic path optimizer consumes.
        """
        value = _validate_point(point, self._check_ready, source_representation_identity)
        assert self._model is not None
        model = cast(Any, self._model)
        responsibilities = np.asarray(model.predict_proba(value[None, :]))[0]  # pyright: ignore[reportUnknownMemberType]
        gradient = np.zeros(value.shape[0], dtype=np.float64)
        for k, weight in enumerate(responsibilities):
            mean = np.asarray(model.means_[k], dtype=np.float64)  # pyright: ignore[reportUnknownMemberType]
            if weight <= 0.0:
                continue
            covariance = _component_covariance(model, k)  # pyright: ignore[reportUnknownMemberType]
            diff = mean - value
            gradient += float(weight) * np.linalg.solve(covariance, diff)
        return gradient

    def evaluate(
        self,
        in_distribution: np.ndarray,
        out_of_distribution: np.ndarray,
        *,
        source_representation_identity: str | None = None,
        split_provenance: dict[str, Any] | None = None,
    ) -> DensityEvaluationReport:
        """Evaluate predeclared ID and OOD sets without refitting or mixing splits."""
        id_result = self.score(in_distribution, source_representation_identity=source_representation_identity)
        ood_result = self.score(out_of_distribution, source_representation_identity=source_representation_identity)
        y_true = np.concatenate(
            [np.zeros(len(id_result.calibrated_ood_score)), np.ones(len(ood_result.calibrated_ood_score))]
        )
        y_score = np.concatenate([id_result.calibrated_ood_score, ood_result.calibrated_ood_score])
        metrics = DensityMetrics(
            auroc=float(roc_auc_score(y_true, y_score)),
            auprc=float(average_precision_score(y_true, y_score)),
            brier_score=float(np.mean((y_score - y_true) ** 2)),
            mean_id_score=float(np.mean(id_result.calibrated_ood_score)),
            mean_ood_score=float(np.mean(ood_result.calibrated_ood_score)),
            n_id=len(id_result.calibrated_ood_score),
            n_ood=len(ood_result.calibrated_ood_score),
        )
        return DensityEvaluationReport(id_result, ood_result, metrics, dict(split_provenance or {}))

    def _raw_ood_score(self, data: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return -np.asarray(self._model.score_samples(data), dtype=np.float64)

    def _check_ready(self, n_features: int, source_identity: str | None) -> None:
        if self._model is None or self._source_identity is None:
            raise RuntimeError("GaussianMixtureDensity has not been fitted")
        if source_identity is not None and source_identity != self._source_identity:
            raise ValueError(
                "cross-space scoring is not allowed: fitted identity "
                f"{self._source_identity!r} != requested {source_identity!r}"
            )
        if int(self._model.means_.shape[1]) != n_features:
            raise ValueError(f"feature dimension mismatch: fitted {self._model.means_.shape[1]}, got {n_features}")


def mahalanobis_baseline(train: np.ndarray, evaluation: np.ndarray) -> np.ndarray:
    """Return an OOD score from a single Gaussian/Mahalanobis baseline."""
    x_train = _validate_batch(train, name="baseline train data")
    x_eval = _validate_batch(evaluation, name="baseline evaluation data")
    if x_train.shape[1] != x_eval.shape[1] or len(x_train) <= x_train.shape[1]:
        raise ValueError("Mahalanobis baseline needs matching dimensions and more samples than dimensions")
    model = EmpiricalCovariance().fit(x_train)
    return np.asarray(model.mahalanobis(x_eval), dtype=np.float64)


def cross_seed_evaluation(
    train: np.ndarray,
    calibration: np.ndarray,
    in_distribution: np.ndarray,
    out_of_distribution: np.ndarray,
    *,
    source_representation_identity: str,
    config: GMMConfig | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    geometry: str = "euclidean",
) -> DensityStabilityReport:
    """Fit/evaluate each seed on identical, caller-declared splits."""
    reports: list[DensityEvaluationReport] = []
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")
    for seed in seed_values:
        cfg = (config or GMMConfig()).model_copy(update={"random_state": seed})
        estimator = GaussianMixtureDensity(cfg).fit(
            train, source_representation_identity=source_representation_identity, geometry=geometry
        )
        estimator.calibrate(calibration)
        reports.append(
            estimator.evaluate(
                in_distribution,
                out_of_distribution,
                source_representation_identity=source_representation_identity,
                split_provenance={"seed": seed, "fit": "train", "calibration": "held_out", "evaluation": "held_out"},
            )
        )
    aurocs = tuple(report.metrics.auroc for report in reports)
    auprcs = tuple(report.metrics.auprc for report in reports)
    auroc_std = float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0
    auprc_std = float(np.std(auprcs, ddof=1)) if len(auprcs) > 1 else 0.0
    divisor = np.sqrt(len(reports))
    return DensityStabilityReport(
        seeds=seed_values,
        aurocs=aurocs,
        auprcs=auprcs,
        mean_auroc=float(np.mean(aurocs)),
        auroc_ci95=float(1.96 * auroc_std / divisor),
        mean_auprc=float(np.mean(auprcs)),
        auprc_ci95=float(1.96 * auprc_std / divisor),
        reports=tuple(reports),
    )
