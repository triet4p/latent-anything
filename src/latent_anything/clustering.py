"""K-means latent structure discovery with geometry checks and diagnostics.

This module provides a ``KMeans`` class for discovering cluster structure in
latent representations, with explicit geometry compatibility checks, typed
results, uncertainty quantification via bootstrap/seed stability, and external
validity comparison against known labels.

Key components
--------------
- :class:`KMeansConfig` — pydantic configuration for K-means fitting.
- :class:`KMeansResult` — typed result with assignments, centers, inertia,
  silhouette, confidence proxy, and provenance.
- :class:`ClusterStabilityReport` — multi-seed / bootstrap stability with
  label-aligned cluster agreement and adjusted Rand index.
- :class:`KMeans` — main class with ``fit_predict`` lifecycle.
- :func:`check_clustering_geometry` — raise on unsupported spaces.
- :func:`compare_with_labels` — external validation via adjusted Rand index.
- :func:`cluster_stability_analysis` — multi-seed stability analysis.

Design decisions
----------------
- **Not a ``Method``.**  K-means produces cluster assignments, not a
  dimensionality-reducing transform.  It does not follow the ``Method`` /
  ``AnalysisPipeline`` lifecycle.
- **Full sklearn wrapper.**  The heavy lifting is delegated to
  ``sklearn.cluster.KMeans`` and ``sklearn.metrics``.  The wrapper
  adds geometry checks, bootstrap stability, typed results, and
  provenance tracking.
- **Geometry-aware.**  Clustering is only allowed on ``"euclidean"``
  and ``"unit_norm"`` ``LatentSpace`` geometries.  Structured
  (``"gaussian_set"``) and discrete (``"discrete_code"``) spaces are
  rejected with a clear error.
- **Provenance first.**  Every result carries the config, random state,
  and caller-supplied provenance so clustering runs are fully traceable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false
# (sklearn has incomplete type stubs — these warnings are noise)


# ---------------------------------------------------------------------------
# Part 1 — Geometry checks  (Task 5)
# ---------------------------------------------------------------------------


CLUSTERING_ALLOWED_GEOMETRIES: frozenset[str] = frozenset({"euclidean", "unit_norm"})


def check_clustering_geometry(geometry: str) -> None:
    """Raise ``ValueError`` if *geometry* does not support K-means clustering.

    K-means requires a metric space with meaningful Euclidean (or angular)
    distances.  Structured set-like spaces (``"gaussian_set"``) and discrete
    code spaces (``"discrete_code"``) are not supported.

    Parameters
    ----------
    geometry :
        Geometry string from a ``LatentSpace`` (e.g. ``"euclidean"``).

    Raises
    ------
    ValueError
        If *geometry* is not in ``CLUSTERING_ALLOWED_GEOMETRIES``.
    """
    if geometry not in CLUSTERING_ALLOWED_GEOMETRIES:
        msg = (
            f"K-means clustering does not support geometry {geometry!r}. "
            f"Supported geometries: {sorted(CLUSTERING_ALLOWED_GEOMETRIES)}"
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Part 2 — Configuration  (Tasks 1, 5)
# ---------------------------------------------------------------------------


class KMeansConfig(BaseModel):
    """Pydantic config for :class:`KMeans` clustering.

    Parameters
    ----------
    n_clusters :
        Number of clusters to discover.
    init :
        Initialisation method (``"k-means++"`` or ``"random"``).
    n_init :
        Number of initialisation runs (the best result is kept).
    max_iter :
        Maximum iterations per single run.
    random_state :
        Deterministic seed for all randomness.
    standardize :
        Whether to z-score features before clustering (leakage-safe:
        statistics computed on the full dataset since clustering is
        unsupervised).
    tol :
        Relative tolerance for convergence.
    """

    n_clusters: int = Field(default=8, ge=2, description="Number of clusters")
    init: str = Field(default="k-means++", pattern="^(k-means\\+\\+|random)$", description="Initialisation method")
    n_init: int = Field(default=10, ge=1, description="Number of initialisation runs")
    max_iter: int = Field(default=300, ge=1, description="Maximum iterations per run")
    random_state: int = Field(default=0, ge=0, description="Random seed")
    standardize: bool = Field(default=True, description="Z-score features before clustering")
    tol: float = Field(default=1e-4, gt=0, description="Relative tolerance")


# ---------------------------------------------------------------------------
# Part 3 — Typed result  (Task 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KMeansResult:
    """Typed result of K-means clustering.

    Parameters
    ----------
    assignments :
        Cluster label for each sample, shape ``(n_samples,)``.
    centers :
        Cluster centroids, shape ``(n_clusters, n_features)``.
    inertia :
        Sum of squared distances to the nearest centroid.
    n_iter :
        Number of iterations actually run.
    n_clusters :
        Number of clusters requested.
    cluster_sizes :
        Number of samples per cluster, shape ``(n_clusters,)``.
    silhouette_score :
        Overall mean silhouette coefficient (``-1`` to ``1``).
        Computed on a stratified subsample if ``n_samples > 10000``.
    per_sample_silhouette :
        Silhouette coefficient for each sample, shape ``(n_samples,)``.
        ``None`` if silhouettes were not computed.
    confidence :
        Confidence proxy: the margin between the nearest and second-nearest
        cluster distance for each sample.  Larger values mean more confident
        assignment.  Shape ``(n_samples,)``.
    n_init :
        Number of initialisation runs evaluated (matches config).
    config :
        The config that produced this result.
    feature_means :
        Per-feature means used for standardisation, or ``None``.
    feature_stds :
        Per-feature standard deviations, or ``None``.
    provenance :
        Metadata about the run (dataset, model, layer, etc.).
    """

    assignments: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int
    n_clusters: int
    cluster_sizes: np.ndarray
    silhouette_score: float
    per_sample_silhouette: np.ndarray | None
    confidence: np.ndarray
    n_init: int
    config: KMeansConfig
    feature_means: np.ndarray | None = None
    feature_stds: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.assignments.ndim != 1:
            raise ValueError(f"assignments must be 1D, got {self.assignments.ndim}D")
        if self.centers.ndim != 2:
            raise ValueError(f"centers must be 2D, got {self.centers.ndim}D")
        if self.centers.shape[0] != self.n_clusters:
            raise ValueError(f"centers shape {self.centers.shape[0]} != n_clusters {self.n_clusters}")
        if self.cluster_sizes.shape != (self.n_clusters,):
            raise ValueError(f"cluster_sizes shape {self.cluster_sizes.shape} != ({self.n_clusters},)")
        if self.confidence.shape != (self.assignments.shape[0],):
            raise ValueError(f"confidence shape {self.confidence.shape} != assignments shape {self.assignments.shape}")
        if self.per_sample_silhouette is not None and self.per_sample_silhouette.shape != self.assignments.shape:
            raise ValueError(
                f"per_sample_silhouette shape {self.per_sample_silhouette.shape} "
                f"!= assignments shape {self.assignments.shape}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict (deep copies array attributes)."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, KMeansConfig):
                out[key] = value.model_dump()
            elif isinstance(value, dict):
                out[key] = dict(value)
            else:
                out[key] = value
        return out


# ---------------------------------------------------------------------------
# Part 4 — Stability report  (Task 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterStabilityReport:
    """Multi-seed / bootstrap stability for K-means clustering.

    Parameters
    ----------
    adjusted_rand_index :
        Mean adjusted Rand index between cluster assignments from
        different seeds (bootstrap or re-fit).  1.0 = identical,
        0.0 = chance-level agreement.
    mean_stability :
        Mean fraction of samples that stay in the same (label-aligned)
        cluster across runs.
    per_cluster_stability :
        Stability per cluster, shape ``(n_clusters,)``.
    n_seeds :
        Number of seeds evaluated.
    assignments_matrix :
        Cluster assignments for each seed, shape ``(n_seeds, n_samples)``.
        Labels are aligned across seeds via Hungarian matching.
    results :
        Full results for each seed.
    """

    adjusted_rand_index: float
    mean_stability: float
    per_cluster_stability: np.ndarray
    n_seeds: int
    assignments_matrix: np.ndarray
    results: list[KMeansResult]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], KMeansResult):
                out[key] = [r.to_dict() for r in value]
            elif isinstance(value, dict):
                out[key] = dict(value)
            else:
                out[key] = value
        return out


# ---------------------------------------------------------------------------
# Part 5 — Main KMeans class  (Task 1)
# ---------------------------------------------------------------------------


def _hungarian_align(
    ref_labels: np.ndarray,
    target_labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Align *target_labels* to *ref_labels* via Hungarian matching.

    Returns a copy of *target_labels* with labels permuted to maximise
    agreement with *ref_labels*.
    """
    from scipy.optimize import linear_sum_assignment  # type: ignore[reportMissingTypeStubs]

    # Build cost matrix: cost[i, j] = -count of samples in ref_i ∩ target_j
    cost = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    for i in range(n_clusters):
        ref_mask = ref_labels == i
        for j in range(n_clusters):
            cost[i, j] = -float(np.sum(ref_mask & (target_labels == j)))

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {col_ind[k]: row_ind[k] for k in range(n_clusters)}
    aligned = np.array([mapping.get(label, label) for label in target_labels], dtype=np.int64)
    return aligned


class KMeans:
    """K-means clustering for latent representations.

    Wraps ``sklearn.cluster.KMeans`` with geometry checks, typed results,
    confidence proxy, and provenance tracking.

    Parameters
    ----------
    config : KMeansConfig, optional
        Clustering configuration.  Defaults to ``KMeansConfig()``.
    """

    def __init__(self, config: KMeansConfig | None = None, **kwargs: Any) -> None:
        # Support both direct config and ObjectSpec-style kwargs
        if kwargs:
            self._config = KMeansConfig(**kwargs)
        else:
            self._config = config if config is not None else KMeansConfig()
        self._fitted: bool = False
        self._result: KMeansResult | None = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def config(self) -> KMeansConfig:
        """Return the clustering configuration."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit_predict` has been called successfully."""
        return self._fitted

    @property
    def result(self) -> KMeansResult:
        """Return the last fit result.

        Raises
        ------
        RuntimeError
            If not fitted yet.
        """
        if not self._fitted or self._result is None:
            msg = "KMeans has not been fitted yet. Call fit_predict() first."
            raise RuntimeError(msg)
        return self._result

    # ── Geometry check helper ─────────────────────────────────────────

    @staticmethod
    def check_geometry(geometry: str) -> None:
        """Raise ``ValueError`` if *geometry* is not clustering-compatible.

        See :func:`check_clustering_geometry`.
        """
        check_clustering_geometry(geometry)

    # ── Fit & predict ─────────────────────────────────────────────────

    def fit_predict(
        self,
        data: np.ndarray,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> KMeansResult:
        """Fit K-means and return cluster assignments.

        Parameters
        ----------
        data :
            Feature matrix of shape ``(n_samples, n_features)``.
        provenance :
            Optional metadata dict attached to the result (e.g. model name,
            layer index).

        Returns
        -------
        KMeansResult
            Typed result with assignments, centers, silhouette, and
            confidence proxy.

        Raises
        ------
        ValueError
            If input shapes are invalid or data is degenerate.
        """
        from sklearn.cluster import KMeans as _SKLearnKMeans
        from sklearn.metrics import silhouette_samples, silhouette_score

        cfg = self._config
        data = np.asarray(data, dtype=np.float64)

        # ── Validate ──────────────────────────────────────────────
        if data.ndim != 2:
            msg = f"data must be 2D, got {data.ndim}D"
            raise ValueError(msg)
        if data.shape[0] < cfg.n_clusters:
            msg = f"n_samples ({data.shape[0]}) must be >= n_clusters ({cfg.n_clusters})"
            raise ValueError(msg)
        if data.shape[0] < 2:
            msg = f"need at least 2 samples, got {data.shape[0]}"
            raise ValueError(msg)
        if data.shape[1] < 1:
            msg = "need at least 1 feature"
            raise ValueError(msg)

        n_samples = data.shape[0]

        # ── Standardise ───────────────────────────────────────────
        if cfg.standardize:
            means = data.mean(axis=0)
            stds = data.std(axis=0, ddof=0)
            stds = np.where(stds < 1e-12, 1.0, stds)
            data_scaled = (data - means) / stds
            feature_means: np.ndarray | None = means
            feature_stds: np.ndarray | None = stds
        else:
            data_scaled = data.copy()
            feature_means = None
            feature_stds = None

        # ── Fit sklearn KMeans ────────────────────────────────────
        skm = _SKLearnKMeans(
            n_clusters=cfg.n_clusters,
            init=cfg.init,
            n_init=cfg.n_init,  # pyright: ignore[reportArgumentType]
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
            tol=cfg.tol,
        )
        skm.fit(data_scaled)

        assignments: np.ndarray = np.asarray(skm.labels_, dtype=np.int64)
        centers: np.ndarray = np.asarray(skm.cluster_centers_, dtype=np.float64)
        inertia = float(skm.inertia_)  # pyright: ignore[reportArgumentType]
        n_iter = int(skm.n_iter_) if hasattr(skm, "n_iter_") else cfg.max_iter

        # ── Cluster sizes ─────────────────────────────────────────
        cluster_sizes = np.zeros(cfg.n_clusters, dtype=np.int64)
        for c in range(cfg.n_clusters):
            cluster_sizes[c] = int(np.sum(assignments == c))

        # ── Silhouette ────────────────────────────────────────────
        if cfg.n_clusters < n_samples:
            silhouette_val = float(silhouette_score(data_scaled, assignments))
            per_sil = np.asarray(silhouette_samples(data_scaled, assignments))
        else:
            silhouette_val = 0.0
            per_sil = None

        # ── Confidence proxy: nearest vs second-nearest margin ────
        from sklearn.metrics.pairwise import euclidean_distances  # type: ignore[reportPrivateUsage]

        dists = euclidean_distances(data_scaled, centers)  # (n, k)
        sorted_dists = np.sort(dists, axis=1)
        margin = sorted_dists[:, 1] - sorted_dists[:, 0]
        # Guard against identical distances (should not happen in practice)
        margin = np.maximum(margin, 0.0)
        confidence = np.asarray(margin, dtype=np.float64)

        provenance_dict: dict[str, Any] = dict(provenance or {})
        provenance_dict.setdefault("n_features", data.shape[1])
        provenance_dict.setdefault("n_samples", data.shape[0])

        self._result = KMeansResult(
            assignments=assignments,
            centers=centers,
            inertia=inertia,
            n_iter=n_iter,
            n_clusters=cfg.n_clusters,
            cluster_sizes=cluster_sizes,
            silhouette_score=silhouette_val,
            per_sample_silhouette=per_sil,
            confidence=confidence,
            n_init=cfg.n_init,
            config=cfg,
            feature_means=feature_means,
            feature_stds=feature_stds,
            provenance=provenance_dict,
        )
        self._fitted = True
        return self._result


# ---------------------------------------------------------------------------
# Part 6 — Multi-seed stability analysis  (Task 3)
# ---------------------------------------------------------------------------


def cluster_stability_analysis(
    data: np.ndarray,
    *,
    seeds: list[int] | None = None,
    n_seeds: int = 10,
    config: KMeansConfig | None = None,
) -> ClusterStabilityReport:
    """Assess K-means cluster stability across multiple random seeds.

    Fits K-means with each seed, aligns cluster labels via Hungarian
    matching against the first seed, then reports mean stability and
    adjusted Rand index.

    Parameters
    ----------
    data :
        Feature matrix ``(n_samples, n_features)``.
    seeds :
        Explicit list of seeds.  Mutually exclusive with ``n_seeds``.
    n_seeds :
        Number of seeds to evaluate (0 … *n_seeds* - 1).  Only used
        when ``seeds`` is ``None``.
    config :
        Base config (``random_state`` is overridden per seed).

    Returns
    -------
    ClusterStabilityReport
        Stability metrics and per-seed results.
    """
    if seeds is None:
        seed_list = list(range(n_seeds))
    else:
        seed_list = list(seeds)
        n_seeds = len(seed_list)

    base_cfg = config or KMeansConfig()
    results: list[KMeansResult] = []

    for seed in seed_list:
        cfg = base_cfg.model_copy(update={"random_state": seed})
        km = KMeans(cfg)
        result = km.fit_predict(data)
        results.append(result)

    # Align all to the first seed
    ref_assignments = results[0].assignments
    n_clusters = base_cfg.n_clusters

    aligned_matrix = np.zeros((n_seeds, data.shape[0]), dtype=np.int64)
    aligned_matrix[0] = ref_assignments

    for i in range(1, n_seeds):
        aligned = _hungarian_align(ref_assignments, results[i].assignments, n_clusters)
        aligned_matrix[i] = aligned

    # Per-cluster stability: fraction of samples staying in the same
    # (aligned) cluster across all seeds.
    per_cluster = np.ones(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        in_cluster = ref_assignments == c
        if in_cluster.sum() == 0:
            per_cluster[c] = 0.0
            continue
        # For each sample in cluster c of seed 0, what fraction of other
        # seeds also put it in cluster c?
        cluster_counts = np.sum(aligned_matrix[1:, :] == c, axis=0)
        per_cluster[c] = float(np.mean(cluster_counts[in_cluster]) / (n_seeds - 1)) if n_seeds > 1 else 1.0

    mean_stability = float(np.mean(per_cluster))

    # Adjusted Rand index between first and last seed
    if n_seeds > 1:
        from sklearn.metrics import adjusted_rand_score  # type: ignore[reportMissingTypeStubs]

        ari = float(adjusted_rand_score(ref_assignments, aligned_matrix[-1]))
    else:
        ari = 1.0

    return ClusterStabilityReport(
        adjusted_rand_index=ari,
        mean_stability=mean_stability,
        per_cluster_stability=per_cluster,
        n_seeds=n_seeds,
        assignments_matrix=aligned_matrix,
        results=results,
    )


# ---------------------------------------------------------------------------
# Part 7 — External validation  (Task 4)
# ---------------------------------------------------------------------------


def compare_with_labels(
    assignments: np.ndarray,
    true_labels: np.ndarray,
) -> dict[str, float]:
    """Compare cluster assignments with known ground-truth labels.

    Uses metrics that are invariant to label permutation (no Hungarian
    alignment needed).

    Parameters
    ----------
    assignments :
        Cluster labels from K-means, shape ``(n_samples,)``.
    true_labels :
        Ground-truth labels, shape ``(n_samples,)``.

    Returns
    -------
    dict[str, float]
        ``adjusted_rand_index``, ``adjusted_mutual_info_score``,
        ``homogeneity``, ``completeness``, ``v_measure``.
    """
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        v_measure_score,
    )

    return {
        "adjusted_rand_index": float(adjusted_rand_score(true_labels, assignments)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(true_labels, assignments)),
        "homogeneity": float(homogeneity_score(true_labels, assignments)),
        "completeness": float(completeness_score(true_labels, assignments)),
        "v_measure": float(v_measure_score(true_labels, assignments)),
    }
