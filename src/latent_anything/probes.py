"""Label-aware linear classification probing for latent representations.

This module provides a ``LinearProbe`` class that trains a logistic regression
classifier on latent features to measure how accessible a label is from the
representation. It is the canonical linear-probing tool for the project and
replaces the earlier centroid-based ``probe_accuracy`` heuristic.

Probes are **not** forced through the ``Method``/``AnalysisPipeline`` lifecycle
because they require labels during fitting. Instead they are config-constructable
and registered under the ``"analysis"`` kind for semantic taxonomy.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (sklearn has incomplete type stubs — these warnings are noise)

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression  # type: ignore[reportMissingTypeStubs]
from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingTypeStubs]

from latent_anything._probe_split import stratified_split

# ── Configuration ──────────────────────────────────────────────────────────


class LinearProbeConfig(BaseModel):
    """Pydantic config for a :class:`LinearProbe`.

    Parameters
    ----------
    C : float
        Inverse regularization strength (smaller = stronger regularization).
        Defaults to 1.0.
    solver : str
        Solver for ``LogisticRegression``.  ``"lbfgs"`` is the default and
        works well for small-to-medium datasets.
    max_iter : int
        Maximum solver iterations.
    test_size : float
        Fraction of the full dataset held out for final testing (0 < value < 1).
    val_size : float
        Fraction of the **training** split held out for validation.
        E.g. ``test_size=0.3, val_size=0.1`` gives 63 % train / 7 % val / 30 %
        test of the full data.
    random_state : int
        Deterministic seed for all randomness (split, solver, shuffling).
    standardize : bool
        Whether to z-score features using statistics computed on the **training**
        split only (leakage guard).
    class_weight : str | None
        ``"balanced"`` automatically adjusts weights inversely proportional to
        class frequencies.  ``None`` leaves all classes equally weighted.
    fit_intercept : bool
        Whether to fit a bias term.
    """

    C: float = Field(default=1.0, gt=0, description="Inverse regularization strength")
    solver: str = Field(default="lbfgs", description="Solver for LogisticRegression")
    max_iter: int = Field(default=1000, ge=100, description="Maximum solver iterations")
    test_size: float = Field(default=0.3, gt=0, lt=1, description="Fraction held out for testing")
    val_size: float = Field(default=0.1, ge=0, lt=1, description="Fraction of train held out for validation")
    random_state: int = Field(default=0, ge=0, description="Deterministic seed")
    standardize: bool = Field(default=True, description="Z-score features on training statistics")
    class_weight: str | None = Field(default="balanced", description="Class-weighting strategy")
    fit_intercept: bool = Field(default=True, description="Fit bias term")


# ── Typed result ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LinearProbeResult:
    """Typed result from fitting a :class:`LinearProbe`.

    Attributes
    ----------
    accuracy : float
        Accuracy on the held-out test split.
    val_accuracy : float
        Accuracy on the held-out validation split.
    classes : np.ndarray
        Unique class labels seen during training.
    predictions : np.ndarray
        Predicted class label for each test sample  (1-D, ``n_test``).
    probabilities : np.ndarray
        Per-class probabilities for test samples  (``n_test × n_classes``).
    coefficients : np.ndarray
        Learned weight coefficients.
        For binary: ``(n_features,)``; for multiclass: ``(n_classes, n_features)``.
    intercept : np.ndarray
        Bias term(s).  Shape ``(1,)`` for binary, ``(n_classes,)`` for multiclass.
    n_iter : int
        Number of solver iterations actually used.
    train_indices : np.ndarray
        Boolean mask (``n_samples``) marking training samples.
    val_indices : np.ndarray
        Boolean mask (``n_samples``) marking validation samples (empty if
        ``val_size == 0``).
    test_indices : np.ndarray
        Boolean mask (``n_samples``) marking test samples.
    feature_means : np.ndarray | None
        Per-feature means used for standardization (``None`` if not standardized).
    feature_stds : np.ndarray | None
        Per-feature standard deviations used for standardization.
    config : LinearProbeConfig
        The config that produced this result.
    provenance : dict
        Metadata about the run (version, call context).
    """

    accuracy: float
    val_accuracy: float
    classes: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    n_iter: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    feature_means: np.ndarray | None
    feature_stds: np.ndarray | None
    config: LinearProbeConfig
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict (deep copies array attributes)."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, LinearProbeConfig):
                out[key] = value.model_dump()
            else:
                out[key] = value
        return out


# ── Control baselines ──────────────────────────────────────────────────────


def _stratified_split(
    labels: np.ndarray,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for the shared probe split implementation."""
    return stratified_split(
        labels,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )


@dataclass(frozen=True)
class ControlBaselines:
    """Accuracy of simple baselines on the same split as the probe.

    Attributes
    ----------
    majority_class : float
        Accuracy of always predicting the most frequent training class.
    shuffled_label : float
        Probe accuracy after shuffling labels (tests label-signal specificity).
    raw_input : float
        Probe accuracy on raw (un-pooled) input features for comparison.
    """

    majority_class: float
    shuffled_label: float
    raw_input: float


def compute_controls(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    random_state: int = 0,
) -> ControlBaselines:
    """Compute baseline accuracies on the **same** train/test split.

    Parameters
    ----------
    features : np.ndarray
        Full feature matrix ``(n_samples, n_features)``.
    labels : np.ndarray
        Full label vector ``(n_samples,)``.
    train_indices : np.ndarray
        Boolean mask for training samples.
    test_indices : np.ndarray
        Boolean mask for test samples.
    random_state : int
        Seed for label shuffling.

    Returns
    -------
    ControlBaselines
        Three baseline accuracies.
    """
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    train_features = features[train_indices]
    test_features = features[test_indices]

    # Majority-class baseline
    classes, counts = np.unique(train_labels, return_counts=True)
    majority = classes[np.argmax(counts)]
    majority_acc = float(np.mean(test_labels == majority))

    # Shuffled-label baseline (fit probe on permuted labels)
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(train_labels)
    # Use a fast one-vs-rest probe on shuffled labels
    shuffled_probe = _fast_probe(train_features, shuffled, test_features, test_labels, random_state)
    shuffled_acc = shuffled_probe.accuracy

    # Raw-input baseline (fit probe on raw features)
    raw_probe = _fast_probe(train_features, labels[train_indices], test_features, test_labels, random_state)
    raw_acc = raw_probe.accuracy

    return ControlBaselines(
        majority_class=majority_acc,
        shuffled_label=shuffled_acc,
        raw_input=raw_acc,
    )


def _fast_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    random_state: int,
) -> LinearProbeResult:
    """Fit a quick logistic-regression probe without config building."""
    scaler = StandardScaler()
    train_x_scaled: np.ndarray = np.asarray(scaler.fit_transform(train_x))
    test_x_scaled: np.ndarray = np.asarray(scaler.transform(test_x))

    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=random_state, class_weight="balanced")
    lr.fit(train_x_scaled, train_y)
    preds: np.ndarray = np.asarray(lr.predict(test_x_scaled))
    probs: np.ndarray = np.asarray(lr.predict_proba(test_x_scaled))
    accuracy = float(np.mean(preds == test_y))
    coefs: np.ndarray = np.asarray(lr.coef_)
    n_classes = len(np.unique(train_y))

    return LinearProbeResult(
        accuracy=accuracy,
        val_accuracy=0.0,
        classes=np.unique(train_y),
        predictions=preds,
        probabilities=probs,
        coefficients=coefs[0] if n_classes == 2 and coefs.shape[0] == 1 else coefs,
        intercept=np.asarray(lr.intercept_),
        n_iter=int(lr.n_iter_[0]) if hasattr(lr, "n_iter_") else 0,
        train_indices=np.full(len(train_x) + len(test_x), False),
        val_indices=np.full(len(train_x) + len(test_x), False),
        test_indices=np.full(len(train_x) + len(test_x), False),
        feature_means=np.asarray(scaler.mean_) if hasattr(scaler, "mean_") else None,
        feature_stds=np.asarray(scaler.scale_) if hasattr(scaler, "scale_") else None,
        config=LinearProbeConfig(random_state=random_state),
        provenance={"method": "fast_probe"},
    )


# ── Cross-seed stability ───────────────────────────────────────────────────


@dataclass(frozen=True)
class CrossSeedReport:
    """Aggregated probe results across multiple random seeds.

    Attributes
    ----------
    accuracies : list[float]
        Test accuracy per seed.
    val_accuracies : list[float]
        Validation accuracy per seed.
    mean_accuracy : float
        Mean test accuracy across seeds.
    ci95 : float
        95 % confidence interval half-width (1.96 × std / sqrt(n_seeds)).
    std_accuracy : float
        Standard deviation of test accuracy across seeds.
    min_accuracy : float
        Minimum test accuracy across seeds.
    max_accuracy : float
        Maximum test accuracy across seeds.
    n_seeds : int
        Number of seeds evaluated.
    results : list[LinearProbeResult]
        Full results for each seed (for coefficient / score inspection).
    """

    accuracies: list[float]
    val_accuracies: list[float]
    mean_accuracy: float
    ci95: float
    std_accuracy: float
    min_accuracy: float
    max_accuracy: float
    n_seeds: int
    results: list[LinearProbeResult]


def cross_seed_evaluation(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seeds: list[int] | range | None = None,
    n_seeds: int = 5,
    config: LinearProbeConfig | None = None,
) -> CrossSeedReport:
    """Evaluate probe stability across multiple random seeds.

    Parameters
    ----------
    features : np.ndarray
        ``(n_samples, n_features)`` feature matrix.
    labels : np.ndarray
        ``(n_samples,)`` label vector.
    seeds : list[int] | range, optional
        Explicit list of seeds.  Mutually exclusive with ``n_seeds``.
    n_seeds : int
        Number of seeds to evaluate (0 … *n_seeds* - 1). Only used when
        ``seeds`` is ``None``.
    config : LinearProbeConfig | None
        Base config (``random_state`` is overridden per seed).

    Returns
    -------
    CrossSeedReport
        Aggregated statistics and per-seed results.
    """
    if seeds is None:
        seeds_list = list(range(n_seeds))
    else:
        seeds_list = list(seeds)
        n_seeds = len(seeds_list)

    base_config = config or LinearProbeConfig()
    results: list[LinearProbeResult] = []

    for seed in seeds_list:
        cfg = base_config.model_copy(update={"random_state": seed})
        probe = LinearProbe(cfg)
        result = probe.fit(features, labels)
        results.append(result)

    accs = np.array([r.accuracy for r in results])
    mean_acc = float(accs.mean())
    std_acc = float(accs.std(ddof=1) if len(accs) > 1 else 0.0)
    ci95 = float(1.96 * std_acc / np.sqrt(len(accs)) if len(accs) > 1 else 0.0)

    return CrossSeedReport(
        accuracies=[r.accuracy for r in results],
        val_accuracies=[r.val_accuracy for r in results],
        mean_accuracy=mean_acc,
        ci95=ci95,
        std_accuracy=std_acc,
        min_accuracy=float(accs.min()),
        max_accuracy=float(accs.max()),
        n_seeds=n_seeds,
        results=results,
    )


# ── Main probe class ───────────────────────────────────────────────────────


class LinearProbe:
    """Label-aware linear classification probe for latent representations.

    Fits a logistic regression classifier with leakage-guarded splitting,
    training-only standardization, and configurable regularization.

    This class does **not** conform to the ``Method`` protocol (which expects
    unlabeled ``fit(data)`` / ``transform(data)``) and is deliberately kept
    outside the ``Method`` / ``AnalysisPipeline`` lifecycle.

    Parameters
    ----------
    config : LinearProbeConfig, optional
        Probe configuration.  Defaults to ``LinearProbeConfig()``.

    Examples
    --------
    >>> import numpy as np
    >>> from latent_anything.probes import LinearProbe
    >>> rng = np.random.default_rng(0)
    >>> features = rng.normal(0, 1, (100, 5))
    >>> features[:50] += 2.0  # make first 50 separable
    >>> labels = np.repeat([0, 1], 50)
    >>> probe = LinearProbe()
    >>> result = probe.fit(features, labels)
    >>> result.accuracy > 0.8
    True
    >>> result.predictions.shape == (30,)  # 30 % test split
    True
    >>> result.probabilities.shape == (30, 2)
    True
    """

    def __init__(self, config: LinearProbeConfig | None = None) -> None:
        self._config = config if config is not None else LinearProbeConfig()
        self._fitted: bool = False
        self._result: LinearProbeResult | None = None
        self._scaler: StandardScaler | None = None
        self._classifier: LogisticRegression | None = None
        self._train_mask: np.ndarray | None = None
        self._val_mask: np.ndarray | None = None
        self._test_mask: np.ndarray | None = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def config(self) -> LinearProbeConfig:
        """Return the probe configuration."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._fitted

    @property
    def result(self) -> LinearProbeResult:
        """Return the last fit result.

        Raises
        ------
        RuntimeError
            If the probe has not been fitted yet.
        """
        if not self._fitted or self._result is None:
            msg = "LinearProbe has not been fitted yet. Call .fit() first."
            raise RuntimeError(msg)
        return self._result

    # ── Fit ───────────────────────────────────────────────────────────

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> LinearProbeResult:
        """Fit the linear probe on labeled features.

        Parameters
        ----------
        features : np.ndarray
            ``(n_samples, n_features)`` feature matrix.
        labels : np.ndarray
            ``(n_samples,)`` label vector.  At least two classes required.
        provenance : dict, optional
            Optional metadata dict attached to the result (e.g. layer index,
            model name).

        Returns
        -------
        LinearProbeResult
            Typed result with accuracy, predictions, coefficients, etc.

        Raises
        ------
        ValueError
            If input shapes are invalid or degenerate.
        """
        cfg = self._config
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)

        # ── Validate ──────────────────────────────────────────────
        if features.ndim != 2:
            msg = f"features must be 2D, got {features.ndim}D"
            raise ValueError(msg)
        if labels.ndim != 1:
            msg = f"labels must be 1D, got {labels.ndim}D"
            raise ValueError(msg)
        if features.shape[0] != labels.shape[0]:
            msg = f"features ({features.shape[0]}) and labels ({labels.shape[0]}) must have same number of samples"
            raise ValueError(msg)
        if features.shape[0] < 2:
            msg = f"need at least 2 samples, got {features.shape[0]}"
            raise ValueError(msg)
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            msg = f"labels must contain at least 2 classes, got {len(unique_classes)}"
            raise ValueError(msg)
        min_samples = np.min([np.sum(labels == c) for c in unique_classes])
        if min_samples < 2:
            msg = f"each class must have at least 2 samples, got minimum {min_samples}"
            raise ValueError(msg)

        # ── Split ─────────────────────────────────────────────────
        train_mask, val_mask, test_mask = _stratified_split(
            labels,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state,
        )
        self._train_mask = train_mask
        self._val_mask = val_mask
        self._test_mask = test_mask

        train_x, train_y = features[train_mask], labels[train_mask]
        val_x, val_y = features[val_mask], labels[val_mask] if val_mask.any() else (None, None)
        test_x, test_y = features[test_mask], labels[test_mask]

        has_val = val_x.shape[0] > 0  # val_x is always an array (may be empty)

        # Leakage guard: standardize on training statistics only
        if cfg.standardize:
            scaler = StandardScaler()
            train_x_scaled = scaler.fit_transform(train_x)
            test_x_scaled: np.ndarray = np.asarray(scaler.transform(test_x))
            val_x_scaled = np.asarray(scaler.transform(val_x)) if has_val else None
            self._scaler = scaler
            feature_means: np.ndarray | None = np.asarray(scaler.mean_) if hasattr(scaler, "mean_") else None
            feature_stds: np.ndarray | None = np.asarray(scaler.scale_) if hasattr(scaler, "scale_") else None
        else:
            train_x_scaled = train_x.copy()
            test_x_scaled = test_x.copy()
            val_x_scaled = val_x.copy() if has_val else None
            feature_means = None
            feature_stds = None

        # ── Train classifier ──────────────────────────────────────
        lr = LogisticRegression(
            C=cfg.C,
            solver=cfg.solver,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
            class_weight=cfg.class_weight,
            fit_intercept=cfg.fit_intercept,
        )
        lr.fit(train_x_scaled, train_y)
        self._classifier = lr

        # ── Evaluate ──────────────────────────────────────────────
        test_preds: np.ndarray = np.asarray(lr.predict(test_x_scaled))
        test_probs: np.ndarray = np.asarray(lr.predict_proba(test_x_scaled))
        accuracy = float(np.mean(test_preds == test_y))

        val_accuracy = 0.0
        if has_val and len(val_y) > 0:
            val_preds: np.ndarray = np.asarray(lr.predict(val_x_scaled))
            val_accuracy = float(np.mean(val_preds == val_y))

        n_classes = len(unique_classes)
        coefs: np.ndarray = np.asarray(lr.coef_)
        # For binary case, store 1D coefficient vector
        coefficients: np.ndarray = coefs[0] if n_classes == 2 and coefs.shape[0] == 1 else coefs

        provenance_dict: dict[str, Any] = dict(provenance or {})
        provenance_dict.setdefault("n_features", features.shape[1])
        provenance_dict.setdefault("n_samples", features.shape[0])
        provenance_dict.setdefault("n_classes", n_classes)

        self._result = LinearProbeResult(
            accuracy=accuracy,
            val_accuracy=val_accuracy,
            classes=unique_classes,
            predictions=test_preds,
            probabilities=test_probs,
            coefficients=coefficients,
            intercept=np.asarray(lr.intercept_),
            n_iter=int(lr.n_iter_[0]) if hasattr(lr, "n_iter_") else 0,
            train_indices=train_mask,
            val_indices=val_mask,
            test_indices=test_mask,
            feature_means=feature_means,
            feature_stds=feature_stds,
            config=cfg,
            provenance=provenance_dict,
        )
        self._fitted = True
        return self._result

    # ── Predict (after fit) ───────────────────────────────────────────

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels for new features using the fitted classifier.

        Parameters
        ----------
        features : np.ndarray
            ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        if not self._fitted or self._classifier is None:
            msg = "LinearProbe has not been fitted yet. Call .fit() first."
            raise RuntimeError(msg)
        x: np.ndarray = np.asarray(features, dtype=np.float64)
        if self._scaler is not None:
            x = np.asarray(self._scaler.transform(x))
        return np.asarray(self._classifier.predict(x))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities for new features.

        Parameters
        ----------
        features : np.ndarray
            ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Per-class probabilities ``(n_samples, n_classes)``.
        """
        if not self._fitted or self._classifier is None:
            msg = "LinearProbe has not been fitted yet. Call .fit() first."
            raise RuntimeError(msg)
        x: np.ndarray = np.asarray(features, dtype=np.float64)
        if self._scaler is not None:
            x = np.asarray(self._scaler.transform(x))
        return np.asarray(self._classifier.predict_proba(x))


# ── Evaluate on representation layers (for real-model integration) ─────────


def evaluate_layers(
    layer_features: Mapping[str | int, np.ndarray],
    labels: np.ndarray,
    *,
    seeds: list[int] | range | None = None,
    n_seeds: int = 5,
    config: LinearProbeConfig | None = None,
) -> dict[str | int, CrossSeedReport]:
    """Evaluate probe accuracy across multiple representation layers.

    Parameters
    ----------
    layer_features : dict[str | int, np.ndarray]
        Mapping from layer key (name or index) to feature matrix.
    labels : np.ndarray
        ``(n_samples,)`` label vector.
    seeds : list[int] | range, optional
        Explicit seed list for cross-seed evaluation.
    n_seeds : int
        Number of seeds (used when ``seeds`` is ``None``).
    config : LinearProbeConfig | None
        Base config.

    Returns
    -------
    dict[str | int, CrossSeedReport]
        Per-layer cross-seed reports.
    """
    reports: dict[str | int, CrossSeedReport] = {}
    for layer_key, feats in layer_features.items():
        reports[layer_key] = cross_seed_evaluation(
            feats,
            labels,
            seeds=seeds,
            n_seeds=n_seeds,
            config=config,
        )
    return reports
