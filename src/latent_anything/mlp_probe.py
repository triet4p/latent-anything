"""Bounded nonlinear MLP probe for latent representation accessibility.

This module provides an ``MLPProbe`` class that trains a small multi-layer
perceptron (MLP) classifier as an information-accessibility upper bound.
It reuses the Sprint 40 :func:`_stratified_split` leakage-guarded splitting
and standardization invariants but keeps its own result type separate from
the linear probe result.

The MLP is deliberately small (default: single hidden layer of 64 units) with
deterministic initialization and early stopping to bound capacity and avoid
overfitting in the probe setting.  The goal is *not* state-of-the-art accuracy
but a calibrated nonlinear baseline: if an MLP can find signal that the linear
probe cannot, that signal is nonlinear-accessible (but not necessarily useful
to a linear decoder).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from latent_anything._mlp_controls import (
    NonlinearControls as NonlinearControls,
)
from latent_anything._mlp_controls import (
    ProbeComparison as ProbeComparison,
)
from latent_anything._mlp_controls import (
    classify_probe_comparison,
)
from latent_anything._mlp_training import train_mlp
from latent_anything._probe_split import stratified_split
from latent_anything.probes import LinearProbeConfig

# Keep pickle/module identity for the long-standing public result types while
# their implementation lives in the focused private controls module.
NonlinearControls.__module__ = __name__
ProbeComparison.__module__ = __name__


# ── Configuration ──────────────────────────────────────────────────────────


class MLPProbeConfig(BaseModel):
    """Pydantic config for a :class:`MLPProbe`.

    Parameters
    ----------
    hidden_sizes : list[int]
        Sizes of hidden layers.  ``[64]`` gives one hidden layer with 64 units.
    activation : str
        Hidden activation: ``"relu"`` or ``"tanh"``.
    max_epochs : int
        Maximum training epochs.
    early_stopping_patience : int
        Number of epochs with no validation improvement before stopping.
    learning_rate : float
        Adam learning rate.
    weight_decay : float
        L2 regularization (AdamW weight decay).
    batch_size : int
        Mini-batch size for training.
    test_size : float
        Fraction held out for testing (matches LinearProbe convention).
    val_size : float
        Fraction of training split held out for validation.
    random_state : int
        Deterministic seed for init, split, and training.
    standardize : bool
        Whether to z-score features using training statistics only.
    """

    hidden_sizes: list[int] = Field(default_factory=lambda: [64], description="Hidden layer sizes")
    activation: str = Field(default="relu", pattern="^(relu|tanh)$", description="Hidden activation")
    max_epochs: int = Field(default=200, ge=1, description="Maximum training epochs")
    early_stopping_patience: int = Field(default=10, ge=1, description="Early stopping patience")
    learning_rate: float = Field(default=1e-3, gt=0, description="Adam learning rate")
    weight_decay: float = Field(default=1e-4, ge=0, description="L2 weight decay")
    batch_size: int = Field(default=32, ge=1, description="Mini-batch size")
    test_size: float = Field(default=0.3, gt=0, lt=1, description="Fraction held out for testing")
    val_size: float = Field(default=0.1, ge=0, lt=1, description="Fraction of train held out for validation")
    random_state: int = Field(default=0, ge=0, description="Deterministic seed")
    standardize: bool = Field(default=True, description="Z-score features on training statistics")


# ── Typed result ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MLPProbeResult:
    """Typed result from fitting an :class:`MLPProbe`.

    Attributes
    ----------
    accuracy : float
        Accuracy on the held-out test split.
    val_accuracy : float
        Accuracy on the held-out validation split.
    classes : np.ndarray
        Unique class labels seen during training.
    predictions : np.ndarray
        Predicted class label for each test sample (1-D, ``n_test``).
    probabilities : np.ndarray
        Per-class probabilities for test samples  (``n_test × n_classes``).
    n_epochs : int
        Number of training epochs actually run.
    stopped_early : bool
        Whether training stopped due to early stopping (not max_epochs).
    architecture : dict
        Architecture description (layer sizes, activation, parameter count).
    n_params : int
        Total number of trainable parameters.
    optimizer : str
        Optimizer name (e.g. ``"Adam"``).
    train_indices : np.ndarray
        Boolean mask (``n_samples``) marking training samples.
    val_indices : np.ndarray
        Boolean mask (``n_samples``) marking validation samples.
    test_indices : np.ndarray
        Boolean mask (``n_samples``) marking test samples.
    feature_means : np.ndarray | None
        Per-feature means used for standardization.
    feature_stds : np.ndarray | None
        Per-feature standard deviations used for standardization.
    config : MLPProbeConfig
        The config that produced this result.
    provenance : dict
        Metadata about the run (seed, model, layer, etc.).
    """

    accuracy: float
    val_accuracy: float
    classes: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    n_epochs: int
    stopped_early: bool
    architecture: dict[str, Any]
    n_params: int
    optimizer: str
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    feature_means: np.ndarray | None
    feature_stds: np.ndarray | None
    config: MLPProbeConfig
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict (deep copies array attributes)."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, MLPProbeConfig):
                out[key] = value.model_dump()
            elif isinstance(value, dict):
                out[key] = dict(value)
            else:
                out[key] = value
        return out


# ── Main probe class ───────────────────────────────────────────────────────


class MLPProbe:
    """Bounded nonlinear probe for latent representation accessibility.

    Trains a small MLP with deterministic initialization, early stopping,
    and training-only feature standardization.  Designed as an upper-bound
    complement to :class:`~latent_anything.probes.LinearProbe`.

    This class does **not** conform to the ``Method`` protocol and is
    deliberately kept outside the ``Method`` / ``AnalysisPipeline`` lifecycle.

    Parameters
    ----------
    config : MLPProbeConfig, optional
        Probe configuration.  Defaults to ``MLPProbeConfig()``.

    Examples
    --------
    >>> import numpy as np
    >>> from latent_anything.mlp_probe import MLPProbe
    >>> rng = np.random.default_rng(0)
    >>> features = rng.normal(0, 1, (100, 5))
    >>> features[:50] += 2.0
    >>> labels = np.repeat([0, 1], 50)
    >>> probe = MLPProbe()
    >>> result = probe.fit(features, labels)
    >>> result.accuracy > 0.8
    True
    >>> result.n_params > 0
    True
    >>> isinstance(result.stopped_early, bool)
    True
    """

    def __init__(self, config: MLPProbeConfig | None = None) -> None:
        self._config = config if config is not None else MLPProbeConfig()
        self._fitted: bool = False
        self._result: MLPProbeResult | None = None
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def config(self) -> MLPProbeConfig:
        """Return the probe configuration."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._fitted

    @property
    def result(self) -> MLPProbeResult:
        """Return the last fit result.

        Raises
        ------
        RuntimeError
            If the probe has not been fitted yet.
        """
        if not self._fitted or self._result is None:
            msg = "MLPProbe has not been fitted yet. Call .fit() first."
            raise RuntimeError(msg)
        return self._result

    # ── Fit ───────────────────────────────────────────────────────────

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> MLPProbeResult:
        """Fit the MLP probe on labeled features.

        Parameters
        ----------
        features : np.ndarray
            ``(n_samples, n_features)`` feature matrix.
        labels : np.ndarray
            ``(n_samples,)`` label vector.  At least two classes required.
        provenance : dict, optional
            Optional metadata dict attached to the result.

        Returns
        -------
        MLPProbeResult
            Typed result with accuracy, predictions, architecture, etc.

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

        n_features = features.shape[1]
        n_classes = len(unique_classes)

        # ── Split (reuse Sprint 40 helper) ────────────────────────
        train_mask, val_mask, test_mask = stratified_split(
            labels,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state,
        )

        train_x, train_y = features[train_mask], labels[train_mask]
        val_x = features[val_mask] if val_mask.any() else None  # type: ignore[reportAssignmentType]
        val_y = labels[val_mask] if val_mask.any() else None  # type: ignore[reportAssignmentType]
        test_x, test_y = features[test_mask], labels[test_mask]
        has_val = val_mask.any() and val_x is not None and val_x.shape[0] > 0

        # ── Standardize (leakage guard) ───────────────────────────
        if cfg.standardize:
            mean_arr = train_x.mean(axis=0)
            std_arr = train_x.std(axis=0, ddof=0)
            std_arr = np.where(std_arr < 1e-12, 1.0, std_arr)  # avoid division by zero

            train_x_scaled = (train_x - mean_arr) / std_arr
            test_x_scaled = (test_x - mean_arr) / std_arr
            val_x_scaled = (val_x - mean_arr) / std_arr if has_val else None
            self._feature_means = mean_arr
            self._feature_stds = std_arr
        else:
            train_x_scaled = train_x.copy()
            test_x_scaled = test_x.copy()
            val_x_scaled = val_x.copy() if has_val and val_x is not None else None
        training = train_mlp(
            train_x_scaled,
            train_y,
            val_x_scaled if has_val else None,
            val_y if has_val else None,
            test_x_scaled,
            test_y,
            n_features=n_features,
            n_classes=n_classes,
            hidden_sizes=cfg.hidden_sizes,
            activation=cfg.activation,
            max_epochs=cfg.max_epochs,
            early_stopping_patience=cfg.early_stopping_patience,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            random_state=cfg.random_state,
        )

        provenance_dict: dict[str, Any] = dict(provenance or {})
        provenance_dict.setdefault("n_features", n_features)
        provenance_dict.setdefault("n_samples", features.shape[0])
        provenance_dict.setdefault("n_classes", n_classes)

        self._result = MLPProbeResult(
            accuracy=training.accuracy,
            val_accuracy=training.val_accuracy,
            classes=unique_classes,
            predictions=training.predictions,
            probabilities=training.probabilities,
            n_epochs=training.n_epochs,
            stopped_early=training.stopped_early,
            architecture=training.architecture,
            n_params=training.n_params,
            optimizer=training.optimizer,
            train_indices=train_mask,
            val_indices=val_mask,
            test_indices=test_mask,
            feature_means=self._feature_means,
            feature_stds=self._feature_stds,
            config=cfg,
            provenance=provenance_dict,
        )
        self._fitted = True
        return self._result

    # ── Predict (after fit) ───────────────────────────────────────────

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Reject prediction because fitted model state is not serialized.

        ``MLPProbe.fit`` stores the fitted predictions in its result, but this
        compact probe intentionally does not retain a serialized estimator for
        inference on new features. Calling this method therefore always raises
        ``NotImplementedError`` after the fitted-state precondition succeeds.

        Parameters
        ----------
        features : np.ndarray
            ``(n_samples, n_features)``.

        Raises
        ------
        RuntimeError
            If the probe has not been fitted yet.
        NotImplementedError
            If the probe is fitted; model-state serialization is not
            implemented. Use ``result.predictions`` from :meth:`fit` instead.
        """
        del features  # unused — model state serialization not yet implemented
        if not self._fitted or self._result is None:
            msg = "MLPProbe has not been fitted yet. Call .fit() first."
            raise RuntimeError(msg)
        # Predict requires model state serialization which is not implemented.
        # Use the result.predictions from .fit() instead.
        msg = (
            "MLPProbe.predict() is not available without model state serialization. "
            "Use the result.predictions from .fit() instead."
        )
        raise NotImplementedError(msg)


# ── Controls & memorization tests ──────────────────────────────────────────


def nonlinear_memorization_test(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    config: MLPProbeConfig | None = None,
    selectivity_threshold: float = 2.0,
) -> NonlinearControls:
    """Check whether an MLP memorizes label noise on the given data.

    Trains an MLP probe on shuffled labels and compares its accuracy to
    chance.  A ratio above ``selectivity_threshold`` (default 2× chance)
    indicates the model is memorizing spurious correlations rather than
    learning a genuine representation-label mapping.

    Parameters
    ----------
    features : np.ndarray
        ``(n_samples, n_features)`` feature matrix.
    labels : np.ndarray
        ``(n_samples,)`` label vector.
    config : MLPProbeConfig, optional
        Probe config (seed is internally overridden for the shuffle).
    selectivity_threshold : float
        Maximum allowed ratio of shuffled accuracy to chance accuracy.
        Default 2.0 (2× chance).
    shuffle_seed_offset : int, optional
        Offset applied to the config's random_state for the shuffle run,
        ensuring a different seed from the real-label run.

    Returns
    -------
    NonlinearControls
        Shuffled-label accuracy, memorization ratio, pass/fail.

    Raises
    ------
    ValueError
        If fewer than 2 classes are present.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        msg = f"need at least 2 classes for memorization test, got {len(unique)}"
        raise ValueError(msg)
    chance = 1.0 / len(unique)

    base_cfg = config if config is not None else MLPProbeConfig()
    assert base_cfg is not None
    shuffle_cfg = base_cfg.model_copy(update={"random_state": base_cfg.random_state + 999})

    rng = np.random.default_rng(42)
    shuffled_labels = rng.permutation(labels)

    probe = MLPProbe(shuffle_cfg)
    result = probe.fit(features, shuffled_labels)
    shuffled_acc = result.accuracy

    ratio = shuffled_acc / chance if chance > 0 else float("inf")
    passed = ratio <= selectivity_threshold

    return NonlinearControls(
        shuffled_label_accuracy=shuffled_acc,
        memorization_ratio=ratio,
        passed_memorization_test=passed,
        chance_accuracy=chance,
    )


# ── Linear vs nonlinear comparison ─────────────────────────────────────────


def compare_probes(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    linear_config: LinearProbeConfig | None = None,
    nonlinear_config: MLPProbeConfig | None = None,
    seed: int = 0,
    accuracy_threshold: float = 0.55,
    gap_threshold: float = 0.05,
) -> ProbeComparison:
    """Compare linear and nonlinear probe on identical data.

    Parameters
    ----------
    features : np.ndarray
        ``(n_samples, n_features)`` feature matrix.
    labels : np.ndarray
        ``(n_samples,)`` label vector.
    linear_config : LinearProbeConfig, optional
        Config for the linear probe.
    nonlinear_config : MLPProbeConfig, optional
        Config for the nonlinear probe.
    seed : int
        Seed for both probes (ensures identical splits).
    accuracy_threshold : float
        Minimum accuracy to consider the probe "supported" (above chance).
    gap_threshold : float
        Minimum gap to consider the difference meaningful.

    Returns
    -------
    ProbeComparison
        Side-by-side comparison with a qualitative classification.
    """
    # Keep the estimator implementation import lazy; the config type is
    # imported above because it is part of this function's public contract.
    from latent_anything.probes import cross_seed_evaluation

    lin_cfg = linear_config or LinearProbeConfig(random_state=seed)
    nonlin_cfg = nonlinear_config or MLPProbeConfig(random_state=seed)

    lin_report = cross_seed_evaluation(features, labels, seeds=[seed], config=lin_cfg)
    lin_acc = lin_report.mean_accuracy
    lin_ci95 = lin_report.ci95

    nl_probe = MLPProbe(nonlin_cfg)
    nl_result = nl_probe.fit(features, labels)
    nl_acc = nl_result.accuracy

    # Memorization check
    mem_ctl = nonlinear_memorization_test(features, labels, config=nonlin_cfg)
    return classify_probe_comparison(
        linear_accuracy=lin_acc,
        nonlinear_accuracy=nl_acc,
        linear_ci95=lin_ci95,
        memorization_prone=not mem_ctl.passed_memorization_test,
        accuracy_threshold=accuracy_threshold,
        gap_threshold=gap_threshold,
    )
