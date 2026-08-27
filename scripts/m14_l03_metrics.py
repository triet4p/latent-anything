"""Fixed-split probe metrics and predeclared uncertainty calculations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[reportMissingTypeStubs]
from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingTypeStubs]

from latent_anything._mlp_training import train_mlp
from latent_anything.methods.pca import PCA
from latent_anything.mlp_probe import MLPProbeConfig
from latent_anything.probes import LinearProbeConfig


def wilson_95(correct: int, total: int) -> tuple[float, float]:
    """Wilson score interval for independent held-out correctness rows."""
    if total < 1 or not 0 <= correct <= total:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    p = correct / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(centre - radius), float(centre + radius)


def paired_bootstrap(
    real: np.ndarray, control: np.ndarray, *, resamples: int = 10000, seed: int = 7901
) -> dict[str, Any]:
    """Bootstrap paired correctness differences, sampling held-out examples."""
    real_ok, control_ok = np.asarray(real, dtype=bool), np.asarray(control, dtype=bool)
    if real_ok.shape != control_ok.shape or real_ok.ndim != 1 or not len(real_ok):
        raise ValueError("paired correctness vectors must be non-empty and equal length")
    observed = real_ok.astype(np.float64) - control_ok.astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(observed), size=(resamples, len(observed)))
    estimates = observed[draws].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(observed.mean()),
        "lower": float(low),
        "upper": float(high),
        "resamples": resamples,
        "seed": seed,
        "sampling_unit": "held-out example",
    }


def _scale(
    features: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(features[train])
    return (
        np.asarray(scaler.transform(features[train])),
        np.asarray(scaler.transform(features[val])),
        np.asarray(scaler.transform(features[test])),
    )


def _linear_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    masks: Mapping[str, np.ndarray],
    config: LinearProbeConfig,
    *,
    shuffled: bool = False,
) -> dict[str, Any]:
    train, val, test = masks["train"], masks["val"], masks["test"]
    train_x, val_x, test_x = (
        _scale(features, train, val, test) if config.standardize else (features[train], features[val], features[test])
    )
    train_y = labels[train].copy()
    if shuffled:
        train_y = np.random.default_rng(7900).permutation(train_y)
    model = LogisticRegression(
        C=config.C,
        solver=config.solver,
        max_iter=config.max_iter,
        random_state=config.random_state,
        class_weight=config.class_weight,
        fit_intercept=config.fit_intercept,
    )
    model.fit(train_x, train_y)
    test_pred = np.asarray(model.predict(test_x))
    val_pred = np.asarray(model.predict(val_x))
    return {
        "predictions": test_pred,
        "val_accuracy": float(np.mean(val_pred == labels[val])),
        "test_accuracy": float(np.mean(test_pred == labels[test])),
        "n_params": int(np.asarray(model.coef_).size + np.asarray(model.intercept_).size),
    }


def evaluate_linear(
    features: np.ndarray, labels: np.ndarray, masks: Mapping[str, np.ndarray], config: LinearProbeConfig
) -> dict[str, Any]:
    """Evaluate a fixed grouped split; private because LinearProbe owns its split."""
    real = _linear_predictions(features, labels, masks, config)
    control = _linear_predictions(features, labels, masks, config, shuffled=True)
    real_ok = real["predictions"] == labels[masks["test"]]
    control_ok = control["predictions"] == labels[masks["test"]]
    real["control_predictions"] = control["predictions"]
    real["bootstrap"] = paired_bootstrap(real_ok, control_ok)
    real["wilson_95"] = wilson_95(int(real_ok.sum()), int(real_ok.size))
    return real


def _mlp_once(
    features: np.ndarray,
    labels: np.ndarray,
    masks: Mapping[str, np.ndarray],
    config: MLPProbeConfig,
    *,
    shuffled: bool = False,
) -> dict[str, Any]:
    train, val, test = masks["train"], masks["val"], masks["test"]
    train_x, val_x, test_x = (
        _scale(features, train, val, test) if config.standardize else (features[train], features[val], features[test])
    )
    train_y = labels[train].copy()
    if shuffled:
        train_y = np.random.default_rng(7900).permutation(train_y)
    result = train_mlp(
        train_x,
        train_y,
        val_x,
        labels[val],
        test_x,
        labels[test],
        n_features=features.shape[1],
        n_classes=len(np.unique(labels)),
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
        max_epochs=config.max_epochs,
        early_stopping_patience=config.early_stopping_patience,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        random_state=config.random_state,
    )
    return {
        "predictions": result.predictions,
        "test_accuracy": result.accuracy,
        "val_accuracy": result.val_accuracy,
        "n_params": result.n_params,
        "architecture": result.architecture,
        "n_epochs": result.n_epochs,
        "stopped_early": result.stopped_early,
    }


def evaluate_mlp(
    features: np.ndarray, labels: np.ndarray, masks: Mapping[str, np.ndarray], config: MLPProbeConfig
) -> dict[str, Any]:
    """Evaluate bounded MLP and train-label-preserving held-out control."""
    real = _mlp_once(features, labels, masks, config)
    control = _mlp_once(features, labels, masks, config, shuffled=True)
    real_ok = real["predictions"] == labels[masks["test"]]
    control_ok = control["predictions"] == labels[masks["test"]]
    real["control_predictions"] = control["predictions"]
    real["control_accuracy"] = float(control["test_accuracy"])
    real["bootstrap"] = paired_bootstrap(real_ok, control_ok)
    real["api_shuffled_label_diagnostic"] = "run with nonlinear_memorization_test; report as API diagnostic only"
    return real


def fit_train_only_pca(
    features: np.ndarray, masks: Mapping[str, np.ndarray], components: int = 32
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the existing PCA method on train rows only and transform all rows."""
    pca = PCA(n_components=components)
    pca.fit(features[masks["train"]])
    transformed = pca.transform(features)
    return transformed, {
        "components": components,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "fit_scope": "train rows only",
    }


def compression_ok(pca32_accuracy: float, full_hidden_accuracy: float, tolerance: float = 0.10) -> bool:
    return bool(pca32_accuracy >= full_hidden_accuracy - tolerance)
