"""Deterministic dictionary learning with held-out sparse-reconstruction metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.decomposition import (  # pyright: ignore[reportMissingTypeStubs]
    DictionaryLearning as _SklearnDictionaryLearning,
)

TransformAlgorithm = Literal["lasso_lars", "lasso_cd", "lars", "omp", "threshold"]


@dataclass(frozen=True)
class DictionaryLearningConfig:
    """Configuration for a bounded, reproducible dictionary-learning fit."""

    n_components: int = 8
    alpha: float = 0.1
    max_iter: int = 500
    tol: float = 1e-6
    random_state: int = 79
    val_fraction: float = 0.2
    transform_algorithm: TransformAlgorithm = "omp"
    transform_n_nonzero_coefs: int = 2

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be between zero and one")
        if self.transform_n_nonzero_coefs < 1:
            raise ValueError("transform_n_nonzero_coefs must be positive")


@dataclass(frozen=True)
class DictionaryLearningEvaluation:
    """Held-out metrics and split metadata from one fitted dictionary."""

    config: DictionaryLearningConfig
    n_train: int
    n_val: int
    train_reconstruction_mse: float
    val_reconstruction_mse: float
    train_mean_l0: float
    val_mean_l0: float
    train_baseline_mse: float
    val_baseline_mse: float
    train_codes: np.ndarray
    val_codes: np.ndarray


class DictionaryLearning:
    """Fit a sparse overcomplete dictionary without validation leakage."""

    def __init__(self, config: DictionaryLearningConfig | None = None) -> None:
        self.config = config or DictionaryLearningConfig()
        self._model: _SklearnDictionaryLearning | None = None
        self._mean: np.ndarray | None = None
        self.train_indices_: np.ndarray | None = None
        self.validation_indices_: np.ndarray | None = None
        self.evaluation_: DictionaryLearningEvaluation | None = None

    @staticmethod
    def _validate(data: np.ndarray, name: str = "data") -> np.ndarray:
        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError(f"{name} must be a two-dimensional array with at least two rows")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        return values

    def fit(self, data: np.ndarray) -> DictionaryLearningEvaluation:
        """Fit on the deterministic train split and score the held-out split."""
        values = self._validate(data)
        rng = np.random.default_rng(self.config.random_state)
        permutation = rng.permutation(values.shape[0])
        split = int(values.shape[0] * (1.0 - self.config.val_fraction))
        if split < 1 or split >= values.shape[0]:
            raise ValueError("val_fraction leaves no train or validation samples")
        self.train_indices_ = permutation[:split]
        self.validation_indices_ = permutation[split:]
        train = values[self.train_indices_]
        validation = values[self.validation_indices_]
        mean = train.mean(axis=0)
        self._mean = mean
        centered_train = train - mean
        centered_validation = validation - mean
        self._model = _SklearnDictionaryLearning(
            n_components=self.config.n_components,
            alpha=self.config.alpha,  # pyright: ignore[reportArgumentType]
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            fit_algorithm="lars",
            transform_algorithm=self.config.transform_algorithm,
            transform_n_nonzero_coefs=self.config.transform_n_nonzero_coefs,
            random_state=self.config.random_state,
        )
        train_codes = np.asarray(self._model.fit_transform(centered_train), dtype=np.float64)
        val_codes = np.asarray(self._model.transform(centered_validation), dtype=np.float64)
        train_reconstruction = self._reconstruct_centered(train_codes) + mean
        val_reconstruction = self._reconstruct_centered(val_codes) + mean
        self.evaluation_ = DictionaryLearningEvaluation(
            config=self.config,
            n_train=len(train),
            n_val=len(validation),
            train_reconstruction_mse=float(np.mean((train - train_reconstruction) ** 2)),
            val_reconstruction_mse=float(np.mean((validation - val_reconstruction) ** 2)),
            train_mean_l0=float(np.count_nonzero(np.abs(train_codes) > 1e-10, axis=1).mean()),
            val_mean_l0=float(np.count_nonzero(np.abs(val_codes) > 1e-10, axis=1).mean()),
            train_baseline_mse=float(np.mean((train - train.mean(axis=0)) ** 2)),
            val_baseline_mse=float(np.mean((validation - train.mean(axis=0)) ** 2)),
            train_codes=train_codes,
            val_codes=val_codes,
        )
        return self.evaluation_

    def _require_fitted(self) -> tuple[_SklearnDictionaryLearning, np.ndarray]:
        if self._model is None or self._mean is None:
            raise RuntimeError("dictionary must be fitted before use")
        return self._model, self._mean

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Encode samples using the fitted sparse dictionary."""
        model, mean = self._require_fitted()
        values = self._validate(data)
        return np.asarray(model.transform(values - mean), dtype=np.float64)

    def _reconstruct_centered(self, codes: np.ndarray) -> np.ndarray:
        model, _ = self._require_fitted()
        return np.asarray(model.inverse_transform(codes), dtype=np.float64)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """Encode and decode samples in the original feature coordinates."""
        _, mean = self._require_fitted()
        values = self._validate(data)
        return self._reconstruct_centered(self.transform(values)) + mean

    @property
    def components_(self) -> np.ndarray:
        """Return learned dictionary atoms as ``(n_components, n_features)``."""
        model, _ = self._require_fitted()
        return np.asarray(model.components_, dtype=np.float64)
