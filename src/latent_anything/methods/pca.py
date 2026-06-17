"""PCA dimensionality reduction method, wrapping scikit-learn."""

from __future__ import annotations

from typing import cast

import numpy as np
from sklearn.decomposition import PCA as _SKLearnPCA  # noqa: N811  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.methods._base import _MethodBase  # pyright: ignore[reportPrivateUsage]


class PCA(_MethodBase):
    """PCA dimensionality reduction method.

    Stateful method: call ``fit`` to learn the transformation, then
    ``transform`` to apply it. Input and output are numpy arrays.

    This is a concrete hardcoded implementation wrapping scikit-learn's PCA.
    It has been migrated to the internal ``_MethodBase`` shape alongside
    UMAP (Rule of Three instance #2, see INCREMENTAL.md §4a).

    Parameters
    ----------
    n_components : int, optional
        Number of principal components to keep. If ``None``,
        ``min(n_samples, n_features)`` is used.
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components
        self._pca: _SKLearnPCA | None = None

    def fit(self, data: np.ndarray) -> None:
        """Fit PCA to the data.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.
        """
        if data.ndim != 2:
            msg = f"Expected 2D array, got {data.ndim}D"
            raise ValueError(msg)
        if data.shape[0] < 1 or data.shape[1] < 1:
            msg = "Data must have at least 1 sample and 1 feature"
            raise ValueError(msg)
        self._pca = _SKLearnPCA(n_components=self.n_components)
        self._pca.fit(data)  # pyright: ignore[reportUnknownMemberType]

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to the fitted PCA space.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Transformed array of shape ``(n_samples, n_components)``.
        """
        if self._pca is None:
            msg = "PCA must be fitted before transform"
            raise RuntimeError(msg)
        result = self._pca.transform(data)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return cast(np.ndarray, result)

    @property
    def components_(self) -> np.ndarray:
        """Principal components (fitted model)."""
        if self._pca is None:
            msg = "PCA must be fitted before accessing components_"
            raise RuntimeError(msg)
        return cast(np.ndarray, self._pca.components_)  # pyright: ignore[reportUnknownMemberType]

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Explained variance ratio per component."""
        if self._pca is None:
            msg = "PCA must be fitted before accessing explained_variance_ratio_"
            raise RuntimeError(msg)
        return cast(np.ndarray, self._pca.explained_variance_ratio_)  # pyright: ignore[reportUnknownMemberType]

    @property
    def mean_(self) -> np.ndarray:
        """Per-feature empirical mean."""
        if self._pca is None:
            msg = "PCA must be fitted before accessing mean_"
            raise RuntimeError(msg)
        return cast(np.ndarray, self._pca.mean_)  # pyright: ignore[reportUnknownMemberType]
