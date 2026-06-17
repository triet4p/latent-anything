"""UMAP dimensionality reduction method, wrapping umap-learn."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from latent_anything.methods._base import _MethodBase  # pyright: ignore[reportPrivateUsage]


class UMAP(_MethodBase):
    """UMAP dimensionality reduction method.

    Stateful method: call ``fit`` to learn the transformation, then
    ``transform`` to apply it. Input and output are numpy arrays.

    This is a concrete hardcoded implementation wrapping ``umap-learn``.
    It will be generalized to the ``_MethodBase`` interface alongside PCA
    (Rule of Three, see INCREMENTAL.md §4a).

    Parameters
    ----------
    n_neighbors : int, optional
        Number of neighbors to use for manifold approximation.
    min_dist : float, optional
        Minimum distance between embedded points.
    n_components : int, optional
        Dimensionality of the target embedding.
    metric : str, optional
        Distance metric to use.
    random_state : int, optional
        Seed for reproducibility.
    **kwargs : Any
        Additional keyword arguments forwarded to ``umap.UMAP``.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        metric: str = "euclidean",
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state
        self._kwargs = kwargs
        self._umap: Any = None

    def fit(self, data: np.ndarray) -> None:
        """Fit UMAP to the data.

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
        import umap  # pyright: ignore[reportMissingTypeStubs]

        self._umap = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            metric=self.metric,
            random_state=self.random_state,
            **self._kwargs,
        )
        self._umap.fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to the fitted UMAP embedding space.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Transformed array of shape ``(n_samples, n_components)``.
        """
        if self._umap is None:
            msg = "UMAP must be fitted before transform"
            raise RuntimeError(msg)
        result = self._umap.transform(data)
        return cast(np.ndarray, result)
