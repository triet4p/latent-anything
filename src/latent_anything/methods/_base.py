"""UNSTABLE — internal base shape for stateful dimensionality-reduction methods.

.. warning::
    Do **not** depend on this class. It is a tentative shared shape sketched
    when instance #2 (UMAP) landed alongside instance #1 (PCA), following
    the Rule of Three (INCREMENTAL.md §4a). It will be **replaced** when
    Method #3 (a different philosophy) arrives in Sprint 6. The underscore
    prefix means it is **internal** — not part of the public API, not in
    ``__all__``, and not exported from the top-level package.
"""

from __future__ import annotations

import numpy as np


class _MethodBase:  # pyright: ignore[reportUnusedClass]
    """Tentative internal base for stateful dimensionality-reduction methods.

    Provides the ``fit_transform`` convenience method (calling ``fit`` then
    ``transform``). Subclasses must implement ``fit`` and ``transform``.

    This shape is **unstable** and will be replaced when a third method with
    a different philosophy appears (Sprint 6). Do not add ``save``/``load``
    or any other abstraction yet.
    """

    def fit(self, data: np.ndarray) -> None:
        """Fit the method to the data.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.
        """
        raise NotImplementedError  # pragma: no cover

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to the fitted embedding space.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Transformed array of shape ``(n_samples, n_components)``.
        """
        raise NotImplementedError  # pragma: no cover

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Transformed array of shape ``(n_samples, n_components)``.
        """
        self.fit(data)
        return self.transform(data)
