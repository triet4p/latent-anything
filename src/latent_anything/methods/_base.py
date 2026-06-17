"""Internal convenience base backed by the frozen ``Method`` Protocol.

Provides ``fit_transform`` as a default implementation for subclasses.
This is an internal convenience — all public API surface is the ``Method``
Protocol. Classes conform to ``Method`` via structural duck-typing, not
inheritance.
"""

from __future__ import annotations

import numpy as np


class _MethodBase:  # pyright: ignore[reportUnusedClass]
    """Internal convenience base backed by the frozen ``Method`` Protocol.

    Provides the ``fit_transform`` convenience method (calling ``fit`` then
    ``transform``). Subclasses must implement ``fit`` and ``transform``.

    This class is **internal** — not part of the public API, not in
    ``__all__``. The public API is the ``Method`` Protocol. Subclasses
    conform to ``Method`` via structural duck-typing.
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
