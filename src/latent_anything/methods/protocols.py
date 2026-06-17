"""Public Protocol for stateful dimensionality-reduction methods.

Frozen at Method #3 (SAE), Sprint 6 — this is the canonical ``Method`` shape
for stateful dimensionality-reduction methods in the latent-anything framework.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Method(Protocol):
    """Structural protocol for stateful dimensionality-reduction methods.

    A ``Method`` represents a learnable transformation from a high-dimensional
    input space to a lower-dimensional latent representation. It follows a
    two-phase lifecycle:

    1. **Fit**: ``fit(data)`` learns the transformation parameters from data.
    2. **Transform**: ``transform(data)`` applies the learned transformation.

    The combined ``fit_transform(data)`` convenience method is provided via
    the internal ``_MethodBase`` base class, but conforming classes may also
    implement it directly if an optimised joint implementation exists.

    All public input and output is ``numpy.ndarray``. Internal implementations
    may use PyTorch, scikit-learn, or other backends, but the public surface
    is always NumPy.

    .. note::

        This is a structural (duck-typed) Protocol. Classes conform by
        providing the required methods with matching signatures — they do
        **not** need to inherit from ``Method`` or import it.

    Parameters
    ----------
    fit(data : np.ndarray) -> None
        Learn the transformation from ``data``.
    transform(data : np.ndarray) -> np.ndarray
        Apply the learned transformation to ``data``.
    """

    def fit(self, data: np.ndarray) -> None:
        """Fit the method to the data.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, n_features)``.
        """
        ...  # pragma: no cover

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
        ...  # pragma: no cover
