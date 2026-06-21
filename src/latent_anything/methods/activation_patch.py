"""Model-mediated activation patching via encode → patch → decode.

This is B-Method #3 — the third Layer B (Manipulation) method, and the
first *model-mediated* B-Method. Unlike Lerp (stateless, pure function)
and SteeringVector (stateful, latent→latent), ActivationPatch works
through a ``FlatBatchDecodableAdapter``: it encodes flat sample batches,
patches the latent representation, and decodes back to data space. The
output is in **data space** (e.g. images), not latent space.

This is the Rule of Three freeze trigger for the ``BMethod`` Protocol.
"""

from __future__ import annotations

import numpy as np

from latent_anything.adapters.protocols import FlatBatchDecodableAdapter
from latent_anything.latent_space import LatentSpace
from latent_anything.trajectory import Trajectory


class ActivationPatch:
    """Model-mediated activation patching via encode → patch → decode.

    Unlike Lerp and SteeringVector which operate directly on latent
    points, ``ActivationPatch`` works through a
    ``FlatBatchDecodableAdapter``: it encodes flat sample batches, patches
    the latent representation, and decodes back to data space. The output
    is in data space (e.g., images), not latent space.

    This is B-Method #3 — the third distinct B-Method pattern,
    triggering the Rule of Three freeze for the ``BMethod`` Protocol.

    Parameters
    ----------
    adapter
        A ``FlatBatchDecodableAdapter`` instance with batch-matrix
        ``encode``/``decode`` semantics and ``latent_space``.
    """

    def __init__(self, adapter: object) -> None:
        if not isinstance(adapter, FlatBatchDecodableAdapter):
            msg = (
                "ActivationPatch requires a FlatBatchDecodableAdapter "
                f"(flat-batch encode + decode + latent_space), "
                f"got {type(adapter).__name__}"
            )
            raise TypeError(msg)
        self._adapter: FlatBatchDecodableAdapter = adapter
        self._delta: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def space(self) -> LatentSpace:
        """Return the adapter's ``LatentSpace``.

        Unlike ``Lerp`` and ``SteeringVector`` (which may return
        ``None``), ``ActivationPatch`` always has a ``LatentSpace``
        because the adapter is required and always provides one.
        """
        return self._adapter.latent_space

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` if ``fit()`` has been called successfully."""
        return self._delta is not None

    @property
    def delta(self) -> np.ndarray:
        """Return the learned patch direction in latent space.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(dim,)`` — the mean patch delta.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if self._delta is None:
            msg = "ActivationPatch not fitted. Call fit() first."
            raise RuntimeError(msg)
        return self._delta.copy()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, source_data: np.ndarray, target_data: np.ndarray) -> None:
        """Learn the patch delta from source and target data.

        Encodes both source and target data, then computes the mean
        latent delta: ``mean(target_latent) - mean(source_latent)``.

        Parameters
        ----------
        source_data : np.ndarray
            2-D array of shape ``(n_samples, n_features)`` — source
            (pre-patch) examples in data space.
        target_data : np.ndarray
            2-D array of shape ``(n_samples, n_features)`` — target
            (desired) examples in data space.

        Raises
        ------
        ValueError
            If either array is not 2-D, has fewer than 1 sample, or
            has mismatched feature dimensions.
        """
        if source_data.ndim != 2:
            msg = f"source_data must be a 2D array (n_samples, n_features), got {source_data.ndim}D"
            raise ValueError(msg)
        if target_data.ndim != 2:
            msg = f"target_data must be a 2D array (n_samples, n_features), got {target_data.ndim}D"
            raise ValueError(msg)
        if source_data.shape[0] == 0:
            msg = "source_data array is empty"
            raise ValueError(msg)
        if target_data.shape[0] == 0:
            msg = "target_data array is empty"
            raise ValueError(msg)
        if source_data.shape[1] != target_data.shape[1]:
            msg = (
                f"Feature dimension mismatch: source has dim {source_data.shape[1]}, "
                f"target has dim {target_data.shape[1]}"
            )
            raise ValueError(msg)

        source_latent: np.ndarray = self._adapter.encode(source_data)
        target_latent: np.ndarray = self._adapter.encode(target_data)

        self._delta = target_latent.mean(axis=0) - source_latent.mean(axis=0)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """Encode → patch latent → decode → return data-space output.

        Parameters
        ----------
        input_data : np.ndarray
            2-D array of shape ``(n_samples, n_features)`` — input
            data in data space.

        Returns
        -------
        np.ndarray
            Patched and decoded data, shape ``(n_samples, decoder_output_dim)``.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if self._delta is None:
            msg = "ActivationPatch not fitted. Call fit() first."
            raise RuntimeError(msg)

        latent: np.ndarray = self._adapter.encode(input_data)
        patched: np.ndarray = latent + self._delta  # broadcast delta across samples
        return self._adapter.decode(patched)

    def apply_trajectory(self, trajectory: Trajectory, **kwargs: float) -> np.ndarray:
        """Patch each latent point in trajectory, decode, return data-space outputs.

        Unlike ``Lerp.apply_trajectory`` and ``SteeringVector.apply_trajectory``
        which return ``Trajectory`` (latent → latent), this method returns
        ``np.ndarray`` (latent → data) because the output is in data space.

        Parameters
        ----------
        trajectory : Trajectory
            Input trajectory of latent points with shape ``(n_points, dim)``.
        **kwargs : float
            Reserved for future use (e.g., blend factor).

        Returns
        -------
        np.ndarray
            Decoded outputs of shape ``(n_points, decoder_output_dim)``.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        _ = kwargs
        if self._delta is None:
            msg = "ActivationPatch not fitted. Call fit() first."
            raise RuntimeError(msg)

        data = trajectory.to_numpy()  # (n_points, dim)
        patched = data + self._delta  # broadcast delta
        return self._adapter.decode(patched)
