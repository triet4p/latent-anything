"""RandomProjection — ModelAdapter #2 (fixed-weight / stateless / pretrained pattern).

A concrete adapter that uses a random Gaussian projection matrix
(Johnson-Lindenstrauss style) as a fixed encoder, with approximate
pseudo-inverse decoding via transpose. This is the second ModelAdapter
instance, demonstrating the stateless/pretrained pattern — fundamentally
different from VAE (stateful, trained-from-scratch).

Conforms to both the ``ModelAdapter`` Protocol (``encode``,
``latent_space``) and the ``DecodableAdapter`` Protocol (``encode``,
``decode``, ``latent_space``). Unlike VAE, there is no ``fit`` method
— weights are fixed at construction.

All input/output is ``numpy.ndarray``. No PyTorch dependency.
"""

from __future__ import annotations

import numpy as np

from latent_anything.latent_space import LatentSpace


class RandomProjection:
    """Fixed random projection encoder/decoder (Johnson-Lindenstrauss style).

    At construction, draws a random Gaussian matrix ``W`` of shape
    ``(latent_dim, input_dim)`` and normalises it by ``1/sqrt(latent_dim)``
    to approximately preserve Euclidean distances (Johnson-Lindenstrauss
    lemma). ``encode`` multiplies input by ``W.T``; ``decode`` multiplies
    latent by ``W`` (transpose ≈ pseudo-inverse for random Gaussian matrices).

    There is **no ``fit`` method** — weights are fixed at construction.
    This is the stateless / pretrained pattern, in contrast to VAE
    (stateful, trained-from-scratch).

    Parameters
    ----------
    input_dim : int
        Dimensionality of input data.
    latent_dim : int
        Dimensionality of the latent space (target dim after projection).
    random_state : int, optional
        Seed for the numpy random generator. Ensures reproducible
        projection matrices.

    Attributes
    ----------
    projection_matrix_ : np.ndarray
        The normalised random projection matrix of shape
        ``(latent_dim, input_dim)``. Set at construction.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        random_state: int | None = None,
    ) -> None:
        if input_dim < 1:
            msg = f"input_dim must be >= 1, got {input_dim}"
            raise ValueError(msg)
        if latent_dim < 1:
            msg = f"latent_dim must be >= 1, got {latent_dim}"
            raise ValueError(msg)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.random_state = random_state

        rng = np.random.default_rng(random_state)
        # W ~ N(0, 1) normalised by 1/sqrt(latent_dim) per JL lemma
        self.projection_matrix_: np.ndarray = rng.normal(size=(latent_dim, input_dim)).astype(np.float64) / (
            latent_dim**0.5
        )

    @property
    def latent_space(self) -> LatentSpace:
        """Return a ``LatentSpace`` describing this projection's latent manifold.

        Returns
        -------
        LatentSpace
            A Euclidean flat latent space of dimension ``latent_dim``.
        """
        return LatentSpace(dim=self.latent_dim, source_model="random_projection")

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data to the latent space via the fixed random projection.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape ``(n_samples, input_dim)``.

        Returns
        -------
        np.ndarray
            Latent vectors of shape ``(n_samples, latent_dim)``.

        Raises
        ------
        ValueError
            If ``data`` is not 2D or has wrong number of features.
        """
        if data.ndim != 2:
            msg = f"Expected 2D array, got {data.ndim}D"
            raise ValueError(msg)
        if data.shape[1] != self.input_dim:
            msg = f"Expected input_dim={self.input_dim}, got data with {data.shape[1]} features"
            raise ValueError(msg)
        return np.asarray(data @ self.projection_matrix_.T, dtype=np.float64)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent vectors back to data space (approximate).

        Uses the transpose of the projection matrix as an approximate
        pseudo-inverse reconstruction. For a random Gaussian matrix
        the transpose is a reasonable linear approximation of the
        inverse mapping.

        Parameters
        ----------
        latent : np.ndarray
            2D array of shape ``(n_samples, latent_dim)``.

        Returns
        -------
        np.ndarray
            Reconstructed data of shape ``(n_samples, input_dim)``.

        Raises
        ------
        ValueError
            If ``latent`` is not 2D or has wrong number of features.
        """
        if latent.ndim != 2:
            msg = f"Expected 2D array, got {latent.ndim}D"
            raise ValueError(msg)
        if latent.shape[1] != self.latent_dim:
            msg = f"Expected latent_dim={self.latent_dim}, got latent with {latent.shape[1]} features"
            raise ValueError(msg)
        return np.asarray(latent @ self.projection_matrix_, dtype=np.float64)
