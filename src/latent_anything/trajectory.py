"""An immutable sequence of latent states."""

from __future__ import annotations

import numpy as np


class Trajectory:
    """An immutable sequence of latent states.

    A Trajectory holds a sequence of latent state vectors as a 2D numpy
    array of shape ``(n_points, dim)``. It is immutable — every operation
    returns a new ``Trajectory`` instance.

    A single-point trajectory is valid (shape ``(1, dim)``).

    Parameters
    ----------
    data : np.ndarray
        2D array of shape ``(n_points, dim)`` containing latent states.
    """

    def __init__(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            msg = f"Expected 2D array (n_points, dim), got {data.ndim}D"
            raise ValueError(msg)
        if data.shape[0] < 1:
            msg = "Trajectory must have at least one point"
            raise ValueError(msg)
        # Store a copy to guarantee immutability.
        self._data = data.copy()

    def __len__(self) -> int:
        """Return the number of points in this trajectory."""
        return self._data.shape[0]

    @property
    def dim(self) -> int:
        """Return the dimensionality of each latent point."""
        return self._data.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the (n_points, dim) shape of the underlying data."""
        return self._data.shape

    def __getitem__(self, key: int | slice) -> Trajectory:
        """Index or slice this trajectory, returning a new Trajectory.

        Integer indexing returns a single-point Trajectory (preserving 2D shape).
        Slice indexing returns a Trajectory with the selected rows.
        """
        if isinstance(key, int):
            # Validate the index before slicing to get a proper IndexError.
            if key < -len(self) or key >= len(self):
                raise IndexError(f"Index {key} out of range for Trajectory with {len(self)} points")
            key = slice(key, key + 1)
        return Trajectory(self._data[key])

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the underlying data array.

        Returns
        -------
        np.ndarray
            A fresh copy of shape ``(n_points, dim)``.
        """
        return self._data.copy()

    def __repr__(self) -> str:
        return f"Trajectory(n_points={len(self)}, dim={self.dim}, shape={self.shape})"
