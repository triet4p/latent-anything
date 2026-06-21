"""GaussianRendererAdapter — ModelAdapter mode (iii): deterministic renderer.

A concrete adapter that treats a fixed-size set of 2D Gaussian
parameters as the latent representation and renders them into an
image via a deterministic numpy-only 2D Gaussian splat renderer.

This is the fourth ModelAdapter instance (and first in mode iii),
demonstrating the **explicit non-latent structured representation**
pattern: ``decode`` is a deterministic non-learned renderer, not a
neural network. This closes the last planned ``ModelAdapter`` ADR
mode.

All input/output is ``numpy.ndarray``. No PyTorch, CUDA, or ``gsplat``
dependency.

Philosophical difference from the three prior adapters:
- **VAE (#1)**: stateful, trained-from-scratch, learned encoder + decoder.
- **RandomProjection (#2)**: stateless, fixed random projection,
  linear encode + decode via linear algebra.
- **HiddenStateAdapter (#3)**: stateless, fixed random MLP, encode-only
  — no decoder, no explicit bottleneck.
- **GaussianRendererAdapter (#4)**: stateless, deterministic, decode-only
  in the sense that ``encode`` is a heuristic — the **intended
  workflow is latent-source-first**: create Gaussian parameters as
  a latent vector, then decode into a rendered image.
"""

from __future__ import annotations

import numpy as np

from latent_anything.latent_space import LatentSpace

# Default Gaussian parameter layout for 2D image rendering.
# position(2) + scale(2) + opacity(1) + color(3) = 8 columns.
_DEFAULT_POSITION_DIM = 2
_DEFAULT_SCALE_DIM = 2
_DEFAULT_COLOR_DIM = 3
_PARAM_DIM = _DEFAULT_POSITION_DIM + _DEFAULT_SCALE_DIM + 1 + _DEFAULT_COLOR_DIM  # 8


class GaussianRendererAdapter:
    """Deterministic 2D Gaussian splat renderer — ModelAdapter mode (iii).

    Treats a fixed-size set of 2D Gaussian parameters as a latent
    representation and renders them into an (H, W, 3) RGB image via
    a tiny numpy-only Gaussian splat renderer.

    This is **latent-source-first**: the intended use is to create
    or manipulate Gaussian parameters directly as latent vectors,
    then decode them into images. ``encode`` is provided as a simple
    heuristic grid-based approximation for testing and demonstration
    — **not** a true inverse of the renderer.

    Conforms to ``ModelAdapter`` and the shape-generic
    ``DecodableAdapter`` Protocol. It intentionally does **not** conform
    to ``FlatBatchDecodableAdapter`` because its public shapes are one
    image ``(H, W, 3)`` ↔ one Gaussian set ``(n_gaussians, 8)``, not
    flat sample batches.

    Parameters
    ----------
    n_gaussians : int
        Number of Gaussians in the set.
    img_height : int
        Output image height in pixels.
    img_width : int
        Output image width in pixels.
    random_state : int, optional
        Seed for the numpy random generator. Used only by the
        heuristic ``encode`` path for grid jitter — does not affect
        the deterministic ``decode``.
    """

    def __init__(
        self,
        n_gaussians: int,
        img_height: int,
        img_width: int,
        random_state: int | None = None,
    ) -> None:
        if n_gaussians < 1:
            msg = f"n_gaussians must be >= 1, got {n_gaussians}"
            raise ValueError(msg)
        if img_height < 1:
            msg = f"img_height must be >= 1, got {img_height}"
            raise ValueError(msg)
        if img_width < 1:
            msg = f"img_width must be >= 1, got {img_width}"
            raise ValueError(msg)

        self._n_gaussians = n_gaussians
        self._img_height = img_height
        self._img_width = img_width
        self._random_state = random_state

        # Pre-compute pixel coordinate grid for decode
        self._xs: np.ndarray = np.arange(img_width, dtype=np.float64)
        self._ys: np.ndarray = np.arange(img_height, dtype=np.float64)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_gaussians(self) -> int:
        """Number of Gaussians in the set."""
        return self._n_gaussians

    @property
    def img_height(self) -> int:
        """Output image height in pixels."""
        return self._img_height

    @property
    def img_width(self) -> int:
        """Output image width in pixels."""
        return self._img_width

    @property
    def latent_space(self) -> LatentSpace:
        """Return a ``LatentSpace`` with ``gaussian_set`` geometry.

        The parameter layout per Gaussian is:
        position(2) + scale(2) + opacity(1) + color(3) = 8 columns.

        Metadata includes the ``gaussian_set_param_layout`` key
        (auto-populated by ``LatentSpace``) and additional info
        about the output image dimensions.
        """
        metadata: dict[str, object] = {
            "exposure_mode": "deterministic_renderer",
            "img_height": self._img_height,
            "img_width": self._img_width,
        }
        return LatentSpace(
            dim=_PARAM_DIM,
            geometry="gaussian_set",
            source_model="gaussian_renderer",
            metadata=metadata,
            n_gaussians=self._n_gaussians,
            position_dim=_DEFAULT_POSITION_DIM,
            scale_dim=_DEFAULT_SCALE_DIM,
            color_dim=_DEFAULT_COLOR_DIM,
        )

    # ------------------------------------------------------------------
    # Decode — deterministic 2D Gaussian splat renderer
    # ------------------------------------------------------------------

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Render Gaussian parameters into an (H, W, 3) RGB image.

        This is a **deterministic** decode: the same latent vector
        always produces the same output. There is no randomness,
        no learned weights, and no neural network involved.

        For each Gaussian, the contribution to pixel ``(x, y)`` is::

            c_i * opacity_i * exp(-0.5 * ((x - px_i)² / sx_i²
                                        + (y - py_i)² / sy_i²))

        where ``c_i`` is the RGB colour. Contributions are accumulated
        additively and clipped to [0, 1]; there is no back-to-front sort
        and no per-pixel normalisation.

        Parameters
        ----------
        latent : np.ndarray
            2D array of shape ``(n_gaussians, 8)`` with columns:
            position (x, y), scale (sx, sy), opacity (o),
            colour (r, g, b). ``n_gaussians`` must match the adapter's
            ``n_gaussians``.

        Returns
        -------
        np.ndarray
            Rendered RGB image of shape ``(img_height, img_width, 3)``,
            dtype ``float64``, values in [0, 1].

        Raises
        ------
        ValueError
            If ``latent`` is not 2D, has wrong shape, or violates
            Gaussian-set numeric constraints.
        """
        if latent.ndim != 2:
            msg = f"Expected 2D array, got {latent.ndim}D"
            raise ValueError(msg)
        if latent.shape[0] != self._n_gaussians:
            msg = f"Expected {self._n_gaussians} Gaussians, got latent with {latent.shape[0]} rows"
            raise ValueError(msg)
        if latent.shape[1] != _PARAM_DIM:
            msg = f"Expected {_PARAM_DIM} parameter columns, got latent with {latent.shape[1]} columns"
            raise ValueError(msg)

        # Validate numeric constraints (same as LatentSpace gaussian_set)
        self._validate_decode_input(latent)

        return self._render(latent)

    def _validate_decode_input(self, latent: np.ndarray) -> None:
        """Validate numeric constraints on decode input.

        Scale must be > 0, opacity in [0, 1], colour in [0, 1].
        """
        pdim = _DEFAULT_POSITION_DIM
        sdim = _DEFAULT_SCALE_DIM

        # Scale > 0
        scale_slice = slice(pdim, pdim + sdim)
        if np.any(latent[:, scale_slice] <= 0):
            msg = "Scale components must be > 0"
            raise ValueError(msg)

        # Opacity in [0, 1]
        opacity_idx = pdim + sdim
        if np.any((latent[:, opacity_idx] < 0) | (latent[:, opacity_idx] > 1)):
            msg = "Opacity must be in [0, 1]"
            raise ValueError(msg)

        # Colour in [0, 1]
        color_start = opacity_idx + 1
        color_end = color_start + _DEFAULT_COLOR_DIM
        if np.any((latent[:, color_start:color_end] < 0) | (latent[:, color_start:color_end] > 1)):
            msg = "Colour channels must be in [0, 1]"
            raise ValueError(msg)

    def _render(self, latent: np.ndarray) -> np.ndarray:
        """Core 2D Gaussian splat rasterisation.

        Accumulates Gaussian contributions additively: each Gaussian's
        colour is weighted by its opacity and 2D Gaussian falloff, then
        summed with the other contributions.
        """
        pdim = _DEFAULT_POSITION_DIM
        sdim = _DEFAULT_SCALE_DIM
        cdim = _DEFAULT_COLOR_DIM
        opacity_idx = pdim + sdim
        color_start = opacity_idx + 1
        color_end = color_start + cdim

        # Extract parameter columns
        positions = latent[:, :pdim]  # (n_g, 2)
        scales = latent[:, pdim:opacity_idx]  # (n_g, 2)
        opacities = latent[:, opacity_idx]  # (n_g,)
        colors = latent[:, color_start:color_end]  # (n_g, 3)

        # Build (H, W) pixel coordinate grids
        xs = self._xs[None, None, :]  # (1, 1, W)
        ys = self._ys[None, :, None]  # (1, H, 1)

        # Broadcast Gaussian params to (n_g, 1, 1) for broadcasting
        px = positions[:, 0, None, None]  # (n_g, 1, 1)
        py = positions[:, 1, None, None]  # (n_g, 1, 1)
        sx = scales[:, 0, None, None]  # (n_g, 1, 1)
        sy = scales[:, 1, None, None]  # (n_g, 1, 1)
        op = opacities[:, None, None]  # (n_g, 1, 1)

        # 2D Gaussian falloff per Gaussian per pixel: (n_g, H, W)
        dx2 = (xs - px) ** 2 / (2.0 * sx**2 + 1e-30)
        dy2 = (ys - py) ** 2 / (2.0 * sy**2 + 1e-30)
        weights: np.ndarray = op * np.exp(-(dx2 + dy2))  # (n_g, H, W)

        # Weighted colour accumulation: (n_g, H, W, 3)
        colour_contrib = weights[..., None] * colors[:, None, None, :]  # (n_g, H, W, 3)

        # Sum over Gaussians (no normalisation — the opacity term already
        # controls contribution weight, and additive composition avoids
        # the single-Gaussian normalisation artifact where dividing by
        # total weight cancels the spatial falloff).
        total_colour = colour_contrib.sum(axis=0)  # (H, W, 3)

        return np.clip(total_colour, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Encode — heuristic grid-based approximation
    # ------------------------------------------------------------------

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Heuristic encode from image to Gaussian parameters (approximate).

        .. note::

            This **is not** a true inverse of ``decode``. The adapter
            is **latent-source-first**: the intended workflow is to
            create Gaussian parameters directly as latent vectors,
            then decode them into images. This ``encode`` is a simple
            heuristic for testing, demonstration, and roundtrip
            verification — it places Gaussians on a regular grid over
            the image and estimates colour from local pixel averages.

        The encoding strategy:
        1. Place ``n_gaussians`` Gaussians on a regular grid over
           the image (rows × cols optimised to approximate the count).
        2. Set scale proportional to grid spacing.
        3. Set opacity to 1.0 (fully opaque).
        4. Sample colour as the average colour in each grid cell.
        5. Apply small random jitter to grid positions for visual
           variety (controlled by ``random_state``).

        Parameters
        ----------
        data : np.ndarray
            3D array of shape ``(img_height, img_width, 3)`` with
            values in [0, 1].

        Returns
        -------
        np.ndarray
            Gaussian parameter array of shape ``(n_gaussians, 8)``
            with columns: position (x, y), scale (sx, sy),
            opacity (o), colour (r, g, b).

        Raises
        ------
        ValueError
            If ``data`` is not 3D, has wrong image dimensions, or
            contains out-of-range values.
        """
        if data.ndim != 3:
            msg = f"Expected 3D array (H, W, C), got {data.ndim}D"
            raise ValueError(msg)
        if data.shape != (self._img_height, self._img_width, _DEFAULT_COLOR_DIM):
            msg = f"Expected image shape ({self._img_height}, {self._img_width}, 3), got {data.shape}"
            raise ValueError(msg)
        if np.any(data < 0) or np.any(data > 1):
            msg = "Image data must be in [0, 1]"
            raise ValueError(msg)

        return self._encode_grid(data)

    def _encode_grid(self, data: np.ndarray) -> np.ndarray:
        """Grid-based heuristic encoding.

        Distributes ``n_gaussians`` Gaussians across a regular grid,
        sampling colour from pixel blocks.
        """
        rng = np.random.default_rng(self._random_state)

        # Compute grid dimensions that best match n_gaussians
        sqrt_val: float = float(np.sqrt(self._n_gaussians * self._img_width / self._img_height))
        n_cols = max(1, int(np.round(sqrt_val)))
        n_rows = max(1, self._n_gaussians // n_cols)
        # Adjust if product doesn't match
        while n_rows * n_cols < self._n_gaussians:
            n_cols += 1
        while n_rows * n_cols > self._n_gaussians and n_rows > 1 and (n_rows - 1) * n_cols >= self._n_gaussians:
            n_rows -= 1

        # Cell dimensions
        cell_h = self._img_height / n_rows
        cell_w = self._img_width / n_cols

        # Grid centre positions
        row_centres = (np.arange(n_rows, dtype=np.float64) + 0.5) * cell_h
        col_centres = (np.arange(n_cols, dtype=np.float64) + 0.5) * cell_w
        grid_y, grid_x = np.meshgrid(row_centres, col_centres, indexing="ij")

        # Flatten to (n_gaussians,) arrays
        gx = grid_x.ravel()[: self._n_gaussians]
        gy = grid_y.ravel()[: self._n_gaussians]

        # Small random jitter (up to 20% of cell size) for visual variety
        jitter_x = rng.uniform(-0.2 * cell_w, 0.2 * cell_w, size=self._n_gaussians)
        jitter_y = rng.uniform(-0.2 * cell_h, 0.2 * cell_h, size=self._n_gaussians)
        gx = np.clip(gx + jitter_x, 0, self._img_width - 1)
        gy = np.clip(gy + jitter_y, 0, self._img_height - 1)

        # Scale proportional to grid spacing
        sx = np.full(self._n_gaussians, cell_w * 0.4, dtype=np.float64)
        sy = np.full(self._n_gaussians, cell_h * 0.4, dtype=np.float64)

        # Opacity = 1.0 (fully opaque)
        opacities = np.ones(self._n_gaussians, dtype=np.float64)

        # Sample colour from the centre pixel of each grid cell
        gy_int = np.round(gy).astype(np.intp)
        gx_int = np.round(gx).astype(np.intp)
        gy_int = np.clip(gy_int, 0, self._img_height - 1)
        gx_int = np.clip(gx_int, 0, self._img_width - 1)
        colors_arr: np.ndarray = data[gy_int, gx_int]  # (n_g, 3)

        # Assemble parameter array: (n_g, 8)
        # Explicitly stack 1D arrays to avoid pyright unknown-type issues
        # from column_stack with dynamically-shaped arrays.
        latent = np.empty((self._n_gaussians, _PARAM_DIM), dtype=np.float64)
        latent[:, 0] = gx  # position x
        latent[:, 1] = gy  # position y
        latent[:, 2] = sx  # scale x
        latent[:, 3] = sy  # scale y
        latent[:, 4] = opacities  # opacity
        latent[:, 5] = colors_arr[:, 0]  # colour r
        latent[:, 6] = colors_arr[:, 1]  # colour g
        latent[:, 7] = colors_arr[:, 2]  # colour b
        return latent

    # ------------------------------------------------------------------
    # No-fit note
    # ------------------------------------------------------------------

    # ``fit`` is deliberately absent. This adapter has no training
    # step — the renderer is deterministic and the heuristic encode
    # is fixed at construction.
