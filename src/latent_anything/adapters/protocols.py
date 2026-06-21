"""Frozen adapter Protocols.

Frozen at ModelAdapter #3 (``HiddenStateAdapter``, Sprint 14). Validated
by 3 instances with differing philosophies:

- **VAE (#1)** — explicit learned latent (mode i): ``encode`` + ``decode``,
  both learned from data via gradient descent.
- **RandomProjection (#2)** — fixed explicit projection (mode i-like):
  ``encode`` + ``decode``, both deterministic linear transforms with
  fixed random weights. No training needed.
- **HiddenStateAdapter (#3)** — no-explicit-latent (mode ii):
  ``encode`` only, returns hidden-state activations. No ``decode``
  because there is no decoder — the latent *is* the hidden activation.

The split into two Protocols reflects the core evidence this sprint
produces:

- ``ModelAdapter`` — the universal surface: ``encode`` + ``latent_space``.
  Every adapter must provide this.
- ``DecodableAdapter`` — the shape-generic decodable surface: extends
  ``ModelAdapter`` with ``decode``. Only adapters with a decoder (learned
  or deterministic) implement this.
- ``FlatBatchDecodableAdapter`` — the narrower batch-matrix surface for
  adapters whose public ``encode``/``decode`` contract is
  ``(n_samples, input_dim)`` → ``(n_samples, latent_dim)`` →
  ``(n_samples, output_dim)``.

Methods like ``ActivationPatch`` that work through ``encode → patch → decode``
and assume flat batch matrices require ``FlatBatchDecodableAdapter``, not just
``DecodableAdapter``. Methods that only need latent-space access work with
``ModelAdapter`` alone.

.. note::

    These are structural (duck-typed) Protocols. Classes conform by
    providing the required properties and methods with matching
    signatures — they do **not** need to inherit from these Protocols
    or import them.

Sprint 16 added ``GaussianRendererAdapter`` (mode iii), which conforms to
``DecodableAdapter`` but intentionally does not conform to
``FlatBatchDecodableAdapter`` because it maps one image to one Gaussian set
and one Gaussian set to one image.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from latent_anything.latent_space import LatentSpace


@runtime_checkable
class ModelAdapter(Protocol):
    """Base protocol for all model adapters.

    Every adapter must provide ``encode`` to transform input data into
    a latent representation, and ``latent_space`` to describe the geometry
    of that representation.

    ``decode`` is deliberately **not** part of this Protocol — not all
    adapters have a decoder (see ``DecodableAdapter``).
    """

    @property
    def latent_space(self) -> LatentSpace:
        """Return a ``LatentSpace`` describing this adapter's latent manifold."""
        ...

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode input data to this adapter's latent representation.

        Parameters
        ----------
        data : np.ndarray
            Adapter-specific input array. Flat-batch adapters use
            ``(n_samples, input_dim)``; structured adapters may use
            image, sequence, or other documented shapes.

        Returns
        -------
        np.ndarray
            Adapter-specific latent representation whose shape is described
            by ``latent_space`` and the concrete adapter documentation.
        """
        ...


@runtime_checkable
class DecodableAdapter(ModelAdapter, Protocol):
    """Protocol for adapters that support decoding back to data space.

    Extends ``ModelAdapter`` with a ``decode`` method. Adapters that
    have a decoder (learned, as in VAE, or deterministic, as in
    ``GaussianRendererAdapter``) conform to this Protocol.
    """

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode a latent representation back to data space.

        Parameters
        ----------
        latent : np.ndarray
            Adapter-specific latent array. Flat-batch adapters use
            ``(n_samples, latent_dim)``; structured adapters may use
            a geometry-specific latent shape.

        Returns
        -------
        np.ndarray
            Adapter-specific decoded data shape documented by the concrete
            adapter.
        """
        ...


@runtime_checkable
class FlatBatchDecodableAdapter(DecodableAdapter, Protocol):
    """Protocol for decodable adapters with flat batch-matrix semantics.

    ``ActivationPatch`` and similar data-space patching methods assume
    encode/decode operate on batches of flat vectors. ``DecodableAdapter``
    alone is broader: it also covers structured decoders such as
    ``GaussianRendererAdapter`` whose public shapes are image/gaussian-set
    rather than flat sample matrices.
    """

    @property
    def supports_flat_batch(self) -> bool:
        """Return ``True`` for adapters with flat batch encode/decode semantics."""
        ...
