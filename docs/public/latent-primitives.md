# Latent primitives and generic autoencoder adapters

## When to use this API

Use these types to give model outputs a shape and coordinate-system description, carry latent batches or sequences, or make an existing encoder/decoder available through the stable adapter protocols. They describe and route representations; they do not train a model, score reconstruction quality, or diagnose arbitrary autoencoders.

## Prerequisites

Use Python `>=3.12,<3.15` and the published `latent-anything==1.0.0` base package:

```bash
python -m pip install "latent-anything==1.0.0"
```

The example below needs no optional extra, downloaded checkpoint, or network access. It uses NumPy arrays on CPU. Save the code as `latent_adapter_example.py` and run `python latent_adapter_example.py` after installing the package.

## Shapes and coordinate meaning

`LatentSpace` describes one point's geometry and point shape. A flat Euclidean space of dimension `d` has point shape `(d,)`. Structured spaces retain their point axes: for example, a Gaussian set has point shape `(n_gaussians, param_dim)`, while `so3` and `se3` points have shapes `(3, 3)` and `(4, 4)`.

`LatentValue(data, space, metadata=...)` associates an array with a space. The trailing axes of `data` must equal `space.shape`; any preceding axes are leading batch axes:

| Representation | Example array shape | Meaning |
| --- | --- | --- |
| One flat point | `(d,)` | One Euclidean vector |
| Flat batch | `(n, d)` | `n` vectors |
| Flat batch and time | `(batch, time, d)` | Leading axes retained before the point shape |
| One structured point | `(n_gaussians, param_dim)` | One Gaussian-set point |
| Structured batch | `(batch, n_gaussians, param_dim)` | Leading batch axis plus one structured point |

The API records leading dimensions as `LatentValue.batch_shape`; it does not infer whether an axis means batch, time, layer, or something else. The caller defines those meanings. `to_numpy()` returns a writable copy.

`Trajectory` is a separate flat sequence container with exactly two axes, `(n_points, dim)`. It does not carry a `LatentSpace`; use a `LatentValue` when geometry and coordinate identity must travel with the values. A structured value or a value with more than one leading axis cannot be converted to a flat `Trajectory`.

### Coordinate identity and provenance

Matching dimension or array shape is not enough to make two values comparable. `LatentValue.identity` is assembled from declared representation/coordinate identity, `LatentSpace.source_model`, and a revision token such as `model_version`, `revision`, or `checkpoint` supplied in value or space metadata. Arithmetic such as `subtract()` requires the same geometry, point shape, stored shape, and the same non-empty identity. Missing identity or a different model, representation, or checkpoint raises `ValueError`.

These fields are caller-supplied provenance, not a checksum or proof that an array really came from the named weights. Record the model, layer/representation, and checkpoint revision consistently; do not relabel same-shaped arrays to make them appear compatible.

## Adapter capabilities

The adapter protocols are structural: a user-owned class conforms by providing the required members and need not inherit from the protocol.

| Protocol | Required capability | Appropriate use |
| --- | --- | --- |
| `ModelAdapter` | `latent_space` and `encode(data)` | Any model that exposes representations, including encoder-only or hidden-state adapters |
| `DecodableAdapter` | `ModelAdapter` plus `decode(latent)` | An adapter with a decoder, including geometry-specific/structured decoders |
| `FlatBatchDecodableAdapter` | `DecodableAdapter` plus `supports_flat_batch` | Flat matrix semantics: `(n, input_dim) -> (n, latent_dim) -> (n, output_dim)` |

Having `decode()` does not imply flat-batch support. A structured renderer can be decodable without satisfying `FlatBatchDecodableAdapter`; operations such as activation patching that assume flat batches need the narrower protocol. The flag describes the adapter's contract, not empirical validation of its model.

## Runnable CPU example: wrap an existing model object

This example wraps a tiny user-owned linear encoder/decoder object. Its fixed NumPy matrices stand in for already-available weights; nothing is trained. Replace the `ExistingLinearAE` instance with your already-constructed or loaded model and keep its public input/output shapes and provenance accurate.

```python
from __future__ import annotations

import numpy as np

from latent_anything import LatentSpace, LatentValue, Trajectory
from latent_anything.adapters import (
    DecodableAdapter,
    FlatBatchDecodableAdapter,
    ModelAdapter,
)


class ExistingLinearAE:
    """Small CPU model object with fixed encoder and decoder weights."""

    def __init__(
        self,
        encoder_weights: np.ndarray,
        decoder_weights: np.ndarray,
        revision: str,
    ) -> None:
        self.encoder_weights = encoder_weights
        self.decoder_weights = decoder_weights
        self.revision = revision

    def encode(self, data: np.ndarray) -> np.ndarray:
        return data @ self.encoder_weights.T

    def decode(self, latent: np.ndarray) -> np.ndarray:
        return latent @ self.decoder_weights.T


class ExistingAEAdapter:
    """Adapt an existing flat-batch encoder/decoder to Latent Anything."""

    def __init__(self, model: ExistingLinearAE) -> None:
        self.model = model

    @property
    def latent_space(self) -> LatentSpace:
        return LatentSpace(
            dim=self.model.encoder_weights.shape[0],
            source_model="tiny-linear-ae",
            metadata={"source_representation_identity": "example:tiny-linear-ae:encoder"},
        )

    @property
    def supports_flat_batch(self) -> bool:
        return True

    def encode(self, data: np.ndarray) -> np.ndarray:
        return self.model.encode(data)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        return self.model.decode(latent)


encoder_v1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
decoder = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
model_v1 = ExistingLinearAE(encoder_v1, decoder, revision="demo-v1")
adapter = ExistingAEAdapter(model_v1)

samples = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0], [2.0, 0.0, -2.0]])
encoded = adapter.encode(samples)
latent = LatentValue(
    encoded,
    adapter.latent_space,
    metadata={"model_version": model_v1.revision},
)
sequence = LatentValue(
    encoded[np.newaxis, :, :],
    adapter.latent_space,
    metadata={"model_version": model_v1.revision},
)
trajectory = Trajectory(encoded)
reconstructed = adapter.decode(latent.to_numpy())

print(f"encode: {encoded.shape}; first={encoded[0].tolist()}")
print(f"LatentValue: {latent.shape}; identity={latent.identity}")
print(f"leading axes: {sequence.shape}; batch_shape={sequence.batch_shape}")
print(f"Trajectory: {trajectory.shape}; decoded: {reconstructed.shape}")
print(
    "protocols:",
    isinstance(adapter, ModelAdapter),
    isinstance(adapter, DecodableAdapter),
    isinstance(adapter, FlatBatchDecodableAdapter),
)

# Same dimensions and representation, different weights revision: not arithmetic-compatible.
encoder_v2 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
model_v2 = ExistingLinearAE(encoder_v2, decoder, revision="demo-v2")
adapter_v2 = ExistingAEAdapter(model_v2)
latent_v2 = LatentValue(
    adapter_v2.encode(samples),
    adapter_v2.latent_space,
    metadata={"model_version": model_v2.revision},
)
try:
    latent.subtract(latent_v2)
except ValueError:
    print("same shape, different checkpoint: arithmetic rejected")
else:
    raise AssertionError("different checkpoint identities must not be combined")
```

Expected output:

```text
encode: (3, 2); first=[1.0, 2.0]
LatentValue: (3, 2); identity=example:tiny-linear-ae:encoder::tiny-linear-ae::demo-v1
leading axes: (1, 3, 2); batch_shape=(1, 3)
Trajectory: (3, 2); decoded: (3, 3)
protocols: True True True
same shape, different checkpoint: arithmetic rejected
```

This proves only that the adapter delegates the declared CPU shapes and that the latent wrapper exposes the expected metadata/identity behavior. The example's matrices are illustrative, untrained weights; it does not establish reconstruction quality or validate arbitrary autoencoders, datasets, or hardware. For independently bounded diagnostic cases, see the [AI engineer guide](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md).

## Contract and next steps

This page follows the `1.0.0` release. Check the [frozen API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md), [API freeze snapshot](https://github.com/triet4p/latent-anything/blob/v1.0.0/artifacts/api_freeze_snapshot_1.0.0.json), and the [adapter protocol definitions](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/adapters/protocols.py) for the released contract. Return to the [guide home](index.md) for other entry paths.
