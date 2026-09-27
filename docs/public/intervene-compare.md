# Intervene on latent representations and compare outcomes

Use this guide to choose an operation for changing a representation, distinguish a returned latent edit from an intervention inside a model's forward execution, and design comparisons that keep the claim within the evidence. These methods apply a declared transformation; they do not decide whether a representation feature is causal or validate an arbitrary model.

## Choose the operation

| Goal | Start with | Do not choose it when… |
| --- | --- | --- |
| Move between two compatible latent endpoints | `Lerp` | You need to learn a concept direction, decode a model output, or attribute a downstream effect. |
| Add a learned contrast direction to a latent vector | `SteeringVector` | You lack contrast examples from the same coordinates, or you need a model-forward patch rather than a changed vector. |
| Add a source-to-target mean latent delta and return decoded data | `ActivationPatch` | Your adapter is only generally decodable, has structured/non-flat batch semantics, or you need to inject an activation into a model's later layers. |
| Keep or remove a fitted Euclidean component | `SubspaceProjection` | The value is not flat Euclidean, its coordinate identity differs from the basis, or you want a decoded/model-forward effect. |
| Reuse one intervention across data batches or trajectories | `InterventionPipeline` | You expect it to add a new method, causal guarantee, or model-hook mechanism. |

For the shape and provenance basics behind these operations, see [latent primitives and adapters](latent-primitives.md). If the question is about whether a representation predicts a property before intervening, start with [inspect and explain representations](inspect-explain.md) and keep that predictive result separate from an intervention result.

## Prerequisites and coordinate identity

Use Python `>=3.12,<3.15` and the released base package:

```bash
python -m pip install "latent-anything==1.0.0"
```

The CPU example below uses NumPy and the base package only. It needs no optional extra, downloaded checkpoint, or network access.

A feature axis has meaning only relative to the producing model, layer/representation, checkpoint, and preprocessing or pooling rule. `LatentValue` carries a declared coordinate identity and `SubspaceProjection` binds its fitted basis to that identity. `Lerp` and `SteeringVector` operate on NumPy arrays and do not compare or carry model/layer/revision identity for you: same-shaped arrays from different coordinates are not thereby compatible. Keep those identities aligned when preparing endpoints, positive/negative groups, source/target data, and evaluation examples. A declaration is caller-supplied provenance, not proof that the values came from the named weights.

## Editing a stored latent is not a model-forward patch

`Lerp` and `SteeringVector` return new latent arrays; neither mutates a model, decodes the result, nor runs the model again. `SubspaceProjection.project()` and `.remove()` likewise return new `LatentValue` objects with operation provenance. These are suitable when the next step consumes the edited latent, or when you will explicitly decode it through your own compatible decoder.

`ActivationPatch` is adapter-mediated: it encodes flat data batches, adds the fitted mean target-minus-source latent delta, and decodes the changed latent batch to data space. Its released implementation does **not** install a hook, replace an intermediate activation during an arbitrary model forward pass, or resume the rest of that model's layers. If you need a true forward-execution intervention, your model integration must expose a supported way to inject the changed state and evaluate the downstream computation. Do not describe an encode/patch/decode round trip as a hook-based intervention unless your adapter actually implements that behavior.

### `Lerp`: interpolate between two points

`Lerp(space=None)` is stateless. Call it with two 1-D arrays of the same shape and `t` in `[0, 1]`; it returns `(1 - t) * a + t * b` by default, or delegates to `LatentSpace.interpolate()` when given a space. Supply a space when the representation uses a supported non-Euclidean geometry. `t=0` and `t=1` are endpoint identity controls. Interpolation says where a point lies between supplied endpoints; it does not infer an attribute or show that the model uses the path.

**Do not use `Lerp`** as a substitute for contrast fitting, subspace removal, or a model output intervention. It only needs matching shapes at the array level; the caller must ensure that the endpoints share the intended coordinates.

### `SteeringVector`: fit a contrast direction

`SteeringVector(space=None).fit(positives, negatives)` expects non-empty matrices `(n_positive, dim)` and `(n_negative, dim)` in the same coordinates. It normalizes `mean(positives) - mean(negatives)` to a unit direction. Calling the fitted method on one vector `(dim,)` adds `strength * direction`; strength `0` is a null/identity control and negative strength reverses the direction. An optional `LatentSpace` can apply its geometry-specific normalization after the addition.

This is a mean-difference direction, not a learned causal mechanism. Check that the contrast groups are aligned and not distinguished by an unintended confound. **Do not use it** when there are no meaningful contrast examples or when you require a decoded or forward-executed model result; it returns only a latent vector (or a changed `Trajectory`).

### `SubspaceProjection`: preserve, remove, or transfer a component

`SubspaceProjection` applies an orthonormal basis to `LatentValue` points in a flat Euclidean space. Fit a basis explicitly with `fit_basis()` or derive one with `fit_pca()`, and supply the source representation identity. For basis matrix `U`, columns are orthonormal and the operation is:

- `project(value)`: keep the component `U Uᵀ z` inside the fitted subspace.
- `remove(value)`: keep the residual `(I - U Uᵀ) z` outside it.
- `transfer(source, target)`: retain the target's outside component and replace its subspace component with the source's.

The point shape must be `(dim,)`, geometry must be Euclidean, and the `LatentValue` identity must equal the basis identity. Calls return new values and append operation provenance; they do not edit the source value or run a model. The origin label (`explicit`, `pca`, `probe`, or `concept`) records how the basis was derived; those origins answer different questions and are not interchangeable. **Do not use it** with a mismatched layer/checkpoint or to assert that a projected component is causal just because it separates labels.

### `ActivationPatch`: adapt flat data batches through encode and decode

`ActivationPatch(adapter)` requires the structural `FlatBatchDecodableAdapter` contract, which is narrower than `DecodableAdapter`: `latent_space`, `supports_flat_batch == True`, and flat matrix `encode`/`decode` semantics. The expected shape flow is `(n_samples, input_dim) -> (n_samples, latent_dim) -> (n_samples, output_dim)`. A `decode()` method alone is insufficient; for example, a structured renderer need not support a flat batch.

`fit(source_data, target_data)` takes two non-empty 2-D data batches with equal input feature width. It encodes both and stores `mean(target_latent) - mean(source_latent)`. Calling the fitted patch encodes the input batch, broadcasts that delta to each row, decodes, and returns data-space output. `apply_latent()` exposes the latent-batch edit without decoding. This fitting recipe defines an average displacement; it does not pair or align source and target rows, learn a per-example replacement, or validate that the decoded outputs remain on a model's data manifold.

**Do not use it** with an encoder-only adapter, a merely decodable structured adapter, or when the desired operation is an internal model hook. Confirm that input, latent, and output batch axes mean the same sample rows and that your adapter preserves the declared flat-matrix contract.

### `InterventionPipeline`: compose, do not reinterpret

`InterventionPipeline(method, adapter=None)` is the released canonical name for the manipulation pipeline. `fit()` delegates to a stateful method. `run_trajectory()` delegates to that method's trajectory operation. `run_data()` needs a `FlatBatchDecodableAdapter`; for methods exposing `apply_latent()`, it encodes, applies the latent edit, and decodes, while other callable methods receive the data batch directly. It does not change the wrapped method's mathematical effect or add model hooks, controls, identity checks, or a causal interpretation. Use it to make that existing path explicit and repeatable, not as a substitute for choosing the right intervention.

## Controls and claim boundaries

For every comparison, define the measured outcome before looking at the intervention result and apply baseline and intervention to the same held-out examples under the same model/checkpoint, preprocessing, target, and evaluation code. At minimum consider:

- **Identity/null:** preserve the input (`t=0`, steering `strength=0`, or a zero patch learned from identical source/target data). The result should match the unmodified path within a declared numerical tolerance.
- **Off-target:** predeclare an unrelated feature, token, layer, or task metric and measure whether the intervention changes it. Report off-target movement instead of hiding it in a single aggregate score.
- **Randomized/null direction:** compare the claimed direction with a shuffled-label, random, or otherwise appropriate null direction. A random direction is a separate control from a zero-strength identity run.
- **Positive/restore control where appropriate:** verify that a known valid direction or restoration changes the intended measure, without using that observation to tune the original threshold after the fact.

A probe, label association, visible edit, or output difference is observational/descriptive until a declared intervention is evaluated against these controls. A well-matched model-forward intervention can support a causal-use claim for that model, inputs, target, and tested pathway; a toy fixture or a latent-only transformation cannot establish general model behavior. Causal claims still need aligned runs, controls, uncertainty/stability checks, and limits stated at the scope actually tested.

For persisted experiments, the released API includes `RunComparisonReport` / `build_comparison_report()` and the `compare-runs` CLI; compare aligned run identities and metric deltas rather than eyeballing separate outputs. See the [frozen API reference and CLI contract](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md#cli-contract) and the [stable run-record facade at the release tag](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/run_record.py). These summarize recorded runs; a metric delta alone does not establish causal attribution.

## Runnable CPU example: compare identity, changed, and off-target results

This fixed 2-D adapter rotates flat inputs into a declared latent basis and inverts that transform on decode. The desired edit is defined in the first input feature; the second is a predeclared off-target feature. The identity patch and zero-strength steering are null controls. The example exercises latent interpolation, steering, subspace removal, adapter-mediated patching, and the intervention pipeline without claiming anything about a trained or nonlinear model.

Run this block after installing `latent-anything==1.0.0`:

```python
from __future__ import annotations

import numpy as np

from latent_anything import (
    InterventionPipeline,
    LatentSpace,
    LatentValue,
    SubspaceProjection,
)
from latent_anything.methods import ActivationPatch, Lerp, SteeringVector
from latent_anything.adapters import FlatBatchDecodableAdapter


class FixedFlatAdapter:
    """A deterministic flat-batch encode/decode fixture on CPU."""

    def __init__(self) -> None:
        self.rotation = np.array([[0.8, 0.6], [-0.6, 0.8]], dtype=np.float64)
        self._space = LatentSpace(
            dim=2,
            source_model="two-feature-fixture",
            metadata={
                "source_representation_identity": "two-feature-fixture:rotated-encoder",
                "model_version": "fixed-v1",
            },
        )

    @property
    def latent_space(self) -> LatentSpace:
        return self._space

    @property
    def supports_flat_batch(self) -> bool:
        return True

    def encode(self, data: np.ndarray) -> np.ndarray:
        return data @ self.rotation.T

    def decode(self, latent: np.ndarray) -> np.ndarray:
        return latent @ self.rotation


adapter = FixedFlatAdapter()
assert isinstance(adapter, FlatBatchDecodableAdapter)

source = np.array([[0.0, 10.0], [0.0, 11.0]])
target = np.array([[1.0, 10.0], [1.0, 11.0]])
evaluation = np.array([[0.0, 12.0], [2.0, 12.0]])

# Stateless midpoint: decode only because this fixture explicitly provides a decoder.
lerp = Lerp(adapter.latent_space)
source_latent = adapter.encode(source[:1])[0]
target_latent = adapter.encode(target[:1])[0]
midpoint = adapter.decode(lerp(source_latent, target_latent, 0.5)[None, :])[0]
assert np.allclose(midpoint, [0.5, 10.0])

# The learned direction shifts the declared first feature; strength=0 is a null control.
steering = SteeringVector(adapter.latent_space)
steering.fit(adapter.encode(target), adapter.encode(source))
start_latent = adapter.encode(evaluation[:1])[0]
null_steer = steering(start_latent, strength=0.0)
steered = adapter.decode(steering(start_latent, strength=0.5)[None, :])[0]
assert np.allclose(null_steer, start_latent)
assert np.allclose(steered, [0.5, 12.0])

# Remove the explicitly declared first-feature subspace from immutable latent values.
latent_values = LatentValue(adapter.encode(evaluation), adapter.latent_space)
signal_basis = adapter.encode(np.array([[1.0, 0.0]]))[0][:, None]
subspace = SubspaceProjection().fit_basis(
    signal_basis,
    source_representation_identity=latent_values.identity,
    origin="explicit",
    provenance={"basis": "fixture first input feature"},
)
removed = subspace.remove(latent_values)
removed_data = adapter.decode(removed.to_numpy())
assert np.allclose(removed_data[:, 0], 0.0)
assert np.allclose(removed_data[:, 1], evaluation[:, 1])

# ActivationPatch learns a mean target-minus-source delta and returns decoded data.
identity_patch = ActivationPatch(adapter)
identity_patch.fit(source, source)
identity_output = InterventionPipeline(identity_patch, adapter=adapter).run_data(evaluation)

patch = ActivationPatch(adapter)
patch.fit(source, target)
patched_output = InterventionPipeline(patch, adapter=adapter).run_data(evaluation)
assert np.allclose(identity_output, evaluation)
assert np.allclose(patched_output, evaluation + np.array([1.0, 0.0]))
off_target_delta = float(np.max(np.abs(patched_output[:, 1] - evaluation[:, 1])))
assert np.isclose(off_target_delta, 0.0, atol=1e-12)

print(f"Lerp midpoint decoded: {np.round(midpoint, 3).tolist()}")
print(f"Steering strength=0.5 decoded: {np.round(steered, 3).tolist()}; strength=0 unchanged: {np.allclose(null_steer, start_latent)}")
print(f"Subspace remove decoded rows: {np.round(removed_data, 3).tolist()}")
print(f"ActivationPatch latent delta: {np.round(patch.delta, 3).tolist()}")
print(f"ActivationPatch identity max |delta|: {np.max(np.abs(identity_output - evaluation)):.3f}")
print(f"ActivationPatch changed rows: {np.round(patched_output, 3).tolist()}")
print(f"Off-target feature max |delta|: {off_target_delta:.3f}")
```

Expected output:

```text
Lerp midpoint decoded: [0.5, 10.0]
Steering strength=0.5 decoded: [0.5, 12.0]; strength=0 unchanged: True
Subspace remove decoded rows: [[0.0, 12.0], [0.0, 12.0]]
ActivationPatch latent delta: [0.8, -0.6]
ActivationPatch identity max |delta|: 0.000
ActivationPatch changed rows: [[1.0, 12.0], [3.0, 12.0]]
Off-target feature max |delta|: 0.000
```

The expected controls are an unchanged identity output, an intended first-feature change of `+1`, and zero movement in the declared second-feature off-target. The observed values establish only that the frozen API delegates these declared shapes and operations correctly on this deterministic linear fixture. They do not validate nonlinear encoders/decoders, establish data-manifold validity, or show that a trained model uses the edited component.

## Bounded comparisons and released contracts

The [AI engineer guide](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md) describes two bounded comparisons: its encoder-lesion case reports the predeclared healthy, benign low-variance, null-shuffle, explanation, and intervention controls plus an aligned healthy-versus-lesion comparison for its frozen linear-autoencoder/digits case; its transformer target-evidence-v2 case records a separate removal intervention, aligned comparison, and observational confounds. These are bounded examples, not validation of the toy fixture or arbitrary models.

For the exact released surface, consult the [frozen `1.0.0` API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md), [API-freeze snapshot](https://github.com/triet4p/latent-anything/blob/v1.0.0/artifacts/api_freeze_snapshot_1.0.0.json), [intervention methods](https://github.com/triet4p/latent-anything/tree/v1.0.0/src/latent_anything/methods), [subspace projection](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/projection.py), [adapter protocols](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/adapters/protocols.py), and [intervention pipeline](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/manipulation_pipeline.py). Return to the [guide home](index.md) to choose another released entry path.
