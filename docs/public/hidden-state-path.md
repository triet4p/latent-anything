# Adapt decoder-free models and inspect hidden states

## When to use this API

Use a decoder-free model's intermediate activation as its representation when the model has no meaningful mapping back to the input space, or when the chosen hidden state is the object you want to inspect. The released `ModelAdapter` protocol needs only `latent_space` and `encode()`; it does not require a decoder. A user-owned adapter can therefore select a layer from an existing deep model and return its activation as a NumPy array.

This guide is about carrying hidden representations into the released `1.0.0` API. It does not train your model, choose a scientifically meaningful layer, or establish that features support a particular task.

## Choose the right route

| Situation | Route | What it means |
| --- | --- | --- |
| You already have a decoder-free model and want its real layer output | Implement the structural [`ModelAdapter`](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/adapters/protocols.py) contract around that model | Your adapter controls inference, layer selection, output conversion, and provenance. It may return sequence-shaped values such as `(batch, tokens, hidden_dim)`. |
| You want the released, lightweight hidden-state-shaped example | `HiddenStateAdapter` from `latent_anything.adapters` | This is a fixed-random NumPy MLP with input `(n_samples, input_dim)` and output `(n_samples, hidden_dim)`. Its two-layer weights are initialized but `encode()` returns only the first ReLU activation; it does not wrap your PyTorch model or a pretrained checkpoint. It has no `decode()`. |
| You specifically need a decoder-only language-model lifecycle | `TransformerLMIntegration` in versions that provide it, with the optional Transformers dependency | This is a model-specific integration for tokenization, forward passes, and transformer analysis, not the generic `ModelAdapter` contract. It is not a frozen `1.0.0` public import; installing an extra does not make an unfrozen name part of the release API. |

For a model you already own, the first route is the direct `1.0.0` extension point. The built-in `HiddenStateAdapter` is useful for exercising downstream methods with a reproducible fixed-random representation, but its `encode()` creates that representation itself. It should not be substituted for capture from the trained model whose hidden states you intend to analyze.

## Prerequisites

Use Python `>=3.12,<3.15` and the published base package:

```bash
python -m pip install "latent-anything==1.0.0"
```

The CPU example below uses PyTorch, which is in the base package, plus NumPy. It uses a tiny fixed-weight model defined in the example and makes no checkpoint or network request. `HiddenStateAdapter` itself uses NumPy and does not require the optional Transformers extra.

The package also declares a `transformers` optional extra. Install it only when the specific optional integration you intend to use is available in your installed version:

```bash
python -m pip install "latent-anything[transformers]==1.0.0"
```

The extra supplies optional dependencies; it does not change the frozen `1.0.0` API surface. The repository's current [`TransformerLMIntegration` implementation](https://github.com/triet4p/latent-anything/blob/main/src/latent_anything/integrations/transformer_lm.py) is a source reference, not a stable `1.0.0` import contract. For the exact revision-backed GPT-2 evidence path, use the pinned setup and executable example linked below rather than guessing an import path.

## Shapes, axes, and coordinate identity

For tokenized sequence input, the common shapes are:

| Value | Shape | Meaning |
| --- | --- | --- |
| Token IDs | `(batch, sequence)` | Input IDs supplied by the model's tokenizer, with a matching attention mask where applicable |
| One selected layer | `(batch, sequence, hidden_dim)` | One hidden vector per token; `hidden_dim` is the final feature axis |
| One selected, pooled layer | `(batch, hidden_dim)` | One vector per input after the explicitly chosen pooling rule |
| Several captured layers | `(layers, batch, sequence, hidden_dim)` | A caller-stacked array; the layer axis is still just a leading axis to `LatentValue` |

Declare `LatentSpace(dim=hidden_dim)` for one flat hidden vector. A `LatentValue` over `(batch, sequence, hidden_dim)` then has point shape `(hidden_dim,)` and `batch_shape == (batch, sequence)`. The wrapper preserves leading axes but does not infer that they mean batch, token, or layer. Record those meanings yourself. If layers have different semantics, keep them as separate values or identify the layer axis explicitly in your metadata rather than treating layer index as another feature.

A matching shape does not imply matching coordinates. Set the representation identity to include the model identifier, exact layer or module path, and capture/pooling convention. Supply an immutable checkpoint revision or a caller-owned checkpoint version in `LatentValue` metadata. Include tokenizer revision and token/padding/pooling rules when they affect the captured vectors. Values from different layers, model revisions, or pooling choices must not be combined just because both end in the same `hidden_dim`.

`HiddenStateAdapter` does not know a user's trained model, layer, or checkpoint revision. Its optional `random_state` makes its generated feature weights reproducible, but the caller must still label which seed/configuration produced the values if they will be compared with another representation.

## Runnable CPU example: adapt one named hidden layer

This fixed-weight two-layer sequence encoder exposes a capture option and returns only the requested named layer. The adapter selects `encoder.layers.0`; no hooks, checkpoint downloads, or model training are involved. The same pattern applies to an existing model if its forward path can expose the selected activation. Run this exact block after installing the base package:

```python
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from latent_anything import LatentSpace, LatentValue
from latent_anything.adapters import ModelAdapter


class TinySequenceModel(nn.Module):
    """Tiny fixed-weight sequence encoder with two named hidden layers."""

    def __init__(self) -> None:
        super().__init__()
        embedding = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.layers = nn.ModuleList((nn.Linear(3, 4), nn.Linear(4, 4)))
        with torch.no_grad():
            self.layers[0].weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                )
            )
            self.layers[0].bias.zero_()
            self.layers[1].weight.copy_(torch.eye(4))
            self.layers[1].bias.zero_()

    def forward(self, token_ids: torch.Tensor, *, capture_layer: str) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        hidden_0 = torch.tanh(self.layers[0](embedded))
        if capture_layer == "encoder.layers.0":
            return hidden_0
        if capture_layer == "encoder.layers.1":
            return torch.tanh(self.layers[1](hidden_0))
        raise ValueError(f"Unknown layer: {capture_layer}")


class TinyHiddenStateAdapter:
    """Expose one model layer through the released NumPy adapter protocol."""

    layer_name = "encoder.layers.0"
    model_revision = "fixed-fixture-v1"

    def __init__(self, model: TinySequenceModel) -> None:
        self.model = model.eval()

    @property
    def latent_space(self) -> LatentSpace:
        return LatentSpace(
            dim=4,
            source_model="tiny-sequence-encoder",
            metadata={
                "source_representation_identity": (
                    f"tiny-sequence-encoder:{self.layer_name}:token-hidden"
                )
            },
        )

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        with torch.inference_mode():
            selected = self.model(ids, capture_layer=self.layer_name)
        return selected.cpu().numpy()


adapter = TinyHiddenStateAdapter(TinySequenceModel())
input_ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
hidden = adapter.encode(input_ids)
value = LatentValue(
    hidden,
    adapter.latent_space,
    metadata={"model_version": adapter.model_revision},
)

assert isinstance(adapter, ModelAdapter)
assert hidden.shape == (2, 3, 4)
assert value.item_shape == (4,)
assert value.batch_shape == (2, 3)
assert "encoder.layers.0" in value.identity
assert adapter.model_revision in value.identity
print(f"input_ids={input_ids.shape}; selected hidden={hidden.shape}")
print(f"point={value.item_shape}; leading_axes={value.batch_shape}")
print(f"layer={adapter.layer_name}; revision={adapter.model_revision}")
print(f"first token hidden={np.round(hidden[0, 0], 4).tolist()}")
```

Expected shape summary (the example also prints the selected fixture activation):

```text
input_ids=(2, 3); selected hidden=(2, 3, 4)
point=(4,); leading_axes=(2, 3)
layer=encoder.layers.0; revision=fixed-fixture-v1
```

The code proves the NumPy adapter boundary, selected-layer delegation, shape/axis handling, and declared identity on this deterministic CPU fixture. It does not test a production transformer, demonstrate feature quality, or validate arbitrary model behavior. A live wrapper should use its model's documented inference path, set evaluation mode where appropriate, and return detached CPU NumPy data at the public adapter boundary.

## Capture deliberately

Choose the layer and token scope before materializing activations. Prefer the model's native hidden-state outputs when available; otherwise install only the hook(s) needed for the selected module and remove them after the forward pass. Avoid storing every layer/token when the analysis needs only one layer or a pooled vector. Record the attention mask, token pooling rule, layer path/index, model and tokenizer revisions, dtype, and any preprocessing that changes coordinates.

For a causal decoder-only language model, the optional source integration's native observation route is `output_hidden_states`; its current implementation reserves hooks for interventions. That implementation detail is not a `1.0.0` API promise. The bounded transformer evidence below captured the pinned model's layer outputs and used last-nonpadding-token pooling, with the target labels kept separate from the capture axes. Do not silently substitute a different tokenizer, pooling rule, layer numbering convention, or padding mask and still call it the same capture.

## Bounded diagnostic examples—not general model validation

The [AI engineer guide](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md) and its [encoder-v3 executable](https://github.com/triet4p/latent-anything/blob/main/scripts/ai_engineer_example_encoder.py) and [transformer executable](https://github.com/triet4p/latent-anything/blob/main/scripts/ai_engineer_example_transformer.py) are two separate, frozen diagnostic cases. Both scripts compose case-specific proof code; their coordinator is not a stable public import.

- **Encoder v3:** a committed four-dimensional linear-autoencoder checkpoint, a pinned train/held-out digits split, and a predeclared weight lesion. Its accepted result is bounded to that model, split, and brightness-bin task. It is not evidence about the fixed-random `HiddenStateAdapter`, arbitrary encoders, or general reconstruction quality.
- **Transformer target-evidence-v2:** this case pairs the frozen v1 transformer manifest with a separately versioned `diagnostic-evidence-v2` target record. It uses GPT-2 `openai-community/gpt2` at revision `e7da7f221d5bf496a48136c0cd264e630fe9fcc8` and the pinned Wikitext validation selection. The row-level section-header target is stored separately from real capture axes. The accepted probe result is evidence of separability under its frozen split and controls; it does not show that GPT-2 causally uses section headers. See the [accepted target-evidence-v2 report](https://github.com/triet4p/latent-anything/blob/main/artifacts/diagnostics/proof-80-25-committed-transformer-v1-target-v2-20260926-115522-629609-4488/diagnostic-report).
- **Separate stability supplement:** the target-evidence-v2 validator pass did not establish coefficient stability because that record's stability threshold was null. A separately frozen supplement applied its own predeclared `coef_stability >= 0.8` gate and observed `0.974878`; it did not mutate or retroactively gate the accepted target-evidence-v2 artifact. It reused the same selected Wikitext source corpus with a different grouped split, so it is not an independent corpus sample and remains a non-causal probe result. Read the [separate supplement report](https://github.com/triet4p/latent-anything/blob/main/artifacts/diagnostics/proof-80-25-committed-transformer-stability-supplement-20260926-115522-629609-4488/reports/5f88ac98fe84ab80ed93dfc5a0e08a9972e2665649e152ecb2030503494135d0.md).

The transformer example requires the exact model/data revisions, a local model snapshot, the optional dependencies, and the recorded overlays described in the [AI engineer guide's prerequisites](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md#prerequisites-and-frozen-inputs). It is a slow, heavyweight evidence replay, not a required smoke for the runnable CPU example on this page. `DiagnosticRequest` is the public request-configuration type; the proof-script `DiagnosticWorkflow` coordinator and model-specific compositions are not stable public imports. There is no no-configuration auto-diagnoser for arbitrary model/data pairs.

## Contract and next steps

For the released names and signatures, see the [frozen `1.0.0` API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md), [API freeze snapshot](https://github.com/triet4p/latent-anything/blob/v1.0.0/artifacts/api_freeze_snapshot_1.0.0.json), and [adapter protocol source at the release tag](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/adapters/protocols.py). Continue to [latent primitives and adapters](latent-primitives.md) for `LatentSpace`, `LatentValue`, and adapter capabilities, or return to the [guide home](index.md).
