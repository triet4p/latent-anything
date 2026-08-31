"""Private model-bound transformer forward, capture, and intervention runtime."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from latent_anything._transformer_analysis import (
    apply_logit_lens,
    compute_token_rank_trajectories,
    compute_top_tokens,
    softmax,
)
from latent_anything._transformer_backend import tokenize
from latent_anything.capture import ActivationCaptureSession


class TransformerRuntimeShapeError(ValueError):
    """Sanitized shape-contract failure from one transformer forward.

    The error deliberately contains only tensor field names and integer
    shapes.  It is safe to carry through a failure envelope without exposing
    prompts, token IDs, or tensor values.
    """

    def __init__(self, field: str, expected: tuple[int, ...], actual: object) -> None:
        self.field = field
        self.expected_shape = expected
        self.actual_shape = actual if isinstance(actual, tuple) else None
        super().__init__(f"transformer_runtime_shape_error:{field}:expected={expected}:actual={actual}")


def _shape(value: object) -> tuple[int, ...] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    try:
        return tuple(int(size) for size in raw)
    except (TypeError, ValueError, OverflowError):
        return None


def _validate_forward_shapes(
    input_ids: object,
    attention_mask: object,
    logits: object,
    native_hidden_states: object,
) -> None:
    """Enforce the full-prompt batch/sequence contract before indexing."""
    input_shape = _shape(input_ids)
    if input_shape is None or len(input_shape) != 2:
        raise TransformerRuntimeShapeError("input_ids", (0, 0), input_shape)
    mask_shape = _shape(attention_mask)
    if mask_shape != input_shape:
        raise TransformerRuntimeShapeError("attention_mask", input_shape, mask_shape)
    logits_shape = _shape(logits)
    expected_logits_prefix = input_shape
    if logits_shape is None or len(logits_shape) != 3 or logits_shape[:2] != expected_logits_prefix:
        raise TransformerRuntimeShapeError("logits", (*expected_logits_prefix, -1), logits_shape)
    if native_hidden_states is None:
        return
    if not isinstance(native_hidden_states, Sequence):
        raise TransformerRuntimeShapeError("hidden_states", (0, *input_shape, -1), None) from None
    states = tuple(cast(Sequence[Any], native_hidden_states))
    for index, state in enumerate(states):
        state_shape = _shape(state)
        if state_shape is None or len(state_shape) != 3 or state_shape[:2] != expected_logits_prefix:
            raise TransformerRuntimeShapeError(f"hidden_states[{index}]", (*expected_logits_prefix, -1), state_shape)


@dataclass(frozen=True)
class RuntimeGenerationResult:
    """Torch-free intermediate values ready for public result conversion."""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    logits: np.ndarray
    hidden_states: tuple[tuple[int, np.ndarray, dict[str, str]], ...]
    lens_results: tuple[tuple[int, np.ndarray, np.ndarray, list[list[list[tuple[int, float]]]], int], ...]
    token_rank_trajectories: tuple[tuple[int, str, list[int], list[float], list[int]], ...]
    raw_block_states: tuple[tuple[int, np.ndarray, dict[str, str]], ...]


def run_generation(
    model: Any,
    tokenizer: Any,
    config: Any,
    request: Any,
    intervention: Any,
    *,
    device: str,
    provenance: str,
    default_num_layers: int,
    raw_capture_layers: Sequence[int] = (),
) -> RuntimeGenerationResult:
    import torch

    encoded = tokenize(tokenizer, request.prompt, max_length=request.max_length, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    input_shape = _shape(input_ids)
    if input_shape is None or len(input_shape) != 2:
        raise TransformerRuntimeShapeError("input_ids", (0, 0), input_shape)
    mask_shape = _shape(attention_mask)
    if mask_shape != input_shape:
        raise TransformerRuntimeShapeError("attention_mask", input_shape, mask_shape)
    _, seq_len = input_shape

    num_layers = int(getattr(config, "num_hidden_layers", default_num_layers))
    raw_layers = tuple(sorted({int(layer) for layer in raw_capture_layers}))
    if any(layer < 0 or layer >= num_layers for layer in raw_layers):
        raise ValueError("raw block capture layer is outside the configured transformer depth")
    if request.capture_layers:
        capture_layers = sorted(request.capture_layers)
    elif request.capture_hidden_states:
        capture_layers = tuple(range(num_layers + 1))
    else:
        capture_layers = ()

    need_intervention = intervention is not None
    need_capture = request.capture_hidden_states or len(capture_layers) > 0
    module_locations: list[str] = []
    raw_locations = [f"transformer.h.{layer}" for layer in raw_layers]
    if need_intervention:
        location = f"transformer.h.{intervention.layer}"
        if location not in module_locations:
            module_locations.append(location)
    for location in raw_locations:
        if location not in module_locations:
            module_locations.append(location)
    if need_capture:
        # Native output_hidden_states is the observation path; hooks are only
        # needed for intervention and therefore do not duplicate capture work.
        pass

    intervention_fn: Any = None
    raw_post_intervention: dict[int, np.ndarray] = {}
    if need_intervention:
        direction_t = torch.tensor(intervention.direction, dtype=torch.float32)
        strength_val = intervention.strength
        token_indices = intervention.token_indices
        target_location = f"transformer.h.{intervention.layer}"

        def intervene_callback(tensor: torch.Tensor, metadata: Any) -> torch.Tensor:
            if metadata.location != target_location:
                return tensor
            modified = tensor.clone()
            delta = strength_val * direction_t.to(device=tensor.device, dtype=tensor.dtype)
            if token_indices is not None:
                for batch_index, sequence_index in token_indices:
                    if batch_index < 0 or sequence_index < 0:
                        raise TransformerRuntimeShapeError(
                            "intervention.token_indices", tuple(modified.shape), (batch_index, sequence_index)
                        )
                    if batch_index >= modified.shape[0] or sequence_index >= modified.shape[1]:
                        raise TransformerRuntimeShapeError(
                            "intervention.token_indices", tuple(modified.shape), (batch_index, sequence_index)
                        )
                    delta_batch = batch_index if delta.shape[0] > 1 else 0
                    delta_sequence = sequence_index if delta.shape[1] > 1 else 0
                    modified[batch_index, sequence_index] = (
                        modified[batch_index, sequence_index] + delta[delta_batch, delta_sequence]
                    )
            else:
                modified = modified + delta
            if intervention.layer in raw_layers:
                values = modified.detach().cpu().numpy().copy()
                values.setflags(write=False)
                raw_post_intervention[int(intervention.layer)] = values
            return modified

        intervention_fn = intervene_callback

    capture_session = nullcontext()
    raw_session: ActivationCaptureSession | None = None
    if module_locations:
        capture_session = ActivationCaptureSession(
            model,
            module_locations,
            source_model_version=provenance,
            intervention=intervention_fn,
        )
        raw_session = cast(ActivationCaptureSession, capture_session)

    with capture_session, torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    final_logits_tensor = getattr(outputs, "logits", None)
    native_hidden_states = getattr(outputs, "hidden_states", None)
    _validate_forward_shapes(input_ids, attention_mask, final_logits_tensor, native_hidden_states)
    if need_capture and native_hidden_states is None:
        raise TransformerRuntimeShapeError("hidden_states", (*input_shape, -1), None)
    if final_logits_tensor is None:
        raise TransformerRuntimeShapeError("logits", (*input_ids.shape, -1), None)
    final_logits = final_logits_tensor.detach().cpu().numpy().copy()
    final_logits.setflags(write=False)
    captured_states: list[tuple[int, np.ndarray, dict[str, str]]] = []
    lens_results: list[tuple[int, np.ndarray, np.ndarray, list[list[list[tuple[int, float]]]], int]] = []

    if native_hidden_states is not None and need_capture:
        for layer_index in capture_layers:
            if layer_index < len(native_hidden_states):
                hidden_state = native_hidden_states[layer_index]
                hidden_values = hidden_state.detach().cpu().numpy().copy()
                hidden_values.setflags(write=False)
                captured_states.append(
                    (
                        layer_index,
                        hidden_values,
                        {
                            "shape": str(tuple(hidden_values.shape)),
                            "source": "native_output_hidden_states",
                        },
                    )
                )

    if native_hidden_states is not None:
        final_native_index = len(native_hidden_states) - 1
        for layer_index in capture_layers:
            if layer_index < len(native_hidden_states):
                logits = apply_logit_lens(
                    model,
                    native_hidden_states[layer_index],
                    apply_final_norm=layer_index != final_native_index,
                )
                probabilities = softmax(logits)
                top_tokens = (
                    compute_top_tokens(probabilities, request.top_k_logit_lens) if request.top_k_logit_lens > 0 else []
                )
                lens_results.append((layer_index, logits, probabilities, top_tokens, request.top_k_logit_lens))

    rank_inputs = [
        type(
            "LensResult",
            (),
            {"layer": layer_index, "probabilities": probabilities},
        )()
        for layer_index, _logits, probabilities, _top_tokens, _top_k in lens_results
    ]
    token_rank_trajectories = compute_token_rank_trajectories(rank_inputs, seq_len, tokenizer)

    raw_block_states_map: dict[int, tuple[np.ndarray, dict[str, str]]] = {}
    if raw_session is not None:
        for capture in raw_session.captures:
            prefix = "transformer.h."
            if capture.metadata.location.startswith(prefix):
                layer = int(capture.metadata.location.removeprefix(prefix))
                if layer in raw_layers:
                    raw_block_states_map[layer] = (
                        capture.values,
                        {
                            "shape": str(tuple(capture.values.shape)),
                            "source": "forward_hook_pre_intervention",
                        },
                    )
    for layer, values in raw_post_intervention.items():
        if layer in raw_layers:
            raw_block_states_map[layer] = (
                values,
                {
                    "shape": str(tuple(values.shape)),
                    "source": "forward_hook_post_intervention",
                },
            )
    for layer in raw_layers:
        captured = raw_block_states_map.get(layer)
        if captured is None:
            raise TransformerRuntimeShapeError(f"raw_block_states[{layer}]", (*input_shape, -1), None)
        values, _metadata = captured
        values_shape = _shape(values)
        if values_shape is None or len(values_shape) != 3 or values_shape[:2] != input_shape:
            raise TransformerRuntimeShapeError(f"raw_block_states[{layer}]", (*input_shape, -1), values_shape)
    raw_block_states = tuple(
        (layer, values, metadata)
        for layer in raw_layers
        if (values_metadata := raw_block_states_map.get(layer)) is not None
        for values, metadata in (values_metadata,)
    )

    input_ids_np = input_ids.detach().cpu().numpy().copy()
    input_ids_np.setflags(write=False)
    attention_mask_np = attention_mask.detach().cpu().numpy().copy()
    attention_mask_np.setflags(write=False)
    return RuntimeGenerationResult(
        input_ids=input_ids_np,
        attention_mask=attention_mask_np,
        logits=final_logits,
        hidden_states=tuple(captured_states),
        lens_results=tuple(lens_results),
        token_rank_trajectories=token_rank_trajectories,
        raw_block_states=raw_block_states,
    )
