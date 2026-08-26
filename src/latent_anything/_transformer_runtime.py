"""Private model-bound transformer forward, capture, and intervention runtime."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from latent_anything._transformer_analysis import (
    apply_logit_lens,
    compute_token_rank_trajectories,
    compute_top_tokens,
    softmax,
)
from latent_anything._transformer_backend import tokenize
from latent_anything.capture import ActivationCaptureSession


@dataclass(frozen=True)
class RuntimeGenerationResult:
    """Torch-free intermediate values ready for public result conversion."""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    logits: np.ndarray
    hidden_states: tuple[tuple[int, np.ndarray, dict[str, str]], ...]
    lens_results: tuple[tuple[int, np.ndarray, np.ndarray, list[list[list[tuple[int, float]]]], int], ...]
    token_rank_trajectories: tuple[tuple[int, str, list[int], list[float], list[int]], ...]


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
) -> RuntimeGenerationResult:
    import torch

    encoded = tokenize(tokenizer, request.prompt, max_length=request.max_length, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    _, seq_len = input_ids.shape

    num_layers = int(getattr(config, "num_hidden_layers", default_num_layers))
    if request.capture_layers:
        capture_layers = sorted(request.capture_layers)
    elif request.capture_hidden_states:
        capture_layers = tuple(range(num_layers + 1))
    else:
        capture_layers = ()

    need_intervention = intervention is not None
    need_capture = request.capture_hidden_states or len(capture_layers) > 0
    module_locations: list[str] = []
    if need_intervention:
        location = f"transformer.h.{intervention.layer}"
        if location not in module_locations:
            module_locations.append(location)
    if need_capture:
        # Native output_hidden_states is the observation path; hooks are only
        # needed for intervention and therefore do not duplicate capture work.
        pass

    intervention_fn: Any = None
    if need_intervention:
        direction_t = torch.tensor(intervention.direction, dtype=torch.float32)
        strength_val = intervention.strength
        token_indices = intervention.token_indices
        target_dtype = direction_t.dtype

        def intervene_callback(tensor: torch.Tensor, _metadata: Any) -> torch.Tensor:
            modified = tensor.clone()
            delta = strength_val * direction_t.to(device=tensor.device, dtype=tensor.dtype)
            if token_indices is not None:
                for batch_index, sequence_index in token_indices:
                    if batch_index < modified.shape[0] and sequence_index < modified.shape[1]:
                        modified[batch_index, sequence_index] = (
                            modified[batch_index, sequence_index] + delta[batch_index, sequence_index]
                        )
            else:
                modified = modified + delta
            return modified.to(dtype=target_dtype)

        intervention_fn = intervene_callback

    capture_session = nullcontext()
    if module_locations and intervention_fn is not None:
        capture_session = ActivationCaptureSession(
            model,
            module_locations,
            source_model_version=provenance,
            intervention=intervention_fn,
        )

    with capture_session, torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    final_logits = outputs.logits.detach().cpu().numpy().copy()
    final_logits.setflags(write=False)
    native_hidden_states = outputs.hidden_states
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
        for layer_index in capture_layers:
            if layer_index < len(native_hidden_states):
                logits = apply_logit_lens(model, native_hidden_states[layer_index])
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
    )
