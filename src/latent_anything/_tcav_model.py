"""Optional PyTorch model-boundary operations for TCAV.

The public TCAV facade owns concept statistics and result assembly.  This
module is the only internal boundary that imports PyTorch at call time: it
captures transformer activations, gradients, and bounded intervention outputs
without exposing tensors through the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from latent_anything._hook_output import extract_primary_tensor, replace_primary_tensor

if TYPE_CHECKING:
    from latent_anything.tcav import TransformerLogitTarget


def compute_transformer_layer_gradient(
    model: Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    layer: int,
    target: TransformerLogitTarget,
) -> np.ndarray:
    """Compute a target-logit gradient at one transformer layer."""
    import torch

    device = next(model.parameters()).device
    input_t = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    mask_t = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
    activation: dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def _hook(_module: Any, _input: Any, output: Any) -> None:
            tensor = extract_primary_tensor(output)
            tensor.retain_grad()
            activation[name] = tensor

        return _hook

    layer_name = f"transformer.h.{layer}"
    handle = None
    for n, m in model.named_modules():
        if n == layer_name:
            handle = m.register_forward_hook(_make_hook(layer_name))
            break

    if handle is None:
        raise ValueError(
            f"Layer {layer} ({layer_name}) not found in model. "
            f"Available layers: {[n for n, _ in model.named_modules() if 'transformer.h.' in n]}"
        )

    try:
        model.zero_grad()
        with torch.enable_grad():
            outputs = model(input_ids=input_t, attention_mask=mask_t, output_hidden_states=False)
            logits = outputs.logits

        batch_idx = target.batch_index
        pos = target.position if target.position >= 0 else logits.shape[1] + target.position
        scalar = logits[batch_idx, pos, target.token_id]
        scalar.backward()

        if layer_name not in activation:
            raise RuntimeError(f"Hook at {layer_name} did not fire during forward pass")
        act = activation[layer_name]
        if act.grad is None:
            raise RuntimeError(
                f"Gradient not available at {layer_name}. "
                "Ensure the model is in eval mode (not train) and gradients are enabled."
            )
        grad: np.ndarray = act.grad[batch_idx, pos].detach().cpu().numpy().copy()
        return grad.astype(np.float64)
    finally:
        handle.remove()


def extract_layer_activation(
    model: Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    layer: int,
    batch_index: int = 0,
    position: int = -1,
) -> np.ndarray:
    """Extract one transformer-layer activation without computing gradients."""
    import torch

    device = next(model.parameters()).device
    input_t = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    mask_t = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
    activation: dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def _hook(_module: Any, _input: Any, output: Any) -> None:
            out = extract_primary_tensor(output)
            activation[name] = out.detach().cpu()

        return _hook

    layer_name = f"transformer.h.{layer}"
    handle = None
    for n, m in model.named_modules():
        if n == layer_name:
            handle = m.register_forward_hook(_make_hook(layer_name))
            break

    if handle is None:
        raise ValueError(f"Layer {layer} not found in model")

    try:
        with torch.no_grad():
            model(input_ids=input_t, attention_mask=mask_t, output_hidden_states=False)
        if layer_name not in activation:
            raise RuntimeError(f"Hook at {layer_name} did not fire")
        act = activation[layer_name].numpy()
        pos = position if position >= 0 else act.shape[1] + position
        return act[batch_index, pos].astype(np.float64)
    finally:
        handle.remove()


def intervention_agreement(
    model: Any,
    target_layer: int,
    concept_direction: np.ndarray,
    input_ids_batch: np.ndarray,
    attention_mask_batch: np.ndarray,
    target: TransformerLogitTarget,
    *,
    strength: float = 1.0,
) -> float:
    """Compare matched positive/negative activation interventions to TCAV."""
    import torch

    device = next(model.parameters()).device
    batch_size = input_ids_batch.shape[0]
    agreements: list[float] = []
    v_c_t = torch.as_tensor(concept_direction, dtype=torch.float32, device=device)

    for i in range(batch_size):
        ids = input_ids_batch[i : i + 1]
        mask = attention_mask_batch[i : i + 1]
        pos = target.position if target.position >= 0 else ids.shape[1] + target.position
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)
        mask_t = torch.as_tensor(mask, dtype=torch.long, device=device)

        with torch.no_grad():
            baseline_logits = model(input_ids=ids_t, attention_mask=mask_t).logits
            baseline_val = float(baseline_logits[0, pos, target.token_id].cpu().numpy())

        layer_name = f"transformer.h.{target_layer}"

        def _pos_hook(_module: Any, _input: Any, output: Any) -> Any:
            out = extract_primary_tensor(output)
            delta = strength * v_c_t.to(dtype=out.dtype, device=out.device)
            return replace_primary_tensor(output, out + delta)

        pos_handle = None
        for n, m in model.named_modules():
            if n == layer_name:
                pos_handle = m.register_forward_hook(_pos_hook)
                break
        if pos_handle is None:
            raise ValueError(f"Layer {target_layer} not found")
        try:
            with torch.no_grad():
                pos_out = model(input_ids=ids_t, attention_mask=mask_t).logits
                pos_val = float(pos_out[0, pos, target.token_id].cpu().numpy())
        finally:
            pos_handle.remove()

        def _neg_hook(_module: Any, _input: Any, output: Any) -> Any:
            out = extract_primary_tensor(output)
            delta = strength * v_c_t.to(dtype=out.dtype, device=out.device)
            return replace_primary_tensor(output, out - delta)

        neg_handle = None
        for n, m in model.named_modules():
            if n == layer_name:
                neg_handle = m.register_forward_hook(_neg_hook)
                break
        if neg_handle is None:
            raise ValueError(f"Layer {target_layer} not found")
        try:
            with torch.no_grad():
                neg_out = model(input_ids=ids_t, attention_mask=mask_t).logits
                neg_val = float(neg_out[0, pos, target.token_id].cpu().numpy())
        finally:
            neg_handle.remove()

        grad = compute_transformer_layer_gradient(
            model,
            input_ids=ids,
            attention_mask=mask,
            layer=target_layer,
            target=target,
        )
        ddt = float(np.dot(grad, concept_direction))
        if ddt > 0:
            agrees_positive = pos_val > baseline_val
            agrees_negative = neg_val < baseline_val
        elif ddt < 0:
            agrees_positive = pos_val < baseline_val
            agrees_negative = neg_val > baseline_val
        else:
            agrees_positive = True
            agrees_negative = True
        agreements.append(1.0 if (agrees_positive and agrees_negative) else 0.0)

    return float(np.mean(agreements))
