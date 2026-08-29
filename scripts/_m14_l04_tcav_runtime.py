"""Model-boundary helpers for the M14 L04 TCAV protocol."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from latent_anything._hook_output import extract_primary_tensor, replace_primary_tensor


class RealExecutionError(RuntimeError):
    """Execution failure carrying sanitized resource facts."""

    def __init__(self, message: str, resources: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.resources = dict(resources)


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_digest(model: Any) -> str:
    """Hash parameters in canonical named, ordered, shape-aware form."""
    digest = hashlib.sha256()
    named_parameters = model.named_parameters()
    for name, parameter in named_parameters:
        value = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_rows(integration: Any, rows: Sequence[Mapping[str, Any]], max_length: int) -> list[dict[str, Any]]:
    prompts = tuple(str(row["prompt"]) for row in rows)
    encoded = integration.tokenize(prompts, max_length=max_length, return_tensors="pt")
    ids = encoded["input_ids"].detach().cpu().numpy()
    masks = encoded["attention_mask"].detach().cpu().numpy()
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        mask = np.asarray(masks[index], dtype=np.int64)
        if not np.any(mask):
            raise ValueError(f"row {row['row_id']!r} has no non-padding token")
        result.append(
            {
                "row_id": str(row["row_id"]),
                "group_id": str(row["group_id"]),
                "causal_pair_id": str(row["causal_pair_id"]),
                "split": str(row["split"]),
                "input_ids": np.asarray(ids[index], dtype=np.int64),
                "attention_mask": mask,
                "target_position": int(np.flatnonzero(mask).max()),
            }
        )
    return result


def resolve_target_token(tokenizer: Any, text: str) -> tuple[int, str]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, Sequence) or len(values) != 1:
        raise ValueError(f"target text {text!r} must resolve to exactly one token")
    token_id = int(values[0])
    token_string = str(tokenizer.decode([token_id])) if hasattr(tokenizer, "decode") else text
    return token_id, token_string


def capture_activations(model: Any, rows: Sequence[Mapping[str, Any]], layer: int) -> np.ndarray:
    """Capture one layer activation per row using the existing hook primitive."""
    from latent_anything._tcav_model import extract_layer_activation

    return np.stack(
        [
            extract_layer_activation(
                model,
                np.asarray(row["input_ids"])[None, :],
                np.asarray(row["attention_mask"])[None, :],
                layer,
                0,
                int(row["target_position"]),
            )
            for row in rows
        ],
        axis=0,
    ).astype(np.float64)


def task_margin_gradient(
    model: Any,
    row: Mapping[str, Any],
    *,
    layer: int,
    target_token: int,
    other_token: int,
) -> np.ndarray:
    """Compute ∇(target-logit − other-logit) in one hooked forward."""
    import torch

    device = next(model.parameters()).device
    input_t = torch.as_tensor(np.asarray(row["input_ids"])[None, :], dtype=torch.long, device=device)
    mask_t = torch.as_tensor(np.asarray(row["attention_mask"])[None, :], dtype=torch.long, device=device)
    captured: dict[str, torch.Tensor] = {}
    name = f"transformer.h.{layer}"

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        tensor = extract_primary_tensor(output)
        tensor.retain_grad()
        captured[name] = tensor

    handle = next(
        (module.register_forward_hook(hook) for module_name, module in model.named_modules() if module_name == name),
        None,
    )
    if handle is None:
        raise ValueError(f"Layer {layer} ({name}) not found in model")
    try:
        model.zero_grad()
        with torch.enable_grad():
            output = model(input_ids=input_t, attention_mask=mask_t, output_hidden_states=False)
            logits = output.logits
            position = int(row["target_position"])
            scalar = logits[0, position, target_token] - logits[0, position, other_token]
        scalar.backward()
        activation = captured.get(name)
        if activation is None or activation.grad is None:
            raise RuntimeError("TCAV layer gradient was not captured")
        return activation.grad[0, int(row["target_position"])].detach().cpu().numpy().astype(np.float64, copy=True)
    finally:
        handle.remove()


def task_margin(model: Any, row: Mapping[str, Any], *, target_token: int, other_token: int) -> float:
    import torch

    device = next(model.parameters()).device
    ids = torch.as_tensor(np.asarray(row["input_ids"])[None, :], dtype=torch.long, device=device)
    mask = torch.as_tensor(np.asarray(row["attention_mask"])[None, :], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    pos = int(row["target_position"])
    return float((logits[0, pos, target_token] - logits[0, pos, other_token]).detach().cpu())


def intervened_margin(
    model: Any,
    row: Mapping[str, Any],
    *,
    layer: int,
    direction: np.ndarray,
    target_token: int,
    other_token: int,
    strength: float,
) -> float:
    import torch

    device = next(model.parameters()).device
    ids = torch.as_tensor(np.asarray(row["input_ids"])[None, :], dtype=torch.long, device=device)
    mask = torch.as_tensor(np.asarray(row["attention_mask"])[None, :], dtype=torch.long, device=device)
    vector = torch.as_tensor(direction, dtype=torch.float32, device=device)
    name = f"transformer.h.{layer}"

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = extract_primary_tensor(output)
        delta = vector.to(dtype=tensor.dtype, device=tensor.device) * float(strength)
        position = int(row["target_position"])
        changed = tensor.clone()
        changed[:, position, :] = changed[:, position, :] + delta.reshape(-1)
        return replace_primary_tensor(output, changed)

    handle = next(
        (module.register_forward_hook(hook) for module_name, module in model.named_modules() if module_name == name),
        None,
    )
    if handle is None:
        raise ValueError(f"Layer {layer} ({name}) not found in model")
    try:
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        pos = int(row["target_position"])
        return float((logits[0, pos, target_token] - logits[0, pos, other_token]).detach().cpu())
    finally:
        handle.remove()


__all__ = [
    "RealExecutionError",
    "capture_activations",
    "intervened_margin",
    "parameter_digest",
    "read_rows",
    "resolve_target_token",
    "seed_everything",
    "task_margin",
    "task_margin_gradient",
]
