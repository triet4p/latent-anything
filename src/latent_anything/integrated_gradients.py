"""Activation-space Integrated Gradients for decoder-only transformers.

The first attribution contract is deliberately narrow: one residual-block
output, one batch/token activation, and one scalar next-token logit.  The
module does not attribute input tokens or expose PyTorch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from latent_anything.tcav import TransformerLogitTarget

BaselineKind = Literal["zero", "batch_mean", "explicit"]
IntegrationRule = Literal["trapezoid", "riemann_left", "riemann_right"]


class IntegratedGradientsConfig(BaseModel):
    """Configuration for one activation-space attribution run."""

    target_layer: int = Field(default=6, ge=0)
    activation_position: int = Field(default=-1)
    activation_batch_index: int = Field(default=0, ge=0)
    baseline: BaselineKind = "zero"
    integration_rule: IntegrationRule = "trapezoid"
    n_steps: int = Field(default=32, ge=2, le=4096)


@dataclass(frozen=True)
class IntegratedGradientsResult:
    """Typed NumPy result for one scalar-target attribution path."""

    attributions: np.ndarray
    input_activation: np.ndarray
    baseline_activation: np.ndarray
    target_input: float
    target_baseline: float
    attribution_sum: float
    completeness_delta: float
    completeness_error: float
    convergence_delta: float
    n_steps: int
    integration_rule: IntegrationRule
    baseline_kind: BaselineKind
    target_layer: int
    activation_position: int
    target: TransformerLogitTarget
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        arrays = (self.attributions, self.input_activation, self.baseline_activation)
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("activation-space attribution arrays must be one-dimensional")
        if self.attributions.shape != self.input_activation.shape:
            raise ValueError("attributions and input_activation must have the same shape")
        if self.baseline_activation.shape != self.input_activation.shape:
            raise ValueError("baseline_activation and input_activation must have the same shape")
        if self.n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        for array in arrays:
            array.setflags(write=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary without large activation arrays."""
        return {
            "attribution_sum": self.attribution_sum,
            "completeness_delta": self.completeness_delta,
            "completeness_error": self.completeness_error,
            "convergence_delta": self.convergence_delta,
            "n_steps": self.n_steps,
            "integration_rule": self.integration_rule,
            "baseline_kind": self.baseline_kind,
            "target_layer": self.target_layer,
            "activation_position": self.activation_position,
            "target": self.target.model_dump(),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class SensitivityReport:
    """Bounded comparison of IG runs under declared perturbations."""

    step_counts: tuple[int, ...]
    completeness_errors: tuple[float, ...]
    attribution_cosines: tuple[float, ...]
    baseline_kinds: tuple[str, ...]
    target_token_ids: tuple[int, ...]
    randomization_cosine: float


def _as_readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _resolve_position(position: int, size: int, name: str) -> int:
    resolved = position if position >= 0 else size + position
    if not 0 <= resolved < size:
        raise ValueError(f"{name}={position} is outside dimension of size {size}")
    return resolved


def _layer_module(model: Any, layer: int) -> Any:
    name = f"transformer.h.{layer}"
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    raise ValueError(f"Layer {layer} ({name}) was not found in the model")


def _first_output(output: Any) -> Any:
    return output if hasattr(output, "shape") else output[0]


def _replace_first_output(output: Any, replacement: Any) -> Any:
    if hasattr(output, "shape"):
        return replacement
    values = list(output)
    values[0] = replacement
    return tuple(values)


def _target_scalar(logits: Any, target: TransformerLogitTarget) -> Any:
    batch = target.batch_index
    position = _resolve_position(target.position, int(logits.shape[1]), "target.position")
    if batch >= int(logits.shape[0]):
        raise ValueError(f"target.batch_index={batch} is outside batch of size {logits.shape[0]}")
    if target.token_id >= int(logits.shape[2]):
        raise ValueError(f"target.token_id={target.token_id} is outside vocabulary of size {logits.shape[2]}")
    return logits[batch, position, target.token_id]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


class IntegratedGradients:
    """Compute activation-space Integrated Gradients through a transformer hook."""

    def __init__(self, config: IntegratedGradientsConfig | None = None) -> None:
        self.config = config or IntegratedGradientsConfig()

    def compute(
        self,
        model: Any,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        target: TransformerLogitTarget,
        *,
        baseline_activation: np.ndarray | None = None,
        source_model_version: str = "",
    ) -> IntegratedGradientsResult:
        """Integrate gradients along a straight activation-space path."""
        import torch

        ids = np.asarray(input_ids)
        mask = np.asarray(attention_mask)
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError("input_ids must be 2D and attention_mask must have the same shape")
        device = next(model.parameters()).device
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)
        mask_t = torch.as_tensor(mask, dtype=torch.long, device=device)
        module = _layer_module(model, self.config.target_layer)
        captured: dict[str, Any] = {}

        def capture_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            activation = _first_output(output)
            batch = self.config.activation_batch_index
            position = _resolve_position(
                self.config.activation_position,
                int(activation.shape[1]),
                "activation_position",
            )
            if batch >= int(activation.shape[0]):
                raise ValueError("activation_batch_index is outside the input batch")
            captured["full"] = activation
            captured["selected"] = activation[batch, position]
            return output

        handle = module.register_forward_hook(capture_hook)
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                model(input_ids=ids_t, attention_mask=mask_t, output_hidden_states=False)
                input_full = captured["full"].detach()
                input_vector = captured["selected"].detach()
                if baseline_activation is not None:
                    baseline_kind: BaselineKind = "explicit"
                    baseline_vector = np.asarray(baseline_activation, dtype=np.float64)
                elif self.config.baseline == "zero":
                    baseline_kind = "zero"
                    baseline_vector = np.zeros(tuple(input_vector.shape), dtype=np.float64)
                else:
                    baseline_kind = "batch_mean"
                    baseline_vector = (
                        input_full.detach()[:, self.config.activation_position, :].mean(dim=0).cpu().numpy()
                    )
                if baseline_vector.shape != tuple(input_vector.shape):
                    raise ValueError(f"baseline_activation must have shape {tuple(input_vector.shape)}")
                baseline_full = input_full.detach().clone()
                batch = self.config.activation_batch_index
                position = _resolve_position(
                    self.config.activation_position,
                    int(input_full.shape[1]),
                    "activation_position",
                )
                baseline_full[batch, position] = torch.as_tensor(baseline_vector, device=device, dtype=input_full.dtype)

                baseline_target = self._forward_with_activation(
                    model, ids_t, mask_t, baseline_full, target, module, torch
                )
                input_target = self._forward_with_activation(model, ids_t, mask_t, input_full, target, module, torch)
                gradients: list[Any] = []
                for step in range(self.config.n_steps + 1):
                    alpha = step / self.config.n_steps
                    path_full = baseline_full + alpha * (input_full - baseline_full)
                    gradients.append(
                        self._gradient_at(
                            model,
                            ids_t,
                            mask_t,
                            path_full,
                            target,
                            module,
                            self.config.activation_batch_index,
                            position,
                            torch,
                        )
                    )

                stacked = np.stack(gradients, axis=0)
                delta = _as_readonly(np.asarray(input_vector.detach().cpu().numpy()) - baseline_vector)
                if self.config.integration_rule == "trapezoid":
                    average = (stacked[0] + stacked[-1] + 2.0 * stacked[1:-1].sum(axis=0)) / (2.0 * self.config.n_steps)
                elif self.config.integration_rule == "riemann_left":
                    average = stacked[:-1].mean(axis=0)
                else:
                    average = stacked[1:].mean(axis=0)
                attributions = _as_readonly(delta * average)
                attribution_sum = float(attributions.sum())
                completeness_delta = float(input_target - baseline_target)
                completeness_error = float(attribution_sum - completeness_delta)
                provenance = {
                    "source_model_version": source_model_version,
                    "target_layer": self.config.target_layer,
                    "activation_position": position,
                    "target": target.model_dump(),
                    "baseline": baseline_kind,
                    "integration_rule": self.config.integration_rule,
                }
                return IntegratedGradientsResult(
                    attributions=attributions,
                    input_activation=_as_readonly(input_vector.detach().cpu().numpy()),
                    baseline_activation=_as_readonly(baseline_vector),
                    target_input=float(input_target),
                    target_baseline=float(baseline_target),
                    attribution_sum=attribution_sum,
                    completeness_delta=completeness_delta,
                    completeness_error=completeness_error,
                    convergence_delta=float(abs(completeness_error)),
                    n_steps=self.config.n_steps,
                    integration_rule=self.config.integration_rule,
                    baseline_kind=baseline_kind,
                    target_layer=self.config.target_layer,
                    activation_position=position,
                    target=target,
                    provenance=provenance,
                )
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)

    @staticmethod
    def _forward_with_activation(
        model: Any,
        ids: Any,
        mask: Any,
        activation: Any,
        target: Any,
        module: Any,
        torch: Any,
    ) -> float:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            return _replace_first_output(output, activation)

        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                outputs = model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
                return float(_target_scalar(outputs.logits, target).item())
        finally:
            handle.remove()

    @staticmethod
    def _gradient_at(
        model: Any,
        ids: Any,
        mask: Any,
        activation: Any,
        target: Any,
        module: Any,
        batch_index: int,
        position: int,
        torch: Any,
    ) -> np.ndarray:
        path = activation.detach().clone().requires_grad_(True)

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            return _replace_first_output(output, path)

        handle = module.register_forward_hook(hook)
        try:
            output = model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
            scalar = _target_scalar(output.logits, target)
            gradient = torch.autograd.grad(scalar, path, retain_graph=False, create_graph=False)[0]
            return gradient[batch_index, position].detach().cpu().numpy().astype(np.float64)
        finally:
            handle.remove()


def evaluate_sensitivity(
    runs: tuple[IntegratedGradientsResult, ...],
    *,
    randomized: IntegratedGradientsResult | None = None,
) -> SensitivityReport:
    """Summarize bounded step, baseline, target, and randomization checks."""
    if not runs:
        raise ValueError("runs must not be empty")
    reference = runs[0].attributions
    return SensitivityReport(
        step_counts=tuple(run.n_steps for run in runs),
        completeness_errors=tuple(abs(run.completeness_error) for run in runs),
        attribution_cosines=tuple(_cosine(reference, run.attributions) for run in runs),
        baseline_kinds=tuple(run.baseline_kind for run in runs),
        target_token_ids=tuple(run.target.token_id for run in runs),
        randomization_cosine=0.0 if randomized is None else _cosine(reference, randomized.attributions),
    )


def compute_integrated_gradients(
    model: Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    target: TransformerLogitTarget,
    *,
    config: IntegratedGradientsConfig | None = None,
    baseline_activation: np.ndarray | None = None,
    source_model_version: str = "",
) -> IntegratedGradientsResult:
    """Functional convenience wrapper around :class:`IntegratedGradients`."""
    return IntegratedGradients(config).compute(
        model,
        input_ids,
        attention_mask,
        target,
        baseline_activation=baseline_activation,
        source_model_version=source_model_version,
    )
