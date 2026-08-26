"""Private SmolVLA intervention measurements and report assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from latent_anything.integrations.lerobot_smolvla import (
        SmolVLAActionSelection,
        SmolVLAIntervention,
        SmolVLAInterventionMeasurement,
        SmolVLAPolicyAdapter,
    )


def measure_smolvla_intervention(
    adapter: SmolVLAPolicyAdapter,
    samples: Sequence[Mapping[str, object]],
    *,
    noise: np.ndarray,
    intervention: SmolVLAIntervention,
    alternate_prompt_sample: Mapping[str, object] | None = None,
    camera_swapped_sample: Mapping[str, object] | None = None,
) -> SmolVLAInterventionMeasurement:
    """Measure action, representation, prompt, and camera-order effects."""

    from latent_anything.integrations.lerobot_smolvla import SmolVLAInterventionMeasurement

    if not samples:
        raise ValueError("at least one sample is required")
    noise_array = np.array(noise, copy=True)
    adapter.reset()
    baseline: list[SmolVLAActionSelection] = []
    for sample in samples:
        baseline.append(adapter.select_action(sample, noise=noise_array))
    adapter.reset()
    intervened: list[SmolVLAActionSelection] = []
    for sample in samples:
        intervened.append(adapter.select_action(sample, noise=noise_array, intervention=intervention))
    deltas = [
        np.asarray(item.action_array).reshape(-1) - np.asarray(base.action_array).reshape(-1)
        for item, base in zip(intervened, baseline, strict=True)
    ]
    changes = np.stack(deltas)
    action_change_norm = float(np.mean(np.linalg.norm(changes, axis=1)))
    action_change_per_dim = np.mean(np.abs(changes), axis=0)

    induced = measure_induced_action_direction(adapter, intervention.direction, adapter.action_dim)
    unit = induced / np.linalg.norm(induced)
    projections = changes @ unit
    on_target_norm = float(np.mean(np.abs(projections)))
    residuals = changes - projections[:, None] * unit[None, :]
    off_target_norm = float(np.mean(np.linalg.norm(residuals, axis=1)))
    total = on_target_norm + off_target_norm
    on_target_fraction = on_target_norm / total if total > 0.0 else 0.0

    drift = measure_representation_drift(baseline, intervened)
    first_step_drift = measure_first_step_drift(baseline[0], intervened[0])
    prompt_sensitivity = 0.0
    if alternate_prompt_sample is not None:
        adapter.reset()
        alternate = adapter.select_action(alternate_prompt_sample, noise=noise_array)
        prompt_sensitivity = float(np.linalg.norm(alternate.action_array - baseline[0].action_array))
    camera_order_sensitivity = 0.0
    if camera_swapped_sample is not None:
        adapter.reset()
        swapped = adapter.select_action(camera_swapped_sample, noise=noise_array)
        camera_order_sensitivity = float(np.linalg.norm(swapped.action_array - baseline[0].action_array))

    return SmolVLAInterventionMeasurement(
        action_change_norm=action_change_norm,
        action_change_per_dim=action_change_per_dim,
        on_target_norm=on_target_norm,
        off_target_norm=off_target_norm,
        on_target_fraction=on_target_fraction,
        representation_drift=drift,
        first_step_drift=first_step_drift,
        prompt_sensitivity=prompt_sensitivity,
        camera_order_sensitivity=camera_order_sensitivity,
        metadata={
            "measurement": "smolvla_action_expert_intervention",
            "samples": len(samples),
            "intervention": intervention.to_dict(),
            "causal_environment_effect": False,
            "off_target_definition": (
                "component of the action change orthogonal to the direction induced by the expert "
                "direction through action_out_proj"
            ),
        },
    )


def measure_induced_action_direction(
    adapter: SmolVLAPolicyAdapter, direction: np.ndarray, action_dim: int
) -> np.ndarray:
    """Project an expert-space direction into action space via the policy head."""

    policy = adapter.context.policy
    if not isinstance(policy, nn.Module):
        raise TypeError("SmolVLA policy must be a torch.nn.Module to derive the induced direction")
    action_out_proj = getattr(getattr(policy, "model", None), "action_out_proj", None)
    weight = getattr(action_out_proj, "weight", None)
    if not isinstance(weight, Tensor):
        raise TypeError("SmolVLA policy must expose model.action_out_proj.weight")
    matrix = _to_numpy(weight.detach())
    if matrix.ndim != 2 or matrix.shape[1] != adapter.expert_dim:
        raise ValueError(f"action_out_proj weight shape {matrix.shape} does not match expert_dim={adapter.expert_dim}")
    induced = matrix @ direction
    if induced.shape[0] < action_dim:
        raise ValueError("action_out_proj output is smaller than the declared action dimension")
    return induced[:action_dim]


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    detached = getattr(value, "detach", None)
    current = detached() if callable(detached) else value
    cpu = getattr(current, "cpu", None)
    current = cpu() if callable(cpu) else current
    if isinstance(current, torch.Tensor) and current.dtype in (torch.bfloat16, torch.float16):
        current = current.float()
    numpy = getattr(current, "numpy", None)
    current = numpy() if callable(numpy) else current
    return np.array(current, copy=True)


def _expert_reprs(selection: SmolVLAActionSelection) -> list[np.ndarray]:
    return [
        representation.latent.values
        for representation in selection.representations
        if representation.kind == "action_expert"
    ]


def measure_mean_token_delta(before: np.ndarray, after: np.ndarray) -> float:
    if before.shape != after.shape:
        raise ValueError(f"expert capture shapes differ: {before.shape} vs {after.shape}")
    return float(np.mean(np.linalg.norm(after - before, axis=-1)))


def measure_representation_drift(
    baseline: Sequence[SmolVLAActionSelection],
    intervened: Sequence[SmolVLAActionSelection],
) -> float:
    per_step: list[float] = []
    for base, item in zip(baseline, intervened, strict=True):
        base_reprs = _expert_reprs(base)
        item_reprs = _expert_reprs(item)
        if len(base_reprs) != len(item_reprs):
            raise ValueError("baseline and intervened queries produced different denoising capture counts")
        per_step.extend(
            measure_mean_token_delta(base_values, item_values)
            for base_values, item_values in zip(base_reprs, item_reprs, strict=True)
        )
    if not per_step:
        raise ValueError("no action-expert captures were produced for drift measurement")
    return float(np.mean(per_step))


def measure_first_step_drift(baseline: SmolVLAActionSelection, intervened: SmolVLAActionSelection) -> float:
    base_reprs = _expert_reprs(baseline)
    item_reprs = _expert_reprs(intervened)
    if not base_reprs or not item_reprs:
        raise ValueError("no action-expert captures were produced for first-step drift")
    return measure_mean_token_delta(base_reprs[0], item_reprs[0])
