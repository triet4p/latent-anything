"""Private SmolVLA hook-session, capture, and intervention runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from latent_anything.capture import CaptureMetadata
from latent_anything.integrations.lerobot import captured_latent

if TYPE_CHECKING:
    from latent_anything.integrations.lerobot_smolvla import (
        SmolVLAIntervention,
        SmolVLAPolicyMetadata,
        SmolVLARepresentation,
        SmolVLARepresentationKind,
        SmolVLATokenMetadata,
    )


@dataclass(frozen=True)
class SmolVLAQueryResult:
    """Torch-backed query output before public action post-processing."""

    raw_action: object
    representations: tuple[SmolVLARepresentation, ...]
    denoising_steps: int


def smolvla_to_numpy(value: object) -> np.ndarray:
    """Convert a captured tensor-like value to an owned NumPy array."""

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


def smolvla_tensor_output(output: object) -> Tensor:
    """Extract a tensor from a SmolVLA module output."""

    if isinstance(output, Tensor):
        return output
    tensor = getattr(output, "last_hidden_state", None)
    if not isinstance(tensor, Tensor):
        raise TypeError(f"capture location returned {type(output).__name__}, expected Tensor output")
    return tensor


def smolvla_capture_metadata(location: str, tensor: Tensor, call_index: int, version: str) -> CaptureMetadata:
    """Build capture metadata for one official SmolVLA module seam."""

    return CaptureMetadata(
        location=location,
        call_index=call_index,
        shape=tuple(int(size) for size in tensor.shape),
        batch_axis=0 if tensor.ndim >= 1 else None,
        sequence_axis=1 if tensor.ndim >= 2 else None,
        device=str(tensor.device),
        dtype=str(tensor.dtype).removeprefix("torch."),
        source_model_version=f"lerobot-{version}",
    )


class SmolVLAHookSession(AbstractContextManager["SmolVLAHookSession"]):
    """Exception-safe forward-hook session for one SmolVLA action query."""

    def __init__(
        self,
        policy: nn.Module,
        callbacks: Mapping[str, Callable[[nn.Module, tuple[object, ...], object], object]],
    ) -> None:
        modules: dict[str, nn.Module] = dict(policy.named_modules())  # pyright: ignore[reportUnknownArgumentType]
        missing = [location for location in callbacks if location not in modules or location == ""]
        if missing:
            raise KeyError(f"Unknown SmolVLA capture location(s): {missing}")
        self._modules = {location: modules[location] for location in callbacks}
        self._callbacks = dict(callbacks)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> SmolVLAHookSession:
        for location, callback in self._callbacks.items():
            self._handles.append(self._modules[location].register_forward_hook(callback))
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def is_active(self) -> bool:
        """Whether this session currently owns registered hooks."""

        return bool(self._handles)


def smolvla_policy_device(policy: nn.Module) -> torch.device:
    """Return the device of the policy's first parameter."""

    try:
        first = next(policy.parameters())
    except StopIteration:
        raise TypeError("SmolVLA policy must own at least one parameter") from None
    return first.device


def smolvla_noise_to_tensor(value: np.ndarray | None, *, device: torch.device) -> Tensor | None:
    """Convert public NumPy noise to the upstream action-query tensor."""

    return None if value is None else Tensor(value).to(dtype=torch.float32, device=device)


def run_smolvla_query(
    policy: nn.Module,
    prepared: Mapping[str, object],
    select: Callable[..., object],
    *,
    present_cameras: Sequence[str],
    vision_location: str,
    language_location: str,
    state_location: str,
    expert_location: str,
    expert_dim: int,
    checkpoint_repo_id: str,
    checkpoint_revision: str,
    coordinate_identity: str,
    lerobot_version: str,
    metadata: SmolVLAPolicyMetadata,
    episode_step: int,
    intervention: SmolVLAIntervention | None,
    noise: np.ndarray | None,
) -> SmolVLAQueryResult:
    """Run one official SmolVLA query with ordered capture/intervention hooks."""

    from latent_anything.integrations.lerobot_smolvla import (
        SmolVLARepresentation,
        SmolVLATokenMetadata,
    )

    representations: list[SmolVLARepresentation] = []
    prefix_offset = 0
    vision_calls = 0
    language_calls = 0
    expert_calls = 0

    def capture(
        location: str,
        kind: SmolVLARepresentationKind,
        tensor: Tensor,
        token: SmolVLATokenMetadata,
    ) -> None:
        values = smolvla_to_numpy(tensor)
        if values.ndim == 2:
            values = values[None, :]
        if values.ndim != 3 or values.shape[0] != 1:
            raise ValueError(f"SmolVLA {kind} capture must be 2D or 3D with batch size 1; got shape {values.shape}")
        latent = captured_latent(
            values[0],
            provenance={
                "kind": kind,
                "episode_step": episode_step,
                "policy_repo_id": checkpoint_repo_id,
                "policy_revision": checkpoint_revision,
                "coordinate_identity": coordinate_identity,
                "token_metadata": token.to_dict(),
            },
        )
        representations.append(
            SmolVLARepresentation(
                kind=kind,
                latent=latent,
                capture_metadata=smolvla_capture_metadata(location, tensor, len(representations), lerobot_version),
                episode_step=episode_step,
                token=token,
                metadata=metadata,
            )
        )

    def vision_hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
        nonlocal prefix_offset, vision_calls
        del module, inputs
        tensor = smolvla_tensor_output(output)
        if vision_calls >= len(present_cameras):
            raise ValueError(
                f"vision encoder called {vision_calls + 1} times but only {len(present_cameras)} cameras are present"
            )
        camera = present_cameras[vision_calls]
        vision_calls += 1
        token = SmolVLATokenMetadata(
            modality="vision",
            token_count=int(tensor.shape[1]),
            prefix_offset=prefix_offset,
            camera=camera,
        )
        capture(vision_location, "vision_context", tensor, token)
        prefix_offset += int(tensor.shape[1])
        return output

    def language_hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
        nonlocal prefix_offset, language_calls
        del module, inputs
        tensor = smolvla_tensor_output(output)
        if language_calls != 0:
            raise ValueError(f"language embedding called {language_calls + 1} times per query")
        language_calls += 1
        token = SmolVLATokenMetadata(
            modality="language",
            token_count=int(tensor.shape[1]),
            prefix_offset=prefix_offset,
        )
        capture(language_location, "language_context", tensor, token)
        prefix_offset += int(tensor.shape[1])
        return output

    def state_hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
        del module, inputs
        tensor = smolvla_tensor_output(output)
        token = SmolVLATokenMetadata(
            modality="state",
            token_count=1,
            prefix_offset=prefix_offset,
        )
        capture(state_location, "state_context", tensor, token)
        return output

    def expert_hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
        nonlocal expert_calls
        del module, inputs
        tensor = smolvla_tensor_output(output)
        call_index = expert_calls
        expert_calls += 1
        if tensor.shape[-1] != expert_dim:
            raise ValueError(
                "action-expert hidden shape "
                f"{tuple(int(size) for size in tensor.shape)} does not match expert_dim={expert_dim}"
            )
        result = tensor
        if intervention is not None and intervention.strength != 0.0:
            direction = torch.as_tensor(
                np.asarray(intervention.direction).copy(), device=tensor.device, dtype=tensor.dtype
            )
            result = tensor + direction * intervention.strength
        token = SmolVLATokenMetadata(
            modality="action_expert",
            token_count=int(result.shape[1]),
            prefix_offset=None,
            denoising_step=call_index,
        )
        capture(expert_location, "action_expert", result, token)
        return result

    session = SmolVLAHookSession(
        policy,
        {
            vision_location: vision_hook,
            language_location: language_hook,
            state_location: state_hook,
            expert_location: expert_hook,
        },
    )
    with session:
        raw_action = select(prepared, noise=smolvla_noise_to_tensor(noise, device=smolvla_policy_device(policy)))
    return SmolVLAQueryResult(
        raw_action=raw_action,
        representations=tuple(representations),
        denoising_steps=expert_calls,
    )
