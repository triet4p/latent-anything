"""LeRobot SmolVLA capture and bounded intervention on the official pipeline.

The adapter keeps SmolVLA policy construction, image preparation, flow-matching
denoising, action queueing, and normalization upstream-owned. It observes four
module seams that the official ``select_action`` path executes: the SigLIP
vision encoder, the language token embedding table, the state projection, and
the action-expert final norm. Token and modality metadata record where each
capture sits inside the model's prefix/suffix sequences. One additive
intervention on the action-expert representation supports strength control and
bit-exact no-change identity at strength zero.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from importlib import import_module
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from latent_anything.capture import CaptureMetadata
from latent_anything.integrations.lerobot import (
    LeRobotAPI,
    LeRobotCapturedLatent,
    LeRobotPolicyContext,
    captured_latent,
    load_lerobot_api,
)
from latent_anything.latent_space import LatentSpace

SMOLVLA_POLICY_REPO_ID = "lerobot/smolvla_libero"
SMOLVLA_POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
SMOLVLA_DATASET_REPO_ID = "lerobot/libero"
SMOLVLA_DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
SMOLVLA_LEROBOT_VERSION = "0.6.1"
SMOLVLA_ENV_TYPE = "libero"
SMOLVLA_ENV_TASK = "libero_spatial"
SMOLVLA_VLM_BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOLVLA_VISION_LOCATION = "model.vlm_with_expert.vlm.model.vision_model"
SMOLVLA_LANGUAGE_LOCATION = "model.vlm_with_expert.vlm.model.text_model.embed_tokens"
SMOLVLA_STATE_LOCATION = "model.state_proj"
SMOLVLA_EXPERT_LOCATION = "model.vlm_with_expert.lm_expert.norm"
SMOLVLA_MAX_INTERVENTION_STRENGTH = 100.0
SMOLVLA_SUPPORTED_LEROBOT_VERSION = "0.6.x"
# The model card's documented dataset-to-policy camera mapping. The official
# preprocessor applies the same rename, and make_policy's rename_map records
# that the dataset feature names intentionally differ from the policy config.
SMOLVLA_RENAME_MAP: dict[str, str] = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


@dataclass(frozen=True)
class SmolVLAHardwareProfile:
    """Reproducible hardware requirements for the pinned SmolVLA pair."""

    vlm_backbone: str = SMOLVLA_VLM_BACKBONE
    num_vlm_layers: int = 16
    num_expert_layers: int = 16
    expert_width_multiplier: float = 0.75
    dtype: str = "bfloat16"
    recommended_min_gpu_memory_gb: int = 16
    cpu_feasible: bool = True
    notes: tuple[str, ...] = (
        "SmolVLM2-500M-Video-Instruct backbone with SigLIP vision encoder; "
        "the 16-layer action expert runs at 75% of the VLM hidden width.",
        "CPU inference is feasible for single queries but not realtime; the "
        "marked checkpoint intervention lane requires a CUDA device.",
        "Checkpoint and dataset revisions are immutable pins; the model must "
        "run in the same bfloat16 coordinate system it was trained in.",
    )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible hardware profile."""
        return asdict(self)


SMOLVLA_HARDWARE_PROFILE = SmolVLAHardwareProfile()


@dataclass(frozen=True)
class SmolVLACheckpointSpec:
    """Immutable public identity for the SmolVLA checkpoint and paired dataset.

    The pair is the model card's own documented training configuration:
    ``lerobot/smolvla_libero`` trained on ``lerobot/libero`` for the
    ``libero_spatial`` LIBERO-10 benchmark.
    """

    policy_repo_id: str = SMOLVLA_POLICY_REPO_ID
    policy_revision: str = SMOLVLA_POLICY_REVISION
    dataset_repo_id: str = SMOLVLA_DATASET_REPO_ID
    dataset_revision: str = SMOLVLA_DATASET_REVISION
    lerobot_version: str = SMOLVLA_LEROBOT_VERSION
    environment_type: str = SMOLVLA_ENV_TYPE
    environment_task: str = SMOLVLA_ENV_TASK

    def __post_init__(self) -> None:
        for name in (
            "policy_repo_id",
            "policy_revision",
            "dataset_repo_id",
            "dataset_revision",
            "environment_type",
            "environment_task",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible checkpoint identity."""
        return cast(dict[str, str], asdict(self))


DEFAULT_SMOLVLA_CHECKPOINT = SmolVLACheckpointSpec()


@dataclass(frozen=True)
class SmolVLAPolicyMetadata:
    """Capture-point, dimension, and coordinate identity for one adapter."""

    checkpoint: SmolVLACheckpointSpec
    vision_location: str
    language_location: str
    state_location: str
    expert_location: str
    context_dim: int
    expert_dim: int
    action_dim: int
    max_action_dim: int
    chunk_size: int
    num_steps: int
    coordinate_identity: str
    hardware_profile: SmolVLAHardwareProfile = SMOLVLA_HARDWARE_PROFILE

    def to_dict(self) -> dict[str, object]:
        """Return serializable metadata without upstream objects."""
        result = asdict(self)
        result["checkpoint"] = self.checkpoint.to_dict()
        result["hardware_profile"] = self.hardware_profile.to_dict()
        return result


SmolVLAModality = Literal["vision", "language", "state", "action_expert"]
SmolVLARepresentationKind = Literal["vision_context", "language_context", "state_context", "action_expert"]


@dataclass(frozen=True)
class SmolVLATokenMetadata:
    """Token and modality position of one capture inside the model input."""

    modality: SmolVLAModality
    token_count: int
    prefix_offset: int | None
    camera: str | None = None
    denoising_step: int | None = None

    def __post_init__(self) -> None:
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        if self.modality == "vision" and not self.camera:
            raise ValueError("vision captures must name their camera")
        if self.modality != "action_expert" and self.prefix_offset is None:
            raise ValueError(f"{self.modality} captures must declare their prefix offset")
        if self.modality == "action_expert" and self.prefix_offset is not None:
            raise ValueError("action-expert captures live in the suffix, not the prefix")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible token descriptor."""
        return asdict(self)


@dataclass(frozen=True)
class SmolVLARepresentation:
    """One modality capture with latent, capture, and token metadata."""

    kind: SmolVLARepresentationKind
    latent: LeRobotCapturedLatent
    capture_metadata: CaptureMetadata
    episode_step: int
    token: SmolVLATokenMetadata
    metadata: SmolVLAPolicyMetadata

    def to_dict(self) -> dict[str, object]:
        """Return metadata and shape information without serializing arrays."""
        return {
            "kind": self.kind,
            "capture_metadata": asdict(self.capture_metadata),
            "latent_shape": list(self.latent.values.shape),
            "latent_dtype": str(self.latent.values.dtype),
            "episode_step": self.episode_step,
            "token": self.token.to_dict(),
            "metadata": self.metadata.to_dict(),
            "provenance": dict(self.latent.provenance),
        }


@dataclass(frozen=True)
class SmolVLAActionSelection:
    """Official post-processed action plus all captures from one query."""

    action: object
    action_array: np.ndarray
    representations: tuple[SmolVLARepresentation, ...]
    denoising_steps: int

    def __post_init__(self) -> None:
        values = np.array(self.action_array, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "action_array", values)

    def of_kind(self, kind: SmolVLARepresentationKind) -> tuple[SmolVLARepresentation, ...]:
        """Return representations of one kind in capture order."""
        return tuple(representation for representation in self.representations if representation.kind == kind)


@dataclass(frozen=True)
class SmolVLAIntervention:
    """One additive, bounded intervention on the action-expert representation.

    The direction is added to every action-expert token at each denoising
    step, scaled by ``strength``. Strength zero returns the unchanged output,
    so baseline and intervened actions are bit-identical.
    """

    direction: np.ndarray
    strength: float
    location: str = SMOLVLA_EXPERT_LOCATION
    max_strength: float = SMOLVLA_MAX_INTERVENTION_STRENGTH

    def __post_init__(self) -> None:
        direction = np.array(self.direction, copy=True)
        if direction.ndim != 1 or direction.size < 1:
            raise ValueError("intervention direction must be a non-empty 1D vector")
        if not np.all(np.isfinite(direction)):
            raise ValueError("intervention direction must contain only finite values")
        direction.setflags(write=False)
        object.__setattr__(self, "direction", direction)
        if not np.isfinite(self.strength):
            raise ValueError("intervention strength must be finite")
        if abs(self.strength) > self.max_strength:
            raise ValueError(f"intervention strength {self.strength} exceeds the bounded maximum {self.max_strength}")
        if not self.location:
            raise ValueError("intervention location must not be empty")

    @property
    def applied_norm(self) -> float:
        """Return the total perturbation norm ``|strength| * ||direction||``."""
        return float(abs(self.strength) * np.linalg.norm(self.direction))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable intervention descriptor without the array."""
        return {
            "location": self.location,
            "strength": self.strength,
            "max_strength": self.max_strength,
            "direction_norm": float(np.linalg.norm(self.direction)),
            "applied_norm": self.applied_norm,
            "direction_shape": list(self.direction.shape),
        }


@dataclass(frozen=True)
class SmolVLAInterventionMeasurement:
    """Quantitative effect of one intervention and prompt/camera sensitivity."""

    action_change_norm: float
    action_change_per_dim: np.ndarray
    on_target_norm: float
    off_target_norm: float
    on_target_fraction: float
    representation_drift: float
    first_step_drift: float
    prompt_sensitivity: float
    camera_order_sensitivity: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.array(self.action_change_per_dim, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "action_change_per_dim", values)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible measurement summary."""
        return {
            "action_change_norm": self.action_change_norm,
            "action_change_per_dim": self.action_change_per_dim.tolist(),
            "on_target_norm": self.on_target_norm,
            "off_target_norm": self.off_target_norm,
            "on_target_fraction": self.on_target_fraction,
            "representation_drift": self.representation_drift,
            "first_step_drift": self.first_step_drift,
            "prompt_sensitivity": self.prompt_sensitivity,
            "camera_order_sensitivity": self.camera_order_sensitivity,
            "metadata": dict(self.metadata),
        }


def _to_numpy(value: object) -> np.ndarray:
    """Convert a tensor-like or array-like value into an owned NumPy array.

    The SmolVLA backbone runs in bfloat16, which NumPy cannot represent;
    bfloat16/float16 tensors are upcast to float32 losslessly.
    """

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


def _action_to_numpy(value: object) -> np.ndarray:
    """Extract the canonical action value from a LeRobot post-processor result."""

    if isinstance(value, Mapping):
        for key in ("action", "actions"):
            if key in value:
                return _to_numpy(value[key])
        raise TypeError("LeRobot post-processor mapping must contain 'action' or 'actions'")
    return _to_numpy(value)


def _tensor_output(output: object) -> Tensor:
    """Extract the tensor from a module output or an output object."""

    if isinstance(output, Tensor):
        return output
    tensor = getattr(output, "last_hidden_state", None)
    if not isinstance(tensor, Tensor):
        raise TypeError(f"capture location returned {type(output).__name__}, expected Tensor output")
    return tensor


def _capture_metadata(location: str, tensor: Tensor, call_index: int, version: str) -> CaptureMetadata:
    """Build shared capture metadata for a tensor observed through a hook."""

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


class _SmolVLAHookSession(AbstractContextManager["_SmolVLAHookSession"]):
    """Exception-safe forward-hook session for one action query.

    Hooks are registered on entry and always removed on exit, even when the
    policy forward raises, so no intervention or capture can leak into a later
    query.
    """

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

    def __enter__(self) -> _SmolVLAHookSession:
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


class SmolVLAPolicyAdapter:
    """Capture SmolVLA context and action-expert representations.

    Vision, language, state, and action-expert captures record token/modality
    metadata. The adapter never reimplements image preparation, flow-matching
    denoising, normalization, or action queueing; it observes the official
    ``preprocess -> select_action -> postprocess`` path.
    """

    def __init__(
        self,
        context: LeRobotPolicyContext,
        *,
        checkpoint: SmolVLACheckpointSpec = DEFAULT_SMOLVLA_CHECKPOINT,
        vision_location: str = SMOLVLA_VISION_LOCATION,
        language_location: str = SMOLVLA_LANGUAGE_LOCATION,
        state_location: str = SMOLVLA_STATE_LOCATION,
        expert_location: str = SMOLVLA_EXPERT_LOCATION,
    ) -> None:
        for name, location in (
            ("vision_location", vision_location),
            ("language_location", language_location),
            ("state_location", state_location),
            ("expert_location", expert_location),
        ):
            if not location:
                raise ValueError(f"{name} must not be empty")
        policy = context.policy
        if not isinstance(policy, nn.Module):
            raise TypeError("SmolVLA context must contain a torch.nn.Module policy")
        vla_model = getattr(policy, "model", None)
        vlm_with_expert = getattr(vla_model, "vlm_with_expert", None)
        state_proj = getattr(vla_model, "state_proj", None)
        action_out_proj = getattr(vla_model, "action_out_proj", None)
        if not isinstance(vla_model, nn.Module) or not isinstance(vlm_with_expert, nn.Module):
            raise TypeError("SmolVLA policy must expose model.vlm_with_expert as a torch.nn.Module")
        if not isinstance(state_proj, nn.Module) or not isinstance(action_out_proj, nn.Module):
            raise TypeError("SmolVLA policy must expose model.state_proj and model.action_out_proj")
        config = getattr(policy, "config", None)
        action_shape = getattr(getattr(config, "action_feature", None), "shape", ())
        action_dim = int(action_shape[0]) if action_shape else 0
        if action_dim < 1:
            raise ValueError("SmolVLA policy must expose a positive action dimension")
        max_action_dim = int(getattr(config, "max_action_dim", 0))
        if max_action_dim < 1:
            raise ValueError("SmolVLA policy must expose a positive max_action_dim")
        chunk_size = int(getattr(config, "chunk_size", 0))
        if chunk_size < 1:
            raise ValueError("SmolVLA policy must expose a positive chunk_size")
        num_steps = int(getattr(config, "num_steps", 0))
        if num_steps < 1:
            raise ValueError("SmolVLA policy must expose a positive num_steps")
        expert_dim = int(getattr(vlm_with_expert, "expert_hidden_size", 0))
        if expert_dim < 1:
            expert_dim = int(getattr(action_out_proj, "in_features", 0))
        if expert_dim < 1:
            raise ValueError("SmolVLA policy must expose an action-expert hidden dimension")
        context_dim = int(getattr(state_proj, "out_features", 0))
        if context_dim < 1:
            vlm_config = getattr(getattr(vlm_with_expert, "config", None), "text_config", None)
            context_dim = int(getattr(vlm_config, "hidden_size", 0))
        if context_dim < 1:
            raise ValueError("SmolVLA policy must expose a VLM context hidden dimension")

        image_features = dict(getattr(config, "image_features", {}) or {})
        self._image_features = tuple(str(key) for key in image_features)
        self.context = context
        self.checkpoint = checkpoint
        self.vision_location = vision_location
        self.language_location = language_location
        self.state_location = state_location
        self.expert_location = expert_location
        self._context_dim = context_dim
        self._expert_dim = expert_dim
        self._action_dim = action_dim
        self._max_action_dim = max_action_dim
        self._chunk_size = chunk_size
        self._num_steps = num_steps
        self.metadata = SmolVLAPolicyMetadata(
            checkpoint=checkpoint,
            vision_location=vision_location,
            language_location=language_location,
            state_location=state_location,
            expert_location=expert_location,
            context_dim=context_dim,
            expert_dim=expert_dim,
            action_dim=action_dim,
            max_action_dim=max_action_dim,
            chunk_size=chunk_size,
            num_steps=num_steps,
            coordinate_identity=(
                f"smolvla:{checkpoint.policy_repo_id}@{checkpoint.policy_revision}:"
                f"context+expert:{context_dim}+{expert_dim}"
            ),
        )

    @property
    def context_space(self) -> LatentSpace:
        """Return the Euclidean space of VLM context hidden states."""

        return LatentSpace(
            dim=self._context_dim,
            source_model=self.checkpoint.policy_repo_id,
            metadata={
                "representation_role": "vision_language_state_context",
                "policy_revision": self.checkpoint.policy_revision,
                "coordinate_identity": self.metadata.coordinate_identity,
            },
        )

    @property
    def expert_space(self) -> LatentSpace:
        """Return the Euclidean space of action-expert hidden states."""

        return LatentSpace(
            dim=self._expert_dim,
            source_model=self.checkpoint.policy_repo_id,
            metadata={
                "representation_role": "action_expert",
                "policy_revision": self.checkpoint.policy_revision,
                "coordinate_identity": self.metadata.coordinate_identity,
            },
        )

    @property
    def action_dim(self) -> int:
        """Return the unpadded action dimension of the pinned policy."""

        return self._action_dim

    @property
    def expert_dim(self) -> int:
        """Return the action-expert hidden dimension of the pinned policy."""

        return self._expert_dim

    @property
    def chunk_size(self) -> int:
        """Return the action-chunk size of the pinned policy."""

        return self._chunk_size

    @property
    def device(self) -> str:
        """Return the device the pinned policy lives on."""

        policy = self.context.policy
        if not isinstance(policy, nn.Module):
            raise TypeError("SmolVLA policy must be a torch.nn.Module for device resolution")
        return str(_policy_device(policy))

    @property
    def num_steps(self) -> int:
        """Return the denoising step count of the pinned policy."""

        return self._num_steps

    def reset(self) -> None:
        """Reset LeRobot's action queue at an episode boundary."""

        reset = getattr(self.context.policy, "reset", None)
        if not callable(reset):
            raise TypeError("SmolVLA policy must expose LeRobot's reset() action-selection lifecycle")
        reset()

    def select_action(
        self,
        sample: Mapping[str, object],
        *,
        noise: np.ndarray | None = None,
        intervention: SmolVLAIntervention | None = None,
        episode_step: int = 0,
    ) -> SmolVLAActionSelection:
        """Preprocess, select, capture, and postprocess one observation.

        Hooks observe the official action-selection path for the lifetime of
        the call only. A queue hit executes no model query and produces no
        captures.
        """

        preprocess = self.context.preprocessor
        select = getattr(self.context.policy, "select_action", None)
        postprocess = self.context.postprocessor
        if not callable(preprocess) or not callable(select) or not callable(postprocess):
            raise TypeError(
                "SmolVLA context must expose callable preprocessor, policy.select_action, and postprocessor"
            )
        policy = self.context.policy
        if not isinstance(policy, nn.Module):
            raise TypeError("SmolVLA policy must be a torch.nn.Module for capture")
        prepared = cast(Mapping[str, object], preprocess(sample))
        present_cameras = [key for key in self._image_features if key in prepared]
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
            values = _to_numpy(tensor)
            if values.ndim == 2:
                values = values[None, :]
            if values.ndim != 3 or values.shape[0] != 1:
                raise ValueError(f"SmolVLA {kind} capture must be 2D or 3D with batch size 1; got shape {values.shape}")
            latent = captured_latent(
                values[0],
                provenance={
                    "kind": kind,
                    "episode_step": episode_step,
                    "policy_repo_id": self.checkpoint.policy_repo_id,
                    "policy_revision": self.checkpoint.policy_revision,
                    "coordinate_identity": self.metadata.coordinate_identity,
                    "token_metadata": token.to_dict(),
                },
            )
            representation = SmolVLARepresentation(
                kind=kind,
                latent=latent,
                capture_metadata=_capture_metadata(
                    location, tensor, len(representations), self.checkpoint.lerobot_version
                ),
                episode_step=episode_step,
                token=token,
                metadata=self.metadata,
            )
            representations.append(representation)

        def vision_hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
            nonlocal prefix_offset, vision_calls
            del module, inputs
            tensor = _tensor_output(output)
            if vision_calls >= len(present_cameras):
                raise ValueError(
                    f"vision encoder called {vision_calls + 1} times but only "
                    f"{len(present_cameras)} cameras are present"
                )
            camera = present_cameras[vision_calls]
            vision_calls += 1
            token = SmolVLATokenMetadata(
                modality="vision",
                token_count=int(tensor.shape[1]),
                prefix_offset=prefix_offset,
                camera=camera,
            )
            capture(self.vision_location, "vision_context", tensor, token)
            prefix_offset += int(tensor.shape[1])
            return output

        def language_hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
            nonlocal prefix_offset, language_calls
            del module, inputs
            tensor = _tensor_output(output)
            if language_calls != 0:
                raise ValueError(f"language embedding called {language_calls + 1} times per query")
            language_calls += 1
            token = SmolVLATokenMetadata(
                modality="language",
                token_count=int(tensor.shape[1]),
                prefix_offset=prefix_offset,
            )
            capture(self.language_location, "language_context", tensor, token)
            prefix_offset += int(tensor.shape[1])
            return output

        def state_hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
            del module, inputs
            tensor = _tensor_output(output)
            token = SmolVLATokenMetadata(
                modality="state",
                token_count=1,
                prefix_offset=prefix_offset,
            )
            capture(self.state_location, "state_context", tensor, token)
            return output

        def expert_hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> object:
            nonlocal expert_calls
            del module, inputs
            tensor = _tensor_output(output)
            call_index = expert_calls
            expert_calls += 1
            if tensor.shape[-1] != self._expert_dim:
                raise ValueError(
                    f"action-expert hidden shape {tuple(int(s) for s in tensor.shape)} does not match "
                    f"expert_dim={self._expert_dim}"
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
            capture(self.expert_location, "action_expert", result, token)
            return result

        session = _SmolVLAHookSession(
            policy,
            {
                self.vision_location: vision_hook,
                self.language_location: language_hook,
                self.state_location: state_hook,
                self.expert_location: expert_hook,
            },
        )
        with session:
            raw_action = select(prepared, noise=_noise_to_tensor(noise, device=_policy_device(policy)))
        action = postprocess(raw_action)
        return SmolVLAActionSelection(
            action=action,
            action_array=_action_to_numpy(action),
            representations=tuple(representations),
            denoising_steps=expert_calls,
        )


def _policy_device(policy: nn.Module) -> torch.device:
    """Return the device of the policy's first parameter."""

    try:
        first = next(policy.parameters())
    except StopIteration:
        raise TypeError("SmolVLA policy must own at least one parameter") from None
    return first.device


def _noise_to_tensor(value: np.ndarray | None, *, device: torch.device) -> Tensor | None:
    """Convert the public NumPy noise boundary to the upstream tensor type.

    LeRobot's default action-chunk noise is float32 on the policy device; the
    action expert is created in float32, so a fixed seed noise must match both
    the dtype and the device of the model.
    """

    return None if value is None else Tensor(value).to(dtype=torch.float32, device=device)


def load_smolvla_policy(
    checkpoint: SmolVLACheckpointSpec = DEFAULT_SMOLVLA_CHECKPOINT,
    *,
    api: LeRobotAPI | None = None,
    dataset_meta: object | None = None,
    device: str = "cpu",
) -> SmolVLAPolicyAdapter:
    """Load the pinned SmolVLA policy and official LeRobot processors.

    The factory path is intentionally upstream-owned: ``SmolVLAConfig`` and
    ``LeRobotDatasetMetadata`` come from LeRobot, while policy and processor
    construction go through ``LeRobotAPI.make_policy`` and
    ``make_pre_post_processors``. The device step of the loaded processor
    pipelines is overridden to the requested device.
    """

    upstream_api = api if api is not None else load_lerobot_api()
    smolvla_config_module = import_module("lerobot.policies.smolvla.configuration_smolvla")
    config_type = getattr(smolvla_config_module, "SmolVLAConfig")  # noqa: B009 - optional upstream symbol
    config_loader = getattr(config_type, "from_pretrained")  # noqa: B009 - optional upstream symbol
    config = config_loader(checkpoint.policy_repo_id, revision=checkpoint.policy_revision)
    config.pretrained_path = checkpoint.policy_repo_id
    config.pretrained_revision = checkpoint.policy_revision
    config.device = device

    resolved_meta = dataset_meta
    if resolved_meta is None:
        dataset_module = import_module("lerobot.datasets")
        metadata_type = getattr(dataset_module, "LeRobotDatasetMetadata")  # noqa: B009 - optional upstream symbol
        resolved_meta = metadata_type(checkpoint.dataset_repo_id, revision=checkpoint.dataset_revision)

    policy = upstream_api.make_policy(
        config,
        ds_meta=resolved_meta,
        rename_map=SMOLVLA_RENAME_MAP,
    )
    processors = upstream_api.make_pre_post_processors(
        config,
        pretrained_path=checkpoint.policy_repo_id,
        pretrained_revision=checkpoint.policy_revision,
        dataset_stats=getattr(resolved_meta, "stats", None),
        dataset_meta=resolved_meta,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    preprocessor, postprocessor = cast(tuple[object, object], processors)
    context = LeRobotPolicyContext(
        policy_name="smolvla",
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=resolved_meta,
        metadata={
            "policy_repo_id": checkpoint.policy_repo_id,
            "policy_revision": checkpoint.policy_revision,
            "dataset_repo_id": checkpoint.dataset_repo_id,
            "dataset_revision": checkpoint.dataset_revision,
            "environment_type": checkpoint.environment_type,
            "environment_task": checkpoint.environment_task,
        },
    )
    return SmolVLAPolicyAdapter(context, checkpoint=checkpoint)


def measure_smolvla_intervention(
    adapter: SmolVLAPolicyAdapter,
    samples: Sequence[Mapping[str, object]],
    *,
    noise: np.ndarray,
    intervention: SmolVLAIntervention,
    alternate_prompt_sample: Mapping[str, object] | None = None,
    camera_swapped_sample: Mapping[str, object] | None = None,
) -> SmolVLAInterventionMeasurement:
    """Measure one bounded intervention and prompt/camera-order sensitivity.

    ``samples`` must be spaced at least ``chunk_size`` apart so every query
    executes a fresh model pass. The on/off-target decomposition projects each
    action change onto the action-space direction induced by the expert
    direction through the policy's own ``action_out_proj`` mapping.
    """

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

    induced = _induced_action_direction(adapter, intervention.direction, adapter.action_dim)
    unit = induced / np.linalg.norm(induced)
    projections = changes @ unit
    on_target_norm = float(np.mean(np.abs(projections)))
    residuals = changes - projections[:, None] * unit[None, :]
    off_target_norm = float(np.mean(np.linalg.norm(residuals, axis=1)))
    total = on_target_norm + off_target_norm
    on_target_fraction = on_target_norm / total if total > 0.0 else 0.0

    drift = _representation_drift(baseline, intervened)
    first_step_drift = _first_step_drift(baseline[0], intervened[0])

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


def _induced_action_direction(adapter: SmolVLAPolicyAdapter, direction: np.ndarray, action_dim: int) -> np.ndarray:
    """Project an expert-space direction into action space via the policy's own head."""

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


def _expert_reprs(
    selection: SmolVLAActionSelection,
) -> list[np.ndarray]:
    """Return action-expert captures in denoising order for one selection."""

    return [
        representation.latent.values
        for representation in selection.representations
        if representation.kind == "action_expert"
    ]


def _mean_token_delta(before: np.ndarray, after: np.ndarray) -> float:
    """Mean per-token Euclidean displacement between two expert captures."""

    if before.shape != after.shape:
        raise ValueError(f"expert capture shapes differ: {before.shape} vs {after.shape}")
    deltas = after - before
    return float(np.mean(np.linalg.norm(deltas, axis=-1)))


def _representation_drift(
    baseline: Sequence[SmolVLAActionSelection],
    intervened: Sequence[SmolVLAActionSelection],
) -> float:
    """Mean per-step per-token expert displacement across executed queries."""

    per_step: list[float] = []
    for base, item in zip(baseline, intervened, strict=True):
        base_reprs = _expert_reprs(base)
        item_reprs = _expert_reprs(item)
        if len(base_reprs) != len(item_reprs):
            raise ValueError("baseline and intervened queries produced different denoising capture counts")
        per_step.extend(
            _mean_token_delta(base_values, item_values)
            for base_values, item_values in zip(base_reprs, item_reprs, strict=True)
        )
    if not per_step:
        raise ValueError("no action-expert captures were produced for drift measurement")
    return float(np.mean(per_step))


def _first_step_drift(
    baseline: SmolVLAActionSelection,
    intervened: SmolVLAActionSelection,
) -> float:
    """Expert per-token displacement at the first denoising step of the first query."""

    base_reprs = _expert_reprs(baseline)
    item_reprs = _expert_reprs(intervened)
    if not base_reprs or not item_reprs:
        raise ValueError("no action-expert captures were produced for first-step drift")
    return _mean_token_delta(base_reprs[0], item_reprs[0])


__all__ = [
    "DEFAULT_SMOLVLA_CHECKPOINT",
    "SMOLVLA_DATASET_REPO_ID",
    "SMOLVLA_DATASET_REVISION",
    "SMOLVLA_ENV_TASK",
    "SMOLVLA_ENV_TYPE",
    "SMOLVLA_EXPERT_LOCATION",
    "SMOLVLA_HARDWARE_PROFILE",
    "SMOLVLA_LANGUAGE_LOCATION",
    "SMOLVLA_LEROBOT_VERSION",
    "SMOLVLA_MAX_INTERVENTION_STRENGTH",
    "SMOLVLA_POLICY_REPO_ID",
    "SMOLVLA_POLICY_REVISION",
    "SMOLVLA_RENAME_MAP",
    "SMOLVLA_STATE_LOCATION",
    "SMOLVLA_SUPPORTED_LEROBOT_VERSION",
    "SMOLVLA_VISION_LOCATION",
    "SmolVLAActionSelection",
    "SmolVLACheckpointSpec",
    "SmolVLAHardwareProfile",
    "SmolVLAIntervention",
    "SmolVLAInterventionMeasurement",
    "SmolVLAPolicyAdapter",
    "SmolVLAPolicyMetadata",
    "SmolVLARepresentation",
    "SmolVLATokenMetadata",
    "load_smolvla_policy",
    "measure_smolvla_intervention",
]
