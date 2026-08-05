"""LeRobot Diffusion Policy capture and observational analysis.

The adapter keeps policy construction, normalization, queueing, and action
semantics upstream-owned. It observes the global conditioning tensor entering
the denoiser and the denoiser output at each diffusion timestep, preserving
episode time and diffusion time as separate axes in the trace types.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import import_module
from typing import Literal, cast

import numpy as np
from torch import Tensor, nn

from latent_anything.capture import CaptureMetadata
from latent_anything.density import GaussianMixtureDensity, GMMConfig
from latent_anything.integrations.lerobot import (
    LeRobotAPI,
    LeRobotCapturedLatent,
    LeRobotPolicyContext,
    captured_latent,
    load_lerobot_api,
)
from latent_anything.latent_space import LatentSpace
from latent_anything.methods.pca import PCA
from latent_anything.probes import ControlBaselines, LinearProbe, LinearProbeConfig, LinearProbeResult, compute_controls
from latent_anything.trajectory import Trajectory

DIFFUSION_POLICY_REPO_ID = "LeTau/diffusion_aloha_insertion"
DIFFUSION_POLICY_REVISION = "6126e33"
DIFFUSION_DATASET_REPO_ID = "lerobot/aloha_sim_insertion_human_image"
DIFFUSION_DATASET_REVISION = "d93d36a"
DIFFUSION_LEROBOT_VERSION = "0.6.x"
DIFFUSION_ENV_TYPE = "aloha"
DIFFUSION_ENV_TASK = "AlohaInsertion-v0"
DIFFUSION_CONDITIONING_LOCATION = "diffusion.unet.global_cond"
DIFFUSION_DENOISING_LOCATION = "diffusion.unet"


@dataclass(frozen=True)
class DiffusionCheckpointSpec:
    """Immutable public identity for the Diffusion checkpoint and dataset."""

    policy_repo_id: str = DIFFUSION_POLICY_REPO_ID
    policy_revision: str = DIFFUSION_POLICY_REVISION
    dataset_repo_id: str = DIFFUSION_DATASET_REPO_ID
    dataset_revision: str = DIFFUSION_DATASET_REVISION
    lerobot_version: str = DIFFUSION_LEROBOT_VERSION
    environment_type: str = DIFFUSION_ENV_TYPE
    environment_task: str = DIFFUSION_ENV_TASK

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


DEFAULT_DIFFUSION_CHECKPOINT = DiffusionCheckpointSpec()


@dataclass(frozen=True)
class DiffusionPolicyMetadata:
    """Capture locations and coordinate identity for one adapter."""

    checkpoint: DiffusionCheckpointSpec
    conditioning_location: str
    denoising_location: str
    conditioning_dim: int
    denoising_dim: int
    coordinate_identity: str

    def to_dict(self) -> dict[str, object]:
        """Return serializable metadata without upstream objects."""

        result = asdict(self)
        result["checkpoint"] = self.checkpoint.to_dict()
        return result


DiffusionRepresentationKind = Literal["observation_conditioning", "denoising_action"]


@dataclass(frozen=True)
class DiffusionRepresentation:
    """One flattened representation with explicit diffusion-axis metadata."""

    kind: DiffusionRepresentationKind
    latent: LeRobotCapturedLatent
    capture_metadata: CaptureMetadata
    episode_step: int
    denoising_step: int | None
    diffusion_timestep: int | None
    metadata: DiffusionPolicyMetadata

    def to_dict(self) -> dict[str, object]:
        """Return metadata and shape information without serializing arrays."""

        return {
            "kind": self.kind,
            "capture_metadata": asdict(self.capture_metadata),
            "latent_shape": list(self.latent.values.shape),
            "latent_dtype": str(self.latent.values.dtype),
            "episode_step": self.episode_step,
            "denoising_step": self.denoising_step,
            "diffusion_timestep": self.diffusion_timestep,
            "metadata": self.metadata.to_dict(),
            "provenance": dict(self.latent.provenance),
        }


@dataclass(frozen=True)
class DiffusionActionSelection:
    """Official post-processed action plus all captures from one query."""

    action: object
    action_array: np.ndarray
    representations: tuple[DiffusionRepresentation, ...]

    def __post_init__(self) -> None:
        values = np.array(self.action_array, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "action_array", values)


@dataclass(frozen=True)
class DiffusionEpisodeTrace:
    """Captured policy observations for one labeled episode."""

    episode_id: str
    outcome: Literal["success", "failure"]
    selections: tuple[DiffusionActionSelection, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.selections:
            raise ValueError("a Diffusion episode trace must contain at least one captured selection")

    @property
    def conditioning_trajectory(self) -> Trajectory:
        """Return conditioning representations in environment/episode order."""

        values = np.stack(
            [_representation(selection, "observation_conditioning").latent.values for selection in self.selections]
        )
        return Trajectory(values, metadata={"episode_id": self.episode_id, "outcome": self.outcome, **self.metadata})

    def denoising_by_timestep(self, timestep: int) -> np.ndarray:
        """Return denoising representations for one diffusion timestep."""

        values = [
            representation.latent.values
            for selection in self.selections
            for representation in selection.representations
            if representation.kind == "denoising_action" and representation.diffusion_timestep == timestep
        ]
        if not values:
            raise ValueError(f"no denoising captures for timestep {timestep}")
        return np.stack(values)


@dataclass(frozen=True)
class DiffusionAnalysisResult:
    """Projection, probing, density, controls, and two-axis trajectory metrics."""

    projected_conditioning: np.ndarray
    projection_explained_variance: np.ndarray
    conditioning_probe: LinearProbeResult
    conditioning_controls: ControlBaselines
    conditioning_density_auroc: float
    episode_time_lengths: Mapping[str, float]
    timestep_time_lengths: Mapping[str, Mapping[str, float]]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible observational analysis summary."""

        return {
            "projected_conditioning_shape": list(self.projected_conditioning.shape),
            "projection_explained_variance": self.projection_explained_variance.tolist(),
            "conditioning_probe": self.conditioning_probe.to_dict(),
            "conditioning_controls": asdict(self.conditioning_controls),
            "conditioning_density_auroc": self.conditioning_density_auroc,
            "episode_time_lengths": dict(self.episode_time_lengths),
            "timestep_time_lengths": {key: dict(value) for key, value in self.timestep_time_lengths.items()},
            "metadata": dict(self.metadata),
        }


def _to_numpy(value: object) -> np.ndarray:
    """Convert a tensor-like value into an owned NumPy array."""

    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    detached = getattr(value, "detach", None)
    current = detached() if callable(detached) else value
    cpu = getattr(current, "cpu", None)
    current = cpu() if callable(cpu) else current
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


def _representation(selection: DiffusionActionSelection, kind: DiffusionRepresentationKind) -> DiffusionRepresentation:
    """Return the unique representation of a requested kind from a selection."""

    values = [representation for representation in selection.representations if representation.kind == kind]
    if len(values) != 1:
        raise ValueError(f"expected one {kind} representation, got {len(values)}")
    return values[0]


class DiffusionPolicyAdapter:
    """Observe LeRobot Diffusion conditioning and denoising representations."""

    def __init__(
        self,
        context: LeRobotPolicyContext,
        *,
        checkpoint: DiffusionCheckpointSpec = DEFAULT_DIFFUSION_CHECKPOINT,
        conditioning_location: str = DIFFUSION_CONDITIONING_LOCATION,
        denoising_location: str = DIFFUSION_DENOISING_LOCATION,
    ) -> None:
        if not conditioning_location or not denoising_location:
            raise ValueError("capture locations must not be empty")
        policy = context.policy
        diffusion = getattr(policy, "diffusion", None)
        unet = getattr(diffusion, "unet", None)
        if not isinstance(policy, nn.Module) or not isinstance(unet, nn.Module):
            raise TypeError("Diffusion context must contain a torch.nn.Module policy with policy.diffusion.unet")
        config = getattr(policy, "config", None)
        action_dim = getattr(getattr(config, "action_feature", None), "shape", (None,))[0]
        if not isinstance(action_dim, int) or action_dim < 1:
            action_dim = int(getattr(getattr(diffusion, "config", None), "action_dim", 0))
        if action_dim < 1:
            raise ValueError("Diffusion policy must expose a positive action dimension")
        conditioning_dim = int(getattr(diffusion, "conditioning_dim", 0))
        if conditioning_dim < 1:
            conditioning_dim = int(getattr(getattr(diffusion, "config", None), "conditioning_dim", 0))
        if conditioning_dim < 1:
            conditioning_dim = int(getattr(config, "conditioning_dim", 0))
        if conditioning_dim < 1:
            state_shape = getattr(getattr(config, "robot_state_feature", None), "shape", ())
            env_shape = getattr(getattr(config, "env_state_feature", None), "shape", ())
            state_dim = int(state_shape[0]) if state_shape else 0
            env_dim = int(env_shape[0]) if env_shape else 0
            image_features = getattr(config, "image_features", {}) or {}
            keypoints = int(getattr(config, "spatial_softmax_num_keypoints", 32))
            observation_dim = state_dim + env_dim + len(image_features) * keypoints * 2
            conditioning_dim = observation_dim * int(getattr(config, "n_obs_steps", 1))
        if conditioning_dim < 1:
            raise ValueError("Diffusion policy must expose inferable conditioning features")
        horizon = int(getattr(config, "horizon", 1))
        self._denoising_dim = action_dim * horizon
        self.context = context
        self.checkpoint = checkpoint
        self.conditioning_location = conditioning_location
        self.denoising_location = denoising_location
        self._conditioning_dim = conditioning_dim
        self.metadata = DiffusionPolicyMetadata(
            checkpoint=checkpoint,
            conditioning_location=conditioning_location,
            denoising_location=denoising_location,
            conditioning_dim=conditioning_dim,
            denoising_dim=action_dim,
            coordinate_identity=(
                f"diffusion:{checkpoint.policy_repo_id}@{checkpoint.policy_revision}:"
                f"conditioning+denoising:{conditioning_dim}+{action_dim}"
            ),
        )

    @property
    def conditioning_space(self) -> LatentSpace:
        """Return the Euclidean space of observation conditioning vectors."""

        return LatentSpace(
            dim=self._conditioning_dim,
            source_model=self.checkpoint.policy_repo_id,
            metadata={
                "representation_role": "observation_conditioning",
                "policy_revision": self.checkpoint.policy_revision,
                "coordinate_identity": self.metadata.coordinate_identity,
            },
        )

    @property
    def denoising_space(self) -> LatentSpace:
        """Return the Euclidean space of flattened denoiser action outputs."""

        return LatentSpace(
            dim=self._denoising_dim,
            source_model=self.checkpoint.policy_repo_id,
            metadata={
                "representation_role": "denoising_action",
                "policy_revision": self.checkpoint.policy_revision,
                "coordinate_identity": self.metadata.coordinate_identity,
            },
        )

    def reset(self) -> None:
        """Reset LeRobot's observation and action queues at an episode boundary."""

        reset = getattr(self.context.policy, "reset", None)
        if not callable(reset):
            raise TypeError("Diffusion policy must expose LeRobot's reset() action-selection lifecycle")
        reset()

    def select_action(
        self,
        sample: Mapping[str, object],
        *,
        episode_step: int = 0,
        noise: np.ndarray | None = None,
    ) -> DiffusionActionSelection:
        """Preprocess, select, capture, and postprocess one observation."""

        preprocess = self.context.preprocessor
        select = getattr(self.context.policy, "select_action", None)
        postprocess = self.context.postprocessor
        if not callable(preprocess) or not callable(select) or not callable(postprocess):
            raise TypeError(
                "Diffusion context must expose callable preprocessor, policy.select_action, and postprocessor"
            )
        policy = self.context.policy
        if not isinstance(policy, nn.Module):
            raise TypeError("Diffusion policy must be a torch.nn.Module for capture")

        prepared = preprocess(sample)
        captures: list[DiffusionRepresentation] = []
        call_index = 0

        def hook(
            module: nn.Module,
            inputs: tuple[object, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> object:
            nonlocal call_index
            del module
            if not isinstance(output, Tensor):
                raise TypeError("Diffusion denoiser must return a Tensor")
            global_cond = kwargs.get("global_cond")
            if not isinstance(global_cond, Tensor):
                raise TypeError("Diffusion denoiser call must provide Tensor global_cond")
            timestep = _scalar_timestep(inputs[1] if len(inputs) > 1 else None)
            cond = _flatten_batch(global_cond)
            denoising = _flatten_batch(output)
            if cond.shape[0] != 1 or denoising.shape[0] != 1:
                raise ValueError("Diffusion capture currently requires batch size 1")
            if cond.shape[1] != self._conditioning_dim:
                raise ValueError(f"conditioning shape {cond.shape[1]} does not match {self._conditioning_dim}")
            if denoising.shape[1] != self._denoising_dim:
                raise ValueError(f"denoising shape {denoising.shape[1]} does not match {self._denoising_dim}")
            conditioning = captured_latent(
                cond[0],
                provenance={
                    "kind": "observation_conditioning",
                    "episode_step": episode_step,
                    "policy_repo_id": self.checkpoint.policy_repo_id,
                    "policy_revision": self.checkpoint.policy_revision,
                    "coordinate_identity": self.metadata.coordinate_identity,
                },
            )
            denoising_latent = captured_latent(
                denoising[0],
                provenance={
                    "kind": "denoising_action",
                    "episode_step": episode_step,
                    "denoising_step": call_index,
                    "diffusion_timestep": timestep,
                    "policy_repo_id": self.checkpoint.policy_repo_id,
                    "policy_revision": self.checkpoint.policy_revision,
                    "coordinate_identity": self.metadata.coordinate_identity,
                },
            )
            cond_metadata = _capture_metadata(
                self.conditioning_location, global_cond, 0, self.checkpoint.lerobot_version
            )
            denoising_metadata = _capture_metadata(
                self.denoising_location, output, call_index, self.checkpoint.lerobot_version
            )
            denoising_representation = DiffusionRepresentation(
                kind="denoising_action",
                latent=denoising_latent,
                capture_metadata=denoising_metadata,
                episode_step=episode_step,
                denoising_step=call_index,
                diffusion_timestep=timestep,
                metadata=self.metadata,
            )
            captures.append(denoising_representation)
            if call_index == 0:
                captures.insert(
                    0,
                    DiffusionRepresentation(
                        kind="observation_conditioning",
                        latent=conditioning,
                        capture_metadata=cond_metadata,
                        episode_step=episode_step,
                        denoising_step=None,
                        diffusion_timestep=None,
                        metadata=self.metadata,
                    ),
                )
            call_index += 1
            return output

        diffusion = policy.diffusion  # type: ignore[attr-defined]
        unet = cast(nn.Module, diffusion.unet)  # type: ignore[attr-defined]
        handle = unet.register_forward_hook(hook, with_kwargs=True)
        try:
            raw_action = select(prepared, noise=_noise_to_tensor(noise))
        finally:
            handle.remove()
        action = postprocess(raw_action)
        return DiffusionActionSelection(
            action=action, action_array=_action_to_numpy(action), representations=tuple(captures)
        )

    def capture_episode(
        self,
        samples: Sequence[Mapping[str, object]],
        *,
        episode_id: str,
        outcome: Literal["success", "failure"],
        metadata: Mapping[str, object] | None = None,
    ) -> DiffusionEpisodeTrace:
        """Capture only denoising calls that occur during normal queue rollout."""

        self.reset()
        selections: list[DiffusionActionSelection] = []
        for episode_step, sample in enumerate(samples):
            selected = self.select_action(sample, episode_step=episode_step)
            if selected.representations:
                selections.append(selected)
        if not selections:
            raise RuntimeError("the episode produced no Diffusion denoising captures")
        return DiffusionEpisodeTrace(episode_id, outcome, tuple(selections), dict(metadata or {}))


def _flatten_batch(value: Tensor) -> np.ndarray:
    """Flatten all non-batch axes while retaining one sample dimension."""

    return _to_numpy(value).reshape(int(value.shape[0]), -1)


def _noise_to_tensor(value: np.ndarray | None) -> Tensor | None:
    """Convert the public NumPy noise boundary to the upstream tensor type."""

    return None if value is None else Tensor(value)


def _scalar_timestep(value: object) -> int:
    """Extract a scalar scheduler timestep from a denoiser call."""

    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError(f"expected one timestep for batch size 1, got shape {array.shape}")
    return int(array.reshape(-1)[0])


def _capture_metadata(location: str, tensor: Tensor, call_index: int, version: str) -> CaptureMetadata:
    """Build shared capture metadata for a tensor observed through a hook."""

    return CaptureMetadata(
        location=location,
        call_index=call_index,
        shape=tuple(int(size) for size in tensor.shape),
        batch_axis=0 if tensor.ndim >= 1 else None,
        sequence_axis=None,
        device=str(tensor.device),
        dtype=str(tensor.dtype).removeprefix("torch."),
        source_model_version=f"lerobot-{version}",
    )


def load_diffusion_policy(
    checkpoint: DiffusionCheckpointSpec = DEFAULT_DIFFUSION_CHECKPOINT,
    *,
    api: LeRobotAPI | None = None,
    dataset_meta: object | None = None,
    device: str = "cpu",
) -> DiffusionPolicyAdapter:
    """Load a pinned Diffusion policy through official LeRobot factories."""

    upstream_api = api if api is not None else load_lerobot_api()
    config_module = import_module("lerobot.policies.diffusion.configuration_diffusion")
    config_type = getattr(config_module, "DiffusionConfig")  # noqa: B009 - optional upstream symbol
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

    policy = upstream_api.make_policy(config, ds_meta=resolved_meta)
    processors = upstream_api.make_pre_post_processors(
        config,
        pretrained_path=checkpoint.policy_repo_id,
        pretrained_revision=checkpoint.policy_revision,
        dataset_stats=getattr(resolved_meta, "stats", None),
        dataset_meta=resolved_meta,
    )
    preprocessor, postprocessor = cast(tuple[object, object], processors)
    context = LeRobotPolicyContext(
        policy_name="diffusion",
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
    return DiffusionPolicyAdapter(context, checkpoint=checkpoint)


def analyze_diffusion_traces(
    traces: Sequence[DiffusionEpisodeTrace],
    *,
    n_components: int = 2,
    probe_config: LinearProbeConfig | None = None,
    random_state: int = 0,
) -> DiffusionAnalysisResult:
    """Analyze conditioning labels and keep episode/timestep trajectories distinct."""

    if len(traces) < 2:
        raise ValueError("at least two Diffusion episode traces are required")
    outcomes = {trace.outcome for trace in traces}
    if outcomes != {"success", "failure"}:
        raise ValueError("Diffusion analysis requires both success and failure traces")

    features = np.vstack([trace.conditioning_trajectory.to_numpy() for trace in traces])
    labels = np.concatenate(
        [
            np.full(len(trace.conditioning_trajectory), int(trace.outcome == "success"), dtype=np.int64)
            for trace in traces
        ]
    )
    projection = PCA(n_components=n_components)
    projection.fit(features)
    projected = projection.transform(features)
    probe = LinearProbe(probe_config or LinearProbeConfig(random_state=random_state))
    probe_result = probe.fit(
        features, labels, provenance={"analysis": "diffusion_observational", "label": "episode_outcome"}
    )
    controls = compute_controls(
        features,
        labels,
        train_indices=probe_result.train_indices,
        test_indices=probe_result.test_indices,
        random_state=random_state,
    )

    success_features = np.vstack(
        [trace.conditioning_trajectory.to_numpy() for trace in traces if trace.outcome == "success"]
    )
    failure_features = np.vstack(
        [trace.conditioning_trajectory.to_numpy() for trace in traces if trace.outcome == "failure"]
    )
    split = max(features.shape[1] * 3, success_features.shape[0] // 2)
    if split >= success_features.shape[0]:
        split = success_features.shape[0] - 1
    density = GaussianMixtureDensity(GMMConfig(n_components=1, random_state=random_state)).fit(
        success_features[:split],
        source_representation_identity="diffusion:observation_conditioning",
    )
    density_report = density.evaluate(
        success_features[split:],
        failure_features,
        source_representation_identity="diffusion:observation_conditioning",
        split_provenance={"fit": "success_prefix", "evaluation": "success_suffix_vs_failure"},
    )

    space = LatentSpace(dim=features.shape[1], source_model="diffusion_observation_conditioning")
    episode_lengths: dict[str, float] = {}
    timestep_lengths: dict[str, dict[str, float]] = {}
    for trace in traces:
        condition_values = trace.conditioning_trajectory.to_numpy()
        episode_lengths[trace.episode_id] = _trajectory_length(space, condition_values)
        per_timestep: dict[str, float] = {}
        timesteps = sorted(
            {
                int(representation.diffusion_timestep)
                for selection in trace.selections
                for representation in selection.representations
                if representation.kind == "denoising_action" and representation.diffusion_timestep is not None
            }
        )
        for timestep in timesteps:
            values = trace.denoising_by_timestep(timestep)
            denoising_space = LatentSpace(dim=values.shape[1], source_model="diffusion_denoising_action")
            per_timestep[str(timestep)] = _trajectory_length(denoising_space, values)
        timestep_lengths[trace.episode_id] = per_timestep

    return DiffusionAnalysisResult(
        projected_conditioning=projected,
        projection_explained_variance=projection.explained_variance_ratio_,
        conditioning_probe=probe_result,
        conditioning_controls=controls,
        conditioning_density_auroc=density_report.metrics.auroc,
        episode_time_lengths=episode_lengths,
        timestep_time_lengths=timestep_lengths,
        metadata={
            "analysis": "observational_diffusion_representation",
            "axes": ["episode_time", "action_chunk_position", "diffusion_timestep"],
            "negative_controls": ["majority_class", "shuffled_label", "raw_input_not_used"],
            "causal_intervention": False,
        },
    )


def _trajectory_length(space: LatentSpace, values: np.ndarray) -> float:
    """Compute consecutive path length on one explicitly chosen representation space."""

    if len(values) < 2:
        return 0.0
    return float(sum(space.distance(before, after) for before, after in zip(values[:-1], values[1:], strict=True)))


__all__ = [
    "DEFAULT_DIFFUSION_CHECKPOINT",
    "DIFFUSION_CONDITIONING_LOCATION",
    "DIFFUSION_DATASET_REPO_ID",
    "DIFFUSION_DATASET_REVISION",
    "DIFFUSION_ENV_TASK",
    "DIFFUSION_ENV_TYPE",
    "DIFFUSION_LEROBOT_VERSION",
    "DIFFUSION_DENOISING_LOCATION",
    "DIFFUSION_POLICY_REPO_ID",
    "DIFFUSION_POLICY_REVISION",
    "DiffusionActionSelection",
    "DiffusionAnalysisResult",
    "DiffusionCheckpointSpec",
    "DiffusionEpisodeTrace",
    "DiffusionPolicyAdapter",
    "DiffusionPolicyMetadata",
    "DiffusionRepresentation",
    "analyze_diffusion_traces",
    "load_diffusion_policy",
]
