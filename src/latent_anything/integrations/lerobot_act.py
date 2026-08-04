"""ACT policy capture and observational analysis on the LeRobot path.

The module keeps LeRobot policy, processor, and dataset objects upstream-owned.
It adds only the capture lifecycle, immutable checkpoint provenance, and a
small analysis result around the first decoder query token that feeds ACT's
first selected action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import import_module
from typing import Literal, cast

import numpy as np
from torch import nn

from latent_anything.capture import ActivationCaptureSession, CaptureMetadata
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

ACT_POLICY_REPO_ID = "lerobot/act_aloha_sim_insertion_human"
ACT_POLICY_REVISION = "33259aa86eb45fdf85350280044a33d9d50e40c3"
ACT_DATASET_REPO_ID = "lerobot/aloha_sim_insertion_human"
ACT_DATASET_REVISION = "cc571a3c661df81b566dbfde3d5c1e85fcdf7884"
ACT_LEROBOT_VERSION = "0.6.1"
ACT_CAPTURE_LOCATION = "model.decoder"


@dataclass(frozen=True)
class ACTCheckpointSpec:
    """Immutable public identity for the ACT checkpoint and paired dataset."""

    policy_repo_id: str = ACT_POLICY_REPO_ID
    policy_revision: str = ACT_POLICY_REVISION
    dataset_repo_id: str = ACT_DATASET_REPO_ID
    dataset_revision: str = ACT_DATASET_REVISION
    lerobot_version: str = ACT_LEROBOT_VERSION

    def __post_init__(self) -> None:
        for name in ("policy_repo_id", "policy_revision", "dataset_repo_id", "dataset_revision"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible checkpoint identity."""

        return cast(dict[str, str], asdict(self))


DEFAULT_ACT_CHECKPOINT = ACTCheckpointSpec()


@dataclass(frozen=True)
class ACTPolicyMetadata:
    """Capture-point and model provenance for one ACT adapter."""

    checkpoint: ACTCheckpointSpec
    capture_location: str
    representation_role: str
    representation_dim: int
    coordinate_identity: str

    def to_dict(self) -> dict[str, object]:
        """Return serializable metadata without upstream objects."""

        result = asdict(self)
        result["checkpoint"] = self.checkpoint.to_dict()
        return result


@dataclass(frozen=True)
class ACTRepresentation:
    """One first-action decoder-query representation and its provenance."""

    latent: LeRobotCapturedLatent
    capture_metadata: CaptureMetadata
    metadata: ACTPolicyMetadata

    def to_dict(self) -> dict[str, object]:
        """Return metadata and shape information without serializing arrays."""

        return {
            "capture_metadata": asdict(self.capture_metadata),
            "latent_shape": list(self.latent.values.shape),
            "latent_dtype": str(self.latent.values.dtype),
            "metadata": self.metadata.to_dict(),
            "provenance": dict(self.latent.provenance),
        }


@dataclass(frozen=True)
class ACTActionSelection:
    """Official post-processed action plus an optional captured query token."""

    action: object
    action_array: np.ndarray
    representation: ACTRepresentation | None

    def __post_init__(self) -> None:
        values = np.array(self.action_array, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "action_array", values)


@dataclass(frozen=True)
class ACTEpisodeTrace:
    """Captured representations from one labeled episode."""

    episode_id: str
    outcome: Literal["success", "failure"]
    representations: tuple[ACTRepresentation, ...]
    actions: tuple[np.ndarray, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.representations:
            raise ValueError("an ACT episode trace must contain at least one representation")
        if len(self.representations) != len(self.actions):
            raise ValueError("representations and actions must have matching lengths")

    @property
    def trajectory(self) -> Trajectory:
        """Return the first-action decoder representations as a trajectory."""

        values = np.stack([representation.latent.values for representation in self.representations])
        return Trajectory(values, metadata={"episode_id": self.episode_id, "outcome": self.outcome, **self.metadata})


@dataclass(frozen=True)
class ACTAnalysisResult:
    """Projection, linear probe, controls, and trajectory metrics."""

    projected: np.ndarray
    projection_explained_variance: np.ndarray
    probe: LinearProbeResult
    controls: ControlBaselines
    trajectory_lengths: Mapping[str, float]
    trajectory_velocity_means: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible observational analysis summary."""

        return {
            "projected_shape": list(self.projected.shape),
            "projection_explained_variance": self.projection_explained_variance.tolist(),
            "probe": self.probe.to_dict(),
            "controls": asdict(self.controls),
            "trajectory_lengths": dict(self.trajectory_lengths),
            "trajectory_velocity_means": dict(self.trajectory_velocity_means),
            "metadata": dict(self.metadata),
        }


def _to_numpy(value: object) -> np.ndarray:
    """Convert a tensor-like or array-like value into an owned NumPy array."""

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


def _first_query_token(values: np.ndarray) -> np.ndarray:
    """Select the token that feeds ACT's first action from decoder output."""

    if values.ndim == 3:
        if values.shape[1] != 1:
            raise ValueError(
                "ACT capture currently requires batch size 1 so the first decoder query is unambiguous; "
                f"got decoder output shape {values.shape}"
            )
        return np.array(values[0, 0], copy=True)
    if values.ndim == 2:
        return np.array(values[0], copy=True)
    if values.ndim == 1:
        return np.array(values, copy=True)
    raise ValueError(f"ACT decoder capture must be 1D, 2D, or 3D; got shape {values.shape}")


class ACTPolicyAdapter:
    """Capture ACT decoder query tokens around normal LeRobot inference."""

    def __init__(
        self,
        context: LeRobotPolicyContext,
        *,
        checkpoint: ACTCheckpointSpec = DEFAULT_ACT_CHECKPOINT,
        capture_location: str = ACT_CAPTURE_LOCATION,
        representation_dim: int | None = None,
    ) -> None:
        if not capture_location:
            raise ValueError("capture_location must not be empty")
        policy_config = getattr(context.policy, "config", None)
        inferred_dim = getattr(policy_config, "dim_model", None)
        resolved_dim = representation_dim if representation_dim is not None else inferred_dim
        if not isinstance(resolved_dim, int) or resolved_dim < 1:
            raise ValueError("representation_dim or policy.config.dim_model must be a positive integer")
        self.context = context
        self.checkpoint = checkpoint
        self.capture_location = capture_location
        self._representation_dim = resolved_dim
        identity = (
            f"act:{checkpoint.policy_repo_id}@{checkpoint.policy_revision}:"
            f"{capture_location}:first-query:{resolved_dim}"
        )
        self.metadata = ACTPolicyMetadata(
            checkpoint=checkpoint,
            capture_location=capture_location,
            representation_role="first_action_decoder_query",
            representation_dim=resolved_dim,
            coordinate_identity=identity,
        )

    @property
    def latent_space(self) -> LatentSpace:
        """Return the Euclidean space of one ACT decoder query token."""

        return LatentSpace(
            dim=self._representation_dim,
            source_model=self.checkpoint.policy_repo_id,
            metadata={
                "representation_role": self.metadata.representation_role,
                "capture_location": self.capture_location,
                "policy_revision": self.checkpoint.policy_revision,
                "dataset_repo_id": self.checkpoint.dataset_repo_id,
                "dataset_revision": self.checkpoint.dataset_revision,
                "coordinate_identity": self.metadata.coordinate_identity,
            },
        )

    def reset(self) -> None:
        """Reset LeRobot's action queue at an episode boundary."""

        reset = getattr(self.context.policy, "reset", None)
        if not callable(reset):
            raise TypeError("ACT policy must expose LeRobot's reset() action-selection lifecycle")
        reset()

    def select_action(self, sample: Mapping[str, object]) -> ACTActionSelection:
        """Preprocess, select, capture, and postprocess one observation."""

        preprocess = self.context.preprocessor
        select = getattr(self.context.policy, "select_action", None)
        postprocess = self.context.postprocessor
        if not callable(preprocess) or not callable(select) or not callable(postprocess):
            raise TypeError("ACT context must expose callable preprocessor, policy.select_action, and postprocessor")

        prepared = preprocess(sample)
        policy = self.context.policy
        if not isinstance(policy, nn.Module):
            raise TypeError("ACT policy must be a torch.nn.Module for shared activation capture")
        with ActivationCaptureSession(
            policy,
            [self.capture_location],
            source_model_version=f"lerobot-{self.checkpoint.lerobot_version}",
        ) as session:
            raw_action = select(prepared)
        action = postprocess(raw_action)
        action_array = _action_to_numpy(action)
        if not session.captures:
            return ACTActionSelection(action=action, action_array=action_array, representation=None)
        capture = session.captures[-1]
        query = _first_query_token(capture.values)
        if query.shape != (self._representation_dim,):
            raise ValueError(
                f"ACT first decoder query shape {query.shape} does not match representation_dim="
                f"{self._representation_dim}"
            )
        latent = captured_latent(
            query,
            provenance={
                "coordinate_identity": self.metadata.coordinate_identity,
                "capture_location": self.capture_location,
                "capture_shape": list(capture.values.shape),
                "query_index": 0,
                "policy_repo_id": self.checkpoint.policy_repo_id,
                "policy_revision": self.checkpoint.policy_revision,
                "dataset_repo_id": self.checkpoint.dataset_repo_id,
                "dataset_revision": self.checkpoint.dataset_revision,
            },
        )
        representation = ACTRepresentation(latent=latent, capture_metadata=capture.metadata, metadata=self.metadata)
        return ACTActionSelection(action=action, action_array=action_array, representation=representation)

    def capture_episode(
        self,
        samples: Sequence[Mapping[str, object]],
        *,
        episode_id: str,
        outcome: Literal["success", "failure"],
        metadata: Mapping[str, object] | None = None,
    ) -> ACTEpisodeTrace:
        """Capture an episode while preserving LeRobot's action queue semantics."""

        self.reset()
        representations: list[ACTRepresentation] = []
        actions: list[np.ndarray] = []
        for sample in samples:
            selection = self.select_action(sample)
            if selection.representation is not None:
                representations.append(selection.representation)
                actions.append(selection.action_array)
        if not representations:
            raise RuntimeError("the episode produced no ACT decoder query captures")
        return ACTEpisodeTrace(
            episode_id=episode_id,
            outcome=outcome,
            representations=tuple(representations),
            actions=tuple(actions),
            metadata=dict(metadata or {}),
        )


def load_act_policy(
    checkpoint: ACTCheckpointSpec = DEFAULT_ACT_CHECKPOINT,
    *,
    api: LeRobotAPI | None = None,
    dataset_meta: object | None = None,
    device: str = "cpu",
    capture_location: str = ACT_CAPTURE_LOCATION,
) -> ACTPolicyAdapter:
    """Load a pinned ACT policy and official LeRobot processors.

    The factory path is intentionally upstream-owned: ``ACTConfig`` and
    ``LeRobotDatasetMetadata`` come from LeRobot, while policy and processor
    construction goes through ``LeRobotAPI.make_policy`` and
    ``make_pre_post_processors``.
    """

    upstream_api = api if api is not None else load_lerobot_api()
    act_config_module = import_module("lerobot.policies.act.configuration_act")
    act_config_type = getattr(act_config_module, "ACTConfig")  # noqa: B009 - optional upstream symbol
    config_loader = getattr(act_config_type, "from_pretrained")  # noqa: B009 - optional upstream symbol
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
        policy_name="act",
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=resolved_meta,
        metadata={
            "policy_repo_id": checkpoint.policy_repo_id,
            "policy_revision": checkpoint.policy_revision,
            "dataset_repo_id": checkpoint.dataset_repo_id,
            "dataset_revision": checkpoint.dataset_revision,
            "lerobot_version": checkpoint.lerobot_version,
        },
    )
    return ACTPolicyAdapter(context, checkpoint=checkpoint, capture_location=capture_location)


def analyze_act_traces(
    traces: Sequence[ACTEpisodeTrace],
    *,
    n_components: int = 2,
    probe_config: LinearProbeConfig | None = None,
    random_state: int = 0,
) -> ACTAnalysisResult:
    """Run PCA, label probing, controls, and trajectory metrics on ACT traces."""

    if len(traces) < 2:
        raise ValueError("at least two ACT episode traces are required")
    outcomes = {trace.outcome for trace in traces}
    if outcomes != {"success", "failure"}:
        raise ValueError("ACT analysis requires both success and failure traces")

    features = np.vstack([trace.trajectory.to_numpy() for trace in traces])
    labels = np.concatenate(
        [np.full(len(trace.trajectory), 1 if trace.outcome == "success" else 0, dtype=np.int64) for trace in traces]
    )
    projection = PCA(n_components=n_components)
    projection.fit(features)
    projected = projection.transform(features)
    config = probe_config or LinearProbeConfig(random_state=random_state)
    probe = LinearProbe(config)
    probe_result = probe.fit(
        features,
        labels,
        provenance={"analysis": "act_observational", "label": "episode_outcome"},
    )
    controls = compute_controls(
        features,
        labels,
        train_indices=probe_result.train_indices,
        test_indices=probe_result.test_indices,
        random_state=random_state,
    )
    space = LatentSpace(dim=features.shape[1], source_model="act_decoder_query")
    lengths: dict[str, float] = {}
    velocity_means: dict[str, float] = {}
    for trace in traces:
        trajectory = trace.trajectory
        trajectory_values = trajectory.to_numpy()
        velocities = np.asarray(
            [
                space.distance(before, after)
                for before, after in zip(trajectory_values[:-1], trajectory_values[1:], strict=True)
            ],
            dtype=np.float64,
        )
        lengths[trace.episode_id] = float(np.sum(velocities))
        velocity_means[trace.episode_id] = float(np.mean(velocities)) if velocities.size else 0.0
    return ACTAnalysisResult(
        projected=projected,
        projection_explained_variance=projection.explained_variance_ratio_,
        probe=probe_result,
        controls=controls,
        trajectory_lengths=lengths,
        trajectory_velocity_means=velocity_means,
        metadata={
            "analysis": "observational_act_representation",
            "controls": ["majority_class", "shuffled_label", "raw_input"],
            "causal_intervention": False,
        },
    )


__all__ = [
    "ACTActionSelection",
    "ACTAnalysisResult",
    "ACTCheckpointSpec",
    "ACTEpisodeTrace",
    "ACTPolicyAdapter",
    "ACTPolicyMetadata",
    "ACTRepresentation",
    "ACT_CAPTURE_LOCATION",
    "ACT_DATASET_REPO_ID",
    "ACT_DATASET_REVISION",
    "ACT_LEROBOT_VERSION",
    "ACT_POLICY_REPO_ID",
    "ACT_POLICY_REVISION",
    "DEFAULT_ACT_CHECKPOINT",
    "analyze_act_traces",
    "load_act_policy",
]
