"""Private LeRobot/SmolVLA checkpoint and processor construction."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from latent_anything.integrations.lerobot import LeRobotAPI, LeRobotPolicyContext, load_lerobot_api

if TYPE_CHECKING:
    from latent_anything.integrations.lerobot_smolvla import SmolVLACheckpointSpec, SmolVLAPolicyAdapter


def load_smolvla_policy(
    checkpoint: SmolVLACheckpointSpec,
    *,
    api: LeRobotAPI | None = None,
    dataset_meta: object | None = None,
    device: str = "cpu",
) -> SmolVLAPolicyAdapter:
    """Construct the pinned policy through official LeRobot factories."""

    from latent_anything.integrations.lerobot_smolvla import (
        SMOLVLA_RENAME_MAP,
        SmolVLAPolicyAdapter,
    )

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
