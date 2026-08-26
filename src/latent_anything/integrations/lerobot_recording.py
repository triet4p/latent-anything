"""Run-record helpers at the LeRobot integration boundary.

These helpers accept bridge-owned result objects or JSON-like mappings.  They
never reach into LeRobot internals; model and dataset provenance is supplied by
the existing adapter/checkpoint metadata.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import cast

from latent_anything.integrations.lerobot import LeRobotEvaluationResult
from latent_anything.run_record import FileSystemRunRecorder, RunRecord
from latent_anything.runtime.profiling import RuntimeProfile

try:
    _framework_version = metadata.version("latent-anything")
except metadata.PackageNotFoundError:
    _framework_version = "unknown"


@dataclass(frozen=True)
class CapturePoint:
    """One supported, named representation seam."""

    policy: str
    name: str
    location: str
    representation: str
    supports_intervention: bool

    def to_dict(self) -> dict[str, object]:
        """Return this capture-point contract as a JSON-compatible mapping."""
        return cast(dict[str, object], asdict(self))


SUPPORTED_CAPTURE_POINTS: tuple[CapturePoint, ...] = (
    CapturePoint("act", "decoder_query", "model.decoder", "first_action_decoder_query", False),
    CapturePoint("diffusion", "conditioning", "diffusion.unet.global_cond", "observation_conditioning", False),
    CapturePoint("diffusion", "denoising", "diffusion.unet", "denoising_action", False),
    CapturePoint(
        "smolvla",
        "vision_context",
        "model.vlm_with_expert.vlm.model.vision_model",
        "vision_context",
        False,
    ),
    CapturePoint(
        "smolvla",
        "language_context",
        "model.vlm_with_expert.vlm.model.text_model.embed_tokens",
        "language_context",
        False,
    ),
    CapturePoint("smolvla", "state_context", "model.state_proj", "state_context", False),
    CapturePoint(
        "smolvla",
        "action_expert",
        "model.vlm_with_expert.lm_expert.norm",
        "action_expert",
        True,
    ),
)


def supported_capture_points(policy: str | None = None) -> tuple[CapturePoint, ...]:
    """Return capture seams, optionally filtered by policy kind."""

    if policy is None:
        return SUPPORTED_CAPTURE_POINTS
    normalized = policy.lower()
    return tuple(point for point in SUPPORTED_CAPTURE_POINTS if point.policy == normalized)


def _to_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return {str(key): item for key, item in converted.items()}
    return {"value": str(value)}


def _record_result(
    recorder: FileSystemRunRecorder,
    *,
    name: str,
    kind: str,
    result: object,
    config: Mapping[str, object],
    model_revisions: Mapping[str, str] | None = None,
    dataset_revisions: Mapping[str, str] | None = None,
    seeds: Sequence[int] = (),
    environment: Mapping[str, object] | None = None,
    code_version: str = "",
    metrics: Mapping[str, float] | None = None,
    runtime_profile: RuntimeProfile | None = None,
    theory_evidence_ids: Sequence[str] = (),
    parent_run_ids: Sequence[str] = (),
) -> RunRecord:
    payload = _to_mapping(result)
    record = recorder.start(
        name,
        config=dict(config),
        code_version=code_version,
        framework_version=_framework_version,
        model_revisions=dict(model_revisions or {}),
        dataset_revisions=dict(dataset_revisions or {}),
        seeds=tuple(seeds),
        environment=dict(environment or {}),
        runtime_profile=runtime_profile,
        theory_evidence_ids=tuple(theory_evidence_ids),
        parent_run_ids=tuple(parent_run_ids),
        metadata={"integration": "lerobot", "run_kind": kind},
    )
    try:
        recorder.add_artifact(
            record.run_id,
            json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n",
            name=f"{kind}.json",
            media_type="application/json",
        )
        current = recorder.get(record.run_id)
        return recorder.complete(current.run_id, metrics=dict(metrics or {}), artifacts=current.artifacts)
    except Exception as error:
        recorder.fail(record.run_id, str(error))
        raise


def record_lerobot_dataset_inspection(
    recorder: FileSystemRunRecorder,
    result: object,
    *,
    config: Mapping[str, object],
    dataset_revisions: Mapping[str, str],
    environment: Mapping[str, object] | None = None,
    code_version: str = "",
    theory_evidence_ids: Sequence[str] = (),
) -> RunRecord:
    """Record schema/episode inspection evidence without loading model state."""

    return _record_result(
        recorder,
        name="lerobot-dataset-inspection",
        kind="dataset_inspection",
        result=result,
        config=config,
        dataset_revisions=dataset_revisions,
        environment=environment,
        code_version=code_version,
        theory_evidence_ids=theory_evidence_ids,
    )


def record_lerobot_policy_capture(
    recorder: FileSystemRunRecorder,
    result: object,
    *,
    config: Mapping[str, object],
    model_revisions: Mapping[str, str],
    dataset_revisions: Mapping[str, str] | None = None,
    seeds: Sequence[int] = (),
    environment: Mapping[str, object] | None = None,
    code_version: str = "",
    runtime_profile: RuntimeProfile | None = None,
    theory_evidence_ids: Sequence[str] = (),
    parent_run_ids: Sequence[str] = (),
) -> RunRecord:
    """Record observational policy-representation capture evidence."""

    return _record_result(
        recorder,
        name="lerobot-policy-capture",
        kind="policy_capture",
        result=result,
        config=config,
        model_revisions=model_revisions,
        dataset_revisions=dataset_revisions,
        seeds=seeds,
        environment=environment,
        code_version=code_version,
        runtime_profile=runtime_profile,
        theory_evidence_ids=theory_evidence_ids,
        parent_run_ids=parent_run_ids,
    )


def record_lerobot_intervention(
    recorder: FileSystemRunRecorder,
    result: object,
    *,
    config: Mapping[str, object],
    model_revisions: Mapping[str, str],
    dataset_revisions: Mapping[str, str] | None = None,
    seeds: Sequence[int] = (),
    environment: Mapping[str, object] | None = None,
    code_version: str = "",
    runtime_profile: RuntimeProfile | None = None,
    theory_evidence_ids: Sequence[str] = (),
    parent_run_ids: Sequence[str] = (),
) -> RunRecord:
    """Record a bounded intervention result and its declared provenance."""

    return _record_result(
        recorder,
        name="lerobot-intervention",
        kind="intervention",
        result=result,
        config=config,
        model_revisions=model_revisions,
        dataset_revisions=dataset_revisions,
        seeds=seeds,
        environment=environment,
        code_version=code_version,
        runtime_profile=runtime_profile,
        theory_evidence_ids=theory_evidence_ids,
        parent_run_ids=parent_run_ids,
    )


def record_lerobot_evaluation(
    recorder: FileSystemRunRecorder,
    result: LeRobotEvaluationResult | Mapping[str, object] | object,
    *,
    config: Mapping[str, object],
    model_revisions: Mapping[str, str],
    dataset_revisions: Mapping[str, str] | None = None,
    seeds: Sequence[int] = (),
    environment: Mapping[str, object] | None = None,
    code_version: str = "",
    metrics: Mapping[str, float] | None = None,
    runtime_profile: RuntimeProfile | None = None,
    theory_evidence_ids: Sequence[str] = (),
    parent_run_ids: Sequence[str] = (),
) -> RunRecord:
    """Record environment/evaluation evidence and preserve parent linkage."""

    payload = _to_mapping(result)
    result_metrics = dict(metrics or {})
    if isinstance(result, LeRobotEvaluationResult):
        result_metrics.update(result.metrics)
        if result.success_rate is not None:
            result_metrics.setdefault("success_rate", result.success_rate)
    return _record_result(
        recorder,
        name="lerobot-evaluation",
        kind="evaluation",
        result=payload,
        config=config,
        model_revisions=model_revisions,
        dataset_revisions=dataset_revisions,
        seeds=seeds,
        environment=environment,
        code_version=code_version,
        metrics=result_metrics,
        runtime_profile=runtime_profile,
        theory_evidence_ids=theory_evidence_ids,
        parent_run_ids=parent_run_ids,
    )


__all__ = [
    "CapturePoint",
    "SUPPORTED_CAPTURE_POINTS",
    "record_lerobot_dataset_inspection",
    "record_lerobot_evaluation",
    "record_lerobot_intervention",
    "record_lerobot_policy_capture",
    "supported_capture_points",
]
