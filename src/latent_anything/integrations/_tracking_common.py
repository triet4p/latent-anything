"""Provider-neutral validation/state helpers for optional tracking adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from latent_anything.experiment_recorder import (
    MAX_CONFIG_BYTES,
    MAX_METRIC_EVENTS,
    RecorderContractError,
    RecorderRunInfo,
    canonical_recorder_json,
    compute_recorder_identity,
    validate_recorder_mapping,
    validate_recorder_metrics,
    validate_recorder_name,
    validate_recorder_tags,
)


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """Validated provider-independent start arguments."""

    name: str
    config: dict[str, object]
    tags: dict[str, str]
    parent_run_id: str | None
    identity: str
    provider_tags: dict[str, str]


def prepare_run(
    name: str,
    *,
    config: Mapping[str, object] | None,
    tags: Mapping[str, str] | None,
    parent_run_id: str | None,
    code_version: str,
    framework_version: str,
    model_revisions: Mapping[str, str] | None,
    dataset_revisions: Mapping[str, str] | None,
    seeds: Sequence[int],
    environment: Mapping[str, object] | None,
    metadata: Mapping[str, object] | None,
) -> PreparedRun:
    resolved_name = validate_recorder_name(name)
    resolved_config = validate_recorder_mapping(config, field="config")
    resolved_tags = validate_recorder_tags(tags)
    resolved_metadata = validate_recorder_mapping(metadata, field="metadata")
    resolved_parent = _validate_parent(parent_run_id)
    identity = compute_recorder_identity(
        name=resolved_name,
        config=resolved_config,
        tags=resolved_tags,
        parent_run_id=resolved_parent,
        code_version=code_version,
        framework_version=framework_version,
        model_revisions=model_revisions,
        dataset_revisions=dataset_revisions,
        seeds=seeds,
        environment=environment,
        metadata=resolved_metadata,
    )
    provider_tags = {
        **resolved_tags,
        "latent_anything.identity": identity,
        "latent_anything.contract": "experiment-recorder-v1",
    }
    if resolved_parent is not None:
        provider_tags["latent_anything.parent_run_id"] = resolved_parent
    return PreparedRun(
        name=resolved_name,
        config=resolved_config,
        tags=resolved_tags,
        parent_run_id=resolved_parent,
        identity=identity,
        provider_tags=provider_tags,
    )


def _validate_parent(parent_run_id: str | None) -> str | None:
    if parent_run_id is None:
        return None
    if (
        not parent_run_id
        or len(parent_run_id) > 256
        or any(character in parent_run_id for character in ("\x00", "\n", "\r"))
    ):
        raise RecorderContractError("parent_run_id must be a bounded identifier")
    return parent_run_id


@dataclass(slots=True)
class ExternalRunState:
    """Shared lifecycle/step validation for SDK-backed run handles."""

    info: RecorderRunInfo
    params: dict[str, object]
    tags: dict[str, str]
    latest_metrics: dict[str, float] = field(default_factory=dict)
    last_step: int = -1
    metric_events: int = 0

    def require_running(self) -> None:
        if self.info.status != "running":
            raise RecorderContractError(f"run is already {self.info.status}")

    def prepare_params(self, params: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        self.require_running()
        values = validate_recorder_mapping(params, field="params")
        for key, value in values.items():
            if key in self.params and self.params[key] != value:
                raise RecorderContractError(f"parameter {key!r} cannot change after it is recorded")
        candidate = {**self.params, **values}
        if len(canonical_recorder_json(candidate)) > MAX_CONFIG_BYTES:
            raise RecorderContractError("recorder params exceed the cumulative serialized size bound")
        return values, candidate

    def commit_params(self, candidate: dict[str, object]) -> None:
        self.params = candidate

    def prepare_metrics(
        self, metrics: Mapping[str, float], step: int
    ) -> tuple[dict[str, float], dict[str, float], int, int]:
        self.require_running()
        if type(step) is not int or step < 0 or step < self.last_step:
            raise RecorderContractError("metric steps must be non-negative and non-decreasing")
        if self.metric_events >= MAX_METRIC_EVENTS:
            raise RecorderContractError("metric event bound exceeded")
        values = validate_recorder_metrics(metrics)
        candidate = {**self.latest_metrics, **values}
        if len(canonical_recorder_json(candidate)) > MAX_CONFIG_BYTES:
            raise RecorderContractError("recorder metrics exceed the cumulative serialized size bound")
        return values, candidate, step, self.metric_events + 1

    def commit_metrics(self, candidate: dict[str, float], step: int, metric_events: int) -> None:
        self.last_step = step
        self.metric_events = metric_events
        self.latest_metrics = candidate

    def prepare_tags(self, tags: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        self.require_running()
        values = validate_recorder_tags(tags)
        candidate = {**self.tags, **values}
        if len(canonical_recorder_json(candidate)) > MAX_CONFIG_BYTES:
            raise RecorderContractError("recorder tags exceed the cumulative serialized size bound")
        return values, candidate

    def commit_tags(self, candidate: dict[str, str]) -> None:
        self.tags = candidate

    def record_params(self, params: Mapping[str, object]) -> dict[str, object]:
        values, candidate = self.prepare_params(params)
        self.commit_params(candidate)
        return values

    def record_metrics(self, metrics: Mapping[str, float], step: int) -> dict[str, float]:
        values, candidate, resolved_step, metric_events = self.prepare_metrics(metrics, step)
        self.commit_metrics(candidate, resolved_step, metric_events)
        return values

    def record_tags(self, tags: Mapping[str, str]) -> dict[str, str]:
        values, candidate = self.prepare_tags(tags)
        self.commit_tags(candidate)
        return values

    def finish(self, *, status: str) -> RecorderRunInfo:
        self.require_running()
        if status not in {"completed", "failed"}:
            raise RecorderContractError("unsupported terminal recorder status")
        self.info = RecorderRunInfo(
            run_id=self.info.run_id,
            identity=self.info.identity,
            name=self.info.name,
            status="completed" if status == "completed" else "failed",
            backend=self.info.backend,
            parent_run_id=self.info.parent_run_id,
        )
        return self.info
