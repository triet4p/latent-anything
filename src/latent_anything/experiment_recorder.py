"""Small, validated experiment-recorder contract and local adapter.

The local filesystem recorder remains the source of truth for run identity and
content-addressed artifacts.  This module adds only the common lifecycle and
logging surface that optional tracking adapters can implement without exposing
their SDK types through the public API.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from latent_anything._recorder_contract import (
    MAX_ARTIFACT_BYTES,
    MAX_CONFIG_BYTES,
    MAX_METRIC_EVENTS,
    MAX_STRING_LENGTH,
    canonical_json,
    read_artifact,
    safe_artifact_path,
    validate_artifact_name,
    validate_mapping,
    validate_metrics,
    validate_name,
    validate_seeds,
    validate_string_mapping,
    validate_tags,
)
from latent_anything.run_record import (
    ArtifactRef,
    FileSystemRunRecorder,
    RunRecord,
    RunStatus,
    compute_run_identity,
)

_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class RecorderContractError(ValueError):
    """Raised when recorder input or lifecycle state violates the contract."""


@dataclass(frozen=True, slots=True)
class RecorderArtifact:
    """Backend-neutral content-addressed artifact result."""

    name: str
    digest: str
    size_bytes: int
    media_type: str = "application/octet-stream"
    uri: str | None = None

    def __post_init__(self) -> None:
        if not _SHA256_DIGEST.fullmatch(self.digest):
            raise RecorderContractError("artifact digest must be a lowercase SHA-256 hex digest")
        if self.size_bytes < 0 or self.size_bytes > MAX_ARTIFACT_BYTES:
            raise RecorderContractError("artifact size exceeds the recorder bound")
        _validate_artifact_name(self.name)


@dataclass(frozen=True, slots=True)
class RecorderRunInfo:
    """Stable information shared by local and optional tracking runs."""

    run_id: str
    identity: str
    name: str
    status: RunStatus
    backend: str
    parent_run_id: str | None = None


@runtime_checkable
class ExperimentRun(Protocol):
    """Lifecycle and logging surface proven by the local recorder."""

    @property
    def info(self) -> RecorderRunInfo:
        """Return the current immutable run information."""
        ...

    def log_params(self, params: Mapping[str, object]) -> None:
        """Record immutable parameter values."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Record finite metrics at a non-decreasing step."""
        ...

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Record bounded string tags."""
        ...

    def log_artifact(
        self,
        content: bytes | bytearray | memoryview | str | Path,
        *,
        name: str,
        media_type: str = "application/octet-stream",
    ) -> RecorderArtifact:
        """Record one bounded artifact and return its checksum."""
        ...

    def child(self, name: str, *, config: Mapping[str, object] | None = None) -> ExperimentRun:
        """Start a child run linked to this run."""
        ...

    def finish(self) -> RecorderRunInfo:
        """Finish a successful run."""
        ...

    def fail(self, error: str) -> RecorderRunInfo:
        """Finish a failed run with a bounded diagnostic."""
        ...


@runtime_checkable
class ExperimentRecorder(Protocol):
    """Backend-neutral recorder factory."""

    @property
    def backend_name(self) -> str:
        """Return a stable backend identifier."""
        ...

    def start_run(
        self,
        name: str,
        *,
        config: Mapping[str, object] | None = None,
        tags: Mapping[str, str] | None = None,
        parent_run_id: str | None = None,
        resume_run_id: str | None = None,
        code_version: str = "",
        framework_version: str = "",
        model_revisions: Mapping[str, str] | None = None,
        dataset_revisions: Mapping[str, str] | None = None,
        seeds: Sequence[int] = (),
        environment: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ExperimentRun:
        """Start or resume one reproducible run."""
        ...


def safe_recorder_artifact_path(root: str | Path, name: str) -> Path:
    """Resolve a validated artifact name below a local temporary root."""
    return safe_artifact_path(root, name, error_type=RecorderContractError)


def _canonical_json(value: object) -> bytes:
    return canonical_json(value, error_type=RecorderContractError)


def _validate_name(value: str, *, field: str = "name") -> str:
    return validate_name(value, field=field, error_type=RecorderContractError)


def _validate_artifact_name(name: str) -> str:
    return validate_artifact_name(name, error_type=RecorderContractError)


def _validate_mapping(value: Mapping[str, object] | None, *, field: str) -> dict[str, object]:
    return validate_mapping(value, field=field, error_type=RecorderContractError)


def _validate_tags(tags: Mapping[str, str] | None) -> dict[str, str]:
    return validate_tags(tags, error_type=RecorderContractError)


def _validate_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    return validate_metrics(metrics, error_type=RecorderContractError)


def _validate_string_mapping(value: Mapping[str, str] | None, *, field: str) -> dict[str, str]:
    return validate_string_mapping(value, field=field, error_type=RecorderContractError)


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    return validate_seeds(seeds, error_type=RecorderContractError)


def _read_artifact(content: bytes | bytearray | memoryview | str | Path) -> bytes:
    return read_artifact(content, error_type=RecorderContractError)


def validate_recorder_name(value: str, *, field: str = "name") -> str:
    """Validate a backend run or field name."""

    return _validate_name(value, field=field)


def validate_recorder_artifact_name(name: str) -> str:
    """Validate a safe relative artifact path."""

    return _validate_artifact_name(name)


def validate_recorder_mapping(value: Mapping[str, object] | None, *, field: str) -> dict[str, object]:
    """Normalize bounded canonical-JSON mapping input."""

    return _validate_mapping(value, field=field)


def validate_recorder_tags(tags: Mapping[str, str] | None) -> dict[str, str]:
    """Validate bounded string tags."""

    return _validate_tags(tags)


def validate_recorder_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Validate finite metric values."""

    return _validate_metrics(metrics)


def read_recorder_artifact(content: bytes | bytearray | memoryview | str | Path) -> bytes:
    """Read one bounded artifact payload."""

    return _read_artifact(content)


def canonical_recorder_json(value: object) -> bytes:
    """Return a deterministic canonical JSON representation."""

    return _canonical_json(value)


def recorder_artifact_from_bytes(
    data: bytes, *, name: str, media_type: str, uri: str | None = None
) -> RecorderArtifact:
    """Build a checksum-bearing artifact result from bounded bytes."""

    _validate_artifact_name(name)
    return RecorderArtifact(
        name=name,
        digest=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        media_type=media_type,
        uri=uri,
    )


def compute_recorder_identity(
    *,
    name: str,
    config: Mapping[str, object] | None = None,
    tags: Mapping[str, str] | None = None,
    parent_run_id: str | None = None,
    code_version: str = "",
    framework_version: str = "",
    model_revisions: Mapping[str, str] | None = None,
    dataset_revisions: Mapping[str, str] | None = None,
    seeds: Sequence[int] = (),
    environment: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Compute the local recorder identity for an equivalent backend run."""

    _validate_name(name)
    resolved_tags = _validate_tags(tags)
    resolved_metadata = _validate_mapping(metadata, field="metadata")
    resolved_metadata["recorder_tags"] = resolved_tags
    resolved_code_version = code_version or os.environ.get("LATENT_ANYTHING_CODE_VERSION", "working-tree")
    if framework_version:
        resolved_framework_version = framework_version
    else:
        try:
            resolved_framework_version = importlib_metadata.version("latent-anything")
        except importlib_metadata.PackageNotFoundError:
            resolved_framework_version = "unknown"
    return compute_run_identity(
        name=name,
        config=_validate_mapping(config, field="config"),
        code_version=resolved_code_version,
        framework_version=resolved_framework_version,
        model_revisions=_validate_string_mapping(model_revisions, field="model_revisions"),
        dataset_revisions=_validate_string_mapping(dataset_revisions, field="dataset_revisions"),
        seeds=_validate_seeds(seeds),
        environment=_validate_mapping(environment, field="environment"),
        parent_run_ids=() if parent_run_id is None else (parent_run_id,),
        metadata=resolved_metadata,
    )


class LocalExperimentRecorder:
    """Adapter exposing ``FileSystemRunRecorder`` through the common contract."""

    backend_name = "filesystem"

    def __init__(self, root: str | Path) -> None:
        self._recorder = FileSystemRunRecorder(root)

    def start_run(
        self,
        name: str,
        *,
        config: Mapping[str, object] | None = None,
        tags: Mapping[str, str] | None = None,
        parent_run_id: str | None = None,
        resume_run_id: str | None = None,
        code_version: str = "",
        framework_version: str = "",
        model_revisions: Mapping[str, str] | None = None,
        dataset_revisions: Mapping[str, str] | None = None,
        seeds: Sequence[int] = (),
        environment: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> LocalExperimentRun:
        """Start or resume a local run after validating identity inputs."""
        _validate_name(name)
        resolved_config = _validate_mapping(config, field="config")
        resolved_tags = _validate_tags(tags)
        resolved_environment = _validate_mapping(environment, field="environment")
        resolved_model_revisions = _validate_string_mapping(model_revisions, field="model_revisions")
        resolved_dataset_revisions = _validate_string_mapping(dataset_revisions, field="dataset_revisions")
        resolved_seeds = _validate_seeds(seeds)
        resolved_metadata = _validate_mapping(metadata, field="metadata")
        resolved_metadata["recorder_tags"] = resolved_tags
        if resume_run_id is not None:
            record = self._recorder.get(resume_run_id)
            self._validate_resume_inputs(
                record,
                name=name,
                config=config,
                tags=tags,
                parent_run_id=parent_run_id,
                code_version=code_version,
                framework_version=framework_version,
                model_revisions=model_revisions,
                dataset_revisions=dataset_revisions,
                seeds=seeds,
                environment=environment,
                metadata=metadata,
            )
        else:
            record = self._recorder.start(
                name,
                config=resolved_config,
                parent_run_ids=() if parent_run_id is None else (parent_run_id,),
                code_version=code_version,
                framework_version=framework_version,
                model_revisions=resolved_model_revisions,
                dataset_revisions=resolved_dataset_revisions,
                seeds=resolved_seeds,
                environment=resolved_environment,
                metadata=resolved_metadata,
            )
        expected = compute_recorder_identity(
            name=record.name,
            config=record.config,
            tags=(
                resolved_tags
                if resume_run_id is None
                else cast(Mapping[str, str], record.metadata.get("recorder_tags", {}))
            ),
            parent_run_id=(
                parent_run_id
                if resume_run_id is None
                else (record.parent_run_ids[0] if record.parent_run_ids else None)
            ),
            code_version=record.code_version,
            framework_version=record.framework_version,
            model_revisions=record.model_revisions,
            dataset_revisions=record.dataset_revisions,
            seeds=record.seeds,
            environment=record.environment,
            metadata={key: value for key, value in record.metadata.items() if key != "recorder_tags"},
        )
        if expected != record.identity:
            raise RecorderContractError("local run identity does not match recorder inputs")
        return LocalExperimentRun(self, record)

    @staticmethod
    def _validate_resume_inputs(
        record: RunRecord,
        *,
        name: str,
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
    ) -> None:
        """Reject explicit resume inputs that differ from the stored identity."""

        if name != record.name:
            raise RecorderContractError("local resume name does not match stored identity")
        if config is not None and _validate_mapping(config, field="config") != dict(record.config):
            raise RecorderContractError("local resume config does not match stored identity")
        stored_tags = cast(Mapping[str, str], record.metadata.get("recorder_tags", {}))
        if tags is not None and _validate_tags(tags) != dict(stored_tags):
            raise RecorderContractError("local resume tags do not match stored identity")
        stored_parent = record.parent_run_ids[0] if record.parent_run_ids else None
        if parent_run_id is not None and parent_run_id != stored_parent:
            raise RecorderContractError("local resume parent does not match stored identity")
        if code_version and code_version != record.code_version:
            raise RecorderContractError("local resume code version does not match stored identity")
        if framework_version and framework_version != record.framework_version:
            raise RecorderContractError("local resume framework version does not match stored identity")
        if model_revisions is not None and {str(key): str(value) for key, value in model_revisions.items()} != dict(
            record.model_revisions
        ):
            raise RecorderContractError("local resume model revisions do not match stored identity")
        if dataset_revisions is not None and {str(key): str(value) for key, value in dataset_revisions.items()} != dict(
            record.dataset_revisions
        ):
            raise RecorderContractError("local resume dataset revisions do not match stored identity")
        if seeds and tuple(seeds) != record.seeds:
            raise RecorderContractError("local resume seeds do not match stored identity")
        if environment is not None and _validate_mapping(environment, field="environment") != dict(record.environment):
            raise RecorderContractError("local resume environment does not match stored identity")
        if metadata is not None:
            resolved_metadata = _validate_mapping(metadata, field="metadata")
            stored_metadata = {key: value for key, value in record.metadata.items() if key != "recorder_tags"}
            if resolved_metadata != stored_metadata:
                raise RecorderContractError("local resume metadata does not match stored identity")

    def get_record(self, run_id: str) -> RunRecord:
        """Return the immutable local record identified by ``run_id``."""
        return self._recorder.get(run_id)

    def update_record(self, run_id: str, **changes: object) -> RunRecord:
        """Apply validated lifecycle changes and return the updated record."""
        return self._recorder.update(run_id, **changes)

    def add_artifact(
        self,
        run_id: str,
        content: bytes,
        *,
        name: str,
        media_type: str,
    ) -> ArtifactRef:
        """Store bytes under the run's safe content-addressed artifact layout."""
        return self._recorder.add_artifact(run_id, content, name=name, media_type=media_type)

    def add_json_artifact(self, run_id: str, value: object, *, name: str) -> ArtifactRef:
        """Canonicalize and store one JSON artifact for a local run."""
        return self._recorder.add_json_artifact(run_id, value, name=name)

    def read_artifact(self, reference: ArtifactRef) -> bytes:
        """Read and checksum-verify a previously stored artifact."""
        return self._recorder.read_artifact(reference)

    def complete(self, run_id: str, *, metrics: Mapping[str, float]) -> RunRecord:
        """Mark a local run completed with its final numeric metrics."""
        return self._recorder.complete(run_id, metrics=metrics)

    def fail(self, run_id: str, error: str) -> RunRecord:
        """Mark a local run failed with a bounded diagnostic string."""
        return self._recorder.fail(run_id, error)


class LocalExperimentRun:
    """Mutable contract handle backed by one local immutable ``RunRecord``."""

    def __init__(self, owner: LocalExperimentRecorder, record: RunRecord) -> None:
        self._owner = owner
        self._record = record
        self._params = dict(record.config)
        self._tags = dict(cast(Mapping[str, str], record.metadata.get("recorder_tags", {})))
        self._last_step = -1
        self._metric_events = 0
        self._metric_history: list[dict[str, object]] = []
        self._latest_metrics: dict[str, float] = dict(record.metrics)
        self._state_index = 0
        self._restore_state(record)

    def _restore_state(self, record: RunRecord) -> None:
        state_refs = sorted(
            (reference for reference in record.artifacts if reference.name.startswith("recorder-state-")),
            key=lambda reference: reference.name,
        )
        if not state_refs:
            return
        latest = state_refs[-1]
        try:
            state = json.loads(self._owner.read_artifact(latest).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RecorderContractError("local recorder state artifact is invalid") from error
        if not isinstance(state, Mapping):
            raise RecorderContractError("local recorder state artifact must contain an object")
        raw_params = state.get("params")
        raw_tags = state.get("tags")
        self._params = _validate_mapping(
            cast(Mapping[str, object], raw_params) if isinstance(raw_params, Mapping) else {}, field="params"
        )
        self._tags = _validate_tags(cast(Mapping[str, str], raw_tags) if isinstance(raw_tags, Mapping) else {})
        history = state.get("metric_history", [])
        if not isinstance(history, list) or len(history) > MAX_METRIC_EVENTS:
            raise RecorderContractError("local recorder metric history is invalid")
        for event in history:
            if not isinstance(event, Mapping) or type(event.get("step")) is not int:
                raise RecorderContractError("local recorder metric history is invalid")
            metrics = event.get("metrics")
            if not isinstance(metrics, Mapping):
                raise RecorderContractError("local recorder metric history is invalid")
            self._metric_history.append(
                {"step": cast(int, event["step"]), "metrics": _validate_metrics(cast(Mapping[str, float], metrics))}
            )
        self._metric_events = len(self._metric_history)
        if self._metric_history:
            self._last_step = cast(int, self._metric_history[-1]["step"])
            self._latest_metrics = dict(cast(Mapping[str, float], self._metric_history[-1]["metrics"]))
        try:
            self._state_index = int(latest.name.removeprefix("recorder-state-").removesuffix(".json")) + 1
        except ValueError as error:
            raise RecorderContractError("local recorder state artifact name is invalid") from error

    def _persist_state(self) -> None:
        state = {"params": self._params, "tags": self._tags, "metric_history": self._metric_history}
        if len(_canonical_json(state)) > MAX_CONFIG_BYTES:
            raise RecorderContractError("recorder state exceeds the serialized size bound")
        self._owner.add_json_artifact(
            self._record.run_id,
            state,
            name=f"recorder-state-{self._state_index:08d}.json",
        )
        self._state_index += 1
        self._record = self._owner.get_record(self._record.run_id)

    def _ensure_state_size(
        self,
        *,
        params: Mapping[str, object] | None = None,
        tags: Mapping[str, str] | None = None,
        metric_history: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        state = {
            "params": self._params if params is None else params,
            "tags": self._tags if tags is None else tags,
            "metric_history": self._metric_history if metric_history is None else metric_history,
        }
        if len(_canonical_json(state)) > MAX_CONFIG_BYTES:
            raise RecorderContractError("recorder state exceeds the serialized size bound")

    @property
    def info(self) -> RecorderRunInfo:
        """Return the current lifecycle identity and status for this run handle."""
        record = self._owner.get_record(self._record.run_id)
        self._record = record
        return RecorderRunInfo(
            run_id=record.run_id,
            identity=record.identity,
            name=record.name,
            status=record.status,
            backend=self._owner.backend_name,
            parent_run_id=record.parent_run_ids[0] if record.parent_run_ids else None,
        )

    def _require_running(self) -> None:
        if self.info.status != "running":
            raise RecorderContractError(f"run is already {self.info.status}")

    def log_params(self, params: Mapping[str, object]) -> None:
        """Merge immutable-after-first-write parameters and persist recorder state."""
        self._require_running()
        values = _validate_mapping(params, field="params")
        for key, value in values.items():
            if key in self._params and self._params[key] != value:
                raise RecorderContractError(f"parameter {key!r} cannot change after it is recorded")
        candidate = {**self._params, **values}
        self._ensure_state_size(params=candidate)
        self._params = candidate
        self._persist_state()

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Append non-decreasing step metrics and persist bounded local history."""
        self._require_running()
        if type(step) is not int or step < 0 or step < self._last_step:
            raise RecorderContractError("metric steps must be non-negative and non-decreasing")
        if self._metric_events >= MAX_METRIC_EVENTS:
            raise RecorderContractError("metric event bound exceeded")
        values = _validate_metrics(metrics)
        candidate_history: list[dict[str, object]] = [*self._metric_history, {"step": step, "metrics": values}]
        self._ensure_state_size(metric_history=candidate_history)
        self._last_step = step
        self._metric_events += 1
        self._metric_history = candidate_history
        self._latest_metrics = {**self._latest_metrics, **values}
        self._owner.update_record(self._record.run_id, metrics=self._latest_metrics)
        self._persist_state()

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Merge validated tags into the local run state."""
        self._require_running()
        candidate = {**self._tags, **_validate_tags(tags)}
        self._ensure_state_size(tags=candidate)
        self._tags = candidate
        self._persist_state()

    def log_artifact(
        self,
        content: bytes | bytearray | memoryview | str | Path,
        *,
        name: str,
        media_type: str = "application/octet-stream",
    ) -> RecorderArtifact:
        """Read, safely store, and return metadata for a run artifact."""
        self._require_running()
        data = _read_artifact(content)
        reference = self._owner.add_artifact(self._record.run_id, data, name=name, media_type=media_type)
        return RecorderArtifact(
            name=reference.name,
            digest=reference.digest,
            size_bytes=reference.size_bytes,
            media_type=reference.media_type,
            uri=reference.relative_path,
        )

    def child(self, name: str, *, config: Mapping[str, object] | None = None) -> LocalExperimentRun:
        """Start a child run linked to this still-running local run."""
        self._require_running()
        child = self._owner.start_run(name, config=config, parent_run_id=self._record.run_id)
        current = self._owner.get_record(self._record.run_id)
        if child.info.run_id not in current.child_run_ids:
            self._owner.update_record(
                self._record.run_id,
                child_run_ids=current.child_run_ids + (child.info.run_id,),
            )
        self._record = self._owner.get_record(self._record.run_id)
        return child

    def finish(self) -> RecorderRunInfo:
        """Complete the run with accumulated metrics and return its final info."""
        self._require_running()
        self._owner.complete(self._record.run_id, metrics=self._latest_metrics)
        return self.info

    def fail(self, error: str) -> RecorderRunInfo:
        """Fail the run with a bounded diagnostic and return its final info."""
        self._require_running()
        if not error or len(error) > MAX_STRING_LENGTH:
            raise RecorderContractError("failure diagnostic must be a bounded non-empty string")
        self._owner.fail(self._record.run_id, error)
        return self.info


__all__ = [
    "ExperimentRecorder",
    "ExperimentRun",
    "LocalExperimentRecorder",
    "LocalExperimentRun",
    "MAX_ARTIFACT_BYTES",
    "MAX_CONFIG_BYTES",
    "MAX_METRIC_EVENTS",
    "RecorderArtifact",
    "RecorderContractError",
    "RecorderRunInfo",
    "canonical_recorder_json",
    "compute_recorder_identity",
    "read_recorder_artifact",
    "recorder_artifact_from_bytes",
    "validate_recorder_mapping",
    "validate_recorder_metrics",
    "validate_recorder_artifact_name",
    "validate_recorder_name",
    "validate_recorder_tags",
]
