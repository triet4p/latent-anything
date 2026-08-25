"""Small, validated experiment-recorder contract and local adapter.

The local filesystem recorder remains the source of truth for run identity and
content-addressed artifacts.  This module adds only the common lifecycle and
logging surface that optional tracking adapters can implement without exposing
their SDK types through the public API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol, cast, runtime_checkable

from latent_anything.run_record import (
    ArtifactRef,
    FileSystemRunRecorder,
    RunRecord,
    RunStatus,
    compute_run_identity,
)

MAX_MAPPING_ENTRIES = 256
MAX_KEY_LENGTH = 128
MAX_STRING_LENGTH = 4096
MAX_CONFIG_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_METRIC_EVENTS = 4096
MAX_TAGS = 128
MAX_SEQUENCE_ITEMS = 4096
MAX_NESTING_DEPTH = 16
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_KEY = re.compile(
    r"(?:secret|token|password|passwd|api[_-]?key|access[_-]?key|private[_-]?key|credential|authorization|auth)"
)
_SENSITIVE_VALUE = re.compile(
    r"(?:^sk-[A-Za-z0-9]|^gh[pousr]_[A-Za-z0-9]|^xox[baprs]-|^bearer\s|BEGIN [A-Z ]*PRIVATE KEY)",
    re.I,
)


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


def _normalize_json(value: object, *, active: set[int] | None = None, depth: int = 0) -> object:
    """Return bounded canonical-JSON input without accepting object code."""

    if depth > MAX_NESTING_DEPTH:
        raise RecorderContractError("recorder values exceed the nesting bound")
    active_ids = set() if active is None else active
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ENTRIES:
            raise RecorderContractError("recorder mappings exceed the entry bound")
        value_id = id(value)
        if value_id in active_ids:
            raise RecorderContractError("recorder values must not contain cycles")
        active_ids.add(value_id)
        try:
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key or len(key) > MAX_KEY_LENGTH:
                    raise RecorderContractError("recorder mapping keys must be bounded non-empty strings")
                _reject_sensitive_key(key)
                result[key] = _normalize_json(item, active=active_ids, depth=depth + 1)
        finally:
            active_ids.remove(value_id)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise RecorderContractError("recorder sequences exceed the entry bound")
        value_id = id(value)
        if value_id in active_ids:
            raise RecorderContractError("recorder values must not contain cycles")
        active_ids.add(value_id)
        try:
            return [_normalize_json(item, active=active_ids, depth=depth + 1) for item in value]
        finally:
            active_ids.remove(value_id)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
            raise RecorderContractError("recorder strings exceed the size bound")
        if isinstance(value, str) and _SENSITIVE_VALUE.search(value):
            raise RecorderContractError("recorder values must not contain secret-like material")
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise RecorderContractError("recorder values must contain finite floats")
        return value
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is value:
            raise RecorderContractError(f"unsupported recorder value: {type(value).__name__}")
        return _normalize_json(scalar, active=active_ids, depth=depth + 1)
    raise RecorderContractError(f"unsupported recorder value: {type(value).__name__}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(_normalize_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _validate_name(value: str, *, field: str = "name") -> str:
    if not value or len(value) > MAX_STRING_LENGTH:
        raise RecorderContractError(f"{field} must be a non-empty bounded string")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise RecorderContractError(f"{field} must not contain path separators")
    return value


def _validate_artifact_name(name: str) -> str:
    if type(name) is not str or not name or len(name) > MAX_STRING_LENGTH:
        raise RecorderContractError("artifact name must be a non-empty bounded string")
    if "\\" in name or "\x00" in name or "%" in name or ":" in name:
        raise RecorderContractError("artifact name must use canonical POSIX separators")
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "/".join(path.parts) != name
    ):
        raise RecorderContractError("artifact name must be a safe relative POSIX path")
    return name


def safe_recorder_artifact_path(root: str | Path, name: str) -> Path:
    """Resolve a validated artifact name below a local temporary root."""

    safe_name = _validate_artifact_name(name)
    if _has_reparse_component(Path(root)):
        raise RecorderContractError("artifact root must not be a symlink or reparse point")
    try:
        canonical_root = Path(root).resolve(strict=True)
    except OSError as error:
        raise RecorderContractError("artifact root cannot be resolved safely") from error
    target = canonical_root.joinpath(*safe_name.split("/"))
    try:
        resolved_target = target.resolve(strict=False)
    except OSError as error:
        raise RecorderContractError("artifact path cannot be resolved safely") from error
    try:
        resolved_target.relative_to(canonical_root)
    except ValueError as error:
        raise RecorderContractError("artifact path escapes its temporary root") from error
    current = canonical_root
    for part in safe_name.split("/"):
        current /= part
        try:
            info = os.stat(current, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RecorderContractError("artifact path cannot be inspected safely") from error
        if stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400):
            raise RecorderContractError("artifact path must not traverse symlinks or reparse points")
    return target


def _validate_mapping(value: Mapping[str, object] | None, *, field: str) -> dict[str, object]:
    if value is None:
        return {}
    if len(value) > MAX_MAPPING_ENTRIES:
        raise RecorderContractError(f"{field} exceeds the entry bound")
    result: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str or not key or len(key) > MAX_KEY_LENGTH:
            raise RecorderContractError(f"{field} keys must be bounded non-empty strings")
        _reject_sensitive_key(key)
        result[key] = _normalize_json(item)
    if len(_canonical_json(result)) > MAX_CONFIG_BYTES:
        raise RecorderContractError(f"{field} exceeds the serialized size bound")
    return result


def _validate_tags(tags: Mapping[str, str] | None) -> dict[str, str]:
    if tags is None:
        return {}
    if len(tags) > MAX_TAGS:
        raise RecorderContractError("tags exceed the entry bound")
    result: dict[str, str] = {}
    for key, value in tags.items():
        if type(key) is not str or not key or len(key) > MAX_KEY_LENGTH:
            raise RecorderContractError("tag keys must be bounded non-empty strings")
        _reject_sensitive_key(key)
        if type(value) is not str or len(value) > MAX_STRING_LENGTH:
            raise RecorderContractError("tag values must be bounded strings")
        if _SENSITIVE_VALUE.search(value):
            raise RecorderContractError("tag values must not contain secret-like material")
        result[key] = value
    if len(_canonical_json(result)) > MAX_CONFIG_BYTES:
        raise RecorderContractError("tags exceed the serialized size bound")
    return result


def _validate_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    if len(metrics) > MAX_MAPPING_ENTRIES:
        raise RecorderContractError("metrics exceed the entry bound")
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if type(key) is not str or not key or len(key) > MAX_KEY_LENGTH:
            raise RecorderContractError("metric keys must be bounded non-empty strings")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise RecorderContractError("metrics must contain finite numeric values") from error
        if type(value) is bool or not math.isfinite(numeric_value):
            raise RecorderContractError("metrics must contain finite numeric values")
        result[key] = numeric_value
    return result


def _reject_sensitive_key(key: str) -> None:
    if _SENSITIVE_KEY.search(key.lower().replace("-", "_")):
        raise RecorderContractError("recorder keys must not contain secret-like names")


def _validate_string_mapping(value: Mapping[str, str] | None, *, field: str) -> dict[str, str]:
    if value is None:
        return {}
    validated = _validate_mapping(cast(Mapping[str, object], value), field=field)
    result: dict[str, str] = {}
    for key, item in validated.items():
        if not isinstance(item, str):
            raise RecorderContractError(f"{field} values must be bounded strings")
        result[key] = item
    return result


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if len(seeds) > MAX_SEQUENCE_ITEMS:
        raise RecorderContractError("seeds exceed the entry bound")
    result: list[int] = []
    for seed in seeds:
        if type(seed) is bool or type(seed) is not int or seed < 0:
            raise RecorderContractError("seeds must be non-negative integers")
        result.append(seed)
    return tuple(result)


def _read_artifact(content: bytes | bytearray | memoryview | str | Path) -> bytes:
    if type(content) is bytes:
        if len(content) > MAX_ARTIFACT_BYTES:
            raise RecorderContractError("artifact exceeds the size bound")
        return content
    if type(content) is bytearray:
        if len(content) > MAX_ARTIFACT_BYTES:
            raise RecorderContractError("artifact exceeds the size bound")
        return bytes(content)
    if isinstance(content, memoryview):
        if content.nbytes > MAX_ARTIFACT_BYTES:
            raise RecorderContractError("artifact exceeds the size bound")
        return content.tobytes()
    if isinstance(content, (str, Path)):
        return _read_artifact_path(Path(content))
    raise RecorderContractError("artifact content must be bytes, bytearray, memoryview, or a path")


def _read_artifact_path(path: Path) -> bytes:
    if _has_reparse_component(path):
        raise RecorderContractError("artifact path must not traverse symlinks or reparse points")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RecorderContractError("artifact path cannot be opened safely") from error
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_ARTIFACT_BYTES:
                raise RecorderContractError("artifact must be a bounded regular file")
            data = handle.read(MAX_ARTIFACT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except RecorderContractError:
        raise
    except OSError as error:
        raise RecorderContractError("artifact path could not be read safely") from error
    if len(data) > MAX_ARTIFACT_BYTES:
        raise RecorderContractError("artifact exceeds the size bound")
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RecorderContractError("artifact changed while it was being read")
    return data


def _has_reparse_component(path: Path) -> bool:
    candidate = path if path.is_absolute() else (Path.cwd() / path)
    current = Path(candidate.anchor) if candidate.anchor else Path()
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for part in parts:
        current /= part
        try:
            info = os.stat(current, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RecorderContractError("artifact path cannot be inspected safely") from error
        if stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400):
            return True
    return False


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
        return self._recorder.get(run_id)

    def update_record(self, run_id: str, **changes: object) -> RunRecord:
        return self._recorder.update(run_id, **changes)

    def add_artifact(
        self,
        run_id: str,
        content: bytes,
        *,
        name: str,
        media_type: str,
    ) -> ArtifactRef:
        return self._recorder.add_artifact(run_id, content, name=name, media_type=media_type)

    def add_json_artifact(self, run_id: str, value: object, *, name: str) -> ArtifactRef:
        return self._recorder.add_json_artifact(run_id, value, name=name)

    def read_artifact(self, reference: ArtifactRef) -> bytes:
        return self._recorder.read_artifact(reference)

    def complete(self, run_id: str, *, metrics: Mapping[str, float]) -> RunRecord:
        return self._recorder.complete(run_id, metrics=metrics)

    def fail(self, run_id: str, error: str) -> RunRecord:
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
        self._require_running()
        self._owner.complete(self._record.run_id, metrics=self._latest_metrics)
        return self.info

    def fail(self, error: str) -> RecorderRunInfo:
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
