"""Versioned, local run evidence for reproducible experiments.

The recorder is intentionally small and filesystem-backed.  It stores the
configuration and provenance needed to compare runs without taking ownership
of model code, datasets, or an external tracking service.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypedDict, cast

from latent_anything.reward_value import HoldoutEvaluation, RewardValueEvaluationResult
from latent_anything.runtime.profiling import RuntimeProfile

RUN_RECORD_SCHEMA_VERSION = 1
RunStatus = Literal["running", "completed", "failed", "interrupted"]
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _default_framework_version() -> str:
    try:
        return metadata.version("latent-anything")
    except metadata.PackageNotFoundError:
        return "unknown"


def _freeze_json_value(value: object, *, active: set[int] | None = None) -> object:
    """Return a deeply immutable, JSON-compatible snapshot."""

    active_ids = set() if active is None else active
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in active_ids:
            raise ValueError("run-record inputs must not contain cycles")
        active_ids.add(value_id)
        try:
            frozen = {
                key: _freeze_json_value(item, active=active_ids) for key, item in value.items() if isinstance(key, str)
            }
        finally:
            active_ids.remove(value_id)
        if len(frozen) != len(value):
            raise TypeError("run-record mappings must use string keys")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        value_id = id(value)
        if value_id in active_ids:
            raise ValueError("run-record inputs must not contain cycles")
        active_ids.add(value_id)
        try:
            return tuple(_freeze_json_value(item, active=active_ids) for item in value)
        finally:
            active_ids.remove(value_id)
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError(f"unsupported run-record value: {type(value).__name__}") from error
        if scalar is value:
            raise TypeError(f"unsupported run-record value: {type(value).__name__}")
        return _freeze_json_value(scalar, active=active_ids)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("run-record inputs must contain only finite floats")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported run-record value: {type(value).__name__}")


def _json_value(value: object) -> object:
    """Return a deterministic JSON-compatible representation."""

    frozen = _freeze_json_value(value)
    if isinstance(frozen, Mapping):
        return {key: _json_value(item) for key, item in frozen.items()}
    if isinstance(frozen, tuple):
        return [_json_value(item) for item in frozen]
    return frozen


def _canonical_json(value: object) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _mapping(value: object) -> dict[str, object]:
    converted = _json_value(value)
    return cast(dict[str, object], converted) if isinstance(converted, dict) else {}


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


class _RunCreateKwargs(TypedDict, total=False):
    config: Mapping[str, object] | None
    code_version: str
    framework_version: str
    model_revisions: Mapping[str, str] | None
    dataset_revisions: Mapping[str, str] | None
    seeds: Sequence[int]
    environment: Mapping[str, object] | None
    parent_run_ids: Sequence[str]
    runtime_profile: RuntimeProfile | None
    theory_evidence_ids: Sequence[str]
    metadata: Mapping[str, object] | None
    status: RunStatus


def runtime_profile_metadata(profile: RuntimeProfile | None) -> dict[str, object]:
    """Serialize an existing runtime profile into run metadata."""

    if profile is None:
        return {}
    events = [
        {
            "stage": event.stage,
            "duration_seconds": event.duration_seconds,
            "metadata": dict(event.metadata),
        }
        for event in profile.events
    ]
    return {
        "total_seconds": profile.total_seconds,
        "stage_totals": profile.stage_totals(),
        "events": events,
    }


def compute_run_identity(
    *,
    name: str,
    config: Mapping[str, object],
    code_version: str,
    framework_version: str,
    model_revisions: Mapping[str, str],
    dataset_revisions: Mapping[str, str],
    seeds: Sequence[int],
    environment: Mapping[str, object],
    parent_run_ids: Sequence[str],
    metadata: Mapping[str, object],
) -> str:
    """Hash reproducible inputs while excluding lifecycle timestamps/status."""

    payload = {
        "name": name,
        "config": config,
        "code_version": code_version,
        "framework_version": framework_version,
        "model_revisions": model_revisions,
        "dataset_revisions": dataset_revisions,
        "seeds": list(seeds),
        "environment": environment,
        "parent_run_ids": list(parent_run_ids),
        "metadata": metadata,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    """Content-addressed reference to one artifact stored by a recorder."""

    name: str
    digest: str
    size_bytes: int
    relative_path: str
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("artifact name must not be empty")
        if len(self.digest) != 64 or any(character not in "0123456789abcdef" for character in self.digest):
            raise ValueError("artifact digest must be a lowercase SHA-256 hex digest")
        if self.size_bytes < 0:
            raise ValueError("artifact size must be non-negative")
        if self.relative_path != f"artifacts/{self.digest}":
            raise ValueError("artifact relative_path must be exactly artifacts/<digest>")

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ArtifactRef:
        return cls(
            name=str(value["name"]),
            digest=str(value["digest"]),
            size_bytes=int(str(value["size_bytes"])),
            relative_path=str(value["relative_path"]),
            media_type=str(value.get("media_type", "application/octet-stream")),
        )


@dataclass(frozen=True)
class RunRecord:
    """Schema-v1 record for one reproducible experiment run."""

    run_id: str
    name: str
    status: RunStatus
    created_at: str
    updated_at: str
    config: Mapping[str, object] = field(default_factory=dict)
    code_version: str = ""
    framework_version: str = ""
    model_revisions: Mapping[str, str] = field(default_factory=dict)
    dataset_revisions: Mapping[str, str] = field(default_factory=dict)
    seeds: tuple[int, ...] = ()
    environment: Mapping[str, object] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    parent_run_ids: tuple[str, ...] = ()
    child_run_ids: tuple[str, ...] = ()
    runtime_profile: Mapping[str, object] = field(default_factory=dict)
    theory_evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    error: str | None = None
    schema_version: int = RUN_RECORD_SCHEMA_VERSION
    identity: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != RUN_RECORD_SCHEMA_VERSION:
            raise ValueError(f"unsupported run record schema version: {self.schema_version}")
        if not self.run_id or not self.name:
            raise ValueError("run_id and name must not be empty")
        if self.status not in {"running", "completed", "failed", "interrupted"}:
            raise ValueError(f"unsupported run status: {self.status}")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if any(isinstance(value, bool) or not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("metrics must contain finite numeric values")
        for field_name in (
            "config",
            "model_revisions",
            "dataset_revisions",
            "environment",
            "metrics",
            "runtime_profile",
            "metadata",
        ):
            object.__setattr__(self, field_name, _freeze_json_value(getattr(self, field_name)))
        expected = compute_run_identity(
            name=self.name,
            config=self.config,
            code_version=self.code_version,
            framework_version=self.framework_version,
            model_revisions=self.model_revisions,
            dataset_revisions=self.dataset_revisions,
            seeds=self.seeds,
            environment=self.environment,
            parent_run_ids=self.parent_run_ids,
            metadata=self.metadata,
        )
        if self.identity and self.identity != expected:
            raise ValueError("run identity does not match reproducible run inputs")
        object.__setattr__(self, "identity", expected)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        config: Mapping[str, object] | None = None,
        code_version: str = "",
        framework_version: str = "",
        model_revisions: Mapping[str, str] | None = None,
        dataset_revisions: Mapping[str, str] | None = None,
        seeds: Sequence[int] = (),
        environment: Mapping[str, object] | None = None,
        parent_run_ids: Sequence[str] = (),
        runtime_profile: RuntimeProfile | None = None,
        theory_evidence_ids: Sequence[str] = (),
        metadata: Mapping[str, object] | None = None,
        status: RunStatus = "running",
    ) -> RunRecord:
        """Create a record with a stable identity and UTC lifecycle timestamps."""

        resolved_code_version = code_version or os.environ.get("LATENT_ANYTHING_CODE_VERSION", "working-tree")
        resolved_framework_version = framework_version or _default_framework_version()
        now = _now()
        return cls(
            run_id="pending",
            name=name,
            status=status,
            created_at=now,
            updated_at=now,
            config=dict(config or {}),
            code_version=resolved_code_version,
            framework_version=resolved_framework_version,
            model_revisions=dict(model_revisions or {}),
            dataset_revisions=dict(dataset_revisions or {}),
            seeds=tuple(seeds),
            environment=dict(environment or {}),
            parent_run_ids=tuple(parent_run_ids),
            runtime_profile=runtime_profile_metadata(runtime_profile),
            theory_evidence_ids=tuple(theory_evidence_ids),
            metadata=dict(metadata or {}),
            identity=compute_run_identity(
                name=name,
                config=config or {},
                code_version=resolved_code_version,
                framework_version=resolved_framework_version,
                model_revisions=model_revisions or {},
                dataset_revisions=dataset_revisions or {},
                seeds=seeds,
                environment=environment or {},
                parent_run_ids=parent_run_ids,
                metadata=metadata or {},
            ),
        ).with_run_id()

    def with_run_id(self) -> RunRecord:
        """Replace the temporary constructor id with the identity prefix."""

        return replace(self, run_id=self.identity[:16])

    def transition(
        self,
        status: RunStatus,
        *,
        metrics: Mapping[str, float] | None = None,
        error: str | None = None,
        artifacts: Sequence[ArtifactRef] | None = None,
        child_run_ids: Sequence[str] | None = None,
    ) -> RunRecord:
        """Return a lifecycle-updated copy without changing the identity."""

        return replace(
            self,
            status=status,
            updated_at=_now(),
            metrics=dict(metrics) if metrics is not None else self.metrics,
            error=error,
            artifacts=tuple(artifacts) if artifacts is not None else self.artifacts,
            child_run_ids=tuple(child_run_ids) if child_run_ids is not None else self.child_run_ids,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible schema snapshot."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "identity": self.identity,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": _json_value(self.config),
            "code_version": self.code_version,
            "framework_version": self.framework_version,
            "model_revisions": _json_value(self.model_revisions),
            "dataset_revisions": _json_value(self.dataset_revisions),
            "seeds": list(self.seeds),
            "environment": _json_value(self.environment),
            "metrics": _json_value(self.metrics),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "parent_run_ids": list(self.parent_run_ids),
            "child_run_ids": list(self.child_run_ids),
            "runtime_profile": _json_value(self.runtime_profile),
            "theory_evidence_ids": list(self.theory_evidence_ids),
            "metadata": _json_value(self.metadata),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunRecord:
        """Load schema v1 and migrate the pre-versioned legacy shape."""

        migrated = migrate_run_record(payload)
        artifacts = tuple(
            ArtifactRef.from_dict(item)
            for item in _sequence(migrated.get("artifacts", []))
            if isinstance(item, Mapping)
        )
        status = cast(RunStatus, str(migrated.get("status", "completed")))
        return cls(
            run_id=str(migrated["run_id"]),
            name=str(migrated.get("name", "unnamed")),
            status=status,
            created_at=str(migrated.get("created_at", _now())),
            updated_at=str(migrated.get("updated_at", migrated.get("created_at", _now()))),
            config=_mapping(migrated.get("config", {})),
            code_version=str(migrated.get("code_version", "")),
            framework_version=str(migrated.get("framework_version", "")),
            model_revisions={
                str(key): str(value) for key, value in _mapping(migrated.get("model_revisions", {})).items()
            },
            dataset_revisions={
                str(key): str(value) for key, value in _mapping(migrated.get("dataset_revisions", {})).items()
            },
            seeds=tuple(int(str(seed)) for seed in _sequence(migrated.get("seeds", []))),
            environment=_mapping(migrated.get("environment", {})),
            metrics={str(key): float(str(value)) for key, value in _mapping(migrated.get("metrics", {})).items()},
            artifacts=artifacts,
            parent_run_ids=tuple(str(value) for value in _sequence(migrated.get("parent_run_ids", []))),
            child_run_ids=tuple(str(value) for value in _sequence(migrated.get("child_run_ids", []))),
            runtime_profile=_mapping(migrated.get("runtime_profile", {})),
            theory_evidence_ids=tuple(str(value) for value in _sequence(migrated.get("theory_evidence_ids", []))),
            metadata=_mapping(migrated.get("metadata", {})),
            error=str(migrated["error"]) if migrated.get("error") is not None else None,
            schema_version=int(str(migrated.get("schema_version", RUN_RECORD_SCHEMA_VERSION))),
            identity=str(migrated.get("identity", "")),
        )


def migrate_run_record(payload: Mapping[str, object]) -> dict[str, object]:
    """Migrate a legacy unversioned record into the current schema."""

    result = dict(payload)
    version = int(str(result.get("schema_version", 0)))
    if version > RUN_RECORD_SCHEMA_VERSION:
        raise ValueError(f"run record schema {version} is newer than supported schema {RUN_RECORD_SCHEMA_VERSION}")
    artifacts = result.get("artifacts")
    if version <= RUN_RECORD_SCHEMA_VERSION and isinstance(artifacts, list):
        migrated_artifacts: list[object] = []
        for item in artifacts:
            if isinstance(item, Mapping):
                migrated_item = dict(item)
                digest = migrated_item.get("digest")
                relative_path = migrated_item.get("relative_path")
                if (
                    isinstance(digest, str)
                    and _SHA256_DIGEST.fullmatch(digest) is not None
                    and relative_path == f"artifacts\\{digest}"
                ):
                    migrated_item["relative_path"] = f"artifacts/{digest}"
                migrated_artifacts.append(migrated_item)
            else:
                migrated_artifacts.append(item)
        result["artifacts"] = migrated_artifacts
    result.setdefault("name", result.get("run_name", "legacy-run"))
    result.setdefault("run_id", result.get("id", "legacy"))
    result.setdefault("status", "completed")
    result.setdefault("created_at", _now())
    result.setdefault("updated_at", result["created_at"])
    result.setdefault("model_revisions", {})
    result.setdefault("dataset_revisions", {})
    result.setdefault("seeds", [])
    result.setdefault("environment", {})
    result.setdefault("metrics", {})
    result.setdefault("artifacts", [])
    result.setdefault("parent_run_ids", result.get("parents", []))
    result.setdefault("child_run_ids", result.get("children", []))
    result.setdefault("runtime_profile", {})
    result.setdefault("theory_evidence_ids", [])
    result.setdefault("metadata", {})
    result["schema_version"] = RUN_RECORD_SCHEMA_VERSION
    result.pop("run_name", None)
    result.pop("id", None)
    result.pop("parents", None)
    result.pop("children", None)
    return result


class DuplicateRunError(RuntimeError):
    """Raised when a different reproducible identity occupies a run id."""


class FileSystemRunRecorder:
    """Atomic JSON run store with content-addressed artifact files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.artifacts_dir = self.root / "artifacts"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a simple file name")
        return self.runs_dir / f"{run_id}.json"

    def _atomic_write(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
                temporary = handle.name
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def save(self, record: RunRecord) -> RunRecord:
        """Atomically persist a record, rejecting identity collisions."""

        path = self._run_path(record.run_id)
        if path.exists():
            existing = self.get(record.run_id)
            if existing.identity != record.identity:
                raise DuplicateRunError(f"run id {record.run_id!r} already belongs to another identity")
        self._atomic_write(path, json.dumps(record.to_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n")
        return record

    def start(self, name: str, **kwargs: object) -> RunRecord:
        """Create or reuse a run with the same reproducible identity."""

        record = RunRecord.create(name, **cast(_RunCreateKwargs, kwargs))
        path = self._run_path(record.run_id)
        if path.exists():
            existing = self.get(record.run_id)
            if existing.identity != record.identity:
                raise DuplicateRunError(f"identity collision for run id {record.run_id!r}")
            return existing
        return self.save(record)

    def get(self, run_id: str) -> RunRecord:
        path = self._run_path(run_id)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"run file {path} must contain a JSON object")
        return RunRecord.from_dict(payload)

    def list(self, *, status: RunStatus | None = None) -> tuple[RunRecord, ...]:
        records: list[RunRecord] = []
        for path in sorted(self.runs_dir.glob("*.json")):
            record = self.get(path.stem)
            if status is None or record.status == status:
                records.append(record)
        return tuple(records)

    def update(self, run_id: str, **changes: object) -> RunRecord:
        current = self.get(run_id)
        updated = replace(current, updated_at=_now(), **changes)
        return self.save(updated)

    def complete(
        self,
        run_id: str,
        *,
        metrics: Mapping[str, float] | None = None,
        artifacts: Sequence[ArtifactRef] | None = None,
    ) -> RunRecord:
        record = self.get(run_id).transition("completed", metrics=metrics, artifacts=artifacts)
        return self.save(record)

    def complete_evaluation(
        self,
        run_id: str,
        evaluation: RewardValueEvaluationResult | HoldoutEvaluation,
        *,
        artifact_name: str = "reward_value_evaluation.json",
    ) -> RunRecord:
        """Persist reward/value evidence and complete the associated run.

        The full typed result is stored as a content-addressed JSON artifact;
        its flat metrics are copied into the run record for comparisons.
        """

        self.add_json_artifact(run_id, evaluation.to_dict(), name=artifact_name)
        return self.complete(run_id, metrics=evaluation.to_metrics())

    def fail(self, run_id: str, error: str) -> RunRecord:
        record = self.get(run_id).transition("failed", error=error)
        return self.save(record)

    def recover_interrupted(self) -> tuple[RunRecord, ...]:
        """Mark all unfinished records as interrupted after process restart."""

        recovered: list[RunRecord] = []
        for record in self.list(status="running"):
            recovered_record = record.transition("interrupted", error="recorder recovery: run was still active")
            recovered.append(self.save(recovered_record))
        return tuple(recovered)

    def add_artifact(
        self,
        run_id: str,
        content: bytes | bytearray | memoryview | str | Path,
        *,
        name: str,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        """Store bytes by SHA-256 and attach the reference to a run."""

        data = Path(content).read_bytes() if isinstance(content, (str, Path)) else bytes(content)
        digest = hashlib.sha256(data).hexdigest()
        artifact_path = self.artifacts_dir / digest
        if not artifact_path.exists():
            self._atomic_write(artifact_path, data)
        reference = ArtifactRef(
            name=name,
            digest=digest,
            size_bytes=len(data),
            relative_path=f"artifacts/{digest}",
            media_type=media_type,
        )
        record = self.get(run_id)
        if reference not in record.artifacts:
            self.save(replace(record, artifacts=record.artifacts + (reference,), updated_at=_now()))
        return reference

    def add_json_artifact(self, run_id: str, value: object, *, name: str) -> ArtifactRef:
        return self.add_artifact(run_id, _canonical_json(value) + b"\n", name=name, media_type="application/json")

    def read_artifact(self, reference: ArtifactRef) -> bytes:
        expected_path = self.artifacts_dir / reference.digest
        artifacts_dir = self.artifacts_dir.resolve()
        path = expected_path.resolve()
        if not path.is_relative_to(artifacts_dir):
            raise ValueError("artifact path resolves outside the recorder artifacts directory")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != reference.digest:
            raise ValueError(f"artifact digest mismatch for {reference.relative_path}")
        return data


@dataclass(frozen=True)
class RunComparisonReport:
    """Metric comparison across at least two recorded runs."""

    title: str
    baseline_run_id: str
    runs: tuple[Mapping[str, object], ...]
    metric_deltas: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "title": self.title,
            "baseline_run_id": self.baseline_run_id,
            "runs": [_json_value(run) for run in self.runs],
            "metric_deltas": _json_value(self.metric_deltas),
        }


def build_comparison_report(
    records: Sequence[RunRecord], *, title: str = "Latent Anything run comparison"
) -> RunComparisonReport:
    """Compare metrics and provenance for two or more records."""

    if len(records) < 2:
        raise ValueError("comparison requires at least two run records")
    baseline = records[0]
    baseline_metrics = dict(baseline.metrics)
    deltas: dict[str, dict[str, float]] = {}
    for record in records[1:]:
        deltas[record.run_id] = {
            metric: value - baseline_metrics[metric]
            for metric, value in record.metrics.items()
            if metric in baseline_metrics
        }
    rows: list[Mapping[str, object]] = [
        {
            "run_id": record.run_id,
            "name": record.name,
            "status": record.status,
            "identity": record.identity,
            "model_revisions": dict(record.model_revisions),
            "dataset_revisions": dict(record.dataset_revisions),
            "seeds": list(record.seeds),
            "metrics": dict(record.metrics),
            "theory_evidence_ids": list(record.theory_evidence_ids),
        }
        for record in records
    ]
    return RunComparisonReport(
        title=title,
        baseline_run_id=baseline.run_id,
        runs=tuple(rows),
        metric_deltas=deltas,
    )


__all__ = [
    "ArtifactRef",
    "DuplicateRunError",
    "FileSystemRunRecorder",
    "RUN_RECORD_SCHEMA_VERSION",
    "RunComparisonReport",
    "RunRecord",
    "RunStatus",
    "build_comparison_report",
    "compute_run_identity",
    "migrate_run_record",
    "runtime_profile_metadata",
]
