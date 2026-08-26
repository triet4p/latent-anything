"""Private implementation of the frozen run-record schema value objects."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata
from typing import Literal, cast

from latent_anything._run_record_codec import (
    RUN_RECORD_SCHEMA_VERSION,
    compute_run_identity,
    freeze_json_value,
    json_value,
    mapping,
    migrate_run_record,
    now,
    runtime_profile_metadata,
    sequence,
)
from latent_anything.runtime.profiling import RuntimeProfile

RunStatus = Literal["running", "completed", "failed", "interrupted"]


def _default_framework_version() -> str:
    try:
        return metadata.version("latent-anything")
    except metadata.PackageNotFoundError:
        return "unknown"


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
            object.__setattr__(self, field_name, freeze_json_value(getattr(self, field_name)))
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
        current = now()
        return cls(
            run_id="pending",
            name=name,
            status=status,
            created_at=current,
            updated_at=current,
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
            updated_at=now(),
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
            "config": json_value(self.config),
            "code_version": self.code_version,
            "framework_version": self.framework_version,
            "model_revisions": json_value(self.model_revisions),
            "dataset_revisions": json_value(self.dataset_revisions),
            "seeds": list(self.seeds),
            "environment": json_value(self.environment),
            "metrics": json_value(self.metrics),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "parent_run_ids": list(self.parent_run_ids),
            "child_run_ids": list(self.child_run_ids),
            "runtime_profile": json_value(self.runtime_profile),
            "theory_evidence_ids": list(self.theory_evidence_ids),
            "metadata": json_value(self.metadata),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunRecord:
        """Load schema v1 and migrate the pre-versioned legacy shape."""
        migrated = migrate_run_record(payload)
        artifacts = tuple(
            ArtifactRef.from_dict(item) for item in sequence(migrated.get("artifacts", [])) if isinstance(item, Mapping)
        )
        status = cast(RunStatus, str(migrated.get("status", "completed")))
        return cls(
            run_id=str(migrated["run_id"]),
            name=str(migrated.get("name", "unnamed")),
            status=status,
            created_at=str(migrated.get("created_at", now())),
            updated_at=str(migrated.get("updated_at", migrated.get("created_at", now()))),
            config=mapping(migrated.get("config", {})),
            code_version=str(migrated.get("code_version", "")),
            framework_version=str(migrated.get("framework_version", "")),
            model_revisions={
                str(key): str(value) for key, value in mapping(migrated.get("model_revisions", {})).items()
            },
            dataset_revisions={
                str(key): str(value) for key, value in mapping(migrated.get("dataset_revisions", {})).items()
            },
            seeds=tuple(int(str(seed)) for seed in sequence(migrated.get("seeds", []))),
            environment=mapping(migrated.get("environment", {})),
            metrics={str(key): float(str(value)) for key, value in mapping(migrated.get("metrics", {})).items()},
            artifacts=artifacts,
            parent_run_ids=tuple(str(value) for value in sequence(migrated.get("parent_run_ids", []))),
            child_run_ids=tuple(str(value) for value in sequence(migrated.get("child_run_ids", []))),
            runtime_profile=mapping(migrated.get("runtime_profile", {})),
            theory_evidence_ids=tuple(str(value) for value in sequence(migrated.get("theory_evidence_ids", []))),
            metadata=mapping(migrated.get("metadata", {})),
            error=str(migrated["error"]) if migrated.get("error") is not None else None,
            schema_version=int(str(migrated.get("schema_version", RUN_RECORD_SCHEMA_VERSION))),
            identity=str(migrated.get("identity", "")),
        )
