"""Filesystem persistence and content-addressed artifact lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

from latent_anything._run_record_codec import canonical_json, now
from latent_anything._run_record_schema import ArtifactRef, RunRecord, RunStatus
from latent_anything.adapters.jepa import JEPAEvaluationReport
from latent_anything.artifact_store import ArtifactStore, StoredArtifact
from latent_anything.cem import CEMPlanResult
from latent_anything.mppi import MPPIPlanResult
from latent_anything.reward_value import HoldoutEvaluation, RewardValueEvaluationResult
from latent_anything.runtime.profiling import RuntimeProfile


class RunCreateKwargs(TypedDict, total=False):
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
        record = RunRecord.create(name, **cast(RunCreateKwargs, kwargs))
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
        updated = replace(current, updated_at=now(), **changes)
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
        """Persist reward/value evidence and complete the associated run."""
        self.add_json_artifact(run_id, evaluation.to_dict(), name=artifact_name)
        return self.complete(run_id, metrics=evaluation.to_metrics())

    def complete_cem_plan(
        self,
        run_id: str,
        plan: CEMPlanResult,
        *,
        artifact_name: str = "cem_plan.json",
    ) -> RunRecord:
        """Persist a CEM plan and its optimization/runtime metrics."""
        self.add_json_artifact(run_id, plan.to_dict(), name=artifact_name)
        metrics = {
            "predicted_return": plan.predicted_return,
            "cem_iterations": float(len(plan.candidate_statistics)),
            "planning_seconds": plan.runtime_profile.total_seconds,
        }
        return self.complete(run_id, metrics=metrics)

    def complete_mppi_plan(
        self,
        run_id: str,
        plan: MPPIPlanResult,
        *,
        artifact_name: str = "mppi_plan.json",
    ) -> RunRecord:
        """Persist an MPPI plan and its weighting/runtime metrics."""
        self.add_json_artifact(run_id, plan.to_dict(), name=artifact_name)
        metrics = {
            "predicted_return": plan.predicted_return,
            "mppi_iterations": float(len(plan.candidate_statistics)),
            "mppi_samples": float(plan.sample_count),
            "mppi_effective_sample_size": plan.effective_sample_size,
            "planning_seconds": plan.runtime_profile.total_seconds,
        }
        return self.complete(run_id, metrics=metrics)

    def complete_jepa_evaluation(
        self,
        run_id: str,
        evaluation: JEPAEvaluationReport,
        *,
        artifact_name: str = "jepa_evaluation.json",
    ) -> RunRecord:
        """Persist decoder-free JEPA prediction/rollout evidence."""
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
            self.save(replace(record, artifacts=record.artifacts + (reference,), updated_at=now()))
        return reference

    def add_json_artifact(self, run_id: str, value: object, *, name: str) -> ArtifactRef:
        return self.add_artifact(run_id, canonical_json(value) + b"\n", name=name, media_type="application/json")

    def add_portable_artifact(
        self,
        run_id: str,
        payload: bytes,
        *,
        name: str,
        artifact_type: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef:
        """Store a checksummed portable envelope and attach its content reference."""
        store = ArtifactStore(self.root)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.artifacts_dir, prefix=".portable-", delete=False) as handle:
                temporary_path = Path(handle.name)
            relative_temporary = f"artifacts/{temporary_path.name}"
            store.write(relative_temporary, payload, artifact_type=artifact_type, metadata=dict(metadata or {}))
            envelope = temporary_path.read_bytes()
            digest = hashlib.sha256(envelope).hexdigest()
            target = self.artifacts_dir / digest
            if target.exists():
                if target.read_bytes() != envelope:
                    raise ValueError("portable artifact digest collision")
                temporary_path.unlink(missing_ok=True)
                temporary_path = None
            else:
                os.replace(temporary_path, target)
                temporary_path = None
            reference = ArtifactRef(
                name=name,
                digest=digest,
                size_bytes=len(envelope),
                relative_path=f"artifacts/{digest}",
                media_type="application/vnd.latent-anything.portable",
            )
            record = self.get(run_id)
            if reference not in record.artifacts:
                self.save(replace(record, artifacts=record.artifacts + (reference,), updated_at=now()))
            return reference
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def read_portable_artifact(self, reference: ArtifactRef) -> StoredArtifact:
        """Validate and return a stored portable envelope."""
        self.read_artifact(reference)
        return ArtifactStore(self.root).read(reference.relative_path)

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
