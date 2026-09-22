"""Architecture-neutral diagnostic workflow state machine (Sprint 80.8).

Coordinates ``capture -> detect -> localize -> explain -> intervene -> compare
-> report`` in exact order by invoking caller-supplied stage executors. The
coordinator owns ordering, typed stage contracts, deterministic identity, and
resumable artifact boundaries only. It embeds no detector, explainer, or
intervention algorithm and contains no model-family branches.

Declaration reuses the reviewed 80.6 surface (:class:`DiagnosticRequest` /
:class:`DiagnosticResult`) and the frozen manifest contract
(:func:`validate_manifest`, :func:`manifest_digest`). Lifecycle maps onto
:attr:`DiagnosticResult.status`, ``completed_stages``, ``stage_results``, and
``artifact_refs``. No new top-level export is added; consumers import this
private module directly, mirroring ``_capture_binding`` and
``_diagnostic_validator``.

Non-goals: detector/explainer/intervention algorithms (80.9+), real benchmark
execution, artifact persistence (80.21), and report rendering (80.22).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, cast

from latent_anything._benchmark_manifest import (
    BenchmarkManifestValidationError,
    manifest_digest,
    validate_manifest,
)
from latent_anything._portable_contract import PortableNodeError, canonical_json
from latent_anything.diagnostics import DiagnosticRequest, DiagnosticResult

WORKFLOW_STAGES: tuple[str, ...] = (
    "capture",
    "detect",
    "localize",
    "explain",
    "intervene",
    "compare",
    "report",
)
"""Exact execution order. Single source for the coordinator; must match ``diagnostics``."""

StageOutcome = Literal["completed", "not_applicable", "unsupported"]
"""Honest per-stage outcomes. Non-applicable stages are recorded, never promoted."""

_STAGE_OUTCOMES: frozenset[str] = frozenset({"completed", "not_applicable", "unsupported"})
_CHECKPOINT_STATUSES: frozenset[str] = frozenset({"running", "completed", "failed"})


class WorkflowError(ValueError):
    """Raised for invalid workflow inputs, configs, or checkpoints (fail-closed)."""


class StageContractError(WorkflowError):
    """Raised when a stage executor returns a contract-violating output."""


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"{name} must be a non-empty string")
    return value


def _sha256_text(payload: object) -> str:
    try:
        text = canonical_json(payload)
    except PortableNodeError as exc:
        raise WorkflowError(f"workflow payload is not canonical JSON: {exc}") from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _request_digest(request: DiagnosticRequest) -> str:
    return _sha256_text(request.to_dict())


def _manifest_digest(manifest: Mapping[str, object]) -> str:
    try:
        return manifest_digest(manifest)
    except (BenchmarkManifestValidationError, ValueError, TypeError, KeyError) as exc:
        raise WorkflowError(f"invalid benchmark manifest: {exc}") from exc


def _config_digest(versions: Mapping[str, str]) -> str:
    return _sha256_text({stage: versions[stage] for stage in sorted(versions)})


def _workflow_identity(*, request_digest: str, manifest_digest_value: str, config_digest: str) -> str:
    return _sha256_text(
        {"config": config_digest, "manifest": manifest_digest_value, "request": request_digest}
    )


def _stage_digest(
    *,
    workflow_identity: str,
    stage: str,
    outcome: str,
    payload: Mapping[str, object],
    artifact_refs: Sequence[str],
) -> str:
    return _sha256_text(
        {
            "artifacts": list(artifact_refs),
            "outcome": outcome,
            "payload": dict(payload),
            "stage": stage,
            "workflow": workflow_identity,
        }
    )


def _freeze_payload(payload: Mapping[str, object], *, name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise StageContractError(f"{name} must be a mapping")
    for key in payload:
        if not isinstance(key, str):
            raise StageContractError(f"{name} must use string keys")
    try:
        canonical_json(dict(payload))
    except PortableNodeError as exc:
        raise StageContractError(f"{name} is not canonical JSON: {exc}") from exc
    return MappingProxyType(dict(payload))


def _freeze_refs(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise StageContractError(f"{name} must be a list of strings")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise StageContractError(f"{name} must be a list of non-empty strings")
    if len(set(items)) != len(items):
        raise StageContractError(f"{name} must not contain duplicates")
    return items


@dataclass(frozen=True)
class StageOutput:
    """One executor-produced stage result with an explicit honest outcome."""

    stage: str
    outcome: StageOutcome
    payload: Mapping[str, object]
    artifact_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in WORKFLOW_STAGES:
            raise StageContractError(f"unknown workflow stage: {self.stage!r}")
        if self.outcome not in _STAGE_OUTCOMES:
            raise StageContractError(f"unsupported stage outcome: {self.outcome!r}")
        object.__setattr__(self, "payload", _freeze_payload(self.payload, name=f"{self.stage}.payload"))
        object.__setattr__(self, "artifact_refs", _freeze_refs(self.artifact_refs, name=f"{self.stage}.artifact_refs"))

    def digest(self, workflow_identity: str) -> str:
        """Return the deterministic digest binding this output to one workflow run."""
        _non_empty_string(workflow_identity, name="workflow_identity")
        return _stage_digest(
            workflow_identity=workflow_identity,
            stage=self.stage,
            outcome=self.outcome,
            payload=self.payload,
            artifact_refs=self.artifact_refs,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this stage output."""
        return {
            "artifact_refs": list(self.artifact_refs),
            "outcome": self.outcome,
            "payload": dict(self.payload),
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StageOutput:
        """Rebuild a stage output, failing closed on malformed input."""
        if not isinstance(value, Mapping):
            raise WorkflowError("stage output must be a mapping")
        if set(value) != {"artifact_refs", "outcome", "payload", "stage"}:
            raise WorkflowError("stage output fields are invalid")
        payload = value["payload"]
        if not isinstance(payload, Mapping):
            raise WorkflowError("stage output payload must be a mapping")
        refs = value["artifact_refs"]
        if isinstance(refs, str | bytes) or not isinstance(refs, Sequence):
            raise WorkflowError("stage output artifact_refs must be a list")
        try:
            return cls(
                stage=_non_empty_string(value["stage"], name="stage"),
                outcome=cast(StageOutcome, _non_empty_string(value["outcome"], name="outcome")),
                payload=dict(cast(Mapping[str, object], payload)),
                artifact_refs=tuple(refs),
            )
        except StageContractError as exc:
            raise WorkflowError(f"invalid stage output: {exc}") from exc


@dataclass(frozen=True)
class StageInvocation:
    """Typed input handed to exactly one stage executor."""

    stage: str
    request: DiagnosticRequest
    manifest: Mapping[str, object]
    prior: tuple[StageOutput, ...]
    workflow_identity: str
    request_digest: str
    manifest_digest: str
    config_digest: str

    def __post_init__(self) -> None:
        if self.stage not in WORKFLOW_STAGES:
            raise WorkflowError(f"unknown workflow stage: {self.stage!r}")
        if not isinstance(self.request, DiagnosticRequest):
            raise WorkflowError("invocation request must be a DiagnosticRequest")
        if not isinstance(self.manifest, Mapping):
            raise WorkflowError("invocation manifest must be a mapping")
        expected_prior = WORKFLOW_STAGES[: WORKFLOW_STAGES.index(self.stage)]
        if tuple(item.stage for item in self.prior) != expected_prior:
            raise WorkflowError(f"invocation prior outputs must be exactly {list(expected_prior)!r} in order")
        for item in self.prior:
            if not isinstance(item, StageOutput):
                raise WorkflowError("invocation prior outputs must be StageOutput items")
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(self, "prior", tuple(self.prior))
        for name in ("workflow_identity", "request_digest", "manifest_digest", "config_digest"):
            _non_empty_string(getattr(self, name), name=name)


class StageExecutor(Protocol):
    """Caller-supplied stage behavior. Method algorithms live here, never in the coordinator."""

    def __call__(self, invocation: StageInvocation) -> StageOutput:
        """Run one stage from its typed invocation and return its typed output."""
        ...


@dataclass(frozen=True)
class FailureInfo:
    """Honest failure metadata for the stage that stopped a run."""

    stage: str
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if self.stage not in WORKFLOW_STAGES:
            raise WorkflowError(f"unknown workflow stage: {self.stage!r}")
        _non_empty_string(self.error_type, name="failure error_type")
        if not isinstance(self.message, str) or not self.message.strip():
            raise WorkflowError("failure message must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this failure."""
        return {"error_type": self.error_type, "message": self.message, "stage": self.stage}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FailureInfo:
        """Rebuild failure metadata, failing closed on malformed input."""
        if not isinstance(value, Mapping):
            raise WorkflowError("failure must be a mapping")
        if set(value) != {"error_type", "message", "stage"}:
            raise WorkflowError("failure fields are invalid")
        message = value["message"]
        if not isinstance(message, str):
            raise WorkflowError("failure message must be a string")
        return cls(
            stage=_non_empty_string(value["stage"], name="failure stage"),
            error_type=_non_empty_string(value["error_type"], name="failure error_type"),
            message=message,
        )


@dataclass(frozen=True)
class WorkflowCheckpoint:
    """Resumable artifact boundary for one workflow run."""

    request_id: str
    manifest_id: str
    request_digest: str
    manifest_digest: str
    config_digest: str
    workflow_identity: str
    completed_stages: tuple[str, ...]
    outputs: tuple[StageOutput, ...]
    output_digests: tuple[str, ...]
    artifact_refs: tuple[str, ...]
    next_stage: str | None
    status: str
    failure: FailureInfo | None

    def __post_init__(self) -> None:
        _non_empty_string(self.request_id, name="request_id")
        _non_empty_string(self.manifest_id, name="manifest_id")
        for name in ("request_digest", "manifest_digest", "config_digest", "workflow_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise WorkflowError(f"{name} must be a 64-character hex digest")
        if self.status not in _CHECKPOINT_STATUSES:
            raise WorkflowError(f"unsupported checkpoint status: {self.status!r}")
        completed = tuple(self.completed_stages)
        if tuple(completed) != WORKFLOW_STAGES[: len(completed)]:
            raise WorkflowError(f"checkpoint stages must be an exact prefix in order, got {list(completed)!r}")
        if len(set(completed)) != len(completed):
            raise WorkflowError("checkpoint stages must not contain duplicates")
        object.__setattr__(self, "completed_stages", completed)
        outputs = tuple(self.outputs)
        for item in outputs:
            if not isinstance(item, StageOutput):
                raise WorkflowError("checkpoint outputs must be StageOutput items")
        object.__setattr__(self, "outputs", outputs)
        if tuple(item.stage for item in outputs) != completed:
            raise WorkflowError("checkpoint outputs must align exactly with completed stages")
        digests = tuple(self.output_digests)
        if len(digests) != len(outputs):
            raise WorkflowError("checkpoint digests must align with outputs")
        for digest in digests:
            if not isinstance(digest, str) or len(digest) != 64:
                raise WorkflowError("checkpoint output digests must be 64-character hex digests")
        for item, digest in zip(outputs, digests, strict=True):
            if item.digest(self.workflow_identity) != digest:
                raise WorkflowError(f"checkpoint digest mismatch for stage {item.stage!r}")
        object.__setattr__(self, "output_digests", digests)
        refs = _freeze_refs(self.artifact_refs, name="checkpoint artifact_refs")
        expected_refs: list[str] = [ref for item in outputs for ref in item.artifact_refs]
        if list(refs) != expected_refs:
            raise WorkflowError("checkpoint artifact refs must concatenate stage outputs in order")
        object.__setattr__(self, "artifact_refs", refs)
        if self.status == "completed":
            if len(completed) != len(WORKFLOW_STAGES) or self.next_stage is not None or self.failure is not None:
                raise WorkflowError("completed checkpoints must cover all stages with no next stage or failure")
        elif self.status == "running":
            if len(completed) >= len(WORKFLOW_STAGES) or self.failure is not None:
                raise WorkflowError("running checkpoints must have a remaining stage and no failure")
            if self.next_stage != WORKFLOW_STAGES[len(completed)]:
                raise WorkflowError("running checkpoint next stage must follow completed stages")
        else:
            if not isinstance(self.failure, FailureInfo):
                raise WorkflowError("failed checkpoints must carry failure metadata")
            if self.next_stage != WORKFLOW_STAGES[len(completed)]:
                raise WorkflowError("failed checkpoint next stage must be the failed stage")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of this checkpoint."""
        return {
            "artifact_refs": list(self.artifact_refs),
            "completed_stages": list(self.completed_stages),
            "config_digest": self.config_digest,
            "failure": None if self.failure is None else self.failure.to_dict(),
            "manifest_digest": self.manifest_digest,
            "manifest_id": self.manifest_id,
            "next_stage": self.next_stage,
            "output_digests": list(self.output_digests),
            "outputs": [item.to_dict() for item in self.outputs],
            "request_digest": self.request_digest,
            "request_id": self.request_id,
            "status": self.status,
            "workflow_identity": self.workflow_identity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> WorkflowCheckpoint:
        """Rebuild a checkpoint, failing closed on malformed input."""
        if not isinstance(value, Mapping):
            raise WorkflowError("checkpoint must be a mapping")
        expected = {
            "artifact_refs",
            "completed_stages",
            "config_digest",
            "failure",
            "manifest_digest",
            "manifest_id",
            "next_stage",
            "output_digests",
            "outputs",
            "request_digest",
            "request_id",
            "status",
            "workflow_identity",
        }
        if set(value) != expected:
            raise WorkflowError("checkpoint fields are invalid")
        raw_outputs = value["outputs"]
        raw_stages = value["completed_stages"]
        raw_digests = value["output_digests"]
        raw_refs = value["artifact_refs"]
        if (
            isinstance(raw_outputs, str | bytes)
            or not isinstance(raw_outputs, Sequence)
            or isinstance(raw_stages, str | bytes)
            or not isinstance(raw_stages, Sequence)
            or isinstance(raw_digests, str | bytes)
            or not isinstance(raw_digests, Sequence)
            or isinstance(raw_refs, str | bytes)
            or not isinstance(raw_refs, Sequence)
        ):
            raise WorkflowError("checkpoint sequences are invalid")
        next_stage = value["next_stage"]
        if next_stage is not None and not isinstance(next_stage, str):
            raise WorkflowError("checkpoint next_stage must be a stage name or null")
        raw_failure = value["failure"]
        if raw_failure is not None and not isinstance(raw_failure, Mapping):
            raise WorkflowError("checkpoint failure must be an object or null")
        return cls(
            request_id=_non_empty_string(value["request_id"], name="request_id"),
            manifest_id=_non_empty_string(value["manifest_id"], name="manifest_id"),
            request_digest=_non_empty_string(value["request_digest"], name="request_digest"),
            manifest_digest=_non_empty_string(value["manifest_digest"], name="manifest_digest"),
            config_digest=_non_empty_string(value["config_digest"], name="config_digest"),
            workflow_identity=_non_empty_string(value["workflow_identity"], name="workflow_identity"),
            completed_stages=tuple(raw_stages),
            outputs=tuple(
                StageOutput.from_dict(cast(Mapping[str, object], item)) for item in raw_outputs
            ),
            output_digests=tuple(raw_digests),
            artifact_refs=tuple(raw_refs),
            next_stage=next_stage,
            status=_non_empty_string(value["status"], name="status"),
            failure=None if raw_failure is None else FailureInfo.from_dict(cast(Mapping[str, object], raw_failure)),
        )


class DiagnosticWorkflow:
    """Method-agnostic coordinator enforcing exact stage order and resume identity."""

    def __init__(
        self,
        executors: Mapping[str, Callable[[StageInvocation], StageOutput]],
        versions: Mapping[str, str],
    ) -> None:
        if not isinstance(executors, Mapping) or not isinstance(versions, Mapping):
            raise WorkflowError("executors and versions must be mappings")
        if set(executors) != set(WORKFLOW_STAGES):
            raise WorkflowError(f"executors must cover exactly {list(WORKFLOW_STAGES)!r}")
        if set(versions) != set(WORKFLOW_STAGES):
            raise WorkflowError(f"versions must cover exactly {list(WORKFLOW_STAGES)!r}")
        for stage in WORKFLOW_STAGES:
            if not callable(executors[stage]):
                raise WorkflowError(f"executor for {stage!r} must be callable")
            _non_empty_string(versions[stage], name=f"version for {stage!r}")
        self._executors: dict[str, Callable[[StageInvocation], StageOutput]] = dict(executors)
        self._versions: dict[str, str] = {stage: versions[stage] for stage in WORKFLOW_STAGES}
        self._config_digest = _config_digest(self._versions)

    @property
    def stages(self) -> tuple[str, ...]:
        """Return the exact execution order."""
        return WORKFLOW_STAGES

    @property
    def versions(self) -> dict[str, str]:
        """Return the stage executor versions bound into the config digest."""
        return dict(self._versions)

    @property
    def config_digest(self) -> str:
        """Return the deterministic digest of the executor configuration."""
        return self._config_digest

    def run(
        self,
        request: DiagnosticRequest,
        manifest: Mapping[str, object],
        *,
        stop_after: str | None = None,
    ) -> tuple[DiagnosticResult, WorkflowCheckpoint]:
        """Run from the first stage, optionally stopping early at a resumable boundary."""
        identities = self._checked_identities(request, manifest)
        stop = self._checked_stop(stop_after)
        return self._execute(request, manifest, prior=(), digests=(), stop_after=stop, identities=identities)

    def resume(
        self,
        checkpoint: WorkflowCheckpoint,
        request: DiagnosticRequest,
        manifest: Mapping[str, object],
        *,
        stop_after: str | None = None,
        restart: bool = False,
    ) -> tuple[DiagnosticResult, WorkflowCheckpoint]:
        """Resume after a running checkpoint without rerunning completed stages."""
        if not isinstance(checkpoint, WorkflowCheckpoint):
            raise WorkflowError("checkpoint must be a WorkflowCheckpoint")
        if not isinstance(restart, bool):
            raise WorkflowError("restart must be boolean")
        identities = self._checked_identities(request, manifest)
        stop = self._checked_stop(stop_after)
        self._verified_checkpoint(checkpoint, identities, request=request, manifest=manifest)
        if checkpoint.status != "running" and not restart:
            raise WorkflowError(f"checkpoint with status {checkpoint.status!r} requires restart=True")
        if restart:
            return self._execute(request, manifest, prior=(), digests=(), stop_after=stop, identities=identities)
        if stop is not None and WORKFLOW_STAGES.index(stop) < len(checkpoint.completed_stages):
            raise WorkflowError("stop_after must not precede the resumed position")
        return self._execute(
            request,
            manifest,
            prior=checkpoint.outputs,
            digests=checkpoint.output_digests,
            stop_after=stop,
            identities=identities,
        )

    def _checked_identities(
        self, request: DiagnosticRequest, manifest: Mapping[str, object]
    ) -> dict[str, str]:
        if not isinstance(request, DiagnosticRequest):
            raise WorkflowError("request must be a DiagnosticRequest")
        if not isinstance(manifest, Mapping):
            raise WorkflowError("manifest must be a mapping")
        try:
            validate_manifest(manifest)
        except BenchmarkManifestValidationError as exc:
            raise WorkflowError(f"invalid benchmark manifest: {exc}") from exc
        manifest_id = manifest.get("manifest_id")
        if manifest_id != request.manifest_id:
            raise WorkflowError(
                f"request manifest {request.manifest_id!r} != manifest {manifest_id!r}"
            )
        request_digest = _request_digest(request)
        manifest_digest_value = _manifest_digest(manifest)
        return {
            "config_digest": self._config_digest,
            "manifest_digest": manifest_digest_value,
            "request_digest": request_digest,
            "workflow_identity": _workflow_identity(
                request_digest=request_digest,
                manifest_digest_value=manifest_digest_value,
                config_digest=self._config_digest,
            ),
        }

    def _verified_checkpoint(
        self,
        checkpoint: WorkflowCheckpoint,
        identities: Mapping[str, str],
        *,
        request: DiagnosticRequest,
        manifest: Mapping[str, object],
    ) -> None:
        if checkpoint.request_id != request.request_id:
            raise WorkflowError("checkpoint request_id does not match the request")
        manifest_id = manifest.get("manifest_id")
        if checkpoint.manifest_id != manifest_id or checkpoint.manifest_id != request.manifest_id:
            raise WorkflowError("checkpoint manifest_id does not match the request and manifest")
        for name in ("request_digest", "manifest_digest", "config_digest", "workflow_identity"):
            if getattr(checkpoint, name) != identities[name]:
                raise WorkflowError(f"checkpoint {name} does not match current inputs")

    @staticmethod
    def _checked_stop(stop_after: str | None) -> str | None:
        if stop_after is None:
            return None
        if stop_after not in WORKFLOW_STAGES:
            raise WorkflowError(f"unknown stop_after stage: {stop_after!r}")
        return stop_after

    def _execute(
        self,
        request: DiagnosticRequest,
        manifest: Mapping[str, object],
        *,
        prior: tuple[StageOutput, ...],
        digests: tuple[str, ...],
        stop_after: str | None,
        identities: Mapping[str, str],
    ) -> tuple[DiagnosticResult, WorkflowCheckpoint]:
        outputs: list[StageOutput] = list(prior)
        output_digests: list[str] = list(digests)
        workflow_identity = identities["workflow_identity"]
        for stage in WORKFLOW_STAGES[len(outputs) :]:
            invocation = StageInvocation(
                stage=stage,
                request=request,
                manifest=manifest,
                prior=tuple(outputs),
                workflow_identity=workflow_identity,
                request_digest=identities["request_digest"],
                manifest_digest=identities["manifest_digest"],
                config_digest=identities["config_digest"],
            )
            try:
                produced = self._executors[stage](invocation)
            except StageContractError as exc:
                return self._failed(request, manifest, outputs, output_digests, identities, stage, type(exc).__name__, str(exc))
            except Exception as exc:  # noqa: BLE001 - stage failures become honest failed state
                return self._failed(request, manifest, outputs, output_digests, identities, stage, type(exc).__name__, str(exc))
            if not isinstance(produced, StageOutput):
                return self._failed(
                    request,
                    manifest,
                    outputs,
                    output_digests,
                    identities,
                    stage,
                    "StageContractError",
                    "executor must return a StageOutput",
                )
            if produced.stage != stage:
                return self._failed(
                    request,
                    manifest,
                    outputs,
                    output_digests,
                    identities,
                    stage,
                    "StageContractError",
                    f"executor returned stage {produced.stage!r} for {stage!r}",
                )
            outputs.append(produced)
            output_digests.append(produced.digest(workflow_identity))
            if stop_after == stage:
                break
        if len(outputs) < len(WORKFLOW_STAGES):
            return self._partial(request, manifest, outputs, output_digests, identities)
        return self._completed(request, manifest, outputs, output_digests, identities)

    def _stage_results(
        self,
        outputs: Sequence[StageOutput],
        digests: Sequence[str],
        identities: Mapping[str, str],
        failure: FailureInfo | None,
    ) -> dict[str, object]:
        stages: dict[str, object] = {}
        for item, digest in zip(outputs, digests, strict=True):
            stages[item.stage] = {
                "artifact_refs": list(item.artifact_refs),
                "outcome": item.outcome,
                "output_digest": digest,
                "payload": dict(item.payload),
            }
        return {
            "config_digest": identities["config_digest"],
            "failure": None if failure is None else failure.to_dict(),
            "manifest_digest": identities["manifest_digest"],
            "request_digest": identities["request_digest"],
            "stages": stages,
            "workflow_identity": identities["workflow_identity"],
        }

    def _refs(self, outputs: Sequence[StageOutput]) -> tuple[str, ...]:
        return tuple(ref for item in outputs for ref in item.artifact_refs)

    def _checkpoint(
        self,
        request: DiagnosticRequest,
        manifest: Mapping[str, object],
        outputs: Sequence[StageOutput],
        digests: Sequence[str],
        identities: Mapping[str, str],
        *,
        status: str,
        failure: FailureInfo | None,
    ) -> WorkflowCheckpoint:
        manifest_id = cast(str, cast(Mapping[str, object], manifest).get("manifest_id"))
        completed = tuple(item.stage for item in outputs)
        if status == "completed":
            next_stage: str | None = None
        else:
            next_stage = WORKFLOW_STAGES[len(completed)]
        return WorkflowCheckpoint(
            request_id=request.request_id,
            manifest_id=manifest_id,
            request_digest=identities["request_digest"],
            manifest_digest=identities["manifest_digest"],
            config_digest=identities["config_digest"],
            workflow_identity=identities["workflow_identity"],
            completed_stages=completed,
            outputs=tuple(outputs),
            output_digests=tuple(digests),
            artifact_refs=self._refs(outputs),
            next_stage=next_stage,
            status=status,
            failure=failure,
        )

    def _completed(
        self,
        request: DiagnosticRequest,
        manifest: Mapping[str, object],
        outputs: Sequence[StageOutput],
        digests: Sequence[str],
        identities: Mapping[str, str],
    ) -> tuple[DiagnosticResult, WorkflowCheckpoint]:
        checkpoint = self._checkpoint(
            request, manifest, outputs, digests, identities, status="completed", failure=None
        )
        report_id: str | None = None
        report = next((item for item in outputs if item.stage == "report"), None)
        if report is not None and report.outcome == "completed":
            candidate = report.payload.get("report_id")
            if isinstance(candidate, str) and candidate.strip():
                report_id = candidate
        result = DiagnosticResult(
            request_id=request.request_id,
            manifest_id=request.manifest_id,
            status="completed",
            completed_stages=tuple(item.stage for item in outputs),
            artifact_refs=self._refs(outputs),
            report_id=report_id,
            message=f"workflow completed ({len(outputs)} stages)",
            stage_results=self._stage_results(outputs, digests, identities, None),
        )
        return result, checkpoint

    def _partial(
        self,
        request: DiagnosticRequest,
        manifest: Mapping[str, object],
        outputs: Sequence[StageOutput],
        digests: Sequence[str],
        identities: Mapping[str, str],
    ) -> tuple[DiagnosticResult, WorkflowCheckpoint]:
        checkpoint = self._checkpoint(
            request, manifest, outputs, digests, identities, status="running", failure=None
        )
        last = outputs[-1].stage if outputs else "none"
        result = DiagnosticResult(
            request_id=request.request_id,
            manifest_id=request.manifest_id,
            status="running",
            completed_stages=tuple(item.stage for item in outputs),
            artifact_refs=self._refs(outputs),
            report_id=None,
            message=f"workflow stopped after {last}; next {checkpoint.next_stage}",
            stage_results=self._stage_results(outputs, digests, identities, None),
        )
        return result, checkpoint

    def _failed(
        self,
        request: DiagnosticRequest,
        manifest: Mapping[str, object],
        outputs: Sequence[StageOutput],
        digests: Sequence[str],
        identities: Mapping[str, str],
        stage: str,
        error_type: str,
        message: str,
    ) -> tuple[DiagnosticResult, WorkflowCheckpoint]:
        detail = message.strip() or "stage executor failed"
        failure = FailureInfo(stage=stage, error_type=error_type or "StageError", message=detail)
        checkpoint = self._checkpoint(
            request, manifest, outputs, digests, identities, status="failed", failure=failure
        )
        result = DiagnosticResult(
            request_id=request.request_id,
            manifest_id=request.manifest_id,
            status="failed",
            completed_stages=tuple(item.stage for item in outputs),
            artifact_refs=self._refs(outputs),
            report_id=None,
            message=f"stage {stage} failed ({failure.error_type}): {failure.message}",
            stage_results=self._stage_results(outputs, digests, identities, failure),
        )
        return result, checkpoint

    def __call__(
        self,
        request: DiagnosticRequest,
        manifest: Mapping[str, object],
        *,
        stop_after: str | None = None,
    ) -> tuple[DiagnosticResult, WorkflowCheckpoint]:
        """Run from the first stage; keeps the coordinator usable as a supplied callable."""
        return self.run(request, manifest, stop_after=stop_after)


__all__ = [
    "WORKFLOW_STAGES",
    "DiagnosticWorkflow",
    "FailureInfo",
    "StageExecutor",
    "StageInvocation",
    "StageOutcome",
    "StageOutput",
    "WorkflowCheckpoint",
    "WorkflowError",
    "StageContractError",
]
