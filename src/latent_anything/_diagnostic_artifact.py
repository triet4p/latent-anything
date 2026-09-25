"""Content-addressed diagnostic artifact assembly and persistence
(Sprint 80.21).

One architecture-neutral private seam turns a completed diagnostic
workflow (request, frozen manifest, result, checkpoint) plus the
structured report payload into a self-contained, content-addressed run
artifact, reusing the run-record lane (``ArtifactRef``,
``FileSystemRunRecorder`` atomic writes, ``compute_run_identity``), the
frozen report schema (80.3), the independent validator (80.4), and the
provisional evidence references produced by80.16–80.20.

Assembly (pure, deterministic, timestamp-free):
- rejects incomplete/nonterminal workflows: the result and checkpoint
  must both be ``completed`` and cover exactly ``WORKFLOW_STAGES`` with
  agreeing stage outcomes and payloads (stage-chain disagreement fails
  closed before anything is written);
- registers evidence where the frozen report schema expects it, without
  fabricating evidence or rendering prose: the capture provenance block
  is rebuilt from the real bound capture record (its provisional
  artifact references are remapped everywhere and rejected outright when
  a provisional name collides with a registered evidence or stage name),
  interventions-section rows are registered for every recorded trial
  (caller-supplied rows must be unique, backed by a real registered
  trial record, and agree with it — extra, duplicate, or mismatched rows
  fail closed before any write, and caller evidence references are
  preserved alongside the registered record blob),
  (``intervention-<id>-record`` rows whose evidence is the persisted
  stage record bytes), comparison rows register their
  ``comparison-<id>-record`` evidence from the compare-stage records,
  explanation references register the real hypothesis rows, and
  provisional claim control references are replaced by the executed
  manifest controls with recorded outcomes;
- derives the validator inputs from real stage evidence: family evidence
  and control outcomes from the detect-stage payload, applicability from
  the declared families, and artifact digests from the actual persisted
  bytes;
- runs the independent validator (80.4) and persists its ``passed``
  result bound to the report and input digests; a failing report is
  rejected before any write.

Persistence:
- preflight (workflow completeness/agreement checks, evidence
  registration, independent validation, blob-conflict scan) fails before
  any write: no run record and no files are created;
- every stage record, the registered report, and every evidence record
  are written as canonical-JSON bytes through
  ``FileSystemRunRecorder.add_artifact`` (atomic temp-file + rename,
  SHA-256 file name, ``artifacts/<digest>`` relative paths). Existing
  blobs with matching bytes are reused (idempotent); existing blobs with
  conflicting bytes are rejected before any write;
- the run record is created as ``running`` after preflight, all blobs
  and the document are written, and only then does it transition
  atomically to ``completed``. A mid-write exception therefore leaves a
  recoverable non-completed record — never one surfaced as completed —
  and a retry with identical inputs reuses the deterministic identity,
  skips matching blobs, and completes safely; an already completed run
  is never downgraded or rewritten;
- the artifact document itself is canonical JSON with no timestamps,
  randomness, absolute paths, machine secrets, or cwd dependence, so
  identical input yields an identical digest and identical bytes; the
  run record (timestamped lifecycle file, excluded from identity) is
  addressed by ``identity[:16]`` and stores the artifact digest in
  metadata, giving idempotent re-persists and deterministic lookup.

Loading (fresh process/root):
- uses only persisted files under the root plus the declared external
  manifest: verifies schema, manifest/taxonomy/report-schema hashes,
  request round-trip, run-metadata agreement, the full stage chain
  (order, outcomes, ``StageOutput.digest`` per stage), the report digest,
  every blob's existence/hash/path containment (recorder ``read_artifact``
  rejects path escapes and symlink escapes), inventory/digest agreement,
  and re-runs the independent validator — any validator-result,
  report, provenance, or stage-chain disagreement fails closed.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np

from latent_anything._benchmark_manifest import manifest_digest
from latent_anything._diagnostic_report import load_schema, validate_report_shape
from latent_anything._diagnostic_validator import ReportValidationError, validate_diagnostic_report
from latent_anything._diagnostic_workflow import WORKFLOW_STAGES, StageOutput, WorkflowCheckpoint
from latent_anything._representation_taxonomy import load_taxonomy
from latent_anything._run_record_codec import canonical_json
from latent_anything._run_record_persistence import FileSystemRunRecorder
from latent_anything._run_record_schema import ArtifactRef
from latent_anything._target_evidence import EVIDENCE_CONTRACT_V1, EVIDENCE_CONTRACT_V2
from latent_anything.diagnostics import DiagnosticRequest, DiagnosticResult

DIAGNOSTIC_ARTIFACT_SCHEMA = "diagnostic-artifact-v1"
"""Frozen schema identity for the persisted diagnostic artifact document."""

_VALIDATOR_ID = "diagnostic-report-validator"
_ALLOWED_CONTROL_STATUSES = frozenset({"passed", "failed"})
_DIGEST_LENGTH = 64
_REPORT_SECTION_IDS = (
    "symptoms",
    "localization",
    "hypotheses",
    "statistical_evidence",
    "interventions",
    "comparisons",
    "claims",
)


class DiagnosticArtifactError(ValueError):
    """Raised when a diagnostic artifact cannot be assembled, persisted, or loaded."""


def _require_request(value: object) -> DiagnosticRequest:
    if not isinstance(value, DiagnosticRequest):
        raise DiagnosticArtifactError("request must be a DiagnosticRequest")
    return value


def _require_result(value: object) -> DiagnosticResult:
    if not isinstance(value, DiagnosticResult):
        raise DiagnosticArtifactError("result must be a DiagnosticResult")
    return value


def _require_checkpoint(value: object) -> WorkflowCheckpoint:
    if not isinstance(value, WorkflowCheckpoint):
        raise DiagnosticArtifactError("checkpoint must be a WorkflowCheckpoint")
    return value


def _require_manifest(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DiagnosticArtifactError("manifest must be a mapping")
    return value


def _require_stage_entries(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DiagnosticArtifactError("artifact must declare its stage chain")
    entries: list[Mapping[str, object]] = []
    for item in value:
        entries.append(_require_mapping(item, name="stage chain entries must be objects"))
    return tuple(entries)


def _require_blob(value: object, digest: str, *, name: str) -> bytes:
    if not isinstance(value, bytes) or _digest_of(value) != digest:
        raise DiagnosticArtifactError(f"{name} blob is missing or mis-hashed")
    return value


def _require_digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH:
        raise DiagnosticArtifactError("run record does not reference a diagnostic artifact digest")
    return value


def _require_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DiagnosticArtifactError("run record does not declare the artifact size")
    return int(value)


def _require_run_id(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise DiagnosticArtifactError(f"invalid run id: {value!r}")
    return value


def _require_family_evidence(value: object) -> Mapping[str, Mapping[str, str]]:
    if not isinstance(value, Mapping):
        raise DiagnosticArtifactError("detect family_evidence must be a mapping")
    for family_id, status in value.items():
        if not isinstance(family_id, str) or not isinstance(status, Mapping):
            raise DiagnosticArtifactError("detect family_evidence must map families to evidence mappings")
        for key, item in status.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise DiagnosticArtifactError("detect family_evidence must map families to string mappings")
    return value


def _require_str_mapping(value: object, *, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise DiagnosticArtifactError(f"{name} must be a mapping")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise DiagnosticArtifactError(f"{name} must map strings to strings")
    return value


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DiagnosticArtifactError(f"{name} must be a mapping")
    return value


def _require_outputs(value: object) -> tuple[StageOutput, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DiagnosticArtifactError("checkpoint outputs must be StageOutput items")
    items = tuple(value)
    for item in items:
        if not isinstance(item, StageOutput):
            raise DiagnosticArtifactError("checkpoint outputs must be StageOutput items")
    return items


def _require_digests(value: object, *, count: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DiagnosticArtifactError("checkpoint output digests must be digests")
    items = tuple(value)
    if len(items) != count:
        raise DiagnosticArtifactError("checkpoint digests must align with outputs")
    for item in items:
        if not isinstance(item, str):
            raise DiagnosticArtifactError("checkpoint output digests must be digests")
    return items


def _require_stage_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DiagnosticArtifactError(f"{name} completed_stages must be a stage tuple")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise DiagnosticArtifactError(f"{name} completed_stages must be a stage tuple")
    return items


def _require_refs(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DiagnosticArtifactError(f"{name} must be a list of strings")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise DiagnosticArtifactError(f"{name} must be a list of strings")
    return items


def _require_non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DiagnosticArtifactError(f"{name} must be a non-empty string")
    return value


def _require_sequence(value: object, *, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DiagnosticArtifactError(f"{name} must be a sequence")
    return tuple(value)


def _require_axes(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise DiagnosticArtifactError("bound capture must declare axes")
    return tuple(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticArtifactError(message)


def _canonical_bytes(value: object) -> bytes:
    try:
        return canonical_json(value) + b"\n"
    except Exception as exc:  # noqa: BLE001 - any serialization gap fails closed
        raise DiagnosticArtifactError(f"value is not canonical JSON: {exc}") from exc


def _bounded_environment() -> dict[str, str]:
    """Bounded, secret-free environment metadata that is stable per machine."""
    return {
        "numpy": str(np.__version__),
        "python": platform.python_version(),
        "system": f"{platform.system()}-{platform.machine()}",
    }


def _digest_of(data: bytes) -> str:
    return sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Workflow input validation (incomplete/nonterminal results reject)
# ---------------------------------------------------------------------------


def _stage_records(
    request: object,
    result: object,
    checkpoint: object,
) -> list[dict[str, object]]:
    request = _require_request(request)
    result = _require_result(result)
    checkpoint = _require_checkpoint(checkpoint)
    _require(result.status == "completed", "diagnostic result must be completed")
    _require(checkpoint.status == "completed", "workflow checkpoint must be completed")
    _require(result.request_id == request.request_id, "result request_id does not match the request")
    _require(
        result.manifest_id == request.manifest_id,
        "result manifest_id does not match the request",
    )
    result_stages = _require_stage_tuple(result.completed_stages, name="diagnostic result")
    checkpoint_stages = _require_stage_tuple(checkpoint.completed_stages, name="workflow checkpoint")
    _require(
        result_stages == checkpoint_stages == tuple(WORKFLOW_STAGES),
        f"diagnostic result must cover exactly {list(WORKFLOW_STAGES)!r}",
    )
    stage_results = _require_mapping(result.stage_results, name="stage_results")
    _require(stage_results.get("failure") is None, "diagnostic result records a failure")
    rows = _require_mapping(stage_results.get("stages"), name="stages")
    outputs = _require_outputs(checkpoint.outputs)
    digests = _require_digests(checkpoint.output_digests, count=len(outputs))

    records: list[dict[str, object]] = []
    for output, digest in zip(outputs, digests, strict=True):
        row = _require_mapping(rows.get(output.stage), name=f"stage {output.stage!r}")
        _require(
            row.get("outcome") == output.outcome,
            f"stage disagreement for {output.stage!r}: result outcome "
            f"{row.get('outcome')!r} != checkpoint outcome {output.outcome!r}",
        )
        payload_bytes = _canonical_bytes(row.get("payload"))
        stage_bytes = _canonical_bytes(output.to_dict())
        _require(
            payload_bytes == _canonical_bytes(output.payload),
            f"stage payload disagreement for {output.stage!r}",
        )
        _require(
            output.digest(checkpoint.workflow_identity) == digest,
            f"stage digest disagreement for {output.stage!r}",
        )
        output_refs = _require_refs(output.artifact_refs, name=f"stage {output.stage!r} artifact_refs")
        records.append(
            {
                "artifact_refs": list(output_refs),
                "outcome": output.outcome,
                "output_digest": digest,
                "record_digest": _digest_of(stage_bytes),
                "stage": output.stage,
            }
        )
    _require(
        [record["stage"] for record in records] == list(WORKFLOW_STAGES),
        "checkpoint stage chain must follow the workflow stage order",
    )
    return records


# ---------------------------------------------------------------------------
# Evidence registration (provisional refs become registered records/blobs)
# ---------------------------------------------------------------------------


def _section_rows(report: Mapping[str, object], section: str) -> list[dict[str, object]]:
    value = report.get(section)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DiagnosticArtifactError(f"report section {section!r} must be a list")
    rows: list[dict[str, object]] = []
    for item in value:
        rows.append(dict(_require_mapping(item, name=f"{section} row")))
    return rows


def _register_evidence(
    report: Mapping[str, object],
    *,
    request: DiagnosticRequest,
    manifest: Mapping[str, object],
    payloads: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], list[tuple[str, bytes]], dict[str, object]]:
    """Return the registered report, evidence blobs, and validator inputs."""
    try:
        registered: dict[str, object] = json.loads(_canonical_bytes(report).decode("utf-8"))
    except DiagnosticArtifactError:
        raise
    except Exception as exc:  # noqa: BLE001 - any serialization gap fails closed
        raise DiagnosticArtifactError(f"report is not canonical JSON: {exc}") from exc
    validate_report_shape(registered)

    evidence: list[tuple[str, bytes]] = []
    evidence_names: set[str] = set()

    def _add_evidence(name: str, value: object) -> None:
        _require(name not in evidence_names, f"duplicate evidence reference {name!r}")
        evidence_names.add(name)
        evidence.append((name, _canonical_bytes(value)))

    # 1. Capture provenance is rebuilt from the real bound capture record;
    #    the caller's provisional capture artifact references are remapped.
    capture_payload = _require_mapping(
        payloads.get("capture"), name="registering evidence requires a real capture stage payload"
    )
    bound_capture = _require_mapping(
        capture_payload.get("capture"),
        name="registering evidence requires a real bound capture stage payload",
    )
    model = _require_mapping(manifest.get("model"), name="manifest must declare model")
    dataset = _require_mapping(manifest.get("dataset"), name="manifest must declare dataset")
    representation = _require_mapping(manifest.get("representation"), name="manifest must declare representation")
    _require(
        bound_capture.get("capture_id") == request.capture.capture_id,
        "bound capture_id disagrees with the request capture selection",
    )
    _require(
        bound_capture.get("representation_identity") == request.capture.representation_identity,
        "bound representation disagrees with the request capture selection",
    )
    _require(
        bound_capture.get("manifest_id") == request.manifest_id,
        "bound capture manifest disagrees with the request",
    )
    old_capture_refs: set[str] = set()
    provenance = _require_mapping(registered.get("capture_provenance"), name="report must declare capture_provenance")
    captures = provenance.get("captures")
    if isinstance(captures, Sequence) and not isinstance(captures, str | bytes):
        for raw in captures:
            if isinstance(raw, Mapping):
                refs = raw.get("artifact_refs")
                if isinstance(refs, Sequence) and not isinstance(refs, str | bytes):
                    old_capture_refs.update(str(item) for item in refs)
    capture_ref = f"capture-{request.capture.capture_id}-record"
    axes = _require_axes(bound_capture.get("axes"))
    _add_evidence(capture_ref, dict(capture_payload))
    registered["capture_provenance"] = {
        "captures": [
            {
                "artifact_refs": [capture_ref],
                "axes": [str(item) for item in axes],
                "capture_id": str(bound_capture.get("capture_id")),
                "dataset_split": str(cast(Mapping[str, object], dataset).get("split_identity")),
                "model_revision": str(cast(Mapping[str, object], model).get("revision")),
                "representation_identity": str(cast(Mapping[str, object], representation).get("identity")),
            }
        ]
    }
    # 1b. Target-label provenance (v2 evidence contract): when the detect
    #    payload carries the real evaluated target provenance, build and
    #    validate the content-addressed target record, bind it against the
    #    report's declared target block, and register it as evidence. When
    #    absent, evidence_contract stays v1 and legacy behavior is unchanged.
    detect_payload_for_target = payloads.get("detect")
    target_block_raw = registered.get("target_evidence")
    evidence_contract = EVIDENCE_CONTRACT_V1
    if target_block_raw is not None:
        from latent_anything._target_evidence import (
            TARGET_EVIDENCE_SCHEMA_VERSION,
            TargetEvidenceError,
            link_for_record,
            validate_target_record,
        )

        provenance = _require_mapping(
            cast(Mapping[str, object], detect_payload_for_target).get("target_provenance"),
            name="detect payload must carry target_provenance for the v2 evidence contract",
        )
        declared_block = _require_mapping(target_block_raw, name="report must declare target_evidence for v2")
        try:
            candidate: dict[str, object] = {
                "capacity": str(provenance.get("capacity")),
                "dataset_split_identity": str(cast(Mapping[str, object], dataset).get("split_identity")),
                "eval_indices": list(cast(Sequence[object], provenance.get("eval_indices"))),
                "eval_split_identity": str(provenance.get("eval_split_identity")),
                "label_digest": str(provenance.get("label_digest")),
                "labels": list(cast(Sequence[object], provenance.get("labels"))),
                "representation_identity": str(cast(Mapping[str, object], representation).get("identity")),
                "rule": str(provenance.get("rule")),
                "rule_digest": _digest_of(str(provenance.get("rule")).encode("utf-8")),
                "rule_kind": str(provenance.get("rule_kind")),
                "sample_digest": str(provenance.get("sample_digest")),
                "sample_ids": list(cast(Sequence[object], provenance.get("sample_ids"))),
                "schema_version": TARGET_EVIDENCE_SCHEMA_VERSION,
                "target_id": str(provenance.get("target_id")),
                "train_indices": list(cast(Sequence[object], provenance.get("train_indices"))),
                "train_split_identity": str(provenance.get("train_split_identity")),
            }
            validate_target_record(candidate)
        except TargetEvidenceError as exc:
            raise DiagnosticArtifactError(f"target evidence failed validation; nothing was persisted: {exc}") from exc
        for key in ("rule", "rule_digest", "rule_kind", "target_id"):
            _require(
                str(declared_block.get(key)) == str(candidate[key]),
                f"declared target {key} disagrees with evaluated detect-stage provenance",
            )
        _require(
            str(declared_block.get("label_digest")) == str(provenance.get("label_digest")),
            "declared target labels disagree with the evaluated detect payload",
        )
        _require(
            str(declared_block.get("sample_digest")) == str(provenance.get("sample_digest")),
            "declared target sample identities disagree with the evaluated detect payload",
        )
        link = link_for_record(candidate)
        _require(
            str(declared_block.get("record_digest")) == str(link["record_digest"]),
            "declared target record digest disagrees with the evaluated target evidence",
        )
        target_ref = f"target-{str(candidate['target_id'])}-record"
        _add_evidence(target_ref, dict(candidate))
        registered["target_evidence"] = {
            "evidence_refs": [target_ref],
            "label_digest": str(candidate["label_digest"]),
            "record_digest": str(link["record_digest"]),
            "rule": str(candidate["rule"]),
            "rule_digest": str(candidate["rule_digest"]),
            "rule_kind": str(candidate["rule_kind"]),
            "sample_digest": str(candidate["sample_digest"]),
            "target_id": str(candidate["target_id"]),
        }
        evidence_contract = EVIDENCE_CONTRACT_V2
    if evidence_contract == EVIDENCE_CONTRACT_V2:
        registered["evidence_contract"] = evidence_contract
    else:
        registered.pop("target_evidence", None)
        registered.pop("evidence_contract", None)
    # 2. Interventions-section rows are registered for every recorded trial.
    #    Caller-supplied rows are never dropped silently: each must be
    #    unique, backed by a real registered trial record, and agree with
    #    it; extra, duplicate, or mismatched rows fail closed before any
    #    write. Caller evidence references are preserved and augmented
    #    with the registered record blob.
    caller_rows: dict[str, Mapping[str, object]] = {}
    registered_intervention_ids: set[str] = set()
    for row in _section_rows(registered, "interventions"):
        row_id = str(row.get("id") or "")
        _require(bool(row_id), "interventions rows must declare an id")
        _require(
            row_id not in caller_rows,
            f"duplicate interventions row id {row_id!r}",
        )
        caller_rows[row_id] = row
    registered_rows: dict[str, dict[str, object]] = {}
    intervene_payload = payloads.get("intervene")
    if isinstance(intervene_payload, Mapping):
        trials = intervene_payload.get("trials")
        if isinstance(trials, Sequence) and not isinstance(trials, str | bytes):
            for raw in trials:
                _require(isinstance(raw, Mapping), "intervene trials must be objects")
                trial = cast(Mapping[str, object], raw)
                trial_id = trial.get("intervention_id")
                _require(
                    isinstance(trial_id, str) and bool(trial_id),
                    "intervene trials must declare intervention_id",
                )
                conclusion = trial.get("conclusion")
                _require(
                    isinstance(conclusion, str) and bool(conclusion),
                    f"intervene trial {trial_id!r} must declare a conclusion",
                )
                row_id = f"intervention-{trial_id}-record"
                blob_name = f"stage-record-intervene-{trial_id}"
                _add_evidence(blob_name, dict(trial))
                registered_intervention_ids.add(row_id)
                caller_refs: list[str] = []
                caller_row = caller_rows.get(row_id)
                if caller_row is not None:
                    _require(
                        str(caller_row.get("status")) == conclusion,
                        f"interventions row {row_id!r} status "
                        f"{caller_row.get('status')!r} disagrees with the recorded trial "
                        f"conclusion {conclusion!r}",
                    )
                    _require(
                        str(caller_row.get("target")) == str(trial.get("target")),
                        f"interventions row {row_id!r} target disagrees with the recorded trial",
                    )
                    _require(
                        str(caller_row.get("intervention")) == str(trial.get("kind")),
                        f"interventions row {row_id!r} intervention kind disagrees with the recorded trial",
                    )
                    raw_refs = caller_row.get("evidence_refs")
                    if isinstance(raw_refs, Sequence) and not isinstance(raw_refs, str | bytes):
                        for item in raw_refs:
                            _require(
                                isinstance(item, str) and bool(item),
                                f"interventions row {row_id!r} evidence_refs must be non-empty strings",
                            )
                            caller_refs.append(str(item))
                merged_refs = list(caller_refs)
                if blob_name not in merged_refs:
                    merged_refs.append(blob_name)
                registered_rows[row_id] = {
                    "control_refs": [],
                    "evidence_refs": merged_refs,
                    "id": row_id,
                    "intervention": str(trial.get("kind")),
                    "outcome": {
                        "conclusion": conclusion,
                        "record_digest": _digest_of(_canonical_bytes(dict(trial))),
                    },
                    "status": conclusion,
                    "target": str(trial.get("target")),
                }
    for row_id in caller_rows:
        _require(
            row_id in registered_intervention_ids,
            f"interventions row {row_id!r} is not backed by a registered trial record",
        )
    if registered_rows:
        registered["interventions"] = [registered_rows[key] for key in sorted(registered_rows)]

    # 3. Comparison rows register their provisional record references.
    compare_payload = payloads.get("compare")
    comparison_records: dict[str, Mapping[str, object]] = {}
    if isinstance(compare_payload, Mapping):
        comparisons = compare_payload.get("comparisons")
        if isinstance(comparisons, Sequence) and not isinstance(comparisons, str | bytes):
            for raw in comparisons:
                if isinstance(raw, Mapping):
                    comparison_id = raw.get("comparison_id")
                    if isinstance(comparison_id, str) and comparison_id:
                        comparison_records[comparison_id] = raw
    comparison_rows = _section_rows(registered, "comparisons")
    for row in comparison_rows:
        row_id = str(row.get("id"))
        refs = row.get("evidence_refs")
        if not (isinstance(refs, Sequence) and not isinstance(refs, str | bytes)):
            continue
        expected_ref = f"comparison-{row_id}-record"
        if expected_ref not in set(str(item) for item in refs):
            continue
        record = _require_mapping(
            comparison_records.get(row_id),
            name=f"comparison row {row_id!r} has no recorded compare-stage record",
        )
        _add_evidence(expected_ref, dict(record))

    # 4. Explanation references register the real hypothesis rows.
    explain_payload = payloads.get("explain")
    hypothesis_rows: dict[str, Mapping[str, object]] = {}
    if isinstance(explain_payload, Mapping):
        hypotheses = explain_payload.get("hypotheses")
        if isinstance(hypotheses, Sequence) and not isinstance(hypotheses, str | bytes):
            for raw in hypotheses:
                if isinstance(raw, Mapping):
                    hypothesis_id = raw.get("hypothesis_id")
                    if isinstance(hypothesis_id, str) and hypothesis_id:
                        hypothesis_rows[hypothesis_id] = raw

    remap = {old: capture_ref for old in old_capture_refs if old != capture_ref}

    def _remap_refs(values: Sequence[object]) -> list[str]:
        return [remap.get(str(value), str(value)) for value in values]

    # 5. Claim registration: remap capture references, replace provisional
    #    control references with the executed manifest controls, and require
    #    causal claims to point at a registered intervention row.
    detect_evidence = _detect_validator_evidence(payloads)
    control_outcomes = _require_str_mapping(detect_evidence["control_outcomes"], name="control_outcomes")
    executed_controls = sorted(control_outcomes)
    _require(bool(executed_controls), "no executed manifest control outcomes were recorded")

    registered_claims: list[dict[str, object]] = []
    for raw_claim in _section_rows(registered, "claims"):
        claim = dict(raw_claim)
        refs = claim.get("evidence_refs")
        if isinstance(refs, Sequence) and not isinstance(refs, str | bytes):
            resolved_refs = _remap_refs(refs)
            for reference in resolved_refs:
                if reference.startswith("explanation-") and reference.endswith("-record"):
                    hypothesis_id = reference[len("explanation-") : -len("-record")]
                    hypothesis = _require_mapping(
                        hypothesis_rows.get(hypothesis_id),
                        name=f"explanation reference {reference!r} has no recorded explain-stage row",
                    )
                    _add_evidence(reference, dict(hypothesis))
            claim["evidence_refs"] = resolved_refs
        control_refs = claim.get("control_refs")
        declared_controls = (
            [str(item) for item in control_refs]
            if isinstance(control_refs, Sequence) and not isinstance(control_refs, str | bytes)
            else []
        )
        if not set(declared_controls).issubset(set(control_outcomes)):
            claim["control_refs"] = list(executed_controls)
        if claim.get("kind") == "causal_result":
            claim_refs = claim.get("evidence_refs")
            claim_ref_list = (
                [str(item) for item in claim_refs]
                if isinstance(claim_refs, Sequence) and not isinstance(claim_refs, str | bytes)
                else []
            )
            _require(
                bool(registered_intervention_ids.intersection(claim_ref_list)),
                f"causal claim {claim.get('id')!r} does not reference a registered intervention record",
            )
        registered_claims.append(claim)
    registered["claims"] = registered_claims

    # Remap capture references in every other section as well.
    for section in _REPORT_SECTION_IDS:
        if section == "claims":
            continue
        rows = _section_rows(registered, section)
        updated: list[dict[str, object]] = []
        for row in rows:
            refs = row.get("evidence_refs")
            if isinstance(refs, Sequence) and not isinstance(refs, str | bytes):
                row = dict(row)
                row["evidence_refs"] = _remap_refs(refs)
            updated.append(row)
        registered[section] = updated

    family_evidence = _require_family_evidence(detect_evidence["family_evidence"])
    validator_input = {
        "applicability": detect_evidence["applicability"],
        "artifact_digests": {},  # filled from the blob inventory after assembly
        "control_outcomes": dict(control_outcomes),
        "family_evidence": {family_id: dict(status) for family_id, status in family_evidence.items()},
    }
    # F3: a provisional capture reference that collides with any registered
    # evidence or stage name fails closed instead of silently retargeting
    # unrelated evidence to the capture record.
    reserved_names = (
        set(evidence_names) | {f"stage-record-{stage}" for stage in WORKFLOW_STAGES} | {"diagnostic-report"}
    )
    for old in sorted(old_capture_refs):
        if old == capture_ref:
            continue
        _require(
            old not in reserved_names,
            f"provisional capture reference {old!r} collides with a registered evidence name",
        )
    validate_report_shape(registered)
    return registered, evidence, validator_input


def _detect_validator_evidence(
    payloads: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Derive validator evidence from the real detect-stage payload."""
    detect_payload = _require_mapping(
        payloads.get("detect"), name="registering evidence requires a real detect stage payload"
    )
    families = _require_sequence(detect_payload.get("families"), name="detect families")
    _require(bool(families), "detect payload must record at least one family")
    family_evidence: dict[str, dict[str, str]] = {}
    control_outcomes: dict[str, str] = {}
    applicability: dict[str, str] = {}
    for raw in families:
        family = _require_mapping(raw, name="detect families must be objects")
        family_id = _require_non_empty_string(family.get("family_id"), name="detect families must declare family_id")
        evidence_status = _require_mapping(
            family.get("evidence_status"),
            name=f"detect family {family_id!r} must declare evidence_status",
        )
        _require(bool(evidence_status), f"detect family {family_id!r} must declare evidence_status")
        family_evidence[family_id] = {str(key): str(value) for key, value in evidence_status.items()}
        applicability[family_id] = "applicable"
        outcomes = family.get("control_outcomes")
        if isinstance(outcomes, Mapping):
            for key, value in outcomes.items():
                if str(value) in _ALLOWED_CONTROL_STATUSES:
                    control_outcomes[str(key)] = str(value)
    _require(bool(family_evidence), "detect payload recorded no family evidence")
    _require(
        bool(control_outcomes),
        "detect payload recorded no passed/failed manifest control outcomes",
    )
    return {
        "applicability": applicability,
        "control_outcomes": control_outcomes,
        "family_evidence": family_evidence,
    }


# ---------------------------------------------------------------------------
# Assembly (pure): registered report + blobs + validator result + document
# ---------------------------------------------------------------------------


def _taxonomy_digest() -> str:
    return _digest_of(canonical_json(load_taxonomy()))


def _report_schema_digest() -> str:
    return _digest_of(canonical_json(load_schema()))


def _assemble(
    *,
    request: DiagnosticRequest,
    manifest: Mapping[str, object],
    result: DiagnosticResult,
    checkpoint: WorkflowCheckpoint,
    report: Mapping[str, object],
) -> tuple[bytes, str, bytes, str, list[tuple[str, bytes]], dict[str, object]]:
    stage_records = _stage_records(request, result, checkpoint)
    checkpoint = _require_checkpoint(checkpoint)
    assemble_outputs = _require_outputs(checkpoint.outputs)
    payloads = {item.stage: dict(_require_mapping(item.payload, name="stage payload")) for item in assemble_outputs}
    registered, evidence, validator_input = _register_evidence(
        report, request=request, manifest=manifest, payloads=payloads
    )
    report_bytes = _canonical_bytes(registered)
    report_digest = _digest_of(report_bytes)

    manifest_id = manifest.get("manifest_id")
    schema_version = manifest.get("schema_version")
    _require(
        isinstance(manifest_id, str) and bool(manifest_id),
        "manifest must declare manifest_id",
    )
    _require(
        isinstance(schema_version, str) and bool(schema_version),
        "manifest must declare schema_version",
    )
    try:
        manifest_sha = manifest_digest(manifest)
    except Exception as exc:  # noqa: BLE001 - manifest contract failures fail closed
        raise DiagnosticArtifactError(f"manifest digest failed: {exc}") from exc

    stage_payloads: list[tuple[str, bytes]] = []
    blob_rows: list[dict[str, object]] = []
    seen_names: set[str] = set()
    seen_names.add("diagnostic-report")
    for output in assemble_outputs:
        stage_bytes = _canonical_bytes(output.to_dict())
        name = f"stage-record-{output.stage}"
        _require(name not in seen_names, f"duplicate blob name {name!r}")
        seen_names.add(name)
        stage_payloads.append((name, stage_bytes))
        blob_rows.append(
            {
                "digest": _digest_of(stage_bytes),
                "media_type": "application/json",
                "name": name,
                "relative_path": f"artifacts/{_digest_of(stage_bytes)}",
                "size_bytes": len(stage_bytes),
            }
        )
    for name, data in evidence:
        _require(name not in seen_names, f"duplicate blob name {name!r}")
        seen_names.add(name)
        blob_rows.append(
            {
                "digest": _digest_of(data),
                "media_type": "application/json",
                "name": name,
                "relative_path": f"artifacts/{_digest_of(data)}",
                "size_bytes": len(data),
            }
        )
    blob_rows.append(
        {
            "digest": report_digest,
            "media_type": "application/json",
            "name": "diagnostic-report",
            "relative_path": f"artifacts/{report_digest}",
            "size_bytes": len(report_bytes),
        }
    )
    validator_input["artifact_digests"] = {str(row["name"]): str(row["digest"]) for row in blob_rows}

    seeds_block = manifest.get("seeds")
    document: dict[str, object] = {
        "blobs": blob_rows,
        "environment": _bounded_environment(),
        "hashes": {
            "manifest_sha256": manifest_sha,
            "report_schema_sha256": _report_schema_digest(),
            "taxonomy_sha256": _taxonomy_digest(),
        },
        "manifest": {
            "manifest_id": manifest_id,
            "schema_version": schema_version,
        },
        "report_digest": report_digest,
        "request": request.to_dict(),
        "result": {
            "artifact_refs": list(result.artifact_refs),
            "completed_stages": list(result.completed_stages),
            "report_id": result.report_id,
            "request_id": result.request_id,
            "status": result.status,
        },
        "schema": DIAGNOSTIC_ARTIFACT_SCHEMA,
        "seeds": json.loads(_canonical_bytes(seeds_block).decode("utf-8")) if isinstance(seeds_block, Mapping) else {},
        "stages": stage_records,
        "validator_input": validator_input,
        "validator_result": {},
        "workflow": {
            "config_digest": checkpoint.config_digest,
            "manifest_digest": checkpoint.manifest_digest,
            "request_digest": checkpoint.request_digest,
            "workflow_identity": checkpoint.workflow_identity,
        },
    }

    # Independent validation over the registered report and real evidence.
    evidence_bytes = {name: data for name, data in evidence}
    evidence_digests = {name: _digest_of(data) for name, data in evidence}
    target_input: dict[str, object] | None = None
    target_provenance: Mapping[str, object] | None = None
    target_block = registered.get("target_evidence")
    if target_block is not None:
        target_map = _require_mapping(target_block, name="target evidence")
        target_refs = _require_refs(target_map.get("evidence_refs"), name="target evidence refs")
        _require(len(target_refs) == 1, "v2 target evidence must register exactly one target record")
        target_name = target_refs[0]
        data = evidence_bytes.get(target_name)
        if not isinstance(data, bytes):
            raise DiagnosticArtifactError(f"target record blob {target_name!r} is missing")
        try:
            parsed_target = json.loads(data.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - any parse gap fails closed
            raise DiagnosticArtifactError(f"target record blob is not valid JSON: {exc}") from exc
        _require(isinstance(parsed_target, Mapping), "target record blob must be an object")
        target_input = dict(parsed_target)
        target_input["record_digest"] = _digest_of(data)
        detect_payload = _require_mapping(payloads.get("detect"), name="detect stage payload")
        target_provenance = _require_mapping(
            detect_payload.get("target_provenance"),
            name="detect payload must carry target_provenance for v2",
        )
    try:
        validate_diagnostic_report(
            registered,
            manifest,
            applicability=cast(Mapping[str, str], validator_input["applicability"]),
            family_evidence=cast(Mapping[str, Mapping[str, str]], validator_input["family_evidence"]),
            control_outcomes=cast(Mapping[str, str], validator_input["control_outcomes"]),
            artifacts=evidence_bytes,
            artifact_digests=evidence_digests,
            target_evidence=target_input,
            target_provenance=target_provenance,
        )
    except ReportValidationError as exc:
        raise DiagnosticArtifactError(f"report failed independent validation; nothing was persisted: {exc}") from exc

    document["validator_result"] = {
        "inputs_digest": _digest_of(
            _canonical_bytes(
                {
                    "applicability": validator_input["applicability"],
                    "artifact_digests": validator_input["artifact_digests"],
                    "control_outcomes": validator_input["control_outcomes"],
                    "family_evidence": validator_input["family_evidence"],
                }
            )
        ),
        "report_digest": report_digest,
        "status": "passed",
        "validator": _VALIDATOR_ID,
    }

    document_bytes = _canonical_bytes(document)
    document_digest = _digest_of(document_bytes)
    all_blobs: list[tuple[str, bytes]] = [*stage_payloads, *evidence, ("diagnostic-report", report_bytes)]
    return document_bytes, document_digest, report_bytes, report_digest, all_blobs, registered


# ---------------------------------------------------------------------------
# Public assembly/persistence/load seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticArtifactHandle:
    """Locator for one persisted diagnostic artifact."""

    run_id: str
    artifact_digest: str
    report_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_digest": self.artifact_digest,
            "report_digest": self.report_digest,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class LoadedDiagnosticArtifact:
    """One revalidated diagnostic artifact loaded from persisted files."""

    run_id: str
    artifact_digest: str
    document: Mapping[str, object]
    report: Mapping[str, object]


def persist_diagnostic_artifact(
    root: str | Path,
    *,
    request: object,
    manifest: object,
    result: object,
    checkpoint: object,
    report: object,
) -> DiagnosticArtifactHandle:
    request = _require_request(request)
    manifest = _require_manifest(manifest)
    result = _require_result(result)
    checkpoint = _require_checkpoint(checkpoint)
    report = _require_mapping(report, name="report")
    """Assemble, validate, and persist one content-addressed diagnostic artifact.

    The workflow must be completed and terminal; the registered report
    must pass the independent validator; every blob is stored by the
    SHA-256 of its actual bytes through the atomic run recorder.

    Preflight (assembly, independent validation, and the blob-conflict
    scan) fails before any write: no run record, no files. After
    preflight the run record is created as ``running``, the blobs and
    document are written, and the record then transitions atomically to
    ``completed``; a mid-write exception leaves a recoverable
    non-completed record that loading refuses, and an identical retry
    reuses the deterministic identity and completes safely. Identical
    input is idempotent (no bytes are rewritten); an already completed
    run is never downgraded or rewritten.
    """
    document_bytes, document_digest, _, report_digest, blobs, _ = _assemble(
        request=request,
        manifest=manifest,
        result=result,
        checkpoint=checkpoint,
        report=report,
    )

    manifest_id = str(manifest.get("manifest_id"))
    model = manifest.get("model")
    dataset = manifest.get("dataset")
    seeds_source = _require_mapping(manifest.get("seeds"), name="manifest must declare seeds")
    seed_values: list[int] = []
    for key in ("training", "evaluation", "controls"):
        rows = seeds_source.get(key)
        if isinstance(rows, Sequence) and not isinstance(rows, str | bytes):
            for raw in rows:
                if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                    seed_values.append(int(raw))
    environment = _bounded_environment()

    recorder = FileSystemRunRecorder(root)
    # Conflict pre-scan: no partial writes when bytes would disagree.
    for name, data in blobs:
        path = recorder.artifacts_dir / _digest_of(data)
        if path.exists():
            _require(
                path.read_bytes() == data,
                f"conflicting blob for digest {_digest_of(data)} ({name!r}) already exists",
            )
    document_path = recorder.artifacts_dir / document_digest
    if document_path.exists():
        _require(
            document_path.read_bytes() == document_bytes,
            f"conflicting artifact document for digest {document_digest} already exists",
        )

    record = recorder.start(
        f"diagnostic:{request.request_id}",
        config={
            "artifact_digest": document_digest,
            "manifest_id": manifest_id,
            "report_digest": report_digest,
        },
        code_version="working-tree",
        framework_version="latent-anything-diagnostic",
        model_revisions=(
            {str(cast(Mapping[str, object], model).get("id")): str(cast(Mapping[str, object], model).get("revision"))}
            if isinstance(model, Mapping)
            else {}
        ),
        dataset_revisions=(
            {
                str(cast(Mapping[str, object], dataset).get("id")): str(
                    cast(Mapping[str, object], dataset).get("revision")
                )
            }
            if isinstance(dataset, Mapping)
            else {}
        ),
        seeds=sorted(set(seed_values)),
        environment=environment,
        metadata={
            "diagnostic_artifact_digest": document_digest,
            "diagnostic_artifact_size": len(document_bytes),
            "diagnostic_request_id": request.request_id,
            "schema": DIAGNOSTIC_ARTIFACT_SCHEMA,
        },
        status="running",
    )
    # Mid-write phase: blobs and the document are persisted before the run
    # transitions to completed, so an interrupted write leaves a
    # recoverable non-completed record (never one surfaced as completed).
    for name, data in blobs:
        recorder.add_artifact(record.run_id, data, name=name, media_type="application/json")
    document_ref = recorder.add_artifact(
        record.run_id, document_bytes, name="diagnostic-artifact", media_type="application/json"
    )
    _require(document_ref.digest == document_digest, "artifact document digest drifted")
    current = recorder.get(record.run_id)
    if current.status != "completed":
        recorder.complete(record.run_id)
    # An already completed identical run is never downgraded or rewritten.
    return DiagnosticArtifactHandle(
        run_id=record.run_id,
        artifact_digest=document_digest,
        report_digest=report_digest,
    )


def load_diagnostic_artifact(
    root: str | Path,
    *,
    run_id: object,
    manifest: object,
) -> LoadedDiagnosticArtifact:
    """Load and fully revalidate one persisted diagnostic artifact.

    Uses only persisted files under ``root`` plus the declared external
    manifest: schema and hash identities, request round-trip, stage
    chain, report digest, every blob's bytes/hash/path containment, and
    the independent validator result must all agree or the load fails
    closed.
    """
    manifest = _require_manifest(manifest)
    run_id = _require_run_id(run_id)
    recorder = FileSystemRunRecorder(root)
    try:
        record = recorder.get(run_id)
    except FileNotFoundError as exc:
        raise DiagnosticArtifactError(f"run record {run_id!r} is missing") from exc
    metadata = record.metadata
    document_digest = _require_digest(metadata.get("diagnostic_artifact_digest"))
    document_size = _require_size(metadata.get("diagnostic_artifact_size"))
    metadata_request = _require_non_empty_string(
        metadata.get("diagnostic_request_id"), name="run record diagnostic request id"
    )
    _require(
        metadata.get("schema") == DIAGNOSTIC_ARTIFACT_SCHEMA,
        "run record schema disagrees with the diagnostic artifact schema",
    )
    _require(record.status == "completed", "run record must be completed")

    document_ref = ArtifactRef(
        name="diagnostic-artifact",
        digest=document_digest,
        size_bytes=int(document_size),
        relative_path=f"artifacts/{document_digest}",
        media_type="application/json",
    )
    try:
        document_bytes = recorder.read_artifact(document_ref)
    except FileNotFoundError as exc:
        raise DiagnosticArtifactError(f"artifact document blob {document_digest} is missing") from exc
    except ValueError as exc:
        raise DiagnosticArtifactError(f"artifact document blob rejected: {exc}") from exc
    try:
        document = json.loads(document_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any parse gap fails closed
        raise DiagnosticArtifactError(f"artifact document is not valid JSON: {exc}") from exc
    _require(isinstance(document, Mapping), "artifact document must be a JSON object")
    _require(
        document.get("schema") == DIAGNOSTIC_ARTIFACT_SCHEMA,
        f"artifact schema must be {DIAGNOSTIC_ARTIFACT_SCHEMA!r}",
    )

    # Hash identities against the declared external manifest and frozen contracts.
    try:
        expected_manifest_sha = manifest_digest(manifest)
    except Exception as exc:  # noqa: BLE001 - manifest contract failures fail closed
        raise DiagnosticArtifactError(f"manifest digest failed: {exc}") from exc
    hashes = document.get("hashes")
    _require(isinstance(hashes, Mapping), "artifact must declare hashes")
    _require(
        hashes.get("manifest_sha256") == expected_manifest_sha,
        "manifest hash disagrees with the persisted artifact",
    )
    _require(
        hashes.get("taxonomy_sha256") == _taxonomy_digest(),
        "taxonomy hash disagrees with the frozen taxonomy",
    )
    _require(
        hashes.get("report_schema_sha256") == _report_schema_digest(),
        "report schema hash disagrees with the frozen schema",
    )
    manifest_block = document.get("manifest")
    _require(isinstance(manifest_block, Mapping), "artifact must declare its manifest identity")
    _require(
        manifest_block.get("manifest_id") == manifest.get("manifest_id"),
        "manifest_id disagrees with the persisted artifact",
    )

    request_block = document.get("request")
    _require(isinstance(request_block, Mapping), "artifact must declare its request")
    try:
        request = DiagnosticRequest.from_dict(cast(Mapping[str, object], request_block))
    except Exception as exc:  # noqa: BLE001 - request round-trip failures fail closed
        raise DiagnosticArtifactError(f"persisted request is invalid: {exc}") from exc
    _require(request.to_dict() == dict(request_block), "persisted request did not round-trip")
    _require(
        request.manifest_id == manifest.get("manifest_id"),
        "persisted request disagrees with the manifest",
    )
    _require(
        metadata_request == request.request_id,
        "run record request id disagrees with the persisted request",
    )

    # Blob inventory: unique names, existing bytes, matching hashes, contained paths.
    raw_blobs = document.get("blobs")
    _require(
        isinstance(raw_blobs, Sequence) and not isinstance(raw_blobs, str | bytes) and len(raw_blobs) > 0,
        "artifact must declare its blob inventory",
    )
    refs: list[ArtifactRef] = []
    names: set[str] = set()
    digest_by_name: dict[str, str] = {}
    for index, raw in enumerate(raw_blobs):
        _require(isinstance(raw, Mapping), f"blobs[{index}] must be an object")
        try:
            ref = ArtifactRef.from_dict(cast(Mapping[str, object], raw))
        except Exception as exc:  # noqa: BLE001 - ref contract failures fail closed
            raise DiagnosticArtifactError(f"blobs[{index}] is invalid: {exc}") from exc
        _require(ref.name not in names, f"duplicate blob name {ref.name!r}")
        names.add(ref.name)
        digest_by_name[ref.name] = ref.digest
        refs.append(ref)
    blob_bytes: dict[str, bytes] = {}
    for ref in refs:
        try:
            blob_bytes[ref.name] = recorder.read_artifact(ref)
        except FileNotFoundError as exc:
            raise DiagnosticArtifactError(f"blob {ref.name!r} ({ref.relative_path}) is missing") from exc
        except ValueError as exc:
            raise DiagnosticArtifactError(f"blob {ref.name!r} rejected: {exc}") from exc

    validator_input = document.get("validator_input")
    _require(isinstance(validator_input, Mapping), "artifact must declare its validator input")
    persisted_digests = validator_input.get("artifact_digests")
    _require(
        isinstance(persisted_digests, Mapping),
        "artifact validator input must declare artifact digests",
    )
    _require(
        {str(key): str(value) for key, value in persisted_digests.items()} == digest_by_name,
        "persisted validator artifact digests disagree with the blob inventory",
    )

    # Stage chain: order, outcomes, per-stage workflow digests, payload blobs.
    workflow = _require_mapping(document.get("workflow"), name="artifact must declare its workflow identity")
    workflow_identity = _require_digest(workflow.get("workflow_identity"))
    stage_entries = _require_stage_entries(document.get("stages"))
    stage_names = [str(entry.get("stage")) for entry in stage_entries]
    _require(
        stage_names == list(WORKFLOW_STAGES),
        f"persisted stage chain must be exactly {list(WORKFLOW_STAGES)!r}",
    )
    stage_payloads: dict[str, Mapping[str, object]] = {}
    for entry in stage_entries:
        stage = str(entry.get("stage"))
        record_digest = _require_digest(entry.get("record_digest"))
        payload = _require_blob(blob_bytes.get(f"stage-record-{stage}"), record_digest, name=f"stage {stage!r}")
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - any parse gap fails closed
            raise DiagnosticArtifactError(f"stage record {stage!r} is not valid JSON: {exc}") from exc
        stage_payloads[stage] = _require_mapping(parsed.get("payload"), name=f"stage record {stage!r} payload")
        try:
            output = StageOutput.from_dict(cast(Mapping[str, object], parsed))
        except Exception as exc:  # noqa: BLE001 - stage contract failures fail closed
            raise DiagnosticArtifactError(f"stage record {stage!r} is invalid: {exc}") from exc
        _require(
            output.stage == stage and output.outcome == entry.get("outcome"),
            f"stage record {stage!r} disagrees with the persisted chain",
        )
        output_refs = _require_refs(output.artifact_refs, name=f"stage {stage!r} artifact_refs")
        entry_refs = _require_refs(entry.get("artifact_refs") or (), name=f"stage {stage!r} artifact_refs")
        _require(
            list(output_refs) == list(entry_refs),
            f"stage record {stage!r} artifact_refs disagree with the persisted chain",
        )
        _require(
            output.digest(str(workflow_identity)) == entry.get("output_digest"),
            f"stage digest for {stage!r} disagrees with the workflow identity",
        )

    result_block = document.get("result")
    _require(isinstance(result_block, Mapping), "artifact must declare its result summary")
    _require(
        result_block.get("status") == "completed",
        "persisted diagnostic result must be completed",
    )
    _require(
        list(result_block.get("completed_stages") or ()) == list(WORKFLOW_STAGES),
        "persisted diagnostic result must cover every workflow stage",
    )
    _require(
        result_block.get("request_id") == request.request_id,
        "persisted result request id disagrees with the request",
    )

    # Report bytes and validator-result agreement.
    report_digest = _require_digest(document.get("report_digest"))
    report_bytes = _require_blob(blob_bytes.get("diagnostic-report"), report_digest, name="report")
    try:
        report = json.loads(report_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any parse gap fails closed
        raise DiagnosticArtifactError(f"report blob is not valid JSON: {exc}") from exc
    _require(isinstance(report, Mapping), "report blob must be a JSON object")

    validator_result = document.get("validator_result")
    _require(isinstance(validator_result, Mapping), "artifact must declare its validator result")
    _require(
        validator_result.get("status") == "passed" and validator_result.get("validator") == _VALIDATOR_ID,
        "persisted validator result must be a passing diagnostic-report validation",
    )
    _require(
        validator_result.get("report_digest") == report_digest,
        "validator result disagrees with the persisted report",
    )
    recomputed_inputs_digest = _digest_of(
        _canonical_bytes(
            {
                "applicability": validator_input.get("applicability"),
                "artifact_digests": validator_input.get("artifact_digests"),
                "control_outcomes": validator_input.get("control_outcomes"),
                "family_evidence": validator_input.get("family_evidence"),
            }
        )
    )
    _require(
        validator_result.get("inputs_digest") == recomputed_inputs_digest,
        "validator result disagrees with the persisted validator input",
    )
    target_reload: dict[str, object] | None = None
    target_provenance: Mapping[str, object] | None = None
    target_block = report.get("target_evidence")
    if target_block is not None:
        target_map = _require_mapping(target_block, name="persisted target evidence")
        target_refs = _require_refs(target_map.get("evidence_refs"), name="persisted target evidence refs")
        _require(len(target_refs) == 1, "persisted v2 target evidence must reference exactly one record")
        target_name = target_refs[0]
        blob_data = blob_bytes.get(target_name)
        if not isinstance(blob_data, bytes):
            raise DiagnosticArtifactError(f"persisted target record {target_name!r} is missing")
        try:
            parsed_reload = json.loads(blob_data.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - any parse gap fails closed
            raise DiagnosticArtifactError(f"target record blob is not valid JSON: {exc}") from exc
        _require(isinstance(parsed_reload, Mapping), "persisted target record must be an object")
        target_reload = dict(parsed_reload)
        target_reload["record_digest"] = _digest_of(blob_data)
        target_provenance = _require_mapping(
            stage_payloads["detect"].get("target_provenance"),
            name="persisted detect stage must carry target_provenance for v2",
        )
    try:
        validate_diagnostic_report(
            cast(Mapping[str, object], report),
            manifest,
            applicability=cast(Mapping[str, str], validator_input.get("applicability")),
            family_evidence=cast(Mapping[str, Mapping[str, str]], validator_input.get("family_evidence")),
            control_outcomes=cast(Mapping[str, str], validator_input.get("control_outcomes")),
            artifacts=blob_bytes,
            artifact_digests=cast(Mapping[str, str], persisted_digests),
            target_evidence=target_reload,
            target_provenance=target_provenance,
        )
    except ReportValidationError as exc:
        raise DiagnosticArtifactError(
            f"persisted report failed revalidation (validator-result disagreement): {exc}"
        ) from exc

    return LoadedDiagnosticArtifact(
        run_id=run_id,
        artifact_digest=document_digest,
        document=document,
        report=report,
    )


def registered_artifact_bytes(
    loaded: LoadedDiagnosticArtifact,
    root: str | Path,
) -> dict[str, bytes]:
    """Return every persisted blob by reference name for one loaded artifact."""
    recorder = FileSystemRunRecorder(root)
    raw_blobs = loaded.document.get("blobs")
    result: dict[str, bytes] = {}
    if isinstance(raw_blobs, Sequence) and not isinstance(raw_blobs, str | bytes):
        for raw in raw_blobs:
            if isinstance(raw, Mapping):
                ref = ArtifactRef.from_dict(cast(Mapping[str, object], raw))
                result[ref.name] = recorder.read_artifact(ref)
    return result


__all__ = [
    "DIAGNOSTIC_ARTIFACT_SCHEMA",
    "DiagnosticArtifactError",
    "DiagnosticArtifactHandle",
    "LoadedDiagnosticArtifact",
    "load_diagnostic_artifact",
    "persist_diagnostic_artifact",
    "registered_artifact_bytes",
]
