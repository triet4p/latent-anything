"""Deterministic AI-engineer report renderer for the frozen diagnostic
artifact (Sprint 80.22).

One architecture-neutral private renderer turns a successfully loaded and
independently validated80.21 diagnostic artifact into a concise,
byte-for-byte deterministic text report driven only by validated
structured fields — no heuristics, no model workflows, no prose
invention.

Contract:
- rendering re-loads the artifact first (full80.21 revalidation: schema,
  manifest/taxonomy/report-schema hashes, stage chain, blob hashes and
  path containment, independent validator result), re-verifies the
  registered report digest, opens and parses every cited record blob
  (intervention rows: intervention identity, conclusion/status
  agreement, canonical record digest, target/kind, and either detect metric
  evidence or request-declared aligned comparison-task metric evidence;
  comparison rows: comparison identity and classification),
  and refuses invalid, tampered, or contradictory artifacts with
  `DiagnosticArtifactError`/`ReportRenderingError`;
- the report names, in fixed order: identifiers (artifact/run/schema/
  manifest/workflow/validator identities), the diagnostic symptom with
  its taxonomy family and bounded conclusion, the supported location and
  axes or an explicit non-applicability line, evidence strength with
  uncertainty intervals, control outcomes, family evidence, explanation
  outcomes, a complete content-addressed evidence index, causal state per
  registered intervention (support / falsification / inconclusive /
  unsupported, never promoted), aligned representation-versus-task
  comparison with signed deltas and tolerances, declared limitations plus
  every negative result (never silently omitted), and one concrete next
  action whose references must resolve to registered rows or blobs;
- contradiction and hygiene checks fail closed: unknown statuses,
  unsupported/unknown outcomes, unregistered evidence references,
  record-digest disagreements, report-stage disagreement, causal claims
  without a supported intervention row, failed controls presented as
  success, and non-registered next-action references all raise
  `ReportRenderingError`; when a required manifest control failed the
  causal section renders an explicit blocked line instead of any success
  shape;
- every variable string is escaped so untrusted labels cannot alter the
  section structure; the output contains no absolute paths, no secrets,
  no volatile timestamps, and no nondeterministic ordering;
- `persist_rendered_report` stores the bytes through the existing
  content-addressed recorder conventions (idempotent, conflict-checked)
  and materializes the declared `OutputSelection` copy without mutating
  the validated source artifact.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from latent_anything._diagnostic_artifact import (
    DIAGNOSTIC_ARTIFACT_SCHEMA,
    load_diagnostic_artifact,
    registered_artifact_bytes,
)
from latent_anything._run_record_codec import canonical_json
from latent_anything._run_record_persistence import FileSystemRunRecorder
from latent_anything.diagnostics import DiagnosticRequest, OutputSelection

RENDERED_REPORT_SCHEMA = "diagnostic-report-render-v1"
"""Frozen identity line for one rendered diagnostic report."""

_EVIDENCE_STATUSES = frozenset({"observed", "supported", "falsified", "inconclusive", "unsupported"})
_CONCLUSIONS = frozenset({"supported", "falsified", "inconclusive", "unsupported"})
_FAMILY_OUTCOMES = frozenset({"supported", "inconclusive", "unsupported"})
_EXPLANATION_OUTCOMES = frozenset({"supported", "inconclusive", "unsupported", "omitted"})
_COMPARISON_STATUS_BY_CLASSIFICATION = {
    "representation_only": "observed",
    "task_only": "observed",
    "both": "observed",
    "neither": "observed",
    "inconclusive": "inconclusive",
    "unsupported": "unsupported",
}
_ROW_SECTIONS = (
    "symptoms",
    "localization",
    "hypotheses",
    "statistical_evidence",
    "interventions",
    "comparisons",
    "claims",
)


class ReportRenderingError(ValueError):
    """Raised when a validated artifact cannot be rendered without contradiction."""


def _require_run_id(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ReportRenderingError(f"invalid run id: {value!r}")
    return value


def _require_output(value: object) -> OutputSelection:
    if not isinstance(value, OutputSelection):
        raise ReportRenderingError("output must be an OutputSelection")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ReportRenderingError(name)
    return value


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ReportRenderingError(f"{name} must be a 64-character hex digest")
    return value


def _require_sequence(value: object, *, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReportRenderingError(f"{name} must be a sequence")
    return tuple(value)


def _require_blob(value: object, *, name: str) -> bytes:
    if not isinstance(value, bytes):
        raise ReportRenderingError(f"{name} is missing")
    return value


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReportRenderingError(f"{name} must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportRenderingError(message)


def _digest_of(data: bytes) -> str:
    return sha256(data).hexdigest()


def _escape(value: object) -> str:
    """Render one untrusted value on a single line without altering structure."""
    if not isinstance(value, str):
        raise ReportRenderingError(f"expected a string, got {type(value).__name__}")
    out: list[str] = []
    for character in value:
        code = ord(character)
        if character == "\\":
            out.append("\\\\")
        elif character == "\n":
            out.append("\\n")
        elif character == "\r":
            out.append("\\r")
        elif character == "\t":
            out.append("\\t")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\x{code:02x}")
        else:
            out.append(character)
    return "".join(out)


def _number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportRenderingError(f"expected a number, got {value!r}")
    numeric = float(value)
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        raise ReportRenderingError("expected a finite number")
    return repr(numeric)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReportRenderingError(f"{name} must be a non-empty string")
    return value


def _rows(report: Mapping[str, object], section: str) -> list[Mapping[str, object]]:
    value = report.get(section)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ReportRenderingError(f"report section {section!r} must be a list")
    rows: list[Mapping[str, object]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ReportRenderingError(f"report section {section!r} must hold objects")
        rows.append(raw)
    return rows


def _status(value: object, *, name: str) -> str:
    status = _string(value, name=name)
    _require(status in _EVIDENCE_STATUSES, f"{name} has unknown status {status!r}")
    return status


def _string_list(value: object, *, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ReportRenderingError(f"{name} must be a list of strings")
    result: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw:
            raise ReportRenderingError(f"{name} must hold non-empty strings")
        result.append(raw)
    return result


def _payload(blobs: Mapping[str, bytes], stage: str) -> Mapping[str, object]:
    data = _require_blob(blobs.get(f"stage-record-{stage}"), name=f"stage record blob for {stage!r}")
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any parse gap fails closed
        raise ReportRenderingError(f"stage record {stage!r} is not valid JSON: {exc}") from exc
    payload = _require_mapping(parsed, name=f"stage record {stage!r} must be an object")
    return _require_mapping(payload.get("payload"), name=f"stage record {stage!r} must declare a payload")


def _record_json(blobs: Mapping[str, bytes], name: str) -> Mapping[str, object]:
    data = _require_blob(blobs.get(name), name=f"evidence blob {name!r}")
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any parse gap fails closed
        raise ReportRenderingError(f"evidence blob {name!r} is not valid JSON: {exc}") from exc
    return _require_mapping(parsed, name=f"evidence blob {name!r} must be an object")


def _registered_names(loaded_document: Mapping[str, object]) -> dict[str, str]:
    raw_blobs = _require_sequence(loaded_document.get("blobs"), name="artifact must declare its blob inventory")
    inventory: dict[str, str] = {}
    for raw in raw_blobs:
        entry = _require_mapping(raw, name="blob inventory entries must be objects")
        name = _string(entry.get("name"), name="blob name")
        digest = _string(entry.get("digest"), name="blob digest")
        _require(name not in inventory, f"duplicate blob name {name!r}")
        inventory[name] = digest
    return inventory


def _audit_all_references(report: Mapping[str, object], row_ids: set[str], inventory: Mapping[str, str]) -> None:
    """Every evidence reference must resolve to a registered row or blob."""
    known = set(inventory) | row_ids
    provenance = _require_mapping(report.get("capture_provenance"), name="report must declare capture_provenance")
    captures = _require_sequence(provenance.get("captures"), name="capture provenance captures must be a list")
    for raw_capture in captures:
        capture = _require_mapping(raw_capture, name="capture rows must be objects")
        for reference in _string_list(capture.get("artifact_refs"), name="capture artifact_refs"):
            _require(reference in known, f"unregistered evidence reference: {reference!r}")
    target_block = report.get("target_evidence")
    if isinstance(target_block, Mapping):
        for reference in _string_list(target_block.get("evidence_refs"), name="target evidence_refs"):
            _require(reference in known, f"unregistered evidence reference: {reference!r}")


def render_diagnostic_report(
    root: str | Path,
    *,
    run_id: object,
    manifest: Mapping[str, object],
) -> bytes:
    """Render one validated diagnostic artifact into deterministic report bytes.

    Re-runs the full80.21 loader (so invalid, tampered, mis-hashed, or
    validator-disagreeing artifacts refuse rendering), re-verifies the
    registered report digest, opens and parses every cited record blob,
    and cross-checks each registered row against its record —
    intervention rows must match the record's intervention identity,
    conclusion, target, kind, detect or request-declared aligned comparison-task
    metric evidence, and canonical digest; comparison rows must match the record's
    comparison identity and classification — before emitting the fixed section order
    with escaped values and content-addressed evidence links.
    """
    loaded = load_diagnostic_artifact(root, run_id=run_id, manifest=manifest)
    document = loaded.document
    _require(
        document.get("schema") == DIAGNOSTIC_ARTIFACT_SCHEMA,
        "artifact schema disagreement",
    )
    validator_result = _require_mapping(
        document.get("validator_result"), name="artifact must declare its validator result"
    )
    _require(
        validator_result.get("status") == "passed",
        "artifact validator result is not passing; refusing to render",
    )
    # The registered report must be byte-identical to the persisted report.
    report_bytes = canonical_json(loaded.report) + b"\n"
    _require(
        _inventory_digest(document, "diagnostic-report") == sha256(report_bytes).hexdigest(),
        "registered report does not match the persisted report digest",
    )
    report = loaded.report
    report_id = _string(report.get("report_id"), name="report_id")
    _require(
        report.get("schema_version") == "diagnostic-report-schema-v1",
        "report schema disagreement",
    )
    inventory = _registered_names(document)
    blobs = registered_artifact_bytes(loaded, root)
    _require(set(blobs) == set(inventory), "persisted blob set disagrees with the inventory")
    for name, data in blobs.items():
        _require(
            sha256(data).hexdigest() == inventory[name],
            f"evidence blob {name!r} re-hashes to an unexpected digest",
        )

    row_ids: set[str] = set()
    for section in _ROW_SECTIONS:
        for row in _rows(report, section):
            row_id = _string(row.get("id"), name=f"{section} id")
            _require(row_id not in row_ids, f"duplicate row id {row_id!r}")
            row_ids.add(row_id)
    _audit_all_references(report, row_ids, inventory)

    request = DiagnosticRequest.from_dict(cast(Mapping[str, object], document["request"]))
    requested_comparison_metrics = {
        comparison.comparison_id: frozenset(comparison.metric_ids) for comparison in request.comparisons
    }
    reported_comparison_metrics = {
        _string(row.get("id"), name="comparison row id"): frozenset(
            _string_list(row.get("metric_ids"), name="comparison metric_ids")
        )
        for row in _rows(report, "comparisons")
    }
    comparison_task_metric_ids: set[str] = set()
    comparison_payload = _payload(blobs, "compare")
    comparison_records = _require_sequence(
        comparison_payload.get("comparisons"), name="compare stage record must declare comparisons"
    )
    for index, raw_comparison in enumerate(comparison_records):
        comparison = _require_mapping(raw_comparison, name=f"compare stage comparisons[{index}]")
        comparison_id = _string(comparison.get("comparison_id"), name="comparison_id")
        requested_metrics = requested_comparison_metrics.get(comparison_id)
        _require(requested_metrics is not None, f"comparison {comparison_id!r} is not request-declared")
        task = _require_mapping(comparison.get("task"), name=f"comparison {comparison_id!r} task")
        task_metric_id = _string(task.get("metric_id"), name="comparison task metric_id")
        assert requested_metrics is not None
        _require(
            task_metric_id in requested_metrics,
            f"comparison {comparison_id!r} task metric {task_metric_id!r} is not request-declared",
        )
        report_metrics = reported_comparison_metrics.get(comparison_id)
        _require(
            report_metrics is not None and task_metric_id in report_metrics,
            f"comparison {comparison_id!r} task metric {task_metric_id!r} is not report-declared",
        )
        comparison_task_metric_ids.add(task_metric_id)
    hashes = _require_mapping(document.get("hashes"), name="artifact must declare hashes")
    workflow = _require_mapping(document.get("workflow"), name="artifact must declare its workflow identity")
    manifest_block = _require_mapping(document.get("manifest"), name="artifact must declare its manifest identity")
    report_stage_payload = _payload(blobs, "report")
    stage_report_id = report_stage_payload.get("report_id")
    _require(
        stage_report_id == report_id,
        f"report stage declares report_id {stage_report_id!r} but the report is {report_id!r}",
    )

    lines: list[str] = [f"# Diagnostic Report {_escape(report_id)}", ""]

    # ---------------------------------------------------------------- Identifiers
    lines.append("## Identifiers")
    lines.append(f"render schema: {RENDERED_REPORT_SCHEMA}")
    lines.append(f"artifact schema: {DIAGNOSTIC_ARTIFACT_SCHEMA}")
    lines.append(f"report schema: {_escape(_string(report.get('schema_version'), name='schema_version'))}")
    lines.append(f"run id: {_escape(loaded.run_id)}")
    lines.append(f"artifact digest: {loaded.artifact_digest}")
    lines.append(f"report digest: {_inventory_digest(document, 'diagnostic-report')}")
    lines.append(f"manifest: {_escape(_string(manifest_block.get('manifest_id'), name='manifest_id'))}")
    lines.append(f"manifest sha256: {_escape(_string(hashes.get('manifest_sha256'), name='manifest_sha256'))}")
    lines.append(f"taxonomy sha256: {_escape(_string(hashes.get('taxonomy_sha256'), name='taxonomy_sha256'))}")
    lines.append(
        f"report schema sha256: {_escape(_string(hashes.get('report_schema_sha256'), name='report_schema_sha256'))}"
    )
    lines.append(f"workflow identity: {_escape(_string(workflow.get('workflow_identity'), name='workflow_identity'))}")
    lines.append(f"request id: {_escape(request.request_id)}")
    lines.append("validator: passed")
    contract = report.get("evidence_contract")
    if contract is not None:
        lines.append(f"evidence contract: {_escape(_string(contract, name='evidence contract'))}")
    target_block = report.get("target_evidence")
    if isinstance(target_block, Mapping):
        target_map = _require_mapping(target_block, name="report must declare its target evidence")
        lines.append(f"target: {_escape(_string(target_map.get('target_id'), name='target id'))}")
        lines.append(f"target rule: {_escape(_string(target_map.get('rule'), name='target rule'))}")
        lines.append(f"target record: {_escape(_string(target_map.get('record_digest'), name='target record'))}")
    lines.append("")

    # ---------------------------------------------------------------- Symptom
    detect_payload = _payload(blobs, "detect")
    families = _require_sequence(
        detect_payload.get("families"), name="detect stage record must declare families to render a symptom"
    )
    _require(bool(families), "detect stage record must declare families to render a symptom")
    detect_metric_ids: set[str] = set()
    lines.append("## Symptom")
    symptom_rows = _rows(report, "symptoms")
    _require(bool(symptom_rows), "report must declare at least one symptom")
    for row in symptom_rows:
        lines.append(f"description: {_escape(_string(row.get('description'), name='description'))}")
        symptom_metrics = _string_list(row.get("metric_ids"), name="symptom metrics")
        lines.append(f"metrics: {', '.join(_escape(item) for item in symptom_metrics)}")
        lines.append(f"status: {_status(row.get('status'), name='symptom status')}")
    for raw_family in families:
        family = _require_mapping(raw_family, name="detect families must be objects")
        family_id = _string(family.get("family_id"), name="family_id")
        observed = family.get("observed_metrics")
        if isinstance(observed, Mapping):
            for key in observed:
                detect_metric_ids.add(str(key))
        outcome = _string(family.get("outcome"), name="family outcome")
        _require(outcome in _FAMILY_OUTCOMES, f"unknown family outcome {outcome!r}")
        lines.append(f"family: {_escape(family_id)}")
        lines.append(f"bounded conclusion: {outcome}")
        reason = family.get("reason")
        if isinstance(reason, str) and reason:
            lines.append(f"reason: {_escape(reason)}")
    hypothesis_rows = _rows(report, "hypotheses")
    for row in hypothesis_rows:
        lines.append(f"hypothesis: {_escape(_string(row.get('statement'), name='hypothesis'))}")
        lines.append(f"hypothesis status: {_status(row.get('status'), name='hypothesis status')}")
    lines.append("")

    # ---------------------------------------------------------------- Location
    lines.append("## Location")
    capture_row = None
    provenance = _require_mapping(report.get("capture_provenance"), name="capture_provenance")
    captures = _require_sequence(provenance.get("captures"), name="capture provenance captures")
    if captures and isinstance(captures[0], Mapping):
        capture_row = captures[0]
    if capture_row is not None:
        capture_map = _require_mapping(capture_row, name="capture row")
        axes = _string_list(capture_map.get("axes"), name="capture axes")
        lines.append(f"axes: {', '.join(_escape(item) for item in axes)}")
        lines.append(
            f"representation: {_escape(_string(capture_map.get('representation_identity'), name='representation'))}"
        )
    localization_rows = [row for row in _rows(report, "localization") if row.get("status") in ("observed", "supported")]
    if not localization_rows:
        lines.append("location: not applicable — no supported location was recorded")
    else:
        for row in localization_rows:
            confidence = _number(row.get("confidence"))
            status = _status(row.get("status"), name="localization status")
            lines.append(
                f"location: {_escape(_string(row.get('axis'), name='axis'))} "
                f"{_escape(_string(row.get('selection'), name='selection'))} (confidence {confidence}, "
                f"status {_escape(status)})"
            )
    lines.append("")

    # ---------------------------------------------------------------- Evidence
    lines.append("## Evidence")
    for row in _rows(report, "statistical_evidence"):
        metric_id = _string(row.get("metric_id"), name="statistical metric_id")
        estimate = _number(row.get("estimate"))
        status = _status(row.get("status"), name="statistical status")
        uncertainty = row.get("uncertainty")
        interval_text = ""
        if isinstance(uncertainty, Mapping) and "lower" in uncertainty and "upper" in uncertainty:
            interval_text = f", interval [{_number(uncertainty.get('lower'))}, {_number(uncertainty.get('upper'))}]"
        lines.append(f"metric {_escape(metric_id)}: estimate {estimate}{interval_text}, status {status}")
    validator_input = _require_mapping(
        document.get("validator_input"), name="artifact must declare its validator input"
    )
    control_outcomes = _require_mapping(
        validator_input.get("control_outcomes"), name="validator input must declare control outcomes"
    )
    for control_id in sorted(control_outcomes):
        outcome = control_outcomes[control_id]
        _require(
            outcome in ("passed", "failed"),
            f"unknown control outcome for {control_id!r}: {outcome!r}",
        )
        lines.append(f"control {_escape(control_id)}: {outcome}")
    family_evidence = _require_mapping(
        validator_input.get("family_evidence"), name="validator input must declare family evidence"
    )
    for family_id in sorted(family_evidence):
        evidence = _require_mapping(family_evidence[family_id], name="family evidence must be an object")
        rendered = ", ".join(f"{_escape(key)}={_escape(evidence[key])}" for key in sorted(evidence))
        lines.append(f"family evidence {_escape(family_id)}: {rendered}")
    explain_payload = _payload(blobs, "explain")
    explain_evidence = explain_payload.get("family_evidence")
    if isinstance(explain_evidence, Mapping):
        for hypothesis_id in sorted(explain_evidence):
            entry = _require_mapping(explain_evidence[hypothesis_id], name="explain evidence entries must be objects")
            outcome = _string(entry.get("outcome"), name="explanation outcome")
            _require(
                outcome in _EXPLANATION_OUTCOMES,
                f"unknown explanation outcome {outcome!r}",
            )
            method = _string(entry.get("method"), name="explanation method")
            lines.append(f"explanation {_escape(hypothesis_id)}: {_escape(method)} outcome {outcome}")
    for name in inventory:
        lines.append(f"evidence {_escape(name)}: {inventory[name]}")
    lines.append("")

    # ---------------------------------------------------------------- Causal
    lines.append("## Causal")
    intervention_rows = _rows(report, "interventions")
    if not intervention_rows:
        lines.append("causal: unsupported — no intervention records were registered")
    failed_controls = sorted(control_id for control_id, outcome in control_outcomes.items() if outcome == "failed")
    causal_claim_rows = [row for row in _rows(report, "claims") if row.get("kind") == "causal_result"]
    intervention_status = {
        _string(row.get("id"), name="intervention row id"): _status(row.get("status"), name="intervention status")
        for row in intervention_rows
    }
    for row in intervention_rows:
        row_id = _string(row.get("id"), name="intervention row id")
        status = intervention_status[row_id]
        _require(status in _EVIDENCE_STATUSES, f"unknown intervention status {status!r}")
        _require(
            row_id.startswith("intervention-") and row_id.endswith("-record"),
            f"intervention row id {row_id!r} is not a registered record id",
        )
        stripped_id = row_id[len("intervention-") : -len("-record")]
        outcome = _require_mapping(row.get("outcome"), name="intervention rows must declare an outcome")
        conclusion = _string(outcome.get("conclusion"), name="intervention conclusion")
        _require(
            conclusion == status,
            f"intervention {row_id!r} status {status!r} disagrees with its recorded conclusion {conclusion!r}",
        )
        evidence_refs = _string_list(row.get("evidence_refs"), name="intervention evidence_refs")
        record_refs = [reference for reference in evidence_refs if reference in inventory]
        _require(
            len(record_refs) == 1,
            f"intervention {row_id!r} must reference exactly one registered record blob",
        )
        record_digest = _require_digest(outcome.get("record_digest"), name=f"intervention {row_id!r} record digest")
        # F1: open, parse, and cross-check the cited record bytes before
        # any causal prose is rendered — identity, conclusion, target,
        # kind, metric, and canonical digest must all agree.
        record_bytes = blobs[record_refs[0]]
        try:
            record = json.loads(record_bytes.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - any parse gap fails closed
            raise ReportRenderingError(f"intervention {row_id!r} record blob is not valid JSON: {exc}") from exc
        record = _require_mapping(record, name=f"intervention {row_id!r} record blob must be an object")
        _require(
            record.get("intervention_id") == stripped_id,
            f"intervention {row_id!r} record names intervention {record.get('intervention_id')!r}, not {stripped_id!r}",
        )
        _string(record.get("intervention_id"), name="record intervention_id")
        record_conclusion = _string(record.get("conclusion"), name="record conclusion")
        _require(
            record_conclusion in _CONCLUSIONS,
            f"intervention {row_id!r} record has unknown conclusion {record_conclusion!r}",
        )
        _require(
            record_conclusion == status,
            f"intervention {row_id!r} status {status!r} disagrees with record conclusion {record_conclusion!r}",
        )
        _require(
            record.get("target") == row.get("target"),
            f"intervention {row_id!r} target disagrees with its record",
        )
        _require(
            record.get("kind") == row.get("intervention"),
            f"intervention {row_id!r} kind disagrees with its record",
        )
        record_metric = _string(record.get("metric_id"), name="record metric_id")
        _require(
            record_metric in detect_metric_ids or record_metric in comparison_task_metric_ids,
            f"intervention {row_id!r} record metric {record_metric!r} has no prior detect or "
            "request-declared aligned comparison-task metric evidence",
        )
        _require(
            _digest_of(record_bytes) == record_digest,
            f"intervention {row_id!r} record digest disagrees with the cited record bytes",
        )
        _require(
            record_digest == inventory[record_refs[0]],
            f"intervention {row_id!r} record digest disagrees with its registered blob",
        )
        lines.append(
            f"intervention {_escape(_string(row.get('intervention'), name='intervention'))} on "
            f"{_escape(_string(row.get('target'), name='target'))}: {row_id}, conclusion {status}"
        )
        lines.append(f"record {_escape(record_refs[0])}: {inventory[record_refs[0]]}")
    for row in causal_claim_rows:
        claim_id = _string(row.get("id"), name="claim id")
        status = _status(row.get("status"), name="claim status")
        claim_allowed = _require_bool(row.get("claim_allowed"), name="claim_allowed must be boolean")
        if status in ("supported", "falsified"):
            _require(claim_allowed is True, f"claim {claim_id!r} must allow its {status} status")
        else:
            _require(claim_allowed is False, f"claim {claim_id!r} must not allow its {status} status")
        if status == "supported":
            supporting = [
                reference
                for reference in _string_list(row.get("evidence_refs"), name="claim evidence_refs")
                if intervention_status.get(reference) == "supported"
            ]
            _require(
                bool(supporting),
                f"claim {claim_id!r} is supported but references no supported intervention record",
            )
            _require(
                not failed_controls,
                f"claim {claim_id!r} is supported while required control(s) {', '.join(failed_controls)} failed",
            )
        allowed_text = "yes" if claim_allowed else "no"
        lines.append(f"claim {claim_id}: {status} (claim allowed {allowed_text})")
    if failed_controls:
        blocked = ", ".join(_escape(control_id) for control_id in failed_controls)
        lines.append(f"causal promotion blocked: required control(s) {blocked} failed; no success is claimed")
    lines.append("")

    # ---------------------------------------------------------------- Comparison
    lines.append("## Comparison")
    comparison_rows = _rows(report, "comparisons")
    if not comparison_rows:
        lines.append("comparison: not applicable — no comparison records were registered")
    for row in comparison_rows:
        row_id = _string(row.get("id"), name="comparison row id")
        status = _status(row.get("status"), name="comparison status")
        metric_ids = _string_list(row.get("metric_ids"), name="comparison metric_ids")
        evidence_refs = _string_list(row.get("evidence_refs"), name="comparison evidence_refs")
        record_refs = [reference for reference in evidence_refs if reference in inventory]
        _require(
            len(record_refs) == 1,
            f"comparison {row_id!r} must reference exactly one registered record blob",
        )
        record = _record_json(blobs, record_refs[0])
        _require(
            record.get("comparison_id") == row_id,
            f"comparison {row_id!r} record names comparison {record.get('comparison_id')!r}, not {row_id!r}",
        )
        _string(record.get("comparison_id"), name="comparison_id")
        classification = _string(record.get("classification"), name="comparison classification")
        _require(
            classification in _COMPARISON_STATUS_BY_CLASSIFICATION,
            f"comparison {row_id!r} has unknown classification {classification!r}",
        )
        _require(
            _COMPARISON_STATUS_BY_CLASSIFICATION[classification] == status,
            f"comparison {row_id!r} status {status!r} disagrees with its recorded classification {classification!r}",
        )
        baseline = _require_mapping(record.get("baseline"), name=f"comparison {row_id!r} baseline")
        candidate = _require_mapping(record.get("candidate"), name=f"comparison {row_id!r} candidate")
        lines.append(
            f"comparison {row_id}: {status} (classification {classification}), "
            f"baseline {_escape(_string(baseline.get('run_id'), name='run_id'))} "
            f"({_escape(_string(baseline.get('checkpoint_id'), name='checkpoint_id'))}), "
            f"candidate {_escape(_string(candidate.get('run_id'), name='run_id'))} "
            f"({_escape(_string(candidate.get('checkpoint_id'), name='checkpoint_id'))})"
        )
        lines.append(f"metrics: {', '.join(_escape(item) for item in metric_ids)}")
        representation = _require_mapping(record.get("representation"), name=f"comparison {row_id!r} representation")
        task = _require_mapping(record.get("task"), name=f"comparison {row_id!r} task")
        for label, block in (("representation", representation), ("task", task)):
            metric_id = _string(block.get("metric_id"), name=f"{label} metric_id")
            signed = _number(block.get("signed_delta"))
            absolute = _number(block.get("absolute_delta"))
            tolerance = _number(block.get("tolerance"))
            changed = _require_bool(block.get("changed"), name="changed must be boolean")
            lines.append(
                f"{label} {_escape(metric_id)}: signed delta {signed}, absolute delta "
                f"{absolute}, tolerance {tolerance}, changed {'yes' if changed else 'no'}"
            )
        lines.append(f"record {_escape(record_refs[0])}: {inventory[record_refs[0]]}")
    lines.append("")

    # ---------------------------------------------------------------- Limitations
    lines.append("## Limitations")
    limitation_rows = _rows(report, "limitations")
    _require(bool(limitation_rows), "report must declare at least one limitation")
    for row in limitation_rows:
        affects = _string_list(row.get("affects"), name="limitation affects")
        blocking = _require_bool(row.get("blocking"), name="limitation blocking must be boolean")
        lines.append(
            f"limitation {_escape(_string(row.get('id'), name='id'))}: "
            f"{_escape(_string(row.get('description'), name='description'))} "
            f"(affects {', '.join(_escape(item) for item in affects)}; "
            f"blocking {'yes' if blocking else 'no'})"
        )
    negative_states = ("falsified", "inconclusive", "unsupported")
    for row in intervention_rows:
        status = _status(row.get("status"), name="intervention status")
        if status in negative_states:
            lines.append(f"negative result: intervention {_string(row.get('id'), name='intervention id')} — {status}")
    for row in comparison_rows:
        status = _status(row.get("status"), name="comparison status")
        if status in negative_states:
            lines.append(f"negative result: comparison {_string(row.get('id'), name='comparison id')} — {status}")
    lines.append("")

    # ---------------------------------------------------------------- Next action
    lines.append("## Next action")
    next_action = _require_mapping(report.get("next_action"), name="report must declare its next action")
    action = _string(next_action.get("action"), name="next action action")
    rationale = _string(next_action.get("rationale"), name="next action rationale")
    lines.append(f"action: {_escape(action)}")
    lines.append(f"rationale: {_escape(rationale)}")
    required = _string_list(next_action.get("required_evidence_refs"), name="next action required_evidence_refs")
    known = set(inventory) | row_ids
    if not required:
        lines.append("requires evidence: none declared")
    for reference in required:
        _require(
            reference in known,
            f"next action requires unregistered evidence reference {reference!r}",
        )
        lines.append(f"requires evidence {_escape(reference)}")
    lines.append("")

    rendered = ("\n".join(lines)).encode("utf-8")
    _require(len(rendered) > 0, "rendered report must not be empty")
    return rendered


def _inventory_digest(document: Mapping[str, object], name: str) -> str:
    raw_blobs = document.get("blobs")
    if isinstance(raw_blobs, Sequence) and not isinstance(raw_blobs, str | bytes):
        for raw in raw_blobs:
            if isinstance(raw, Mapping) and raw.get("name") == name:
                digest = raw.get("digest")
                if isinstance(digest, str):
                    return digest
    raise ReportRenderingError(f"blob {name!r} is not registered in the artifact inventory")


@dataclass(frozen=True)
class RenderedReportHandle:
    """Locator for one persisted rendered report."""

    run_id: str
    digest: str
    artifact_name: str
    relative_path: str
    declared_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_name": self.artifact_name,
            "declared_path": self.declared_path,
            "digest": self.digest,
            "relative_path": self.relative_path,
            "run_id": self.run_id,
        }


def persist_rendered_report(
    root: str | Path,
    *,
    run_id: object,
    manifest: Mapping[str, object],
    output: object,
) -> RenderedReportHandle:
    """Render and persist one report without mutating the source artifact.

    The bytes are stored content-addressed on the same run through the
    atomic recorder (conflicting bytes reject before any write), and the
    declared ``OutputSelection`` copy is materialized at
    ``output_location/artifact_name``. ``include_report`` must be true:
    an output selection that does not request a report fails closed.
    """
    output = _require_output(output)
    run_id = _require_run_id(run_id)
    if not output.include_report:
        raise ReportRenderingError("output selection does not request a report")
    artifact_name = output.artifact_name
    if Path(artifact_name).name != artifact_name:
        raise ReportRenderingError("artifact_name must be a simple file name")
    rendered = render_diagnostic_report(root, run_id=run_id, manifest=manifest)
    digest = sha256(rendered).hexdigest()
    recorder = FileSystemRunRecorder(root)
    stored_path = recorder.artifacts_dir / digest
    if stored_path.exists() and stored_path.read_bytes() != rendered:
        raise ReportRenderingError(f"conflicting rendered report blob for digest {digest} already exists")
    reference = recorder.add_artifact(run_id, rendered, name=f"{artifact_name}.rendered", media_type="text/markdown")
    _require(reference.digest == digest, "rendered report digest drifted during persistence")
    declared = Path(output.output_location) / artifact_name
    declared.parent.mkdir(parents=True, exist_ok=True)
    if declared.exists() and declared.read_bytes() != rendered:
        raise ReportRenderingError(f"conflicting declared report at {output.output_location}/{artifact_name}")
    if not declared.exists():
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=declared.parent, prefix=f".{declared.name}.", delete=False) as handle:
                temporary = handle.name
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, declared)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
    return RenderedReportHandle(
        run_id=run_id,
        digest=digest,
        artifact_name=reference.name,
        relative_path=reference.relative_path,
        declared_path=str(declared),
    )


__all__ = [
    "RENDERED_REPORT_SCHEMA",
    "RenderedReportHandle",
    "ReportRenderingError",
    "persist_rendered_report",
    "render_diagnostic_report",
]
