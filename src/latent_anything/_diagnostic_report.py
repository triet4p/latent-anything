"""Shape and semantic-kind checks for the frozen diagnostic report schema.

This module intentionally stops at report construction semantics. Cross-artifact
hashes, taxonomy references, control outcomes, and conclusion-strength review
remain independent concerns for Sprint 80.4.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from latent_anything._artifact_path import resolve_artifact_path

SCHEMA_VERSION = "diagnostic-report-schema-v1"
SCHEMA_PATH = resolve_artifact_path("diagnostic_report_schema_v1.json")

_EVIDENCE_KINDS = frozenset({"observation", "explanation", "causal_result", "unsupported_conclusion"})
_EVIDENCE_STATUSES = frozenset({"observed", "supported", "falsified", "inconclusive", "unsupported"})
_ALLOWED_STATUS_BY_KIND = {
    "observation": frozenset({"observed", "inconclusive", "unsupported"}),
    "explanation": frozenset({"supported", "inconclusive", "unsupported"}),
    "causal_result": frozenset({"supported", "falsified", "inconclusive", "unsupported"}),
    "unsupported_conclusion": frozenset({"unsupported"}),
}
_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "report_id",
        "status",
        "capture_provenance",
        "symptoms",
        "localization",
        "hypotheses",
        "statistical_evidence",
        "interventions",
        "comparisons",
        "limitations",
        "next_action",
        "claims",
    }
)
_OPTIONAL_TOP_FIELDS = frozenset({"target_evidence", "evidence_contract"})
"""Optional v2 evidence-contract fields. Frozen v1 reports omit both."""


class DiagnosticReportValidationError(ValueError):
    """Raised when a report is malformed or conflates evidence semantics."""


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DiagnosticReportValidationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DiagnosticReportValidationError(f"{name} must be a non-empty string")
    return value


def _string_list(value: object, *, name: str, minimum: int = 0) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DiagnosticReportValidationError(f"{name} must be a list of strings")
    result = list(value)
    if len(result) < minimum or not all(isinstance(item, str) and item for item in result):
        raise DiagnosticReportValidationError(f"{name} must contain at least {minimum} non-empty strings")
    if len(set(result)) != len(result):
        raise DiagnosticReportValidationError(f"{name} contains duplicate references")
    return result


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], *, name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise DiagnosticReportValidationError(f"{name} fields are invalid ({'; '.join(details)})")


def _validate_schema_shape(schema: Mapping[str, object]) -> None:
    if schema.get("schema_version") != SCHEMA_VERSION or schema.get("schema_id") != "diagnostic-report":
        raise DiagnosticReportValidationError("unsupported diagnostic-report schema version or identity")
    if schema.get("status") != "frozen" or schema.get("report_status") != "declared":
        raise DiagnosticReportValidationError("diagnostic-report schema is not frozen")
    required = frozenset(_string_list(schema.get("required_fields"), name="required_fields", minimum=1))
    if required != _TOP_FIELDS:
        raise DiagnosticReportValidationError("schema required fields are incomplete or unstable")
    kinds = set(_string_list(schema.get("evidence_kinds"), name="evidence_kinds", minimum=1))
    if kinds != set(_EVIDENCE_KINDS):
        raise DiagnosticReportValidationError("evidence kinds are incomplete or unstable")
    statuses = set(_string_list(schema.get("evidence_statuses"), name="evidence_statuses", minimum=1))
    if statuses != set(_EVIDENCE_STATUSES):
        raise DiagnosticReportValidationError("evidence statuses are incomplete or unstable")
    semantics = _mapping(schema.get("evidence_semantics"), name="evidence_semantics")
    for kind in _EVIDENCE_KINDS:
        _mapping(semantics.get(kind), name=f"evidence_semantics.{kind}")
    sections = _mapping(schema.get("sections"), name="sections")
    if frozenset(sections) != _TOP_FIELDS - {"schema_version", "report_id", "status"}:
        raise DiagnosticReportValidationError("report sections are incomplete or unstable")
    fail_closed = _mapping(schema.get("fail_closed"), name="fail_closed")
    for case_name in ("malformed", "conflated_evidence_kind", "unsupported_conclusion"):
        case = _mapping(fail_closed.get(case_name), name=f"fail_closed.{case_name}")
        if case.get("claim_allowed") is not False:
            raise DiagnosticReportValidationError(f"fail_closed.{case_name} must be non-promotable")


def load_schema(path: Path = SCHEMA_PATH) -> Mapping[str, object]:
    """Load and validate the canonical report schema document."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticReportValidationError(f"diagnostic-report schema cannot be loaded from {path}") from exc
    schema = _mapping(raw, name="schema")
    _validate_schema_shape(schema)
    return schema


def _validate_capture_provenance(value: Mapping[str, object]) -> None:
    _exact_keys(value, frozenset({"captures"}), name="capture_provenance")
    captures = value.get("captures")
    if isinstance(captures, (str, bytes)) or not isinstance(captures, Sequence) or not captures:
        raise DiagnosticReportValidationError("capture_provenance.captures must be non-empty")
    identities: set[str] = set()
    fields = frozenset(
        {"capture_id", "model_revision", "dataset_split", "representation_identity", "axes", "artifact_refs"}
    )
    for index, raw_capture in enumerate(captures):
        capture = _mapping(raw_capture, name=f"capture_provenance.captures[{index}]")
        _exact_keys(capture, fields, name=f"capture_provenance.captures[{index}]")
        capture_id = _string(capture.get("capture_id"), name=f"captures[{index}].capture_id")
        if capture_id in identities:
            raise DiagnosticReportValidationError("capture identifiers must be unique")
        identities.add(capture_id)
        for field in ("model_revision", "dataset_split", "representation_identity"):
            _string(capture.get(field), name=f"captures[{index}].{field}")
        _string_list(capture.get("axes"), name=f"captures[{index}].axes", minimum=1)
        _string_list(capture.get("artifact_refs"), name=f"captures[{index}].artifact_refs", minimum=1)


def _validate_item_list(value: object, *, name: str, fields: frozenset[str], minimum: int = 0) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DiagnosticReportValidationError(f"{name} must be a list")
    if len(value) < minimum:
        raise DiagnosticReportValidationError(f"{name} must contain at least {minimum} item(s)")
    for index, raw_item in enumerate(value):
        item = _mapping(raw_item, name=f"{name}[{index}]")
        _exact_keys(item, fields, name=f"{name}[{index}]")
        _string(item.get("id"), name=f"{name}[{index}].id")
        if "status" in fields:
            _validate_status(item.get("status"), name=f"{name}[{index}].status")


def _validate_status(value: object, *, name: str) -> str:
    status = _string(value, name=name)
    if status not in _EVIDENCE_STATUSES:
        raise DiagnosticReportValidationError(f"{name} is unsupported")
    return status


def _validate_claims(value: object) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise DiagnosticReportValidationError("claims must be a non-empty list")
    fields = frozenset(
        {
            "id",
            "kind",
            "status",
            "claim",
            "evidence_refs",
            "control_refs",
            "causal",
            "claim_allowed",
            "missing_evidence",
        }
    )
    identities: set[str] = set()
    for index, raw_claim in enumerate(value):
        claim = _mapping(raw_claim, name=f"claims[{index}]")
        _exact_keys(claim, fields, name=f"claims[{index}]")
        claim_id = _string(claim.get("id"), name=f"claims[{index}].id")
        if claim_id in identities:
            raise DiagnosticReportValidationError("claim identifiers must be unique")
        identities.add(claim_id)
        kind = _string(claim.get("kind"), name=f"claims[{index}].kind")
        if kind not in _EVIDENCE_KINDS:
            raise DiagnosticReportValidationError(f"claims[{index}].kind is unsupported")
        status = _validate_status(claim.get("status"), name=f"claims[{index}].status")
        if status not in _ALLOWED_STATUS_BY_KIND[kind]:
            raise DiagnosticReportValidationError(f"{kind} cannot use status {status}")
        _string(claim.get("claim"), name=f"claims[{index}].claim")
        evidence_refs = _string_list(claim.get("evidence_refs"), name=f"claims[{index}].evidence_refs")
        control_refs = _string_list(claim.get("control_refs"), name=f"claims[{index}].control_refs")
        missing = _string_list(claim.get("missing_evidence"), name=f"claims[{index}].missing_evidence")
        causal = claim.get("causal")
        claim_allowed = claim.get("claim_allowed")
        if not isinstance(causal, bool) or not isinstance(claim_allowed, bool):
            raise DiagnosticReportValidationError(f"claims[{index}] causal and claim_allowed must be boolean")
        if kind == "observation" and causal:
            raise DiagnosticReportValidationError("observations cannot be causal")
        if kind == "explanation" and causal:
            raise DiagnosticReportValidationError("explanations cannot be causal results")
        if kind == "causal_result" and not causal:
            raise DiagnosticReportValidationError("causal_result must declare causal=true")
        if kind == "causal_result" and not control_refs:
            raise DiagnosticReportValidationError("causal_result requires control references")
        if kind in {"explanation", "causal_result"} and not evidence_refs:
            raise DiagnosticReportValidationError(f"{kind} requires evidence references")
        if kind == "unsupported_conclusion":
            if causal:
                raise DiagnosticReportValidationError("unsupported conclusions cannot be causal results")
            if status != "unsupported" or claim_allowed or not missing:
                raise DiagnosticReportValidationError("unsupported conclusions must be explicit and non-promotable")
        elif status in {"unsupported", "inconclusive"} and claim_allowed:
            raise DiagnosticReportValidationError(f"{kind} cannot allow an {status} claim")
        elif status in {"observed", "supported", "falsified"} and not claim_allowed:
            raise DiagnosticReportValidationError(f"{kind} must allow a non-unsupported {status} result")
        if kind == "observation" and status == "observed" and not evidence_refs:
            raise DiagnosticReportValidationError("observed claims require evidence references")


def _validate_target_evidence_shape(value: object, contract: object) -> None:
    """Validate the optional v2 target-evidence block without 80.4 semantics."""
    if value is None and contract is None:
        return
    if (value is None) != (contract is None):
        raise DiagnosticReportValidationError("target_evidence and evidence_contract must appear together")
    if contract not in ("diagnostic-evidence-v1", "diagnostic-evidence-v2"):
        raise DiagnosticReportValidationError("evidence_contract must be a known evidence version")
    if contract == "diagnostic-evidence-v1" and value is not None:
        raise DiagnosticReportValidationError("v1 reports must not declare target_evidence")
    block = _mapping(value, name="target_evidence")
    expected = frozenset(
        {
            "evidence_refs",
            "label_digest",
            "record_digest",
            "rule",
            "rule_digest",
            "rule_kind",
            "sample_digest",
            "target_id",
        }
    )
    _exact_keys(block, expected, name="target_evidence")
    for field in ("target_id", "rule", "rule_kind"):
        _string(block.get(field), name=f"target_evidence.{field}")
    for field in ("rule_digest", "label_digest", "sample_digest", "record_digest"):
        digest = block.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise DiagnosticReportValidationError(f"target_evidence.{field} must be a 64-character hex digest")
    refs = block.get("evidence_refs")
    if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence) or not refs:
        raise DiagnosticReportValidationError("target_evidence.evidence_refs must be non-empty")
    for item in refs:
        _string(item, name="target_evidence.evidence_refs")


def validate_report_shape(report: Mapping[str, object], *, schema: Mapping[str, object] | None = None) -> None:
    """Validate report structure and evidence-kind semantics without 80.4 checks."""

    if schema is None:
        load_schema()
    else:
        _validate_schema_shape(schema)
    actual = frozenset(report)
    if not actual.issuperset(_TOP_FIELDS) or not actual.issubset(_TOP_FIELDS | _OPTIONAL_TOP_FIELDS):
        raise DiagnosticReportValidationError(
            f"report fields are invalid (missing {sorted(_TOP_FIELDS - actual)}; "
            f"unexpected {sorted(actual - _TOP_FIELDS - _OPTIONAL_TOP_FIELDS)})"
        )
    _validate_target_evidence_shape(report.get("target_evidence"), report.get("evidence_contract"))
    _string(report.get("report_id"), name="report.report_id")
    _validate_capture_provenance(_mapping(report.get("capture_provenance"), name="capture_provenance"))
    _validate_item_list(
        report.get("symptoms"),
        name="symptoms",
        fields=frozenset({"id", "description", "metric_ids", "evidence_refs", "status"}),
    )
    _validate_item_list(
        report.get("localization"),
        name="localization",
        fields=frozenset({"id", "axis", "selection", "evidence_refs", "confidence", "status"}),
    )
    localization = report.get("localization")
    assert isinstance(localization, Sequence) and not isinstance(localization, (str, bytes))
    for index, item in enumerate(localization):
        row = _mapping(item, name=f"localization[{index}]")
        _string(row.get("axis"), name=f"localization[{index}].axis")
        _string(row.get("selection"), name=f"localization[{index}].selection")
        _string_list(row.get("evidence_refs"), name=f"localization[{index}].evidence_refs")
        confidence = row.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise DiagnosticReportValidationError(f"localization[{index}].confidence must be between zero and one")
        _validate_status(row.get("status"), name=f"localization[{index}].status")
    _validate_item_list(
        report.get("hypotheses"),
        name="hypotheses",
        fields=frozenset({"id", "statement", "evidence_refs", "alternatives", "status"}),
    )
    _validate_item_list(
        report.get("statistical_evidence"),
        name="statistical_evidence",
        fields=frozenset({"id", "metric_id", "estimate", "uncertainty", "control_refs", "evidence_refs", "status"}),
    )
    _validate_item_list(
        report.get("interventions"),
        name="interventions",
        fields=frozenset({"id", "intervention", "target", "control_refs", "outcome", "evidence_refs", "status"}),
    )
    _validate_item_list(
        report.get("comparisons"),
        name="comparisons",
        fields=frozenset({"id", "baseline", "candidate", "alignment", "metric_ids", "evidence_refs", "status"}),
    )
    _validate_item_list(
        report.get("limitations"),
        name="limitations",
        fields=frozenset({"id", "description", "affects", "blocking"}),
        minimum=1,
    )
    limitations = report.get("limitations")
    assert isinstance(limitations, Sequence) and not isinstance(limitations, (str, bytes))
    for index, item in enumerate(limitations):
        row = _mapping(item, name=f"limitations[{index}]")
        _string(row.get("description"), name=f"limitations[{index}].description")
        _string_list(row.get("affects"), name=f"limitations[{index}].affects", minimum=1)
        if not isinstance(row.get("blocking"), bool):
            raise DiagnosticReportValidationError(f"limitations[{index}].blocking must be boolean")
    next_action = _mapping(report.get("next_action"), name="next_action")
    _exact_keys(next_action, frozenset({"action", "rationale", "required_evidence_refs"}), name="next_action")
    _string(next_action.get("action"), name="next_action.action")
    _string(next_action.get("rationale"), name="next_action.rationale")
    _string_list(next_action.get("required_evidence_refs"), name="next_action.required_evidence_refs")
    _validate_claims(report.get("claims"))


__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "DiagnosticReportValidationError",
    "load_schema",
    "validate_report_shape",
]
