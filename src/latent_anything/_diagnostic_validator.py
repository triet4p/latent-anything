"""Independent validation for frozen diagnostic reports (Sprint 80.4).

This module consumes the frozen representation-problem taxonomy (80.1),
benchmark-manifest schema (80.2), and diagnostic-report schema (80.3)
without changing their semantics and without constructing reports. It
validates an externally supplied machine-readable report together with
externally supplied evidence, control outcomes, and artifacts.

A report is accepted only when every check below holds:

- taxonomy references resolve: manifest metric families are known
  taxonomy families, and every metric referenced by the report is
  declared in the manifest;
- evidence is complete: every family used by the manifest has all of
  its required evidence observed, and every evidence reference resolves
  to a known section entry or a verified artifact;
- controls are present and passing: every required manifest control is
  referenced by the report, every control reference is declared, every
  required control has a recorded outcome, and failed controls block the
  associated supported conclusion;
- artifacts verify: every capture artifact has a declared digest and a
  matching payload, and digest mismatches or missing payloads reject;
- provenance agrees: every capture repeats the manifest model revision,
  dataset split identity, and representation identity;
- conclusions match evidence strength: supported explanations and causal
  results require complete family evidence, supporting measurement rows,
  passed controls, and no admitted missing evidence. Explicit
  unsupported, inconclusive, non-applicable, and falsified results are
  preserved without being promoted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import cast

from latent_anything._benchmark_manifest import validate_manifest
from latent_anything._diagnostic_report import validate_report_shape
from latent_anything._representation_taxonomy import (
    evaluate_claim,
    load_taxonomy,
    validate_taxonomy,
)
from latent_anything._run_record_codec import canonical_json

_PASSED = "passed"
_FAILED = "failed"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_APPLICABILITY_STATES = frozenset({"applicable", "not_applicable", "unsupported"})
_PROMOTED_STATUSES = frozenset({"observed", "supported"})


class ReportValidationError(ValueError):
    """Raised when a diagnostic report fails independent validation."""


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.match(value) is None:
        raise ReportValidationError(f"{name} is malformed")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReportValidationError(f"{name} must be a non-empty string")
    return value


def _string_list(value: object, *, name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReportValidationError(f"{name} must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ReportValidationError(f"{name} must be a list of strings")
        result.append(item)
    return result


def _rows(report: Mapping[str, object], section: str) -> list[Mapping[str, object]]:
    value = report.get(section)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReportValidationError(f"{section} must be a list")
    return [_mapping(item, name=f"{section}[{index}]") for index, item in enumerate(value)]


def _row_id(row: Mapping[str, object], *, section: str) -> str:
    return _string(row.get("id"), name=f"{section}.id")


def _metric_ids_of(row: Mapping[str, object]) -> list[str]:
    if "metric_id" in row:
        return [_string(row.get("metric_id"), name="metric reference")]
    if "metric_ids" in row:
        return _string_list(row.get("metric_ids"), name="metric references")
    return []


def _claim_metrics(
    claim: Mapping[str, object],
    by_id: Mapping[str, list[tuple[str, Mapping[str, object]]]],
) -> set[str]:
    """Collect metric ids linked to a claim through evidence references."""
    linked: set[str] = set()
    visited: set[str] = set()
    queue = _string_list(claim.get("evidence_refs"), name="claim.evidence_refs")
    while queue:
        reference = queue.pop()
        if reference in visited:
            continue
        visited.add(reference)
        for section, row in by_id.get(reference, ()):
            linked.update(_metric_ids_of(row))
            if section == "claims":
                queue.extend(_string_list(row.get("evidence_refs"), name="claim.evidence_refs"))
    return linked


def validate_diagnostic_report(
    report: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    taxonomy: Mapping[str, object] | None = None,
    applicability: Mapping[str, str] | None = None,
    family_evidence: Mapping[str, Mapping[str, str]] | None = None,
    control_outcomes: Mapping[str, str] | None = None,
    artifacts: Mapping[str, bytes] | None = None,
    artifact_digests: Mapping[str, str] | None = None,
    target_evidence: Mapping[str, object] | None = None,
    target_provenance: Mapping[str, object] | None = None,
) -> None:
    """Validate an externally supplied report against frozen contracts.

    ``family_evidence`` maps each taxonomy family to its evidence-id
    status map, ``applicability`` maps each family to its declared
    applicability state, ``control_outcomes`` maps each manifest control
    to ``"passed"`` or ``"failed"``, ``artifacts`` carries artifact
    payloads by reference, and ``artifact_digests`` carries the expected
    SHA-256 hex digest per artifact reference. ``target_evidence`` carries
    the v2 content-addressed target-label record; ``target_provenance`` is
    the matching evaluated-target provenance from the detect-stage payload.
    Under the legacy v1 contract both v2 inputs must be absent.
    """

    validate_report_shape(report)
    validate_manifest(manifest)
    document = load_taxonomy() if taxonomy is None else taxonomy
    validate_taxonomy(document)

    families: dict[str, Mapping[str, object]] = {}
    required_axes: dict[str, tuple[str, ...]] = {}
    raw_families = document.get("families")
    assert isinstance(raw_families, Sequence) and not isinstance(raw_families, (str, bytes))
    for item in raw_families:
        family = _mapping(item, name="family")
        family_id = _string(family.get("id"), name="family.id")
        families[family_id] = family
        axes = _mapping(family.get("applicability"), name=f"{family_id}.applicability").get("required_axes")
        if isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)):
            required_axes[family_id] = tuple(entry for entry in axes if isinstance(entry, str))
        else:
            required_axes[family_id] = ()

    metric_family: dict[str, str] = {}
    raw_metrics = manifest.get("metrics")
    assert isinstance(raw_metrics, Sequence) and not isinstance(raw_metrics, (str, bytes))
    for raw_metric in raw_metrics:
        metric = _mapping(raw_metric, name="metric")
        metric_id = _string(metric.get("id"), name="metric.id")
        family_id = _string(metric.get("taxonomy_family_id"), name="metric.taxonomy_family_id")
        if family_id not in families:
            raise ReportValidationError(f"unknown taxonomy family: {family_id}")
        metric_family[metric_id] = family_id

    manifest_controls: dict[str, set[str]] = {}
    required_controls: set[str] = set()
    raw_controls = manifest.get("controls")
    assert isinstance(raw_controls, Sequence) and not isinstance(raw_controls, (str, bytes))
    for raw_control in raw_controls:
        control = _mapping(raw_control, name="control")
        control_id = _string(control.get("id"), name="control.id")
        linked = set(_string_list(control.get("metric_ids"), name="control.metric_ids"))
        manifest_controls[control_id] = linked
        if control.get("required") is True:
            required_controls.add(control_id)

    applied_applicability: dict[str, str] = {}
    for family_id in metric_family.values():
        state = "applicable" if applicability is None else applicability.get(family_id, "applicable")
        if state not in _APPLICABILITY_STATES:
            raise ReportValidationError(f"unsupported applicability state: {state!r}")
        applied_applicability[family_id] = state
    if applicability is not None:
        for family_id in applicability:
            if family_id not in families:
                raise ReportValidationError(f"unknown taxonomy family: {family_id}")

    evidence = dict(family_evidence) if family_evidence is not None else {}
    for family_id in evidence:
        if family_id not in families:
            raise ReportValidationError(f"unknown taxonomy family: {family_id}")

    outcomes = dict(control_outcomes) if control_outcomes is not None else {}
    for control_id, outcome in outcomes.items():
        if control_id not in manifest_controls:
            raise ReportValidationError(f"unknown control reference: {control_id}")
        if outcome not in {_PASSED, _FAILED}:
            raise ReportValidationError(f"control outcome for {control_id} must be passed or failed")

    digests = dict(artifact_digests) if artifact_digests is not None else {}
    for reference, digest in digests.items():
        _require_digest(digest, name=f"artifact digest for {reference}")
    payloads = dict(artifacts) if artifacts is not None else {}

    by_id: dict[str, list[tuple[str, Mapping[str, object]]]] = {}
    for section in (
        "symptoms",
        "localization",
        "hypotheses",
        "statistical_evidence",
        "interventions",
        "comparisons",
        "claims",
    ):
        for row in _rows(report, section):
            by_id.setdefault(_row_id(row, section=section), []).append((section, row))

    for section in ("symptoms", "statistical_evidence", "comparisons"):
        for row in _rows(report, section):
            for metric_id in _metric_ids_of(row):
                if metric_id not in metric_family:
                    raise ReportValidationError(f"unknown metric reference: {metric_id}")

    for section in ("statistical_evidence", "interventions", "claims"):
        for row in _rows(report, section):
            for control_id in _string_list(row.get("control_refs"), name=f"{section}.control_refs"):
                if control_id not in manifest_controls:
                    raise ReportValidationError(f"unknown control reference: {control_id}")

    def _resolve_evidence(reference: str) -> None:
        if reference in by_id:
            return
        expected = digests.get(reference)
        if expected is None:
            raise ReportValidationError(f"unknown evidence reference: {reference}")
        payload = payloads.get(reference)
        if payload is None:
            raise ReportValidationError(f"artifact {reference} is missing")
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ReportValidationError(f"artifact digest mismatch for {reference}")

    capture_provenance = _mapping(report.get("capture_provenance"), name="capture_provenance")
    raw_captures = capture_provenance.get("captures")
    assert isinstance(raw_captures, Sequence) and not isinstance(raw_captures, (str, bytes))
    model = _mapping(manifest.get("model"), name="model")
    dataset = _mapping(manifest.get("dataset"), name="dataset")
    representation = _mapping(manifest.get("representation"), name="representation")
    expected_revision = _string(model.get("revision"), name="model.revision")
    expected_split = _string(dataset.get("split_identity"), name="dataset.split_identity")
    expected_identity = _string(representation.get("identity"), name="representation.identity")
    for index, raw_capture in enumerate(raw_captures):
        capture = _mapping(raw_capture, name=f"captures[{index}]")
        if capture.get("model_revision") != expected_revision:
            raise ReportValidationError(f"provenance mismatch for captures[{index}].model_revision")
        if capture.get("dataset_split") != expected_split:
            raise ReportValidationError(f"provenance mismatch for captures[{index}].dataset_split")
        if capture.get("representation_identity") != expected_identity:
            raise ReportValidationError(f"provenance mismatch for captures[{index}].representation_identity")
        for reference in _string_list(capture.get("artifact_refs"), name=f"captures[{index}].artifact_refs"):
            _resolve_evidence(reference)

    for section in (
        "symptoms",
        "localization",
        "hypotheses",
        "statistical_evidence",
        "interventions",
        "comparisons",
        "claims",
    ):
        for row in _rows(report, section):
            if "evidence_refs" in row:
                for reference in _string_list(row.get("evidence_refs"), name=f"{section}.evidence_refs"):
                    _resolve_evidence(reference)

    referenced_controls: set[str] = set()
    for section in ("statistical_evidence", "interventions", "claims"):
        for row in _rows(report, section):
            referenced_controls.update(_string_list(row.get("control_refs"), name=f"{section}.control_refs"))
    for control_id in sorted(required_controls):
        if control_id not in referenced_controls:
            raise ReportValidationError(f"required control {control_id} is missing")
        if control_id not in outcomes:
            raise ReportValidationError(f"control outcome for {control_id} is missing")

    for section in ("statistical_evidence", "interventions"):
        for row in _rows(report, section):
            if row.get("status") != "supported":
                continue
            for control_id in _string_list(row.get("control_refs"), name=f"{section}.control_refs"):
                if outcomes.get(control_id) == _FAILED:
                    raise ReportValidationError(f"failed control {control_id} blocks {section} {row.get('id')}")
                if control_id not in outcomes:
                    raise ReportValidationError(f"control outcome for {control_id} is missing")

    incomplete_families: set[str] = set()
    mismatched_families: set[str] = set()
    for family_id in sorted(set(metric_family.values())):
        status = evidence.get(family_id)
        if status is None:
            incomplete_families.add(family_id)
            continue
        decision = evaluate_claim(
            family_id,
            applicability=applied_applicability[family_id],
            evidence_status=dict(status),
            taxonomy=document,
        )
        if not decision.claim_allowed or decision.outcome != "supported":
            incomplete_families.add(family_id)
    captures = [_mapping(item, name="capture") for item in raw_captures]
    target_block = report.get("target_evidence")
    resolved_target: Mapping[str, object] | None = None
    if target_block is not None or target_evidence is not None:
        from latent_anything._target_evidence import (
            ACTIVATION_AXES,
            EVIDENCE_CONTRACT_V2,
            PROVENANCE_TOKENS,
            TargetEvidenceError,
            validate_target_record,
        )

        declared = _mapping(target_block, name="target_evidence")
        supplied = _mapping(target_evidence, name="target_evidence input")
        record = {key: value for key, value in supplied.items() if key != "record_digest"}
        for key in ("target_id", "rule", "rule_digest", "rule_kind", "label_digest", "sample_digest", "record_digest"):
            if _string(declared.get(key), name=f"target_evidence.{key}") != _string(
                supplied.get(key), name=f"target input.{key}"
            ):
                raise ReportValidationError(f"target evidence {key} disagrees with the validated record")
        try:
            validate_target_record(record)
        except TargetEvidenceError as exc:
            raise ReportValidationError(f"target evidence is invalid: {exc}") from exc
        if _string(report.get("evidence_contract"), name="report.evidence_contract") != EVIDENCE_CONTRACT_V2:
            raise ReportValidationError("target evidence must declare the v2 evidence contract")
        if target_provenance is None:
            raise ReportValidationError("v2 target evidence requires evaluated detect-stage provenance")
        evaluated = _mapping(target_provenance, name="evaluated target provenance")
        for key in (
            "capacity",
            "eval_indices",
            "eval_split_identity",
            "label_digest",
            "labels",
            "rule",
            "rule_kind",
            "sample_digest",
            "sample_ids",
            "target_id",
            "train_indices",
            "train_split_identity",
        ):
            if record.get(key) != evaluated.get(key):
                raise ReportValidationError(f"target evidence does not match evaluated target provenance at {key}")
        record_digest = _require_digest(supplied.get("record_digest"), name="target record digest")
        record_bytes = canonical_json(record) + b"\n"
        if record_digest != hashlib.sha256(record_bytes).hexdigest():
            raise ReportValidationError("target record digest does not match the validated record bytes")
        target_ref = f"target-{_string(record.get('target_id'), name='target_id')}-record"
        target_refs = _string_list(declared.get("evidence_refs"), name="target_evidence.evidence_refs")
        if target_refs != [target_ref]:
            raise ReportValidationError("target evidence must reference its single content-addressed record")
        _resolve_evidence(target_ref)
        if digests.get(target_ref) != record_digest or payloads.get(target_ref) != record_bytes:
            raise ReportValidationError("target record blob disagrees with its content-addressed evidence")
        expected_split = _string(dataset.get("split_identity"), name="dataset.split_identity")
        if _string(record.get("dataset_split_identity"), name="target dataset split") != expected_split:
            raise ReportValidationError("target dataset split disagrees with the manifest")
        expected_representation = _string(representation.get("identity"), name="representation.identity")
        if _string(record.get("representation_identity"), name="target representation") != expected_representation:
            raise ReportValidationError("target representation disagrees with the manifest")
        raw_representation_axes = representation.get("axes")
        if not isinstance(raw_representation_axes, Sequence) or isinstance(raw_representation_axes, (str, bytes)):
            raise ReportValidationError("manifest representation axes are invalid")
        declared_axes = {
            _string(_mapping(axis, name=f"representation.axes[{index}]").get("name"), name="axis.name")
            for index, axis in enumerate(raw_representation_axes)
        }
        allowed_capture_axes = declared_axes | {"feature"}
        resolved_target = record
        for index, capture in enumerate(captures):
            capture_axis_names = _string_list(capture.get("axes"), name=f"captures[{index}].axes")
            axes = set(capture_axis_names)
            smuggled = axes & set(PROVENANCE_TOKENS)
            if smuggled:
                raise ReportValidationError(f"capture axes must not alias target provenance: {sorted(smuggled)}")
            unknown = axes - set(ACTIVATION_AXES)
            if unknown:
                raise ReportValidationError(f"capture axes contain unknown axis: {sorted(unknown)}")
            undeclared = axes - allowed_capture_axes
            if undeclared:
                raise ReportValidationError(
                    f"capture axes are not declared by the manifest or binder: {sorted(undeclared)}"
                )
            capture_references = _string_list(
                capture.get("artifact_refs"),
                name=f"captures[{index}].artifact_refs",
            )
            if not capture_references:
                raise ReportValidationError(f"captures[{index}] must reference its bound capture record")
            for reference in capture_references:
                capture_bytes = payloads.get(reference)
                if not isinstance(capture_bytes, bytes):
                    raise ReportValidationError(f"bound capture artifact {reference!r} is missing")
                try:
                    capture_document = json.loads(capture_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ReportValidationError(f"bound capture artifact {reference!r} is not valid JSON") from exc
                bound_payload = _mapping(capture_document, name=f"capture artifact {reference}")
                bound_capture = _mapping(bound_payload.get("capture"), name=f"capture artifact {reference}.capture")
                bound_axes = _string_list(
                    bound_capture.get("axes"),
                    name=f"capture artifact {reference}.capture.axes",
                )
                if capture_axis_names != bound_axes:
                    raise ReportValidationError(
                        f"capture axes disagree with the registered bound capture at captures[{index}]"
                    )
                for report_key, bound_key in (
                    ("capture_id", "capture_id"),
                    ("dataset_split", "split_identity"),
                    ("model_revision", "model_revision"),
                    ("representation_identity", "representation_identity"),
                ):
                    if capture.get(report_key) != bound_capture.get(bound_key):
                        raise ReportValidationError(
                            f"capture provenance disagrees with its bound artifact at captures[{index}].{report_key}"
                        )
    for family_id in sorted(set(metric_family.values())):
        if applied_applicability[family_id] != "applicable":
            continue
        needed = set(required_axes.get(family_id, ()))
        for index, capture in enumerate(captures):
            axes = set(_string_list(capture.get("axes"), name=f"captures[{index}].axes"))
            if resolved_target is not None:
                # Target evidence supplies distinct sample identity and label
                # provenance; neither is fabricated as an activation-array axis.
                axes |= {"sample", "label"}
            if not needed.issubset(axes):
                mismatched_families.add(family_id)

    failed_controls = {control_id for control_id, outcome in outcomes.items() if outcome == _FAILED}
    causal = _mapping(manifest.get("causal_expectation"), name="causal_expectation")
    causal_applicable = causal.get("applicable") is True

    for claim in _rows(report, "claims"):
        kind = _string(claim.get("kind"), name="claim.kind")
        status = _string(claim.get("status"), name="claim.status")
        claim_allowed = claim.get("claim_allowed")
        if kind == "unsupported_conclusion" or status not in _PROMOTED_STATUSES or claim_allowed is not True:
            continue
        if _string_list(claim.get("missing_evidence"), name="claim.missing_evidence"):
            raise ReportValidationError(f"claim {claim.get('id')} is stronger than its evidence supports")
        linked = _claim_metrics(claim, by_id)
        linked_families = sorted({metric_family[metric_id] for metric_id in linked if metric_id in metric_family})
        if kind in {"explanation", "causal_result"}:
            if kind == "explanation" and not linked:
                raise ReportValidationError(f"claim {claim.get('id')} is stronger than its evidence supports")
            families_to_gate = linked_families or sorted(set(metric_family.values()))
            for family_id in families_to_gate:
                if family_id in incomplete_families:
                    raise ReportValidationError(f"required evidence for {family_id} is incomplete")
                if family_id in mismatched_families:
                    raise ReportValidationError(f"taxonomy applicability mismatch for {family_id}")
        if kind == "observation":
            for control_id in _string_list(claim.get("control_refs"), name="claim.control_refs"):
                if outcomes.get(control_id) == _FAILED:
                    raise ReportValidationError(f"failed control {control_id} blocks claim {claim.get('id')}")
            continue
        supporting_measurements = 0
        weak_reference = False
        for reference in _string_list(claim.get("evidence_refs"), name="claim.evidence_refs"):
            for section, row in by_id.get(reference, []):
                if section in {"statistical_evidence", "interventions", "comparisons"}:
                    if row.get("status") in _PROMOTED_STATUSES:
                        supporting_measurements += 1
                    else:
                        weak_reference = True
        if weak_reference or supporting_measurements == 0:
            raise ReportValidationError(f"claim {claim.get('id')} is stronger than its evidence supports")
        if kind == "causal_result":
            if not causal_applicable:
                raise ReportValidationError(f"claim {claim.get('id')} is stronger than its evidence supports")
            intervention_support = any(
                section == "interventions" and row.get("status") == "supported"
                for reference in _string_list(claim.get("evidence_refs"), name="claim.evidence_refs")
                for section, row in by_id.get(reference, [])
            )
            if not intervention_support:
                raise ReportValidationError(f"claim {claim.get('id')} is stronger than its evidence supports")
        for control_id in _string_list(claim.get("control_refs"), name="claim.control_refs"):
            if outcomes.get(control_id) != _PASSED:
                raise ReportValidationError(f"claim {claim.get('id')} is stronger than its evidence supports")
        for control_id in sorted(failed_controls):
            if manifest_controls[control_id] & linked or not linked:
                raise ReportValidationError(f"failed control {control_id} blocks claim {claim.get('id')}")


__all__ = [
    "ReportValidationError",
    "validate_diagnostic_report",
]
