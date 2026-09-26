"""Consumer-observable tests for the independent diagnostic-report validator."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from latent_anything._benchmark_manifest import SCHEMA_VERSION as MANIFEST_SCHEMA_VERSION
from latent_anything._benchmark_manifest import manifest_digest
from latent_anything._diagnostic_report import SCHEMA_VERSION as REPORT_SCHEMA_VERSION
from latent_anything._diagnostic_validator import ReportValidationError, validate_diagnostic_report
from latent_anything._representation_taxonomy import load_taxonomy


def _manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "validator-fixture",
        "status": "predeclared",
        "model": {
            "id": "model-reference",
            "revision": "model-revision-immutable",
            "revision_kind": "digest",
            "revision_immutable": True,
            "artifact_digest": "a" * 64,
        },
        "dataset": {
            "id": "dataset-reference",
            "revision": "dataset-revision",
            "split": "heldout",
            "split_identity": "split-identity-1",
            "data_digest": "b" * 64,
        },
        "representation": {
            "identity": "representation-reference",
            "axes": [
                {"name": "layer", "selection": "layer-1", "identity": "layer-identity-1"},
                {"name": "slice", "selection": "all", "identity": "slice-identity-1"},
            ],
        },
        "defect": {
            "classification": "known",
            "declaration": "predeclared defect description",
            "predeclared": True,
            "source": "fixture-provenance",
        },
        "metrics": [
            {
                "id": "metric-1",
                "taxonomy_family_id": "collapse_rank_loss",
                "direction": "lower_is_better",
                "unit": "ratio",
                "estimator": "declared-estimator",
                "aggregation": "mean-with-interval",
            },
        ],
        "seeds": {"training": [11], "evaluation": [13], "controls": [17], "independent": True},
        "controls": [
            {
                "id": "control-required",
                "kind": "null",
                "metric_ids": ["metric-1"],
                "required": True,
                "expected_behavior": "no claimed effect",
            },
        ],
        "uncertainty": {
            "method": "bootstrap",
            "confidence_level": 0.95,
            "repetitions": 200,
            "unit_of_analysis": "sample",
            "predeclared": True,
        },
        "causal_expectation": {
            "applicable": True,
            "expectation": "intervention changes metric-1 in the declared direction",
            "target_metric_ids": ["metric-1"],
            "falsification_rule": "the expected direction is absent under the held-out evaluation",
            "non_applicable_reason": "not_applicable",
        },
        "thresholds": [
            {"metric_id": "metric-1", "comparator": "<", "value": 0.4, "tolerance": 0.01, "predeclared": True},
        ],
        "commitment": {
            "locked": True,
            "declared_at": "2026-09-21T00:00:00Z",
            "manifest_sha256": "",
            "canonicalization": "json-sort-keys-no-whitespace-utf8",
        },
    }
    commitment = manifest["commitment"]
    assert isinstance(commitment, dict)
    commitment["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def _report() -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": "validator-report",
        "status": "declared",
        "capture_provenance": {
            "captures": [
                {
                    "capture_id": "capture-1",
                    "model_revision": "model-revision-immutable",
                    "dataset_split": "split-identity-1",
                    "representation_identity": "representation-reference",
                    "axes": ["sample", "feature"],
                    "artifact_refs": ["capture-artifact-1"],
                },
            ],
        },
        "symptoms": [
            {
                "id": "symptom-1",
                "description": "Observed metric movement",
                "metric_ids": ["metric-1"],
                "evidence_refs": ["obs-1"],
                "status": "observed",
            },
        ],
        "localization": [
            {
                "id": "location-1",
                "axis": "layer",
                "selection": "layer-1",
                "evidence_refs": ["obs-1"],
                "confidence": 0.8,
                "status": "observed",
            },
        ],
        "hypotheses": [
            {
                "id": "hypothesis-1",
                "statement": "A declared representation defect explains the symptom",
                "evidence_refs": ["obs-1", "stats-1"],
                "alternatives": ["sampling artifact"],
                "status": "supported",
            },
        ],
        "statistical_evidence": [
            {
                "id": "stats-1",
                "metric_id": "metric-1",
                "estimate": 0.2,
                "uncertainty": {"kind": "interval", "lower": 0.1, "upper": 0.3},
                "control_refs": ["control-required"],
                "evidence_refs": ["stats-artifact-1"],
                "status": "supported",
            },
        ],
        "interventions": [
            {
                "id": "intervention-1",
                "intervention": "declared intervention",
                "target": "representation-reference",
                "control_refs": ["control-required"],
                "outcome": {"metric_delta": 0.4},
                "evidence_refs": ["causal-artifact-1"],
                "status": "supported",
            },
        ],
        "comparisons": [
            {
                "id": "comparison-1",
                "baseline": "run-a",
                "candidate": "run-b",
                "alignment": {"dataset_split": "same", "representation": "same"},
                "metric_ids": ["metric-1"],
                "evidence_refs": ["comparison-artifact-1"],
                "status": "supported",
            },
        ],
        "limitations": [
            {
                "id": "limitation-1",
                "description": "A declared evidence boundary remains",
                "affects": ["causal interpretation"],
                "blocking": False,
            },
        ],
        "next_action": {
            "action": "Collect the declared next evidence set",
            "rationale": "The next evidence gate is explicit",
            "required_evidence_refs": ["control-required"],
        },
        "claims": [
            {
                "id": "obs-1",
                "kind": "observation",
                "status": "observed",
                "claim": "The representation metric changed",
                "evidence_refs": ["capture-artifact-1"],
                "control_refs": [],
                "causal": False,
                "claim_allowed": True,
                "missing_evidence": [],
            },
            {
                "id": "explanation-1",
                "kind": "explanation",
                "status": "supported",
                "claim": "The hypothesis explains the observed symptom",
                "evidence_refs": ["stats-1"],
                "control_refs": ["control-required"],
                "causal": False,
                "claim_allowed": True,
                "missing_evidence": [],
            },
            {
                "id": "causal-1",
                "kind": "causal_result",
                "status": "supported",
                "claim": "The intervention produced the expected effect",
                "evidence_refs": ["intervention-1"],
                "control_refs": ["control-required"],
                "causal": True,
                "claim_allowed": True,
                "missing_evidence": [],
            },
            {
                "id": "unsupported-1",
                "kind": "unsupported_conclusion",
                "status": "unsupported",
                "claim": "A stronger conclusion is not supported",
                "evidence_refs": [],
                "control_refs": [],
                "causal": False,
                "claim_allowed": False,
                "missing_evidence": ["independent control"],
            },
        ],
    }


def _inputs(**overrides: object) -> dict[str, object]:
    document = load_taxonomy()
    families = document["families"]
    assert isinstance(families, list)
    evidence: dict[str, dict[str, str]] = {}
    for family in families:
        assert isinstance(family, dict)
        required = family["required_evidence"]
        assert isinstance(required, list)
        evidence[str(family["id"])] = {str(item["id"]): "observed" for item in required if isinstance(item, dict)}
    payloads = {
        "capture-artifact-1": b"capture-bytes",
        "stats-artifact-1": b"stats-bytes",
        "causal-artifact-1": b"causal-bytes",
        "comparison-artifact-1": b"comparison-bytes",
    }
    digests = {key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()}
    bundle: dict[str, object] = {
        "applicability": {"collapse_rank_loss": "applicable"},
        "family_evidence": evidence,
        "control_outcomes": {"control-required": "passed"},
        "artifacts": payloads,
        "artifact_digests": digests,
    }
    bundle.update(overrides)
    return bundle


def _validate(
    report: dict[str, object] | None = None, manifest: dict[str, object] | None = None, **overrides: object
) -> None:
    validate_diagnostic_report(
        report if report is not None else _report(),
        manifest if manifest is not None else _manifest(),
        **_inputs(**overrides),
    )  # type: ignore[arg-type]


def test_validator_accepts_declared_non_applicable_family_without_promotion() -> None:
    report = _report()
    for section in ("statistical_evidence", "interventions"):
        rows = report[section]
        assert isinstance(rows, list)
        row = rows[0]
        assert isinstance(row, dict)
        row["status"] = "inconclusive"
    claims = report["claims"]
    assert isinstance(claims, list)
    for entry in (claims[1], claims[2]):
        assert isinstance(entry, dict)
        entry["status"] = "inconclusive"
        entry["claim_allowed"] = False
    inputs = _inputs(applicability={"collapse_rank_loss": "not_applicable"})
    validate_diagnostic_report(report, _manifest(), **inputs)  # type: ignore[arg-type]


def test_validator_preserves_explicit_falsified_causal_result() -> None:
    report = _report()
    claims = report["claims"]
    assert isinstance(claims, list)
    causal = claims[2]
    assert isinstance(causal, dict)
    causal["status"] = "falsified"
    validate_diagnostic_report(report, _manifest(), **_inputs())  # type: ignore[arg-type]


def test_missing_required_control_reference_is_rejected() -> None:
    report = _report()
    for section in ("statistical_evidence", "interventions"):
        rows = report[section]
        assert isinstance(rows, list)
        row = rows[0]
        assert isinstance(row, dict)
        row["control_refs"] = ["control-undeclared"]
    claims = report["claims"]
    assert isinstance(claims, list)
    for entry in (claims[1], claims[2]):
        assert isinstance(entry, dict)
        entry["control_refs"] = ["control-undeclared"]
    with pytest.raises(ReportValidationError, match="unknown control reference"):
        _validate(report)


def test_unreferenced_required_control_is_rejected() -> None:
    report = _report()
    manifest = _manifest()
    controls = manifest["controls"]
    assert isinstance(controls, list)
    assert isinstance(controls[0], dict)
    controls.append(
        {
            "id": "control-extra",
            "kind": "null",
            "metric_ids": ["metric-1"],
            "required": True,
            "expected_behavior": "no claimed effect",
        }
    )
    commitment = manifest["commitment"]
    assert isinstance(commitment, dict)
    commitment["manifest_sha256"] = manifest_digest(manifest)
    with pytest.raises(ReportValidationError, match="required control"):
        validate_diagnostic_report(report, manifest, **_inputs())  # type: ignore[arg-type]


def test_failed_required_control_blocks_supported_conclusion() -> None:
    with pytest.raises(ReportValidationError, match="blocks"):
        _validate(control_outcomes={"control-required": "failed"})


def test_missing_control_outcome_is_rejected() -> None:
    with pytest.raises(ReportValidationError, match="control outcome"):
        _validate(control_outcomes={})


def test_mismatched_provenance_is_rejected() -> None:
    report = _report()
    capture_provenance = report["capture_provenance"]
    assert isinstance(capture_provenance, dict)
    captures = capture_provenance["captures"]
    assert isinstance(captures, list)
    capture = captures[0]
    assert isinstance(capture, dict)
    capture["model_revision"] = "another-revision"
    with pytest.raises(ReportValidationError, match="provenance mismatch"):
        _validate(report)


def test_artifact_digest_mismatch_is_rejected() -> None:
    report = _report()
    capture_provenance = report["capture_provenance"]
    assert isinstance(capture_provenance, dict)
    captures = capture_provenance["captures"]
    assert isinstance(captures, list)
    capture = captures[0]
    assert isinstance(capture, dict)
    capture["artifact_refs"] = ["tampered-artifact"]
    payloads = {"tampered-artifact": b"actual-bytes"}
    digests = {"tampered-artifact": hashlib.sha256(b"other-bytes").hexdigest()}
    with pytest.raises(ReportValidationError, match="digest mismatch"):
        _validate(report, artifacts=payloads, artifact_digests=digests)


def test_missing_artifact_payload_is_rejected() -> None:
    with pytest.raises(ReportValidationError, match="is missing"):
        _validate(artifacts={}, artifact_digests={"capture-artifact-1": hashlib.sha256(b"capture-bytes").hexdigest()})


def test_incomplete_family_evidence_is_rejected() -> None:
    document = load_taxonomy()
    families = document["families"]
    assert isinstance(families, list)
    first = families[0]
    assert isinstance(first, dict)
    required = first["required_evidence"]
    assert isinstance(required, list)
    assert isinstance(required[0], dict)
    evidence: dict[str, dict[str, str]] = {
        "collapse_rank_loss": {str(required[0]["id"]): "observed"},
    }
    with pytest.raises(ReportValidationError, match="required evidence"):
        _validate(family_evidence=evidence)


def test_unknown_taxonomy_reference_is_rejected() -> None:
    manifest = _manifest()
    metrics = manifest["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[0]
    assert isinstance(metric, dict)
    metric["taxonomy_family_id"] = "not_a_family"
    commitment = manifest["commitment"]
    assert isinstance(commitment, dict)
    commitment["manifest_sha256"] = manifest_digest(manifest)
    with pytest.raises(ReportValidationError, match="unknown taxonomy family"):
        _validate(manifest=manifest)


def test_unknown_metric_reference_is_rejected() -> None:
    report = _report()
    symptoms = report["symptoms"]
    assert isinstance(symptoms, list)
    symptom = symptoms[0]
    assert isinstance(symptom, dict)
    symptom["metric_ids"] = ["metric-unknown"]
    with pytest.raises(ReportValidationError, match="unknown metric reference"):
        _validate(report)


def test_mismatched_taxonomy_axes_are_rejected() -> None:
    report = deepcopy(_report())
    capture_provenance = report["capture_provenance"]
    assert isinstance(capture_provenance, dict)
    captures = capture_provenance["captures"]
    assert isinstance(captures, list)
    capture = captures[0]
    assert isinstance(capture, dict)
    capture["axes"] = ["feature"]
    with pytest.raises(ReportValidationError, match="applicability mismatch"):
        _validate(report)


def test_overclaimed_conclusion_without_measurement_support_is_rejected() -> None:
    report = _report()
    claims = report["claims"]
    assert isinstance(claims, list)
    supported = claims[1]
    assert isinstance(supported, dict)
    supported["evidence_refs"] = ["obs-1"]
    with pytest.raises(ReportValidationError, match="stronger than its evidence supports"):
        _validate(report)


def test_inconclusive_result_is_preserved_without_promotion() -> None:
    report = _report()
    claims = report["claims"]
    assert isinstance(claims, list)
    for entry in (claims[1], claims[2]):
        assert isinstance(entry, dict)
        entry["status"] = "inconclusive"
        entry["claim_allowed"] = False
    statistical = report["statistical_evidence"]
    assert isinstance(statistical, list)
    row = statistical[0]
    assert isinstance(row, dict)
    row["status"] = "inconclusive"
    interventions = report["interventions"]
    assert isinstance(interventions, list)
    intervention = interventions[0]
    assert isinstance(intervention, dict)
    intervention["status"] = "inconclusive"
    _validate(report)
