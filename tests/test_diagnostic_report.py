"""Consumer-observable tests for the frozen diagnostic-report schema."""

from __future__ import annotations

from copy import deepcopy

import pytest

from latent_anything._diagnostic_report import (
    SCHEMA_VERSION,
    DiagnosticReportValidationError,
    load_schema,
    validate_report_shape,
)


def _report() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report_id": "report-fixture",
        "status": "declared",
        "capture_provenance": {
            "captures": [
                {
                    "capture_id": "capture-1",
                    "model_revision": "model@immutable-revision",
                    "dataset_split": "dataset@split",
                    "representation_identity": "representation@layer-1",
                    "axes": ["sample", "feature"],
                    "artifact_refs": ["capture-artifact-1"],
                }
            ]
        },
        "symptoms": [
            {
                "id": "symptom-1",
                "description": "Observed metric movement",
                "metric_ids": ["metric-1"],
                "evidence_refs": ["obs-1"],
                "status": "observed",
            }
        ],
        "localization": [
            {
                "id": "location-1",
                "axis": "layer",
                "selection": "layer-1",
                "evidence_refs": ["obs-1"],
                "confidence": 0.8,
                "status": "observed",
            }
        ],
        "hypotheses": [
            {
                "id": "hypothesis-1",
                "statement": "A declared representation defect explains the symptom",
                "evidence_refs": ["obs-1", "stats-1"],
                "alternatives": ["sampling artifact"],
                "status": "supported",
            }
        ],
        "statistical_evidence": [
            {
                "id": "stats-1",
                "metric_id": "metric-1",
                "estimate": 0.2,
                "uncertainty": {"kind": "interval", "lower": 0.1, "upper": 0.3},
                "control_refs": ["control-1"],
                "evidence_refs": ["stats-artifact-1"],
                "status": "supported",
            }
        ],
        "interventions": [
            {
                "id": "intervention-1",
                "intervention": "declared intervention",
                "target": "representation@layer-1",
                "control_refs": ["control-1"],
                "outcome": {"metric_delta": 0.4},
                "evidence_refs": ["causal-artifact-1"],
                "status": "supported",
            }
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
            }
        ],
        "limitations": [
            {
                "id": "limitation-1",
                "description": "A declared evidence boundary remains",
                "affects": ["causal interpretation"],
                "blocking": False,
            }
        ],
        "next_action": {
            "action": "Collect the declared missing control",
            "rationale": "The next evidence gate is explicit",
            "required_evidence_refs": ["control-1"],
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
                "evidence_refs": ["obs-1", "stats-1"],
                "control_refs": [],
                "causal": False,
                "claim_allowed": True,
                "missing_evidence": [],
            },
            {
                "id": "causal-1",
                "kind": "causal_result",
                "status": "falsified",
                "claim": "The intervention did not produce the expected effect",
                "evidence_refs": ["causal-artifact-1"],
                "control_refs": ["control-1"],
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


def test_frozen_schema_names_all_report_sections_and_evidence_kinds() -> None:
    schema = load_schema()

    assert schema["schema_version"] == SCHEMA_VERSION
    assert set(schema["required_fields"]) == {
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
    assert set(schema["evidence_kinds"]) == {
        "observation",
        "explanation",
        "causal_result",
        "unsupported_conclusion",
    }


def test_report_shape_accepts_distinct_observation_explanation_causal_and_unsupported_claims() -> None:
    validate_report_shape(_report())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["claims"][1].__setitem__("causal", True), "explanations cannot be causal"),
        (lambda value: value["claims"][2].__setitem__("causal", False), "causal_result must declare causal=true"),
        (
            lambda value: value["claims"][3].__setitem__("claim_allowed", True),
            "unsupported conclusions must be explicit",
        ),
        (lambda value: value.pop("localization"), "report fields are invalid"),
    ],
)
def test_malformed_or_conflated_report_fails_closed(mutation: object, message: str) -> None:
    value = _report()
    assert callable(mutation)
    mutation(value)  # type: ignore[operator]

    with pytest.raises(DiagnosticReportValidationError, match=message):
        validate_report_shape(value)


def test_unsupported_conclusion_requires_explicit_missing_evidence() -> None:
    value = deepcopy(_report())
    claims = value["claims"]
    assert isinstance(claims, list)
    assert isinstance(claims[3], dict)
    claims[3]["missing_evidence"] = []

    with pytest.raises(DiagnosticReportValidationError, match="unsupported conclusions"):
        validate_report_shape(value)
