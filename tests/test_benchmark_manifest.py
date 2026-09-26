"""Consumer-observable tests for the frozen benchmark-manifest contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from latent_anything._benchmark_manifest import (
    SCHEMA_VERSION,
    BenchmarkManifestValidationError,
    load_schema,
    manifest_digest,
    validate_manifest,
)


def _manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": "architecture-neutral-fixture",
        "status": "predeclared",
        "model": {
            "id": "model-reference",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "revision_kind": "commit",
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
            "declaration": "predeclared defect or counterexample description",
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
            {
                "id": "metric-2",
                "taxonomy_family_id": "density_ood_distribution_drift",
                "direction": "higher_is_better",
                "unit": "score",
                "estimator": "declared-estimator",
                "aggregation": "median",
            },
        ],
        "seeds": {"training": [11], "evaluation": [13], "controls": [17], "independent": True},
        "controls": [
            {
                "id": "control-null",
                "kind": "null",
                "metric_ids": ["metric-1", "metric-2"],
                "required": True,
                "expected_behavior": "no claimed effect",
            }
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
            {"metric_id": "metric-2", "comparator": ">=", "value": 0.7, "tolerance": 0.02, "predeclared": True},
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


def test_frozen_schema_names_all_required_benchmark_contract_sections() -> None:
    schema = load_schema()

    assert schema["schema_version"] == SCHEMA_VERSION
    assert schema["manifest_status"] == "predeclared"
    required = schema["required_fields"]
    assert isinstance(required, list)
    assert {
        "model",
        "dataset",
        "representation",
        "defect",
        "metrics",
        "seeds",
        "controls",
        "uncertainty",
        "causal_expectation",
        "thresholds",
        "commitment",
    }.issubset(required)


def test_complete_predeclared_manifest_validates() -> None:
    validate_manifest(_manifest())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["model"].pop("revision"), "model fields are invalid"),
        (lambda value: value.__setitem__("status", "completed"), "predeclared status"),
        (lambda value: value["thresholds"].pop(), "every metric must have exactly one"),
        (lambda value: value["controls"].clear(), "controls must be a non-empty list"),
    ],
)
def test_incomplete_or_post_hoc_manifest_fails_closed(mutation: object, message: str) -> None:
    value = _manifest()
    assert callable(mutation)
    mutation(value)  # type: ignore[operator]

    with pytest.raises(BenchmarkManifestValidationError, match=message):
        validate_manifest(value)


def test_mutating_locked_manifest_after_commitment_fails_closed() -> None:
    value = _manifest()
    mutated = deepcopy(value)
    metrics = mutated["metrics"]
    assert isinstance(metrics, list)
    assert isinstance(metrics[0], dict)
    metrics[0]["aggregation"] = "post-hoc-choice"

    with pytest.raises(BenchmarkManifestValidationError, match="digest does not match"):
        validate_manifest(mutated)


def test_non_applicable_causal_expectation_is_explicit_and_valid() -> None:
    value = _manifest()
    causal = value["causal_expectation"]
    assert isinstance(causal, dict)
    causal.update(
        {
            "applicable": False,
            "expectation": "not_applicable",
            "target_metric_ids": [],
            "falsification_rule": "not_applicable",
            "non_applicable_reason": "representation exposes no intervention axis",
        }
    )
    value["commitment"]["manifest_sha256"] = manifest_digest(value)  # type: ignore[index]

    validate_manifest(value)
