"""Consumer-observable tests for the frozen representation-problem taxonomy."""

from __future__ import annotations

from pathlib import Path

import pytest

from latent_anything._representation_taxonomy import (
    TAXONOMY_PATH,
    TAXONOMY_SCHEMA_VERSION,
    TaxonomyValidationError,
    evaluate_claim,
    load_taxonomy,
    validate_taxonomy,
)

EXPECTED_FAMILIES = {
    "collapse_rank_loss",
    "anisotropy_inactive_dimensions",
    "redundancy_superposition",
    "separability_probe_leakage",
    "density_ood_distribution_drift",
    "sparse_feature_instability",
    "sequence_trajectory_drift",
}


def _family(document: dict[str, object], family_id: str) -> dict[str, object]:
    families = document["families"]
    assert isinstance(families, list)
    for family in families:
        assert isinstance(family, dict)
        if family["id"] == family_id:
            return family
    raise AssertionError(f"missing family {family_id}")


def _observed_evidence(document: dict[str, object], family_id: str) -> dict[str, str]:
    family = _family(document, family_id)
    evidence = family["required_evidence"]
    assert isinstance(evidence, list)
    return {item["id"]: "observed" for item in evidence if isinstance(item, dict)}


def test_frozen_taxonomy_contains_all_problem_families_and_required_evidence() -> None:
    document = dict(load_taxonomy())

    assert document["schema_version"] == TAXONOMY_SCHEMA_VERSION
    assert Path(TAXONOMY_PATH).is_file()
    families = document["families"]
    assert isinstance(families, list)
    assert {family["id"] for family in families if isinstance(family, dict)} == EXPECTED_FAMILIES
    for family_id in EXPECTED_FAMILIES:
        family = _family(document, family_id)
        assert family["applicability"]["required_axes"]
        assert family["required_evidence"]
        assert family["unsupported_case"]["claim_allowed"] is False


def test_applicable_claim_requires_every_declared_evidence_item() -> None:
    document = dict(load_taxonomy())
    family_id = "collapse_rank_loss"
    evidence = _observed_evidence(document, family_id)

    decision = evaluate_claim(family_id, applicability="applicable", evidence_status=evidence, taxonomy=document)

    assert decision.outcome == "supported"
    assert decision.claim_allowed is True
    assert decision.missing_evidence == ()

    evidence.pop(next(iter(evidence)))
    incomplete = evaluate_claim(family_id, applicability="applicable", evidence_status=evidence, taxonomy=document)
    assert incomplete.outcome == "inconclusive"
    assert incomplete.claim_allowed is False
    assert incomplete.missing_evidence


@pytest.mark.parametrize("applicability", ["not_applicable", "unsupported"])
def test_non_applicable_and_unsupported_claims_fail_closed(applicability: str) -> None:
    document = dict(load_taxonomy())
    evidence = _observed_evidence(document, "sequence_trajectory_drift")

    decision = evaluate_claim(
        "sequence_trajectory_drift",
        applicability=applicability,
        evidence_status=evidence,
        taxonomy=document,
    )

    assert decision.outcome == "unsupported"
    assert decision.claim_allowed is False


def test_unknown_family_and_invalid_applicability_fail_closed() -> None:
    with pytest.raises(TaxonomyValidationError, match="unknown taxonomy family"):
        evaluate_claim("model_specific_detector", applicability="applicable", evidence_status={})
    with pytest.raises(TaxonomyValidationError, match="unsupported applicability state"):
        evaluate_claim("collapse_rank_loss", applicability="maybe", evidence_status={})


def test_taxonomy_rejects_mutated_version() -> None:
    document = dict(load_taxonomy())
    document["schema_version"] = "representation-problem-taxonomy-v0"

    with pytest.raises(TaxonomyValidationError, match="unsupported taxonomy schema version"):
        validate_taxonomy(document)
