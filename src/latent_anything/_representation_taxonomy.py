"""Validation helpers for the frozen representation-problem taxonomy.

The taxonomy is deliberately data-driven: this module does not select a model,
metric, detector, or architecture. It only validates the shared contract and
ensures unsupported or non-applicable claims cannot be promoted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from latent_anything._artifact_path import resolve_artifact_path

TAXONOMY_SCHEMA_VERSION = "representation-problem-taxonomy-v1"
TAXONOMY_PATH = resolve_artifact_path("representation_problem_taxonomy_v1.json")

_EXPECTED_FAMILY_IDS = (
    "collapse_rank_loss",
    "anisotropy_inactive_dimensions",
    "redundancy_superposition",
    "separability_probe_leakage",
    "density_ood_distribution_drift",
    "sparse_feature_instability",
    "sequence_trajectory_drift",
)
_APPLICABILITY_STATES = frozenset({"applicable", "not_applicable", "unsupported"})
_EVIDENCE_STATUSES = frozenset({"observed", "missing", "invalid", "not_evaluated"})


class TaxonomyValidationError(ValueError):
    """Raised when the frozen taxonomy or a claim evaluation is malformed."""


ClaimOutcome = Literal["supported", "inconclusive", "unsupported"]


@dataclass(frozen=True)
class ClaimDecision:
    """Consumer-visible result of applying taxonomy evidence semantics."""

    outcome: ClaimOutcome
    claim_allowed: bool
    missing_evidence: tuple[str, ...]
    reason: str


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TaxonomyValidationError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _require_string_map(value: object, *, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TaxonomyValidationError(f"{name} must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TaxonomyValidationError(f"{name} keys and values must be strings")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TaxonomyValidationError(f"{name} must be a non-empty string")
    return value


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaxonomyValidationError(f"{name} must be a sequence of strings")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise TaxonomyValidationError(f"{name} must contain only non-empty strings")
    return result


def _validate_case(value: object, *, name: str) -> None:
    case = _mapping(value, name=name)
    if case.get("outcome") != "unsupported" or case.get("claim_allowed") is not False:
        raise TaxonomyValidationError(f"{name} must fail closed")
    _string(case.get("evidence_status"), name=f"{name}.evidence_status")


def validate_taxonomy(document: Mapping[str, object]) -> None:
    """Validate structural and fail-closed invariants of a taxonomy document."""

    if document.get("schema_version") != TAXONOMY_SCHEMA_VERSION:
        raise TaxonomyValidationError("unsupported taxonomy schema version")
    if document.get("taxonomy_id") != "representation-problem" or document.get("status") != "frozen":
        raise TaxonomyValidationError("taxonomy identity or status is invalid")
    if _string_sequence(document.get("applicability_states"), name="applicability_states") != (
        "applicable",
        "not_applicable",
        "unsupported",
    ):
        raise TaxonomyValidationError("applicability states are not stable")
    evidence_statuses = set(_string_sequence(document.get("evidence_statuses"), name="evidence_statuses"))
    if evidence_statuses != set(_EVIDENCE_STATUSES):
        raise TaxonomyValidationError("evidence statuses are incomplete")
    _validate_case(document.get("unsupported_case"), name="unsupported_case")
    _validate_case(document.get("not_applicable_case"), name="not_applicable_case")

    raw_families = document.get("families")
    if isinstance(raw_families, (str, bytes)) or not isinstance(raw_families, Sequence):
        raise TaxonomyValidationError("families must be a sequence")
    families = tuple(_mapping(item, name="family") for item in raw_families)
    family_ids = tuple(_string(family.get("id"), name="family.id") for family in families)
    if family_ids != _EXPECTED_FAMILY_IDS:
        raise TaxonomyValidationError("taxonomy family identifiers or order changed")

    for family in families:
        _string(family.get("label"), name=f"{family_ids[families.index(family)]}.label")
        applicability = _mapping(family.get("applicability"), name=f"{family.get('id')}.applicability")
        required_axes = _string_sequence(
            applicability.get("required_axes"), name=f"{family.get('id')}.applicability.required_axes"
        )
        if not required_axes or len(set(required_axes)) != len(required_axes):
            raise TaxonomyValidationError(f"{family.get('id')} required axes are invalid")
        _string_sequence(
            applicability.get("non_applicable_when"), name=f"{family.get('id')}.applicability.non_applicable_when"
        )
        _string_sequence(
            applicability.get("unsupported_when"), name=f"{family.get('id')}.applicability.unsupported_when"
        )
        raw_evidence = family.get("required_evidence")
        if isinstance(raw_evidence, (str, bytes)) or not isinstance(raw_evidence, Sequence) or not raw_evidence:
            raise TaxonomyValidationError(f"{family.get('id')} required evidence is empty")
        evidence_ids: list[str] = []
        for item in raw_evidence:
            evidence = _mapping(item, name=f"{family.get('id')}.required_evidence.item")
            evidence_id = _string(evidence.get("id"), name=f"{family.get('id')}.required_evidence.id")
            evidence_ids.append(evidence_id)
            _string(evidence.get("kind"), name=f"{evidence_id}.kind")
            _string(evidence.get("description"), name=f"{evidence_id}.description")
            minimum_records = evidence.get("minimum_records")
            if isinstance(minimum_records, bool) or not isinstance(minimum_records, int) or minimum_records < 1:
                raise TaxonomyValidationError(f"{evidence_id}.minimum_records must be positive")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise TaxonomyValidationError(f"{family.get('id')} evidence identifiers are not unique")
        _validate_case(family.get("unsupported_case"), name=f"{family.get('id')}.unsupported_case")
        if "not_applicable_case" in family:
            _validate_case(family.get("not_applicable_case"), name=f"{family.get('id')}.not_applicable_case")


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Mapping[str, object]:
    """Load and validate the canonical taxonomy JSON document."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaxonomyValidationError(f"taxonomy cannot be loaded from {path}") from exc
    result = _mapping(document, name="taxonomy")
    validate_taxonomy(result)
    return result


def _family(document: Mapping[str, object], family_id: str) -> Mapping[str, object]:
    families = document["families"]
    assert isinstance(families, Sequence) and not isinstance(families, (str, bytes))
    for item in families:
        family = _mapping(item, name="family")
        if family.get("id") == family_id:
            return family
    raise TaxonomyValidationError(f"unknown taxonomy family: {family_id}")


def evaluate_claim(
    family_id: str,
    *,
    applicability: str,
    evidence_status: object,
    taxonomy: Mapping[str, object] | None = None,
) -> ClaimDecision:
    """Apply fail-closed applicability and evidence rules for one claim.

    A supported outcome is possible only when applicability is explicitly
    ``applicable`` and every required evidence item is explicitly ``observed``.
    ``not_applicable`` and ``unsupported`` always return a non-promotable
    unsupported outcome, regardless of any supplied evidence.
    """

    document = load_taxonomy() if taxonomy is None else taxonomy
    validate_taxonomy(document)
    family = _family(document, family_id)
    if applicability not in _APPLICABILITY_STATES:
        raise TaxonomyValidationError(f"unsupported applicability state: {applicability!r}")
    evidence_status = _require_string_map(evidence_status, name="evidence_status")
    required_items = family["required_evidence"]
    if isinstance(required_items, (str, bytes)) or not isinstance(required_items, Sequence):
        raise TaxonomyValidationError(f"{family_id}.required_evidence must be a sequence of strings")
    required = _string_sequence(
        tuple(_mapping(item, name="evidence").get("id") for item in required_items),
        name=f"{family_id}.required_evidence",
    )
    unknown = tuple(sorted(set(evidence_status).difference(required)))
    if unknown:
        return ClaimDecision("unsupported", False, (), f"unknown evidence identifiers: {', '.join(unknown)}")
    invalid = tuple(sorted(key for key, value in evidence_status.items() if value not in _EVIDENCE_STATUSES))
    if invalid:
        return ClaimDecision("unsupported", False, (), f"invalid evidence status for: {', '.join(invalid)}")
    if applicability != "applicable":
        reason = "declared non-applicability" if applicability == "not_applicable" else "declared unsupported"
        return ClaimDecision("unsupported", False, (), reason)
    missing = tuple(item for item in required if evidence_status.get(item) != "observed")
    if missing:
        return ClaimDecision("inconclusive", False, missing, "required evidence is not fully observed")
    return ClaimDecision("supported", True, (), "all required evidence is observed")


__all__ = [
    "TAXONOMY_PATH",
    "TAXONOMY_SCHEMA_VERSION",
    "ClaimDecision",
    "TaxonomyValidationError",
    "evaluate_claim",
    "load_taxonomy",
    "validate_taxonomy",
]
