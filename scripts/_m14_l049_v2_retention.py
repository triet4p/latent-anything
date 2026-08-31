"""Single-artifact retention boundary for L04.9 v2.

This is intentionally separate from the v1 exact-three-member transaction.
It only describes a validated JSON artifact and never deletes or mutates raw
evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts._m14_l049_v2_schema import canonical_digest, is_digest

V2_RETENTION_SCHEMA = "m14-l04.9-v2-single-artifact-retention-v1"


def build_retention_record(artifact: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Build a non-destructive retention record for one v2 stage artifact."""
    artifact_sha = artifact.get("artifact_sha256")
    if (
        not isinstance(artifact_sha, str)
        or artifact_sha != canonical_digest(dict(artifact), "artifact_sha256")
        or not is_digest(artifact.get("attestation_sha256"))
        or not is_digest(artifact.get("source_sha256"))
    ):
        raise ValueError("v2 artifact digest is invalid")
    return {
        "schema_version": V2_RETENTION_SCHEMA,
        "stage": str(stage),
        "status": artifact.get("status"),
        "use_case": str(stage),
        "source_sha256": artifact.get("source_sha256"),
        "artifact_sha256": artifact_sha,
        "attestation_sha256": artifact.get("attestation_sha256"),
        "member_count": 1,
        "raw_retention_status": "not_applicable_single_json_artifact",
        "repository_promotion": False,
    }


def validate_retention_record(record: Mapping[str, Any], artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_fields = {
        "schema_version",
        "stage",
        "status",
        "use_case",
        "source_sha256",
        "artifact_sha256",
        "attestation_sha256",
        "member_count",
        "raw_retention_status",
        "repository_promotion",
    }
    if (
        set(record) != expected_fields
        or record.get("schema_version") != V2_RETENTION_SCHEMA
        or record.get("member_count") != 1
    ):
        errors.append("v2 retention schema is invalid")
    if (
        record.get("artifact_sha256") != artifact.get("artifact_sha256")
        or record.get("artifact_sha256") != _artifact_digest(artifact)
        or record.get("stage") != artifact.get("stage")
        or record.get("status") != artifact.get("status")
        or record.get("use_case") != artifact.get("stage")
        or record.get("source_sha256") != artifact.get("source_sha256")
        or record.get("attestation_sha256") != artifact.get("attestation_sha256")
    ):
        errors.append("v2 retention stage/status/source/artifact binding is invalid")
    if record.get("raw_retention_status") != "not_applicable_single_json_artifact":
        errors.append("v2 retention raw status is invalid")
    if record.get("repository_promotion") is not False:
        errors.append("v2 retention must remain non-promoting")
    if not is_digest(record.get("artifact_sha256")) or not is_digest(record.get("attestation_sha256")):
        errors.append("v2 retention digests are malformed")
    return errors


def _artifact_digest(artifact: Mapping[str, Any]) -> str | None:
    try:
        value = artifact.get("artifact_sha256")
        return (
            value if isinstance(value, str) and value == canonical_digest(dict(artifact), "artifact_sha256") else None
        )
    except (TypeError, ValueError, OverflowError):
        return None


__all__ = ["V2_RETENTION_SCHEMA", "build_retention_record", "validate_retention_record"]
