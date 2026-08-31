"""Build-only transport metadata for the separate v2 single-artifact lane."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts._m14_l049_v2_schema import (
    CommitmentPolicy,
    canonical_digest,
    canonical_json_bytes,
    digest_bytes,
    is_digest,
    pinned_commitment_policy,
)

V2_TRANSPORT_SCHEMA = "m14-l04.9-v2-transport-v1"
_TOP_LEVEL_CLIS = {
    "stage_a_train_selection": "m14_l049_v2_stage_a.py",
    "stage_b_holdout_evaluation": "m14_l049_v2_stage_b.py",
}


def _top_level_cli_digest(stage: str) -> str | None:
    filename = _TOP_LEVEL_CLIS.get(stage)
    if filename is None:
        return None
    try:
        return digest_bytes(Path(__file__).with_name(filename).read_bytes())
    except OSError:
        return None


def build_transport_metadata(
    stage: str,
    artifact: Mapping[str, Any],
    addendum: Mapping[str, Any],
    *,
    source_commit_sha: str,
    source_tree_sha256: str,
    cli_sha256: str | None = None,
    policy: CommitmentPolicy | None = None,
) -> dict[str, Any]:
    """Describe marker-only transport without invoking SSH or remote execution."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit_sha):
        raise ValueError("source commit must be exactly 40 lowercase hexadecimal characters")
    if source_commit_sha == "0" * 40:
        raise ValueError("source commit must be an owner-supplied nonzero commitment")
    resolved_cli_sha = cli_sha256 or _top_level_cli_digest(str(stage)) or "0" * 64
    if not is_digest(source_tree_sha256) or not is_digest(resolved_cli_sha):
        raise ValueError("source tree and CLI commitments must be 64-hex digests")
    if source_tree_sha256 == "0" * 64 or resolved_cli_sha == "0" * 64:
        raise ValueError("source tree and CLI commitments must be owner-supplied nonzero digests")
    resolved_policy = policy or pinned_commitment_policy()
    if dict(addendum) != resolved_policy.expected_addendum():
        raise ValueError("transport addendum does not match the immutable commitment policy")
    artifact_sha = artifact.get("artifact_sha256")
    if not isinstance(artifact_sha, str) or artifact_sha != canonical_digest(dict(artifact), "artifact_sha256"):
        raise ValueError("v2 artifact digest is invalid")
    payload_sha = digest_bytes(canonical_json_bytes(dict(artifact)))
    bootstrap_sha = canonical_digest({"stage": str(stage), "payload_sha256": payload_sha}, "bootstrap_sha256")
    metadata: dict[str, Any] = {
        "schema_version": V2_TRANSPORT_SCHEMA,
        "stage": str(stage),
        "use_case": str(stage),
        "source_commit_sha": source_commit_sha,
        "source_tree_sha256": source_tree_sha256,
        "cli_sha256": resolved_cli_sha,
        "artifact_sha256": artifact_sha,
        "addendum_sha256": resolved_policy.addendum_sha256,
        "status": artifact.get("status"),
        "attestation_sha256": artifact.get("attestation_sha256"),
        "payload_sha256": payload_sha,
        "bootstrap_sha256": bootstrap_sha,
        "decode_sha256": payload_sha,
        "decoded_payload_sha256": payload_sha,
        "decode_ok": True,
        "cli_invocation_count": 1,
        "status_marker_count": 1,
        "bundle_marker_count": 3,
        "bundle_member_count": 3,
        "cleanup_marker_count": 1,
        "raw_before_parse": True,
        "postprocess_status": "PASS",
        "bundle_sha256": "",
        "protocol": "single canonical JSON artifact; marker-only stdout",
        "v1_exact_three_member_protocol": "unchanged and not reused",
        "network": "owner-gated; not invoked in Phase A",
    }
    metadata["bundle_sha256"] = _bundle_digest(metadata)
    metadata["transport_sha256"] = canonical_digest(metadata, "transport_sha256")
    return metadata


def validate_transport_metadata(
    metadata: Mapping[str, Any],
    stage: str,
    artifact: Mapping[str, Any],
    addendum: Mapping[str, Any],
    *,
    expected_source_commit_sha: str,
    expected_source_tree_sha256: str,
    expected_cli_sha256: str | None = None,
    policy: CommitmentPolicy | None = None,
) -> list[str]:
    """Validate marker/decode/hash metadata without performing transport."""
    errors: list[str] = []
    resolved_policy = policy or pinned_commitment_policy()
    expected_fields = {
        "schema_version",
        "stage",
        "use_case",
        "source_commit_sha",
        "source_tree_sha256",
        "cli_sha256",
        "artifact_sha256",
        "addendum_sha256",
        "status",
        "attestation_sha256",
        "payload_sha256",
        "bootstrap_sha256",
        "decode_sha256",
        "decoded_payload_sha256",
        "decode_ok",
        "cli_invocation_count",
        "status_marker_count",
        "bundle_marker_count",
        "bundle_member_count",
        "cleanup_marker_count",
        "raw_before_parse",
        "postprocess_status",
        "bundle_sha256",
        "protocol",
        "v1_exact_three_member_protocol",
        "network",
        "transport_sha256",
    }
    if set(metadata) != expected_fields:
        errors.append("v2 transport fields are invalid")
    try:
        artifact_sha = canonical_digest(dict(artifact), "artifact_sha256")
        addendum_sha = canonical_digest(dict(addendum), "addendum_sha256")
        payload_sha = digest_bytes(canonical_json_bytes(dict(artifact)))
        unsigned = dict(metadata)
        unsigned.pop("transport_sha256", None)
        expected_transport = canonical_digest(unsigned, "transport_sha256")
    except (TypeError, ValueError, OverflowError):
        artifact_sha = addendum_sha = payload_sha = expected_transport = None
    expected_cli = expected_cli_sha256 or _top_level_cli_digest(stage)
    if expected_cli is None:
        errors.append("v2 transport owner CLI commitment is required")
    expected = {
        "schema_version": V2_TRANSPORT_SCHEMA,
        "stage": stage,
        "use_case": stage,
        "source_commit_sha": expected_source_commit_sha,
        "source_tree_sha256": expected_source_tree_sha256,
        "cli_sha256": expected_cli if expected_cli is not None else metadata.get("cli_sha256"),
        "artifact_sha256": artifact_sha,
        "addendum_sha256": resolved_policy.addendum_sha256,
        "status": artifact.get("status"),
        "attestation_sha256": artifact.get("attestation_sha256"),
        "payload_sha256": payload_sha,
        "bootstrap_sha256": canonical_digest({"stage": stage, "payload_sha256": payload_sha}, "bootstrap_sha256"),
        "decode_sha256": payload_sha,
        "decoded_payload_sha256": payload_sha,
        "decode_ok": True,
        "cli_invocation_count": 1,
        "status_marker_count": 1,
        "bundle_marker_count": 3,
        "bundle_member_count": 3,
        "cleanup_marker_count": 1,
        "raw_before_parse": True,
        "postprocess_status": "PASS",
        "protocol": "single canonical JSON artifact; marker-only stdout",
        "v1_exact_three_member_protocol": "unchanged and not reused",
        "network": "owner-gated; not invoked in Phase A",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(f"v2 transport {key} binding is invalid")
    if addendum_sha != resolved_policy.addendum_sha256 or dict(addendum) != resolved_policy.expected_addendum():
        errors.append("v2 transport addendum does not match the immutable commitment policy")
    if not isinstance(metadata.get("source_commit_sha"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", metadata["source_commit_sha"]
    ):
        errors.append("v2 transport source commit must be exactly 40 lowercase hexadecimal characters")
    if metadata.get("source_commit_sha") == "0" * 40:
        errors.append("v2 transport source commit is only a placeholder")
    for field in ("source_tree_sha256", "cli_sha256"):
        if not is_digest(metadata.get(field)):
            errors.append(f"v2 transport {field} is malformed")
        elif metadata.get(field) == "0" * 64:
            errors.append(f"v2 transport {field} is only a placeholder")
    if not is_digest(metadata.get("artifact_sha256")) or not is_digest(metadata.get("addendum_sha256")):
        errors.append("v2 transport digest is malformed")
    if not is_digest(metadata.get("attestation_sha256")):
        errors.append("v2 transport attestation digest is malformed")
    attestation = artifact.get("runtime_attestation")
    if (
        not isinstance(attestation, Mapping)
        or attestation.get("attestation_sha256") != metadata.get("attestation_sha256")
        or attestation.get("cli_sha256") != metadata.get("cli_sha256")
    ):
        errors.append("v2 transport artifact/attestation/CLI binding is invalid")
    if metadata.get("bundle_sha256") != _bundle_digest(metadata):
        errors.append("v2 transport bundle digest is invalid")
    if metadata.get("transport_sha256") != expected_transport:
        errors.append("v2 transport self-digest is invalid")
    return errors


def _bundle_digest(metadata: Mapping[str, Any]) -> str | None:
    try:
        unsigned = dict(metadata)
        unsigned.pop("bundle_sha256", None)
        unsigned.pop("transport_sha256", None)
        return canonical_digest(unsigned, "bundle_sha256")
    except (TypeError, ValueError, OverflowError):
        return None


__all__ = ["V2_TRANSPORT_SCHEMA", "build_transport_metadata", "validate_transport_metadata"]
