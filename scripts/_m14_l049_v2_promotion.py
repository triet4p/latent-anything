"""Audit-gated local D3 promotion for L04.9 v2.

Stage B itself never emits D3.  This module is the only local path that can
create a D3 record, and it requires a separately retained, reopened,
transport-audited three-member bundle.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from scripts._m14_l049_v2_schema import CommitmentPolicy, canonical_digest, is_digest
from scripts._m14_l049_v2_transport import validate_transport_metadata
from scripts._m14_l049_v2_validate_stage_b import validate_stage_b_impl

V2_PROMOTION_SCHEMA = "m14-l04.9-v2-d3-promotion-v1"
PROMOTION_MEMBER_KINDS = ("partial", "run", "failure")


def _bounded_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and type(value) is not bool and 0 <= value < (1 << 63)


def _valid_audit(
    audit: object,
    transport: Mapping[str, Any],
    stage_b: Mapping[str, Any],
    *,
    expected_source_commit_sha: str,
    expected_source_tree_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(audit, Mapping):
        return ["v2 promotion retention audit is missing"]
    expected = {
        "schema_version",
        "stage",
        "status",
        "source_commit_sha",
        "source_tree_sha256",
        "cli_sha256",
        "cli_invocation_count",
        "cleanup_marker_count",
        "raw_before_parse",
        "raw_status",
        "raw_absent",
        "reopen_validation",
        "transport_sha256",
        "members",
        "pending_audit",
        "pending_audit_sha256",
        "raw_capture_bytes",
        "raw_capture_sha256",
        "raw_predelete_reopen_validation",
        "deleted_verified",
        "source_sha256",
        "one_cli_invocation",
        "cleanup_status",
        "audit_sha256",
    }
    if set(audit) != expected:
        errors.append("v2 promotion retention audit fields are invalid")
    if audit.get("schema_version") != "m14-l04.9-v2-retained-triplet-audit-v1":
        errors.append("v2 promotion retention audit schema is invalid")
    if audit.get("stage") != stage_b.get("stage") or audit.get("status") != stage_b.get("status"):
        errors.append("v2 promotion retention audit stage/status binding is invalid")
    for field in ("source_tree_sha256", "cli_sha256"):
        value = audit.get(field)
        if not isinstance(value, str) or not is_digest(value):
            errors.append(f"v2 promotion audit {field} is malformed")
    source_commit = audit.get("source_commit_sha")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        errors.append("v2 promotion source commit must be exactly 40 hexadecimal characters")
    if source_commit == "0" * 40:
        errors.append("v2 promotion source commit is only a placeholder")
    for field in ("source_tree_sha256", "cli_sha256"):
        if audit.get(field) == "0" * 64:
            errors.append(f"v2 promotion {field} is only a placeholder")
    if audit.get("source_commit_sha") != transport.get("source_commit_sha"):
        errors.append("v2 promotion source commit binding is invalid")
    if audit.get("source_tree_sha256") != transport.get("source_tree_sha256"):
        errors.append("v2 promotion source tree binding is invalid")
    if audit.get("source_commit_sha") != expected_source_commit_sha:
        errors.append("v2 promotion source commit does not match owner baseline")
    if audit.get("source_tree_sha256") != expected_source_tree_sha256:
        errors.append("v2 promotion source tree does not match owner baseline")
    if audit.get("cli_sha256") != transport.get("cli_sha256"):
        errors.append("v2 promotion CLI binding is invalid")
    if audit.get("cli_invocation_count") != 1 or audit.get("cleanup_marker_count") != 1:
        errors.append("v2 promotion audit marker counts are invalid")
    # The final audit is the authority for a deleted raw capture: this
    # validator cannot (and must not pretend to) recompute bytes that are gone.
    raw_capture_bytes = audit.get("raw_capture_bytes")
    if (
        audit.get("raw_before_parse") is not True
        or audit.get("raw_status") != "deleted_verified"
        or audit.get("raw_absent") is not True
    ):
        errors.append("v2 promotion raw retention is not deleted_verified/absent")
    if (
        audit.get("deleted_verified") is not True
        or audit.get("raw_predelete_reopen_validation") != "PASS"
        or audit.get("source_sha256") != stage_b.get("source_sha256")
        or audit.get("one_cli_invocation") is not True
        or audit.get("cleanup_status") != "PASS"
        or not _bounded_nonnegative_int(raw_capture_bytes)
        or not is_digest(audit.get("raw_capture_sha256"))
    ):
        errors.append("v2 promotion final raw audit chain is invalid")
    pending = audit.get("pending_audit")
    pending_bytes = pending.get("raw_capture_bytes") if isinstance(pending, Mapping) else None
    if (
        not isinstance(pending, Mapping)
        or set(pending)
        != {
            "schema_version",
            "stage",
            "status",
            "raw_capture_bytes",
            "raw_capture_sha256",
            "reopen_validation",
            "pending_audit_sha256",
        }
        or pending.get("schema_version") != "m14-l04.9-v2-pending-retention-audit-v1"
        or pending.get("stage") != stage_b.get("stage")
        or pending.get("status") != "quarantined_pending_delete"
        or not _bounded_nonnegative_int(pending_bytes)
        or not is_digest(pending.get("raw_capture_sha256"))
        or pending.get("reopen_validation") != "PASS"
        or pending.get("pending_audit_sha256") != canonical_digest(dict(pending), "pending_audit_sha256")
        or audit.get("pending_audit_sha256") != pending.get("pending_audit_sha256")
        or audit.get("raw_capture_bytes") != pending_bytes
        or audit.get("raw_capture_sha256") != pending.get("raw_capture_sha256")
    ):
        errors.append("v2 promotion pending-audit predecessor chain is invalid")
    if audit.get("reopen_validation") != "PASS" or audit.get("transport_sha256") != transport.get("transport_sha256"):
        errors.append("v2 promotion audit reopen/transport binding is invalid")
    members = audit.get("members")
    if not isinstance(members, Mapping) or set(members) != set(PROMOTION_MEMBER_KINDS):
        errors.append("v2 promotion retained triplet is incomplete")
    else:
        for kind in PROMOTION_MEMBER_KINDS:
            member = members.get(kind)
            member_map = cast(Mapping[str, Any], member) if isinstance(member, Mapping) else {}
            member_bytes = member_map.get("bytes")
            invalid_member = (
                not isinstance(member, Mapping)
                or set(member) != {"path", "bytes", "sha256", "reopen_validation"}
                or not _bounded_nonnegative_int(member_bytes)
                or not is_digest(member_map.get("sha256"))
                or member_map.get("reopen_validation") != "PASS"
                or not isinstance(member_map.get("path"), str)
            )
            if invalid_member:
                errors.append(f"v2 promotion retained {kind} member is invalid")
            else:
                try:
                    path = Path(str(member_map["path"]))
                    raw = path.read_bytes()
                    if (
                        path.is_symlink()
                        or not path.is_file()
                        or len(raw) != member_bytes
                        or hashlib.sha256(raw).hexdigest() != member_map["sha256"]
                    ):
                        errors.append(f"v2 promotion retained {kind} member bytes are invalid")
                    try:
                        json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        errors.append(f"v2 promotion retained {kind} envelope is invalid")
                except (OSError, TypeError, ValueError, OverflowError):
                    errors.append(f"v2 promotion retained {kind} member path is unreadable")
    try:
        if audit.get("audit_sha256") != canonical_digest(dict(audit), "audit_sha256"):
            errors.append("v2 promotion retention audit self-digest is invalid")
    except (TypeError, ValueError, OverflowError):
        errors.append("v2 promotion retention audit self-digest is invalid")
    return errors


def _validate_promotion_record_impl(
    record: Mapping[str, Any],
    stage_b: Mapping[str, Any],
    candidate: Mapping[str, Any],
    addendum: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    holdout_seed: bytes,
    transport: Mapping[str, Any],
    retention_audit: Mapping[str, Any],
    *,
    expected_source_commit_sha: str,
    expected_source_tree_sha256: str,
    policy: CommitmentPolicy,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema_version",
        "stage",
        "status",
        "evidence_level",
        "evidence_eligible",
        "repository_promotion",
        "promotion_candidate",
        "stage_b_artifact_sha256",
        "stage_b_attestation_sha256",
        "candidate_artifact_sha256",
        "parent_plan_sha256",
        "addendum_schema",
        "source_commit_sha",
        "source_tree_sha256",
        "cli_sha256",
        "transport_sha256",
        "retention_audit_sha256",
        "retained_member_sha256",
        "promotion_sha256",
    }
    if set(record) != expected:
        errors.append("v2 promotion record fields are invalid")
    stage_b_errors = validate_stage_b_impl(
        stage_b, holdout_rows, holdout_seed, candidate, addendum, train_rows, policy=policy
    )
    errors.extend(f"Stage B prerequisite: {error}" for error in stage_b_errors)
    if stage_b.get("evidence_level") != "D2" or stage_b.get("evidence_eligible") is not False:
        errors.append("v2 promotion requires a D2-ineligible Stage B artifact")
    if stage_b.get("promotion_candidate") is not True:
        errors.append("v2 promotion requires Stage B promotion_candidate=true")
    errors.extend(
        validate_transport_metadata(
            transport,
            str(stage_b.get("stage")),
            stage_b,
            addendum,
            expected_source_commit_sha=expected_source_commit_sha,
            expected_source_tree_sha256=expected_source_tree_sha256,
            expected_cli_sha256=transport.get("cli_sha256"),
            policy=policy,
        )
    )
    errors.extend(
        _valid_audit(
            retention_audit,
            transport,
            stage_b,
            expected_source_commit_sha=expected_source_commit_sha,
            expected_source_tree_sha256=expected_source_tree_sha256,
        )
    )
    if record.get("schema_version") != V2_PROMOTION_SCHEMA or record.get("stage") != stage_b.get("stage"):
        errors.append("v2 promotion schema/stage is invalid")
    expected_bindings = {
        "stage_b_artifact_sha256": stage_b.get("artifact_sha256"),
        "stage_b_attestation_sha256": stage_b.get("attestation_sha256"),
        "candidate_artifact_sha256": candidate.get("artifact_sha256"),
        "parent_plan_sha256": policy.parent_plan_sha256,
        "addendum_schema": policy.expected_addendum().get("schema_version"),
        "source_commit_sha": expected_source_commit_sha,
        "source_tree_sha256": expected_source_tree_sha256,
        "cli_sha256": transport.get("cli_sha256"),
        "transport_sha256": transport.get("transport_sha256"),
        "retention_audit_sha256": retention_audit.get("audit_sha256"),
    }
    for field, expected_value in expected_bindings.items():
        if record.get(field) != expected_value:
            errors.append(f"v2 promotion {field} binding is invalid")
    members = retention_audit.get("members")
    expected_member_hashes = (
        {kind: members[kind]["sha256"] for kind in PROMOTION_MEMBER_KINDS}
        if isinstance(members, Mapping)
        and all(isinstance(members.get(kind), Mapping) for kind in PROMOTION_MEMBER_KINDS)
        else {}
    )
    if record.get("retained_member_sha256") != expected_member_hashes:
        errors.append("v2 promotion retained member hashes are invalid")
    if record.get("status") != "accepted" or record.get("evidence_level") != "D3":
        errors.append("v2 promotion status/evidence level is invalid")
    if record.get("evidence_eligible") is not True or record.get("repository_promotion") is not True:
        errors.append("v2 promotion eligibility is invalid")
    if record.get("promotion_candidate") is not True:
        errors.append("v2 promotion candidate binding is invalid")
    try:
        if record.get("promotion_sha256") != canonical_digest(dict(record), "promotion_sha256"):
            errors.append("v2 promotion self-digest is invalid")
    except (TypeError, ValueError, OverflowError):
        errors.append("v2 promotion self-digest is invalid")
    return errors


def validate_promotion_record(
    record: Mapping[str, Any],
    stage_b: Mapping[str, Any],
    candidate: Mapping[str, Any],
    addendum: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    holdout_seed: bytes,
    transport: Mapping[str, Any],
    retention_audit: Mapping[str, Any],
    *,
    expected_source_commit_sha: str,
    expected_source_tree_sha256: str,
    policy: CommitmentPolicy,
) -> list[str]:
    """Validate promotion input without allowing malformed JSON to escape."""
    try:
        return _validate_promotion_record_impl(
            record,
            stage_b,
            candidate,
            addendum,
            train_rows,
            holdout_rows,
            holdout_seed,
            transport,
            retention_audit,
            expected_source_commit_sha=expected_source_commit_sha,
            expected_source_tree_sha256=expected_source_tree_sha256,
            policy=policy,
        )
    except (TypeError, ValueError, OverflowError, OSError, UnicodeError) as exc:
        return [f"v2 promotion malformed input: {type(exc).__name__}"]


def build_promotion_record(
    stage_b: Mapping[str, Any],
    candidate: Mapping[str, Any],
    addendum: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    holdout_seed: bytes,
    transport: Mapping[str, Any],
    retention_audit: Mapping[str, Any],
    *,
    expected_source_commit_sha: str,
    expected_source_tree_sha256: str,
    policy: CommitmentPolicy,
) -> dict[str, Any]:
    """Create D3 only after all independent prerequisite validators pass."""
    members = retention_audit.get("members")
    if not isinstance(members, Mapping) or any(
        not isinstance(members.get(kind), Mapping) for kind in PROMOTION_MEMBER_KINDS
    ):
        raise ValueError("promotion prerequisites failed: retained triplet is malformed")
    record: dict[str, Any] = {
        "schema_version": V2_PROMOTION_SCHEMA,
        "stage": stage_b["stage"],
        "status": "accepted",
        "evidence_level": "D3",
        "evidence_eligible": True,
        "repository_promotion": True,
        "promotion_candidate": True,
        "stage_b_artifact_sha256": stage_b["artifact_sha256"],
        "stage_b_attestation_sha256": stage_b["attestation_sha256"],
        "candidate_artifact_sha256": candidate["artifact_sha256"],
        "parent_plan_sha256": policy.parent_plan_sha256,
        "addendum_schema": policy.expected_addendum().get("schema_version"),
        "source_commit_sha": expected_source_commit_sha,
        "source_tree_sha256": expected_source_tree_sha256,
        "cli_sha256": transport["cli_sha256"],
        "transport_sha256": transport["transport_sha256"],
        "retention_audit_sha256": retention_audit["audit_sha256"],
        "retained_member_sha256": {kind: members[kind]["sha256"] for kind in PROMOTION_MEMBER_KINDS},
    }
    record["promotion_sha256"] = canonical_digest(record, "promotion_sha256")
    final_errors = validate_promotion_record(
        record,
        stage_b,
        candidate,
        addendum,
        train_rows,
        holdout_rows,
        holdout_seed,
        transport,
        retention_audit,
        expected_source_commit_sha=expected_source_commit_sha,
        expected_source_tree_sha256=expected_source_tree_sha256,
        policy=policy,
    )
    if final_errors:
        raise ValueError("promotion record validation failed: " + "; ".join(final_errors))
    return record


__all__ = ["PROMOTION_MEMBER_KINDS", "V2_PROMOTION_SCHEMA", "build_promotion_record", "validate_promotion_record"]
