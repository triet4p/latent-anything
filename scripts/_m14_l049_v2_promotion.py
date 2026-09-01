"""Audit-gated local D3 promotion for L04.9 v2.

Stage B itself never emits D3.  This module is the only local path that can
create a D3 record, and it requires a separately retained, reopened,
transport-audited three-member bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from scripts._m14_l049_v2_fixture import read_rows
from scripts._m14_l049_v2_inputs import (
    CANONICAL_STAGE_B_HOLDOUT,
    CANONICAL_STAGE_B_MANIFEST,
    CANONICAL_STAGE_B_SEED,
    SOURCE_KEYED_STAGE_B_CANDIDATE,
)
from scripts._m14_l049_v2_schema import (
    EXPECTED_ADDENDUM_SHA256,
    EXPECTED_HOLDOUT_CONTENT_SHA256,
    EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256,
    PARENT_PLAN_SHA256,
    CommitmentPolicy,
    canonical_digest,
    is_digest,
)
from scripts._m14_l049_v2_transport import validate_transport_metadata
from scripts._m14_l049_v2_validate_stage_b import validate_stage_b_impl

LEGACY_V2_PROMOTION_SCHEMA = "m14-l04.9-v2-d3-promotion-v1"
REAL_V2_PROMOTION_SCHEMA = "m14-l04.9-v2-d3-promotion-real-v2"
PROMOTION_MEMBER_KINDS = ("partial", "run", "failure")

# These paths and digests are part of the real-evidence contract.  They are
# intentionally fixed rather than taken from caller-provided mappings.
REAL_CANONICAL_PATHS = MappingProxyType(
    {
        "d1_assessment": Path(
            "artifacts/m14/l04-explanations.ssh.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.d1-assessment.sidecar.json"
        ),
        "d1_audit": Path(
            "artifacts/m14/l04-explanations.ssh.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.audit.json"
        ),
        "d1_candidate": Path(
            "artifacts/m14/l04-explanations.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.candidate.json"
        ),
        "d2_assessment": Path(
            "artifacts/m14/l04-explanations.ssh.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.d2-assessment.sidecar.json"
        ),
        "d2_audit": Path(
            "artifacts/m14/l04-explanations.ssh.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.audit.json"
        ),
        "provisioning_assessment": Path(
            "artifacts/m14/l04-explanations.L049V2StageB.provisioning-assessment.sidecar.json"
        ),
        "manifest": CANONICAL_STAGE_B_MANIFEST,
        "holdout": CANONICAL_STAGE_B_HOLDOUT,
        "seed": CANONICAL_STAGE_B_SEED,
        "candidate": SOURCE_KEYED_STAGE_B_CANDIDATE,
        "addendum": Path("artifacts/m14/l04-explanations.v2.addendum.json"),
        "train": Path("artifacts/m14/l04-l049-v2-train.jsonl"),
        "d1_failure": Path(
            "artifacts/m14/l04-explanations.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.failure.json"
        ),
        "d1_partial": Path(
            "artifacts/m14/l04-explanations.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.partial.json"
        ),
        "d1_run": Path("artifacts/m14/l04-explanations.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.run.json"),
        "d2_failure": Path(
            "artifacts/m14/l04-explanations.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.failure.json"
        ),
        "d2_partial": Path(
            "artifacts/m14/l04-explanations.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.partial.json"
        ),
        "d2_run": Path("artifacts/m14/l04-explanations.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.run.json"),
    }
)

REAL_CANONICAL_FILE_SHA256 = MappingProxyType(
    {
        "d1_assessment": "735d7fca2a157aaaefdcbb2667b95ff9fd91f6445b70c78cc9cee1e82b790d66",
        "d1_audit": "a1b60ec6804e0468716398c75c9e3508a1c982c0b312fcd8fb1c5aab737e166d",
        "d1_candidate": "29bcd20ab494092abbb074bff5d99d091ec288d261a0399f97f2e2fb4f092aa2",
        "d2_assessment": "1fc621818f89c932dc46d0f80ca22aa2aaabf1f19c869a62fb0bcf71b818070f",
        "d2_audit": "c8a308655103a75845ae45a0cc0a8029408958e4c9c01335db1a22854b0cef85",
        "provisioning_assessment": "7bbc7276a44cc2ae0e68a2e6ca09c35d22f718c94392c7e712bc3ac0f9a0804c",
        "manifest": "2849b07fd719a0a761f433892fcc031c2ab17012a538daba322dd6fa50674974",
        "holdout": "295ef5f558315c629d68e2d0216567a67163e5ef4adaaf3bbc9fe8a4da96dd5f",
        "seed": "b8e5e28908c2d2925a5bf5dcc69d852b4e31584f23f0ced2903a70f10d36b5e1",
        "candidate": "29bcd20ab494092abbb074bff5d99d091ec288d261a0399f97f2e2fb4f092aa2",
        "addendum": "3573e98257dd1922dcf6a70be97e0f651944f707291ece3ae4c01f7d73260b6d",
        "train": "f4cb7b52f946263a99113b9ebd8b24a74f66b49cd17fce77c15a044ec671a9e9",
        "d1_failure": "a40f645d7e8cbb6ccf76765287ff09d592b70ea9f5e4d284e1a5c9c74d489afe",
        "d1_partial": "f5fff08f0de818bb4ef91157b7e94d9c200343afcc3e6c53444b041fa840eee2",
        "d1_run": "0123b4dbd38b921c5174dfcb87c2e5bd08fdd08cc66db4374191d31a061fed9a",
        "d2_failure": "b8c2000afbec9900f706034dfe742761ef95034f94aa9606b49c9686336144e2",
        "d2_partial": "18f60e97ce21ff88a1fa27c1b3e23e0f3bbcc7898ee7440e1a1804b4f695f0eb",
        "d2_run": "929c53129bf8285055a689c11da05675f473cd4a5984bc8efe82a8f34886a210",
    }
)

REAL_SOURCE_COMMIT_SHA = "6af20749b305f591d2c90d868cb09e71f623bdd0"
REAL_SOURCE_TREE_OID = "a0f1fb55c8d112128d81f3942132657100eac00f"
REAL_D1_SOURCE_COMMIT_SHA = "76a45ea74fbb2843b7d109855c2c387ab98b3e47"
REAL_D1_SOURCE_TREE_OID = "392d241719b10fe6a946f20d203b9e0ff0f5f46c"
REAL_PROVISIONING_SOURCE_COMMIT_SHA = "7d1e23fdbc385909f964df05360f01027d3b6c35"
REAL_PROVISIONING_SOURCE_TREE_OID = "5f43b035a043faf97237cd87aa621bec61c805b1"
REAL_D1_PENDING_AUDIT_SHA256 = "0c81ddedac08d2747d20982f4f2e221183ed9e380504917550b6cdfd680f9d7c"
REAL_D1_PENDING_AUDIT_BYTES = 3243
REAL_RAW_CAPTURE_PATH = (
    "artifacts/m14/l04-explanations.ssh.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.raw.txt"
)
REAL_BUNDLE_MEMBER_PATHS = MappingProxyType(
    {
        (
            "artifacts/m14/l04-explanations.L049V2StageB.attempt1.failure.json"
        ): "b8c2000afbec9900f706034dfe742761ef95034f94aa9606b49c9686336144e2",
        (
            "artifacts/m14/l04-explanations.L049V2StageB.attempt1.partial.json"
        ): "18f60e97ce21ff88a1fa27c1b3e23e0f3bbcc7898ee7440e1a1804b4f695f0eb",
        (
            "artifacts/m14/l04-explanations.L049V2StageB.attempt1.run.json"
        ): "929c53129bf8285055a689c11da05675f473cd4a5984bc8efe82a8f34886a210",
    }
)


@dataclass(frozen=True)
class RealEvidenceCommitment:
    """Immutable owner-supplied commitment for one retained repository file."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class RealPromotionPolicy:
    """Independent pins required before the real-evidence D3 path can run."""

    source_commit_sha: str
    source_tree_algorithm: str
    source_tree_oid: str
    d1_assessment: RealEvidenceCommitment
    d1_assessment_canonical_sha256: str
    d1_audit: RealEvidenceCommitment
    d1_candidate: RealEvidenceCommitment
    d1_source_commit_sha: str
    d1_source_tree_algorithm: str
    d1_source_tree_oid: str
    d1_pending_sidecar_sha256: str
    d1_pending_audit_sha256: str
    d1_pending_audit_bytes: int
    provisioning_source_commit_sha: str
    provisioning_source_tree_algorithm: str
    provisioning_source_tree_oid: str
    d2_assessment: RealEvidenceCommitment
    d2_assessment_canonical_sha256: str
    d2_audit: RealEvidenceCommitment
    d2_pending_sidecar_sha256: str
    d2_pending_audit_sha256: str
    d2_pending_audit_bytes: int
    provisioning_assessment: RealEvidenceCommitment
    provisioning_assessment_canonical_sha256: str
    manifest: RealEvidenceCommitment
    holdout: RealEvidenceCommitment
    seed: RealEvidenceCommitment
    candidate: RealEvidenceCommitment
    parent_plan_sha256: str
    addendum_schema: str
    candidate_artifact_sha256: str
    stage_b_artifact_sha256: str
    stage_b_attestation_sha256: str
    cli_sha256: str
    transport_payload_sha256: str
    transport_decode_sha256: str
    raw_capture: RealEvidenceCommitment
    bundle: RealEvidenceCommitment
    bundle_members: tuple[RealEvidenceCommitment, ...]
    triad: tuple[RealEvidenceCommitment, ...]

    def file_commitments(self) -> tuple[RealEvidenceCommitment, ...]:
        return (
            self.d1_assessment,
            self.d1_audit,
            self.d1_candidate,
            self.d2_assessment,
            self.d2_audit,
            self.provisioning_assessment,
            self.manifest,
            self.holdout,
            self.seed,
            self.candidate,
            *self.triad,
        )


def _bounded_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and type(value) is not bool and 0 <= value < (1 << 63)


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return a typed, read-only view for defensive evidence inspection."""
    return value if isinstance(value, Mapping) else {}


def _mapping_inputs_valid(*values: object) -> bool:
    return all(isinstance(value, Mapping) for value in values)


def _repo_root_matches(repo_root: Path) -> bool:
    """Require an actual top-level Git worktree before reading evidence."""
    try:
        requested = repo_root.resolve(strict=True)
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        text_value = result.stdout.strip()
        if result.returncode != 0 or not text_value or "\n" in text_value or "\r" in text_value:
            return False
        reported = Path(text_value).resolve(strict=True)
        return os.path.normcase(str(reported)) == os.path.normcase(str(requested))
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError, RuntimeError):
        return False


def _tracked_exact(repo_root: Path, relative: Path) -> bool:
    """Require exact index membership, with no alternate or parent-repo lookup."""
    try:
        value = relative.as_posix()
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", value],
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == value
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return False


def _canonical_raw(repo_root: Path, name: str) -> tuple[Path, bytes] | None:
    """Read only a fixed, tracked, non-symlinked canonical file."""
    relative = REAL_CANONICAL_PATHS.get(name)
    expected_sha = REAL_CANONICAL_FILE_SHA256.get(name)
    if relative is None or expected_sha is None or not _repo_root_matches(repo_root):
        return None
    try:
        root = repo_root.resolve(strict=True)
        path = root / relative
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                return None
        if not path.is_file() or not _tracked_exact(root, relative):
            return None
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            return None
        return path, raw
    except (OSError, RuntimeError, ValueError):
        return None


def _canonical_json(repo_root: Path, name: str) -> Mapping[str, Any] | None:
    resolved = _canonical_raw(repo_root, name)
    if resolved is None:
        return None
    try:
        value = json.loads(resolved[1].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, Mapping) else None


def _canonical_mapping_matches(repo_root: Path, name: str, value: object) -> bool:
    expected = _canonical_json(repo_root, name)
    if not isinstance(value, Mapping) or expected is None:
        return False
    try:
        return _strict_canonical_json_bytes(value) == _strict_canonical_json_bytes(expected)
    except (TypeError, ValueError, OverflowError):
        return False


def _strict_canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON-shaped values while preserving primitive type identity."""

    def normalize(item: object) -> object:
        if item is None or isinstance(item, (str, bool)):
            return item
        if isinstance(item, int):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("non-finite JSON number")
            return item
        if isinstance(item, Mapping):
            normalized: dict[str, object] = {}
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                normalized[key] = normalize(nested)
            return normalized
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [normalize(nested) for nested in item]
        raise TypeError("value is not JSON-shaped")

    encoded = json.dumps(normalize(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (encoded + "\n").encode("utf-8")


def _canonical_rows_matches(repo_root: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> bool:
    resolved = _canonical_raw(repo_root, name)
    if resolved is None:
        return False
    try:
        _raw, expected_rows = read_rows(resolved[0])
        actual = [dict(row) for row in rows]
        expected = [dict(row) for row in expected_rows]
        return _strict_canonical_json_bytes(actual) == _strict_canonical_json_bytes(expected)
    except (OSError, TypeError, ValueError, UnicodeError, OverflowError):
        return False


def _rows_input_valid(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(row, Mapping) for row in value)
    )


def _sidecar_shape_valid(value: object) -> bool:
    """Reject malformed known object slots before any nested access."""
    if not isinstance(value, Mapping):
        return False
    mapping_fields = ("source", "assessment", "evidence", "retention", "inputs")
    if any(field in value and not isinstance(value.get(field), Mapping) for field in mapping_fields):
        return False
    evidence = value.get("evidence")
    if isinstance(evidence, Mapping) and any(
        field in evidence and not isinstance(evidence.get(field), Mapping)
        for field in ("audit", "bundle", "candidate", "raw_capture", "triad")
    ):
        return False
    retention = value.get("retention")
    return not (
        isinstance(retention, Mapping)
        and "finalize_delete" in retention
        and not isinstance(retention.get("finalize_delete"), Mapping)
    )


def _is_runtime_instance(value: object, expected: type[Any]) -> bool:
    """Keep public runtime guards active even when static types are narrow."""
    return isinstance(value, expected)


def _repository_tree_errors(
    repo_root: Path,
    source_commit_sha: str,
    tree_algorithm: str,
    tree_oid: str,
) -> list[str]:
    """Cross-check the explicit tree commitment against independent Git metadata."""
    if not _repo_root_matches(repo_root):
        return ["real promotion repository root is invalid"]
    if tree_algorithm != "sha1" or not re.fullmatch(r"[0-9a-f]{40}", tree_oid):
        return ["real promotion source tree commitment is malformed"]
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit_sha):
        return ["real promotion source commit commitment is malformed"]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root.resolve()), "rev-parse", f"{source_commit_sha}^{{tree}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ["real promotion repository tree metadata is unavailable"]
    observed = result.stdout.strip().lower() if result.returncode == 0 else ""
    if observed != tree_oid:
        return ["real promotion source tree does not match repository metadata"]
    return []


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
    if record.get("schema_version") != LEGACY_V2_PROMOTION_SCHEMA or record.get("stage") != stage_b.get("stage"):
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


def validate_legacy_promotion_record(
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
    if (
        not _mapping_inputs_valid(record, stage_b, candidate, addendum, transport, retention_audit)
        or not _rows_input_valid(train_rows)
        or not _rows_input_valid(holdout_rows)
        or not _is_runtime_instance(holdout_seed, bytes)
        or not _is_runtime_instance(policy, CommitmentPolicy)
    ):
        return ["v2 promotion malformed input"]
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
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, OSError, UnicodeError):
        return ["v2 promotion malformed input"]


def build_legacy_promotion_record(
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
        "schema_version": LEGACY_V2_PROMOTION_SCHEMA,
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
    final_errors = validate_legacy_promotion_record(
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


REAL_PROMOTION_FIELDS = {
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
    "candidate_file_sha256",
    "parent_plan_sha256",
    "addendum_schema",
    "source_commit_sha",
    "source_tree",
    "d1_source_commit_sha",
    "d1_source_tree",
    "provisioning_source_commit_sha",
    "provisioning_source_tree",
    "d1_pending_sidecar_sha256",
    "d1_pending_audit_sha256",
    "d1_pending_audit_bytes",
    "d2_pending_audit_sha256",
    "d2_pending_audit_bytes",
    "cli_sha256",
    "transport_payload_sha256",
    "transport_decode_sha256",
    "transport_decode_match",
    "bundle_bytes",
    "bundle_sha256",
    "bundle_member_sha256",
    "retention_audit_sha256",
    "retention_audit_schema",
    "d1_assessment_sha256",
    "d1_audit_sha256",
    "d1_candidate_file_sha256",
    "d2_assessment_sha256",
    "d2_audit_sha256",
    "provisioning_assessment_sha256",
    "provisioning_manifest_sha256",
    "provisioning_holdout_sha256",
    "provisioning_seed_commitment_sha256",
    "retained_member_sha256",
    "pending_retention_sidecar_sha256",
    "promotion_sha256",
}


def _canonical_bound_relative_path(value: object) -> str | None:
    """Validate an untrusted bound path lexically before ``Path`` can normalize it."""
    if not isinstance(value, str) or not value:
        return None
    if "\\" in value or "\x00" in value:
        return None
    if value.startswith("/") or value.endswith("/") or "//" in value:
        return None
    if re.match(r"^[A-Za-z]:", value):
        return None
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        return None
    return value


def _safe_bound_file(repo_root: Path, relative: object) -> tuple[Path, bytes] | None:
    canonical_relative = _canonical_bound_relative_path(relative)
    if canonical_relative is None:
        return None
    try:
        root = repo_root.resolve()
        if not _repo_root_matches(root):
            return None
        relative_path = Path(canonical_relative)
        if relative_path.as_posix() != canonical_relative:
            return None
        if not _tracked_exact(root, relative_path):
            return None
        candidate = root / relative_path
        cursor = root
        for part in relative_path.parts:
            cursor /= part
            if cursor.is_symlink():
                return None
        path = candidate.resolve()
        path.relative_to(root)
        if not path.is_file():
            return None
        return path, path.read_bytes()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _file_record_errors(
    repo_root: Path,
    item: object,
    expected: RealEvidenceCommitment | None = None,
) -> tuple[list[str], bytes | None]:
    if not isinstance(item, Mapping):
        return ["real promotion evidence file record is malformed"], None
    path = item.get("path")
    expected_bytes = item.get("bytes")
    expected_sha = item.get("sha256") or item.get("commitment_sha256")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
        or not is_digest(expected_sha)
    ):
        return ["real promotion evidence file commitment is malformed"], None
    if expected is not None and (
        path != expected.path or expected_bytes != expected.bytes or expected_sha != expected.sha256
    ):
        return ["real promotion evidence file does not match owner policy"], None
    resolved = _safe_bound_file(repo_root, path)
    if resolved is None:
        return ["real promotion evidence file is missing or unsafe"], None
    _path, raw = resolved
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha:
        return ["real promotion evidence file commitment does not match"], None
    return [], raw


def _load_bound_json(repo_root: Path, item: object) -> Mapping[str, Any] | None:
    """Load a committed JSON evidence record only after path containment checks."""
    if not isinstance(item, Mapping):
        return None
    resolved = _safe_bound_file(repo_root, item.get("path"))
    if resolved is None:
        return None
    try:
        value = json.loads(resolved[1].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _sidecar_digest_errors(sidecar: object, expected_schema: str) -> list[str]:
    if not isinstance(sidecar, Mapping):
        return ["real promotion sidecar is missing"]
    errors: list[str] = []
    if sidecar.get("schema_version") != expected_schema:
        errors.append("real promotion sidecar schema is invalid")
    digest = sidecar.get("sidecar_sha256")
    if not is_digest(digest) or digest != canonical_digest(dict(sidecar), "sidecar_sha256"):
        errors.append("real promotion sidecar self-digest is invalid")
    return errors


def _policy_errors(policy: object) -> list[str]:
    if not isinstance(policy, RealPromotionPolicy):
        return ["real promotion owner policy is malformed"]
    errors: list[str] = []
    if (
        policy.source_tree_algorithm != "sha1"
        or not re.fullmatch(r"[0-9a-f]{40}", policy.source_tree_oid)
        or not re.fullmatch(r"[0-9a-f]{40}", policy.source_commit_sha)
        or policy.d1_source_tree_algorithm != "sha1"
        or not re.fullmatch(r"[0-9a-f]{40}", policy.d1_source_tree_oid)
        or not re.fullmatch(r"[0-9a-f]{40}", policy.d1_source_commit_sha)
        or policy.provisioning_source_tree_algorithm != "sha1"
        or not re.fullmatch(r"[0-9a-f]{40}", policy.provisioning_source_tree_oid)
        or not re.fullmatch(r"[0-9a-f]{40}", policy.provisioning_source_commit_sha)
    ):
        errors.append("real promotion owner source commitment is malformed")
    digest_values = (
        policy.d2_pending_sidecar_sha256,
        policy.d2_pending_audit_sha256,
        policy.d1_pending_sidecar_sha256,
        policy.d1_pending_audit_sha256,
        policy.parent_plan_sha256,
        policy.candidate_artifact_sha256,
        policy.stage_b_artifact_sha256,
        policy.stage_b_attestation_sha256,
        policy.cli_sha256,
        policy.transport_payload_sha256,
        policy.transport_decode_sha256,
        policy.d1_assessment_canonical_sha256,
        policy.d2_assessment_canonical_sha256,
        policy.provisioning_assessment_canonical_sha256,
    )
    if any(not is_digest(value) for value in digest_values):
        errors.append("real promotion owner digest commitment is malformed")
    if not _bounded_nonnegative_int(policy.d2_pending_audit_bytes):
        errors.append("real promotion owner predecessor audit size is malformed")
    if not _bounded_nonnegative_int(policy.d1_pending_audit_bytes):
        errors.append("real promotion owner D1 predecessor audit size is malformed")
    addendum_schema = cast(object, policy.addendum_schema)
    if not isinstance(addendum_schema, str) or not addendum_schema:
        errors.append("real promotion owner addendum schema commitment is malformed")
    commitments = policy.file_commitments()
    paths: dict[str, tuple[int, str]] = {}
    for item in commitments:
        path_value = cast(object, item.path)
        if (
            not isinstance(path_value, str)
            or not path_value
            or not _bounded_nonnegative_int(item.bytes)
            or not is_digest(item.sha256)
            or (path_value in paths and paths[path_value] != (item.bytes, item.sha256))
        ):
            errors.append("real promotion owner file commitment is malformed")
        if isinstance(path_value, str):
            paths[path_value] = (item.bytes, item.sha256)
    if not policy.bundle_members or not policy.triad:
        errors.append("real promotion owner retained commitments are incomplete")
    return errors


def load_real_promotion_policy(repo_root: Path) -> RealPromotionPolicy:
    """Build the real policy exclusively from pinned canonical repository files.

    Caller-provided mappings are deliberately not accepted here.  Every JSON
    source is checked against a fixed path and byte digest before any fields
    are used to construct the immutable policy.
    """
    if not _repo_root_matches(repo_root):
        raise ValueError("real promotion canonical policy unavailable")
    loaded: dict[str, Mapping[str, Any]] = {}
    for name in (
        "d1_assessment",
        "d1_audit",
        "d1_candidate",
        "d2_assessment",
        "d2_audit",
        "provisioning_assessment",
        "manifest",
        "candidate",
        "addendum",
        "d1_failure",
        "d1_partial",
        "d1_run",
        "d2_failure",
        "d2_partial",
        "d2_run",
    ):
        value = _canonical_json(repo_root, name)
        if value is None:
            raise ValueError("real promotion canonical policy unavailable")
        loaded[name] = value
    d1 = loaded["d1_assessment"]
    d2 = loaded["d2_assessment"]
    provisioning = loaded["provisioning_assessment"]
    d2_audit = loaded["d2_audit"]
    candidate = loaded["candidate"]
    addendum = loaded["addendum"]
    d1_evidence = _as_mapping(d1.get("evidence"))
    d2_evidence = _as_mapping(d2.get("evidence"))
    d2_inputs = _as_mapping(d2.get("inputs"))
    partial = loaded["d2_partial"]
    stage_b = _as_mapping(partial.get("artifact"))
    if (
        d1.get("source", {}).get("commit_sha") != REAL_D1_SOURCE_COMMIT_SHA
        or d1.get("source", {}).get("tree_sha256") != REAL_D1_SOURCE_TREE_OID
        or d2.get("source", {}).get("commit_sha") != REAL_SOURCE_COMMIT_SHA
        or d2.get("source", {}).get("tree_sha256") != REAL_SOURCE_TREE_OID
        or provisioning.get("source", {}).get("commit_sha") != REAL_PROVISIONING_SOURCE_COMMIT_SHA
        or provisioning.get("source", {}).get("tree_sha256") != REAL_PROVISIONING_SOURCE_TREE_OID
        or d2_audit.get("source_sha") != REAL_SOURCE_COMMIT_SHA
        or d2_audit.get("use_case") != "L049V2StageB"
        or not stage_b
    ):
        raise ValueError("real promotion canonical policy unavailable")
    try:
        addendum_policy = CommitmentPolicy.from_addendum(dict(addendum))
        if (
            addendum.get("addendum_sha256") != EXPECTED_ADDENDUM_SHA256
            or addendum_policy.parent_plan_sha256 != PARENT_PLAN_SHA256
        ):
            raise ValueError

        def commitment(name: str) -> RealEvidenceCommitment:
            path = REAL_CANONICAL_PATHS[name]
            raw = _canonical_raw(repo_root, name)
            if raw is None:
                raise ValueError
            return RealEvidenceCommitment(path.as_posix(), len(raw[1]), REAL_CANONICAL_FILE_SHA256[name])

        def sidecar_commitment(name: str) -> RealEvidenceCommitment:
            return commitment(name)

        d1_candidate = _as_mapping(d1_evidence.get("candidate"))
        d2_audit_map = d2_audit
        d2_triad = _as_mapping(d2_evidence.get("triad"))
        bundle = _as_mapping(d2_audit_map.get("bundle"))
        bundle_members = _as_mapping(bundle.get("members"))
        if set(bundle_members) != set(REAL_BUNDLE_MEMBER_PATHS) or set(d2_triad) != set(PROMOTION_MEMBER_KINDS):
            raise ValueError
        triad: list[RealEvidenceCommitment] = []
        for kind in PROMOTION_MEMBER_KINDS:
            item = _as_mapping(d2_triad.get(kind))
            expected_name = f"d2_{kind}"
            if item.get("path") != REAL_CANONICAL_PATHS[expected_name].as_posix():
                raise ValueError
            triad.append(RealEvidenceCommitment(item["path"], int(item["bytes"]), str(item["sha256"])))
        manifest_item = _as_mapping(d2_inputs.get("manifest"))
        holdout_item = _as_mapping(d2_inputs.get("holdout"))
        seed_item = _as_mapping(d2_inputs.get("seed"))
        candidate_item = _as_mapping(d2_inputs.get("candidate"))
        if (
            manifest_item.get("path") != REAL_CANONICAL_PATHS["manifest"].as_posix()
            or holdout_item.get("path") != REAL_CANONICAL_PATHS["holdout"].as_posix()
            or seed_item.get("path") != REAL_CANONICAL_PATHS["seed"].as_posix()
            or candidate_item.get("path") != REAL_CANONICAL_PATHS["candidate"].as_posix()
            or d1_candidate.get("path") != REAL_CANONICAL_PATHS["candidate"].as_posix()
        ):
            raise ValueError
        holdout_bound = _canonical_raw(repo_root, "holdout")
        seed_bound = _canonical_raw(repo_root, "seed")
        if holdout_bound is None or seed_bound is None:
            raise ValueError
        return RealPromotionPolicy(
            source_commit_sha=REAL_SOURCE_COMMIT_SHA,
            source_tree_algorithm="sha1",
            source_tree_oid=REAL_SOURCE_TREE_OID,
            d1_assessment=sidecar_commitment("d1_assessment"),
            d1_assessment_canonical_sha256=str(d1["sidecar_sha256"]),
            d1_audit=commitment("d1_audit"),
            d1_candidate=commitment("d1_candidate"),
            d1_source_commit_sha=REAL_D1_SOURCE_COMMIT_SHA,
            d1_source_tree_algorithm="sha1",
            d1_source_tree_oid=REAL_D1_SOURCE_TREE_OID,
            d1_pending_sidecar_sha256=str(d1["retention"]["previous_pending_sidecar_sha256"]),
            d1_pending_audit_sha256=REAL_D1_PENDING_AUDIT_SHA256,
            d1_pending_audit_bytes=REAL_D1_PENDING_AUDIT_BYTES,
            provisioning_source_commit_sha=REAL_PROVISIONING_SOURCE_COMMIT_SHA,
            provisioning_source_tree_algorithm="sha1",
            provisioning_source_tree_oid=REAL_PROVISIONING_SOURCE_TREE_OID,
            d2_assessment=sidecar_commitment("d2_assessment"),
            d2_assessment_canonical_sha256=str(d2["sidecar_sha256"]),
            d2_audit=commitment("d2_audit"),
            d2_pending_sidecar_sha256=str(d2["retention"]["previous_pending_sidecar_sha256"]),
            d2_pending_audit_sha256=str(d2_evidence["audit"]["prior_pending_sha256"]),
            d2_pending_audit_bytes=int(d2_evidence["audit"]["prior_pending_bytes"]),
            provisioning_assessment=sidecar_commitment("provisioning_assessment"),
            provisioning_assessment_canonical_sha256=str(provisioning["sidecar_sha256"]),
            manifest=commitment("manifest"),
            holdout=RealEvidenceCommitment(
                REAL_CANONICAL_PATHS["holdout"].as_posix(),
                len(holdout_bound[1]),
                EXPECTED_HOLDOUT_CONTENT_SHA256,
            ),
            seed=RealEvidenceCommitment(
                REAL_CANONICAL_PATHS["seed"].as_posix(),
                len(seed_bound[1]),
                EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256,
            ),
            candidate=commitment("candidate"),
            parent_plan_sha256=addendum_policy.parent_plan_sha256,
            addendum_schema=str(addendum.get("schema_version")),
            candidate_artifact_sha256=str(candidate["artifact_sha256"]),
            stage_b_artifact_sha256=str(stage_b["artifact_sha256"]),
            stage_b_attestation_sha256=str(stage_b["attestation_sha256"]),
            cli_sha256=str(_as_mapping(stage_b.get("runtime_attestation")).get("cli_sha256")),
            transport_payload_sha256=str(d2_audit_map["transport"]["payload_sha256"]),
            transport_decode_sha256=str(d2_audit_map["transport"]["decode_sha256"]),
            raw_capture=RealEvidenceCommitment(
                REAL_RAW_CAPTURE_PATH,
                int(d2_audit_map["raw_capture"]["bytes"]),
                str(d2_audit_map["raw_capture"]["sha256"]),
            ),
            bundle=RealEvidenceCommitment("<bundle>", int(bundle["bytes"]), str(bundle["sha256"])),
            bundle_members=tuple(
                RealEvidenceCommitment(name, int(item["bytes"]), str(item["sha256"]))
                for name, item in bundle_members.items()
            ),
            triad=tuple(triad),
        )
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        raise ValueError("real promotion canonical policy unavailable") from None


def _validate_finalized_sidecar_lifecycle(sidecar: Mapping[str, Any], *, stage: str) -> list[str]:
    """Validate the official raw-only finalization contract for one sidecar."""
    retention = _as_mapping(sidecar.get("retention"))
    evidence = _as_mapping(sidecar.get("evidence"))
    raw = _as_mapping(evidence.get("raw_capture"))
    errors: list[str] = []
    if stage == "L049V2StageB":
        expected_retention = {
            "raw_retention_status",
            "raw_present",
            "raw_local_absence_verified",
            "standard_finalize",
            "retention_finalized",
            "repository_promotion",
            "previous_pending_sidecar_sha256",
            "finalize_delete",
            "recommendation",
            "remote_checkout_absence",
        }
        if set(retention) != expected_retention or not (
            retention.get("raw_retention_status") == "deleted_verified"
            and retention.get("raw_present") is False
            and retention.get("raw_local_absence_verified") is True
            and retention.get("standard_finalize") is True
            and retention.get("retention_finalized") is True
            and retention.get("repository_promotion") is False
            and retention.get("remote_checkout_absence") == "not_independently_verified"
        ):
            errors.append("real promotion D2 finalized retention lifecycle is invalid")
        finalize = _as_mapping(retention.get("finalize_delete"))
        if set(finalize) != {
            "mode",
            "dry_run",
            "raw_target_only",
            "audit_lifecycle",
            "quarantine_status",
            "triad_survives",
            "candidate_survives",
        } or finalize != {
            "mode": "official_finalize_delete",
            "dry_run": "PASS",
            "raw_target_only": True,
            "audit_lifecycle": "deleted_verified",
            "quarantine_status": "absent_verified",
            "triad_survives": "yes",
            "candidate_survives": "yes",
        }:
            errors.append("real promotion D2 finalize-delete proof is invalid")
        expected_raw_keys = {
            "path",
            "bytes",
            "sha256",
            "present",
            "present_before_finalize",
            "absence_verified_by",
            "local_absence_proof",
            "written_before_parse",
            "raw_status",
            "quarantine_status",
        }
    else:
        expected_retention = {
            "finalize_delete",
            "previous_pending_sidecar_sha256",
            "raw_local_absence_verified",
            "recommendation",
            "remote_checkout_absence_proof",
            "repository_promotion",
            "standard_finalize",
            "status",
        }
        if set(retention) != expected_retention or not (
            retention.get("status") == "deleted_verified"
            and retention.get("raw_local_absence_verified") is True
            and retention.get("repository_promotion") is False
            and retention.get("standard_finalize") is True
            and retention.get("remote_checkout_absence_proof") is False
        ):
            errors.append("real promotion D1 finalized retention lifecycle is invalid")
        finalize = _as_mapping(retention.get("finalize_delete"))
        if set(finalize) != {
            "audit_lifecycle_mutation",
            "candidate_survives",
            "dry_run_required_before_execution",
            "executed",
            "raw_status",
            "raw_target_only",
            "remote_cleanup_absence_proof",
            "triad_survives",
        } or finalize != {
            "audit_lifecycle_mutation": "deleted_verified",
            "candidate_survives": "yes",
            "dry_run_required_before_execution": True,
            "executed": "official_finalize_delete",
            "raw_status": "deleted_verified",
            "raw_target_only": True,
            "remote_cleanup_absence_proof": "not_available",
            "triad_survives": "yes",
        }:
            errors.append("real promotion D1 finalize-delete proof is invalid")
        expected_raw_keys = {
            "absence_verified_by",
            "bytes",
            "local_absence_proof",
            "path",
            "present",
            "present_before_finalize",
            "sha256",
            "written_before_parse",
        }
    if set(raw) != expected_raw_keys or not (
        raw.get("present") is False
        and raw.get("present_before_finalize") is True
        and raw.get("local_absence_proof") is True
        and raw.get("written_before_parse") is True
        and raw.get("absence_verified_by") == "official_finalize_delete"
    ):
        errors.append("real promotion finalized raw-capture proof is invalid")
    if (
        not isinstance(raw.get("path"), str)
        or not raw.get("path", "").startswith("artifacts/m14/")
        or "\\" in raw.get("path", "")
        or raw.get("path", "").startswith("/")
    ):
        errors.append("real promotion finalized raw path is invalid")
    if stage == "L049V2StageB" and (
        raw.get("raw_status") != "deleted_verified" or raw.get("quarantine_status") != "absent_verified"
    ):
        errors.append("real promotion D2 raw-capture lifecycle is invalid")
    if not _bounded_nonnegative_int(raw.get("bytes")) or not is_digest(raw.get("sha256")):
        errors.append("real promotion finalized raw-capture commitment is invalid")
    return errors


def _validate_real_sidecars(
    d1: Mapping[str, Any],
    d2: Mapping[str, Any],
    provisioning: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_source_commit_sha: str,
    expected_source_tree_algorithm: str,
    expected_source_tree_oid: str,
    policy: CommitmentPolicy,
    real_policy: RealPromotionPolicy,
) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    if not _repo_root_matches(repo_root):
        errors.append("real promotion repository root is invalid")
    errors.extend(_policy_errors(real_policy))
    try:
        canonical_policy = load_real_promotion_policy(repo_root)
    except ValueError:
        canonical_policy = None
    if canonical_policy is None or real_policy != canonical_policy:
        errors.append("real promotion policy is not independently canonical")
    canonical_addendum = _canonical_json(repo_root, "addendum")
    if (
        canonical_addendum is None
        or not _is_runtime_instance(policy, CommitmentPolicy)
        or policy != CommitmentPolicy.from_addendum(dict(canonical_addendum))
    ):
        errors.append("real promotion commitment policy is not independently canonical")
    for name, value in (
        ("d1_assessment", d1),
        ("d2_assessment", d2),
        ("provisioning_assessment", provisioning),
        ("candidate", candidate),
    ):
        if not _canonical_mapping_matches(repo_root, name, value):
            errors.append("real promotion canonical evidence mapping is invalid")
    if (
        expected_source_commit_sha != real_policy.source_commit_sha
        or expected_source_tree_algorithm != real_policy.source_tree_algorithm
        or expected_source_tree_oid != real_policy.source_tree_oid
    ):
        errors.append("real promotion source commitment differs from owner policy")
    errors.extend(_sidecar_digest_errors(d1, "m14-l04.9-v2-d1-retention-assessment-v1"))
    errors.extend(_sidecar_digest_errors(d2, "m14-l04.9-v2-stage-b-d2-assessment-v1"))
    errors.extend(_sidecar_digest_errors(provisioning, "m14-l04.9-v2-stage-b-provisioning-assessment-v1"))
    errors.extend(_validate_finalized_sidecar_lifecycle(d1, stage="L049V2StageA"))
    errors.extend(_validate_finalized_sidecar_lifecycle(d2, stage="L049V2StageB"))
    d2_source = _as_mapping(d2.get("source"))
    d1_source = _as_mapping(d1.get("source"))
    p_source = _as_mapping(provisioning.get("source"))
    if real_policy.addendum_schema != policy.expected_addendum().get("schema_version"):
        errors.append("real promotion addendum schema is not policy-bound")
    if real_policy.parent_plan_sha256 != policy.parent_plan_sha256:
        errors.append("real promotion parent plan is not policy-bound")
    d1_retention = _as_mapping(d1.get("retention"))
    d2_retention = _as_mapping(d2.get("retention"))
    if d1_retention.get("previous_pending_sidecar_sha256") != real_policy.d1_pending_sidecar_sha256:
        errors.append("real promotion D1 predecessor sidecar is not policy-bound")
    if d2_retention.get("previous_pending_sidecar_sha256") != real_policy.d2_pending_sidecar_sha256:
        errors.append("real promotion D2 predecessor sidecar is not policy-bound")
    for commitment in real_policy.file_commitments():
        file_errors, _raw = _file_record_errors(
            repo_root,
            {"path": commitment.path, "bytes": commitment.bytes, "sha256": commitment.sha256},
            commitment,
        )
        errors.extend(file_errors)
    if (
        d2.get("status") != "deleted_verified"
        or d2_source.get("commit_sha") != expected_source_commit_sha
        or d2_source.get("use_case") != "L049V2StageB"
        or d2_source.get("exact_source_verified") is not True
        or d2_source.get("tree_sha256") != expected_source_tree_oid
    ):
        errors.append("real promotion D2 source/retention binding is invalid")
    if (
        d1.get("status") != "deleted_verified"
        or d1_source.get("use_case") != "L049V2StageA"
        or d1_source.get("exact_source_verified") is not True
        or d1_source.get("commit_sha") != real_policy.d1_source_commit_sha
        or d1_source.get("tree_sha256") != real_policy.d1_source_tree_oid
        or not isinstance(d1_source.get("commit_sha"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", d1_source.get("commit_sha", ""))
        or not re.fullmatch(r"[0-9a-f]{40}", str(d1_source.get("tree_sha256", "")))
    ):
        errors.append("real promotion D1 source/retention binding is invalid")
    if (
        provisioning.get("status") != "ready_for_stage_b"
        or p_source.get("use_case") != "L049V2StageB"
        or p_source.get("exact_source_verified") is not True
        or p_source.get("commit_sha") != real_policy.provisioning_source_commit_sha
        or p_source.get("tree_sha256") != real_policy.provisioning_source_tree_oid
        or not re.fullmatch(r"[0-9a-f]{40}", str(p_source.get("commit_sha", "")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(p_source.get("tree_sha256", "")))
    ):
        errors.append("real promotion provisioning source binding is invalid")
    if expected_source_tree_algorithm != "sha1" or not re.fullmatch(r"[0-9a-f]{40}", expected_source_tree_oid):
        errors.append("real promotion source tree must be an explicit sha1 object id")
    for source_record, source_algorithm, source_oid in (
        (d1_source, real_policy.d1_source_tree_algorithm, real_policy.d1_source_tree_oid),
        (p_source, real_policy.provisioning_source_tree_algorithm, real_policy.provisioning_source_tree_oid),
    ):
        commit = source_record.get("commit_sha")
        errors.extend(_repository_tree_errors(repo_root, str(commit), source_algorithm, source_oid))
    d2_assessment = _as_mapping(d2.get("assessment"))
    if (
        d2_assessment.get("evidence_level") != "D2"
        or d2_assessment.get("evaluation_complete") is not True
        or d2_assessment.get("evidence_eligible") is not True
        or d2_assessment.get("promotion_candidate") is not True
        or d2_assessment.get("repository_promotion") is not False
        or d2_assessment.get("semantic_finalization") is not False
        or d2_assessment.get("retention_finalized") is not True
    ):
        errors.append("real promotion D2 assessment is not finalized and eligible")
    d1_assessment = _as_mapping(d1.get("assessment"))
    if (
        d1_assessment.get("evidence_level") != "D1"
        or d1_assessment.get("evidence_eligible") is not True
        or d1_assessment.get("selected_candidate") != {"layer": 10, "offset": 0}
        or d1_assessment.get("stage_b_access") is not False
        or d1_assessment.get("repository_promotion") is not False
    ):
        errors.append("real promotion D1 assessment is invalid")
    p_assessment = _as_mapping(provisioning.get("assessment"))
    if (
        p_assessment.get("d2") is not False
        or p_assessment.get("d3") is not False
        or p_assessment.get("evaluation") is not False
        or p_assessment.get("promotion") is not False
    ):
        errors.append("real promotion provisioning assessment is not a pre-run snapshot")
    d1_evidence = _as_mapping(d1.get("evidence"))
    candidate_raw = _as_mapping(d2.get("inputs")).get("candidate")
    d1_candidate_raw = d1_evidence.get("candidate")
    candidate_record = _as_mapping(candidate_raw)
    d1_candidate_record = _as_mapping(d1_candidate_raw)
    for sidecar, _commitment, canonical_sidecar_sha in (
        (d1, real_policy.d1_assessment, real_policy.d1_assessment_canonical_sha256),
        (d2, real_policy.d2_assessment, real_policy.d2_assessment_canonical_sha256),
        (provisioning, real_policy.provisioning_assessment, real_policy.provisioning_assessment_canonical_sha256),
    ):
        if sidecar.get("sidecar_sha256") != canonical_sidecar_sha:
            errors.append("real promotion sidecar does not match owner policy")
    if not isinstance(candidate_raw, Mapping) or not isinstance(d1_candidate_raw, Mapping):
        errors.append("real promotion candidate sidecar records are missing")
    else:
        if candidate_record.get("canonical_digest") != candidate.get("artifact_sha256"):
            errors.append("real promotion D2 candidate artifact binding is invalid")
        if d1_candidate_record.get("canonical_digest") != candidate.get("artifact_sha256"):
            errors.append("real promotion D1 candidate artifact binding is invalid")
        if candidate_record.get("sha256") != d1_candidate_record.get("sha256"):
            errors.append("real promotion D1/D2 candidate file binding is invalid")
        if (
            candidate_record.get("path") != real_policy.candidate.path
            or candidate_record.get("bytes") != real_policy.candidate.bytes
            or candidate_record.get("sha256") != real_policy.candidate.sha256
            or d1_candidate_record.get("path") != real_policy.d1_candidate.path
            or d1_candidate_record.get("bytes") != real_policy.d1_candidate.bytes
            or d1_candidate_record.get("sha256") != real_policy.d1_candidate.sha256
        ):
            errors.append("real promotion candidate path commitment is invalid")
    p_inputs = _as_mapping(provisioning.get("inputs"))
    expected_inputs = {
        "manifest": real_policy.manifest,
        "holdout": real_policy.holdout,
        "seed": real_policy.seed,
        "candidate": real_policy.candidate,
    }
    input_digests: dict[str, str] = {}
    for name in ("manifest", "holdout", "seed", "candidate"):
        item = p_inputs.get(name)
        file_errors, raw = _file_record_errors(repo_root, item, expected_inputs[name])
        errors.extend(file_errors)
        if raw is not None and isinstance(item, Mapping):
            input_digests[name] = str(item.get("sha256") or item.get("commitment_sha256"))
    if input_digests.get("candidate") != candidate_record.get("sha256"):
        errors.append("real promotion provisioning candidate commitment is invalid")
    if candidate.get("artifact_sha256") != real_policy.candidate_artifact_sha256:
        errors.append("real promotion candidate artifact is not policy-bound")
    manifest_input = _as_mapping(p_inputs.get("manifest"))
    holdout_input = _as_mapping(p_inputs.get("holdout"))
    seed_input = _as_mapping(p_inputs.get("seed"))
    authoring = _as_mapping(policy.expected_addendum().get("authoring"))
    if manifest_input.get("canonical_digest") != authoring.get("manifest_sha256"):
        errors.append("real promotion manifest commitment is invalid")
    if seed_input.get("commitment_sha256") != policy.holdout_seed_commitment_sha256:
        errors.append("real promotion seed commitment is invalid")
    if holdout_input.get("commitment_valid") is not True:
        errors.append("real promotion holdout commitment is invalid")
    return errors, {
        "d1_assessment_sha256": str(d1.get("sidecar_sha256", "")),
        "d2_assessment_sha256": str(d2.get("sidecar_sha256", "")),
        "provisioning_assessment_sha256": str(provisioning.get("sidecar_sha256", "")),
        "d1_candidate_file_sha256": str(d1_candidate_record.get("sha256", "")),
        "candidate_file_sha256": str(candidate_record.get("sha256", "")),
        "provisioning_manifest_sha256": str(manifest_input.get("sha256", "")),
        "provisioning_holdout_sha256": str(holdout_input.get("sha256", "")),
        "provisioning_seed_commitment_sha256": str(seed_input.get("commitment_sha256", "")),
    }


def _validate_official_audit(
    audit: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_source_commit_sha: str,
    expected_use_case: str,
    real_policy: RealPromotionPolicy,
) -> list[str]:
    errors: list[str] = []
    if not _repo_root_matches(repo_root):
        errors.append("real promotion repository root is invalid")
    audit_name = "d2_audit" if expected_use_case == "L049V2StageB" else "d1_audit"
    if not _canonical_mapping_matches(repo_root, audit_name, audit):
        errors.append("real promotion official audit mapping is not canonical")
    expected_audit = real_policy.d2_audit if expected_use_case == "L049V2StageB" else real_policy.d1_audit
    side_audit = _as_mapping(_as_mapping(sidecar.get("evidence")).get("audit"))
    if (
        side_audit.get("path") != expected_audit.path
        or side_audit.get("bytes") != expected_audit.bytes
        or side_audit.get("sha256") != expected_audit.sha256
    ):
        errors.append("real promotion official audit does not match owner policy")
    if set(audit) != {
        "attempt",
        "bundle",
        "envelopes",
        "final_payload",
        "marker_exits",
        "mode",
        "promoted",
        "raw_capture",
        "raw_status",
        "schema_version",
        "source_sha",
        "transport",
        "use_case",
        "validation",
    }:
        errors.append("real promotion official audit fields are invalid")
    if (
        audit.get("schema_version") != "m14-l04-remote-retention-audit-v2"
        or audit.get("attempt") != "attempt1"
        or audit.get("mode") != "retained_pending_finalize"
        or audit.get("source_sha") != expected_source_commit_sha
        or audit.get("use_case") != expected_use_case
        or audit.get("raw_status") != "deleted_verified"
        or audit.get("promoted") is not False
        or audit.get("validation") != {"archive": "PASS", "envelopes": "PASS"}
    ):
        errors.append("real promotion official audit lifecycle is invalid")
    raw = _as_mapping(audit.get("raw_capture"))
    quarantine = _as_mapping(raw.get("quarantine"))
    side_raw = _as_mapping(_as_mapping(sidecar.get("evidence")).get("raw_capture"))
    if (
        set(raw) != {"bytes", "path", "quarantine", "sha256"}
        or set(quarantine) != {"bytes", "path", "sha256", "status"}
        or not _bounded_nonnegative_int(raw.get("bytes"))
        or not is_digest(raw.get("sha256"))
        or raw.get("path") != "<raw-capture-path>"
        or quarantine.get("status") != "absent_verified"
        or raw.get("bytes") != side_raw.get("bytes")
        or raw.get("sha256") != side_raw.get("sha256")
        or side_raw.get("present") is not False
        or side_raw.get("present_before_finalize") is not True
        or side_raw.get("local_absence_proof") is not True
        or side_raw.get("written_before_parse") is not True
        or side_raw.get("absence_verified_by") != "official_finalize_delete"
        or quarantine.get("bytes") != raw.get("bytes")
        or quarantine.get("sha256") != raw.get("sha256")
        or quarantine.get("path") != "<raw-quarantine-path>"
    ):
        errors.append("real promotion official raw deletion proof is invalid")
    if expected_use_case == "L049V2StageB" and (
        side_raw.get("path") != real_policy.raw_capture.path
        or raw.get("bytes") != real_policy.raw_capture.bytes
        or raw.get("sha256") != real_policy.raw_capture.sha256
    ):
        errors.append("real promotion D2 raw commitment is invalid")
    markers = _as_mapping(audit.get("marker_exits"))
    if markers != {
        "transport_decode": 0,
        "cli": 0,
        "bundle": 0,
        "final": 0,
        "transport_cleanup": "PASS",
        "remote_cleanup": "PASS",
    }:
        errors.append("real promotion official marker proof is invalid")
    envelopes = _as_mapping(audit.get("envelopes"))
    if set(envelopes) != set(PROMOTION_MEMBER_KINDS):
        errors.append("real promotion official envelope proof is invalid")
    if any(envelope != "PASS" for envelope in envelopes.values()):
        errors.append("real promotion official envelope proof is invalid")
    transport = _as_mapping(audit.get("transport"))
    if (
        set(transport) != {"payload_sha256", "decode_sha256", "decode_match", "workdir"}
        or not is_digest(transport.get("payload_sha256"))
        or transport.get("decode_sha256") != transport.get("payload_sha256")
        or transport.get("decode_match") != "PASS"
        or transport.get("workdir") != "<remote-workdir>"
    ):
        errors.append("real promotion official transport proof is invalid")
    if expected_use_case == "L049V2StageB" and (
        transport.get("payload_sha256") != real_policy.transport_payload_sha256
        or transport.get("decode_sha256") != real_policy.transport_decode_sha256
    ):
        errors.append("real promotion D2 transport commitment is invalid")
    bundle = _as_mapping(audit.get("bundle"))
    members = _as_mapping(bundle.get("members"))
    side_evidence = _as_mapping(sidecar.get("evidence"))
    side_bundle = _as_mapping(side_evidence.get("bundle"))
    expected_names = side_bundle.get("member_names")
    if (
        not _bounded_nonnegative_int(bundle.get("bytes"))
        or not is_digest(bundle.get("sha256"))
        or not isinstance(expected_names, list)
        or sorted(members) != sorted(expected_names)
        or bundle.get("announced_members") != members
        or bundle.get("announced_bytes") != bundle.get("bytes")
        or bundle.get("announced_sha256") != bundle.get("sha256")
        or (
            side_bundle.get("present") is not True
            if expected_use_case == "L049V2StageB"
            else side_bundle.get("present_in_raw_capture") is not True
        )
        or side_bundle.get("bytes") != bundle.get("bytes")
        or side_bundle.get("sha256") != bundle.get("sha256")
        or (
            side_bundle.get("member_reopen_validation") != "PASS"
            if expected_use_case == "L049V2StageB"
            else side_bundle.get("present_in_raw_capture") is not True
        )
    ):
        errors.append("real promotion official bundle proof is invalid")
    if expected_use_case == "L049V2StageB" and (
        bundle.get("bytes") != real_policy.bundle.bytes or bundle.get("sha256") != real_policy.bundle.sha256
    ):
        errors.append("real promotion D2 bundle commitment is invalid")
    if expected_use_case == "L049V2StageB":
        expected_bundle_members = {item.path: (item.bytes, item.sha256) for item in real_policy.bundle_members}
        actual_bundle_members = {
            str(name): (item.get("bytes"), item.get("sha256"))
            for name, item in members.items()
            if isinstance(item, Mapping)
        }
        if actual_bundle_members != expected_bundle_members:
            errors.append("real promotion D2 bundle member commitments are invalid")
    for member in members.values():
        if (
            not isinstance(member, Mapping)
            or not _bounded_nonnegative_int(member.get("bytes"))
            or not is_digest(member.get("sha256"))
        ):
            errors.append("real promotion official bundle member proof is invalid")
    final = _as_mapping(audit.get("final_payload"))
    paths = _as_mapping(final.get("paths"))
    side_triad = _as_mapping(side_evidence.get("triad"))
    expected_triad = {
        str(item.get("path", "")).replace("artifacts/m14/", ""): item
        for item in side_triad.values()
        if isinstance(item, Mapping)
    }
    if final.get("reopen_validation") != "PASS" or set(paths) != set(expected_triad):
        errors.append("real promotion official retained triad binding is invalid")
    for name, item in paths.items():
        if not isinstance(item, Mapping) or item.get("path") != f"artifacts/m14/{name}":
            errors.append("real promotion official triad path binding is invalid")
            continue
        file_errors, _raw = _file_record_errors(
            repo_root,
            {
                "path": item.get("path"),
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
            },
        )
        errors.extend(file_errors)
        expected = expected_triad.get(name)
        if isinstance(expected, Mapping) and (
            item.get("bytes") != expected.get("bytes") or item.get("sha256") != expected.get("sha256")
        ):
            errors.append("real promotion official triad digest binding is invalid")
    if expected_use_case == "L049V2StageB":
        expected_triad_commitments = {item.path: (item.bytes, item.sha256) for item in real_policy.triad}
        actual_triad_commitments = {
            str(item.get("path")): (item.get("bytes"), item.get("sha256"))
            for item in paths.values()
            if isinstance(item, Mapping)
        }
        if actual_triad_commitments != expected_triad_commitments:
            errors.append("real promotion D2 triad commitments are invalid")
    return errors


def _validate_real_promotion_record_impl(
    record: Mapping[str, Any],
    stage_b: Mapping[str, Any],
    candidate: Mapping[str, Any],
    addendum: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    holdout_seed: bytes,
    transport: Mapping[str, Any],
    retention_audit: Mapping[str, Any],
    d1_assessment: Mapping[str, Any],
    d2_assessment: Mapping[str, Any],
    provisioning_assessment: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_source_commit_sha: str,
    expected_source_tree_algorithm: str,
    expected_source_tree_oid: str,
    policy: CommitmentPolicy,
    real_policy: RealPromotionPolicy,
) -> list[str]:
    errors: list[str] = []
    if not _repo_root_matches(repo_root):
        errors.append("real promotion repository root is invalid")
    if not _canonical_mapping_matches(repo_root, "addendum", addendum):
        errors.append("real promotion addendum mapping is not canonical")
    canonical_addendum = _canonical_json(repo_root, "addendum")
    if (
        canonical_addendum is None
        or not _is_runtime_instance(policy, CommitmentPolicy)
        or policy != CommitmentPolicy.from_addendum(dict(canonical_addendum))
    ):
        errors.append("real promotion commitment policy is not independently canonical")
    if not _canonical_mapping_matches(repo_root, "candidate", candidate):
        errors.append("real promotion candidate mapping is not canonical")
    partial = _canonical_json(repo_root, "d2_partial")
    if partial is None or partial.get("artifact") != dict(stage_b):
        errors.append("real promotion Stage B artifact mapping is not canonical")
    for name in ("d2_failure", "d2_partial", "d2_run"):
        envelope = _canonical_json(repo_root, name)
        if envelope is None or envelope.get("artifact_sha256") != stage_b.get("artifact_sha256"):
            errors.append("real promotion Stage B envelope binding is invalid")
    if not _canonical_rows_matches(repo_root, "train", train_rows):
        errors.append("real promotion train rows are not canonical")
    if not _canonical_rows_matches(repo_root, "holdout", holdout_rows):
        errors.append("real promotion holdout rows are not canonical")
    seed_raw = _canonical_raw(repo_root, "seed")
    if seed_raw is None or seed_raw[1] != holdout_seed:
        errors.append("real promotion holdout seed is not canonical")
    errors.extend(
        _repository_tree_errors(
            repo_root,
            expected_source_commit_sha,
            expected_source_tree_algorithm,
            expected_source_tree_oid,
        )
    )
    if set(record) != REAL_PROMOTION_FIELDS:
        errors.append("real promotion record fields are invalid")
    errors.extend(
        f"Stage B prerequisite: {error}"
        for error in validate_stage_b_impl(
            stage_b, holdout_rows, holdout_seed, candidate, addendum, train_rows, policy=policy
        )
    )
    runtime_attestation = _as_mapping(stage_b.get("runtime_attestation"))
    if (
        stage_b.get("artifact_sha256") != real_policy.stage_b_artifact_sha256
        or stage_b.get("attestation_sha256") != real_policy.stage_b_attestation_sha256
        or runtime_attestation.get("cli_sha256") != real_policy.cli_sha256
        or candidate.get("artifact_sha256") != real_policy.candidate_artifact_sha256
        or transport.get("payload_sha256") != real_policy.transport_payload_sha256
        or transport.get("decode_sha256") != real_policy.transport_decode_sha256
    ):
        errors.append("real promotion runtime commitment differs from owner policy")
    errors.extend(
        _validate_real_sidecars(
            d1_assessment,
            d2_assessment,
            provisioning_assessment,
            candidate,
            repo_root=repo_root,
            expected_source_commit_sha=expected_source_commit_sha,
            expected_source_tree_algorithm=expected_source_tree_algorithm,
            expected_source_tree_oid=expected_source_tree_oid,
            policy=policy,
            real_policy=real_policy,
        )[0]
    )
    errors.extend(
        _validate_official_audit(
            retention_audit,
            d2_assessment,
            repo_root=repo_root,
            expected_source_commit_sha=expected_source_commit_sha,
            expected_use_case="L049V2StageB",
            real_policy=real_policy,
        )
    )
    official_transport = _as_mapping(retention_audit.get("transport"))
    if dict(transport) != dict(official_transport):
        errors.append("real promotion transport metadata is not bound to official audit")
    d2_sidecar_evidence = _as_mapping(d2_assessment.get("evidence"))
    d2_audit_record = _as_mapping(d2_sidecar_evidence.get("audit"))
    d1_evidence = _as_mapping(d1_assessment.get("evidence"))
    d1_audit_record = _as_mapping(d1_evidence.get("audit"))
    d1_source = _as_mapping(d1_assessment.get("source"))
    d1_audit = _load_bound_json(repo_root, d1_audit_record)
    d1_audit_file_errors, _d1_audit_raw = _file_record_errors(repo_root, d1_audit_record)
    errors.extend(d1_audit_file_errors)
    if d1_audit is None:
        errors.append("real promotion D1 official audit is missing or malformed")
    else:
        errors.extend(
            _validate_official_audit(
                d1_audit,
                d1_assessment,
                repo_root=repo_root,
                expected_source_commit_sha=str(d1_source.get("commit_sha", "")),
                expected_use_case="L049V2StageA",
                real_policy=real_policy,
            )
        )
    d2_audit_bound = _safe_bound_file(repo_root, d2_audit_record.get("path"))
    d2_audit_sha = d2_audit_record.get("sha256")
    d2_audit_matches = False
    if d2_audit_bound is not None:
        try:
            d2_audit_matches = (
                is_digest(d2_audit_sha)
                and hashlib.sha256(d2_audit_bound[1]).hexdigest() == d2_audit_sha
                and json.loads(d2_audit_bound[1].decode("utf-8")) == dict(retention_audit)
            )
        except (UnicodeError, json.JSONDecodeError):
            d2_audit_matches = False
    if not d2_audit_matches:
        errors.append("real promotion D2 official audit file binding is invalid")
    if (
        d2_audit_record.get("prior_pending_sha256") != real_policy.d2_pending_audit_sha256
        or d2_audit_record.get("prior_pending_bytes") != real_policy.d2_pending_audit_bytes
        or d2_audit_record.get("prior_pending_sha256")
        != _as_mapping(d2_sidecar_evidence.get("audit")).get("prior_pending_sha256")
        or not _bounded_nonnegative_int(d2_audit_record.get("prior_pending_bytes"))
    ):
        errors.append("real promotion D2 pending audit predecessor binding is invalid")
    predecessor = _as_mapping(d2_assessment.get("retention"))
    if predecessor.get("previous_pending_sidecar_sha256") != real_policy.d2_pending_sidecar_sha256:
        errors.append("real promotion pending D2 predecessor binding is invalid")
    if (
        d1_assessment.get("retention", {}).get("previous_pending_sidecar_sha256")
        != real_policy.d1_pending_sidecar_sha256
    ):
        errors.append("real promotion pending D1 predecessor binding is invalid")
    if (
        record.get("schema_version") != REAL_V2_PROMOTION_SCHEMA
        or record.get("stage") != stage_b.get("stage")
        or record.get("status") != "accepted"
        or record.get("evidence_level") != "D3"
        or record.get("evidence_eligible") is not True
        or record.get("repository_promotion") is not True
        or record.get("promotion_candidate") is not True
    ):
        errors.append("real promotion status/eligibility is invalid")
    expected = {
        "stage_b_artifact_sha256": real_policy.stage_b_artifact_sha256,
        "stage_b_attestation_sha256": real_policy.stage_b_attestation_sha256,
        "candidate_artifact_sha256": real_policy.candidate_artifact_sha256,
        "candidate_file_sha256": real_policy.candidate.sha256,
        "parent_plan_sha256": policy.parent_plan_sha256,
        "addendum_schema": policy.expected_addendum().get("schema_version"),
        "source_commit_sha": expected_source_commit_sha,
        "source_tree": {"algorithm": expected_source_tree_algorithm, "oid": expected_source_tree_oid},
        "d1_source_commit_sha": real_policy.d1_source_commit_sha,
        "d1_source_tree": {
            "algorithm": real_policy.d1_source_tree_algorithm,
            "oid": real_policy.d1_source_tree_oid,
        },
        "provisioning_source_commit_sha": real_policy.provisioning_source_commit_sha,
        "provisioning_source_tree": {
            "algorithm": real_policy.provisioning_source_tree_algorithm,
            "oid": real_policy.provisioning_source_tree_oid,
        },
        "d1_pending_sidecar_sha256": real_policy.d1_pending_sidecar_sha256,
        "d1_pending_audit_sha256": real_policy.d1_pending_audit_sha256,
        "d1_pending_audit_bytes": real_policy.d1_pending_audit_bytes,
        "d2_pending_audit_sha256": real_policy.d2_pending_audit_sha256,
        "d2_pending_audit_bytes": real_policy.d2_pending_audit_bytes,
        "cli_sha256": real_policy.cli_sha256,
        "transport_payload_sha256": real_policy.transport_payload_sha256,
        "transport_decode_sha256": real_policy.transport_decode_sha256,
        "transport_decode_match": transport.get("decode_match"),
        "bundle_bytes": real_policy.bundle.bytes,
        "bundle_sha256": real_policy.bundle.sha256,
        "bundle_member_sha256": {item.path: item.sha256 for item in real_policy.bundle_members},
        "retention_audit_sha256": real_policy.d2_audit.sha256,
        "retention_audit_schema": retention_audit.get("schema_version"),
        "d1_assessment_sha256": real_policy.d1_assessment_canonical_sha256,
        "d1_audit_sha256": real_policy.d1_audit.sha256,
        "d1_candidate_file_sha256": real_policy.d1_candidate.sha256,
        "d2_assessment_sha256": real_policy.d2_assessment_canonical_sha256,
        "d2_audit_sha256": real_policy.d2_audit.sha256,
        "provisioning_assessment_sha256": real_policy.provisioning_assessment_canonical_sha256,
        "provisioning_manifest_sha256": real_policy.manifest.sha256,
        "provisioning_holdout_sha256": real_policy.holdout.sha256,
        "provisioning_seed_commitment_sha256": real_policy.seed.sha256,
        "pending_retention_sidecar_sha256": real_policy.d2_pending_sidecar_sha256,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            errors.append(f"real promotion {field} binding is invalid")
    members = _as_mapping(_as_mapping(retention_audit.get("final_payload")).get("paths"))
    expected_members = {
        str(Path(str(item.get("path", ""))).name): item.get("sha256")
        for item in members.values()
        if isinstance(item, Mapping)
    }
    if record.get("retained_member_sha256") != expected_members:
        errors.append("real promotion retained member hashes are invalid")
    try:
        if record.get("promotion_sha256") != canonical_digest(dict(record), "promotion_sha256"):
            errors.append("real promotion self-digest is invalid")
    except (TypeError, ValueError, OverflowError):
        errors.append("real promotion self-digest is invalid")
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
    d1_assessment: Mapping[str, Any],
    d2_assessment: Mapping[str, Any],
    provisioning_assessment: Mapping[str, Any],
    repo_root: Path,
    expected_source_commit_sha: str,
    expected_source_tree_algorithm: str,
    expected_source_tree_oid: str,
    policy: CommitmentPolicy,
    real_policy: RealPromotionPolicy,
) -> list[str]:
    """Validate the real-evidence D3 record without calling the builder."""
    if (
        not _mapping_inputs_valid(
            record,
            stage_b,
            candidate,
            addendum,
            transport,
            retention_audit,
            d1_assessment,
            d2_assessment,
            provisioning_assessment,
        )
        or not _rows_input_valid(train_rows)
        or not _rows_input_valid(holdout_rows)
        or not all(_sidecar_shape_valid(value) for value in (d1_assessment, d2_assessment, provisioning_assessment))
        or not _is_runtime_instance(real_policy, RealPromotionPolicy)
        or not _is_runtime_instance(policy, CommitmentPolicy)
        or not _is_runtime_instance(holdout_seed, bytes)
        or not _is_runtime_instance(repo_root, Path)
    ):
        return ["real promotion malformed input"]
    try:
        return _validate_real_promotion_record_impl(
            record,
            stage_b,
            candidate,
            addendum,
            train_rows,
            holdout_rows,
            holdout_seed,
            transport,
            retention_audit,
            d1_assessment,
            d2_assessment,
            provisioning_assessment,
            repo_root=repo_root,
            expected_source_commit_sha=expected_source_commit_sha,
            expected_source_tree_algorithm=expected_source_tree_algorithm,
            expected_source_tree_oid=expected_source_tree_oid,
            policy=policy,
            real_policy=real_policy,
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, OSError, UnicodeError):
        return ["real promotion malformed input"]


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
    d1_assessment: Mapping[str, Any],
    d2_assessment: Mapping[str, Any],
    provisioning_assessment: Mapping[str, Any],
    repo_root: Path,
    expected_source_commit_sha: str,
    expected_source_tree_algorithm: str,
    expected_source_tree_oid: str,
    policy: CommitmentPolicy,
    real_policy: RealPromotionPolicy,
) -> dict[str, Any]:
    """Create a real-evidence D3 record after every prerequisite passes."""
    if (
        not _mapping_inputs_valid(
            stage_b,
            candidate,
            addendum,
            transport,
            retention_audit,
            d1_assessment,
            d2_assessment,
            provisioning_assessment,
        )
        or not _is_runtime_instance(real_policy, RealPromotionPolicy)
        or not _is_runtime_instance(policy, CommitmentPolicy)
        or not _is_runtime_instance(holdout_seed, bytes)
        or not _is_runtime_instance(repo_root, Path)
    ):
        raise ValueError("real promotion malformed input")
    sidecar_errors, sidecar_bindings = _validate_real_sidecars(
        d1_assessment,
        d2_assessment,
        provisioning_assessment,
        candidate,
        repo_root=repo_root,
        expected_source_commit_sha=expected_source_commit_sha,
        expected_source_tree_algorithm=expected_source_tree_algorithm,
        expected_source_tree_oid=expected_source_tree_oid,
        policy=policy,
        real_policy=real_policy,
    )
    audit_errors = _validate_official_audit(
        retention_audit,
        d2_assessment,
        repo_root=repo_root,
        expected_source_commit_sha=expected_source_commit_sha,
        expected_use_case="L049V2StageB",
        real_policy=real_policy,
    )
    prerequisite_errors = (
        sidecar_errors
        + audit_errors
        + validate_stage_b_impl(stage_b, holdout_rows, holdout_seed, candidate, addendum, train_rows, policy=policy)
    )
    if prerequisite_errors:
        raise ValueError("real promotion prerequisites failed: " + "; ".join(prerequisite_errors))
    d2_evidence = d2_assessment["evidence"]
    d1_evidence = d1_assessment["evidence"]
    p_inputs = provisioning_assessment["inputs"]
    paths = retention_audit["final_payload"]["paths"]
    record: dict[str, Any] = {
        "schema_version": REAL_V2_PROMOTION_SCHEMA,
        "stage": stage_b["stage"],
        "status": "accepted",
        "evidence_level": "D3",
        "evidence_eligible": True,
        "repository_promotion": True,
        "promotion_candidate": True,
        "stage_b_artifact_sha256": stage_b["artifact_sha256"],
        "stage_b_attestation_sha256": stage_b["attestation_sha256"],
        "candidate_artifact_sha256": candidate["artifact_sha256"],
        "candidate_file_sha256": sidecar_bindings["candidate_file_sha256"],
        "parent_plan_sha256": policy.parent_plan_sha256,
        "addendum_schema": policy.expected_addendum()["schema_version"],
        "source_commit_sha": expected_source_commit_sha,
        "source_tree": {"algorithm": expected_source_tree_algorithm, "oid": expected_source_tree_oid},
        "d1_source_commit_sha": d1_assessment["source"]["commit_sha"],
        "d1_source_tree": {"algorithm": "sha1", "oid": d1_assessment["source"]["tree_sha256"]},
        "provisioning_source_commit_sha": provisioning_assessment["source"]["commit_sha"],
        "provisioning_source_tree": {
            "algorithm": "sha1",
            "oid": provisioning_assessment["source"]["tree_sha256"],
        },
        "d1_pending_sidecar_sha256": d1_assessment["retention"]["previous_pending_sidecar_sha256"],
        "d1_pending_audit_sha256": real_policy.d1_pending_audit_sha256,
        "d1_pending_audit_bytes": real_policy.d1_pending_audit_bytes,
        "d2_pending_audit_sha256": d2_evidence["audit"]["prior_pending_sha256"],
        "d2_pending_audit_bytes": d2_evidence["audit"]["prior_pending_bytes"],
        "cli_sha256": stage_b["runtime_attestation"]["cli_sha256"],
        "transport_payload_sha256": transport["payload_sha256"],
        "transport_decode_sha256": transport["decode_sha256"],
        "transport_decode_match": transport["decode_match"],
        "bundle_bytes": retention_audit["bundle"]["bytes"],
        "bundle_sha256": retention_audit["bundle"]["sha256"],
        "bundle_member_sha256": {
            str(name): item["sha256"] for name, item in retention_audit["bundle"]["members"].items()
        },
        "retention_audit_sha256": d2_evidence["audit"]["sha256"],
        "retention_audit_schema": retention_audit["schema_version"],
        "d1_assessment_sha256": d1_assessment["sidecar_sha256"],
        "d1_audit_sha256": d1_evidence["audit"]["sha256"],
        "d1_candidate_file_sha256": d1_evidence["candidate"]["sha256"],
        "d2_assessment_sha256": d2_assessment["sidecar_sha256"],
        "d2_audit_sha256": d2_evidence["audit"]["sha256"],
        "provisioning_assessment_sha256": provisioning_assessment["sidecar_sha256"],
        "provisioning_manifest_sha256": p_inputs["manifest"]["sha256"],
        "provisioning_holdout_sha256": p_inputs["holdout"]["sha256"],
        "provisioning_seed_commitment_sha256": p_inputs["seed"]["commitment_sha256"],
        "retained_member_sha256": {str(Path(str(item["path"])).name): item["sha256"] for item in paths.values()},
        "pending_retention_sidecar_sha256": d2_assessment["retention"]["previous_pending_sidecar_sha256"],
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
        d1_assessment=d1_assessment,
        d2_assessment=d2_assessment,
        provisioning_assessment=provisioning_assessment,
        repo_root=repo_root,
        expected_source_commit_sha=expected_source_commit_sha,
        expected_source_tree_algorithm=expected_source_tree_algorithm,
        expected_source_tree_oid=expected_source_tree_oid,
        policy=policy,
        real_policy=real_policy,
    )
    if final_errors:
        raise ValueError("real promotion record validation failed: " + "; ".join(final_errors))
    return record


__all__ = [
    "LEGACY_V2_PROMOTION_SCHEMA",
    "PROMOTION_MEMBER_KINDS",
    "RealEvidenceCommitment",
    "RealPromotionPolicy",
    "REAL_V2_PROMOTION_SCHEMA",
    "build_legacy_promotion_record",
    "build_promotion_record",
    "load_real_promotion_policy",
    "validate_legacy_promotion_record",
    "validate_promotion_record",
]
