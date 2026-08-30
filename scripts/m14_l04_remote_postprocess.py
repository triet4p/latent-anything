"""CLI for fail-closed L04 remote evidence retention and finalization."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from scripts._m14_l04_retention_parser import (
    SHA1_RE,
    SHA256_RE,
    USE_CASES,
    RetentionError,
    inspect_archive,
    int_marker,
    json_load,
    member_attempt,
    parse_capture,
    parse_member_markers,
    sha256,
    size_marker,
    validate_triplet,
)
from scripts._m14_l04_retention_transaction import (
    delete_quarantine,
    install_payloads,
    json_bytes,
    quarantine_raw,
    reopen_payloads,
    restore_quarantine,
    restore_raw_snapshot,
    write_atomic,
)
from scripts.m14_l04_contract import FIXTURE_PATH, PLAN_PATH, load_and_validate, load_plan


def _audit_base(
    raw: bytes,
    source_sha: str,
    use_case: str,
    markers: dict[str, str],
    member_markers: dict[str, tuple[int, str]],
    archive: bytes,
    observed_members: dict[str, tuple[int, str]],
    attempt: str,
) -> dict[str, Any]:
    return {
        "schema_version": "m14-l04-remote-retention-audit-v2",
        "source_sha": source_sha,
        "use_case": use_case,
        "attempt": attempt,
        "raw_capture": {"bytes": len(raw), "sha256": sha256(raw), "path": "<raw-capture-path>"},
        "marker_exits": {
            "transport_decode": int_marker(markers, "L04_TRANSPORT_DECODE_STATUS"),
            "cli": int_marker(markers, "L04_CLI_STATUS"),
            "bundle": int_marker(markers, "L04_BUNDLE_STATUS"),
            "final": int_marker(markers, "L04_STATUS"),
            "transport_cleanup": markers["L04_TRANSPORT_CLEANUP"],
            "remote_cleanup": markers["L04_CLEANUP"],
        },
        "transport": {
            "payload_sha256": markers["L04_TRANSPORT_PAYLOAD_SHA256"],
            "decode_sha256": markers["L04_TRANSPORT_DECODE_SHA256"],
            "decode_match": markers["L04_TRANSPORT_DECODE_MATCH"],
            "workdir": "<remote-workdir>",
        },
        "bundle": {
            "bytes": len(archive),
            "sha256": sha256(archive),
            "announced_bytes": size_marker(markers, "L04_BUNDLE_BYTES"),
            "announced_sha256": markers["L04_BUNDLE_SHA256"],
            "members": {
                path: {"bytes": size, "sha256": digest} for path, (size, digest) in sorted(observed_members.items())
            },
            "announced_members": {
                path: {"bytes": size, "sha256": digest} for path, (size, digest) in sorted(member_markers.items())
            },
        },
        "validation": {"archive": "PASS", "envelopes": "PASS"},
        "final_payload": {"paths": {}, "reopen_validation": "PENDING"},
        "raw_status": "not_touched",
        "promoted": False,
    }


def _parse_and_validate(
    raw: bytes, source_sha: str, use_case: str, plan_path: Path, fixture_path: Path
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    if not SHA1_RE.fullmatch(source_sha) or source_sha != source_sha.lower():
        raise RetentionError("source SHA must be lowercase 40-character hexadecimal")
    if use_case not in USE_CASES:
        raise RetentionError(f"invalid use-case {use_case!r}")
    markers, member_values, archive = parse_capture(raw)
    if markers["L04_USE_CASE"] != use_case or markers["L04_CODE_SHA"] != source_sha:
        raise RetentionError("capture use-case/source SHA markers do not match expected values")
    if (
        markers["L04_TRANSPORT_DECODE_MATCH"] != "PASS"
        or markers["L04_TRANSPORT_CLEANUP"] != "PASS"
        or int_marker(markers, "L04_TRANSPORT_DECODE_STATUS") != 0
        or markers["L04_TRANSPORT_DECODE_SHA256"] != markers["L04_TRANSPORT_PAYLOAD_SHA256"]
        or SHA256_RE.fullmatch(markers["L04_TRANSPORT_PAYLOAD_SHA256"]) is None
        or markers["L04_CLEANUP"] != "PASS"
    ):
        raise RetentionError("transport decode/cleanup did not pass")
    cli_status = int_marker(markers, "L04_CLI_STATUS")
    bundle_status = int_marker(markers, "L04_BUNDLE_STATUS")
    final_status = int_marker(markers, "L04_STATUS")
    expected_final = cli_status if cli_status != 0 else bundle_status if bundle_status != 0 else 0
    if final_status != expected_final:
        raise RetentionError("bundle/final status markers are inconsistent")
    if len(archive) != size_marker(markers, "L04_BUNDLE_BYTES") or sha256(archive) != markers["L04_BUNDLE_SHA256"]:
        raise RetentionError("announced bundle size/SHA-256 does not match decoded archive")
    attempt = member_attempt(member_values, use_case)
    announced_members = parse_member_markers(member_values, use_case, attempt)
    load_and_validate(plan_path, fixture_path)
    plan = load_plan(plan_path)
    archive_attempt, files, observed_members = inspect_archive(archive, use_case, announced_members)
    if archive_attempt != attempt:
        raise RetentionError("archive and marker attempts do not match")
    envelopes = validate_triplet(files, plan, source_sha, use_case, attempt)
    audit = _audit_base(raw, source_sha, use_case, markers, announced_members, archive, observed_members, attempt)
    audit["envelopes"] = {kind: "PASS" for kind in envelopes}
    return audit, files, plan


def retain_capture(
    *,
    raw_capture_path: Path,
    source_sha: str,
    use_case: str,
    artifact_dir: Path,
    audit_path: Path,
    plan_path: Path = PLAN_PATH,
    fixture_path: Path = FIXTURE_PATH,
    retain: bool = False,
    validate_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate and optionally retain; this operation never deletes raw bytes."""
    if retain == validate_only:
        raise RetentionError("choose exactly one of retain or validate-only")
    raw = raw_capture_path.read_bytes()
    audit, files, plan = _parse_and_validate(raw, source_sha, use_case, plan_path, fixture_path)
    if dry_run:
        audit["mode"] = "dry-run"
        return audit
    if validate_only:
        audit["mode"] = "validate-only"
        audit["raw_status"] = "retained_pending_finalize"
        return audit
    audit["mode"] = "retained_pending_finalize"
    audit["raw_status"] = "retained_pending_finalize"
    created: list[Path] = []
    try:
        created = install_payloads(artifact_dir, files)
        final_hashes = reopen_payloads(artifact_dir, files, plan, source_sha, use_case, audit["attempt"])
        audit["final_payload"] = {"paths": final_hashes, "reopen_validation": "PASS"}
        write_atomic(audit_path, json_bytes(audit), replace=False)
        if json_load(audit_path.read_bytes(), "audit") != audit:
            raise RetentionError("audit reopen verification failed")
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return audit


def finalize_delete(
    *,
    raw_capture_path: Path,
    source_sha: str,
    use_case: str,
    artifact_dir: Path,
    audit_path: Path,
    plan_path: Path = PLAN_PATH,
    fixture_path: Path = FIXTURE_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Finalize a pending audit through same-directory quarantine and deletion."""
    prior_audit_bytes = audit_path.read_bytes()
    prior = json_load(prior_audit_bytes, "audit")
    if prior.get("schema_version") != "m14-l04-remote-retention-audit-v2":
        raise RetentionError("audit schema is not finalizable")
    if prior.get("source_sha") != source_sha or prior.get("use_case") != use_case:
        raise RetentionError("audit provenance does not match finalization request")
    if prior.get("mode") != "retained_pending_finalize":
        raise RetentionError("audit mode is not retained_pending_finalize")
    if prior.get("raw_status") != "retained_pending_finalize":
        raise RetentionError("audit is not pending finalization")
    raw = raw_capture_path.read_bytes()
    raw_record = prior.get("raw_capture")
    if (
        not isinstance(raw_record, dict)
        or raw_record.get("bytes") != len(raw)
        or raw_record.get("sha256") != sha256(raw)
    ):
        raise RetentionError("pending audit raw capture hash/size does not match the snapshot")
    reconstructed, files, plan = _parse_and_validate(raw, source_sha, use_case, plan_path, fixture_path)
    reconstructed["mode"] = "retained_pending_finalize"
    reconstructed["raw_status"] = "retained_pending_finalize"
    reconstructed["final_payload"] = {
        "paths": reopen_payloads(artifact_dir, files, plan, source_sha, use_case, str(reconstructed["attempt"])),
        "reopen_validation": "PASS",
    }
    # Lifecycle fields are allowed to advance, but every provenance, marker,
    # bundle, member, validator, and final-path field must match the raw source.
    prior_core = copy.deepcopy(prior)
    reconstructed_core = copy.deepcopy(reconstructed)
    for field in ("raw_status", "final_payload"):
        prior_core.pop(field, None)
        reconstructed_core.pop(field, None)
    if prior_core != reconstructed_core:
        raise RetentionError("pending audit does not match reparsed raw evidence")
    expected_pending = copy.deepcopy(reconstructed)
    expected_pending["mode"] = "retained_pending_finalize"
    expected_pending["raw_status"] = "retained_pending_finalize"
    expected_pending["final_payload"] = {
        "paths": reopen_payloads(artifact_dir, files, plan, source_sha, use_case, str(expected_pending["attempt"])),
        "reopen_validation": "PASS",
    }
    stored_pending = json_load(prior_audit_bytes, "audit")
    if stored_pending.get("final_payload") != expected_pending["final_payload"]:
        raise RetentionError("pending audit final payload does not match reopened files")
    if dry_run:
        result = dict(stored_pending)
        result["mode"] = "dry-run-finalize"
        return result
    pending = expected_pending
    quarantine = quarantine_raw(raw_capture_path)
    quarantine_audit = copy.deepcopy(pending)
    quarantine_audit["raw_status"] = "quarantined_pending_delete"
    quarantine_audit["raw_capture"]["quarantine"] = {
        "path": "<raw-quarantine-path>",
        "bytes": len(raw),
        "sha256": sha256(raw),
        "status": "pending_delete",
    }
    try:
        write_atomic(audit_path, json_bytes(quarantine_audit), replace=True)
        if json_load(audit_path.read_bytes(), "audit") != quarantine_audit:
            raise RetentionError("quarantine audit reopen verification failed")
    except Exception as exc:
        try:
            restore_quarantine(quarantine, raw_capture_path, raw)
            write_atomic(audit_path, prior_audit_bytes, replace=True)
            if audit_path.read_bytes() != prior_audit_bytes:
                raise RetentionError("pending audit restoration was not verified")
        except Exception as restore_exc:
            raise RetentionError(f"quarantine rollback failed; retained at {quarantine}") from restore_exc
        raise exc
    try:
        delete_quarantine(quarantine)
        finalized = copy.deepcopy(quarantine_audit)
        finalized["raw_status"] = "deleted_verified"
        finalized["raw_capture"]["quarantine"]["status"] = "absent_verified"
        write_atomic(audit_path, json_bytes(finalized), replace=True)
        if json_load(audit_path.read_bytes(), "audit") != finalized:
            raise RetentionError("final audit reopen verification failed")
        if raw_capture_path.exists() or quarantine.exists():
            raise RetentionError("original raw and quarantine paths were not both absent")
        return finalized
    except Exception as exc:
        if quarantine.exists():
            # Deletion did not complete; leave the quarantine and its
            # truthful pending audit in place for a later retry.
            try:
                pending_quarantine_bytes = json_bytes(quarantine_audit)
                write_atomic(audit_path, pending_quarantine_bytes, replace=True)
                if audit_path.read_bytes() != pending_quarantine_bytes:
                    raise RetentionError("quarantine audit restoration was not verified")
            except Exception as pending_exc:
                raise RetentionError(
                    "quarantine deletion failed and pending audit could not be restored"
                ) from pending_exc
            raise exc
        # The quarantine has been consumed. Recreate the raw capture from the
        # one in-memory snapshot before restoring the exact pending audit;
        # never leave an audit pointing at a missing quarantine.
        raw_restored = False
        try:
            restore_raw_snapshot(raw_capture_path, raw)
            if not raw_capture_path.is_file() or raw_capture_path.read_bytes() != raw:
                raise RetentionError("raw snapshot restoration was not byte-exact")
            raw_restored = True
            write_atomic(audit_path, prior_audit_bytes, replace=True)
            restored_audit = audit_path.read_bytes()
            if restored_audit != prior_audit_bytes or json_load(restored_audit, "audit") != json_load(
                prior_audit_bytes, "audit"
            ):
                raise RetentionError("pending audit restoration was not byte-exact")
            reopen_payloads(artifact_dir, files, plan, source_sha, use_case, str(prior["attempt"]))
        except Exception as restore_exc:
            # Publish a distinct recovery state when possible, with no false
            # pending/deleted success claim.
            fatal = copy.deepcopy(quarantine_audit)
            fatal_status = "pending_audit_restore_failed" if raw_restored else "raw_restore_failed"
            fatal["raw_status"] = fatal_status
            fatal["raw_capture"]["quarantine"]["status"] = (
                "absent_pending_audit_restore_failed" if raw_restored else "absent_restore_failed"
            )
            fatal["recovery"] = {
                "status": "fatal",
                "snapshot": "restored" if raw_restored else "in_memory",
                "final_audit_error": type(exc).__name__,
                "restore_error": type(restore_exc).__name__,
            }
            try:
                write_atomic(audit_path, json_bytes(fatal), replace=True)
                if json_load(audit_path.read_bytes(), "audit") != fatal:
                    raise RetentionError("fatal recovery audit reopen verification failed")
            except Exception as fatal_audit_exc:
                raise RetentionError(
                    "final audit failed and raw snapshot restoration failed; fatal audit publication failed"
                ) from fatal_audit_exc
            raise RetentionError(f"final audit failed and recovery failed; raw_status={fatal_status}") from restore_exc
        raise RetentionError("final audit publication failed; raw snapshot and pending audit restored") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--retain", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--finalize-delete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--raw-capture", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--use-case", choices=USE_CASES, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.finalize_delete:
            result = finalize_delete(
                raw_capture_path=args.raw_capture,
                source_sha=args.source_sha,
                use_case=args.use_case,
                artifact_dir=args.artifact_dir,
                audit_path=args.audit,
                plan_path=args.plan,
                fixture_path=args.fixture,
                dry_run=args.dry_run,
            )
        else:
            result = retain_capture(
                raw_capture_path=args.raw_capture,
                source_sha=args.source_sha,
                use_case=args.use_case,
                artifact_dir=args.artifact_dir,
                audit_path=args.audit,
                plan_path=args.plan,
                fixture_path=args.fixture,
                retain=args.retain,
                validate_only=args.validate_only,
                dry_run=args.dry_run,
            )
    except Exception as exc:  # noqa: BLE001 - CLI must emit structured failures
        print(
            json.dumps(
                {
                    "schema_version": "m14-l04-retention-result-v1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
