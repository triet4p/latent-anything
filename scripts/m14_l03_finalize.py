"""Side-effect-free finalization after the remote wrapper proves cleanup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts.m14_l03_envelope import (
    SHA1_RE,
    record_digest,
    source_digests,
    validate_artifact,
)
from scripts.m14_l03_plan import plan_digest


def build_report(plan: Mapping[str, Any], artifact: Mapping[str, Any], run_record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the preliminary JSON object captured before remote cleanup."""
    return {
        "schema_version": "m14-l03-report-v1",
        "status": "runner_completed_cleanup_pending",
        "plan_sha256": plan_digest(plan),
        "artifact": dict(artifact),
        "run_record": dict(run_record),
        "run_record_sha256": record_digest(run_record),
    }


def validate_report(report: Mapping[str, Any], plan: Mapping[str, Any]) -> list[str]:
    """Validate preliminary or finalized success report linkage."""
    errors: list[str] = []
    artifact = report.get("artifact")
    run_record = report.get("run_record")
    if report.get("schema_version") != "m14-l03-report-v1" or report.get("status") not in {
        "runner_completed_cleanup_pending",
        "success",
    }:
        errors.append("report identity/status is invalid")
    if not isinstance(artifact, Mapping) or not isinstance(run_record, Mapping):
        return errors + ["success report must contain artifact and run_record objects"]
    errors.extend(validate_artifact(artifact, plan, source_digests()))
    if report.get("plan_sha256") != plan_digest(plan):
        errors.append("report plan digest is invalid")
    if report.get("run_record_sha256") != record_digest(run_record):
        errors.append("report run-record digest is invalid")
    if run_record.get("artifact_sha256") != artifact.get("artifact_sha256"):
        errors.append("run record does not link to artifact digest")
    if run_record.get("plan_sha256") != plan_digest(plan):
        errors.append("run record plan digest is invalid")
    if run_record.get("status") not in {"runner_completed_cleanup_pending", "completed"} or not SHA1_RE.fullmatch(
        str(run_record.get("git_sha", ""))
    ):
        errors.append("run record status or git SHA is invalid")
    if (
        run_record.get("status") == "runner_completed_cleanup_pending"
        and run_record.get("cleanup_status") != "pending_external_wrapper"
    ):
        errors.append("preliminary run must keep cleanup pending")
    if run_record.get("status") == "completed" and run_record.get("cleanup_status") != "completed_external_wrapper":
        errors.append("final run must carry completed wrapper cleanup evidence")
    if run_record.get("accepted_record_ids") != artifact.get("accepted_record_ids"):
        errors.append("run record accepted IDs do not link to artifact")
    if run_record.get("git_sha") != artifact.get("provenance", {}).get("git_sha"):
        errors.append("run record git SHA does not link to artifact")
    current = source_digests()
    for field in ("runner_source_sha256", "contract_source_sha256", "implementation_source_sha256"):
        if run_record.get(field) != current.get(field):
            errors.append(f"run record {field} does not match current source")
    return errors


def finalize_run_record(preliminary: Mapping[str, Any], cleanup_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Finalize only after the remote wrapper proves isolated cleanup."""
    required = ("checkout_removed", "cache_removed", "output_captured")
    if any(cleanup_evidence.get(key) is not True for key in required):
        raise ValueError("cleanup evidence must affirm checkout, cache, and output handling")
    result = dict(preliminary)
    was_failure = str(preliminary.get("status")) == "runner_failed_cleanup_pending"
    result["status"] = "failed" if was_failure else "completed"
    result["cleanup_status"] = "completed_external_wrapper"
    result["cleanup"] = "external wrapper confirmed isolated checkout/cache/output cleanup"
    result["cleanup_evidence"] = dict(cleanup_evidence)
    result["run_record_sha256"] = record_digest(result)
    return result


def finalize_report(
    report: Mapping[str, Any], cleanup_evidence: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the final report without changing artifact metrics or IDs."""
    preliminary = report.get("run_record")
    artifact = report.get("artifact")
    if not isinstance(preliminary, Mapping):
        raise ValueError("preliminary report must contain a run_record")
    final_run = finalize_run_record(preliminary, cleanup_evidence)
    result = dict(report)
    result["run_record"] = final_run
    result["run_record_sha256"] = record_digest(final_run)
    if not isinstance(artifact, Mapping):
        if report.get("status") != "failed":
            raise ValueError("only failed preliminary reports may omit an artifact")
        result["status"] = "failed"
        return result
    result["status"] = "success"
    if validate_artifact(artifact, plan, source_digests()):
        raise ValueError("preliminary artifact became invalid during finalization")
    return result
