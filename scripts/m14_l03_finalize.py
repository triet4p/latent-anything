"""Side-effect-free finalization after the remote wrapper proves cleanup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts.m14_l03_envelope import (
    SHA1_RE,
    SHA256_RE,
    record_digest,
    source_digests,
    validate_artifact,
)
from scripts.m14_l03_plan import plan_digest

_CLEANUP_BOOL_FIELDS = ("checkout_removed", "cache_removed", "output_captured")
_CLEANUP_PATH_PREFIXES = ("/tmp/", "/var/tmp/")


def validate_cleanup_evidence(evidence: Any) -> list[str]:
    """Validate narrowly scoped cleanup proof from the external wrapper."""
    errors: list[str] = []
    if not isinstance(evidence, Mapping):
        return ["cleanup evidence must be an object"]
    for field in _CLEANUP_BOOL_FIELDS:
        if evidence.get(field) is not True:
            errors.append(f"cleanup evidence {field} must be true")
    exit_status = evidence.get("wrapper_exit_status")
    if isinstance(exit_status, bool) or not isinstance(exit_status, int) or exit_status != 0:
        errors.append("cleanup evidence wrapper_exit_status must be integer zero")
    checkout = evidence.get("isolated_checkout_path")
    if (
        not isinstance(checkout, str)
        or not checkout.startswith(_CLEANUP_PATH_PREFIXES)
        or checkout.rstrip("/") in {"/tmp", "/var/tmp"}
    ):
        errors.append("cleanup evidence must identify a scoped isolated checkout path")
    caches = evidence.get("isolated_cache_paths")
    if (
        not isinstance(caches, list)
        or not caches
        or any(
            not isinstance(path, str)
            or not path.startswith(_CLEANUP_PATH_PREFIXES)
            or path.rstrip("/") in {"/tmp", "/var/tmp"}
            for path in caches
        )
    ):
        errors.append("cleanup evidence must identify scoped isolated cache paths")
    output_digest = evidence.get("captured_output_sha256")
    if not isinstance(output_digest, str) or not SHA256_RE.fullmatch(output_digest):
        errors.append("cleanup evidence must identify captured output by SHA-256")
    return errors


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
    run_status = run_record.get("status")
    if run_status not in {"runner_completed_cleanup_pending", "completed"} or not SHA1_RE.fullmatch(
        str(run_record.get("git_sha", ""))
    ):
        errors.append("run record status or git SHA is invalid")
    if (
        run_status == "runner_completed_cleanup_pending"
        and run_record.get("cleanup_status") != "pending_external_wrapper"
    ):
        errors.append("preliminary run must keep cleanup pending")
    if run_status == "completed":
        if run_record.get("cleanup_status") != "completed_external_wrapper":
            errors.append("final run must carry completed wrapper cleanup evidence")
        errors.extend(validate_cleanup_evidence(run_record.get("cleanup_evidence", {})))
    if report.get("status") == "runner_completed_cleanup_pending" and run_status != "runner_completed_cleanup_pending":
        errors.append("preliminary report must contain a preliminary run")
    if report.get("status") == "success" and run_status != "completed":
        errors.append("success report must contain a finalized run")
    if run_record.get("accepted_record_ids") != artifact.get("accepted_record_ids"):
        errors.append("run record accepted IDs do not link to artifact")
    if run_record.get("git_sha") != artifact.get("provenance", {}).get("git_sha"):
        errors.append("run record git SHA does not link to artifact")
    current = source_digests()
    for field in ("runner_source_sha256", "contract_source_sha256", "implementation_source_sha256"):
        if run_record.get(field) != current.get(field):
            errors.append(f"run record {field} does not match current source")
    if run_record.get("implementation_source_files") != current.get("implementation_source_files"):
        errors.append("run record implementation source files do not match current source")
    return errors


def finalize_run_record(preliminary: Mapping[str, Any], cleanup_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Finalize only after the remote wrapper proves isolated cleanup."""
    if preliminary.get("status") not in {"runner_completed_cleanup_pending", "runner_failed_cleanup_pending"}:
        raise ValueError("only a preliminary run may be finalized")
    if preliminary.get("cleanup_status") != "pending_external_wrapper":
        raise ValueError("preliminary run must keep cleanup pending")
    cleanup_errors = validate_cleanup_evidence(cleanup_evidence)
    if cleanup_errors:
        raise ValueError("; ".join(cleanup_errors))
    result = dict(preliminary)
    was_failure = str(preliminary.get("status")) == "runner_failed_cleanup_pending"
    result["status"] = "failed" if was_failure else "completed"
    result["cleanup_status"] = "completed_external_wrapper"
    result["cleanup"] = "external wrapper confirmed isolated checkout/cache/output cleanup"
    result["cleanup_evidence"] = dict(cleanup_evidence)
    result["run_record_sha256"] = record_digest(result)
    return result


def validate_failure_report(
    report: Mapping[str, Any], plan: Mapping[str, Any], *, finalized: bool = False
) -> list[str]:
    """Validate a failed report before or after external-wrapper cleanup."""
    errors: list[str] = []
    if report.get("schema_version") != "m14-l03-failure-v1" or report.get("status") != "failed":
        errors.append("failure report identity/status is invalid")
    if report.get("artifact_written") is not False:
        errors.append("failure report must state that no artifact was written")
    if report.get("artifact") is not None:
        errors.append("failure report must not contain an artifact")
    if report.get("plan_sha256") != plan_digest(plan):
        errors.append("failure report plan digest is invalid")
    if report.get("accepted_record_ids") != [] or report.get("accepted_gap_ids") != []:
        errors.append("failure report accepted IDs must be empty")
    if not SHA1_RE.fullmatch(str(report.get("git_sha", ""))):
        errors.append("failure report git SHA is invalid")
    if not isinstance(report.get("source_digests"), Mapping):
        errors.append("failure report source digests are missing")
    else:
        current = source_digests()
        if report["source_digests"] != current:
            errors.append("failure report source digests do not match current source")
    if not isinstance(report.get("model_attempt"), Mapping):
        errors.append("failure report model attempt is missing")
    else:
        model = plan.get("model", {})
        if report["model_attempt"].get("model_id") != model.get("model_id"):
            errors.append("failure report model ID does not match plan")
        if report["model_attempt"].get("revision") != model.get("revision"):
            errors.append("failure report model revision does not match plan")
    if not isinstance(report.get("runtime_versions"), Mapping):
        errors.append("failure report runtime versions are missing")
    if not isinstance(report.get("resources"), Mapping):
        errors.append("failure report resources are missing")
    if not isinstance(report.get("error_type"), str) or not report["error_type"]:
        errors.append("failure report error type is missing")
    if not isinstance(report.get("error"), str) or not report["error"]:
        errors.append("failure report error message is missing")
    expected_status = "failed" if finalized else "runner_failed_cleanup_pending"
    expected_cleanup = "completed_external_wrapper" if finalized else "pending_external_wrapper"
    if report.get("run_record_status") != "failed":
        errors.append("failure report run-record status is invalid")
    if report.get("cleanup_status") != expected_cleanup:
        errors.append("failure report cleanup status is invalid")
    run_record = report.get("run_record")
    if not isinstance(run_record, Mapping):
        return errors + ["failure report must contain a run_record"]
    if report.get("run_record_sha256") != record_digest(run_record):
        errors.append("failure report run-record digest is invalid")
    if run_record.get("schema_version") != "m14-l03-analysis-run-v1" or run_record.get("lane") != "M14-L03":
        errors.append("failure run record identity is invalid")
    if run_record.get("status") != expected_status:
        errors.append("failure run record status is invalid")
    if run_record.get("cleanup_status") != expected_cleanup:
        errors.append("failure run record cleanup status is invalid")
    if (
        run_record.get("artifact_sha256") is not None
        or run_record.get("accepted_record_ids") != []
        or run_record.get("accepted_gap_ids") != []
    ):
        errors.append("failure run record must have no artifact or accepted IDs")
    if run_record.get("plan_sha256") != plan_digest(plan) or run_record.get("git_sha") != report.get("git_sha"):
        errors.append("failure run record identity linkage is invalid")
    current = source_digests()
    for field in (
        "runner_source_sha256",
        "contract_source_sha256",
        "implementation_source_sha256",
        "implementation_source_files",
    ):
        if run_record.get(field) != current.get(field):
            errors.append(f"failure run record {field} does not match current source")
    if finalized:
        errors.extend(validate_cleanup_evidence(run_record.get("cleanup_evidence", {})))
    elif "cleanup_evidence" in run_record:
        errors.append("preliminary failure run must not claim cleanup evidence")
    return errors


def finalize_report(
    report: Mapping[str, Any], cleanup_evidence: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the final report without changing artifact metrics or IDs."""
    preliminary = report.get("run_record")
    artifact = report.get("artifact")
    if not isinstance(preliminary, Mapping):
        raise ValueError("preliminary report must contain a run_record")
    if not isinstance(artifact, Mapping):
        preliminary_errors = validate_failure_report(report, plan)
        if preliminary_errors:
            raise ValueError("invalid preliminary failure report: " + "; ".join(preliminary_errors))
    else:
        preliminary_errors = validate_report(report, plan)
        if preliminary_errors:
            raise ValueError("invalid preliminary success report: " + "; ".join(preliminary_errors))
    final_run = finalize_run_record(preliminary, cleanup_evidence)
    result = dict(report)
    result["run_record"] = final_run
    result["run_record_sha256"] = record_digest(final_run)
    if not isinstance(artifact, Mapping):
        result["status"] = "failed"
        result["run_record_status"] = final_run["status"]
        result["cleanup_status"] = final_run["cleanup_status"]
        final_errors = validate_failure_report(result, plan, finalized=True)
        if final_errors:
            raise ValueError("invalid finalized failure report: " + "; ".join(final_errors))
        return result
    result["status"] = "success"
    if validate_artifact(artifact, plan, source_digests()):
        raise ValueError("preliminary artifact became invalid during finalization")
    final_errors = validate_report(result, plan)
    if final_errors:
        raise ValueError("invalid finalized success report: " + "; ".join(final_errors))
    return result
