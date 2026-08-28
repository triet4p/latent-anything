"""Private run and failure envelopes for the L04 execution dispatcher."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from scripts._m14_l04_artifact import build_artifact, execution_template
from scripts._m14_l04_digest import canonical_digest, code_sha, runtime_versions, source_digests, source_map_digest
from scripts._m14_l04_io import safe_write
from scripts.m14_l04_contract import plan_digest

COMMAND = "uv run python -m scripts.m14_l04_explanations --run-real"


def build_run_record(
    plan: dict[str, Any],
    artifact: dict[str, Any],
    use_case: str,
    status: str,
    resources: dict[str, Any],
    *,
    artifact_name: str | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": "m14-l04-explanations-run-v1",
        "lane": "L04",
        "artifact_name": artifact_name or f"artifacts/m14/l04-explanations.{use_case}.partial.json",
        "use_case": use_case,
        "artifact_sha256": artifact["artifact_sha256"],
        "plan_sha256": plan_digest(plan),
        "code_sha": code_sha(),
        **source_digests(),
        "command": f"{COMMAND} --use-case {use_case}",
        "model": plan["model"],
        "environment": runtime_versions(),
        "device": resources.get("device", "not used"),
        "resource_peak": resources.get("resource_peak", "not measured"),
        "network": resources.get("network", "not attempted"),
        "credentials_redacted": True,
        "cleanup": resources.get("cleanup", "not applicable; no model was loaded"),
        "status": status,
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "partial_artifact_written": True,
        "accepted_record_ids": list(artifact.get("accepted_record_ids", [])),
        "accepted_gap_ids": list(artifact.get("accepted_gap_ids", [])),
    }
    result["run_record_sha256"] = canonical_digest(result, "run_record_sha256")
    return result


def failure_envelope(
    plan: dict[str, Any],
    use_case: str,
    status: str,
    *,
    error: BaseException | None = None,
    failure_ref: str | None = None,
    run_record: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observed = resources or {}
    result = {
        "schema_version": "m14-l04-explanations-failure-v1",
        "lane": "L04",
        "status": status,
        "use_case": use_case,
        "command": f"{COMMAND} --use-case {use_case}",
        "stage": "dispatch" if error is None else "execution",
        "exception_type": None if error is None else type(error).__name__,
        "exception": None if error is None else str(error),
        "stdout_stderr_sha256": None,
        "stdout_stderr_capture": "not applicable; in-process dispatcher",
        "code_sha": code_sha(),
        **source_digests(),
        "model": plan["model"],
        "plan_sha256": plan_digest(plan),
        "resource": {
            "device": observed.get("device", "not used"),
            "network": observed.get("network", "not attempted"),
            "credentials": "not used",
            "cleanup": observed.get("cleanup", "not applicable; no model was loaded"),
        },
        "network": observed.get("network", "not attempted"),
        "credentials_redacted": True,
        "cleanup": observed.get("cleanup", "not applicable; no model was loaded"),
        "blocker_owner": "explanation",
        "artifact_written": True,
        "failure_ref": failure_ref,
        "run_record": run_record,
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    result["failure_sha256"] = canonical_digest(result, "failure_sha256")
    return result


# Compatibility wrappers keep the old private import stable after validator split.
def validate_artifact(artifact: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    from scripts._m14_l04_validate import validate_artifact as _validate_artifact

    return _validate_artifact(artifact, plan)


def validate_run_record(run: dict[str, Any], artifact: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    from scripts._m14_l04_validate import validate_run_record as _validate_run_record

    return _validate_run_record(run, artifact, plan)


def validate_failure(
    failure: dict[str, Any], plan: dict[str, Any], artifact: dict[str, Any] | None = None
) -> list[str]:
    from scripts._m14_l04_validate import validate_failure as _validate_failure

    return _validate_failure(failure, plan, artifact)


__all__ = [
    "build_artifact",
    "build_run_record",
    "canonical_digest",
    "code_sha",
    "execution_template",
    "failure_envelope",
    "runtime_versions",
    "safe_write",
    "source_digests",
    "source_map_digest",
    "validate_artifact",
    "validate_failure",
    "validate_run_record",
]
