"""Fail-closed validators for L04 dispatcher envelopes."""

from __future__ import annotations

import re
from typing import Any

from scripts._m14_l04_boundary import INTEGRATION_FACTORY
from scripts._m14_l04_digest import canonical_digest, source_map_digest
from scripts.m14_l04_contract import plan_digest

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
PENDING_STATUSES = {
    "not_implemented_pending_L04.4",
    "not_implemented_pending_L04.5",
    "not_implemented_pending_L04.6",
    "not_implemented_pending_L04.8",
    "not_implemented_pending_L04.9",
    "not_implemented_pending_L04.10",
}
INJECTED_STATUS = "injected_offline_non_eligible"
FAILED_STATUS = "failed"
TUNED_USE_CASE = "TunedLogitLens"
EXPECTED_STATUS = {
    "IntegratedGradients": "not_implemented_pending_L04.4",
    "TCAV": "not_implemented_pending_L04.5",
    "DirectLogitLens": "not_implemented_pending_L04.6",
    "TunedLogitLens": "blocked_missing_corpus",
    "Disentanglement": "not_implemented_pending_L04.8",
    "TrueActivationPatching": "not_implemented_pending_L04.9",
    "AdditiveSteering": "not_implemented_pending_L04.10",
}
RECORD_FOR_USE_CASE = {
    "TCAV": "THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018",
    "TunedLogitLens": "THY-T05-LOGIT-LENS-TUNED-LENS",
    "Disentanglement": "THY-T03-DISENTANGLEMENT",
    "TrueActivationPatching": "THY-T05-ACTIVATION-PATCHING",
    "AdditiveSteering": "THY-T05-STEERING-VECTORS-ZOU-ET-AL-2023-REPRESENTATION-ENGINEERING",
}


def _source_errors(value: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    file_map = value.get("implementation_source_files")
    if not isinstance(file_map, dict) or not file_map:
        return [f"{label} implementation source map is missing"]
    valid_keys = all(isinstance(name, str) and bool(name) for name in file_map)
    if not valid_keys:
        errors.append(f"{label} implementation source map has invalid keys")
    valid_hashes = all(
        isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) for digest in file_map.values()
    )
    if not valid_hashes:
        errors.append(f"{label} implementation source map has invalid hashes")
    if file_map.get("m14_l04_explanations.py") != value.get("runner_source_sha256"):
        errors.append(f"{label} runner source hash linkage is invalid")
    if file_map.get("m14_l04_contract.py") != value.get("contract_source_sha256"):
        errors.append(f"{label} contract source hash linkage is invalid")
    if valid_keys and valid_hashes and value.get("implementation_source_sha256") != source_map_digest(file_map):
        errors.append(f"{label} implementation source aggregate is invalid")
    for field in ("runner_source_sha256", "contract_source_sha256", "implementation_source_sha256"):
        if not isinstance(value.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", value[field]):
            errors.append(f"{label} {field} has invalid format")
    return errors


def validate_artifact(artifact: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(plan["artifact_schema"]["required_top_level"]) | {
        "accepted_record_ids",
        "accepted_gap_ids",
        "executions",
    }
    if not required.issubset(artifact):
        errors.append("artifact required fields are incomplete")
    if artifact.get("schema_version") != "m14-l04-explanations-artifact-v1" or artifact.get("lane") != "L04":
        errors.append("artifact identity is invalid")
    if artifact.get("plan_sha256") != plan_digest(plan):
        errors.append("artifact plan digest is invalid")
    if artifact.get("artifact_sha256") != canonical_digest(artifact, "artifact_sha256"):
        errors.append("artifact self-digest is invalid")
    if artifact.get("model") != plan.get("model"):
        errors.append("artifact model provenance is invalid")
    if artifact.get("integration") != "TransformerLMIntegration" or artifact.get("adapter") != "N/A":
        errors.append("artifact integration boundary is invalid")
    expected_exec = plan["real_use_case_checklist"]
    expected_names = [case["use_case"] for case in expected_exec if isinstance(case, dict)]
    if artifact.get("use_case") not in expected_names:
        errors.append("artifact use-case is invalid")
    executions = artifact.get("executions")
    execution_names = (
        [entry.get("use_case") for entry in executions]
        if isinstance(executions, list) and all(isinstance(entry, dict) for entry in executions)
        else None
    )
    if not isinstance(executions, list) or execution_names != expected_names:
        errors.append("artifact execution mappings are invalid")
    if isinstance(executions, list):
        active_status = next(
            (
                entry.get("status")
                for entry in executions
                if isinstance(entry, dict) and entry.get("use_case") == artifact.get("use_case")
            ),
            None,
        )
        for entry, expected in zip(executions, expected_exec, strict=False):
            if not isinstance(entry, dict) or not isinstance(expected, dict):
                errors.append("artifact execution mappings are invalid")
                continue
            for field in ("record_id", "support_only", "model", "integration", "adapter"):
                if entry.get(field) != expected.get(field):
                    errors.append(f"artifact execution {field} mapping is invalid")
            if entry.get("evidence_eligible") is not False or entry.get("acceptance") is not False:
                errors.append("dispatcher artifact cannot contain eligible or accepted evidence")
            expected_status = _expected_status(expected, artifact.get("use_case"), active_status)
            allowed_status = (
                {EXPECTED_STATUS.get(artifact.get("use_case")), INJECTED_STATUS, FAILED_STATUS}
                if expected.get("use_case") == artifact.get("use_case")
                else {expected_status}
            )
            if entry.get("status") not in allowed_status:
                errors.append(f"artifact execution status for {entry.get('use_case')} is invalid")
    expected_records = [item["record_id"] for item in plan["record_order"]]
    records = artifact.get("records")
    record_ids = (
        [entry.get("record_id") for entry in records]
        if isinstance(records, list) and all(isinstance(entry, dict) for entry in records)
        else None
    )
    if not isinstance(records, list) or record_ids != expected_records:
        errors.append("artifact records do not match frozen order")
    if isinstance(records, list):
        for record in records:
            if (
                not isinstance(record, dict)
                or record.get("evidence_level") != "D0"
                or record.get("acceptance") is not False
            ):
                errors.append("dispatcher records cannot contain eligible or accepted evidence")
                break
        active_use_case = artifact.get("use_case")
        active_record = RECORD_FOR_USE_CASE.get(active_use_case)
        execution_list = executions if isinstance(executions, list) else []
        active_status = next(
            (
                entry.get("status")
                for entry in execution_list
                if isinstance(entry, dict) and entry.get("use_case") == active_use_case
            ),
            None,
        )
        for record in records:
            if not isinstance(record, dict):
                continue
            expected_status = (
                active_status
                if record.get("record_id") == active_record
                else (
                    "blocked_missing_corpus"
                    if record.get("record_id") == "THY-T05-LOGIT-LENS-TUNED-LENS"
                    else "not_run"
                )
            )
            if record.get("status") != expected_status:
                errors.append(f"artifact record status for {record.get('record_id')} is invalid")
    if (
        artifact.get("accepted_record_ids") != []
        or artifact.get("accepted_gap_ids") != []
        or artifact.get("evidence_level") != "D0"
    ):
        errors.append("dispatcher artifact must remain unpromoted")
    if artifact.get("controls") != {
        "thresholds_and_controls": plan.get("thresholds_and_controls"),
        "evaluation": "not_run_by_dispatcher",
    }:
        errors.append("artifact thresholds and controls are not the frozen declarations")
    split = artifact.get("split")
    if not isinstance(split, dict) or split.get("group_overlap") != 0:
        errors.append("artifact split group_overlap must be zero")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict) or not SHA1_RE.fullmatch(str(provenance.get("git_sha", ""))):
        errors.append("artifact committed code SHA is invalid")
    if isinstance(provenance, dict):
        if provenance.get("plan_sha256") != plan_digest(plan):
            errors.append("artifact provenance plan digest is invalid")
        plan_model = plan.get("model")
        if (
            not isinstance(plan_model, dict)
            or provenance.get("model_id") != plan_model.get("id")
            or provenance.get("model_revision") != plan_model.get("revision")
        ):
            errors.append("artifact model identity provenance is invalid")
        if provenance.get("integration") != "TransformerLMIntegration" or provenance.get("adapter") != "N/A":
            errors.append("artifact integration provenance is invalid")
        if provenance.get("evidence_origin") not in {"dispatcher-only-no-model", "dependency-injected-offline"}:
            errors.append("artifact evidence origin is invalid")
        if provenance.get("network") != "not attempted" or provenance.get("credentials") != "not used":
            errors.append("artifact resource provenance is invalid")
        if provenance.get("integration_factory") != INTEGRATION_FACTORY:
            errors.append("artifact integration factory identity is invalid")
        errors.extend(_source_errors(provenance, "artifact"))
    if not isinstance(provenance, dict) or artifact.get("use_case") != provenance.get("use_case"):
        errors.append("artifact use-case provenance linkage is invalid")
    return errors


def _expected_status(expected: dict[str, Any], active_use_case: object, active_status: object) -> str:
    use_case = expected.get("use_case")
    if use_case == active_use_case:
        return str(active_status)
    return "blocked_missing_corpus" if use_case == TUNED_USE_CASE else "not_run"


def validate_run_record(run: dict[str, Any], artifact: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if run.get("schema_version") != "m14-l04-explanations-run-v1" or run.get("lane") != "L04":
        errors.append("run record identity is invalid")
    if run.get("plan_sha256") != plan_digest(plan) or run.get("artifact_sha256") != artifact.get("artifact_sha256"):
        errors.append("run record linkage is invalid")
    if run.get("use_case") not in EXPECTED_STATUS or run.get("use_case") != artifact.get("use_case"):
        errors.append("run record use-case linkage is invalid")
    if run.get("model") != plan.get("model"):
        errors.append("run record model provenance is invalid")
    if run.get("status") != _active_status(artifact):
        errors.append("run record status linkage is invalid")
    artifact_name = run.get("artifact_name")
    partial_pattern = rf"^l04-explanations\.{re.escape(str(artifact.get('use_case')))}\.attempt\d+\.partial\.json$"
    if not isinstance(artifact_name, str) or re.fullmatch(partial_pattern, artifact_name) is None:
        errors.append("run record artifact filename is invalid")
    if (
        run.get("command")
        != f"uv run python -m scripts.m14_l04_explanations --run-real --use-case {run.get('use_case')}"
    ):
        errors.append("run record command is invalid")
    if run.get("run_record_sha256") != canonical_digest(run, "run_record_sha256"):
        errors.append("run record self-digest is invalid")
    if run.get("partial_artifact_written") is not True:
        errors.append("run record partial artifact field is invalid")
    if not SHA1_RE.fullmatch(str(run.get("code_sha", ""))):
        errors.append("run record code SHA is invalid")
    errors.extend(_source_errors(run, "run record"))
    return errors


def validate_failure(
    failure: dict[str, Any], plan: dict[str, Any], artifact: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    allowed = {"failed", "blocked_missing_corpus", "injected_offline_non_eligible", *PENDING_STATUSES}
    if failure.get("schema_version") != "m14-l04-explanations-failure-v1" or failure.get("lane") != "L04":
        errors.append("failure identity is invalid")
    if failure.get("status") not in allowed:
        errors.append("failure status is invalid")
    expected_failure_status = EXPECTED_STATUS.get(failure.get("use_case"))
    if failure.get("use_case") in EXPECTED_STATUS and failure.get("status") not in {
        expected_failure_status,
        INJECTED_STATUS,
        FAILED_STATUS,
    }:
        errors.append("failure status is not allowed for this use case")
    if failure.get("plan_sha256") != plan_digest(plan):
        errors.append("failure plan digest is invalid")
    if failure.get("failure_sha256") != canonical_digest(failure, "failure_sha256"):
        errors.append("failure self-digest is invalid")
    if failure.get("artifact_written") is not True:
        errors.append("failure artifact_written must be true after partial write")
    if not SHA1_RE.fullmatch(str(failure.get("code_sha", ""))):
        errors.append("failure code SHA is invalid")
    if failure.get("use_case") not in EXPECTED_STATUS or (
        artifact is not None and failure.get("use_case") != artifact.get("use_case")
    ):
        errors.append("failure use-case linkage is invalid")
    if failure.get("model") != plan.get("model"):
        errors.append("failure model provenance is invalid")
    if artifact is not None and failure.get("status") != _active_status(artifact):
        errors.append("failure status linkage is invalid")
    failure_pattern = rf"^l04-explanations\.{re.escape(str(failure.get('use_case')))}\.attempt\d+\.failure\.json$"
    if not isinstance(failure.get("failure_ref"), str) or re.fullmatch(failure_pattern, failure["failure_ref"]) is None:
        errors.append("failure filename reference is invalid")
    errors.extend(_source_errors(failure, "failure"))
    run_record = failure.get("run_record")
    if not isinstance(run_record, dict):
        errors.append("failure run record is missing")
    elif artifact is not None:
        if run_record.get("artifact_sha256") != artifact.get("artifact_sha256"):
            errors.append("failure/run artifact linkage is invalid")
        if run_record.get("status") != failure.get("status") or run_record.get("use_case") != failure.get("use_case"):
            errors.append("failure/run status linkage is invalid")
        errors.extend(validate_run_record(run_record, artifact, plan))
        artifact_executions = artifact.get("executions")
        if not isinstance(artifact_executions, list):
            artifact_executions = []
        active = next(
            (
                entry
                for entry in artifact_executions
                if isinstance(entry, dict) and entry.get("use_case") == artifact.get("use_case")
            ),
            None,
        )
        if not isinstance(active, dict) or active.get("failure_ref") != failure.get("failure_ref"):
            errors.append("failure/artifact reference linkage is invalid")
    if failure.get("status") in PENDING_STATUSES | {"blocked_missing_corpus", "injected_offline_non_eligible"} and (
        failure.get("exception") is not None or failure.get("exception_type") is not None
    ):
        errors.append("non-failed status cannot contain an exception")
    if failure.get("status") == "failed" and (
        failure.get("exception_type") is None or failure.get("exception") is None
    ):
        errors.append("failed status must retain the execution exception")
    if failure.get("stage") != ("execution" if failure.get("status") == FAILED_STATUS else "dispatch"):
        errors.append("failure stage is inconsistent with status")
    return errors


def _active_status(artifact: dict[str, Any]) -> object:
    executions = artifact.get("executions")
    if not isinstance(executions, list):
        return None
    for entry in executions:
        if isinstance(entry, dict) and entry.get("use_case") == artifact.get("use_case"):
            return entry.get("status")
    return None
