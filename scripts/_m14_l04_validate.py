"""Fail-closed validators for L04 dispatcher envelopes."""

from __future__ import annotations

import re
from typing import Any

from scripts._m14_l04_boundary import INTEGRATION_FACTORY
from scripts._m14_l04_digest import canonical_digest, source_map_digest
from scripts._m14_l04_validate_activation_patching import validate_real_true_activation_patching_execution
from scripts._m14_l04_validate_direct_lens import validate_real_direct_lens_execution
from scripts._m14_l04_validate_disentanglement import validate_real_disentanglement_execution
from scripts._m14_l04_validate_ig import validate_real_ig_execution
from scripts._m14_l04_validate_tcav import validate_real_tcav_execution
from scripts._m14_l04_validate_tuned_lens import validate_real_tuned_lens_execution
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
EXECUTION_STAGES = {
    "dispatch",
    "execution",
    "preflight",
    "dependency_check",
    "cuda_check",
    "model_load",
    "scoring",
    "cleanup",
    "complete",
}
PRE_CUDA_STAGES = {"dispatch", "preflight", "dependency_check"}
CUDA_STAGE_ORDER = (
    "cuda_check",
    "model_load",
    "scoring",
    "cleanup",
    "complete",
)
DEVICE_PLACEHOLDERS = {"", "not used", "not attempted", "cpu", "cuda"}
SAFE_ADDITIVE_CUDA_DEVICE_RE = re.compile(r"^cuda(?::(?:0|[1-9][0-9]*))?$")
REAL_IG_STATUS = "passed_real_cuda"
REAL_DIRECT_LENS_STATUS = "passed_real_cuda"
REAL_TUNED_LENS_STATUS = "passed_real_cuda"
ADDITIVE_COMPLETED_STATUS = "completed_real_cuda_d0"
ADDITIVE_USE_CASE = "AdditiveSteering"
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
GAP_FOR_USE_CASE = {
    "TCAV": "THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018",
}
TCAV_ACCEPTED_RECORD_ID = "t05_tcav"
DISENTANGLEMENT_ACCEPTED_RECORD_ID = "THY-T03-DISENTANGLEMENT"
ACTIVATION_PATCHING_ACCEPTED_RECORD_ID = "THY-T05-ACTIVATION-PATCHING"


def _validate_additive_d0(artifact: dict[str, Any], active: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    """Validate the Phase-A additive contract independently at the envelope boundary."""
    errors: list[str] = []
    try:
        from scripts._m14_l04_artifact import sanitize_additive_resources, sanitize_additive_result

        result_fields = {
            "status",
            "evidence_eligible",
            "acceptance",
            "evidence_level",
            "semantic_candidate",
            "criteria",
            "failure_reason",
            "metrics",
            "controls",
            "raw_summaries",
            "holdout_evidence",
            "token_ids",
            "target_token_strings",
            "layer",
            "native_hidden_state_index",
            "seeds",
            "strength_grid",
            "train_groups",
            "holdout_groups",
            "direction_norm",
            "no_mutation",
            "budget_pass",
            "model_parameter_digest_before",
            "model_parameter_digest_after",
            "provenance",
            "resources",
        }
        from scripts._m14_l04_artifact import ADDITIVE_EXECUTION_BASE_FIELDS, ADDITIVE_LINKED_FIELDS

        status = active.get("status")
        expected_active_fields = set(ADDITIVE_EXECUTION_BASE_FIELDS) | {"resources"}
        if status in {ADDITIVE_COMPLETED_STATUS, FAILED_STATUS} and bool(active.get("metrics")):
            expected_active_fields |= set(ADDITIVE_LINKED_FIELDS)
        if set(active) != expected_active_fields:
            errors.append("additive execution contains unexpected fields")
        if (
            active.get("use_case") != ADDITIVE_USE_CASE
            or active.get("evidence_level") != "D0"
            or active.get("evidence_eligible") is not False
            or active.get("acceptance") is not False
        ):
            errors.append("additive execution must remain exactly D0 and non-accepted")
        if status not in {
            ADDITIVE_COMPLETED_STATUS,
            FAILED_STATUS,
            INJECTED_STATUS,
            EXPECTED_STATUS[ADDITIVE_USE_CASE],
        }:
            errors.append("additive execution status is invalid")
        resources = active.get("resources", artifact.get("resources"))
        if status in {ADDITIVE_COMPLETED_STATUS, FAILED_STATUS} and bool(active.get("metrics")):
            if not isinstance(resources, dict):
                raise ValueError("additive resource schema is missing")
            sanitize_additive_resources(resources)
            if not isinstance(active.get("provenance"), dict):
                errors.append("additive execution provenance is missing")
            else:
                dynamic = active["provenance"]
                result = {key: active.get(key, artifact.get(key)) for key in result_fields}
                result["status"] = status
                result["provenance"] = dynamic
                result["resources"] = resources
                result["evidence_eligible"] = active.get("evidence_eligible")
                result["acceptance"] = active.get("acceptance")
                result["evidence_level"] = active.get("evidence_level")
                sanitize_additive_result(result, resources, plan)
                for field in (*sorted(ADDITIVE_LINKED_FIELDS - {"provenance", "resources"}),):
                    artifact_field = artifact.get(field)
                    if field == "controls" and isinstance(artifact_field, dict):
                        artifact_field = artifact_field.get("additive")
                    if artifact_field != active.get(field):
                        errors.append(f"additive artifact/{field} linkage is invalid")
                if artifact.get("resources") != resources:
                    errors.append("additive artifact/resources linkage is invalid")
                artifact_provenance = artifact.get("provenance")
                if not isinstance(artifact_provenance, dict):
                    errors.append("additive artifact provenance linkage is invalid")
                else:
                    # The artifact adds generic dispatcher provenance around
                    # the handler provenance.  Every retained handler field
                    # must nevertheless be byte-for-byte identical in all
                    # three locations (artifact, execution, and record).
                    additive_provenance_fields = {
                        "runtime",
                        "model_revision",
                        "target_token_ids",
                        "target_token_strings",
                        "target_position",
                        "direction_fit",
                        "network",
                        "device",
                        "execution_attempted",
                        "execution_backend",
                        "stage",
                        "deterministic_algorithms",
                        "runtime_versions",
                        "resource_peak",
                        "budget_pass",
                        "cleanup",
                        "cleanup_complete",
                        "model_parameter_digest_before",
                        "model_parameter_digest_after",
                        "model_parameter_digest_algorithm",
                        "bootstrap_replicates",
                        "aggregation_unit",
                        "off_target_aggregation",
                        "use_case",
                        "shuffled_label_policy",
                        "shuffled_label_cardinality",
                        "shuffled_label_identity_assignment",
                        "execution_result_digest",
                    }
                    for field in additive_provenance_fields:
                        if artifact_provenance.get(field) != active["provenance"].get(field):
                            errors.append(f"additive artifact provenance/{field} linkage is invalid")
        elif status == FAILED_STATUS and not isinstance(active.get("resources"), dict):
            errors.append("additive pre-CUDA failure resources are missing")
        elif isinstance(active.get("resources"), dict):
            # Pre-CUDA failures intentionally have no completed-result fields,
            # but their resource tuple is still a strict retained envelope.
            sanitize_additive_resources(active["resources"])
        if isinstance(active.get("resources"), dict) and artifact.get("resources") != active.get("resources"):
            errors.append("additive execution/resources linkage is invalid")
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"additive D0 schema is invalid: {type(exc).__name__}")
    return errors


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


def _execution_tuple_errors(value: dict[str, Any], label: str) -> list[str]:
    """Validate stage/backend/device/network as one fail-closed execution tuple."""
    errors: list[str] = []
    attempted = value.get("execution_attempted")
    backend = value.get("execution_backend")
    stage = value.get("stage")
    network = value.get("network")
    device = value.get("device")
    if stage is None:
        # Preserve validation of historical D0/D3 artifacts that predate the
        # stage field; all newly constructed envelopes always include it.
        return []
    if not isinstance(attempted, bool) or backend not in {"cuda", "none"} or stage not in EXECUTION_STAGES:
        return [f"{label} execution stage/backend tuple is invalid"]
    if attempted is not (backend == "cuda"):
        errors.append(f"{label} execution attempt/backend provenance is incoherent")
    if value.get("use_case") == ADDITIVE_USE_CASE:
        # ``cuda`` is the canonical generic attempted-device marker for the
        # additive fallback envelope, so it is concrete in this contract.
        is_placeholder = device is None or (isinstance(device, str) and device in DEVICE_PLACEHOLDERS - {"cuda"})
    else:
        is_placeholder = device is None or (isinstance(device, str) and device in DEVICE_PLACEHOLDERS)
    if stage in PRE_CUDA_STAGES or (stage == "cuda_check" and attempted is False):
        if attempted or backend != "none" or not is_placeholder or network != "not attempted":
            errors.append(f"{label} pre-CUDA execution tuple is invalid")
        resource_peak = value.get("resource_peak")
        if resource_peak is not None and resource_peak != "not measured":
            errors.append(f"{label} resource peak was recorded before CUDA execution")
        return errors
    if stage in CUDA_STAGE_ORDER and attempted is not True:
        errors.append(f"{label} post-CUDA stage must record an attempted CUDA backend")
    if attempted is True:
        if is_placeholder or not isinstance(device, str):
            errors.append(f"{label} attempted CUDA execution is missing a concrete device")
        elif value.get("use_case") == ADDITIVE_USE_CASE and SAFE_ADDITIVE_CUDA_DEVICE_RE.fullmatch(device) is None:
            errors.append(f"{label} additive CUDA device provenance is not canonical")
        if network != "enabled":
            errors.append(f"{label} attempted CUDA execution must have enabled network state")
    return errors


def _additive_resource_tuple(value: dict[str, Any]) -> tuple[object, ...]:
    """Return the exact resource/status linkage tuple for additive envelopes."""
    return tuple(
        value.get(field)
        for field in (
            "device",
            "network",
            "execution_backend",
            "stage",
            "execution_attempted",
            "cleanup",
            "resource_peak",
        )
    )


def _additive_tuple_link_errors(source: dict[str, Any], candidate: dict[str, Any], label: str) -> list[str]:
    if _additive_resource_tuple(source) != _additive_resource_tuple(candidate):
        return [f"additive {label} resource tuple linkage is invalid"]
    return []


def validate_artifact(artifact: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    is_additive_artifact = artifact.get("use_case") == ADDITIVE_USE_CASE
    if is_additive_artifact:
        try:
            from scripts._m14_l04_artifact import (
                ADDITIVE_ARTIFACT_BASE_FIELDS,
                ADDITIVE_LINKED_FIELDS,
                reject_additive_sensitive_keys,
                validate_fixture_metadata,
            )

            active = next(
                (
                    item
                    for item in artifact.get("executions", [])
                    if isinstance(item, dict) and item.get("use_case") == ADDITIVE_USE_CASE
                ),
                {},
            )
            # A technically completed additive run is only valid when the
            # complete sanitized result contract is present.  Failed and
            # pre-CUDA envelopes intentionally keep their smaller schemas.
            full = active.get("status") == ADDITIVE_COMPLETED_STATUS or bool(active.get("metrics"))
            expected_keys = set(ADDITIVE_ARTIFACT_BASE_FIELDS) | {"resources"}
            if full:
                expected_keys |= set(ADDITIVE_LINKED_FIELDS)
            if set(artifact) != expected_keys:
                errors.append("additive artifact top-level allowlist is invalid")
            reject_additive_sensitive_keys(artifact)
            fixture = artifact.get("fixture")
            try:
                validate_fixture_metadata(fixture)
            except (TypeError, ValueError, KeyError):
                errors.append("additive fixture metadata is not authored canonical data")
            if isinstance(fixture, dict) and (
                fixture.get("path") != plan.get("fixture", {}).get("path") or fixture.get("rows") != 24
            ):
                errors.append("additive fixture metadata linkage is invalid")
            if artifact.get("tokenization") != plan.get("tokenization_and_sampling"):
                errors.append("additive tokenization linkage is invalid")
            split = artifact.get("split")
            expected_split = {
                "train_groups": plan.get("fixture", {}).get("split", {}).get("train_groups"),
                "holdout_groups": plan.get("fixture", {}).get("split", {}).get("holdout_groups"),
                "group_overlap": 0,
            }
            if split != expected_split:
                errors.append("additive split linkage is invalid")
            provenance = artifact.get("provenance")
            generic_provenance_fields = {
                "runner_source_sha256",
                "contract_source_sha256",
                "implementation_source_sha256",
                "implementation_source_files",
                "git_sha",
                "model_id",
                "model_revision",
                "integration",
                "integration_factory",
                "adapter",
                "evidence_origin",
                "network",
                "device",
                "credentials",
                "cleanup",
                "execution_attempted",
                "execution_backend",
                "stage",
                "resource_peak",
                "use_case",
                "plan_sha256",
            }
            from scripts._m14_l04_artifact import ADDITIVE_PROVENANCE_FIELDS

            allowed_provenance = generic_provenance_fields | set(ADDITIVE_PROVENANCE_FIELDS)
            expected_provenance = allowed_provenance if full else generic_provenance_fields
            if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
                errors.append("additive artifact provenance allowlist is invalid")
        except (TypeError, ValueError, KeyError):
            errors.append("additive artifact schema cannot be inspected")
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
    try:
        artifact_digest_valid = artifact.get("artifact_sha256") == canonical_digest(artifact, "artifact_sha256")
    except (TypeError, ValueError):
        artifact_digest_valid = False
    if not artifact_digest_valid:
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
            if entry.get("use_case") == ADDITIVE_USE_CASE and artifact.get("use_case") == ADDITIVE_USE_CASE:
                errors.extend(_validate_additive_d0(artifact, entry, plan))
                expected_status = _expected_status(expected, artifact.get("use_case"), active_status)
                allowed_status = {
                    expected_status,
                    INJECTED_STATUS,
                    FAILED_STATUS,
                    ADDITIVE_COMPLETED_STATUS,
                }
                if entry.get("status") not in allowed_status:
                    errors.append("additive execution status is invalid")
                continue
            if entry.get("status") == REAL_IG_STATUS and entry.get("use_case") == "IntegratedGradients":
                if entry.get("evidence_eligible") is not True:
                    errors.append("real Integrated Gradients execution must be evidence-eligible")
            elif entry.get("status") == REAL_IG_STATUS and entry.get("use_case") == "TCAV":
                errors.extend(validate_real_tcav_execution(entry, artifact, plan))
            elif entry.get("status") == REAL_DIRECT_LENS_STATUS and entry.get("use_case") == "DirectLogitLens":
                errors.extend(validate_real_direct_lens_execution(entry, artifact, plan))
            elif entry.get("status") == REAL_TUNED_LENS_STATUS and entry.get("use_case") == "TunedLogitLens":
                errors.extend(validate_real_tuned_lens_execution(entry, artifact, plan))
            elif entry.get("status") == REAL_TUNED_LENS_STATUS and entry.get("use_case") == "Disentanglement":
                errors.extend(validate_real_disentanglement_execution(entry, artifact, plan))
            elif entry.get("status") == REAL_TUNED_LENS_STATUS and entry.get("use_case") == "TrueActivationPatching":
                errors.extend(validate_real_true_activation_patching_execution(entry, artifact, plan))
            elif entry.get("evidence_eligible") is not False or entry.get("acceptance") is not False:
                errors.append("dispatcher artifact cannot contain eligible or accepted evidence")
            expected_status = _expected_status(expected, artifact.get("use_case"), active_status)
            active_name = artifact.get("use_case")
            active_key = active_name if isinstance(active_name, str) else ""
            allowed_status = (
                {
                    EXPECTED_STATUS.get(active_key),
                    INJECTED_STATUS,
                    FAILED_STATUS,
                    REAL_IG_STATUS,
                    REAL_TUNED_LENS_STATUS,
                }
                if expected.get("use_case") == artifact.get("use_case")
                else {expected_status}
            )
            if entry.get("status") not in allowed_status:
                errors.append(f"artifact execution status for {entry.get('use_case')} is invalid")
            if entry.get("status") == REAL_IG_STATUS and entry.get("use_case") == "IntegratedGradients":
                errors.extend(validate_real_ig_execution(entry, artifact, plan))
    expected_records = [item["record_id"] for item in plan["record_order"]]
    records = artifact.get("records")
    record_ids = (
        [entry.get("record_id") for entry in records]
        if isinstance(records, list) and all(isinstance(entry, dict) for entry in records)
        else None
    )
    if not isinstance(records, list) or record_ids != expected_records:
        errors.append("artifact records do not match frozen order")
    active_status = _active_status(artifact)
    active_execution = next(
        (
            entry
            for entry in artifact.get("executions", [])
            if isinstance(entry, dict) and entry.get("use_case") == artifact.get("use_case")
        ),
        None,
    )
    tcav_accepted = (
        artifact.get("use_case") == "TCAV"
        and active_status == REAL_IG_STATUS
        and isinstance(active_execution, dict)
        and active_execution.get("evidence_eligible") is True
        and active_execution.get("acceptance") is True
        and artifact.get("evidence_level") == "D3"
        and artifact.get("accepted_record_ids") == [TCAV_ACCEPTED_RECORD_ID]
        and artifact.get("accepted_gap_ids") == [GAP_FOR_USE_CASE["TCAV"]]
    )
    tuned_accepted = (
        artifact.get("use_case") == "TunedLogitLens"
        and active_status == REAL_TUNED_LENS_STATUS
        and isinstance(active_execution, dict)
        and active_execution.get("evidence_eligible") is True
        and active_execution.get("acceptance") is True
        and artifact.get("evidence_level") == "D3"
        and artifact.get("accepted_record_ids") == ["THY-T05-LOGIT-LENS-TUNED-LENS"]
        and artifact.get("accepted_gap_ids") == ["THY-T05-LOGIT-LENS-TUNED-LENS"]
    )
    disentanglement_accepted = (
        artifact.get("use_case") == "Disentanglement"
        and active_status == REAL_TUNED_LENS_STATUS
        and isinstance(active_execution, dict)
        and active_execution.get("evidence_eligible") is True
        and active_execution.get("acceptance") is True
        and artifact.get("evidence_level") == "D2"
        and artifact.get("accepted_record_ids") == [DISENTANGLEMENT_ACCEPTED_RECORD_ID]
        and artifact.get("accepted_gap_ids") == [DISENTANGLEMENT_ACCEPTED_RECORD_ID]
    )
    activation_patching_accepted = (
        artifact.get("use_case") == "TrueActivationPatching"
        and active_status == REAL_TUNED_LENS_STATUS
        and isinstance(active_execution, dict)
        and active_execution.get("evidence_eligible") is True
        and active_execution.get("acceptance") is True
        and artifact.get("evidence_level") == "D3"
        and artifact.get("accepted_record_ids") == [ACTIVATION_PATCHING_ACCEPTED_RECORD_ID]
        and artifact.get("accepted_gap_ids") == [ACTIVATION_PATCHING_ACCEPTED_RECORD_ID]
    )
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict) or (
                not (
                    (
                        tcav_accepted
                        and record.get("record_id") == RECORD_FOR_USE_CASE.get(str(artifact.get("use_case")))
                    )
                    or (
                        tuned_accepted
                        and record.get("record_id") == RECORD_FOR_USE_CASE.get(str(artifact.get("use_case")))
                    )
                    or (
                        disentanglement_accepted
                        and record.get("record_id") == RECORD_FOR_USE_CASE.get(str(artifact.get("use_case")))
                    )
                    or (
                        activation_patching_accepted
                        and record.get("record_id") == RECORD_FOR_USE_CASE.get(str(artifact.get("use_case")))
                    )
                )
                and (record.get("evidence_level") != "D0" or record.get("acceptance") is not False)
            ):
                errors.append("dispatcher records cannot contain eligible or accepted evidence")
                break
        active_use_case = artifact.get("use_case")
        active_key = active_use_case if isinstance(active_use_case, str) else ""
        active_record = RECORD_FOR_USE_CASE.get(active_key)
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
        if is_additive_artifact and isinstance(active_execution, dict):
            active_record = next(
                (
                    record
                    for record in records
                    if isinstance(record, dict)
                    and record.get("record_id") == RECORD_FOR_USE_CASE.get(ADDITIVE_USE_CASE)
                ),
                None,
            )
            if isinstance(active_record, dict):
                from scripts._m14_l04_artifact import (
                    ADDITIVE_LINKED_FIELDS,
                    ADDITIVE_RECORD_BASE_FIELDS,
                )

                expected_record_fields = set(ADDITIVE_RECORD_BASE_FIELDS) | {"resources"}
                if active_execution.get("status") == ADDITIVE_COMPLETED_STATUS or active_execution.get("metrics"):
                    expected_record_fields |= set(ADDITIVE_LINKED_FIELDS)
                if set(active_record) != expected_record_fields:
                    errors.append("additive record allowlist is invalid")
                if active_execution.get("status") == ADDITIVE_COMPLETED_STATUS or (
                    active_execution.get("status") == FAILED_STATUS and bool(active_execution.get("metrics"))
                ):
                    for field in ADDITIVE_LINKED_FIELDS:
                        if active_record.get(field) != active_execution.get(field):
                            errors.append(f"additive execution/record {field} linkage is invalid")
    if tcav_accepted:
        active_record = next(
            (r for r in artifact.get("records", []) if r.get("record_id") == RECORD_FOR_USE_CASE["TCAV"]), {}
        )
        if active_record.get("evidence_level") != "D3" or active_record.get("acceptance") is not True:
            errors.append("accepted TCAV record must be D3 and accepted")
    elif tuned_accepted:
        active_record = next(
            (r for r in artifact.get("records", []) if r.get("record_id") == RECORD_FOR_USE_CASE["TunedLogitLens"]), {}
        )
        if active_record.get("evidence_level") != "D3" or active_record.get("acceptance") is not True:
            errors.append("accepted tuned lens record must be D3 and accepted")
    elif disentanglement_accepted:
        active_record = next(
            (r for r in artifact.get("records", []) if r.get("record_id") == RECORD_FOR_USE_CASE["Disentanglement"]), {}
        )
        if active_record.get("evidence_level") != "D2" or active_record.get("acceptance") is not True:
            errors.append("accepted disentanglement record must be D2 and accepted")
    elif activation_patching_accepted:
        active_record = next(
            (
                r
                for r in artifact.get("records", [])
                if r.get("record_id") == RECORD_FOR_USE_CASE["TrueActivationPatching"]
            ),
            {},
        )
        if active_record.get("evidence_level") != "D3" or active_record.get("acceptance") is not True:
            errors.append("accepted activation patching record must be D3 and accepted")
    elif (
        artifact.get("accepted_record_ids") != []
        or artifact.get("accepted_gap_ids") != []
        or artifact.get("evidence_level") != "D0"
    ):
        errors.append("dispatcher artifact must remain unpromoted")
    expected_dispatcher_controls = {
        "thresholds_and_controls": plan.get("thresholds_and_controls"),
        "evaluation": "not_run_by_dispatcher",
    }
    if is_additive_artifact and isinstance(active_execution, dict) and bool(active_execution.get("metrics")):
        active_controls = active_execution.get("controls")
        expected_dispatcher_controls["additive"] = active_controls
    if artifact.get("controls") != expected_dispatcher_controls:
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
        if provenance.get("evidence_origin") not in {
            "dispatcher-only-no-model",
            "dependency-injected-offline",
            "real-cuda",
        }:
            errors.append("artifact evidence origin is invalid")
        execution_attempted = provenance.get("execution_attempted")
        execution_backend = provenance.get("execution_backend")
        stage = provenance.get("stage")
        if not isinstance(execution_attempted, bool) or execution_backend not in {"cuda", "none"}:
            errors.append("artifact execution attempt/backend provenance is invalid")
        elif execution_attempted is not (execution_backend == "cuda"):
            errors.append("artifact execution attempt/backend provenance is incoherent")
        if stage is not None and stage not in EXECUTION_STAGES:
            errors.append("artifact execution stage provenance is invalid")
        is_real_ig = (
            artifact.get("use_case") == "IntegratedGradients" and provenance.get("evidence_origin") == "real-cuda"
        )
        is_real_tcav = artifact.get("use_case") == "TCAV" and provenance.get("evidence_origin") == "real-cuda"
        is_real_direct = (
            artifact.get("use_case") == "DirectLogitLens" and provenance.get("evidence_origin") == "real-cuda"
        )
        is_real_tuned = (
            artifact.get("use_case") == "TunedLogitLens" and provenance.get("evidence_origin") == "real-cuda"
        )
        is_real_disentanglement = (
            artifact.get("use_case") == "Disentanglement" and provenance.get("evidence_origin") == "real-cuda"
        )
        is_real_activation_patching = (
            artifact.get("use_case") == "TrueActivationPatching" and provenance.get("evidence_origin") == "real-cuda"
        )
        is_real_steering = (
            artifact.get("use_case") == "AdditiveSteering" and provenance.get("evidence_origin") == "real-cuda"
        )
        if provenance.get("evidence_origin") == "real-cuda" and (
            execution_attempted is not True or execution_backend != "cuda"
        ):
            errors.append("real CUDA artifact is missing truthful execution attempt/backend provenance")
        if is_real_ig:
            active_status = _active_status(artifact)
            allowed_networks = {"enabled"} if active_status == REAL_IG_STATUS else {"enabled", "not attempted"}
            if provenance.get("network") not in allowed_networks:
                errors.append("real Integrated Gradients runtime provenance is invalid")
            if provenance.get("device") in {None, "", "not used"}:
                errors.append("real Integrated Gradients CUDA device provenance is missing")
            if active_status == REAL_IG_STATUS:
                if provenance.get("deterministic_algorithms") is not True:
                    errors.append("real Integrated Gradients deterministic setting is missing")
                if not isinstance(provenance.get("runtime_versions"), dict):
                    errors.append("real Integrated Gradients runtime tool versions are missing")
                peak = provenance.get("resource_peak")
                if not isinstance(peak, dict) or not isinstance(peak.get("max_memory_allocated_bytes"), int):
                    errors.append("real Integrated Gradients CUDA peak resource is missing")
        elif is_real_tcav:
            if provenance.get("network") != "enabled":
                errors.append("real TCAV runtime provenance is invalid")
        elif is_real_direct:
            if provenance.get("network") != "enabled":
                errors.append("real direct lens runtime provenance is invalid")
        elif is_real_tuned:
            active_status = _active_status(artifact)
            allowed_networks = {"enabled"} if active_status == REAL_TUNED_LENS_STATUS else {"enabled", "not attempted"}
            if provenance.get("network") not in allowed_networks:
                errors.append("real tuned lens runtime provenance is invalid")
        elif is_real_disentanglement:
            if provenance.get("network") != "enabled":
                errors.append("real disentanglement runtime provenance is invalid")
            if provenance.get("device") in {None, "", "not used"}:
                errors.append("real disentanglement CUDA device provenance is missing")
            if active_status == REAL_TUNED_LENS_STATUS:
                peak = provenance.get("resource_peak")
                if not isinstance(peak, dict) or not isinstance(peak.get("max_memory_allocated_bytes"), int):
                    errors.append("real disentanglement CUDA peak resource is missing")
        elif is_real_activation_patching:
            if provenance.get("network") != "enabled" or provenance.get("device") in {None, "", "not used"}:
                errors.append("real activation patching runtime provenance is invalid")
            if _active_status(artifact) == REAL_TUNED_LENS_STATUS:
                peak = provenance.get("resource_peak")
                if not isinstance(peak, dict) or not isinstance(peak.get("max_memory_allocated_bytes"), int):
                    errors.append("real activation patching CUDA peak resource is missing")
        elif is_real_steering:
            # Additive steering Phase A is a diagnostic runtime only.  Validate
            # its resource provenance while keeping the D0/non-eligible gate
            # above; promotion semantics belong to the later validator task.
            if provenance.get("network") != "enabled" or provenance.get("device") in {None, "", "not used"}:
                errors.append("real additive steering runtime provenance is invalid")
            if _active_status(artifact) in {ADDITIVE_COMPLETED_STATUS, REAL_TUNED_LENS_STATUS}:
                peak = provenance.get("resource_peak")
                if not isinstance(peak, dict) or not isinstance(peak.get("max_memory_allocated_bytes"), int):
                    errors.append("real additive steering CUDA peak resource is missing")
        elif provenance.get("evidence_origin") == "dependency-injected-offline":
            # Injected handlers are offline test seams, but may deliberately
            # model a post-CUDA partial failure. Keep that provenance instead
            # of rewriting it to preflight or treating it as a real claim.
            if (
                provenance.get("network") not in {"enabled", "not attempted"}
                or provenance.get("credentials") != "not used"
            ):
                errors.append("artifact resource provenance is invalid")
        elif provenance.get("network") != "not attempted" or provenance.get("credentials") != "not used":
            errors.append("artifact resource provenance is invalid")
        if provenance.get("integration_factory") != INTEGRATION_FACTORY:
            errors.append("artifact integration factory identity is invalid")
        errors.extend(_execution_tuple_errors(provenance, "artifact"))
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
    if run.get("accepted_record_ids", artifact.get("accepted_record_ids")) != artifact.get("accepted_record_ids"):
        errors.append("run record accepted record IDs do not link to artifact")
    if run.get("accepted_gap_ids", artifact.get("accepted_gap_ids")) != artifact.get("accepted_gap_ids"):
        errors.append("run record accepted gap IDs do not link to artifact")
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
    errors.extend(_execution_tuple_errors(run, "run record"))
    errors.extend(_source_errors(run, "run record"))
    if run.get("use_case") == ADDITIVE_USE_CASE:
        executions = artifact.get("executions")
        active = (
            next(
                (
                    entry
                    for entry in executions
                    if isinstance(entry, dict) and entry.get("use_case") == ADDITIVE_USE_CASE
                ),
                None,
            )
            if isinstance(executions, list)
            else None
        )
        if isinstance(active, dict) and isinstance(active.get("resources"), dict):
            errors.extend(_additive_tuple_link_errors(active["resources"], run, "run record"))
            if run.get("failure_stage") != active["resources"].get("failure_stage"):
                errors.append("additive run record failure stage linkage is invalid")
        if isinstance(active, dict) and active.get("status") == ADDITIVE_COMPLETED_STATUS:
            # Keep the run envelope from becoming an alternate validation
            # path around the complete additive result contract.
            errors.extend(_validate_additive_d0(artifact, active, plan))
    return errors


def validate_failure(
    failure: dict[str, Any], plan: dict[str, Any], artifact: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    allowed = {
        "failed",
        "blocked_missing_corpus",
        "injected_offline_non_eligible",
        REAL_IG_STATUS,
        ADDITIVE_COMPLETED_STATUS,
        *PENDING_STATUSES,
    }
    if failure.get("schema_version") != "m14-l04-explanations-failure-v1" or failure.get("lane") != "L04":
        errors.append("failure identity is invalid")
    if failure.get("status") not in allowed:
        errors.append("failure status is invalid")
    failure_use_case = failure.get("use_case")
    failure_key = failure_use_case if isinstance(failure_use_case, str) else ""
    expected_failure_status = EXPECTED_STATUS.get(failure_key)
    if failure_key in EXPECTED_STATUS and failure.get("status") not in {
        expected_failure_status,
        INJECTED_STATUS,
        FAILED_STATUS,
        REAL_IG_STATUS,
        ADDITIVE_COMPLETED_STATUS,
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
        if failure.get("use_case") == ADDITIVE_USE_CASE and isinstance(active, dict):
            additive_resources = active.get("resources")
            failure_resources = failure.get("resource")
            if isinstance(additive_resources, dict) and isinstance(failure_resources, dict):
                try:
                    from scripts._m14_l04_artifact import sanitize_additive_resources

                    sanitize_additive_resources(
                        {key: value for key, value in failure_resources.items() if key != "credentials"}
                    )
                except (TypeError, ValueError, KeyError):
                    errors.append("additive failure resource schema is invalid")
                errors.extend(_additive_tuple_link_errors(additive_resources, failure_resources, "failure"))
                if failure.get("failure_stage") != additive_resources.get("failure_stage"):
                    errors.append("additive failure stage linkage is invalid")
            if isinstance(additive_resources, dict):
                errors.extend(_additive_tuple_link_errors(additive_resources, run_record, "failure/run"))
            if active.get("status") == ADDITIVE_COMPLETED_STATUS:
                errors.extend(_validate_additive_d0(artifact, active, plan))
    if failure.get("status") in PENDING_STATUSES | {
        "blocked_missing_corpus",
        "injected_offline_non_eligible",
        REAL_IG_STATUS,
        ADDITIVE_COMPLETED_STATUS,
    } and (failure.get("exception") is not None or failure.get("exception_type") is not None):
        errors.append("non-failed status cannot contain an exception")
    if failure.get("status") == "failed" and (
        failure.get("exception_type") is None or failure.get("exception") is None
    ):
        errors.append("failed status must retain the execution exception")
    stage = failure.get("stage")
    if failure.get("status") == FAILED_STATUS:
        if stage not in EXECUTION_STAGES - {"complete"}:
            errors.append("failed execution stage is not a truthful partial stage")
    elif stage not in {"dispatch", "complete"}:
        errors.append("non-failed failure stage is inconsistent with status")
    resource = failure.get("resource")
    if not isinstance(resource, dict):
        errors.append("failure resource envelope is missing")
    else:
        tuple_resource = (
            {**resource, "use_case": ADDITIVE_USE_CASE} if failure.get("use_case") == ADDITIVE_USE_CASE else resource
        )
        errors.extend(_execution_tuple_errors(tuple_resource, "failure resource"))
        if resource.get("stage") is not None and resource.get("stage") != stage:
            errors.append("failure resource stage provenance is incoherent")
        if (
            failure.get("status") == FAILED_STATUS
            and stage == "dispatch"
            and resource.get("execution_attempted") is True
        ):
            errors.append("failed execution cannot claim dispatch stage after starting")
    return errors


def _active_status(artifact: dict[str, Any]) -> object:
    executions = artifact.get("executions")
    if not isinstance(executions, list):
        return None
    for entry in executions:
        if isinstance(entry, dict) and entry.get("use_case") == artifact.get("use_case"):
            return entry.get("status")
    return None
