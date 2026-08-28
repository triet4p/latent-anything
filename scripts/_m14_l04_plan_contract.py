"""Validation rules for the frozen M14 L04 explanation plan."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts._m14_l04_contract_common import (
    canonical_json_bytes,
    digest_bytes,
    exact_keys,
    mapping,
)

PLAN_PATH = Path(__file__).resolve().parents[1] / "artifacts/m14/l04-explanations.plan.json"
SCHEMA_VERSION = "m14-l04-explanations-plan-v1"

EXPECTED_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "lane",
        "status",
        "owner",
        "hash_contract",
        "facts",
        "inferences_and_limits",
        "record_order",
        "scc_and_blockers",
        "architecture_audit",
        "local_checks",
        "model",
        "fixture",
        "tokenization_and_sampling",
        "semantic_contracts",
        "tuned_lens_corpus",
        "real_use_case_checklist",
        "metric_formulas",
        "thresholds_and_controls",
        "resource_budget",
        "artifact_schema",
        "atomic_tasks",
        "remote_cuda_workflow",
        "graphify_checkpoints",
        "plan_sha256",
    }
)
EXPECTED_RECORD_IDS = (
    "THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018",
    "THY-T05-LOGIT-LENS-TUNED-LENS",
    "THY-T03-DISENTANGLEMENT",
    "THY-T05-ACTIVATION-PATCHING",
    "THY-T05-STEERING-VECTORS-ZOU-ET-AL-2023-REPRESENTATION-ENGINEERING",
)
EXPECTED_USE_CASES = (
    "IntegratedGradients",
    "TCAV",
    "DirectLogitLens",
    "TunedLogitLens",
    "Disentanglement",
    "TrueActivationPatching",
    "AdditiveSteering",
)


def plan_digest(plan: Mapping[str, Any]) -> str:
    """Compute the unsigned canonical plan digest without mutating ``plan``."""
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    return digest_bytes(canonical_json_bytes(unsigned))


def _validate_identity_hash_model(plan: Mapping[str, Any], errors: list[str]) -> None:
    """Validate plan identity, hash declaration, and model integration boundary."""
    exact_keys(plan, EXPECTED_PLAN_KEYS, "plan", errors)
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("lane") != "L04":
        errors.append("plan identity must be m14-l04-explanations-plan-v1/L04")
    if plan.get("status") != "design-frozen-pending-implementation":
        errors.append("plan must remain design-frozen-pending-implementation")

    hash_contract = mapping(plan.get("hash_contract"), "hash_contract", errors)
    if hash_contract is not None and not {
        "algorithm",
        "canonicalization",
        "line_endings",
    }.issubset(hash_contract):
        errors.append("hash_contract must declare algorithm, canonicalization, and line_endings")

    model = mapping(plan.get("model"), "model", errors)
    if model is not None:
        if (
            model.get("id") != "openai-community/gpt2"
            or model.get("revision") != "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
        ):
            errors.append("model must remain pinned to the declared GPT-2 revision")
        adapter = mapping(model.get("adapter_applicability"), "model.adapter_applicability", errors)
        if adapter is not None and (
            adapter.get("ModelAdapter") != "N/A (deliberate architecture/ADR)"
            or adapter.get("real_boundary") != "latent_anything.integrations.transformer_lm.TransformerLMIntegration"
        ):
            errors.append("model adapter boundary must remain TransformerLMIntegration with ModelAdapter=N/A")


def _validate_fixture_and_tokenization(plan: Mapping[str, Any], errors: list[str]) -> None:
    """Validate frozen fixture metadata and tokenizer/sampling settings."""
    fixture = mapping(plan.get("fixture"), "fixture", errors)
    if fixture is not None:
        if fixture.get("path") != "artifacts/m14/l04-prompt-factor-fixture.jsonl":
            errors.append("fixture path is not canonical")
        if fixture.get("rows") != 24 or fixture.get("groups") != 12 or fixture.get("pairs") != 12:
            errors.append("fixture must declare exactly 24 rows, 12 groups, and 12 pairs")
        digest_contract = mapping(fixture.get("digest_contract"), "fixture.digest_contract", errors)
        if digest_contract is not None and "independent_recompute" not in digest_contract:
            errors.append("fixture digest contract must require independent recomputation")
        split = mapping(fixture.get("split"), "fixture.split", errors)
        if split is not None:
            if split.get("train_groups") != [f"g{i:02d}" for i in range(1, 9)] or split.get("holdout_groups") != [
                f"g{i:02d}" for i in range(9, 13)
            ]:
                errors.append("fixture split groups must remain g01-g08 train and g09-g12 holdout")
            if split.get("labels") != ["animal_cat", "tone_positive"] or split.get("target_texts") != [
                " true",
                " false",
            ]:
                errors.append("fixture labels and target texts are not the frozen contract")

    token = mapping(plan.get("tokenization_and_sampling"), "tokenization_and_sampling", errors)
    if token is not None:
        frozen = {
            "max_length": 32,
            "padding": "longest within batch",
            "truncation": True,
            "target_position": "last non-padding prompt token; target_text is scored as the next-token class",
            "hidden_layer": 6,
            "native_hidden_state_index": 7,
            "seeds": [17, 29, 41, 53, 67],
            "bootstrap_replicates": 2000,
            "strength_grid": [0.0, 0.25, 0.5, 1.0],
            "ig_steps": [16, 64],
        }
        for key, expected in frozen.items():
            if token.get(key) != expected:
                errors.append(f"tokenization_and_sampling.{key} is not frozen as declared")
        requirement = mapping(token.get("target_token_offline_contract"), "target_token_offline_contract", errors)
        if requirement is not None and (
            requirement.get("target_texts") != [" true", " false"] or requirement.get("expected_token_count") != 1
        ):
            errors.append("offline target-token contract must check one token for true/false")


def _validate_record_order(plan: Mapping[str, Any], errors: list[str]) -> None:
    """Validate the five frozen records and their explicit sequence numbers."""
    records = plan.get("record_order")
    if not isinstance(records, list) or len(records) != 5:
        errors.append("plan must contain exactly five records in record_order")
        return
    record_ids: list[object] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            errors.append(f"record_order[{index - 1}] must be an object")
            continue
        record_ids.append(record.get("record_id"))
        if record.get("order") != index:
            errors.append(f"record_order[{index - 1}] has incorrect order")
    if tuple(record_ids) != EXPECTED_RECORD_IDS:
        errors.append("record_order does not match the five frozen record IDs")


def _validate_use_cases(plan: Mapping[str, Any], errors: list[str]) -> None:
    """Validate the seven real-use-case mappings and integration boundaries."""
    use_cases = plan.get("real_use_case_checklist")
    if not isinstance(use_cases, list) or len(use_cases) != 7:
        errors.append("real_use_case_checklist must contain exactly seven use cases")
        return
    names: list[object] = []
    expected_keys = frozenset(
        {"use_case", "record_id", "support_only", "model", "integration", "adapter", "cuda_server_required", "test"}
    )
    for index, use_case in enumerate(use_cases):
        if not isinstance(use_case, Mapping):
            errors.append(f"real_use_case_checklist[{index}] must be an object")
            continue
        exact_keys(use_case, expected_keys, f"real_use_case_checklist[{index}]", errors)
        names.append(use_case.get("use_case"))
        expected_record_id = (
            None,
            EXPECTED_RECORD_IDS[0],
            None,
            EXPECTED_RECORD_IDS[1],
            EXPECTED_RECORD_IDS[2],
            EXPECTED_RECORD_IDS[3],
            EXPECTED_RECORD_IDS[4],
        )[index]
        if use_case.get("record_id") != expected_record_id or use_case.get("support_only") is not (
            expected_record_id is None
        ):
            errors.append(f"real_use_case_checklist[{index}] has an invalid record/support mapping")
        if (
            use_case.get("model") != "openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
            or use_case.get("integration") != "TransformerLMIntegration"
            or use_case.get("adapter") != "N/A"
            or use_case.get("cuda_server_required") is not True
        ):
            errors.append(f"real_use_case_checklist[{index}] has an invalid model/integration boundary")
    if tuple(names) != EXPECTED_USE_CASES:
        errors.append("real_use_case_checklist order does not match the seven frozen use cases")


def _validate_thresholds(plan: Mapping[str, Any], errors: list[str]) -> None:
    """Validate threshold sections, comparator fields, controls, and values."""
    thresholds = mapping(plan.get("thresholds_and_controls"), "thresholds_and_controls", errors)
    if thresholds is None:
        return
    expected_sections = {
        "integrated_gradients",
        "tcav",
        "disentanglement",
        "activation_patching",
        "steering",
        "lens",
        "policy",
    }
    if set(thresholds) != expected_sections:
        errors.append("thresholds_and_controls must contain exactly the seven declared sections")
    policy = thresholds.get("policy")
    if not isinstance(policy, str) or "strict > comparison" not in policy or "<=" not in policy:
        errors.append("threshold policy must declare strict-min and max/atol comparator semantics")
    for name in expected_sections - {"policy"}:
        section = mapping(thresholds.get(name), f"thresholds_and_controls.{name}", errors)
        if section is None or not isinstance(section.get("controls"), list) or not section.get("controls"):
            errors.append(f"thresholds_and_controls.{name} must declare non-empty controls")

    required_thresholds = {
        "integrated_gradients": {
            "completeness_relative_error_max",
            "step_16_vs_64_attribution_cosine_min",
            "randomized_target_cosine_max",
            "controls",
        },
        "tcav": {
            "heldout_accuracy_min",
            "heldout_accuracy_wilson_lower_min",
            "bootstrap_ci_lower_min",
            "corrected_empirical_p_max",
            "intervention_agreement_min",
            "controls",
        },
        "disentanglement": {"heldout_gain_over_shuffled_min", "bootstrap_ci_lower_min", "controls"},
        "activation_patching": {
            "recovery_ci_lower_strict_gt",
            "off_target_absolute_effect_max",
            "zero_strength_identity_atol",
            "controls",
        },
        "steering": {
            "target_effect_ci_lower_strict_gt_logits",
            "selectivity_ci_lower_strict_gt_logits",
            "off_target_absolute_effect_max_logits",
            "zero_strength_identity_atol",
            "controls",
        },
        "lens": {
            "direct_parity_rtol",
            "direct_parity_atol",
            "tuned_holdout_kl_improvement_strict_gt_nats",
            "tuned_holdout_calibration_ci_lower_strict_gt",
            "controls",
        },
    }
    for name, required in required_thresholds.items():
        section = thresholds.get(name)
        if isinstance(section, Mapping) and not required.issubset(section):
            errors.append(f"thresholds_and_controls.{name} is missing a frozen comparator field")

    frozen_values = {
        "integrated_gradients": {
            "completeness_relative_error_max": 0.001,
            "step_16_vs_64_attribution_cosine_min": 0.95,
            "randomized_target_cosine_max": 0.25,
        },
        "tcav": {
            "heldout_accuracy_min": 0.60,
            "heldout_accuracy_wilson_lower_min": 0.55,
            "bootstrap_ci_lower_min": 0.50,
            "corrected_empirical_p_max": 0.05,
            "intervention_agreement_min": 0.80,
        },
        "disentanglement": {"heldout_gain_over_shuffled_min": 0.10, "bootstrap_ci_lower_min": 0.05},
        "activation_patching": {
            "recovery_ci_lower_strict_gt": 0.10,
            "off_target_absolute_effect_max": 0.10,
            "zero_strength_identity_atol": 1e-6,
        },
        "steering": {
            "target_effect_ci_lower_strict_gt_logits": 0.05,
            "selectivity_ci_lower_strict_gt_logits": 0.05,
            "off_target_absolute_effect_max_logits": 0.10,
            "zero_strength_identity_atol": 1e-6,
        },
        "lens": {
            "direct_parity_rtol": 1e-6,
            "direct_parity_atol": 1e-6,
            "tuned_holdout_kl_improvement_strict_gt_nats": 0.01,
            "tuned_holdout_calibration_ci_lower_strict_gt": 0.01,
        },
    }
    for name, expected in frozen_values.items():
        section = thresholds.get(name)
        if isinstance(section, Mapping):
            for field, value in expected.items():
                if section.get(field) != value:
                    errors.append(f"thresholds_and_controls.{name}.{field} is not frozen as declared")


def _validate_resource_budget(plan: Mapping[str, Any], errors: list[str]) -> None:
    """Validate the resource and network policy for real use cases."""
    resource = mapping(plan.get("resource_budget"), "resource_budget", errors)
    if resource is None:
        return
    required_resource = {
        "device",
        "reference_hardware",
        "vram_peak_gb_max",
        "host_rss_gb_max",
        "runtime_minutes_max",
        "network",
        "failure_policy",
    }
    if not required_resource.issubset(resource):
        errors.append("resource_budget is missing required resource/network/failure fields")
    if resource.get("device") != "CUDA server only for all real model/integration use cases":
        errors.append("resource budget must keep real use cases on the CUDA server")


def _validate_remote_workflow(plan: Mapping[str, Any], errors: list[str]) -> None:
    """Validate required markers in the disposable remote CUDA workflow."""
    remote = mapping(plan.get("remote_cuda_workflow"), "remote_cuda_workflow", errors)
    if remote is None:
        return
    required_remote = {
        "transport_rule",
        "precondition",
        "powershell_commands",
        "capture_rule",
        "exact_sha_rule",
        "cleanup_rule",
    }
    if not required_remote.issubset(remote) or not isinstance(remote.get("powershell_commands"), list):
        errors.append("remote_cuda_workflow must declare transport, capture, exact-SHA, and cleanup protocol")
    text = "\n".join(str(command) for command in remote.get("powershell_commands", []))
    for marker in ("ssh.exe", "LASTEXITCODE", "mktemp", "trap", "git checkout --detach", "base64"):
        if marker not in text:
            errors.append(f"remote protocol is missing required marker {marker!r}")


def _validate_artifact_schema(plan: Mapping[str, Any], errors: list[str]) -> None:
    """Validate the required final and failure artifact schema sections."""
    artifact = mapping(plan.get("artifact_schema"), "artifact_schema", errors)
    if artifact is None:
        return
    for key in (
        "final_artifact",
        "final_schema",
        "required_top_level",
        "record_fields",
        "run_record",
        "failure_record",
    ):
        if key not in artifact:
            errors.append(f"artifact_schema is missing {key}")


def _validate_plan_structure(plan: Mapping[str, Any], errors: list[str]) -> None:
    """Append all structural plan errors in the frozen contract's order."""
    _validate_identity_hash_model(plan, errors)
    _validate_fixture_and_tokenization(plan, errors)
    _validate_record_order(plan, errors)
    _validate_use_cases(plan, errors)
    _validate_thresholds(plan, errors)
    _validate_resource_budget(plan, errors)
    _validate_remote_workflow(plan, errors)
    _validate_artifact_schema(plan, errors)


def validate_plan(plan: Mapping[str, Any]) -> list[str]:
    """Return all structural and digest errors for a frozen plan."""
    errors: list[str] = []
    _validate_plan_structure(plan, errors)
    stored = plan.get("plan_sha256")
    if not isinstance(stored, str) or stored != plan_digest(plan):
        errors.append("plan_sha256 does not match the canonical unsigned plan")
    return errors
