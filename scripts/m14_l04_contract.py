"""Offline contract validation for the frozen M14 L04 explanation plan.

This module deliberately has no model, Transformers, network, or filesystem
write dependency.  Real tokenizer resolution belongs to the future real-run
preflight; the offline check validates only the declared one-token invariant
through an injected tokenization callback.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

PLAN_PATH = Path(__file__).resolve().parents[1] / "artifacts/m14/l04-explanations.plan.json"
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "artifacts/m14/l04-prompt-factor-fixture.jsonl"
SCHEMA_VERSION = "m14-l04-explanations-plan-v1"
FIXTURE_SPLIT_SCHEMA = "l04-fixture-split-v1"
FIXTURE_PAIR_SCHEMA = "l04-fixture-pair-v1"

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
FIXTURE_ROW_KEYS = frozenset(
    {"row_id", "group_id", "causal_pair_id", "condition", "split", "task", "prompt", "target_text", "factor_labels"}
)


class ContractValidationError(ValueError):
    """Raised when a frozen L04 contract is malformed or inconsistent."""


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_json_bytes(raw: bytes, *, source: str) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ContractValidationError(f"{source} must not contain a UTF-8 BOM")
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_non_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractValidationError(f"{source} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ContractValidationError(f"{source} must be a JSON object")
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a JSON object using the plan's immutable canonical encoding."""
    try:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("value cannot be canonically serialized") from exc
    return (encoded + "\n").encode("utf-8")


def _canonical_fixture_payload_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode split/pair payloads while preserving their explicitly listed key order."""
    try:
        encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("fixture payload cannot be canonically serialized") from exc
    return (encoded + "\n").encode("utf-8")


def digest_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


def plan_digest(plan: Mapping[str, Any]) -> str:
    """Compute the unsigned canonical plan digest without mutating ``plan``."""
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    return digest_bytes(canonical_json_bytes(unsigned))


def _mapping(value: object, name: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return None
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str, errors: list[str]) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        errors.append(f"{name} schema mismatch ({'; '.join(details)})")


def _validate_plan_structure(plan: Mapping[str, Any], errors: list[str]) -> None:
    _exact_keys(plan, EXPECTED_PLAN_KEYS, "plan", errors)
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("lane") != "L04":
        errors.append("plan identity must be m14-l04-explanations-plan-v1/L04")
    if plan.get("status") != "design-frozen-pending-implementation":
        errors.append("plan must remain design-frozen-pending-implementation")

    hash_contract = _mapping(plan.get("hash_contract"), "hash_contract", errors)
    if hash_contract is not None and not {
        "algorithm",
        "canonicalization",
        "line_endings",
    }.issubset(hash_contract):
        errors.append("hash_contract must declare algorithm, canonicalization, and line_endings")

    model = _mapping(plan.get("model"), "model", errors)
    if model is not None:
        if (
            model.get("id") != "openai-community/gpt2"
            or model.get("revision") != "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
        ):
            errors.append("model must remain pinned to the declared GPT-2 revision")
        adapter = _mapping(model.get("adapter_applicability"), "model.adapter_applicability", errors)
        if adapter is not None and (
            adapter.get("ModelAdapter") != "N/A (deliberate architecture/ADR)"
            or adapter.get("real_boundary") != "latent_anything.integrations.transformer_lm.TransformerLMIntegration"
        ):
            errors.append("model adapter boundary must remain TransformerLMIntegration with ModelAdapter=N/A")

    fixture = _mapping(plan.get("fixture"), "fixture", errors)
    if fixture is not None:
        if fixture.get("path") != "artifacts/m14/l04-prompt-factor-fixture.jsonl":
            errors.append("fixture path is not canonical")
        if fixture.get("rows") != 24 or fixture.get("groups") != 12 or fixture.get("pairs") != 12:
            errors.append("fixture must declare exactly 24 rows, 12 groups, and 12 pairs")
        digest_contract = _mapping(fixture.get("digest_contract"), "fixture.digest_contract", errors)
        if digest_contract is not None and "independent_recompute" not in digest_contract:
            errors.append("fixture digest contract must require independent recomputation")
        split = _mapping(fixture.get("split"), "fixture.split", errors)
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

    token = _mapping(plan.get("tokenization_and_sampling"), "tokenization_and_sampling", errors)
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
        requirement = _mapping(token.get("target_token_offline_contract"), "target_token_offline_contract", errors)
        if requirement is not None and (
            requirement.get("target_texts") != [" true", " false"] or requirement.get("expected_token_count") != 1
        ):
            errors.append("offline target-token contract must check one token for true/false")

    records = plan.get("record_order")
    if not isinstance(records, list) or len(records) != 5:
        errors.append("plan must contain exactly five records in record_order")
    else:
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

    use_cases = plan.get("real_use_case_checklist")
    if not isinstance(use_cases, list) or len(use_cases) != 7:
        errors.append("real_use_case_checklist must contain exactly seven use cases")
    else:
        names: list[object] = []
        expected_keys = frozenset(
            {"use_case", "record_id", "support_only", "model", "integration", "adapter", "cuda_server_required", "test"}
        )
        for index, use_case in enumerate(use_cases):
            if not isinstance(use_case, Mapping):
                errors.append(f"real_use_case_checklist[{index}] must be an object")
                continue
            _exact_keys(use_case, expected_keys, f"real_use_case_checklist[{index}]", errors)
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

    thresholds = _mapping(plan.get("thresholds_and_controls"), "thresholds_and_controls", errors)
    if thresholds is not None:
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
            section = _mapping(thresholds.get(name), f"thresholds_and_controls.{name}", errors)
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

    resource = _mapping(plan.get("resource_budget"), "resource_budget", errors)
    if resource is not None:
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

    remote = _mapping(plan.get("remote_cuda_workflow"), "remote_cuda_workflow", errors)
    if remote is not None:
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

    artifact = _mapping(plan.get("artifact_schema"), "artifact_schema", errors)
    if artifact is not None:
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


def validate_plan(plan: Mapping[str, Any]) -> list[str]:
    """Return all structural and digest errors for a frozen plan."""
    errors: list[str] = []
    _validate_plan_structure(plan, errors)
    stored = plan.get("plan_sha256")
    if not isinstance(stored, str) or stored != plan_digest(plan):
        errors.append("plan_sha256 does not match the canonical unsigned plan")
    return errors


def _read_fixture(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ContractValidationError("fixture must not contain a UTF-8 BOM")
    if b"\r" in raw:
        raise ContractValidationError("fixture must use LF-only line endings")
    if not raw.endswith(b"\n"):
        raise ContractValidationError("fixture must end with exactly one LF per row")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractValidationError("fixture is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.split("\n")[:-1], start=1):
        if not line:
            raise ContractValidationError(f"fixture row {line_number} is empty")
        try:
            value = json.loads(line, parse_constant=_reject_non_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ContractValidationError(f"fixture row {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ContractValidationError(f"fixture row {line_number} must be an object")
        rows.append(value)
    return raw, rows


def content_digest(raw: bytes) -> str:
    """Digest the exact fixture bytes, without parsing or normalization."""
    return digest_bytes(raw)


def split_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Digest the canonical split payload with its declared field order."""
    split_rows = [
        {"row_id": row["row_id"], "group_id": row["group_id"], "split": row["split"]}
        for row in sorted(rows, key=lambda item: (str(item["group_id"]), str(item["row_id"])))
    ]
    return digest_bytes(_canonical_fixture_payload_bytes({"schema": FIXTURE_SPLIT_SCHEMA, "rows": split_rows}))


def pair_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Digest the canonical one-clean/one-corrupted causal-pair payload."""
    pair_rows: list[dict[str, Any]] = []
    for pair_id, pair in sorted(_group_by(rows, "causal_pair_id").items()):
        clean = next(row for row in pair if row.get("condition") == "clean")
        pair_rows.append(
            {
                "causal_pair_id": pair_id,
                "group_id": clean["group_id"],
                "clean_row_id": clean["row_id"],
                "corrupted_row_id": next(row["row_id"] for row in pair if row.get("condition") == "corrupted"),
                "split": clean["split"],
            }
        )
    return digest_bytes(_canonical_fixture_payload_bytes({"schema": FIXTURE_PAIR_SCHEMA, "pairs": pair_rows}))


def fixture_digests(raw: bytes, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Recompute content, split, and causal-pair digests independently."""
    return {
        "content_sha256": content_digest(raw),
        "split_sha256": split_digest(rows),
        "pair_sha256": pair_digest(rows),
    }


def _group_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return grouped


def validate_fixture(plan: Mapping[str, Any], raw: bytes, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate fixture rows, splits, pairs, labels, and all three digests."""
    errors: list[str] = []
    fixture = _mapping(plan.get("fixture"), "fixture", errors)
    if fixture is None:
        return errors
    if len(rows) != 24:
        errors.append("fixture must contain exactly 24 records")
    if len({row.get("row_id") for row in rows}) != len(rows):
        errors.append("fixture row_id values must be unique")
    if any(frozenset(row) != FIXTURE_ROW_KEYS for row in rows):
        errors.append("fixture rows must use the exact frozen schema")
    for index, row in enumerate(rows):
        if not all(
            isinstance(row.get(key), str) and row.get(key)
            for key in ("row_id", "group_id", "causal_pair_id", "condition", "split", "task", "prompt", "target_text")
        ):
            errors.append(f"fixture row {index} has missing/non-string identity or text")
        labels = row.get("factor_labels")
        if (
            not isinstance(labels, Mapping)
            or set(labels) != {"animal_cat", "tone_positive"}
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1)
                for value in labels.values()
            )
        ):
            errors.append(f"fixture row {index} has invalid factor labels")
            continue
        condition = row.get("condition")
        expected_target = " true" if condition == "clean" else " false" if condition == "corrupted" else None
        if expected_target is None or row.get("target_text") != expected_target:
            errors.append(f"fixture row {index} condition/target mismatch")
        if labels.get("tone_positive") != (1 if condition == "clean" else 0 if condition == "corrupted" else -1):
            errors.append(f"fixture row {index} condition/tone_positive mismatch")

    groups = _group_by(rows, "group_id")
    pairs = _group_by(rows, "causal_pair_id")
    if len(groups) != 12 or len(pairs) != 12:
        errors.append("fixture must contain exactly 12 groups and 12 causal pairs")
    train_groups = {f"g{i:02d}" for i in range(1, 9)}
    holdout_groups = {f"g{i:02d}" for i in range(9, 13)}
    for group_id, group_rows in groups.items():
        splits = {row.get("split") for row in group_rows}
        pair_ids = {row.get("causal_pair_id") for row in group_rows}
        if len(splits) != 1 or len(pair_ids) != 1:
            errors.append(f"group {group_id!r} crosses split or causal pair")
        expected_split = "train" if group_id in train_groups else "holdout" if group_id in holdout_groups else None
        if expected_split is None or splits != {expected_split}:
            errors.append(f"group {group_id!r} has an undeclared split")
    for pair_id, pair_rows in pairs.items():
        conditions = [str(row.get("condition")) for row in pair_rows]
        if len(pair_rows) != 2 or sorted(conditions) != ["clean", "corrupted"]:
            errors.append(f"pair {pair_id!r} must have exactly one clean and one corrupted row")
            continue
        if (
            len({row.get("group_id") for row in pair_rows}) != 1
            or len({row.get("split") for row in pair_rows}) != 1
            or len({row.get("task") for row in pair_rows}) != 1
        ):
            errors.append(f"pair {pair_id!r} crosses group, split, or task")

    try:
        digests = fixture_digests(raw, rows)
    except (KeyError, StopIteration, TypeError) as exc:
        errors.append(f"fixture digest payload cannot be constructed: {exc}")
    else:
        for key in ("content_sha256", "split_sha256", "pair_sha256"):
            if digests[key] != fixture.get(key):
                errors.append(f"fixture {key} does not match its canonical recomputation")
    return errors


def validate_target_tokens(
    tokenize: Callable[[str], object], target_texts: Sequence[str], *, expected_token_count: int = 1
) -> None:
    """Validate target token cardinality through an injected tokenizer seam.

    The callback is intentionally the only tokenizer dependency.  This helper
    records no token IDs and performs no model/tokenizer resolution; it is for
    the future real-run preflight and deterministic fake-only unit tests.
    """
    for target_text in target_texts:
        try:
            encoded = tokenize(target_text)
        except Exception as exc:  # noqa: BLE001 - convert arbitrary tokenizer failures to contract errors
            raise ContractValidationError(f"tokenizer failed for target {target_text!r}") from exc
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
        if isinstance(encoded, (str, bytes)) or not isinstance(encoded, Sequence):
            raise ContractValidationError(f"tokenizer result for {target_text!r} has no token sequence")
        if len(encoded) != expected_token_count:
            raise ContractValidationError(
                f"target {target_text!r} resolves to {len(encoded)} tokens; expected {expected_token_count}"
            )


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    """Load and validate the frozen plan without optional dependencies."""
    plan = _load_json_bytes(path.read_bytes(), source="plan")
    errors = validate_plan(plan)
    if errors:
        raise ContractValidationError("; ".join(errors))
    return plan


def load_fixture(path: Path = FIXTURE_PATH) -> tuple[bytes, list[dict[str, Any]]]:
    """Load the raw UTF-8/LF fixture and its parsed rows."""
    return _read_fixture(path)


def load_and_validate(plan_path: Path = PLAN_PATH, fixture_path: Path = FIXTURE_PATH) -> dict[str, str]:
    """Read the frozen plan/fixture without writing, downloading, or tokenizing.

    The offline check validates only the declared target strings and expected
    one-token count.  It does not establish actual GPT-2 token cardinality;
    that belongs to the future real-run preflight via ``validate_target_tokens``.
    """
    plan = _load_json_bytes(plan_path.read_bytes(), source="plan")
    errors = validate_plan(plan)
    raw, rows = _read_fixture(fixture_path)
    errors.extend(validate_fixture(plan, raw, rows))
    if errors:
        raise ContractValidationError("; ".join(errors))
    return {"plan_sha256": plan_digest(plan), **fixture_digests(raw, rows)}


def check_plan(plan_path: Path = PLAN_PATH, fixture_path: Path = FIXTURE_PATH) -> dict[str, str]:
    """Compatibility facade for the canonical side-effect-free check."""
    return load_and_validate(plan_path, fixture_path)
