"""Fail-closed validator for the M14 L04.9 interchange payload."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from scripts._m14_l04_activation_patching import (
    BOOTSTRAP_REPLICATES,
    LAYER,
    NATIVE_HIDDEN_STATE_INDEX,
    OFF_TARGET_ABSOLUTE_EFFECT_MAX,
    OFF_TARGET_LAYER,
    RECOVERY_CI_LOWER_THRESHOLD,
    SEEDS,
    STRENGTH_GRID,
    TARGET_TOKEN_IDS,
    TARGET_TOKEN_STRINGS,
    ZERO_STRENGTH_IDENTITY_ATOL,
    _metric,  # pyright: ignore[reportPrivateUsage]
    budget_pass,
    deterministic_split_donor_derangement,
    donor_mapping_digest,
)
from scripts._m14_l04_fixture_contract import FIXTURE_PATH, fixture_digests, read_fixture

REAL_ACTIVATION_PATCHING_STATUS = "passed_real_cuda"
RECORD_ID = "THY-T05-ACTIVATION-PATCHING"
SHUFFLED_SEMANTIC = "shuffled donor activation; compatibility key retained"
PARAMETER_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
EVIDENCE_NUMERIC_FIELDS = (
    "clean_margin",
    "corrupted_margin",
    "true_interchange_margin",
    "off_target_layer_margin",
    "off_target_token_margin",
    "shuffled_donor_margin",
    "zero_strength_margin",
    "recovery",
    "off_target_layer_effect",
    "off_target_token_effect",
    "shuffled_donor_effect",
    "zero_strength_error",
)
SUMMARY_METRIC_FIELDS = ("recovery", "off_target", "off_target_layer", "off_target_token", "zero_strength")
COMPLETED_CLEANUP = "CUDA synchronized; model gradients cleared; CUDA cache emptied"
SUMMARY_KEYS = frozenset(
    {
        "seed",
        "train_pairs",
        "holdout_pairs",
        "holdout_evidence",
        "recovery",
        "off_target",
        "off_target_layer",
        "off_target_token",
        "zero_strength",
        "shuffled_direction",
        "finite",
        "bootstrap_replicates",
    }
)
EVIDENCE_KEYS = frozenset(
    {
        "pair_id",
        "group_id",
        "split",
        "clean_row_id",
        "corrupted_row_id",
        "clean_condition",
        "corrupted_condition",
        "clean_target_position",
        "corrupted_target_position",
        "clean_previous_valid_position",
        "corrupted_previous_valid_position",
        "clean_margin",
        "corrupted_margin",
        "true_interchange_margin",
        "off_target_layer_margin",
        "off_target_token_margin",
        "shuffled_donor_margin",
        "zero_strength_margin",
        "recovery",
        "off_target_layer_effect",
        "off_target_token_effect",
        "shuffled_donor_effect",
        "zero_strength_error",
        "strength_grid",
    }
)


def _expected_linkage(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import hashlib

    return [
        {
            "row_id": str(row["row_id"]),
            "group_id": str(row["group_id"]),
            "causal_pair_id": str(row["causal_pair_id"]),
            "condition": str(row["condition"]),
            "split": str(row["split"]),
            "prompt_sha256": hashlib.sha256(str(row["prompt"]).encode()).hexdigest(),
            "target_text_sha256": hashlib.sha256(str(row["target_text"]).encode()).hexdigest(),
            "task_sha256": hashlib.sha256(str(row["task"]).encode()).hexdigest(),
        }
        for row in rows
    ]


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _forbidden(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in {"prompt", "text", "logits", "hidden", "hidden_states"} or _forbidden(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_forbidden(item) for item in value)
    return False


def _resource_errors(entry: Mapping[str, Any], provenance: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    peak = provenance.get("resource_peak")
    if not isinstance(peak, Mapping):
        return ["activation patching resource peak is missing"]
    entry_resources = entry.get("resources")
    entry_peak = entry_resources.get("resource_peak") if isinstance(entry_resources, Mapping) else None
    if not isinstance(entry_peak, Mapping) or dict(entry_peak) != dict(peak):
        errors.append("activation patching execution resource peak is not bound to provenance")
    for field in ("elapsed_seconds", "max_memory_allocated_bytes", "max_memory_reserved_bytes", "max_rss_bytes"):
        if not _finite(peak.get(field)):
            errors.append(f"activation patching resource {field} is invalid")
    expected_budget = budget_pass(peak)
    if provenance.get("budget_pass") is not expected_budget or entry.get("budget_pass") is not expected_budget:
        errors.append("activation patching resource budget did not pass")
    entry_before = entry.get("model_parameter_digest_before")
    entry_after = entry.get("model_parameter_digest_after")
    provenance_before = provenance.get("model_parameter_digest_before")
    provenance_after = provenance.get("model_parameter_digest_after")
    valid_digests = all(
        isinstance(value, str) and PARAMETER_DIGEST_RE.fullmatch(value) is not None
        for value in (entry_before, entry_after, provenance_before, provenance_after)
    )
    if not valid_digests or not (entry_before == entry_after == provenance_before == provenance_after):
        errors.append("activation patching parameter digests are missing, malformed, or mismatched")
    if entry.get("no_mutation") is not True:
        errors.append("activation patching model mutation control failed")
    return errors


def _derived_recovery(item: Mapping[str, Any]) -> float | None:
    clean = item.get("clean_margin")
    corrupted = item.get("corrupted_margin")
    true_interchange = item.get("true_interchange_margin")
    if not all(_finite(value) for value in (clean, corrupted, true_interchange)):
        return None
    clean_value = cast(float, clean)
    corrupted_value = cast(float, corrupted)
    true_interchange_value = cast(float, true_interchange)
    denominator = float(clean_value) - float(corrupted_value)
    if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return None
    return (float(true_interchange_value) - float(corrupted_value)) / abs(denominator)


def _recomputed_recovery(
    evidence: Sequence[Mapping[str, Any]], holdout_groups: set[str], seed: int
) -> tuple[dict[str, Any], bool] | None:
    grouped: dict[str, list[float]] = {}
    declared_matches = True
    for item in evidence:
        group = str(item.get("group_id"))
        derived = _derived_recovery(item)
        if derived is None:
            return None
        if not _finite(item.get("recovery")) or not math.isclose(
            float(item["recovery"]), derived, rel_tol=0.0, abs_tol=1e-12
        ):
            declared_matches = False
        grouped.setdefault(group, []).append(derived)
    if any(group not in grouped or not grouped[group] for group in holdout_groups):
        return None
    return (
        _metric(
            [sum(grouped[group]) / len(grouped[group]) for group in sorted(holdout_groups)],
            seed=seed,
            threshold=RECOVERY_CI_LOWER_THRESHOLD,
            comparator=">",
            units="normalized causal recovery",
        ),
        declared_matches,
    )


def validate_completed_activation_failure_structure(
    entry: dict[str, Any], plan: dict[str, Any], rows: Sequence[Mapping[str, Any]], resources: Mapping[str, Any]
) -> list[str]:
    """Validate the complete-scoring shape before failed-stage normalization."""
    errors: list[str] = []
    result_resources = entry.get("resources")
    provenance = entry.get("provenance")
    controls = entry.get("controls")
    if (
        not isinstance(result_resources, Mapping)
        or not isinstance(provenance, Mapping)
        or not isinstance(controls, Mapping)
    ):
        return ["activation patching completed failure structure is missing nested envelopes"]
    if (
        entry.get("status") != "failed"
        or entry.get("evidence_eligible") is not False
        or entry.get("acceptance") is not False
    ):
        errors.append("activation patching completed failure status/acceptance is invalid")
    if entry.get("evidence_level") != "D0":
        errors.append("activation patching completed failure must be D0")
    if entry.get("record_id") != RECORD_ID or entry.get("support_only") is not False:
        errors.append("activation patching completed failure record boundary is invalid")
    model = plan.get("model")
    if not isinstance(model, Mapping) or provenance.get("model_revision") != model.get("revision"):
        errors.append("activation patching completed failure model provenance is invalid")
    if (
        entry.get("layer") != LAYER
        or entry.get("native_hidden_state_index") != NATIVE_HIDDEN_STATE_INDEX
        or entry.get("token_ids") != TARGET_TOKEN_IDS
        or entry.get("target_token_strings") != TARGET_TOKEN_STRINGS
    ):
        errors.append("activation patching completed failure target linkage is invalid")
    if entry.get("seed") != SEEDS[0] or entry.get("seeds") != list(SEEDS):
        errors.append("activation patching completed failure seeds are not frozen")
    expected_linkage = _expected_linkage(rows)
    if entry.get("fixture_linkage") != expected_linkage:
        errors.append("activation patching completed failure fixture linkage is not exact")
    expected_fields = (
        "metrics",
        "confidence_intervals",
        "raw_summaries",
        "fixture_linkage",
        "budget_pass",
        "no_mutation",
        "model_parameter_digest_before",
        "model_parameter_digest_after",
    )
    if any(field not in entry for field in expected_fields):
        errors.append("activation patching completed failure semantic fields are missing")
    summaries = entry.get("raw_summaries")
    if (
        not isinstance(summaries, list)
        or len(summaries) != len(SEEDS)
        or [item.get("seed") if isinstance(item, Mapping) else None for item in summaries] != list(SEEDS)
    ):
        errors.append("activation patching completed failure summaries are not the frozen ordered seeds")
        summaries = []
    frozen_pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        frozen_pairs.setdefault(str(row["causal_pair_id"]), {})[str(row["condition"])] = row
    train_pairs = sorted(pair for pair, value in frozen_pairs.items() if value["clean"]["split"] == "train")
    holdout_pairs = sorted(pair for pair, value in frozen_pairs.items() if value["clean"]["split"] == "holdout")
    holdout_groups = {str(row["group_id"]) for row in rows if row["split"] == "holdout"}
    independently_failed_gate = False
    for summary in summaries:
        if not isinstance(summary, Mapping) or set(summary) != set(SUMMARY_KEYS):
            errors.append("activation patching completed failure summary schema is invalid")
            continue
        if summary.get("train_pairs") != train_pairs or summary.get("holdout_pairs") != holdout_pairs:
            errors.append("activation patching completed failure pair domains are invalid")
        evidence = summary.get("holdout_evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append("activation patching completed failure evidence is missing")
            continue
        if {str(item.get("pair_id")) for item in evidence if isinstance(item, Mapping)} != set(holdout_pairs):
            errors.append("activation patching completed failure evidence membership is invalid")
        for item in evidence:
            if not isinstance(item, Mapping) or set(item) != set(EVIDENCE_KEYS):
                errors.append("activation patching completed failure evidence schema is invalid")
                continue
            expected = frozen_pairs.get(str(item.get("pair_id")), {})
            if (
                not expected
                or item.get("group_id") != str(expected["clean"]["group_id"])
                or item.get("split") != "holdout"
                or item.get("clean_row_id") != str(expected["clean"]["row_id"])
                or item.get("corrupted_row_id") != str(expected["corrupted"]["row_id"])
                or item.get("clean_condition") != "clean"
                or item.get("corrupted_condition") != "corrupted"
                or str(item.get("group_id")) not in holdout_groups
                or any(not _finite(item.get(field)) for field in EVIDENCE_NUMERIC_FIELDS)
            ):
                errors.append("activation patching completed failure evidence linkage or metrics are invalid")
            curve = item.get("strength_grid")
            if not isinstance(curve, Mapping) or set(curve) != {str(value) for value in STRENGTH_GRID}:
                errors.append("activation patching completed failure strength grid is invalid")
        if isinstance(summary.get("recovery"), Mapping) and all(
            isinstance(item, Mapping) and set(item) == set(EVIDENCE_KEYS) for item in evidence
        ):
            recomputed_recovery = _recomputed_recovery(evidence, holdout_groups, int(summary["seed"]))
            if (
                recomputed_recovery is None
                or not recomputed_recovery[1]
                or summary["recovery"] != recomputed_recovery[0]
            ):
                errors.append("activation patching completed failure recovery was not independently recomputed")
            elif recomputed_recovery[0]["pass"] is False:
                independently_failed_gate = True
    expected_controls = {
        "clean_endpoint",
        "corrupted_endpoint",
        "true_interchange",
        "off_target_layer",
        "off_target_token",
        "off_target_combined",
        "shuffled_direction",
        "zero_strength",
    }
    if set(controls) != expected_controls or any(
        not isinstance(controls.get(name), Mapping) for name in expected_controls
    ):
        errors.append("activation patching completed failure controls are invalid")
    metrics = entry.get("metrics")
    confidence_intervals = entry.get("confidence_intervals")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get("recovery"), Mapping):
        errors.append("activation patching completed failure metrics are invalid")
    if not isinstance(confidence_intervals, Mapping) or set(confidence_intervals) != {str(seed) for seed in SEEDS}:
        errors.append("activation patching completed failure confidence intervals are invalid")
    for mapping in (resources, result_resources, provenance):
        if (
            mapping.get("stage") != "complete"
            or mapping.get("execution_attempted") is not True
            or mapping.get("execution_backend") != "cuda"
            or mapping.get("network") != "enabled"
            or mapping.get("cleanup") != COMPLETED_CLEANUP
        ):
            errors.append("activation patching completed failure execution envelope is invalid")
    errors.extend(_resource_errors(entry, provenance))
    peak = provenance.get("resource_peak")
    if isinstance(peak, Mapping) and not budget_pass(peak):
        independently_failed_gate = True
    if entry.get("no_mutation") is False:
        independently_failed_gate = True
    if not independently_failed_gate:
        errors.append("activation patching completed failure has no independently recomputed failed gate")
    return errors


def validate_real_activation_patching_execution(
    entry: dict[str, Any], artifact: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if entry.get("status") != REAL_ACTIVATION_PATCHING_STATUS:
        return errors
    if entry.get("record_id") != RECORD_ID or entry.get("support_only") is not False:
        errors.append("activation patching record boundary is invalid")
    if (
        entry.get("evidence_level") != "D3"
        or entry.get("evidence_eligible") is not True
        or entry.get("acceptance") is not True
    ):
        errors.append("real activation patching must be accepted D3 evidence")
    if entry.get("layer") != LAYER or entry.get("native_hidden_state_index") != NATIVE_HIDDEN_STATE_INDEX:
        errors.append("activation patching layer/native index is invalid")
    if entry.get("token_ids") != TARGET_TOKEN_IDS or entry.get("target_token_strings") != TARGET_TOKEN_STRINGS:
        errors.append("activation patching target-token linkage is invalid")
    if entry.get("seed") != SEEDS[0] or entry.get("seeds") != list(SEEDS) or artifact.get("seeds") != list(SEEDS):
        errors.append("activation patching seeds are not frozen")
    try:
        raw_fixture, frozen_rows = read_fixture(FIXTURE_PATH)
        expected_fixture = fixture_digests(raw_fixture, frozen_rows)
    except Exception as exc:  # noqa: BLE001 - malformed fixture is a D0 result
        return errors + [f"activation patching frozen fixture is unavailable: {type(exc).__name__}"]
    fixture = artifact.get("fixture")
    if not isinstance(fixture, Mapping) or any(fixture.get(key) != value for key, value in expected_fixture.items()):
        errors.append("activation patching fixture digests are not bound to the frozen fixture")
    plan_fixture = plan.get("fixture")
    expected_train_groups = sorted({str(row["group_id"]) for row in frozen_rows if row["split"] == "train"})
    expected_holdout_groups = sorted({str(row["group_id"]) for row in frozen_rows if row["split"] == "holdout"})
    expected_split = {"train_groups": expected_train_groups, "holdout_groups": expected_holdout_groups}
    if (
        not isinstance(plan_fixture, Mapping)
        or any(plan_fixture.get(key) != value for key, value in expected_fixture.items())
        or plan_fixture.get("path") != "artifacts/m14/l04-prompt-factor-fixture.jsonl"
        or plan_fixture.get("rows") != len(frozen_rows)
        or plan_fixture.get("groups") != len({str(row["group_id"]) for row in frozen_rows})
        or plan_fixture.get("pairs") != len({str(row["causal_pair_id"]) for row in frozen_rows})
        or not isinstance(plan_fixture.get("split"), Mapping)
        or any(plan_fixture["split"].get(key) != value for key, value in expected_split.items())
    ):
        errors.append("activation patching plan fixture digests do not match the frozen fixture")
    if entry.get("fixture_linkage") != _expected_linkage(frozen_rows):
        errors.append("activation patching fixture row linkage is not exact")
    frozen_pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in frozen_rows:
        frozen_pairs.setdefault(str(row["causal_pair_id"]), {})[str(row["condition"])] = row
    expected_mapping_by_seed = {seed: deterministic_split_donor_derangement(frozen_pairs, seed) for seed in SEEDS}
    summaries = artifact.get("raw_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(SEEDS):
        return errors + ["activation patching must retain one summary per frozen seed"]
    summary_seeds = [summary.get("seed") if isinstance(summary, Mapping) else None for summary in summaries]
    if summary_seeds != list(SEEDS):
        errors.append("activation patching summaries must contain the unique frozen seeds in order")
    if _forbidden(summaries):
        errors.append("activation patching summaries contain forbidden prompt or tensor data")
    holdout_groups = set(expected_holdout_groups)
    for summary in summaries:
        if not isinstance(summary, Mapping):
            errors.append("activation patching summary is not an object")
            continue
        seed = summary.get("seed")
        if (
            not isinstance(seed, int)
            or seed not in SEEDS
            or summary.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        ):
            errors.append("activation patching summary seed/bootstrap linkage is invalid")
            continue
        expected_mapping = expected_mapping_by_seed[int(seed)]
        expected_train_pairs = sorted(
            pair for pair, value in frozen_pairs.items() if value["clean"]["split"] == "train"
        )
        expected_holdout_pair_list = sorted(
            pair for pair, value in frozen_pairs.items() if value["clean"]["split"] == "holdout"
        )
        if (
            summary.get("train_pairs") != expected_train_pairs
            or summary.get("holdout_pairs") != expected_holdout_pair_list
        ):
            errors.append("activation patching train/holdout pair domains are invalid")
        evidence = summary.get("holdout_evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append("activation patching holdout evidence is missing")
            continue
        expected_evidence_keys = {
            "pair_id",
            "group_id",
            "split",
            "clean_row_id",
            "corrupted_row_id",
            "clean_condition",
            "corrupted_condition",
            "clean_target_position",
            "corrupted_target_position",
            "clean_previous_valid_position",
            "corrupted_previous_valid_position",
            "clean_margin",
            "corrupted_margin",
            "true_interchange_margin",
            "off_target_layer_margin",
            "off_target_token_margin",
            "shuffled_donor_margin",
            "zero_strength_margin",
            "recovery",
            "off_target_layer_effect",
            "off_target_token_effect",
            "shuffled_donor_effect",
            "zero_strength_error",
            "strength_grid",
        }
        if any(set(item) != expected_evidence_keys for item in evidence if isinstance(item, Mapping)):
            errors.append("activation patching evidence schema contains unexpected nested fields")
        expected_holdout_pairs = {
            str(row["causal_pair_id"])
            for row in frozen_rows
            if row["split"] == "holdout" and row["condition"] == "clean"
        }
        if {str(item.get("pair_id")) for item in evidence if isinstance(item, Mapping)} != expected_holdout_pairs:
            errors.append("activation patching evidence pair membership is not frozen")
        expected_by_pair = {
            str(pair): value for pair, value in frozen_pairs.items() if value["clean"]["split"] == "holdout"
        }
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            expected = expected_by_pair.get(str(item.get("pair_id")))
            if (
                expected is None
                or item.get("group_id") != str(expected["clean"]["group_id"])
                or item.get("split") != str(expected["clean"]["split"])
                or item.get("clean_row_id") != str(expected["clean"]["row_id"])
                or item.get("corrupted_row_id") != str(expected["corrupted"]["row_id"])
                or item.get("clean_condition") != "clean"
                or item.get("corrupted_condition") != "corrupted"
            ):
                errors.append("activation patching evidence row/group/condition linkage is invalid")
            if (
                not isinstance(item.get("clean_target_position"), int)
                or not isinstance(item.get("corrupted_target_position"), int)
                or not isinstance(item.get("clean_previous_valid_position"), int)
                or not isinstance(item.get("corrupted_previous_valid_position"), int)
                or item.get("clean_target_position", 0) < 1
                or item.get("corrupted_target_position", 0) < 1
                or item.get("clean_previous_valid_position", -1) < 0
                or item.get("corrupted_previous_valid_position", -1) < 0
                or item.get("clean_previous_valid_position", 0) >= item.get("clean_target_position", 0)
                or item.get("corrupted_previous_valid_position", 0) >= item.get("corrupted_target_position", 0)
            ):
                errors.append("activation patching previous-token positions are invalid")
        if any(str(item.get("group_id")) not in holdout_groups for item in evidence if isinstance(item, Mapping)):
            errors.append("activation patching evidence includes non-holdout groups")
        if any(
            not isinstance(item, Mapping) or any(not _finite(item.get(key)) for key in EVIDENCE_NUMERIC_FIELDS)
            for item in evidence
        ):
            errors.append("activation patching evidence contains non-finite metrics")
            continue
        for item in evidence:
            derived_recovery = _derived_recovery(item)
            if derived_recovery is None:
                errors.append("activation patching denominator is non-finite or too small")
            if derived_recovery is not None and not math.isclose(
                float(item["recovery"]), derived_recovery, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append("activation patching recovery was not recomputed from endpoints")
            if not math.isclose(
                float(item["off_target_layer_effect"]),
                float(item["off_target_layer_margin"]) - float(item["corrupted_margin"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(item["off_target_token_effect"]),
                float(item["off_target_token_margin"]) - float(item["corrupted_margin"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append("activation patching off-target effects were tampered")
            if abs(float(item["zero_strength_error"])) > ZERO_STRENGTH_IDENTITY_ATOL:
                errors.append("activation patching zero-strength identity failed")
            expected_zero_error = abs(float(item["zero_strength_margin"]) - float(item["corrupted_margin"]))
            if not math.isclose(float(item["zero_strength_error"]), expected_zero_error, rel_tol=0.0, abs_tol=1e-12):
                errors.append("activation patching zero-strength error was not recomputed")
            curve = item.get("strength_grid")
            if (
                not isinstance(curve, Mapping)
                or set(curve) != {str(value) for value in STRENGTH_GRID}
                or any(not _finite(value) for value in curve.values())
            ):
                errors.append("activation patching strength grid is invalid")
            elif not math.isclose(
                float(curve["0.0"]), float(item["corrupted_margin"]), rel_tol=0.0, abs_tol=1e-6
            ) or not math.isclose(
                float(curve["1.0"]), float(item["true_interchange_margin"]), rel_tol=0.0, abs_tol=1e-6
            ):
                errors.append("activation patching strength endpoints are not bound to controls")

        def metric_finite(value: object) -> bool:
            return bool(
                isinstance(value, Mapping)
                and _finite(value.get("point_estimate"))
                and _finite(value.get("threshold"))
                and isinstance(value.get("confidence_interval_95"), list)
                and len(value["confidence_interval_95"]) == 2
                and all(_finite(item) for item in value["confidence_interval_95"])
            )

        expected_summary_finite = (
            all(
                _finite(item.get(key))
                for item in evidence
                if isinstance(item, Mapping)
                for key in EVIDENCE_NUMERIC_FIELDS
            )
            and all(
                isinstance(item.get("strength_grid"), Mapping)
                and all(_finite(value) for value in item["strength_grid"].values())
                for item in evidence
                if isinstance(item, Mapping)
            )
            and all(metric_finite(summary.get(key)) for key in SUMMARY_METRIC_FIELDS)
        )
        if summary.get("finite") is not expected_summary_finite:
            errors.append("activation patching summary finite flag was not recomputed")
        recomputed_recovery = _recomputed_recovery(evidence, holdout_groups, int(seed))
        if (
            recomputed_recovery is None
            or not recomputed_recovery[1]
            or summary.get("recovery") != recomputed_recovery[0]
        ):
            errors.append("activation patching recovery metric was not independently recomputed")

        def group_metric(key: str, evidence_rows: list[Mapping[str, Any]], seed_value: int) -> dict[str, Any]:
            grouped: dict[str, list[float]] = {}
            for item in evidence_rows:
                grouped.setdefault(str(item["group_id"]), []).append(abs(float(item[key])))
            return _metric(
                [sum(values) / len(values) for _, values in sorted(grouped.items())],
                seed=seed_value,
                threshold=OFF_TARGET_ABSOLUTE_EFFECT_MAX,
                comparator="<=",
                units="absolute logit margin effect",
                statistic="max",
            )

        layer_metric = group_metric("off_target_layer_effect", evidence, seed)
        token_metric = group_metric("off_target_token_effect", evidence, seed)
        grouped_combined: dict[str, list[float]] = {}
        for item in evidence:
            grouped_combined.setdefault(str(item["group_id"]), []).append(
                max(abs(float(item["off_target_layer_effect"])), abs(float(item["off_target_token_effect"])))
            )
        combined_metric = _metric(
            [sum(values) / len(values) for _, values in sorted(grouped_combined.items())],
            seed=seed,
            threshold=OFF_TARGET_ABSOLUTE_EFFECT_MAX,
            comparator="<=",
            units="absolute logit margin effect",
            statistic="max",
        )
        off = summary.get("off_target")
        if (
            off != combined_metric
            or summary.get("off_target_layer") != layer_metric
            or summary.get("off_target_token") != token_metric
        ):
            errors.append("activation patching combined off-target maximum failed")
        grouped_zero: dict[str, list[float]] = {}
        for item in evidence:
            grouped_zero.setdefault(str(item["group_id"]), []).append(abs(float(item["zero_strength_error"])))
        zero_metric = _metric(
            [sum(values) / len(values) for _, values in sorted(grouped_zero.items())],
            seed=seed,
            threshold=ZERO_STRENGTH_IDENTITY_ATOL,
            comparator="<=",
            units="absolute logit margin difference",
            statistic="max",
        )
        if summary.get("zero_strength") != zero_metric:
            errors.append("activation patching zero-strength metric was not independently recomputed")
        shuffled = summary.get("shuffled_direction")
        expected_shuffled_finite = all(
            _finite(item.get("shuffled_donor_effect")) for item in evidence if isinstance(item, Mapping)
        )
        if (
            not isinstance(shuffled, Mapping)
            or shuffled.get("semantic") != "shuffled donor activation; compatibility key retained"
            or shuffled.get("finite") is not expected_shuffled_finite
        ):
            errors.append("activation patching shuffled donor control linkage is invalid")
        mapping = None if not isinstance(shuffled, Mapping) else shuffled.get("mapping")
        if (
            not isinstance(mapping, Mapping)
            or dict(mapping) != expected_mapping
            or not isinstance(shuffled, Mapping)
            or shuffled.get("mapping_sha256") != donor_mapping_digest(expected_mapping)
        ):
            errors.append("activation patching shuffled donor mapping is not a non-self derangement")
        elif any(
            str(mapping[source]) not in expected_mapping
            or str(source) == str(mapping[source])
            or frozen_pairs[str(source)]["clean"]["split"] != frozen_pairs[str(mapping[source])]["clean"]["split"]
            for source in expected_mapping
        ):
            errors.append("activation patching shuffled donor mapping crosses split or has invalid range")
    controls = entry.get("controls")
    if not isinstance(controls, Mapping) or any(
        not isinstance(controls.get(name), Mapping) or controls.get(name, {}).get("pass") is not True
        for name in (
            "clean_endpoint",
            "corrupted_endpoint",
            "true_interchange",
            "off_target_layer",
            "off_target_token",
            "off_target_combined",
            "shuffled_direction",
            "zero_strength",
        )
    ):
        errors.append("activation patching controls are missing or failing")
    metrics = entry.get("metrics")
    expected_recovery_metrics = {
        str(summary["seed"]): summary["recovery"] for summary in summaries if isinstance(summary, Mapping)
    }
    if not isinstance(metrics, Mapping) or metrics.get("recovery") != expected_recovery_metrics:
        errors.append("activation patching metrics are not linked to validated summaries")
    if isinstance(controls, Mapping):
        expected_passes = {
            "clean_endpoint": all(
                _finite(item.get("clean_margin"))
                for summary in summaries
                if isinstance(summary, Mapping)
                for item in summary.get("holdout_evidence", [])
                if isinstance(item, Mapping)
            ),
            "corrupted_endpoint": all(
                _finite(item.get("corrupted_margin"))
                for summary in summaries
                if isinstance(summary, Mapping)
                for item in summary.get("holdout_evidence", [])
                if isinstance(item, Mapping)
            ),
            "true_interchange": all(summary.get("recovery", {}).get("pass") is True for summary in summaries),
            "off_target_layer": all(summary.get("off_target_layer", {}).get("pass") is True for summary in summaries),
            "off_target_token": all(summary.get("off_target_token", {}).get("pass") is True for summary in summaries),
            "off_target_combined": all(summary.get("off_target", {}).get("pass") is True for summary in summaries),
            "shuffled_direction": all(
                summary.get("shuffled_direction", {}).get("finite") is True for summary in summaries
            ),
            "zero_strength": all(summary.get("zero_strength", {}).get("pass") is True for summary in summaries),
        }
        for control_name, expected_pass in expected_passes.items():
            actual_control = controls.get(control_name, {})
            if not isinstance(actual_control, Mapping) or actual_control.get("pass") is not expected_pass:
                errors.append(f"activation patching {control_name} pass flag was not recomputed")
        for control_name, summary_key in (
            ("off_target_layer", "off_target_layer"),
            ("off_target_token", "off_target_token"),
            ("off_target_combined", "off_target"),
        ):
            expected = {
                str(summary["seed"]): summary[summary_key] for summary in summaries if isinstance(summary, Mapping)
            }
            actual = controls.get(control_name, {})
            if not isinstance(actual, Mapping) or actual.get("metrics") != expected:
                errors.append(f"activation patching {control_name} metrics are not independently linked")
    provenance = entry.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("network") != "enabled"
        or provenance.get("execution_backend") != "cuda"
        or provenance.get("stage") != "complete"
        or provenance.get("deterministic_algorithms") is not True
    ):
        errors.append("activation patching runtime provenance is invalid")
    else:
        errors.extend(_resource_errors(entry, provenance))
        artifact_provenance = artifact.get("provenance")
        if not isinstance(artifact_provenance, Mapping) or any(
            artifact_provenance.get(field) != provenance.get(field)
            for field in ("model_parameter_digest_before", "model_parameter_digest_after", "resource_peak")
        ):
            errors.append("activation patching artifact provenance is not bound to execution provenance")
        if provenance.get("off_target_controls") != {"layer": OFF_TARGET_LAYER, "token": "previous valid token"}:
            errors.append("activation patching off-target control definition is invalid")
        if provenance.get("strength_grid") != list(STRENGTH_GRID):
            errors.append("activation patching strength grid provenance is invalid")

    def summary_acceptance(value: object) -> bool:
        if not isinstance(value, Mapping):
            return False
        recovery = value.get("recovery")
        off_target = value.get("off_target")
        zero_strength = value.get("zero_strength")
        shuffled = value.get("shuffled_direction")
        return bool(
            isinstance(recovery, Mapping)
            and isinstance(recovery.get("confidence_interval_95"), list)
            and len(recovery["confidence_interval_95"]) == 2
            and _finite(recovery["confidence_interval_95"][0])
            and float(recovery["confidence_interval_95"][0]) > RECOVERY_CI_LOWER_THRESHOLD
            and isinstance(off_target, Mapping)
            and _finite(off_target.get("point_estimate"))
            and float(off_target["point_estimate"]) <= OFF_TARGET_ABSOLUTE_EFFECT_MAX
            and isinstance(zero_strength, Mapping)
            and zero_strength.get("pass") is True
            and value.get("finite") is True
            and isinstance(shuffled, Mapping)
            and shuffled.get("finite") is True
        )

    peak = provenance.get("resource_peak") if isinstance(provenance, Mapping) else None
    expected_acceptance = bool(
        all(summary_acceptance(summary) for summary in summaries)
        and entry.get("no_mutation") is True
        and isinstance(peak, Mapping)
        and budget_pass(peak)
    )
    if entry.get("evidence_eligible") is not expected_acceptance or entry.get("acceptance") is not expected_acceptance:
        errors.append("activation patching acceptance was not bound to recomputed gates")
    return errors


def validate_real_true_activation_patching_execution(
    entry: dict[str, Any], artifact: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    try:
        return validate_real_activation_patching_execution(entry, artifact, plan)
    except Exception as exc:  # noqa: BLE001
        return [f"real activation patching evidence is malformed: {type(exc).__name__}"]


__all__ = [
    "validate_completed_activation_failure_structure",
    "REAL_ACTIVATION_PATCHING_STATUS",
    "validate_real_activation_patching_execution",
    "validate_real_true_activation_patching_execution",
]
