"""Fail-closed validator for the frozen L04.8 execution envelope."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts._m14_l04_contract_common import canonical_json_bytes
from scripts._m14_l04_disentanglement_metrics import (
    BOOTSTRAP_REPLICATES,
    CI_LOWER_THRESHOLD,
    FACTORS,
    HOLDOUT_GROUPS,
    POINT_THRESHOLD,
    SEEDS,
    TRAIN_GROUPS,
    brier_quality,
    deterministic_group_derangement,
    fixture_row_summary,
    mapping_digest,
    metric,
)
from scripts._m14_l04_disentanglement_runtime import (
    CONVERGENCE_GRAD_TOL,
    GPT2_VOCAB_SIZE,
    L2_C,
    LBFGS_MAX_ITER,
    LBFGS_TOLERANCE_CHANGE,
    LBFGS_TOLERANCE_GRAD,
    excluded_columns_digest,
)
from scripts._m14_l04_fixture_contract import FIXTURE_PATH, read_fixture

REAL_DISENTANGLEMENT_STATUS = "passed_real_cuda"
TARGET_TOKEN_IDS = {" true": 2081, " false": 3991}
TARGET_TOKEN_STRINGS = {" true": " true", " false": " false"}
MAX_ELAPSED_SECONDS = 1800.0
MAX_CUDA_ALLOCATED_BYTES = 6 * 1024**3
MAX_RSS_BYTES = 4 * 1024**3
MODEL_DIGEST_ALGORITHM = "sha256/canonical-ordered-named-parameters-v1"
RSS_SOURCE = "resource.getrusage(RUSAGE_SELF).ru_maxrss"
RSS_UNIT = "KiB"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _forbidden_raw(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in {"prompt", "text", "logits", "hidden", "hidden_states"} or _forbidden_raw(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_forbidden_raw(item) for item in value)
    return False


def _groups(summary: Mapping[str, Any], key: str, expected: Sequence[str]) -> bool:
    value = summary.get(key)
    return isinstance(value, Mapping) and sorted(str(item) for item in value) == sorted(expected)


def _validate_group_factor(value: Any, groups: Sequence[str]) -> bool:
    if not isinstance(value, Mapping) or sorted(str(item) for item in value) != sorted(groups):
        return False
    return all(
        isinstance(group_values, Mapping)
        and set(group_values) == set(FACTORS)
        and all(_finite(group_values[factor]) for factor in FACTORS)
        for group_values in value.values()
    )


def _close(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        return False


def _nonnegative_resource_values(values: Sequence[Any]) -> tuple[bool, tuple[float, ...]]:
    """Narrow untrusted resource values before numeric comparisons."""
    normalized: list[float] = []
    for value in values:
        if not _finite(value) or float(value) < 0.0:
            return False, ()
        normalized.append(float(value))
    return True, tuple(normalized)


def _expected_factor_permutation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries = [
        {
            "row_id": str(row["row_id"]),
            "group_id": str(row["group_id"]),
            "causal_pair_id": str(row["causal_pair_id"]),
            "condition": str(row["condition"]),
            "original_labels": {factor: int(row["factor_labels"][factor]) for factor in FACTORS},
            "swapped_labels": {
                FACTORS[0]: int(row["factor_labels"][FACTORS[1]]),
                FACTORS[1]: int(row["factor_labels"][FACTORS[0]]),
            },
        }
        for row in rows
    ]
    return {"rows": entries, "sha256": hashlib.sha256(canonical_json_bytes({"rows": entries})).hexdigest()}


def _expected_tokenization_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "row_order": [fixture_row_summary(row) for row in rows],
        "tokenizer": "pinned-gpt2",
        "vocab_size": GPT2_VOCAB_SIZE,
        "padding": "attention_mask; excluded padding tokens",
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _fixture_binding(
    plan: Mapping[str, Any], entry: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[int]], dict[str, Any], list[str]]:
    """Load and bind all frozen fixture identities needed by the validator."""
    errors: list[str] = []
    if plan.get("fixture", {}).get("split", {}).get("train_groups") != list(TRAIN_GROUPS) or plan.get(
        "fixture", {}
    ).get("split", {}).get("holdout_groups") != list(HOLDOUT_GROUPS):
        errors.append("disentanglement split groups are not the frozen g01-g08/g09-g12 order")
    _raw, fixture_rows = read_fixture(FIXTURE_PATH)
    expected_linkage = [fixture_row_summary(row) for row in fixture_rows]
    if entry.get("fixture_linkage") != expected_linkage:
        errors.append("disentanglement fixture linkage is not the authored fixture")
    expected_holdout = [fixture_row_summary(row) for row in fixture_rows if row["split"] == "holdout"]
    train_fixture = [row for row in fixture_rows if row["split"] == "train"]
    expected_train_counts = {
        factor: [sum(int(row["factor_labels"][factor]) == value for row in train_fixture) for value in (0, 1)]
        for factor in FACTORS
    }
    return (
        fixture_rows,
        expected_holdout,
        expected_train_counts,
        _expected_factor_permutation(train_fixture),
        errors,
    )


def _validate_raw_token_linkage(entry: Mapping[str, Any], fixture_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate digest-only raw-token linkage without exposing token features."""
    errors: list[str] = []
    raw_linkage = entry.get("raw_token_linkage")
    if not isinstance(raw_linkage, Mapping):
        return ["disentanglement raw-token linkage is missing"]
    if (
        raw_linkage.get("row_order") != [str(row["row_id"]) for row in fixture_rows]
        or raw_linkage.get("tokenizer") != "pinned-gpt2"
        or raw_linkage.get("vocab_size") != GPT2_VOCAB_SIZE
        or raw_linkage.get("padding") != "attention_mask; excluded padding tokens"
        or raw_linkage.get("tokenization_digest") != _expected_tokenization_digest(fixture_rows)
    ):
        errors.append("disentanglement raw-token tokenizer/row-order linkage is invalid")
    feature_matrix = raw_linkage.get("feature_matrix")
    if (
        not isinstance(feature_matrix, Mapping)
        or not _sha256(feature_matrix.get("digest"))
        or feature_matrix.get("shape") != [len(fixture_rows), GPT2_VOCAB_SIZE]
        or feature_matrix.get("dtype") != "float64"
        or feature_matrix.get("order") != "C"
        or feature_matrix.get("config") != "binary token presence; attention_mask; no padding; target IDs excluded"
    ):
        errors.append("disentanglement raw-token feature matrix linkage is invalid")
    excluded = raw_linkage.get("excluded_columns")
    if (
        not isinstance(excluded, Mapping)
        or excluded.get("token_ids") != sorted(TARGET_TOKEN_IDS.values())
        or excluded.get("digest") != excluded_columns_digest(len(fixture_rows), sorted(TARGET_TOKEN_IDS.values()))
        or excluded.get("shape") != [len(fixture_rows), len(TARGET_TOKEN_IDS)]
        or excluded.get("dtype") != "float64"
        or excluded.get("order") != "C"
        or excluded.get("all_zero") is not True
    ):
        errors.append("disentanglement raw-token excluded columns are not independently linked as zero")
    return errors


def _validate_resources_and_mutation(entry: Mapping[str, Any], provenance: Mapping[str, Any]) -> list[str]:
    """Validate accepted D2 resource caps and exact model non-mutation."""
    errors: list[str] = []
    peak = provenance.get("resource_peak")
    budget_pass = False
    if not isinstance(peak, Mapping):
        errors.append("real disentanglement resource evidence is missing")
    else:
        values = tuple(
            peak.get(name)
            for name in ("elapsed_seconds", "max_memory_allocated_bytes", "max_memory_reserved_bytes", "max_rss_bytes")
        )
        values_ok, normalized = _nonnegative_resource_values(values)
        if not values_ok:
            errors.append("real disentanglement resource evidence must be finite and nonnegative")
        elif peak.get("rss_source") != RSS_SOURCE or peak.get("rss_unit") != RSS_UNIT:
            errors.append("accepted disentanglement evidence must use normalized process peak RSS")
        else:
            elapsed, allocated, reserved, rss = normalized
            budget_pass = bool(
                elapsed <= MAX_ELAPSED_SECONDS
                and allocated <= MAX_CUDA_ALLOCATED_BYTES
                and reserved <= MAX_CUDA_ALLOCATED_BYTES
                and rss <= MAX_RSS_BYTES
            )
            if not budget_pass:
                errors.append("real disentanglement frozen resource budget failed")
        if peak.get("cuda_device") != provenance.get("device"):
            errors.append("real disentanglement CUDA device/resource linkage is invalid")
    if entry.get("budget_pass") is not budget_pass or provenance.get("budget_pass") is not budget_pass:
        errors.append("real disentanglement budget pass flag was not recomputed")
    before = entry.get("model_parameter_digest_before")
    after = entry.get("model_parameter_digest_after")
    digest_equal = _sha256(before) and _sha256(after) and before == after
    if (
        not digest_equal
        or before != provenance.get("model_parameter_digest_before")
        or after != provenance.get("model_parameter_digest_after")
        or provenance.get("model_parameter_digest_algorithm") != MODEL_DIGEST_ALGORITHM
    ):
        errors.append("real disentanglement model parameter digest provenance is invalid")
    if entry.get("no_mutation") is not digest_equal:
        errors.append("real disentanglement no_mutation was not derived from exact digests")
    return errors


def _recompute_factor_quality(
    evidence: Sequence[Mapping[str, Any]], method: str, groups: Sequence[str]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {group: {} for group in groups}
    for factor in FACTORS:
        by_group: dict[str, list[float]] = {group: [] for group in groups}
        for row in evidence:
            group = str(row["group_id"])
            probability = float(row["predicted_probabilities"][method][factor])
            label = int(row["true_labels"][factor])
            by_group[group].append(brier_quality([probability], [label]))
        for group in groups:
            result[group][factor] = float(np.mean(by_group[group]))
    return result


def _recompute_control_flags(
    evidence: Sequence[Mapping[str, Any]], fit_metadata: Any, expected_train_counts: Mapping[str, list[int]]
) -> dict[str, bool]:
    """Recompute diagnostic controls from retained evidence and fit metadata."""
    flags = {"group_preserving_shuffle": True, "factor_permutation": True, "raw_token_baseline": True}
    for method in ("factor_permutation", "raw_token"):
        flags[method if method == "factor_permutation" else "raw_token_baseline"] = all(
            isinstance(row.get("predicted_probabilities"), Mapping)
            and isinstance(row["predicted_probabilities"].get(method), Mapping)
            and all(_finite(row["predicted_probabilities"][method].get(factor)) for factor in FACTORS)
            for row in evidence
        )
    if not isinstance(fit_metadata, Mapping):
        return flags
    real = fit_metadata.get("real")
    shuffled = fit_metadata.get("shuffled")
    if not isinstance(real, Mapping) or not isinstance(shuffled, Mapping):
        flags["group_preserving_shuffle"] = False
        return flags
    flags["group_preserving_shuffle"] = all(
        isinstance(real.get(factor), Mapping)
        and isinstance(shuffled.get(factor), Mapping)
        and shuffled[factor].get("class_counts") == real[factor].get("class_counts") == expected_train_counts[factor]
        for factor in FACTORS
    )
    return flags


def _validate_fit_metadata(fit_metadata: Any, expected_train_counts: Mapping[str, list[int]]) -> tuple[list[str], bool]:
    """Validate each probe's train-only metadata and class-count linkage."""
    errors: list[str] = []
    group_preserving = True
    if not isinstance(fit_metadata, Mapping) or set(fit_metadata) != {
        "real",
        "shuffled",
        "factor_permutation",
        "raw_token",
    }:
        return ["disentanglement probe metadata is missing"], False
    for fit_name, by_factor in fit_metadata.items():
        if not isinstance(by_factor, Mapping) or set(by_factor) != set(FACTORS):
            errors.append("disentanglement probe factor metadata is malformed")
            continue
        for factor in FACTORS:
            meta = by_factor[factor]
            if (
                not isinstance(meta, Mapping)
                or not isinstance(meta.get("class_counts"), list)
                or len(meta["class_counts"]) != 2
                or not all(isinstance(value, int) and value > 0 for value in meta["class_counts"])
                or not isinstance(meta.get("feature_dim"), int)
                or meta["feature_dim"] <= 0
                or not isinstance(meta.get("standardization_sha256"), str)
                or len(meta["standardization_sha256"]) != 64
                or not isinstance(meta.get("probe_sha256"), str)
                or len(meta["probe_sha256"]) != 64
            ):
                errors.append(f"disentanglement {fit_name}/{factor} metadata is malformed")
        if fit_name == "raw_token" and any(
            isinstance(by_factor.get(factor), Mapping) and by_factor[factor].get("feature_dim") != GPT2_VOCAB_SIZE
            for factor in FACTORS
        ):
            errors.append("disentanglement raw-token baseline vocabulary is not pinned GPT-2")
    real = fit_metadata.get("real")
    shuffled = fit_metadata.get("shuffled")
    permutation = fit_metadata.get("factor_permutation")
    if not (
        isinstance(real, Mapping)
        and isinstance(shuffled, Mapping)
        and isinstance(permutation, Mapping)
        and all(isinstance(real.get(factor), Mapping) for factor in FACTORS)
        and all(isinstance(shuffled.get(factor), Mapping) for factor in FACTORS)
        and all(isinstance(permutation.get(factor), Mapping) for factor in FACTORS)
    ):
        return errors, False
    real_factors = {factor: real[factor] for factor in FACTORS}
    shuffled_factors = {factor: shuffled[factor] for factor in FACTORS}
    permutation_factors = {factor: permutation[factor] for factor in FACTORS}
    if real_factors["animal_cat"].get("class_counts") != expected_train_counts["animal_cat"]:
        errors.append("disentanglement real animal class counts are not fixture-linked")
    if real_factors["tone_positive"].get("class_counts") != expected_train_counts["tone_positive"]:
        errors.append("disentanglement real tone class counts are not fixture-linked")
    for fit_name in ("shuffled", "raw_token"):
        by_factor = fit_metadata.get(fit_name)
        if not isinstance(by_factor, Mapping) or any(
            not isinstance(by_factor.get(factor), Mapping)
            or by_factor[factor].get("class_counts") != expected_train_counts[factor]
            for factor in FACTORS
        ):
            errors.append(f"disentanglement {fit_name} class counts changed")
    if (
        permutation_factors["animal_cat"].get("class_counts") != expected_train_counts["tone_positive"]
        or permutation_factors["tone_positive"].get("class_counts") != expected_train_counts["animal_cat"]
    ):
        errors.append("disentanglement factor permutation counts are invalid")
    group_preserving = all(
        shuffled_factors[factor].get("class_counts") == real_factors[factor].get("class_counts") for factor in FACTORS
    )
    return errors, group_preserving


def _validate_probability_evidence(
    summary: Mapping[str, Any], evidence: Sequence[Any], expected_holdout: Sequence[Mapping[str, Any]]
) -> tuple[list[str], bool]:
    """Validate row identity/probabilities and the deterministic repeat control."""
    errors: list[str] = []
    methods = ("real", "shuffled", "factor_permutation", "raw_token")
    if [item.get("fixture_row_linkage") for item in evidence if isinstance(item, Mapping)] != list(expected_holdout):
        errors.append("disentanglement per-row evidence does not bind the authored holdout rows")
    for item in evidence:
        if not isinstance(item, Mapping) or set(item.get("true_labels", {})) != set(FACTORS):
            errors.append("disentanglement row evidence labels are malformed")
            continue
        linkage = item.get("fixture_row_linkage")
        if (
            not isinstance(linkage, Mapping)
            or any(
                item.get(field) != linkage.get(field) for field in ("row_id", "group_id", "causal_pair_id", "condition")
            )
            or item.get("true_labels") != linkage.get("factor_labels")
        ):
            errors.append("disentanglement row evidence identity/labels are not fixture-linked")
        predicted = item.get("predicted_probabilities")
        if not isinstance(predicted, Mapping) or set(predicted) != set(methods):
            errors.append("disentanglement row evidence methods are malformed")
            continue
        for method_name in methods:
            values = predicted.get(method_name)
            if (
                not isinstance(values, Mapping)
                or set(values) != set(FACTORS)
                or not all(_finite(values[factor]) and 0.0 <= float(values[factor]) <= 1.0 for factor in FACTORS)
            ):
                errors.append("disentanglement row probabilities are malformed")
    repeated = summary.get("seeded_repeat_probabilities")
    repeated_ok = isinstance(repeated, Mapping) and all(
        isinstance(repeated.get(factor), list)
        and len(repeated[factor]) == len(evidence)
        and all(
            isinstance(evidence[index], Mapping)
            and isinstance(evidence[index].get("predicted_probabilities"), Mapping)
            and isinstance(evidence[index]["predicted_probabilities"].get("real"), Mapping)
            and _close(repeated[factor][index], evidence[index]["predicted_probabilities"]["real"][factor])
            for index in range(len(evidence))
        )
        for factor in FACTORS
    )
    if not repeated_ok:
        errors.append("disentanglement seeded repeat does not exactly reproduce real probabilities")
    return errors, repeated_ok


def _validate_summary_metrics(
    summary: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    expected_groups: Sequence[str],
    expected_factor_permutation: Mapping[str, Any],
    gain_threshold: float,
    ci_threshold: float,
) -> tuple[list[str], dict[str, dict[str, dict[str, float]]], bool]:
    """Recompute per-row, group, macro, gain, and bootstrap metrics for one seed."""
    errors: list[str] = []
    methods = ("real", "shuffled", "factor_permutation", "raw_token")
    recomputed_quality = {method: _recompute_factor_quality(evidence, method, expected_groups) for method in methods}
    quality_keys = {
        "real": "real_group_factor_quality",
        "shuffled": "shuffled_group_factor_quality",
        "factor_permutation": "factor_permutation_group_factor_quality",
        "raw_token": "raw_token_group_factor_quality",
    }
    for method_name, key in quality_keys.items():
        reported_quality = summary.get(key)
        if not isinstance(reported_quality, Mapping) or any(
            not isinstance(reported_quality.get(group), Mapping)
            or any(
                not _close(reported_quality[group].get(factor), recomputed_quality[method_name][group][factor])
                for factor in FACTORS
            )
            for group in expected_groups
        ):
            errors.append(f"disentanglement {key} was not recomputed from per-row probabilities")
    for method_name, key in (
        ("real", "real_group_quality"),
        ("shuffled", "shuffled_group_quality"),
        ("factor_permutation", "factor_permutation_group_quality"),
        ("raw_token", "raw_token_group_quality"),
    ):
        expected_macro = {
            group: float(np.mean([recomputed_quality[method_name][group][factor] for factor in FACTORS]))
            for group in expected_groups
        }
        reported_macro = summary.get(key)
        if not isinstance(reported_macro, Mapping) or any(
            not _close(reported_macro.get(group), expected_macro[group]) for group in expected_groups
        ):
            errors.append(f"disentanglement {key} macro aggregation is invalid")
    if not _groups(summary, "real_group_quality", expected_groups):
        errors.append("disentanglement heldout groups are not frozen")
    for key in quality_keys.values():
        if not _validate_group_factor(summary.get(key), expected_groups):
            errors.append(f"disentanglement {key} is malformed")
    gains = summary.get("gain_by_group")
    if (
        not isinstance(gains, Mapping)
        or sorted(str(item) for item in gains) != sorted(expected_groups)
        or not all(_finite(item) for item in gains.values())
    ):
        errors.append("disentanglement group gains are malformed")
        return errors, recomputed_quality, False
    expected_mapping = deterministic_group_derangement(list(TRAIN_GROUPS), int(summary["seed"]))
    if summary.get("shuffled_group_mapping") != expected_mapping:
        errors.append("disentanglement shuffle mapping is not deterministic")
    if summary.get("shuffled_mapping_sha256") != mapping_digest(expected_mapping):
        errors.append("disentanglement shuffle mapping digest is invalid")
    if summary.get("factor_permutation") != {"swapped_factors": list(FACTORS)}:
        errors.append("disentanglement factor permutation linkage is invalid")
    if summary.get("factor_permutation_supervision") != expected_factor_permutation:
        errors.append("disentanglement factor permutation supervision mapping is invalid")
    if any(
        item["source_group"] == item["target_group"] or item.get("slot_reversal") is not True
        for item in expected_mapping
    ):
        errors.append("disentanglement shuffle is not a reversed slot derangement")
    expected_gain = {
        group: recomputed_quality["real"][group]["animal_cat"] / 2.0
        + recomputed_quality["real"][group]["tone_positive"] / 2.0
        - recomputed_quality["shuffled"][group]["animal_cat"] / 2.0
        - recomputed_quality["shuffled"][group]["tone_positive"] / 2.0
        for group in expected_groups
    }
    if any(not _close(gains[group], expected_gain[group]) for group in expected_groups):
        errors.append("disentanglement group gains are not real-minus-shuffled macro quality")
    recomputed = metric(
        [float(expected_gain[group]) for group in expected_groups],
        seed=int(summary["seed"]),
        point_threshold=gain_threshold,
        ci_lower_threshold=ci_threshold,
    )
    reported = summary.get("heldout_gain")
    if not isinstance(reported, Mapping):
        errors.append("disentanglement heldout gain metric is missing")
    else:
        for field in ("point_estimate", "threshold", "ci_lower_threshold"):
            if not _finite(reported.get(field)):
                errors.append(f"disentanglement metric {field} is non-finite")
        if reported.get("threshold") != gain_threshold or reported.get("ci_lower_threshold") != ci_threshold:
            errors.append("disentanglement metric thresholds are not frozen")
        if not np.isclose(
            float(reported.get("point_estimate", np.nan)), recomputed["point_estimate"], rtol=0.0, atol=1e-12
        ):
            errors.append("disentanglement heldout gain point was tampered")
        if reported.get("confidence_interval_95") != recomputed["confidence_interval_95"]:
            errors.append("disentanglement bootstrap interval was tampered")
        if reported.get("pass") is not bool(recomputed["pass"]):
            errors.append("disentanglement metric pass flag was not recomputed")
        if not bool(reported.get("pass")):
            errors.append("disentanglement frozen per-seed gate failed")
    return errors, recomputed_quality, True


def _validate_real_disentanglement_execution(
    entry: dict[str, Any], artifact: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if entry.get("status") != REAL_DISENTANGLEMENT_STATUS:
        return errors
    if entry.get("support_only") is not False or entry.get("evidence_level") != "D2":
        errors.append("real disentanglement must be non-support D2 evidence")
    if entry.get("evidence_eligible") is not True or entry.get("acceptance") is not True:
        errors.append("real disentanglement must be eligible and accepted")
    if entry.get("layer") != 6 or entry.get("native_hidden_state_index") != 7:
        errors.append("real disentanglement layer/native index linkage is invalid")
    token_ids = entry.get("token_ids")
    token_strings = entry.get("target_token_strings")
    if token_ids != TARGET_TOKEN_IDS or token_strings != TARGET_TOKEN_STRINGS:
        errors.append("real disentanglement target token linkage is invalid")
    summaries = entry.get("raw_summaries")
    if not isinstance(summaries, list) or [item.get("seed") for item in summaries if isinstance(item, Mapping)] != list(
        SEEDS
    ):
        errors.append("real disentanglement must retain one summary per frozen seed")
        return errors
    if any(not isinstance(item, Mapping) or _forbidden_raw(item) for item in summaries):
        errors.append("real disentanglement raw summaries retain forbidden model/prompt data")
    thresholds = plan.get("thresholds_and_controls", {}).get("disentanglement", {})
    gain_threshold = float(thresholds.get("heldout_gain_over_shuffled_min", POINT_THRESHOLD))
    ci_threshold = float(thresholds.get("bootstrap_ci_lower_min", CI_LOWER_THRESHOLD))
    expected_groups = list(HOLDOUT_GROUPS)
    (
        fixture_rows,
        expected_holdout,
        expected_train_counts,
        expected_factor_permutation,
        fixture_errors,
    ) = _fixture_binding(plan, entry)
    errors.extend(fixture_errors)
    recomputed_controls = {
        "group_preserving_shuffle": True,
        "factor_permutation": True,
        "raw_token_baseline": True,
        "seeded_repeat": True,
    }
    for summary in summaries:
        if not isinstance(summary, Mapping):
            errors.append("disentanglement seed summary is not an object")
            continue
        if summary.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
            errors.append("disentanglement bootstrap replicate count is not frozen")
        if summary.get("train_groups") != list(TRAIN_GROUPS) or summary.get("holdout_groups") != list(HOLDOUT_GROUPS):
            errors.append("disentanglement seed group linkage is not frozen")
        evidence = summary.get("holdout_evidence")
        if not isinstance(evidence, list):
            errors.append("disentanglement per-row evidence does not bind the authored holdout rows")
            continue
        probability_errors, repeated_ok = _validate_probability_evidence(summary, evidence, expected_holdout)
        errors.extend(probability_errors)
        if not repeated_ok:
            recomputed_controls["seeded_repeat"] = False
        metric_errors, _recomputed_quality, metrics_valid = _validate_summary_metrics(
            summary,
            evidence,
            expected_groups,
            expected_factor_permutation,
            gain_threshold,
            ci_threshold,
        )
        errors.extend(metric_errors)
        if not metrics_valid:
            continue
        if (
            summary.get("seeded_repeat_exact") is not True
            or summary.get("finite") is not True
            or summary.get("factor_count_preserved") is not True
        ):
            errors.append("disentanglement controls are not passing")
        fit_metadata = summary.get("fit_metadata")
        fit_errors, fit_group_preserving = _validate_fit_metadata(fit_metadata, expected_train_counts)
        errors.extend(fit_errors)
        if not fit_group_preserving:
            recomputed_controls["group_preserving_shuffle"] = False
        control_flags = _recompute_control_flags(evidence, fit_metadata, expected_train_counts)
        for name, value in control_flags.items():
            recomputed_controls[name] = recomputed_controls[name] and value
    errors.extend(_validate_raw_token_linkage(entry, fixture_rows))
    metrics = entry.get("metrics", {}).get("heldout_gain_over_shuffled")
    if not isinstance(metrics, Mapping) or set(metrics) != {str(seed) for seed in SEEDS}:
        errors.append("disentanglement metrics are not linked for every frozen seed")
    else:
        for summary in summaries:
            reported = metrics.get(str(summary["seed"]))
            if reported != summary.get("heldout_gain"):
                errors.append("disentanglement metric/summary linkage is invalid")
    if entry.get("confidence_intervals") != {
        str(summary["seed"]): summary["heldout_gain"]["confidence_interval_95"] for summary in summaries
    }:
        errors.append("disentanglement confidence interval linkage is invalid")
    controls = entry.get("controls")
    if not isinstance(controls, Mapping) or any(
        controls.get(name, {}).get("pass") is not recomputed_controls[name] for name in recomputed_controls
    ):
        errors.append("disentanglement controls are missing or trusted without recomputation")
    elif controls.get("raw_token_baseline", {}).get("vocab_size") != GPT2_VOCAB_SIZE:
        errors.append("disentanglement raw-token control vocabulary linkage is invalid")
    provenance = entry.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("real disentanglement runtime provenance is invalid")
    else:
        if (
            provenance.get("network") != "enabled"
            or provenance.get("deterministic_algorithms") is not True
            or provenance.get("execution_attempted") is not True
            or provenance.get("execution_backend") != "cuda"
            or provenance.get("stage") != "complete"
        ):
            errors.append("real disentanglement runtime provenance is invalid")
        errors.extend(_validate_resources_and_mutation(entry, provenance))
        if (
            provenance.get("target_token_ids") != TARGET_TOKEN_IDS
            or provenance.get("target_token_strings") != TARGET_TOKEN_STRINGS
        ):
            errors.append("real disentanglement provenance target-token linkage is invalid")
        if provenance.get("raw_token_excluded_ids") != sorted(TARGET_TOKEN_IDS.values()):
            errors.append("real disentanglement raw-token output classes were not excluded")
        expected_probe = {
            "dtype": "float64 CPU",
            "optimizer": "torch.optim.LBFGS strong_wolfe",
            "max_iter": LBFGS_MAX_ITER,
            "tolerance_grad": LBFGS_TOLERANCE_GRAD,
            "tolerance_change": LBFGS_TOLERANCE_CHANGE,
            "convergence_grad_tol": CONVERGENCE_GRAD_TOL,
            "l2_c": L2_C,
            "class_weight": "balanced",
            "standardization": "train-only; zero variance scale=1",
        }
        if provenance.get("probe") != expected_probe:
            errors.append("real disentanglement probe configuration is not frozen")
    if artifact.get("integration") != "TransformerLMIntegration" or artifact.get("adapter") != "N/A":
        errors.append("real disentanglement integration linkage is invalid")
    return errors


def validate_real_disentanglement_execution(
    entry: dict[str, Any], artifact: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    """Validate malformed evidence without allowing nested data to escape."""
    try:
        return _validate_real_disentanglement_execution(entry, artifact, plan)
    except Exception as exc:  # noqa: BLE001 - validator is a fail-closed boundary
        return [f"real disentanglement evidence is malformed: {type(exc).__name__}"]


__all__ = ["REAL_DISENTANGLEMENT_STATUS", "validate_real_disentanglement_execution"]
