"""Strict validation for the real M14 L04 TCAV execution payload."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from scripts._m14_l04_fixture_contract import FIXTURE_PATH, read_fixture
from scripts._m14_l04_tcav_metrics import bootstrap, corrected_empirical_p, metric, wilson_lower

SEEDS = [17, 29, 41, 53, 67]
NULL_COUNT = 99
BOOTSTRAP_REPLICATES = 2000
METRICS = {
    "heldout_accuracy": ("heldout_accuracy_min", ">"),
    "heldout_accuracy_wilson_lower": ("heldout_accuracy_wilson_lower_min", ">"),
    "bootstrap_ci_lower": ("bootstrap_ci_lower_min", ">"),
    "corrected_empirical_p": ("corrected_empirical_p_max", "<="),
    "intervention_agreement": ("intervention_agreement_min", ">"),
}
CONTROL_NAMES = {
    "shuffled_concept_labels",
    "random_concept_directions",
    "matched_norm_null",
    "off_target_target_token",
    "zero_strength_identity",
}
CONTROL_RAW_NAMES = {
    "group_ids",
    "intervention_agreement",
    "off_target_target_token",
    "zero_strength_identity",
}


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric(value: object, label: str, threshold: float, comparator: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return [f"real TCAV metric {label} is not an object"]
    point = value.get("point_estimate")
    interval = value.get("confidence_interval_95")
    if not _finite(point):
        errors.append(f"real TCAV metric {label} point estimate is not finite")
    if not isinstance(interval, list) or len(interval) != 2 or not all(_finite(item) for item in interval):
        errors.append(f"real TCAV metric {label} CI is not finite")
    elif float(interval[0]) > float(interval[1]):
        errors.append(f"real TCAV metric {label} CI is not ordered")
    if value.get("units") != "dimensionless":
        errors.append(f"real TCAV metric {label} units are invalid")
    if value.get("aggregation_unit") != "independent causal group":
        errors.append(f"real TCAV metric {label} aggregation unit is invalid")
    if value.get("statistic") != "mean":
        errors.append(f"real TCAV metric {label} statistic is invalid")
    if value.get("comparator") != comparator:
        errors.append(f"real TCAV metric {label} comparator is not frozen")
    observed_threshold = value.get("threshold")
    observed_threshold_float = float(cast(float, observed_threshold)) if _finite(observed_threshold) else 0.0
    if not _finite(observed_threshold) or observed_threshold_float != float(threshold):
        errors.append(f"real TCAV metric {label} threshold is not frozen")
    if not isinstance(value.get("pass"), bool):
        errors.append(f"real TCAV metric {label} pass is not boolean")
    if _finite(point) and value.get("comparator") == comparator and _finite(observed_threshold):
        point_float = float(cast(float, point))
        threshold_float = observed_threshold_float
        expected = {
            "<=": point_float <= threshold_float,
            "<": point_float < threshold_float,
            ">=": point_float >= threshold_float,
            ">": point_float > threshold_float,
        }.get(comparator, False)
        if value.get("pass") is not expected:
            errors.append(f"real TCAV metric {label} pass is inconsistent with its estimate")
    return errors


def _raw_errors(
    raw: object, artifact: Mapping[str, Any]
) -> tuple[list[str], list[float], list[float], list[float], list[float]]:
    errors: list[str] = []
    accuracies: list[float] = []
    scores: list[float] = []
    nulls: list[float] = []
    row_correct: list[float] = []
    family_counts = {"shuffled": 0, "random": 0, "matched": 0}
    family_norms: dict[str, list[float]] = {name: [] for name in family_counts}
    try:
        _raw, fixture_rows = read_fixture(FIXTURE_PATH)
    except Exception:  # noqa: BLE001
        return ["real TCAV frozen fixture cannot be loaded"], accuracies, scores, nulls, row_correct
    expected_groups = {str(row["group_id"]) for row in fixture_rows if row["split"] == "holdout"}
    if not isinstance(raw, list) or [item.get("seed") for item in raw if isinstance(item, Mapping)] != SEEDS:
        return ["real TCAV per-seed summaries are incomplete"], accuracies, scores, nulls, row_correct
    expected_nulls_by_seed = {seed: (19 if seed == SEEDS[-1] else 20) for seed in SEEDS}
    expected_summary_fields = {
        "seed",
        "train_groups",
        "holdout_groups",
        "heldout_group_accuracy",
        "heldout_row_correct",
        "heldout_group_tcav",
        "null_group_scores",
        "null_direction_norms",
        "null_families",
    }
    for summary_index, summary in enumerate(raw):
        if not isinstance(summary, Mapping):
            errors.append("real TCAV seed summary is not an object")
            continue
        if set(summary) != expected_summary_fields:
            errors.append("real TCAV seed summary schema is not exact")
        for field in (
            "train_groups",
            "holdout_groups",
            "heldout_group_accuracy",
            "heldout_group_tcav",
            "null_group_scores",
            "heldout_row_correct",
            "null_direction_norms",
            "null_families",
        ):
            if field not in summary:
                errors.append(f"real TCAV seed summary field {field} is missing")
        if summary.get("holdout_groups") != sorted(expected_groups):
            errors.append("real TCAV holdout group coverage is invalid")
        if summary.get("train_groups") != [f"g{i:02d}" for i in range(1, 9)]:
            errors.append("real TCAV train group coverage is invalid")
        for key, _value in summary.items():
            if "prompt" in str(key).lower():
                errors.append("real TCAV raw summaries must not contain prompt text")
        values = summary.get("heldout_row_correct")
        if not isinstance(values, list) or len(values) != 8 or not all(_finite(value) for value in values):
            errors.append("real TCAV heldout row coverage is invalid")
        elif not all(float(value) in {0.0, 1.0} for value in values):
            errors.append("real TCAV heldout row correctness must be binary")
        elif summary_index == 0:
            row_correct.extend(float(value) for value in values)
        for field, destination in (
            ("heldout_group_accuracy", accuracies),
            ("heldout_group_tcav", scores),
            ("null_group_scores", nulls),
        ):
            values = summary.get(field)
            if not isinstance(values, list) or not values or not all(_finite(value) for value in values):
                errors.append(f"real TCAV {field} values are invalid")
            elif field != "null_group_scores" and len(values) != len(expected_groups):
                errors.append(f"real TCAV {field} group count is invalid")
            valid_values = [float(value) for value in cast(list[Any], values) if _finite(value)]
            if field != "null_direction_norms" and not all(0.0 <= value <= 1.0 for value in valid_values):
                errors.append(f"real TCAV {field} values are outside [0, 1]")
            destination.extend(valid_values)
        null_values = summary.get("null_group_scores")
        null_norms = summary.get("null_direction_norms")
        null_families = summary.get("null_families")
        seed_value = summary.get("seed")
        expected_count = expected_nulls_by_seed.get(seed_value, -1) if isinstance(seed_value, int) else -1
        if not isinstance(null_values, list) or len(null_values) != expected_count:
            errors.append("real TCAV per-seed null count is invalid")
        if (
            not isinstance(null_norms, list)
            or not isinstance(null_families, list)
            or len(null_norms) != expected_count
            or len(null_families) != expected_count
            or not all(_finite(value) and float(value) > 0.0 for value in null_norms)
            or not all(value in family_counts for value in null_families)
        ):
            errors.append("real TCAV null family coverage or norms are invalid")
        else:
            for family, norm in zip(null_families, null_norms, strict=True):
                family_counts[str(family)] += 1
                family_norms[str(family)].append(float(norm))
    if len(nulls) != NULL_COUNT or family_counts != {"shuffled": 33, "random": 33, "matched": 33}:
        errors.append("real TCAV null count or family balance is not the frozen 99 (33 each)")
    if not isinstance(artifact.get("fixture"), Mapping) or artifact["fixture"].get("rows") != 24:
        errors.append("real TCAV fixture linkage is invalid")
    concept_direction_norm = artifact.get("provenance", {}).get("concept_direction_norm")
    if _finite(concept_direction_norm):
        for family in ("shuffled", "matched"):
            if family_norms[family] and not all(
                math.isclose(value, float(concept_direction_norm), rel_tol=1e-6, abs_tol=1e-6)
                for value in family_norms[family]
            ):
                errors.append(f"real TCAV {family} null norms do not match learned direction norm")
        if family_norms["random"] and all(
            math.isclose(value, float(concept_direction_norm), rel_tol=1e-6, abs_tol=1e-6)
            for value in family_norms["random"]
        ):
            errors.append("real TCAV random nulls are indistinguishable from matched-norm nulls")
    return errors, accuracies, scores, nulls, row_correct


def validate_real_tcav_execution(
    entry: Mapping[str, Any], artifact: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[str]:
    """Validate metrics, controls, coverage, and all recomputed verdict links."""
    errors: list[str] = []
    thresholds = plan["thresholds_and_controls"]["tcav"]
    if entry.get("status") != "passed_real_cuda":
        return ["real TCAV validation requires passed_real_cuda status"]
    if entry.get("evidence_eligible") is not True or entry.get("acceptance") is not True:
        errors.append("passed real TCAV execution must be eligible and accepted")
    if entry.get("layer") != 6 or entry.get("native_hidden_state_index") != 7:
        errors.append("real TCAV layer/native index linkage is invalid")
    if entry.get("seeds") != SEEDS:
        errors.append("real TCAV seeds are not the frozen five")
    metrics = entry.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRICS):
        errors.append("real TCAV metrics have the wrong schema")
    else:
        for name, (threshold_key, comparator) in METRICS.items():
            errors.extend(_metric(metrics[name], name, float(thresholds[threshold_key]), comparator))
    raw_errors, accuracies, scores, nulls, row_correct = _raw_errors(artifact.get("raw_summaries"), artifact)
    errors.extend(raw_errors)
    if accuracies and scores and nulls and row_correct and isinstance(metrics, Mapping):
        expected_wilson = wilson_lower(round(sum(row_correct)), len(row_correct))
        observed_accuracy = float(sum(accuracies) / len(accuracies))
        observed = float(sum(scores) / len(scores))
        group_count = len(plan["fixture"]["split"]["holdout_groups"])
        grouped_accuracies = [
            float(sum(accuracies[offset + index] for offset in range(0, len(accuracies), group_count)) / len(SEEDS))
            for index in range(group_count)
        ]
        grouped_scores = [
            float(sum(scores[offset + index] for offset in range(0, len(scores), group_count)) / len(SEEDS))
            for index in range(group_count)
        ]
        expected_accuracy_interval = bootstrap(grouped_accuracies, seed=SEEDS[0], replicates=BOOTSTRAP_REPLICATES)
        expected_score_interval = bootstrap(grouped_scores, seed=SEEDS[1], replicates=BOOTSTRAP_REPLICATES)
        expected_p = corrected_empirical_p(observed, nulls)
        if float(metrics["heldout_accuracy"]["point_estimate"]) != observed_accuracy:
            errors.append("real TCAV heldout accuracy is not recomputed from raw groups")
        if metrics["heldout_accuracy"]["confidence_interval_95"] != expected_accuracy_interval:
            errors.append("real TCAV heldout accuracy CI is not recomputed from raw groups")
        if float(metrics["bootstrap_ci_lower"]["point_estimate"]) != float(expected_score_interval[0]):
            errors.append("real TCAV sensitivity point estimate is not recomputed from raw groups")
        if metrics["bootstrap_ci_lower"]["confidence_interval_95"] != expected_score_interval:
            errors.append("real TCAV sensitivity CI is not recomputed from raw groups")
        if float(metrics["heldout_accuracy_wilson_lower"]["point_estimate"]) != expected_wilson:
            errors.append("real TCAV Wilson lower bound is not recomputed from raw groups")
        if float(metrics["corrected_empirical_p"]["point_estimate"]) != expected_p:
            errors.append("real TCAV corrected empirical p is not recomputed from raw nulls")
    controls = entry.get("controls")
    if not isinstance(controls, Mapping) or set(controls) != CONTROL_NAMES:
        errors.append("real TCAV controls have the wrong schema")
    else:
        for name, control in controls.items():
            if (
                not isinstance(control, Mapping)
                or not isinstance(control.get("metrics"), Mapping)
                or not isinstance(control.get("pass"), bool)
            ):
                errors.append(f"real TCAV control {name} schema is invalid")
                continue
            control_metrics = control["metrics"]
            if len(control_metrics) != 1:
                errors.append(f"real TCAV control {name} metric schema is invalid")
                continue
            metric_value = next(iter(control_metrics.values()))
            expected_control_metric = (
                "absolute_margin_difference" if name == "zero_strength_identity" else "tcav_sensitivity"
            )
            if set(control_metrics) != {expected_control_metric}:
                errors.append(f"real TCAV control {name} metric name is invalid")
            if name != "zero_strength_identity":
                reference = control.get("reference")
                if (
                    not isinstance(reference, Mapping)
                    or reference.get("metric") != "primary_sensitivity"
                    or not _finite(reference.get("value"))
                    or not scores
                    or float(reference["value"]) != float(sum(scores) / len(scores))
                ):
                    errors.append(f"real TCAV control {name} reference is not linked to primary sensitivity")
            errors.extend(
                _metric(
                    metric_value,
                    f"control {name}",
                    float(metric_value.get("threshold", 0.0))
                    if isinstance(metric_value, Mapping) and _finite(metric_value.get("threshold"))
                    else 0.0,
                    str(metric_value.get("comparator")) if isinstance(metric_value, Mapping) else "<=",
                )
            )
            if isinstance(metric_value, Mapping) and control.get("pass") is not metric_value.get("pass"):
                errors.append(f"real TCAV control {name} pass is inconsistent with its metric")
    control_raw = entry.get("control_raw")
    holdout_groups = [f"g{i:02d}" for i in range(9, 13)]
    if not isinstance(control_raw, Mapping) or set(control_raw) != CONTROL_RAW_NAMES:
        errors.append("real TCAV control raw values have the wrong schema")
    else:
        if control_raw.get("group_ids") != holdout_groups:
            errors.append("real TCAV control raw group coverage is invalid")
        for name in ("intervention_agreement", "off_target_target_token", "zero_strength_identity"):
            values = control_raw.get(name)
            if (
                not isinstance(values, list)
                or len(values) != len(holdout_groups)
                or not all(_finite(value) for value in values)
            ):
                errors.append(f"real TCAV control raw {name} coverage is invalid")
            elif name != "zero_strength_identity" and not all(0.0 <= float(value) <= 1.0 for value in values):
                errors.append(f"real TCAV control raw {name} values are outside [0, 1]")
            elif name == "zero_strength_identity" and not all(float(value) >= 0.0 for value in values):
                errors.append("real TCAV control raw zero-strength values must be non-negative")
    provenance = entry.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = artifact.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("network") != "enabled"
        or provenance.get("deterministic_algorithms") is not True
    ):
        errors.append("real TCAV runtime provenance is incomplete")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("null_count") != NULL_COUNT
        or provenance.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or provenance.get("direction_fit") != "train groups only"
        or provenance.get("concept_factor") != "tone_positive"
        or not _finite(provenance.get("primary_sensitivity"))
        or provenance.get("runtime") != "real TransformerLMIntegration"
        or provenance.get("target_position") != "last non-padding token"
    ):
        errors.append("real TCAV protocol provenance is invalid")
    if isinstance(provenance, Mapping):
        token_ids = entry.get("token_ids")
        plan_model = plan.get("model")
        if (
            provenance.get("model_id") != (plan_model.get("id") if isinstance(plan_model, Mapping) else None)
            or provenance.get("model_revision")
            != (plan_model.get("revision") if isinstance(plan_model, Mapping) else None)
            or not isinstance(token_ids, Mapping)
            or provenance.get("off_target_token_id") != token_ids.get(" false")
            or token_ids.get(" true") == token_ids.get(" false")
            or provenance.get("target_token_strings") != {" true": " true", " false": " false"}
            or provenance.get("null_family_counts") != {"shuffled": 33, "random": 33, "matched": 33}
        ):
            errors.append("real TCAV token/control provenance is invalid")
    control_raw_valid = isinstance(control_raw, Mapping) and set(control_raw) == CONTROL_RAW_NAMES and all(
        isinstance(control_raw.get(name), list)
        and len(control_raw.get(name, [])) == len(holdout_groups)
        and all(_finite(value) for value in control_raw.get(name, []))
        for name in ("intervention_agreement", "off_target_target_token", "zero_strength_identity")
    )
    if (
        control_raw_valid
        and isinstance(metrics, Mapping)
        and isinstance(controls, Mapping)
        and isinstance(provenance, Mapping)
    ):
        control_raw_map = cast(Mapping[str, Any], control_raw)
        provenance_map = cast(Mapping[str, Any], provenance)
        expected_intervention = metric(
            control_raw_map["intervention_agreement"],
            seed=41,
            threshold=float(thresholds["intervention_agreement_min"]),
            comparator=">",
        )
        _compare_metric(metrics.get("intervention_agreement"), expected_intervention, "intervention_agreement", errors)
        family_values: dict[str, list[float]] = {name: [] for name in ("shuffled", "random", "matched")}
        for summary in artifact.get("raw_summaries", []):
            if isinstance(summary, Mapping):
                for value, family in zip(
                    summary.get("null_group_scores", []), summary.get("null_families", []), strict=False
                ):
                    if family in family_values and _finite(value):
                        family_values[str(family)].append(float(value))
        control_specs = (
            ("shuffled_concept_labels", "shuffled", 17),
            ("random_concept_directions", "random", 18),
            ("matched_norm_null", "matched", 19),
        )
        for control_name, family, seed in control_specs:
            expected_control = metric(
                family_values[family],
                seed=seed,
                threshold=float(provenance_map.get("primary_sensitivity", 0.0)),
                comparator="<",
            )
            actual_control = controls.get(control_name, {})
            _compare_metric(
                actual_control.get("metrics", {}).get("tcav_sensitivity")
                if isinstance(actual_control, Mapping) and isinstance(actual_control.get("metrics"), Mapping)
                else None,
                expected_control,
                f"control {control_name}",
                errors,
            )
        for control_name, raw_name, seed in (
            ("off_target_target_token", "off_target_target_token", 20),
            ("zero_strength_identity", "zero_strength_identity", 21),
        ):
            expected_control = metric(
                control_raw_map[raw_name],
                seed=seed,
                threshold=(
                    float(provenance_map.get("primary_sensitivity", 0.0))
                    if control_name != "zero_strength_identity"
                    else 1e-6
                ),
                comparator="<" if control_name != "zero_strength_identity" else "<=",
            )
            actual_control = controls.get(control_name, {})
            expected_metric_name = (
                "tcav_sensitivity" if control_name != "zero_strength_identity" else "absolute_margin_difference"
            )
            _compare_metric(
                actual_control.get("metrics", {}).get(expected_metric_name)
                if isinstance(actual_control, Mapping) and isinstance(actual_control.get("metrics"), Mapping)
                else None,
                expected_control,
                f"control {control_name}",
                errors,
            )
    return errors


def _compare_metric(actual: object, expected: Mapping[str, Any], label: str, errors: list[str]) -> None:
    """Require every serialized metric field to match its raw-data recomputation."""
    if not isinstance(actual, Mapping):
        errors.append(f"real TCAV {label} metric is missing for recomputation")
        return
    for field in ("point_estimate", "confidence_interval_95", "threshold", "comparator", "pass"):
        if actual.get(field) != expected.get(field):
            errors.append(f"real TCAV {label} metric is not recomputed from raw values")
            break


__all__ = ["validate_real_tcav_execution"]
