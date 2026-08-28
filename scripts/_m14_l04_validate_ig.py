"""Strict schema validation for the real Integrated Gradients envelope."""

from __future__ import annotations

import math
from typing import Any, cast

from scripts._m14_l04_fixture_contract import read_fixture
from scripts.m14_l04_contract import FIXTURE_PATH

VALID_COMPARATORS = {"<=", "<", ">=", ">"}
SEEDS = [17, 29, 41, 53, 67]
CONTROL_NAMES = {"zero_baseline", "batch_mean_baseline", "random_target", "seeded_repeat", "finite/no-mutation"}
REPEAT_THRESHOLD = 1.0 - 1e-8


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _compare(point: float, comparator: str, threshold: float) -> bool:
    if comparator == "<=":
        return point <= threshold
    if comparator == "<":
        return point < threshold
    if comparator == ">=":
        return point >= threshold
    if comparator == ">":
        return point > threshold
    raise ValueError(f"unsupported comparator {comparator!r}")


def _validate_metric(
    value: object,
    label: str,
    *,
    threshold: float | None = None,
    comparator: str | None = None,
    statistic: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"real Integrated Gradients metric {label} is not an object"]
    point = value.get("point_estimate")
    interval = value.get("confidence_interval_95")
    if not _finite_number(point):
        errors.append(f"real Integrated Gradients metric {label} point estimate is not finite")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(_finite_number(item) for item in interval)
        or float(cast(float, interval[0])) > float(cast(float, interval[1]))
    ):
        errors.append(f"real Integrated Gradients metric {label} CI is not finite and ordered")
    if value.get("units") != "dimensionless":
        errors.append(f"real Integrated Gradients metric {label} units are invalid")
    if value.get("aggregation_unit") != "independent causal group":
        errors.append(f"real Integrated Gradients metric {label} aggregation unit is invalid")
    observed_comparator = value.get("comparator")
    if observed_comparator not in VALID_COMPARATORS:
        errors.append(f"real Integrated Gradients metric {label} comparator is invalid")
    observed_threshold = value.get("threshold")
    if not _finite_number(observed_threshold):
        errors.append(f"real Integrated Gradients metric {label} threshold is not finite")
    if not isinstance(value.get("pass"), bool):
        errors.append(f"real Integrated Gradients metric {label} pass is not boolean")
    if statistic is not None and value.get("statistic") != statistic:
        errors.append(f"real Integrated Gradients metric {label} statistic is invalid")
    if (
        threshold is not None
        and _finite_number(observed_threshold)
        and float(cast(float, observed_threshold)) != threshold
    ):
        errors.append(f"real Integrated Gradients metric {label} threshold is not frozen")
    if comparator is not None and observed_comparator != comparator:
        errors.append(f"real Integrated Gradients metric {label} comparator is not frozen")
    if _finite_number(point) and _finite_number(observed_threshold) and observed_comparator in VALID_COMPARATORS:
        recomputed = _compare(
            float(cast(float, point)),
            cast(str, observed_comparator),
            float(cast(float, observed_threshold)),
        )
        if value.get("pass") is not recomputed:
            errors.append(f"real Integrated Gradients metric {label} pass is inconsistent with its estimate")
    return errors


def _metric_set(
    metrics: object,
    expected: dict[str, tuple[float, str, str]],
    label: str,
) -> tuple[list[str], dict[str, bool]]:
    if not isinstance(metrics, dict) or set(metrics) != set(expected):
        return [f"real Integrated Gradients {label} metrics have the wrong schema"], {}
    errors: list[str] = []
    passes: dict[str, bool] = {}
    for name, (threshold, comparator, statistic) in expected.items():
        errors.extend(
            _validate_metric(
                metrics[name], f"{label}.{name}", threshold=threshold, comparator=comparator, statistic=statistic
            )
        )
        if isinstance(metrics[name], dict) and isinstance(metrics[name].get("pass"), bool):
            passes[name] = bool(metrics[name]["pass"])
    return errors, passes


def _control_errors(controls: object, artifact: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    if not isinstance(controls, dict) or set(controls) != CONTROL_NAMES:
        return ["real Integrated Gradients controls have the wrong schema"]
    completeness = float(thresholds["completeness_relative_error_max"])
    random_max = float(thresholds["randomized_target_cosine_max"])
    expected_main = {
        "completeness_relative_error": (completeness, "<=", "mean"),
        "step_16_vs_64_attribution_cosine": (float(thresholds["step_16_vs_64_attribution_cosine_min"]), ">", "median"),
    }
    expected = {
        "zero_baseline": ({"metrics"}, expected_main),
        "batch_mean_baseline": ({"metrics"}, {"completeness_relative_error": (completeness, "<=", "mean")}),
        "random_target": ({"metrics"}, {"attribution_cosine": (random_max, "<=", "mean")}),
        "seeded_repeat": (
            {"metrics", "repeat_count", "seeds"},
            {"attribution_cosine": (REPEAT_THRESHOLD, ">", "mean")},
        ),
        "finite/no-mutation": ({"metrics", "finite_rows", "mutated"}, {"finite_fraction": (1.0, ">=", "mean")}),
    }
    errors: list[str] = []
    for name, (required_fields, metric_specs) in expected.items():
        control = controls[name]
        if not isinstance(control, dict) or set(control) != required_fields | {"pass"}:
            errors.append(f"real Integrated Gradients control {name} has the wrong schema")
            continue
        metric_errors, passes = _metric_set(control.get("metrics"), metric_specs, f"control {name}")
        errors.extend(metric_errors)
        expected_pass = bool(passes) and all(passes.values())
        if name == "seeded_repeat":
            if control.get("repeat_count") != 2 or control.get("seeds") != SEEDS:
                errors.append("real Integrated Gradients seeded_repeat seed/count linkage is invalid")
            expected_pass = expected_pass and control.get("repeat_count") == 2 and control.get("seeds") == SEEDS
        elif name == "finite/no-mutation":
            finite_rows = control.get("finite_rows")
            expected_rows = _finite_row_count(artifact.get("raw_summaries"))
            if not isinstance(finite_rows, int) or isinstance(finite_rows, bool) or finite_rows <= 0:
                errors.append("real Integrated Gradients finite/no-mutation rows are invalid")
                expected_pass = False
            elif expected_rows is not None and finite_rows != expected_rows:
                errors.append("real Integrated Gradients finite/no-mutation rows are not linked")
                expected_pass = False
            if control.get("mutated") is not False:
                errors.append("real Integrated Gradients finite/no-mutation mutation flag is invalid")
                expected_pass = False
        if not isinstance(control.get("pass"), bool):
            errors.append(f"real Integrated Gradients control {name} pass is not boolean")
        elif control.get("pass") is not expected_pass:
            errors.append(f"real Integrated Gradients control {name} pass is inconsistent with its metrics")
    return errors


def _finite_row_count(raw_summaries: object) -> int | None:
    if not isinstance(raw_summaries, list) or not raw_summaries:
        return None
    count = 0
    for summary in raw_summaries:
        if not isinstance(summary, dict) or not isinstance(summary.get("zero_baseline"), list):
            return None
        count += len(summary["zero_baseline"])
    return count


def _validate_raw_summaries(raw_summaries: object, artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        _raw_fixture, fixture_rows = read_fixture(FIXTURE_PATH)
    except (OSError, ValueError):
        return ["real Integrated Gradients frozen fixture cannot be loaded for raw-summary validation"]
    expected = {str(row["row_id"]): (str(row["group_id"]), str(row["split"])) for row in fixture_rows}
    fixture_meta = artifact.get("fixture")
    if not isinstance(fixture_meta, dict) or fixture_meta.get("rows") != len(expected):
        errors.append("real Integrated Gradients raw-summary fixture row count is invalid")
    if (
        not isinstance(raw_summaries, list)
        or [item.get("seed") for item in raw_summaries if isinstance(item, dict)] != SEEDS
    ):
        return errors + ["real Integrated Gradients per-seed summaries are incomplete"]
    zero_keys = {
        "row_id",
        "group_id",
        "split",
        "completeness_relative_error_16",
        "completeness_relative_error_64",
        "step_16_vs_64_attribution_cosine",
        "randomized_target_attribution_cosine",
        "seeded_repeat_cosine",
        "finite",
        "no_mutation",
        "target_token_id",
        "other_token_id",
        "target_position",
    }
    batch_keys = {"row_id", "group_id", "split", "completeness_relative_error"}
    for summary in raw_summaries:
        if not isinstance(summary, dict) or set(summary) != {"seed", "zero_baseline", "batch_mean_baseline"}:
            errors.append("real Integrated Gradients per-seed summary schema is invalid")
            continue
        for field, expected_keys in (("zero_baseline", zero_keys), ("batch_mean_baseline", batch_keys)):
            values = summary[field]
            if not isinstance(values, list) or len(values) != len(expected):
                errors.append(f"real Integrated Gradients {field} row count is invalid")
                continue
            observed: dict[str, tuple[str, str]] = {}
            for item in values:
                if not isinstance(item, dict) or set(item) != expected_keys:
                    errors.append(f"real Integrated Gradients {field} row schema is invalid")
                    continue
                row_id = item.get("row_id")
                group_id = item.get("group_id")
                split = item.get("split")
                if not isinstance(row_id, str) or not isinstance(group_id, str) or not isinstance(split, str):
                    errors.append(f"real Integrated Gradients {field} row identity is invalid")
                    continue
                if row_id in observed:
                    errors.append(f"real Integrated Gradients {field} row ids are not unique")
                observed[row_id] = (group_id, split)
                if "prompt" in item:
                    errors.append(f"real Integrated Gradients {field} must not contain prompt fields")
            if observed != expected:
                errors.append(f"real Integrated Gradients {field} group coverage is invalid")
    return errors


def validate_real_ig_execution(entry: dict[str, Any], artifact: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    """Require a complete, non-leaking support-only IG execution payload."""
    errors: list[str] = []
    if artifact.get("use_case") != "IntegratedGradients":
        return ["only Integrated Gradients may use real CUDA execution status"]
    if entry.get("evidence_eligible") is not True:
        errors.append("real Integrated Gradients execution must be evidence-eligible")
    if entry.get("acceptance") is not True:
        errors.append("real Integrated Gradients execution must have acceptance=true")
    thresholds = plan["thresholds_and_controls"]["integrated_gradients"]
    metric_errors, _ = _metric_set(
        entry.get("metrics"),
        {
            "completeness_relative_error": (float(thresholds["completeness_relative_error_max"]), "<=", "mean"),
            "step_16_vs_64_attribution_cosine": (
                float(thresholds["step_16_vs_64_attribution_cosine_min"]),
                ">",
                "median",
            ),
        },
        "execution",
    )
    errors.extend(metric_errors)
    errors.extend(_control_errors(entry.get("controls"), artifact, thresholds))
    if artifact.get("accepted_record_ids") != [] or artifact.get("accepted_gap_ids") != []:
        errors.append("Integrated Gradients support-only execution cannot promote a ledger record")
    raw_summaries = artifact.get("raw_summaries")
    errors.extend(_validate_raw_summaries(raw_summaries, artifact))
    if any("prompt" in str(item).lower() for item in raw_summaries) if isinstance(raw_summaries, list) else False:
        errors.append("real Integrated Gradients summaries must not contain prompt text")
    token_ids = artifact.get("token_ids")
    if (
        not isinstance(token_ids, dict)
        or set(token_ids) != {" true", " false"}
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in token_ids.values())
    ):
        errors.append("real Integrated Gradients target token ids are missing")
    if entry.get("token_ids") != token_ids:
        errors.append("real Integrated Gradients execution token linkage is invalid")
    if entry.get("layer") != 6 or entry.get("native_hidden_state_index") != 7:
        errors.append("real Integrated Gradients layer linkage is invalid")
    if entry.get("seeds") != SEEDS:
        errors.append("real Integrated Gradients seed declaration is invalid")
    return errors


__all__ = ["validate_real_ig_execution"]
