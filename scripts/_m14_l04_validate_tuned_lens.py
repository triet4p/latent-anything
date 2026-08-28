"""Independent, fail-closed validation for the tuned-lens execution payload."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from scripts._m14_l04_tuned_lens import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEEDS,
    FIT_SEED,
    FITTED_LAYERS,
    MAX_ELAPSED_SECONDS,
    MAX_GPU_BYTES,
    MAX_RSS_BYTES,
    NATIVE_LAYERS,
)
from scripts._m14_l04_tuned_lens_metrics import improvement_metric
from scripts._m14_l04_wikitext_runtime import read_manifest


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _same(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return actual == expected


def _metric_matches(actual: object, expected: Mapping[str, Any], label: str, errors: list[str]) -> None:
    if (
        not isinstance(actual, Mapping)
        or set(actual) != set(expected)
        or any(not _same(actual.get(field), expected.get(field)) for field in expected)
    ):
        errors.append(f"real tuned lens {label} metric is not independently recomputed")


def _control_metric(value: float, *, threshold: float, comparator: str) -> dict[str, Any]:
    passed = value <= threshold if comparator == "<=" else value > threshold
    return {
        "point_estimate": value,
        "confidence_interval_95": [value, value],
        "units": "logits",
        "aggregation_unit": "all validation tokens and vocabulary coordinates",
        "statistic": "global_max",
        "threshold": threshold,
        "comparator": comparator,
        "pass": bool(passed),
    }


def _identity_list(value: object, label: str, errors: list[str]) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        errors.append(f"{label} identity list is invalid")
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _validate_real_tuned_lens_execution(
    entry: Mapping[str, Any], artifact: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[str]:
    """Recompute every acceptance/control value; serialized pass flags are never trusted."""
    errors: list[str] = []
    if entry.get("status") != "passed_real_cuda":
        return ["real tuned lens validation requires passed_real_cuda status"]
    if (
        entry.get("support_only") is not False
        or entry.get("evidence_eligible") is not True
        or entry.get("acceptance") is not True
        or entry.get("evidence_level") != "D3"
    ):
        errors.append("accepted tuned lens linkage is invalid")
    if (
        entry.get("layer") != 6
        or entry.get("native_hidden_state_index") != 7
        or entry.get("seed") != FIT_SEED
        or entry.get("seeds") != list(BOOTSTRAP_SEEDS)
    ):
        errors.append("tuned lens layer/seed linkage is invalid")
    raw_summaries = artifact.get("raw_summaries")
    if not isinstance(raw_summaries, list) or len(raw_summaries) != 1 or not isinstance(raw_summaries[0], Mapping):
        return errors + ["tuned lens raw summary must contain one fit/evaluation record"]
    summary = raw_summaries[0]
    expected_keys = {
        "seed",
        "fit_layers",
        "native_layers",
        "train_rows",
        "validation_rows",
        "rows",
        "train_permutation",
        "validation_permutation",
        "train_objectives",
        "shuffled_train_objectives",
        "translator_digests",
        "shuffled_translator_digests",
        "terminal_logit_max_abs_error",
        "terminal_logit_max_relative_error",
    }
    if set(summary) != expected_keys:
        errors.append("tuned lens raw summary schema is invalid")
        return errors
    if (
        summary.get("seed") != FIT_SEED
        or summary.get("fit_layers") != list(FITTED_LAYERS)
        or summary.get("native_layers") != list(NATIVE_LAYERS)
    ):
        errors.append("tuned lens layer/seed summary is invalid")
    try:
        manifest_path = Path(
            str(entry.get("provenance", {}).get("manifest_path", "artifacts/m14/l04-wikitext-2-manifest.json"))
        )
        manifest, raw_sha = read_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tuned lens manifest cannot be independently validated: {type(exc).__name__}")
        return errors
    provenance = entry.get("provenance")
    source = manifest["source"]
    if not isinstance(provenance, Mapping) or not isinstance(source, Mapping):
        errors.append("tuned lens provenance is invalid")
    else:
        required = {
            "runtime": "real TransformerLMIntegration",
            "model_id": plan.get("model", {}).get("id"),
            "model_revision": plan.get("model", {}).get("revision"),
            "dataset_id": source.get("dataset_id"),
            "dataset_config": source.get("config"),
            "dataset_revision": source.get("revision"),
            "manifest_sha256": raw_sha,
            "manifest_content_sha256": manifest.get("content_sha256"),
            "manifest_split_sha256": manifest.get("split_sha256"),
            "fit_seed": FIT_SEED,
            "bootstrap_seeds": list(BOOTSTRAP_SEEDS),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "fit_layers": list(FITTED_LAYERS),
            "native_layers": list(NATIVE_LAYERS),
            "objective": "tokenwise KL(p_true || q_translated) in nats over every non-padding position",
            "optimizer": "AdamW",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip_norm": 1.0,
            "network": "enabled",
            "deterministic_algorithms": True,
            "train_rows": 8192,
            "validation_rows": 2048,
            "official_train_rows": manifest["splits"]["train"]["official_rows"],
            "official_validation_rows": manifest["splits"]["validation"]["official_rows"],
            "model_forwards_per_corpus_batch": 1,
        }
        if any(not _same(provenance.get(key), value) for key, value in required.items()):
            errors.append("tuned lens optimizer/dataset/manifest provenance is incomplete")
        before = provenance.get("model_parameter_digest_before")
        after = provenance.get("model_parameter_digest_after")

        def valid_digest(value: object) -> bool:
            return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)

        if not valid_digest(before) or not valid_digest(after):
            errors.append("tuned lens model parameter digests are missing or malformed")
        elif before != after:
            errors.append("tuned lens model parameter digest changed during execution")
        if provenance.get("no_mutation") is not True or entry.get("no_mutation") is not True:
            errors.append("tuned lens no_mutation evidence is missing or false")
        peak = provenance.get("resource_peak")
        if not isinstance(peak, Mapping):
            errors.append("tuned lens resource peak evidence is missing")
        else:
            elapsed = peak.get("elapsed_seconds")
            allocated = peak.get("max_memory_allocated_bytes")
            reserved = peak.get("max_memory_reserved_bytes")
            rss = peak.get("max_rss_bytes")
            device = provenance.get("device")
            peak_device = peak.get("cuda_device")
            numeric_peak = (elapsed, allocated, reserved, rss)
            if any(not _finite(value) or float(cast(Any, value)) < 0 for value in numeric_peak):
                errors.append("tuned lens resource peak values are missing, nonnumeric, or negative")
            if not isinstance(device, str) or not device.strip() or peak_device != device:
                errors.append("tuned lens CUDA device provenance is incoherent")
            if all(_finite(value) and float(cast(Any, value)) >= 0 for value in numeric_peak):
                budget_expected = (
                    float(cast(Any, elapsed)) <= MAX_ELAPSED_SECONDS
                    and float(cast(Any, allocated)) <= MAX_GPU_BYTES
                    and float(cast(Any, rss)) <= MAX_RSS_BYTES
                )
                if not budget_expected:
                    errors.append("tuned lens resource measurements exceed the frozen budget caps")
                if provenance.get("budget_pass") is not budget_expected:
                    errors.append("tuned lens budget_pass is not independently recomputed")
    expected_train = manifest["splits"]["train"]["selected"]
    expected_validation = manifest["splits"]["validation"]["selected"]
    train_rows = _identity_list(summary.get("train_rows"), "train", errors)
    validation_rows = _identity_list(summary.get("validation_rows"), "validation", errors)

    def expected_identities(split: str, selected: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
        return [
            {"row_id": f"{split}:{row['index']}", "index": str(row["index"]), "text_sha256": str(row["text_sha256"])}
            for row in selected
        ]

    if train_rows != expected_identities("train", expected_train) or validation_rows != expected_identities(
        "validation", expected_validation
    ):
        errors.append("tuned lens selected row identity/order/hash is not bound to manifest")
    train_ids = {row.get("row_id") for row in train_rows}
    validation_ids = {row.get("row_id") for row in validation_rows}
    if len(train_ids) != len(train_rows) or len(validation_ids) != len(validation_rows) or train_ids & validation_ids:
        errors.append("tuned lens train/holdout rows are not unique and disjoint")
    rng = np.random.default_rng(FIT_SEED)
    if (
        summary.get("train_permutation") != rng.permutation(8192).tolist()
        or summary.get("validation_permutation") != rng.permutation(2048).tolist()
    ):
        errors.append("tuned lens shuffled permutation is not the frozen deterministic permutation")
    for digest_name in ("translator_digests", "shuffled_translator_digests"):
        digests = summary.get(digest_name)
        if (
            not isinstance(digests, Mapping)
            or set(digests) != {str(layer) for layer in FITTED_LAYERS}
            or any(not isinstance(value, str) or len(value) != 64 for value in digests.values())
        ):
            errors.append(f"tuned lens {digest_name} linkage is invalid")
    for objective_name in ("train_objectives", "shuffled_train_objectives"):
        objectives = summary.get(objective_name)
        if (
            not isinstance(objectives, Mapping)
            or set(objectives) != {str(layer) for layer in FITTED_LAYERS}
            or any(not _finite(value) or float(value) < 0 for value in objectives.values())
        ):
            errors.append(f"tuned lens {objective_name} linkage is invalid")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 2048:
        return errors + ["tuned lens validation row coverage is invalid"]
    direct = {layer: [] for layer in NATIVE_LAYERS}
    tuned = {layer: [] for layer in NATIVE_LAYERS}
    improvements: list[float] = []
    shuffled_values: list[float] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append("tuned lens validation row is invalid")
            continue
        expected_row = validation_rows[position] if position < len(validation_rows) else {}
        if position >= len(validation_rows):
            errors.append("tuned lens validation row count is inconsistent with manifest")
        if (
            row.get("row_id") != expected_row.get("row_id")
            or row.get("index") != expected_row.get("index")
            or row.get("text_sha256") != expected_row.get("text_sha256")
            or row.get("split") != "validation"
            or not isinstance(row.get("token_count"), int)
            or row.get("token_count", 0) <= 0
            or row.get("finite") is not True
        ):
            errors.append("tuned lens validation row provenance is invalid")
        row_direct, row_tuned, row_improvement = row.get("direct_kl"), row.get("tuned_kl"), row.get("improvement")
        if (
            not isinstance(row_direct, list)
            or len(row_direct) != 13
            or not all(_finite(value) and float(value) >= 0 for value in row_direct)
            or not isinstance(row_tuned, list)
            or len(row_tuned) != 13
            or not all(_finite(value) and float(value) >= 0 for value in row_tuned)
            or not isinstance(row_improvement, list)
            or len(row_improvement) != 12
            or not all(_finite(value) for value in row_improvement)
        ):
            errors.append("tuned lens KL/improvement row is invalid")
            continue
        expected_improvement = [float(row_direct[layer] - row_tuned[layer]) for layer in FITTED_LAYERS]
        expected_macro = sum(expected_improvement) / 12
        if (
            not _same(row_improvement, expected_improvement)
            or not _same(row.get("macro_improvement"), expected_macro)
            or not _finite(row.get("shuffled_macro_improvement"))
        ):
            errors.append("tuned lens row improvement is not recomputed")
        for layer in NATIVE_LAYERS:
            direct[layer].append(float(row_direct[layer]))
            tuned[layer].append(float(row_tuned[layer]))
        macro_value = row.get("macro_improvement")
        shuffled_value = row.get("shuffled_macro_improvement")
        if not _finite(macro_value) or not _finite(shuffled_value):
            errors.append("tuned lens row macro values are missing or nonnumeric")
            continue
        improvements.append(float(cast(Any, macro_value)))
        shuffled_values.append(float(cast(Any, shuffled_value)))
    threshold = float(
        plan.get("thresholds_and_controls", {}).get("lens", {}).get("tuned_holdout_kl_improvement_strict_gt_nats", 0.01)
    )
    if len(improvements) == 2048:
        point_expected = improvement_metric(improvements, seed=FIT_SEED, threshold=threshold)
        seed_expected = {
            str(seed): improvement_metric(improvements, seed=seed, threshold=threshold) for seed in BOOTSTRAP_SEEDS
        }
        minimum_lower = min(float(value["confidence_interval_95"][0]) for value in seed_expected.values())
        lower_expected = {
            **point_expected,
            "point_estimate": minimum_lower,
            "confidence_interval_95": [minimum_lower, minimum_lower],
            "pass": bool(minimum_lower > threshold),
        }
        metrics = entry.get("metrics")
        if not isinstance(metrics, Mapping):
            errors.append("tuned lens metrics schema is invalid")
        else:
            _metric_matches(metrics.get("tuned_holdout_kl_improvement"), point_expected, "holdout improvement", errors)
            _metric_matches(
                metrics.get("tuned_holdout_calibration_ci_lower"), lower_expected, "calibration lower", errors
            )
        if entry.get("confidence_intervals") != seed_expected:
            errors.append("tuned lens all five bootstrap estimates are not recomputed")
    controls = entry.get("controls")
    if not isinstance(controls, Mapping):
        errors.append("tuned lens controls are missing")
    else:
        expected_control_names = {
            "direct_lens",
            "affine_tuned_lens",
            "train_holdout_separation",
            "shuffled_translator_target",
            "terminal_post_ln_f_parity",
        }
        if set(controls) != expected_control_names:
            errors.append("tuned lens control schema is invalid")
        finite = float(
            all(_finite(item) for values in direct.values() for item in values)
            and all(_finite(item) for values in tuned.values() for item in values)
            and all(_finite(item) for item in shuffled_values)
        )
        abs_error = summary.get("terminal_logit_max_abs_error")
        rel_error = summary.get("terminal_logit_max_relative_error")
        if not _finite(abs_error) or not _finite(rel_error):
            errors.append("tuned lens terminal parity evidence is invalid")
            abs_error = float("inf")
            rel_error = float("inf")
        expected_controls = {
            "direct_lens": {"metrics": {"finite_fraction": finite}, "pass": bool(finite)},
            "affine_tuned_lens": {"metrics": {"finite_fraction": finite}, "pass": bool(finite)},
            "train_holdout_separation": {
                "metrics": {"train_rows": 8192, "validation_rows": 2048, "disjoint": True},
                "pass": len(train_rows) == 8192 and len(validation_rows) == 2048 and not train_ids & validation_ids,
            },
            "shuffled_translator_target": {
                "metrics": {
                    "finite_fraction": float(all(_finite(value) for value in shuffled_values)),
                    "permutation_sha256": hashlib.sha256(
                        np.asarray(summary["train_permutation"], dtype=np.int64).tobytes()
                    ).hexdigest(),
                },
                "pass": all(_finite(value) for value in shuffled_values),
            },
            "terminal_post_ln_f_parity": {
                "metrics": {
                    "max_abs_error": _control_metric(float(cast(Any, abs_error)), threshold=1e-6, comparator="<="),
                    "max_relative_error": _control_metric(float(cast(Any, rel_error)), threshold=1e-6, comparator="<="),
                },
                "pass": float(cast(Any, abs_error)) <= 1e-6 and float(cast(Any, rel_error)) <= 1e-6,
            },
        }
        for name, expected in expected_controls.items():
            actual = controls.get(name)
            if (
                not isinstance(actual, Mapping)
                or not _same(actual.get("metrics"), expected["metrics"])
                or actual.get("pass") is not expected["pass"]
            ):
                errors.append(f"tuned lens control {name} is not independently recomputed")
    if artifact.get("accepted_record_ids") != ["THY-T05-LOGIT-LENS-TUNED-LENS"] or artifact.get("accepted_gap_ids") != [
        "THY-T05-LOGIT-LENS-TUNED-LENS"
    ]:
        errors.append("tuned lens acceptance linkage is invalid")
    return errors


def validate_real_tuned_lens_execution(
    entry: Mapping[str, Any], artifact: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[str]:
    """Fail closed with structured errors for malformed evidence, never exceptions."""
    try:
        return _validate_real_tuned_lens_execution(entry, artifact, plan)
    except Exception as exc:  # noqa: BLE001 - malformed untrusted evidence must not escape
        return [f"tuned lens validation failed closed on malformed evidence: {type(exc).__name__}"]


__all__ = ["validate_real_tuned_lens_execution"]
