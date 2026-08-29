"""Concrete M14 L04.8 disentanglement handler (real CUDA execution only)."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from scripts._m14_l04_contract_common import canonical_json_bytes
from scripts._m14_l04_digest import runtime_versions
from scripts._m14_l04_disentanglement_metrics import (
    BOOTSTRAP_REPLICATES,
    CI_LOWER_THRESHOLD,
    FACTORS,
    POINT_THRESHOLD,
    SEEDS,
    deterministic_group_derangement,
    fixture_row_summary,
    group_factor_quality,
    macro_group_quality,
    mapping_digest,
    metric,
    shuffled_labels,
)
from scripts._m14_l04_disentanglement_runtime import (
    CONVERGENCE_GRAD_TOL,
    GPT2_VOCAB_SIZE,
    L2_C,
    LBFGS_MAX_ITER,
    LBFGS_TOLERANCE_CHANGE,
    LBFGS_TOLERANCE_GRAD,
    binary_token_bow,
    excluded_columns_digest,
    fit_logistic_probe,
    matrix_digest,
    tokenizer_vocab_size,
)
from scripts._m14_l04_fixture_contract import FIXTURE_PATH, content_digest, read_fixture
from scripts._m14_l04_tcav_runtime import (
    RealExecutionError,
    capture_activations,
    parameter_digest,
    read_rows,
    resolve_target_token,
    seed_everything,
)

REAL_STATUS = "passed_real_cuda"
TARGET_TEXTS = (" true", " false")
TARGET_TOKEN_IDS = {" true": 2081, " false": 3991}
TARGET_TOKEN_STRINGS = {" true": " true", " false": " false"}
LAYER = 6
NATIVE_HIDDEN_STATE_INDEX = 7
MAX_ELAPSED_SECONDS = 1800.0
MAX_CUDA_ALLOCATED_BYTES = 6 * 1024**3
MAX_RSS_BYTES = 4 * 1024**3


def _frozen_fixture_linkage(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    raw, expected = read_fixture(FIXTURE_PATH)
    if plan["fixture"]["content_sha256"] != content_digest(raw):
        raise ValueError("plan is not linked to the frozen authored fixture")
    actual = [fixture_row_summary(row) for row in rows]
    frozen = [fixture_row_summary(row) for row in expected]
    if actual != frozen:
        raise ValueError("rows must exactly match the authored fixture order, groups, pairs, and labels")
    return actual


def _rss_measurement() -> tuple[int, str, str]:
    """Return true process peak RSS when available, normalized to bytes."""
    try:
        import resource

        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if raw < 0:
            raise ValueError("process peak RSS is negative")
        if sys.platform == "darwin":
            return raw, "resource.getrusage(RUSAGE_SELF).ru_maxrss", "bytes"
        return raw * 1024, "resource.getrusage(RUSAGE_SELF).ru_maxrss", "bytes"
    except (ImportError, OSError):
        try:
            import psutil

            return int(psutil.Process().memory_info().rss), "psutil.Process.memory_info().rss", "bytes"
        except ImportError as exc:
            raise ValueError("process peak RSS measurement is unavailable") from exc


def _rss_bytes() -> int:
    return _rss_measurement()[0]


def factor_permutation_supervision(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    entries = [
        {
            "row_id": str(row["row_id"]),
            "group_id": str(row["group_id"]),
            "causal_pair_id": str(row["causal_pair_id"]),
            "condition": str(row["condition"]),
            "original_labels": {factor: int(labels[factor][index]) for factor in FACTORS},
            "swapped_labels": {
                FACTORS[0]: int(labels[FACTORS[1]][index]),
                FACTORS[1]: int(labels[FACTORS[0]][index]),
            },
        }
        for index, row in enumerate(rows)
    ]
    return {
        "rows": entries,
        "sha256": hashlib.sha256(canonical_json_bytes({"rows": entries})).hexdigest(),
    }


def tokenization_digest(rows: Sequence[Mapping[str, Any]], *, vocab_size: int) -> str:
    payload = {
        "row_order": [fixture_row_summary(row) for row in rows],
        "tokenizer": "pinned-gpt2",
        "vocab_size": vocab_size,
        "padding": "attention_mask; excluded padding tokens",
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _resource_peak(
    torch: Any,
    device: str,
    started: float,
    peak_rss_bytes: int,
    rss_source: str,
    rss_unit: str,
) -> dict[str, Any]:
    """Finalize resource measurements at the single accepted boundary."""
    return {
        "cuda_device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "max_rss_bytes": max(peak_rss_bytes, _rss_bytes()),
        "rss_source": rss_source,
        "rss_unit": rss_unit,
    }


def budget_pass(resource_peak: Mapping[str, Any]) -> bool:
    """Apply every frozen resource cap without trusting an external flag."""
    values = tuple(
        resource_peak.get(name)
        for name in (
            "elapsed_seconds",
            "max_memory_allocated_bytes",
            "max_memory_reserved_bytes",
            "max_rss_bytes",
        )
    )
    numeric_values: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            return False
        numeric_values.append(float(value))
    elapsed, allocated, reserved, rss = numeric_values
    return bool(
        0.0 <= elapsed <= MAX_ELAPSED_SECONDS
        and 0.0 <= allocated <= MAX_CUDA_ALLOCATED_BYTES
        and 0.0 <= reserved <= MAX_CUDA_ALLOCATED_BYTES
        and 0.0 <= rss <= MAX_RSS_BYTES
    )


def _labels(rows: Sequence[Mapping[str, Any]], factor: str) -> np.ndarray:
    values = np.asarray([int(row["factor_labels"][factor]) for row in rows], dtype=np.int64)
    if np.any((values != 0) & (values != 1)):
        raise ValueError(f"factor {factor!r} is not binary")
    return values


def _fit_quality(
    train_features: np.ndarray,
    train_labels: Mapping[str, Sequence[int]],
    eval_features: np.ndarray,
    eval_rows: Sequence[Mapping[str, Any]],
    eval_labels: Mapping[str, Sequence[int]],
    *,
    torch: Any,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, list[float]], dict[str, Any]]:
    probabilities_by_factor: dict[str, np.ndarray] = {}
    fit_metadata: dict[str, Any] = {}
    for factor in FACTORS:
        probe = fit_logistic_probe(train_features, np.asarray(train_labels[factor]), torch=torch)
        probabilities_by_factor[factor] = probe.predict_proba(eval_features)
        fit_metadata[factor] = {
            "class_counts": [
                int(np.count_nonzero(np.asarray(train_labels[factor]) == 0)),
                int(np.count_nonzero(np.asarray(train_labels[factor]) == 1)),
            ],
            "feature_dim": int(train_features.shape[1]),
            "standardization_sha256": hashlib.sha256(probe.mean.tobytes() + probe.scale.tobytes()).hexdigest(),
            "probe_sha256": hashlib.sha256(
                probe.weights.tobytes() + np.asarray([probe.intercept], dtype=np.float64).tobytes()
            ).hexdigest(),
        }
    # The same predicted probability is intentionally scored against each
    # supervision column; this keeps the frozen probe concrete per factor.
    by_factor: dict[str, dict[str, float]] = {}
    for factor in FACTORS:
        probabilities = probabilities_by_factor[factor]
        labels = {factor: eval_labels[factor]}
        labels.update({other: eval_labels[other] for other in FACTORS if other != factor})
        factor_quality = group_factor_quality(eval_rows, probabilities.tolist(), labels)
        for group, values in factor_quality.items():
            by_factor.setdefault(group, {})[factor] = values[factor]
    return (
        by_factor,
        macro_group_quality(by_factor),
        {factor: probabilities_by_factor[factor].tolist() for factor in FACTORS},
        fit_metadata,
    )


def _assert_count_preservation(original: Mapping[str, Sequence[int]], shuffled: Mapping[str, Sequence[int]]) -> bool:
    return all(
        sum(int(value) for value in original[factor]) == sum(int(value) for value in shuffled[factor])
        for factor in FACTORS
    )


def _run_seed(
    seed: int,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    train_features: np.ndarray,
    holdout_features: np.ndarray,
    train_labels: Mapping[str, Sequence[int]],
    holdout_labels: Mapping[str, Sequence[int]],
    raw_train: np.ndarray,
    raw_holdout: np.ndarray,
    *,
    torch: Any,
    holdout_source: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    real_by_factor, real_group, real_probabilities, real_fit = _fit_quality(
        train_features, train_labels, holdout_features, holdout_rows, holdout_labels, torch=torch
    )
    mapping = deterministic_group_derangement(sorted({str(row["group_id"]) for row in train_rows}), seed)
    shuffled_train = shuffled_labels(train_rows, train_labels, mapping)
    if not _assert_count_preservation(train_labels, shuffled_train):
        raise ValueError("group-preserving shuffle changed factor counts")
    shuffled_by_factor, shuffled_group, shuffled_probabilities, shuffled_fit = _fit_quality(
        train_features, shuffled_train, holdout_features, holdout_rows, holdout_labels, torch=torch
    )
    permutation_train = {FACTORS[0]: list(train_labels[FACTORS[1]]), FACTORS[1]: list(train_labels[FACTORS[0]])}
    permutation_by_factor, permutation_group, permutation_probabilities, permutation_fit = _fit_quality(
        train_features, permutation_train, holdout_features, holdout_rows, holdout_labels, torch=torch
    )
    raw_by_factor, raw_group, raw_probabilities, raw_fit = _fit_quality(
        raw_train, train_labels, raw_holdout, holdout_rows, holdout_labels, torch=torch
    )
    gains = {group: float(real_group[group] - shuffled_group[group]) for group in sorted(real_group)}
    gain_values = [gains[group] for group in sorted(gains)]
    primary = metric(
        gain_values,
        seed=seed,
        point_threshold=POINT_THRESHOLD,
        ci_lower_threshold=CI_LOWER_THRESHOLD,
    )
    # Re-fitting the same real probe is a deterministic repeat control.  The
    # control compares predictions, not serialized tensors or model state.
    repeat_by_factor, _repeat_group, _repeat_probabilities, _repeat_fit = _fit_quality(
        train_features, train_labels, holdout_features, holdout_rows, holdout_labels, torch=torch
    )
    repeat_exact = bool(repeat_by_factor == real_by_factor)
    return {
        "seed": int(seed),
        "train_groups": sorted({str(row["group_id"]) for row in train_rows}),
        "holdout_groups": sorted({str(row["group_id"]) for row in holdout_rows}),
        "real_group_factor_quality": real_by_factor,
        "shuffled_group_factor_quality": shuffled_by_factor,
        "factor_permutation_group_factor_quality": permutation_by_factor,
        "raw_token_group_factor_quality": raw_by_factor,
        "real_group_quality": real_group,
        "shuffled_group_quality": shuffled_group,
        "factor_permutation_group_quality": permutation_group,
        "raw_token_group_quality": raw_group,
        "holdout_evidence": [
            {
                "row_id": str(row["row_id"]),
                "group_id": str(row["group_id"]),
                "causal_pair_id": str(row["causal_pair_id"]),
                "condition": str(row["condition"]),
                "true_labels": {factor: int(holdout_labels[factor][index]) for factor in FACTORS},
                "predicted_probabilities": {
                    "real": {factor: float(real_probabilities[factor][index]) for factor in FACTORS},
                    "shuffled": {factor: float(shuffled_probabilities[factor][index]) for factor in FACTORS},
                    "factor_permutation": {
                        factor: float(permutation_probabilities[factor][index]) for factor in FACTORS
                    },
                    "raw_token": {factor: float(raw_probabilities[factor][index]) for factor in FACTORS},
                },
                "fixture_row_linkage": fixture_row_summary(source_row),
            }
            for index, (row, source_row) in enumerate(zip(holdout_rows, holdout_source or holdout_rows, strict=True))
        ],
        "fit_metadata": {
            "real": real_fit,
            "shuffled": shuffled_fit,
            "factor_permutation": permutation_fit,
            "raw_token": raw_fit,
        },
        "gain_by_group": gains,
        "heldout_gain": primary,
        "shuffled_group_mapping": mapping,
        "shuffled_mapping_sha256": mapping_digest(mapping),
        "factor_permutation": {"swapped_factors": list(FACTORS)},
        "factor_permutation_supervision": factor_permutation_supervision(train_rows, train_labels),
        "seeded_repeat_exact": repeat_exact,
        "seeded_repeat_probabilities": _repeat_probabilities,
        "finite": bool(np.isfinite(np.asarray(gain_values)).all()),
        "factor_count_preserved": True,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _acceptance_passed(summaries: Sequence[Mapping[str, Any]]) -> bool:
    """Apply the frozen per-seed metric and control gates."""
    return all(
        summary["heldout_gain"]["point_estimate"] > POINT_THRESHOLD
        and summary["heldout_gain"]["confidence_interval_95"][0] > CI_LOWER_THRESHOLD
        and summary["seeded_repeat_exact"]
        and summary["finite"]
        and summary["factor_count_preserved"]
        for summary in summaries
    )


def _build_execution_result(
    *,
    accepted: bool,
    summaries: Sequence[Mapping[str, Any]],
    vocab_size: int,
    fixture_linkage: list[dict[str, Any]],
    token_ids: Mapping[str, int],
    token_strings: Mapping[str, str],
    raw_token_linkage: Mapping[str, Any],
    no_mutation: bool,
    before: str,
    after: str,
    within_budget: bool,
    model_spec: Mapping[str, Any],
    resources: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the sanitized result after execution and cleanup are complete."""
    return {
        "status": REAL_STATUS if accepted else "failed",
        "evidence_eligible": accepted,
        "acceptance": accepted,
        "evidence_level": "D2" if accepted else "D0",
        "failure_reason": None if accepted else "one or more frozen L04.8 gates or controls failed",
        "metrics": {"heldout_gain_over_shuffled": {str(item["seed"]): item["heldout_gain"] for item in summaries}},
        "confidence_intervals": {
            str(item["seed"]): item["heldout_gain"]["confidence_interval_95"] for item in summaries
        },
        "controls": {
            "group_preserving_shuffle": {"pass": all(item["factor_count_preserved"] for item in summaries)},
            "factor_permutation": {"pass": all(item["finite"] for item in summaries)},
            "raw_token_baseline": {"pass": all(item["finite"] for item in summaries), "vocab_size": vocab_size},
            "seeded_repeat": {"pass": all(item["seeded_repeat_exact"] for item in summaries)},
        },
        "raw_summaries": list(summaries),
        "fixture_linkage": fixture_linkage,
        "token_ids": dict(token_ids),
        "target_token_strings": dict(token_strings),
        "raw_token_linkage": dict(raw_token_linkage),
        "layer": LAYER,
        "native_hidden_state_index": NATIVE_HIDDEN_STATE_INDEX,
        "seeds": list(SEEDS),
        "no_mutation": no_mutation,
        "model_parameter_digest_before": before,
        "model_parameter_digest_after": after,
        "budget_pass": within_budget,
        "provenance": {
            "runtime": "real TransformerLMIntegration",
            "model_revision": str(model_spec["revision"]),
            "target_token_ids": dict(token_ids),
            "target_token_strings": dict(token_strings),
            "raw_token_excluded_ids": sorted(token_ids.values()),
            "target_position": "last non-padding token",
            "network": "enabled",
            "device": resources["device"],
            "execution_attempted": True,
            "execution_backend": "cuda",
            "stage": "complete",
            "deterministic_algorithms": True,
            "runtime_versions": runtime_versions(),
            "resource_peak": resources["resource_peak"],
            "model_parameter_digest_before": before,
            "model_parameter_digest_after": after,
            "model_parameter_digest_algorithm": "sha256/canonical-ordered-named-parameters-v1",
            "budget_pass": within_budget,
            "cleanup": resources["cleanup"],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "probe": {
                "dtype": "float64 CPU",
                "optimizer": "torch.optim.LBFGS strong_wolfe",
                "max_iter": LBFGS_MAX_ITER,
                "tolerance_grad": LBFGS_TOLERANCE_GRAD,
                "tolerance_change": LBFGS_TOLERANCE_CHANGE,
                "convergence_grad_tol": CONVERGENCE_GRAD_TOL,
                "l2_c": L2_C,
                "class_weight": "balanced",
                "standardization": "train-only; zero variance scale=1",
            },
        },
        "resources": resources,
    }


def _build_raw_token_linkage(
    raw_features: np.ndarray,
    real_rows: Sequence[Mapping[str, Any]],
    fixture_rows: Sequence[Mapping[str, Any]],
    vocab_size: int,
    excluded_ids: Sequence[int],
) -> dict[str, Any]:
    """Build digest-only linkage for the fixed-vocabulary raw-token control."""
    if (
        all(token_id < raw_features.shape[1] for token_id in excluded_ids)
        and np.count_nonzero(raw_features[:, list(excluded_ids)]) != 0
    ):
        raise ValueError("raw-token baseline excluded target-token columns are nonzero")
    return {
        "row_order": [str(row["row_id"]) for row in real_rows],
        "tokenizer": "pinned-gpt2",
        "vocab_size": vocab_size,
        "padding": "attention_mask; excluded padding tokens",
        "tokenization_digest": tokenization_digest(fixture_rows, vocab_size=vocab_size),
        "feature_matrix": {
            "digest": matrix_digest(raw_features, purpose="binary-input-token-bow"),
            "shape": list(raw_features.shape),
            "dtype": str(raw_features.dtype),
            "order": "C",
            "config": "binary token presence; attention_mask; no padding; target IDs excluded",
        },
        "excluded_columns": {
            "token_ids": list(excluded_ids),
            "digest": excluded_columns_digest(len(real_rows), list(excluded_ids)),
            "shape": [len(real_rows), len(excluded_ids)],
            "dtype": "float64",
            "order": "C",
            "all_zero": True,
        },
    }


def _cleanup_failure(torch: Any, model: Any | None, resources: dict[str, Any]) -> None:
    """Release CUDA resources and record a truthful cleanup stage after failure."""
    try:
        if model is not None:
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        resources["cleanup"] = "failure cleanup synchronized; gradients cleared; CUDA cache emptied"
    except Exception as cleanup_error:  # noqa: BLE001 - preserve failure provenance
        resources["cleanup"] = f"failure cleanup incomplete: {type(cleanup_error).__name__}"
    resources["stage"] = "cleanup"


def run_disentanglement(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    integration_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run L04.8 against the concrete TransformerLMIntegration boundary."""
    resources: dict[str, Any] = {
        "device": "not used",
        "network": "not attempted",
        "resource_peak": "not measured",
        "cleanup": "pending",
        "execution_attempted": False,
        "execution_backend": "none",
        "stage": "preflight",
    }
    if os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1":
        raise RealExecutionError("disentanglement requires LATENT_ANYTHING_RUN_NETWORK=1", resources)
    if os.environ.get("LATENT_ANYTHING_NETWORK_DEVICE", "").strip().lower() != "cuda":
        raise RealExecutionError("disentanglement requires LATENT_ANYTHING_NETWORK_DEVICE=cuda", resources)
    try:
        import torch
    except ImportError as exc:
        resources["stage"] = "dependency_check"
        raise RealExecutionError("PyTorch is required for the real disentanglement lane", resources) from exc
    resources["stage"] = "cuda_check"
    if not torch.cuda.is_available():
        resources["stage"] = "preflight"
        raise RealExecutionError("real disentanglement requires an available CUDA device", resources)
    started = time.perf_counter()
    torch.use_deterministic_algorithms(True)
    resources["execution_attempted"] = True
    resources["execution_backend"] = "cuda"
    resources["stage"] = "cuda_check"
    resources["network"] = "enabled"
    resources["device"] = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    model: Any | None = None
    peak_rss_bytes = 0
    _rss_value, rss_source, rss_unit = _rss_measurement()
    try:
        peak_rss_bytes = _rss_bytes()
        resources["stage"] = "model_load"
        if integration_factory is None:
            from scripts._m14_l04_boundary import transformer_integration_type

            integration_factory = transformer_integration_type()
        model_spec = plan["model"]
        integration = integration_factory(
            model_id=str(model_spec["id"]), revision=str(model_spec["revision"]), device="cuda"
        )
        model, tokenizer, config = integration._backend()
        if model is None:
            raise ValueError("TransformerLMIntegration returned no model")
        model.eval()
        before = parameter_digest(model)
        resolved = {text: resolve_target_token(tokenizer, text) for text in TARGET_TEXTS}
        if any(token_string != text for text, (_token_id, token_string) in resolved.items()):
            raise ValueError("target token decode does not exactly match frozen target text")
        token_ids = {text: int(token_id) for text, (token_id, _token_string) in resolved.items()}
        token_strings = {text: str(token_string) for text, (_token_id, token_string) in resolved.items()}
        if token_ids != TARGET_TOKEN_IDS or token_strings != TARGET_TOKEN_STRINGS:
            raise ValueError("target token IDs/strings do not match the pinned GPT-2 tokenizer")
        fixture_linkage = _frozen_fixture_linkage(plan, rows)
        real_rows = read_rows(integration, rows, int(plan["tokenization_and_sampling"]["max_length"]))
        source = {str(row["row_id"]): row for row in rows}
        train_rows = [row for row in real_rows if row["split"] == "train"]
        holdout_rows = [row for row in real_rows if row["split"] == "holdout"]
        train_source = [source[row["row_id"]] for row in train_rows]
        holdout_source = [source[row["row_id"]] for row in holdout_rows]
        train_labels = {factor: _labels(train_source, factor).tolist() for factor in FACTORS}
        holdout_labels = {factor: _labels(holdout_source, factor).tolist() for factor in FACTORS}
        train_features = capture_activations(model, train_rows, LAYER)
        holdout_features = capture_activations(model, holdout_rows, LAYER)
        peak_rss_bytes = max(peak_rss_bytes, _rss_bytes())
        resources["stage"] = "scoring"
        vocab_size = tokenizer_vocab_size(tokenizer, config)
        if vocab_size != GPT2_VOCAB_SIZE:
            raise ValueError("raw-token baseline requires the pinned GPT-2 vocabulary size")
        raw_features = binary_token_bow(real_rows, vocab_size, excluded_token_ids=set(token_ids.values()))
        excluded_ids = sorted(token_ids.values())
        raw_token_linkage = _build_raw_token_linkage(raw_features, real_rows, rows, vocab_size, excluded_ids)
        train_indices = [index for index, row in enumerate(real_rows) if row["split"] == "train"]
        holdout_indices = [index for index, row in enumerate(real_rows) if row["split"] == "holdout"]
        raw_train = raw_features[train_indices]
        raw_holdout = raw_features[holdout_indices]
        summaries = []
        for seed in SEEDS:
            seed_everything(seed, torch)
            summaries.append(
                _run_seed(
                    seed,
                    train_rows,
                    holdout_rows,
                    train_features,
                    holdout_features,
                    train_labels,
                    holdout_labels,
                    raw_train,
                    raw_holdout,
                    torch=torch,
                    holdout_source=holdout_source,
                )
            )
            peak_rss_bytes = max(peak_rss_bytes, _rss_bytes())
        resource_peak = _resource_peak(torch, str(resources["device"]), started, peak_rss_bytes, rss_source, rss_unit)
        resources["resource_peak"] = resource_peak
        within_budget = budget_pass(resource_peak)
        accepted = _acceptance_passed(summaries)
        after = parameter_digest(model)
        no_mutation = after == before
        accepted = bool(accepted and no_mutation and within_budget)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        resources["cleanup"] = "CUDA synchronized; model gradients cleared; CUDA cache emptied"
        resources["stage"] = "complete"
        return _build_execution_result(
            accepted=accepted,
            summaries=summaries,
            vocab_size=vocab_size,
            fixture_linkage=fixture_linkage,
            token_ids=token_ids,
            token_strings=token_strings,
            raw_token_linkage=raw_token_linkage,
            no_mutation=no_mutation,
            before=before,
            after=after,
            within_budget=within_budget,
            model_spec=model_spec,
            resources=resources,
        )
    except RealExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - dispatcher retains a sanitized failure envelope
        if resources.get("execution_attempted") is True and resources.get("resource_peak") == "not measured":
            try:
                resources["resource_peak"] = _resource_peak(
                    torch, str(resources["device"]), started, peak_rss_bytes, rss_source, rss_unit
                )
            except Exception:  # noqa: BLE001 - preserve the original failure
                resources["resource_peak"] = "partial measurement unavailable"
        _cleanup_failure(torch, model, resources)
        raise RealExecutionError(str(exc), resources) from exc


__all__ = ["BOOTSTRAP_REPLICATES", "REAL_STATUS", "SEEDS", "run_disentanglement"]
