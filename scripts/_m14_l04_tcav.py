"""Concrete, lazy M14 L04 TCAV execution handler."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from scripts._m14_l04_digest import runtime_versions
from scripts._m14_l04_tcav_controls import assemble_controls
from scripts._m14_l04_tcav_evaluation import evaluate_cached, evaluate_interventions
from scripts._m14_l04_tcav_metrics import bootstrap, corrected_empirical_p, metric, wilson_lower
from scripts._m14_l04_tcav_runtime import (
    RealExecutionError,
    capture_activations,
    parameter_digest,
    read_rows,
    resolve_target_token,
    task_margin_gradient,
)

REAL_STATUS = "passed_real_cuda"
TARGET_TEXTS = (" true", " false")
SEEDS = (17, 29, 41, 53, 67)
LAYER = 6
NATIVE_HIDDEN_STATE_INDEX = 7
BOOTSTRAP_REPLICATES = 2000
NULL_COUNT = 99
CONCEPT_FACTOR = "tone_positive"


def _metric_from_point(
    point: float,
    interval: Sequence[float],
    *,
    threshold: float,
    comparator: str,
    units: str = "dimensionless",
    aggregation_unit: str = "independent causal group",
) -> dict[str, Any]:
    if not np.isfinite(point) or len(interval) != 2 or not np.isfinite(np.asarray(interval, dtype=float)).all():
        raise ValueError("TCAV metric values must be finite")
    if float(interval[0]) > float(interval[1]):
        raise ValueError("TCAV metric confidence interval must be ordered")
    passed = {
        "<=": point <= threshold,
        "<": point < threshold,
        ">=": point >= threshold,
        ">": point > threshold,
    }[comparator]
    return {
        "point_estimate": float(point),
        "confidence_interval_95": [float(interval[0]), float(interval[1])],
        "units": units,
        "aggregation_unit": aggregation_unit,
        "statistic": "mean",
        "threshold": float(threshold),
        "comparator": comparator,
        "pass": bool(passed),
    }


def run_tcav(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    integration_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the frozen TCAV protocol; factories are private deterministic-test seams."""
    resources: dict[str, Any] = {
        "device": "cuda",
        "network": "not attempted",
        "resource_peak": "not measured",
        "cleanup": "pending",
    }
    if os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1":
        raise RealExecutionError("TCAV requires LATENT_ANYTHING_RUN_NETWORK=1", resources)
    if os.environ.get("LATENT_ANYTHING_NETWORK_DEVICE", "").strip().lower() != "cuda":
        raise RealExecutionError("TCAV requires LATENT_ANYTHING_NETWORK_DEVICE=cuda", resources)
    try:
        import torch
    except ImportError as exc:
        raise RealExecutionError("PyTorch is required for the real TCAV lane", resources) from exc
    if not torch.cuda.is_available():
        raise RealExecutionError("real TCAV requires an available CUDA device", resources)
    torch.use_deterministic_algorithms(True)
    deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    resources["network"] = "enabled"
    resources["device"] = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    if integration_factory is None:
        from scripts._m14_l04_boundary import transformer_integration_type

        integration_factory = transformer_integration_type()
    model_spec = plan["model"]
    integration = integration_factory(
        model_id=str(model_spec["id"]), revision=str(model_spec["revision"]), device="cuda"
    )
    started = time.perf_counter()
    model: Any | None = None
    try:
        model, tokenizer, _config = integration._backend()
        if model is None:
            raise RealExecutionError("TransformerLMIntegration returned no model", resources)
        before = parameter_digest(model)
        resolved_targets = {text: resolve_target_token(tokenizer, text) for text in TARGET_TEXTS}
        if any(token_string != text for text, (_token_id, token_string) in resolved_targets.items()):
            raise ValueError("TCAV target token decode does not exactly match the frozen target text")
        target_ids = {text: token_id for text, (token_id, _token_string) in resolved_targets.items()}
        target_strings = {text: token_string for text, (_token_id, token_string) in resolved_targets.items()}
        target_true, target_false = target_ids[" true"], target_ids[" false"]
        real_rows = read_rows(integration, rows, int(plan["tokenization_and_sampling"]["max_length"]))
        row_lookup = {str(row["row_id"]): row for row in rows}
        labels = np.asarray([int(row_lookup[item["row_id"]]["factor_labels"][CONCEPT_FACTOR]) for item in real_rows])
        train_mask = np.asarray([item["split"] == "train" for item in real_rows])
        holdout_mask = ~train_mask
        train_activations = capture_activations(
            model, [item for item, yes in zip(real_rows, train_mask, strict=True) if yes], LAYER
        )
        holdout_rows = [item for item, yes in zip(real_rows, holdout_mask, strict=True) if yes]
        holdout_labels = labels[holdout_mask]
        holdout_activations = capture_activations(model, holdout_rows, LAYER)
        # One cached task-margin gradient per row; controls below only change vectors/tokens.
        gradients = np.stack(
            [
                task_margin_gradient(
                    model,
                    item,
                    layer=LAYER,
                    target_token=target_true,
                    other_token=target_false,
                )
                for item in real_rows
            ],
            axis=0,
        )
        off_target_gradients = np.stack(
            [
                task_margin_gradient(
                    model,
                    item,
                    layer=LAYER,
                    target_token=target_false,
                    other_token=target_true,
                )
                for item in holdout_rows
            ],
            axis=0,
        )
        train_labels = labels[train_mask]
        thresholds = plan["thresholds_and_controls"]["tcav"]
        evaluation = evaluate_cached(
            train_activations=train_activations,
            train_labels=train_labels,
            holdout_activations=holdout_activations,
            holdout_rows=holdout_rows,
            holdout_labels=holdout_labels.tolist(),
            real_rows=real_rows,
            gradients=gradients,
            seeds=SEEDS,
            torch=torch,
            null_count=NULL_COUNT,
        )
        by_seed = evaluation["by_seed"]
        real_group_accuracy = evaluation["grouped_accuracy"]
        real_group_scores = evaluation["grouped_scores"]
        first_row_correct = evaluation["first_row_correct"]
        null_scores = evaluation["null_scores"]
        null_families = evaluation["null_families"]
        null_family_counts = evaluation["null_family_counts"]
        direction = evaluation["direction"]
        accuracy_interval = metric(
            real_group_accuracy, seed=SEEDS[0], threshold=float(thresholds["heldout_accuracy_min"]), comparator=">"
        )
        wilson = _metric_from_point(
            wilson_lower(round(sum(first_row_correct)), len(first_row_correct)),
            [wilson_lower(round(sum(first_row_correct)), len(first_row_correct))] * 2,
            threshold=float(thresholds["heldout_accuracy_wilson_lower_min"]),
            comparator=">",
        )
        score_interval = bootstrap(real_group_scores, seed=SEEDS[1], replicates=BOOTSTRAP_REPLICATES)
        bootstrap_lower = _metric_from_point(
            score_interval[0],
            score_interval,
            threshold=float(thresholds["bootstrap_ci_lower_min"]),
            comparator=">",
        )
        empirical_p = corrected_empirical_p(float(np.mean(real_group_scores)), null_scores)
        p_metric = _metric_from_point(
            empirical_p,
            [empirical_p, empirical_p],
            threshold=float(thresholds["corrected_empirical_p_max"]),
            comparator="<=",
        )
        intervention_values = evaluate_interventions(
            model=model,
            holdout_rows=holdout_rows,
            real_rows=real_rows,
            gradients=gradients,
            off_target_gradients=off_target_gradients,
            direction=direction,
            target_true=target_true,
            target_false=target_false,
            intervention_threshold=float(thresholds["intervention_agreement_min"]),
        )
        intervention = intervention_values["intervention"]
        intervention_groups = intervention_values["intervention_groups"]
        off_target_groups = intervention_values["off_target_groups"]
        zero_groups = intervention_values["zero_groups"]
        metrics = {
            "heldout_accuracy": accuracy_interval,
            "heldout_accuracy_wilson_lower": wilson,
            "bootstrap_ci_lower": bootstrap_lower,
            "corrected_empirical_p": p_metric,
            "intervention_agreement": intervention,
        }
        controls = assemble_controls(
            shuffled_scores=null_families["shuffled"],
            random_scores=null_families["random"],
            matched_scores=null_families["matched"],
            off_target_scores=off_target_groups,
            zero_differences=zero_groups,
            seed=SEEDS[0],
            sensitivity_reference=float(np.mean(real_group_scores)),
        )
        no_mutation = parameter_digest(model) == before
        all_pass = (
            all(value["pass"] for value in metrics.values())
            and all(value["pass"] for value in controls.values())
            and no_mutation
        )
        accepted = bool(all_pass and len(null_scores) == NULL_COUNT)
        resources["resource_peak"] = {
            "cuda_device": resources["device"],
            "elapsed_seconds": time.perf_counter() - started,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        resources["cleanup"] = "CUDA synchronized; model gradients cleared; CUDA cache emptied"
        return {
            "status": REAL_STATUS if accepted else "failed",
            "evidence_eligible": accepted,
            "acceptance": accepted,
            "evidence_level": "D3" if accepted else "D0",
            "failure_reason": None if accepted else "one or more frozen TCAV gates or controls failed",
            "metrics": metrics,
            "controls": controls,
            "control_raw": {
                "group_ids": sorted({str(item["group_id"]) for item in holdout_rows}),
                "intervention_agreement": intervention_groups,
                "off_target_target_token": off_target_groups,
                "zero_strength_identity": zero_groups,
            },
            "token_ids": {" true": target_true, " false": target_false},
            "target_token_strings": target_strings,
            "layer": LAYER,
            "native_hidden_state_index": NATIVE_HIDDEN_STATE_INDEX,
            "seeds": list(SEEDS),
            "raw_summaries": by_seed,
            "no_mutation": no_mutation,
            "provenance": {
                "runtime": "real TransformerLMIntegration",
                "model_revision": str(model_spec["revision"]),
                "target_token_ids": {" true": target_true, " false": target_false},
                "target_token_strings": target_strings,
                "off_target_token_id": target_false,
                "concept_factor": CONCEPT_FACTOR,
                "primary_sensitivity": float(np.mean(real_group_scores)),
                "concept_direction_norm": float(np.linalg.norm(direction)),
                "null_family_counts": null_family_counts,
                "target_position": "last non-padding token",
                "network": "enabled",
                "device": resources["device"],
                "deterministic_algorithms": deterministic_algorithms,
                "runtime_versions": runtime_versions(),
                "resource_peak": resources["resource_peak"],
                "cleanup": resources["cleanup"],
                "null_count": NULL_COUNT,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "direction_fit": "train groups only",
            },
            "resources": resources,
        }
    except RealExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - dispatcher retains the failure envelope
        try:
            if model is not None:
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            resources["cleanup"] = "failure cleanup synchronized; gradients cleared; CUDA cache emptied"
        except Exception as cleanup_error:  # noqa: BLE001
            resources["cleanup"] = f"failure cleanup incomplete: {type(cleanup_error).__name__}"
        raise RealExecutionError(str(exc), resources) from exc


run_tcav = run_tcav
__all__ = ["BOOTSTRAP_REPLICATES", "NULL_COUNT", "REAL_STATUS", "RealExecutionError", "SEEDS", "run_tcav"]
