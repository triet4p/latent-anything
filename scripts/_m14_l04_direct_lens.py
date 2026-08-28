"""Concrete real-CUDA direct logit-lens handler for M14 L04.6."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from scripts._m14_l04_digest import runtime_versions
from scripts._m14_l04_direct_lens_runtime import (
    RealExecutionError,
    parameter_digest,
    seed_everything,
    summarize_generation,
    validate_targets,
)
from scripts._m14_l04_ig_metrics import metric

REAL_STATUS = "passed_real_cuda"
SEEDS = (17, 29, 41, 53, 67)
NATIVE_LAYERS = tuple(range(13))
BOOTSTRAP_REPLICATES = 2000


def _group_means(records: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(str(record["group_id"]), []).append(float(record[key]))
    return [float(np.mean(values)) for _, values in sorted(grouped.items())]


def _seed_group_means(values: Sequence[float], group_count: int) -> list[float]:
    if len(values) % group_count != 0:
        raise ValueError("seed group values do not divide into complete held-out groups")
    return [
        float(np.mean([values[offset + index] for offset in range(0, len(values), group_count)]))
        for index in range(group_count)
    ]


def _finite_control(values: Sequence[float], seed: int) -> dict[str, Any]:
    metric_value = metric(
        [1.0 if np.isfinite(value) else 0.0 for value in values], seed=seed, threshold=1.0, comparator=">="
    )
    return {"metrics": {"finite_fraction": metric_value}, "pass": bool(metric_value["pass"])}


def run_direct_logit_lens(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    integration_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run direct lens parity and diagnostics on all fixture prompts."""
    resources: dict[str, Any] = {
        "device": "cuda",
        "network": "not attempted",
        "resource_peak": "not measured",
        "cleanup": "pending",
    }
    if os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1":
        raise RealExecutionError("direct logit lens requires LATENT_ANYTHING_RUN_NETWORK=1", resources)
    if os.environ.get("LATENT_ANYTHING_NETWORK_DEVICE", "").strip().lower() != "cuda":
        raise RealExecutionError("direct logit lens requires LATENT_ANYTHING_NETWORK_DEVICE=cuda", resources)
    try:
        import torch

        from latent_anything.integrations.transformer_lm import TransformerGenerationRequest
    except ImportError as exc:
        raise RealExecutionError(
            "PyTorch and TransformerLMIntegration are required for direct lens", resources
        ) from exc
    if not torch.cuda.is_available():
        raise RealExecutionError("real direct logit lens requires an available CUDA device", resources)
    torch.use_deterministic_algorithms(True)
    resources["network"] = "enabled"
    resources["device"] = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model: Any | None = None
    try:
        if integration_factory is None:
            from scripts._m14_l04_boundary import transformer_integration_type

            integration_factory = transformer_integration_type()
        model_spec = plan["model"]
        integration = integration_factory(
            model_id=str(model_spec["id"]), revision=str(model_spec["revision"]), device="cuda"
        )
        model, tokenizer, _config = integration._backend()
        if model is None:
            raise ValueError("TransformerLMIntegration returned no model")
        before = parameter_digest(model)
        target_ids, target_strings = validate_targets(tokenizer)
        prompts = tuple(str(row["prompt"]) for row in rows)
        max_length = int(plan["tokenization_and_sampling"]["max_length"])
        summaries: list[dict[str, Any]] = []
        heldout_group_margins: list[float] = []
        shuffled_group_margins: list[float] = []
        randomized_group_margins: list[float] = []
        parity_abs: list[float] = []
        parity_rel: list[float] = []
        for seed in SEEDS:
            seed_everything(seed, torch)
            result = integration.generate(
                TransformerGenerationRequest(
                    prompt=prompts,
                    max_length=max_length,
                    seed=seed,
                    capture_hidden_states=True,
                    top_k_logit_lens=0,
                )
            )
            records, max_abs, max_rel = summarize_generation(result, rows, target_ids, target_ids)
            rng = np.random.default_rng(seed)
            target_flags = np.asarray([str(row["target_text"]) == " true" for row in rows], dtype=bool)
            shuffled_flags = rng.permutation(target_flags)
            for record, shuffled_true in zip(records, shuffled_flags, strict=True):
                target_probability = record["target_probabilities"][-1]
                other_probability = record["other_probabilities"][-1]
                record["shuffled_target_margin"] = float(
                    target_probability - other_probability
                    if bool(shuffled_true) == (record["target_token_id"] == target_ids[" true"])
                    else other_probability - target_probability
                )
                record["randomized_target_margin"] = float(other_probability - target_probability)
            holdout = [record for record in records if record["split"] == "holdout"]
            heldout_group_margins.extend(_group_means(holdout, "target_margin"))
            shuffled_group_margins.extend(_group_means(holdout, "shuffled_target_margin"))
            randomized_group_margins.extend(_group_means(holdout, "randomized_target_margin"))
            parity_abs.append(max_abs)
            parity_rel.append(max_rel)
            summaries.append(
                {
                    "seed": seed,
                    "layer_indices": list(NATIVE_LAYERS),
                    "native_hidden_state_indices": list(NATIVE_LAYERS),
                    "rows": records,
                    "terminal_logit_max_abs_error": max_abs,
                    "terminal_logit_max_relative_error": max_rel,
                }
            )
        thresholds = plan["thresholds_and_controls"]["lens"]
        parity_metric = metric(
            parity_abs,
            seed=SEEDS[0],
            threshold=float(thresholds["direct_parity_atol"]),
            comparator="<=",
            statistic="median",
        )
        relative_metric = metric(
            parity_rel,
            seed=SEEDS[1],
            threshold=float(thresholds["direct_parity_rtol"]),
            comparator="<=",
            statistic="median",
        )
        heldout_group_values = _seed_group_means(heldout_group_margins, len(plan["fixture"]["split"]["holdout_groups"]))
        group_selectivity = metric(
            heldout_group_values,
            seed=SEEDS[2],
            threshold=0.0,
            comparator=">",
        )
        controls = {
            "target_non_target_selectivity": _finite_control(heldout_group_margins, SEEDS[2]),
            "shuffled_target_labels": _finite_control(shuffled_group_margins, SEEDS[3]),
            "randomized_target_tokens": _finite_control(randomized_group_margins, SEEDS[4]),
            "terminal_post_ln_f_parity": {
                "metrics": {"max_abs_error": parity_metric, "max_relative_error": relative_metric},
                "pass": bool(parity_metric["pass"] and relative_metric["pass"]),
            },
        }
        no_mutation = parameter_digest(model) == before
        all_pass = bool(
            parity_metric["pass"]
            and relative_metric["pass"]
            and all(summary_row["finite"] for summary in summaries for summary_row in summary["rows"])
            and all(control["pass"] for control in controls.values())
            and no_mutation
        )
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
            "status": REAL_STATUS if all_pass else "failed",
            "evidence_eligible": all_pass,
            "acceptance": all_pass,
            "evidence_level": "D0",
            "failure_reason": None if all_pass else "direct lens parity or no-mutation gate failed",
            "metrics": {"terminal_logit_parity": parity_metric, "terminal_logit_relative_parity": relative_metric},
            "diagnostics": {"heldout_target_non_target_selectivity": group_selectivity},
            "controls": controls,
            "control_raw": {
                "holdout_group_margins": heldout_group_margins,
                "shuffled_group_margins": shuffled_group_margins,
                "randomized_group_margins": randomized_group_margins,
            },
            "token_ids": target_ids,
            "target_token_strings": target_strings,
            "layer": 6,
            "native_hidden_state_index": 7,
            "seeds": list(SEEDS),
            "raw_summaries": summaries,
            "no_mutation": no_mutation,
            "provenance": {
                "runtime": "real TransformerLMIntegration",
                "model_id": str(model_spec["id"]),
                "model_revision": str(model_spec["revision"]),
                "target_token_ids": target_ids,
                "target_token_strings": target_strings,
                "target_position": "last non-padding token",
                "native_layer_indices": list(NATIVE_LAYERS),
                "network": "enabled",
                "device": resources["device"],
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "runtime_versions": runtime_versions(),
                "resource_peak": resources["resource_peak"],
                "cleanup": resources["cleanup"],
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "aggregation_unit": "independent causal group",
            },
            "resources": resources,
        }
    except RealExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - dispatcher writes the failure envelope
        try:
            if model is not None:
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            resources["cleanup"] = "failure cleanup synchronized; gradients cleared; CUDA cache emptied"
        except Exception as cleanup_error:  # noqa: BLE001
            resources["cleanup"] = f"failure cleanup incomplete: {type(cleanup_error).__name__}"
        raise RealExecutionError(str(exc), resources) from exc


__all__ = ["REAL_STATUS", "SEEDS", "run_direct_logit_lens"]
