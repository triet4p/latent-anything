"""Concrete M14 L04 Integrated Gradients execution handler.

This module owns the narrow, real-model IG lane.  It deliberately does not
expose prompts in its result: only group identifiers, token ids, metrics, and
runtime provenance are retained by the dispatcher envelope.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from scripts._m14_l04_boundary import transformer_integration_type
from scripts._m14_l04_digest import runtime_versions
from scripts._m14_l04_ig_metrics import cosine as _cosine
from scripts._m14_l04_ig_metrics import group_means as _group_metric
from scripts._m14_l04_ig_metrics import metric as _metric
from scripts._m14_l04_ig_runtime import (
    RealExecutionError,
)
from scripts._m14_l04_ig_runtime import (
    parameter_digest as _parameter_digest,
)
from scripts._m14_l04_ig_runtime import (
    read_rows as _read_rows,
)
from scripts._m14_l04_ig_runtime import (
    seed_everything as _seed_everything,
)
from scripts._m14_l04_ig_runtime import (
    target_attribution as _target_attribution,
)
from scripts.m14_l04_contract import validate_target_tokens

REAL_STATUS = "passed_real_cuda"
TARGET_TEXTS = (" true", " false")
SEEDS = (17, 29, 41, 53, 67)
IG_STEPS = (16, 64)


def run_integrated_gradients(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    integration_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the frozen IG protocol against the pinned model on real CUDA.

    The optional factory is private test dependency injection.  The caller
    (the production dispatcher) still marks all injected handlers ineligible.
    """
    resources: dict[str, Any] = {
        "device": "cuda",
        "network": "not attempted",
        "resource_peak": "not measured",
        "cleanup": "pending",
    }
    if __import__("os").environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1":
        raise RealExecutionError("Integrated Gradients requires LATENT_ANYTHING_RUN_NETWORK=1", resources)
    if __import__("os").environ.get("LATENT_ANYTHING_NETWORK_DEVICE", "").strip().lower() != "cuda":
        raise RealExecutionError("Integrated Gradients requires LATENT_ANYTHING_NETWORK_DEVICE=cuda", resources)
    try:
        import torch
    except ImportError as exc:
        raise RealExecutionError("PyTorch is required for the real Integrated Gradients lane", resources) from exc
    if not torch.cuda.is_available():
        raise RealExecutionError("real Integrated Gradients requires an available CUDA device", resources)
    torch.use_deterministic_algorithms(True)
    deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    resources["network"] = "enabled"
    resources["device"] = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    if integration_factory is None:
        integration_factory = transformer_integration_type()
    model_spec = plan["model"]
    integration = integration_factory(
        model_id=str(model_spec["id"]),
        revision=str(model_spec["revision"]),
        device="cuda",
    )
    started = time.perf_counter()
    model: Any | None = None
    try:
        model, tokenizer, _config = integration._backend()
        if model is None:
            raise RealExecutionError("TransformerLMIntegration returned no model", resources)
        parameters_before = _parameter_digest(model)

        def tokenize_target(text: str) -> list[int]:
            encoded_target = tokenizer(text, add_special_tokens=False)
            values = encoded_target["input_ids"] if isinstance(encoded_target, Mapping) else encoded_target
            if hasattr(values, "tolist"):
                values = values.tolist()
            return list(values)

        validate_target_tokens(tokenize_target, TARGET_TEXTS)
        target_ids = {text: int(tokenize_target(text)[0]) for text in TARGET_TEXTS}
        target_true, target_false = target_ids[" true"], target_ids[" false"]
        real_rows = _read_rows(integration, rows, int(plan["tokenization_and_sampling"]["max_length"]))
        by_seed: list[dict[str, Any]] = []
        all_zero: list[dict[str, Any]] = []
        all_batch_mean: list[dict[str, Any]] = []
        all_random: list[dict[str, Any]] = []
        batch_ids = np.stack([item.input_ids for item in real_rows])
        batch_mask = np.stack([item.attention_mask for item in real_rows])
        deterministic_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(real_rows):
            target = (
                target_true
                if str(next(r["target_text"] for r in rows if r["row_id"] == row.row_id)) == " true"
                else target_false
            )
            other = target_false if target == target_true else target_true
            single_ids = row.input_ids[None, :]
            single_mask = row.attention_mask[None, :]
            a16, d16, e16 = _target_attribution(
                model,
                row,
                target_token=target,
                other_token=other,
                steps=IG_STEPS[0],
                baseline="zero",
                seed=SEEDS[0],
                source_model_version=integration.provenance,
                batch_ids=single_ids,
                batch_mask=single_mask,
                batch_index=0,
            )
            a64, d64, e64 = _target_attribution(
                model,
                row,
                target_token=target,
                other_token=other,
                steps=IG_STEPS[1],
                baseline="zero",
                seed=SEEDS[0],
                source_model_version=integration.provenance,
                batch_ids=single_ids,
                batch_mask=single_mask,
                batch_index=0,
            )
            _batch_attr, batch_delta, batch_error = _target_attribution(
                model,
                row,
                target_token=target,
                other_token=other,
                steps=IG_STEPS[0],
                baseline="batch_mean",
                seed=SEEDS[0],
                source_model_version=integration.provenance,
                batch_ids=batch_ids,
                batch_mask=batch_mask,
                batch_index=row_index,
            )
            deterministic_rows.append(
                {
                    "row": row,
                    "target": target,
                    "other": other,
                    "a16": a16,
                    "a64": a64,
                    "d16": d16,
                    "d64": d64,
                    "e16": e16,
                    "e64": e64,
                    "batch_delta": batch_delta,
                    "batch_error": batch_error,
                    "single_ids": single_ids,
                    "single_mask": single_mask,
                }
            )
        for seed in SEEDS:
            _seed_everything(seed, torch)
            seed_zero: list[dict[str, Any]] = []
            seed_batch: list[dict[str, Any]] = []
            seed_random: list[dict[str, Any]] = []
            for deterministic in deterministic_rows:
                row = deterministic["row"]
                target = deterministic["target"]
                other = deterministic["other"]
                a16 = deterministic["a16"]
                a64 = deterministic["a64"]
                d16 = deterministic["d16"]
                d64 = deterministic["d64"]
                e16 = deterministic["e16"]
                e64 = deterministic["e64"]
                batch_delta = deterministic["batch_delta"]
                batch_error = deterministic["batch_error"]
                stable_offset = int.from_bytes(hashlib.sha256(row.row_id.encode("utf-8")).digest()[:4], "little")
                random_rng = np.random.default_rng(seed + stable_offset)
                random_target = int(random_rng.integers(0, int(model.config.vocab_size)))
                while random_target in {target, other}:
                    random_target = int(random_rng.integers(0, int(model.config.vocab_size)))
                random_attr, _random_delta, _random_error = _target_attribution(
                    model,
                    row,
                    target_token=random_target,
                    other_token=other,
                    steps=IG_STEPS[1],
                    baseline="zero",
                    seed=seed,
                    source_model_version=integration.provenance,
                    batch_ids=deterministic["single_ids"],
                    batch_mask=deterministic["single_mask"],
                    batch_index=0,
                )
                repeat_attr, _repeat_delta, _repeat_error = _target_attribution(
                    model,
                    row,
                    target_token=target,
                    other_token=other,
                    steps=IG_STEPS[1],
                    baseline="zero",
                    seed=seed,
                    source_model_version=integration.provenance,
                    batch_ids=deterministic["single_ids"],
                    batch_mask=deterministic["single_mask"],
                    batch_index=0,
                )
                zero_entry = {
                    "row_id": row.row_id,
                    "group_id": row.group_id,
                    "split": row.split,
                    "completeness_relative_error_16": abs(e16) / max(abs(d16), 1e-12),
                    "completeness_relative_error_64": abs(e64) / max(abs(d64), 1e-12),
                    "step_16_vs_64_attribution_cosine": _cosine(a16, a64),
                    "randomized_target_attribution_cosine": _cosine(a64, random_attr),
                    "seeded_repeat_cosine": _cosine(a64, repeat_attr),
                    "finite": bool(np.isfinite(a16).all() and np.isfinite(a64).all()),
                    "no_mutation": True,
                    "target_token_id": target,
                    "other_token_id": other,
                    "target_position": int(np.flatnonzero(row.attention_mask).max()),
                }
                batch_entry = {
                    "row_id": row.row_id,
                    "group_id": row.group_id,
                    "split": row.split,
                    "completeness_relative_error": abs(batch_error) / max(abs(batch_delta), 1e-12),
                }
                random_entry = {
                    "row_id": row.row_id,
                    "group_id": row.group_id,
                    "split": row.split,
                    "randomized_target_attribution_cosine": zero_entry["randomized_target_attribution_cosine"],
                }
                seed_zero.append(zero_entry)
                seed_batch.append(batch_entry)
                seed_random.append(random_entry)
            all_zero.extend(seed_zero)
            all_batch_mean.extend(seed_batch)
            all_random.extend(seed_random)
            by_seed.append({"seed": seed, "zero_baseline": seed_zero, "batch_mean_baseline": seed_batch})
        zero_completeness = _group_metric(all_zero, "completeness_relative_error_64")
        stability = _group_metric(all_zero, "step_16_vs_64_attribution_cosine")
        random_cosines = _group_metric(all_random, "randomized_target_attribution_cosine")
        repeat_cosines = _group_metric(all_zero, "seeded_repeat_cosine")
        finite_fraction = _group_metric(
            [
                {"group_id": item["group_id"], "finite_fraction": float(item["finite"] and item["no_mutation"])}
                for item in all_zero
            ],
            "finite_fraction",
        )
        batch_completeness = _group_metric(all_batch_mean, "completeness_relative_error")
        threshold = plan["thresholds_and_controls"]["integrated_gradients"]
        metrics = {
            "completeness_relative_error": _metric(
                zero_completeness,
                seed=79,
                threshold=float(threshold["completeness_relative_error_max"]),
                comparator="<=",
            ),
            "step_16_vs_64_attribution_cosine": _metric(
                stability,
                seed=80,
                threshold=float(threshold["step_16_vs_64_attribution_cosine_min"]),
                comparator=">",
                statistic="median",
            ),
        }
        batch_metric = _metric(
            batch_completeness,
            seed=81,
            threshold=float(threshold["completeness_relative_error_max"]),
            comparator="<=",
        )
        controls: dict[str, Any] = {
            "zero_baseline": {
                "metrics": metrics,
                "pass": all(value["pass"] for value in metrics.values()),
            },
            "batch_mean_baseline": {
                "metrics": {"completeness_relative_error": batch_metric},
                "pass": bool(batch_metric["pass"]),
            },
            "random_target": {
                "metrics": {
                    "attribution_cosine": _metric(
                        random_cosines,
                        seed=82,
                        threshold=float(threshold["randomized_target_cosine_max"]),
                        comparator="<=",
                    )
                },
                "pass": False,
            },
            "seeded_repeat": {
                "metrics": {
                    "attribution_cosine": _metric(repeat_cosines, seed=83, threshold=1.0 - 1e-8, comparator=">")
                },
                "repeat_count": 2,
                "seeds": list(SEEDS),
                "pass": False,
            },
            "finite/no-mutation": {
                "metrics": {
                    "finite_fraction": _metric(finite_fraction, seed=84, threshold=1.0, comparator=">="),
                },
                "finite_rows": len(all_zero),
                "mutated": False,
                "pass": False,
            },
        }
        controls["random_target"]["pass"] = bool(controls["random_target"]["metrics"]["attribution_cosine"]["pass"])
        controls["seeded_repeat"]["pass"] = bool(
            controls["seeded_repeat"]["metrics"]["attribution_cosine"]["pass"]
            and controls["seeded_repeat"]["repeat_count"] == 2
            and controls["seeded_repeat"]["seeds"] == list(SEEDS)
        )
        controls["finite/no-mutation"]["pass"] = bool(
            controls["finite/no-mutation"]["metrics"]["finite_fraction"]["pass"]
            and controls["finite/no-mutation"]["finite_rows"] > 0
            and controls["finite/no-mutation"]["mutated"] is False
            and _parameter_digest(model) == parameters_before
        )
        accepted = bool(
            metrics["completeness_relative_error"]["pass"]
            and metrics["step_16_vs_64_attribution_cosine"]["pass"]
            and all(value["pass"] for value in controls.values())
        )
        elapsed = time.perf_counter() - started
        resources["resource_peak"] = {
            "cuda_device": resources["device"],
            "elapsed_seconds": elapsed,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        resources["cleanup"] = "CUDA synchronized; model gradients cleared; CUDA cache emptied"
        status = REAL_STATUS if accepted else "failed"
        return {
            "status": status,
            "evidence_eligible": accepted,
            "acceptance": accepted,
            "failure_reason": None if accepted else "one or more frozen Integrated Gradients gates failed",
            "metrics": metrics,
            "controls": controls,
            "token_ids": {" true": target_true, " false": target_false},
            "layer": 6,
            "native_hidden_state_index": 7,
            "seeds": list(SEEDS),
            "raw_summaries": by_seed,
            "provenance": {
                "runtime": "real TransformerLMIntegration",
                "model_revision": str(model_spec["revision"]),
                "target_token_ids": {" true": target_true, " false": target_false},
                "target_position": "last non-padding token",
                "network": "enabled",
                "device": resources["device"],
                "deterministic_algorithms": deterministic_algorithms,
                "runtime_versions": runtime_versions(),
                "resource_peak": resources["resource_peak"],
                "cleanup": resources["cleanup"],
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
        except Exception as cleanup_error:  # noqa: BLE001 - retain primary failure and cleanup caveat
            resources["cleanup"] = f"failure cleanup incomplete: {type(cleanup_error).__name__}"
        raise RealExecutionError(str(exc), resources) from exc


__all__ = ["REAL_STATUS", "RealExecutionError", "run_integrated_gradients"]
