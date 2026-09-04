"""Private real-CUDA additive steering handler for M14 L04.10.

The handler is deliberately separate from interchange activation patching:
it learns one direction from clean/corrupted *training* pairs and applies
``hidden + strength * direction`` at the selected token during a new forward.
Only sanitized, prompt-free summaries and parameter digests leave this module.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from scripts._m14_l04_contract_common import canonical_json_bytes
from scripts._m14_l04_digest import runtime_versions
from scripts._m14_l04_ig_metrics import metric
from scripts._m14_l04_tcav_runtime import (
    RealExecutionError,
    parameter_digest,
    read_rows,
    resolve_target_token,
    seed_everything,
)

# Phase A is a technically complete CUDA diagnostic, not an accepted evidence
# result.  Keep this status distinct from the promotion-capable handlers.
REAL_STATUS = "completed_real_cuda_d0"
TARGET_TEXTS = (" true", " false")
TARGET_TOKEN_IDS = {" true": 2081, " false": 3991}
TARGET_TOKEN_STRINGS = {" true": " true", " false": " false"}
LAYER = 6
NATIVE_HIDDEN_STATE_INDEX = 7
SEEDS = (17, 29, 41, 53, 67)
BOOTSTRAP_REPLICATES = 2000
STRENGTH_GRID = (0.0, 0.25, 0.5, 1.0)
ZERO_STRENGTH_IDENTITY_ATOL = 1e-6
MAX_ELAPSED_SECONDS = 1800.0
MAX_CUDA_ALLOCATED_BYTES = 6 * 1024**3
MAX_RSS_BYTES = 4 * 1024**3


def _execution_result_digest(value: dict[str, Any]) -> str:
    """Hash the result before its self-referential provenance digest is set."""
    unsigned = deepcopy(value)
    provenance = unsigned.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("execution_result_digest", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


class _HookCleanupError(RuntimeError):
    """Internal marker for a failed forward-hook removal."""


def _module(model: Any, layer: int) -> Any:
    expected = f"transformer.h.{int(layer)}"
    for name, module in model.named_modules():
        if name == expected:
            return module
    raise ValueError(f"layer {layer} ({expected}) not found")


def _extract(output: Any) -> Any:
    from latent_anything._hook_output import extract_primary_tensor

    return extract_primary_tensor(output)


def _replace(output: Any, replacement: Any) -> Any:
    from latent_anything._hook_output import replace_primary_tensor

    return replace_primary_tensor(output, replacement)


def _inputs(model: Any, row: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch

    device = next(model.parameters()).device
    input_ids = torch.as_tensor(np.asarray(row["input_ids"])[None, :], dtype=torch.long, device=device)
    attention_mask = torch.as_tensor(np.asarray(row["attention_mask"])[None, :], dtype=torch.long, device=device)
    return input_ids, attention_mask


def _margin(model: Any, row: Mapping[str, Any], target_token: int, other_token: int) -> float:
    import torch

    input_ids, attention_mask = _inputs(model, row)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        position = int(row["target_position"])
        value = logits[0, position, target_token] - logits[0, position, other_token]
    result = float(value.detach().cpu())
    if not np.isfinite(result):
        raise ValueError("task margin is non-finite")
    return result


def capture_activation(model: Any, row: Mapping[str, Any], *, layer: int = LAYER) -> np.ndarray:
    """Capture one target-position activation, removing the hook always."""
    import torch

    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs_value: Any, output: Any) -> None:
        tensor = _extract(output)
        captured["value"] = tensor.detach().clone()

    handle = _module(model, layer).register_forward_hook(hook)
    try:
        input_ids, attention_mask = _inputs(model, row)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        value = captured.get("value")
        if value is None:
            raise ValueError("steering activation was not captured")
        position = int(row["target_position"])
        result = value[0, position].detach().cpu().numpy().astype(np.float64, copy=True)
        if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
            raise ValueError("steering activation is malformed or non-finite")
        return result
    finally:
        try:
            handle.remove()
        except Exception as exc:  # noqa: BLE001 - convert to sanitized cleanup failure
            raise _HookCleanupError(f"steering hook cleanup failed: {type(exc).__name__}") from exc


def apply_additive_intervention(
    model: Any,
    row: Mapping[str, Any],
    direction: np.ndarray,
    *,
    strength: float,
    layer: int = LAYER,
    token_position: int | None = None,
) -> float:
    """Measure a target margin after ``hidden += strength * direction``."""
    import torch

    vector = np.asarray(direction, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("direction must be a finite one-dimensional vector")
    if isinstance(strength, bool) or not np.isfinite(float(strength)) or float(strength) < 0:
        raise ValueError("strength must be a finite non-negative number")
    input_ids, attention_mask = _inputs(model, row)
    target_token = int(row.get("target_token", TARGET_TOKEN_IDS[" true"]))
    other_token = int(row.get("other_token", TARGET_TOKEN_IDS[" false"]))
    module = _module(model, layer)

    def hook(_module_value: Any, _inputs_value: Any, output: Any) -> Any:
        tensor = _extract(output)
        if int(tensor.shape[-1]) != vector.size:
            raise ValueError("direction hidden dimension does not match layer output")
        delta = torch.as_tensor(vector, dtype=tensor.dtype, device=tensor.device) * float(strength)
        position = int(row["target_position"] if token_position is None else token_position)
        if position < 0 or position >= int(tensor.shape[1]):
            raise ValueError("intervention token position is outside the sequence")
        changed = tensor.clone()
        changed[:, position, :] = changed[:, position, :] + delta
        return _replace(output, changed)

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits
            position = int(row["target_position"])
            value = logits[0, position, target_token] - logits[0, position, other_token]
        result = float(value.detach().cpu())
        if not np.isfinite(result):
            raise ValueError("intervened task margin is non-finite")
        return result
    finally:
        try:
            handle.remove()
        except Exception as exc:  # noqa: BLE001 - convert to sanitized cleanup failure
            raise _HookCleanupError(f"steering hook cleanup failed: {type(exc).__name__}") from exc


def normalized_direction(values: np.ndarray, *, norm: float | None = None) -> np.ndarray:
    """Return a finite L2-normalized direction, optionally at a target norm."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("direction must be a finite one-dimensional vector")
    source_norm = float(np.linalg.norm(vector))
    target_norm = source_norm if norm is None else float(norm)
    if not np.isfinite(target_norm) or target_norm <= 0 or source_norm <= 0:
        raise ValueError("direction norm must be finite and positive")
    return vector / source_norm * target_norm


def budget_pass(resource_peak: Mapping[str, Any]) -> bool:
    names = ("elapsed_seconds", "max_memory_allocated_bytes", "max_memory_reserved_bytes", "max_rss_bytes")
    values = [resource_peak.get(name) for name in names]
    numeric: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        numeric.append(float(value))
    elapsed, allocated, reserved, rss = numeric
    return bool(
        np.isfinite([elapsed, allocated, reserved, rss]).all()
        and 0 <= elapsed <= MAX_ELAPSED_SECONDS
        and 0 <= allocated <= MAX_CUDA_ALLOCATED_BYTES
        and 0 <= reserved <= MAX_CUDA_ALLOCATED_BYTES
        and 0 <= rss <= MAX_RSS_BYTES
    )


def _rss_measurement() -> tuple[int, str, str]:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return (
            value if sys.platform == "darwin" else value * 1024,
            "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "bytes",
        )
    except (ImportError, OSError):
        try:
            import psutil

            return int(psutil.Process().memory_info().rss), "psutil.Process.memory_info().rss", "bytes"
        except ImportError as exc:
            raise ValueError("process peak RSS measurement is unavailable") from exc


def _group_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["group_id"]), []).append(float(row[key]))
    return [float(np.mean(grouped[name])) for name in sorted(grouped)]


def paired_absolute_changes(rows: Sequence[Mapping[str, Any]], values: Sequence[float]) -> list[float]:
    """Return absolute clean/corrupted changes for each causal pair.

    Off-target locality is a paired diagnostic: averaging signed row effects
    first can cancel a real change when the clean and corrupted endpoints move
    in opposite directions.  Keep the pair as the independent unit and take
    the absolute change before any group/bootstrap aggregation.
    """
    if len(rows) != len(values):
        raise ValueError("paired changes require one value per row")
    grouped: dict[str, dict[str, float]] = {}
    group_ids: dict[str, str] = {}
    for row, value in zip(rows, values, strict=True):
        pair_id = str(row["causal_pair_id"])
        condition = str(row["condition"])
        if condition not in {"clean", "corrupted"}:
            raise ValueError("paired changes require clean and corrupted conditions")
        pair = grouped.setdefault(pair_id, {})
        if condition in pair:
            raise ValueError("paired changes require one value per condition")
        pair[condition] = float(value)
        group_ids[pair_id] = str(row["group_id"])
    changes: list[tuple[str, str, float]] = []
    for pair_id, pair in grouped.items():
        if set(pair) != {"clean", "corrupted"}:
            raise ValueError("paired changes require one clean and one corrupted row")
        changes.append((group_ids[pair_id], pair_id, abs(pair["clean"] - pair["corrupted"])))
    return [value for _group_id, _pair_id, value in sorted(changes)]


def _metric(
    values: Sequence[float], *, seed: int, threshold: float, comparator: str, units: str = "logits"
) -> dict[str, Any]:
    return metric(
        values,
        seed=seed,
        threshold=threshold,
        comparator=comparator,
        units=units,
    )


def _fixture_shape(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 24:
        raise ValueError("additive steering requires the frozen 24-row fixture")
    groups = {str(row.get("group_id")) for row in rows}
    train = {str(row.get("group_id")) for row in rows if row.get("split") == "train"}
    holdout = {str(row.get("group_id")) for row in rows if row.get("split") == "holdout"}
    pair_ids = {str(row.get("causal_pair_id")) for row in rows}
    if (
        len(groups) != 12
        or groups != {f"g{index:02d}" for index in range(1, 13)}
        or len(pair_ids) != 12
        or len(train) != 8
        or len(holdout) != 4
        or train & holdout
        or {str(row.get("split")) for row in rows} != {"train", "holdout"}
    ):
        raise ValueError("fixture must contain disjoint 8-train/4-holdout groups")
    for pair_id in pair_ids:
        pair = [row for row in rows if str(row.get("causal_pair_id")) == pair_id]
        if len(pair) != 2 or {str(row.get("condition")) for row in pair} != {"clean", "corrupted"}:
            raise ValueError("every causal pair must contain one clean and one corrupted row")
        if len({str(row.get("split")) for row in pair}) != 1 or len({str(row.get("group_id")) for row in pair}) != 1:
            raise ValueError("causal pair crosses train/holdout split")
    if any(
        len({str(row.get("causal_pair_id")) for row in rows if str(row.get("group_id")) == group_id}) != 1
        for group_id in groups
    ):
        raise ValueError("a fixture group must contain exactly one causal pair")


def _fit_direction(rows: Sequence[Mapping[str, Any]], activations: Sequence[np.ndarray]) -> np.ndarray:
    deltas = _pair_deltas(rows, activations)
    return normalized_direction(np.mean(np.stack(deltas, axis=0), axis=0))


def _pair_deltas(rows: Sequence[Mapping[str, Any]], activations: Sequence[np.ndarray]) -> list[np.ndarray]:
    by_pair: dict[str, dict[str, np.ndarray]] = {}
    for row, activation in zip(rows, activations, strict=True):
        by_pair.setdefault(str(row["causal_pair_id"]), {})[str(row["condition"])] = activation
    deltas: list[np.ndarray] = []
    for pair in by_pair.values():
        if set(pair) != {"clean", "corrupted"}:
            raise ValueError("train pair is missing clean or corrupted activation")
        deltas.append(pair["clean"] - pair["corrupted"])
    if not deltas:
        raise ValueError("no train causal pairs available for steering direction")
    return deltas


def _shuffled_label_direction(
    rows: Sequence[Mapping[str, Any]], activations: Sequence[np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a control from an actual balanced permutation of train labels.

    Labels are shuffled across individual train examples, independently of the
    fitted clean-minus-corrupted pairing.  The permutation is deterministic
    for the seed and only its digest/cardinality leave the runtime.
    """
    if len(rows) != len(activations) or not rows:
        raise ValueError("shuffled-label control requires matched train rows and activations")
    labels = np.asarray([1.0 if row.get("condition") == "clean" else -1.0 for row in rows], dtype=np.float64)
    if not np.isfinite(labels).all() or int(np.sum(labels > 0)) != int(np.sum(labels < 0)):
        raise ValueError("shuffled-label control requires balanced clean/corrupted labels")
    activation_matrix = np.stack([np.asarray(value, dtype=np.float64) for value in activations], axis=0)
    if activation_matrix.ndim != 2 or not np.isfinite(activation_matrix).all():
        raise ValueError("shuffled-label activations are malformed")
    for _ in range(32):
        permutation = rng.permutation(len(labels))
        shuffled = labels[permutation]
        # Comparing index permutations is insufficient: swapping two rows with
        # the same condition leaves the actual label assignment unchanged.
        if np.array_equal(shuffled, labels):
            continue
        candidate = np.mean(activation_matrix * shuffled[:, None], axis=0)
        if np.linalg.norm(candidate) <= 0.0:
            continue
        digest = hashlib.sha256(
            canonical_json_bytes({"ordered_labels": [int(label) for label in shuffled.tolist()]})
        ).hexdigest()
        return normalized_direction(candidate), {
            "policy": "uniform permutation of balanced train-example labels; independent of fitted labels",
            "permutation_digest": digest,
            "row_count": len(labels),
            "positive_count": int(np.sum(shuffled > 0)),
            "negative_count": int(np.sum(shuffled < 0)),
            "identity_permutation": False,
        }
    raise ValueError("shuffled-label permutation produced no valid non-degenerate direction")


def _safe_cleanup(
    torch: Any, model: Any | None, resources: dict[str, Any], *, preserve_failure_stage: bool = False
) -> bool:
    """Release resources and return whether every cleanup step completed."""
    causal_stage = resources.get("stage")
    if (
        preserve_failure_stage
        and resources.get("execution_attempted") is True
        and causal_stage not in {None, "cleanup", "complete"}
    ):
        resources["failure_stage"] = causal_stage
    try:
        if model is not None:
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        resources["cleanup"] = "CUDA synchronized; model gradients cleared; CUDA cache emptied"
        resources["cleanup_complete"] = True
        resources.pop("cleanup_error", None)
    except Exception as exc:  # noqa: BLE001 - preserve cleanup status without details
        resources["cleanup"] = f"failure cleanup incomplete: {type(exc).__name__}"
        resources["cleanup_complete"] = False
        resources["cleanup_error"] = type(exc).__name__
    if resources.get("execution_attempted") is True:
        resources["stage"] = "cleanup"
    return bool(resources["cleanup_complete"])


def run_additive_steering(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    integration_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the frozen train-only additive steering protocol."""
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
        raise RealExecutionError("additive steering requires LATENT_ANYTHING_RUN_NETWORK=1", resources)
    if os.environ.get("LATENT_ANYTHING_NETWORK_DEVICE", "").strip().lower() != "cuda":
        raise RealExecutionError("additive steering requires LATENT_ANYTHING_NETWORK_DEVICE=cuda", resources)
    try:
        import torch
    except ImportError as exc:
        resources["stage"] = "dependency_check"
        raise RealExecutionError("PyTorch is required for additive steering", resources) from exc
    resources["stage"] = "cuda_check"
    try:
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise RealExecutionError("additive steering requires an available CUDA device", resources)
        torch.use_deterministic_algorithms(True)
        # Persist only the canonical logical device identifier; vendor/model
        # names are dynamic strings and are not safe provenance values.
        device_name = "cuda:0"
        torch.cuda.reset_peak_memory_stats()
    except RealExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - preserve a pre-CUDA failure tuple
        raise RealExecutionError("additive steering CUDA preflight failed", resources) from exc
    started = time.perf_counter()
    resources.update(
        {
            "device": device_name,
            "network": "enabled",
            "execution_attempted": True,
            "execution_backend": "cuda",
        }
    )
    model: Any | None = None
    before: str | None = None
    after: str | None = None
    try:
        _fixture_shape(rows)
        model_spec = plan["model"]
        if integration_factory is None:
            from scripts._m14_l04_boundary import transformer_integration_type

            integration_factory = transformer_integration_type()
        integration = integration_factory(
            model_id=str(model_spec["id"]), revision=str(model_spec["revision"]), device="cuda"
        )
        resources["stage"] = "model_load"
        model, tokenizer, _config = integration._backend()
        if model is None:
            raise ValueError("TransformerLMIntegration returned no model")
        model.eval()
        before = parameter_digest(model)
        resolved = {text: resolve_target_token(tokenizer, text) for text in TARGET_TEXTS}
        if resolved != {text: (TARGET_TOKEN_IDS[text], TARGET_TOKEN_STRINGS[text]) for text in TARGET_TEXTS}:
            raise ValueError("target token IDs/strings do not match pinned GPT-2")
        max_length = int(plan["tokenization_and_sampling"]["max_length"])
        real_rows = read_rows(integration, rows, max_length)
        source_by_id = {str(row["row_id"]): row for row in rows}
        for row in real_rows:
            source = source_by_id.get(str(row["row_id"]))
            if source is None or str(source["split"]) not in {"train", "holdout"}:
                raise ValueError("tokenized row linkage is malformed")
        train_rows = [row for row in real_rows if row["split"] == "train"]
        holdout_rows = [row for row in real_rows if row["split"] == "holdout"]
        resources["stage"] = "scoring"
        train_activations = [capture_activation(model, row) for row in train_rows]
        direction = _fit_direction(train_rows, train_activations)
        target_token = TARGET_TOKEN_IDS[" true"]
        other_token = TARGET_TOKEN_IDS[" false"]
        baseline = [_margin(model, row, target_token, other_token) for row in holdout_rows]
        main_rows: list[dict[str, Any]] = []
        for row, baseline_margin in zip(holdout_rows, baseline, strict=True):
            effects = {
                str(strength): float(
                    apply_additive_intervention(
                        model,
                        {**row, "target_token": target_token, "other_token": other_token},
                        direction,
                        strength=strength,
                    )
                    - baseline_margin
                )
                for strength in STRENGTH_GRID
            }
            main_rows.append(
                {
                    "row_id": str(row["row_id"]),
                    "group_id": str(row["group_id"]),
                    "causal_pair_id": str(row["causal_pair_id"]),
                    "condition": str(row["condition"]),
                    "split": str(row["split"]),
                    "baseline_margin": float(baseline_margin),
                    "strength_effects": effects,
                }
            )
        thresholds = plan["thresholds_and_controls"]["steering"]
        metric_declarations = {
            "target_effect": (float(thresholds["target_effect_ci_lower_strict_gt_logits"]), ">", "logits"),
            "selectivity": (float(thresholds["selectivity_ci_lower_strict_gt_logits"]), ">", "logits"),
            "off_target_token": (float(thresholds["off_target_absolute_effect_max_logits"]), "<=", "logits"),
            "zero_strength_identity": (
                float(thresholds["zero_strength_identity_atol"]),
                "<=",
                "absolute logit margin difference",
            ),
        }
        summaries: list[dict[str, Any]] = []
        for seed in SEEDS:
            seed_everything(seed, torch)
            rng = np.random.default_rng(seed)
            random_direction = rng.normal(size=direction.size).astype(np.float64)
            matched_direction = normalized_direction(random_direction, norm=float(np.linalg.norm(direction)))
            shuffled_direction, shuffled_label_provenance = _shuffled_label_direction(
                train_rows, train_activations, rng
            )
            control_specs = {
                "randomized": random_direction,
                "shuffled": shuffled_direction,
                "matched_norm": matched_direction,
            }
            controls: dict[str, list[float]] = {}
            for name, control_direction in control_specs.items():
                controls[name] = _group_values(
                    [
                        {
                            "group_id": row["group_id"],
                            "effect": apply_additive_intervention(
                                model,
                                {**row, "target_token": target_token, "other_token": other_token},
                                control_direction,
                                strength=1.0,
                            )
                            - base,
                        }
                        for row, base in zip(holdout_rows, baseline, strict=True)
                    ],
                    "effect",
                )
            target_group_effects = _group_values(
                [{"group_id": item["group_id"], "effect": item["strength_effects"]["1.0"]} for item in main_rows],
                "effect",
            )
            off_target_effects = []
            for row, base in zip(holdout_rows, baseline, strict=True):
                valid_positions = np.flatnonzero(np.asarray(row["attention_mask"], dtype=bool))
                if len(valid_positions) < 2:
                    raise ValueError("off-target token control requires at least two valid tokens")
                off_target_position = int(valid_positions[-2])
                off_target_margin = apply_additive_intervention(
                    model,
                    {**row, "target_token": target_token, "other_token": other_token},
                    direction,
                    strength=1.0,
                    token_position=off_target_position,
                )
                off_target_effects.append(off_target_margin - base)
            # Preserve clean/corrupted pairing before aggregation.  A signed
            # group mean would let opposite endpoint effects cancel to zero.
            off_target_groups = paired_absolute_changes(holdout_rows, off_target_effects)
            zero_errors = _group_values(
                [
                    {"group_id": item["group_id"], "error": abs(float(item["strength_effects"]["0.0"]))}
                    for item in main_rows
                ],
                "error",
            )
            target_metric = _metric(
                target_group_effects,
                seed=seed,
                threshold=float(thresholds["target_effect_ci_lower_strict_gt_logits"]),
                comparator=">",
            )
            selectivity_values = [
                target - off for target, off in zip(target_group_effects, off_target_groups, strict=True)
            ]
            selectivity_metric = _metric(
                selectivity_values,
                seed=seed,
                threshold=float(thresholds["selectivity_ci_lower_strict_gt_logits"]),
                comparator=">",
            )
            off_target_metric = _metric(
                off_target_groups,
                seed=seed,
                threshold=float(thresholds["off_target_absolute_effect_max_logits"]),
                comparator="<=",
            )
            zero_metric = _metric(
                zero_errors,
                seed=seed,
                threshold=float(thresholds["zero_strength_identity_atol"]),
                comparator="<=",
                units="absolute logit margin difference",
            )
            summaries.append(
                {
                    "seed": seed,
                    "holdout_groups": sorted({str(row["group_id"]) for row in holdout_rows}),
                    "target_effect": target_metric,
                    "selectivity": selectivity_metric,
                    "off_target_token": off_target_metric,
                    "off_target_pair_changes": off_target_groups,
                    "zero_strength_identity": zero_metric,
                    "strength_curve": {
                        strength: _metric(
                            _group_values(
                                [
                                    {"group_id": item["group_id"], "effect": item["strength_effects"][strength]}
                                    for item in main_rows
                                ],
                                "effect",
                            ),
                            seed=seed,
                            threshold=metric_declarations["target_effect"][0],
                            comparator=metric_declarations["target_effect"][1],
                        )
                        for strength in (str(value) for value in STRENGTH_GRID)
                    },
                    "control_effects": {
                        name: _metric(
                            values,
                            seed=seed,
                            threshold=metric_declarations["off_target_token"][0],
                            comparator=metric_declarations["off_target_token"][1],
                            units=metric_declarations["off_target_token"][2],
                        )
                        for name, values in controls.items()
                    },
                    "control_direction_norms": {
                        "randomized": float(np.linalg.norm(random_direction)),
                        "shuffled": float(np.linalg.norm(shuffled_direction)),
                        "matched_norm": float(np.linalg.norm(matched_direction)),
                    },
                    "shuffled_label_provenance": shuffled_label_provenance,
                    "finite": bool(
                        np.isfinite(np.asarray(target_group_effects + selectivity_values + off_target_groups)).all()
                    ),
                }
            )
        peak_rss, rss_source, rss_unit = _rss_measurement()
        resources["resource_peak"] = {
            "cuda_device": resources["device"],
            "elapsed_seconds": time.perf_counter() - started,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "max_rss_bytes": int(peak_rss),
            "rss_source": rss_source,
            "rss_unit": rss_unit,
        }
        within_budget = budget_pass(resources["resource_peak"])
        after = parameter_digest(model)
        no_mutation = after == before
        criteria = {
            "target_effect": all(item["target_effect"]["pass"] for item in summaries),
            "selectivity": all(item["selectivity"]["pass"] for item in summaries),
            "off_target": all(item["off_target_token"]["pass"] for item in summaries),
            "zero_strength_identity": all(item["zero_strength_identity"]["pass"] for item in summaries),
            "controls": all(
                item["control_effects"][name]["pass"]
                for item in summaries
                for name in ("randomized", "shuffled", "matched_norm")
            ),
            "finite": all(item["finite"] for item in summaries),
            "budget": within_budget,
            "no_mutation": no_mutation,
            "parameter_digest_equal": no_mutation,
        }
        cleanup_complete = _safe_cleanup(torch, model, resources)
        criteria["cleanup_complete"] = cleanup_complete
        semantic_candidate = bool(all(criteria.values()))
        if cleanup_complete:
            resources["stage"] = "complete"
        control_passes = {
            "zero_strength": all(item["zero_strength_identity"]["pass"] for item in summaries),
            "randomized_direction": all(item["control_effects"]["randomized"]["pass"] for item in summaries),
            "shuffled_labels": all(item["control_effects"]["shuffled"]["pass"] for item in summaries),
            "off_target_token": all(item["off_target_token"]["pass"] for item in summaries),
            "matched_norm_direction": all(item["control_effects"]["matched_norm"]["pass"] for item in summaries),
        }
        result: dict[str, Any] = {
            # Phase A intentionally reports runtime completion separately from
            # promotion.  Task 2's artifact validator owns any final gate.
            "status": REAL_STATUS if cleanup_complete else "failed",
            "evidence_eligible": False,
            "acceptance": False,
            "evidence_level": "D0",
            "semantic_candidate": semantic_candidate,
            "criteria": criteria,
            "failure_reason": None
            if cleanup_complete and semantic_candidate
            else ("cleanup incomplete" if not cleanup_complete else "one or more additive steering criteria failed"),
            "metrics": {
                "target_effect": {str(item["seed"]): deepcopy(item["target_effect"]) for item in summaries},
                "selectivity": {str(item["seed"]): deepcopy(item["selectivity"]) for item in summaries},
                "off_target_token": {str(item["seed"]): deepcopy(item["off_target_token"]) for item in summaries},
                "zero_strength_identity": {
                    str(item["seed"]): deepcopy(item["zero_strength_identity"]) for item in summaries
                },
            },
            "controls": {
                "zero_strength": {
                    "pass": control_passes["zero_strength"],
                    "by_seed": {str(item["seed"]): deepcopy(item["zero_strength_identity"]) for item in summaries},
                },
                "randomized_direction": {
                    "pass": control_passes["randomized_direction"],
                    "by_seed": {
                        str(item["seed"]): deepcopy(item["control_effects"]["randomized"]) for item in summaries
                    },
                },
                "shuffled_labels": {
                    "pass": control_passes["shuffled_labels"],
                    "by_seed": {str(item["seed"]): deepcopy(item["control_effects"]["shuffled"]) for item in summaries},
                },
                "off_target_token": {
                    "pass": control_passes["off_target_token"],
                    "by_seed": {str(item["seed"]): deepcopy(item["off_target_token"]) for item in summaries},
                },
                "matched_norm_direction": {
                    "pass": control_passes["matched_norm_direction"],
                    "by_seed": {
                        str(item["seed"]): deepcopy(item["control_effects"]["matched_norm"]) for item in summaries
                    },
                },
            },
            "raw_summaries": summaries,
            "holdout_evidence": main_rows,
            "token_ids": dict(TARGET_TOKEN_IDS),
            "target_token_strings": dict(TARGET_TOKEN_STRINGS),
            "layer": LAYER,
            "native_hidden_state_index": NATIVE_HIDDEN_STATE_INDEX,
            "seeds": list(SEEDS),
            "strength_grid": list(STRENGTH_GRID),
            "train_groups": sorted({str(row["group_id"]) for row in train_rows}),
            "holdout_groups": sorted({str(row["group_id"]) for row in holdout_rows}),
            "direction_norm": float(np.linalg.norm(direction)),
            "no_mutation": no_mutation,
            "budget_pass": within_budget,
            "model_parameter_digest_before": before,
            "model_parameter_digest_after": after,
            "provenance": {
                "runtime": "real TransformerLMIntegration",
                "use_case": "AdditiveSteering",
                "model_revision": str(model_spec["revision"]),
                "target_token_ids": dict(TARGET_TOKEN_IDS),
                "target_token_strings": dict(TARGET_TOKEN_STRINGS),
                "target_position": "last non-padding token",
                "direction_fit": "clean-minus-corrupted train pairs only; normalized before holdout scoring",
                "network": "enabled",
                "device": resources["device"],
                "execution_attempted": True,
                "execution_backend": "cuda",
                "stage": resources["stage"],
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "runtime_versions": runtime_versions(),
                "resource_peak": resources["resource_peak"],
                "budget_pass": within_budget,
                "cleanup": resources["cleanup"],
                "cleanup_complete": cleanup_complete,
                "model_parameter_digest_before": before,
                "model_parameter_digest_after": after,
                "model_parameter_digest_algorithm": "sha256/canonical-ordered-named-parameters-v1",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "aggregation_unit": "independent causal group",
                "off_target_aggregation": "mean(abs(clean-corrupted effect per causal pair))",
                "shuffled_label_policy": (
                    "uniform permutation of balanced train-example labels; independent of fitted labels"
                ),
                "shuffled_label_cardinality": {"rows": 16, "positive": 8, "negative": 8},
                "shuffled_label_identity_assignment": False,
            },
            "resources": resources,
        }
        result["provenance"]["execution_result_digest"] = _execution_result_digest(result)
        return result
    except RealExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - sanitized failure, never leak prompts/paths
        _safe_cleanup(torch, model, resources, preserve_failure_stage=True)
        if isinstance(exc, _HookCleanupError):
            resources["cleanup"] = f"failure cleanup incomplete: {type(exc).__name__}"
            resources["cleanup_complete"] = False
            resources["cleanup_error"] = type(exc).__name__
        if model is not None:
            try:
                after = parameter_digest(model)
            except Exception:  # noqa: BLE001 - digest is best-effort on failure
                after = None
        raise RealExecutionError(f"additive steering failed during {resources['stage']}", resources) from exc


run_steering = run_additive_steering

__all__ = [
    "BOOTSTRAP_REPLICATES",
    "LAYER",
    "MAX_CUDA_ALLOCATED_BYTES",
    "MAX_ELAPSED_SECONDS",
    "MAX_RSS_BYTES",
    "NATIVE_HIDDEN_STATE_INDEX",
    "REAL_STATUS",
    "SEEDS",
    "STRENGTH_GRID",
    "TARGET_TOKEN_IDS",
    "TARGET_TOKEN_STRINGS",
    "ZERO_STRENGTH_IDENTITY_ATOL",
    "apply_additive_intervention",
    "budget_pass",
    "capture_activation",
    "normalized_direction",
    "paired_absolute_changes",
    "run_additive_steering",
    "run_steering",
]
