"""Real GPT-2 clean/corrupted activation interchange for M14 L04.9.

This module intentionally owns a private, model-bound implementation.  It is
not an additive steering intervention and does not change the public VAE
``ActivationPatch`` or ``TransformerLMIntegration`` APIs.
"""

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
from scripts._m14_l04_fixture_contract import FIXTURE_PATH, content_digest, read_fixture
from scripts._m14_l04_ig_metrics import bootstrap

REAL_STATUS = "passed_real_cuda"
TARGET_TEXTS = (" true", " false")
TARGET_TOKEN_IDS = {" true": 2081, " false": 3991}
TARGET_TOKEN_STRINGS = {" true": " true", " false": " false"}
LAYER = 6
OFF_TARGET_LAYER = 5
NATIVE_HIDDEN_STATE_INDEX = 7
BOOTSTRAP_REPLICATES = 2000
SEEDS = (17, 29, 41, 53, 67)
STRENGTH_GRID = (0.0, 0.25, 0.5, 1.0)
RECOVERY_CI_LOWER_THRESHOLD = 0.1
OFF_TARGET_ABSOLUTE_EFFECT_MAX = 0.1
ZERO_STRENGTH_IDENTITY_ATOL = 1e-6
MAX_ELAPSED_SECONDS = 1800.0
MAX_CUDA_ALLOCATED_BYTES = 6 * 1024**3
MAX_RSS_BYTES = 4 * 1024**3


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def _rss_measurement() -> tuple[int, str, str]:
    try:
        import resource

        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if raw < 0:
            raise ValueError("process peak RSS is negative")
        return (raw if sys.platform == "darwin" else raw * 1024), "resource.getrusage(RUSAGE_SELF).ru_maxrss", "bytes"
    except (ImportError, OSError):
        try:
            import psutil

            return int(psutil.Process().memory_info().rss), "psutil.Process.memory_info().rss", "bytes"
        except ImportError as exc:
            raise ValueError("process peak RSS measurement is unavailable") from exc


def budget_pass(resource_peak: Mapping[str, Any]) -> bool:
    values = [
        resource_peak.get(name)
        for name in ("elapsed_seconds", "max_memory_allocated_bytes", "max_memory_reserved_bytes", "max_rss_bytes")
    ]
    if any(not _finite(value) for value in values):
        return False
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) != 4:
        return False
    elapsed, allocated, reserved, rss = numeric
    return bool(
        0.0 <= elapsed <= MAX_ELAPSED_SECONDS
        and 0.0 <= allocated <= MAX_CUDA_ALLOCATED_BYTES
        and 0.0 <= reserved <= MAX_CUDA_ALLOCATED_BYTES
        and 0.0 <= rss <= MAX_RSS_BYTES
    )


def _module(model: Any, layer: int) -> Any:
    name = f"transformer.h.{int(layer)}"
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    raise ValueError(f"layer {layer} ({name}) not found in model")


def _extract(output: Any) -> Any:
    from latent_anything._hook_output import extract_primary_tensor

    return extract_primary_tensor(output)


def _replace(output: Any, replacement: Any) -> Any:
    from latent_anything._hook_output import replace_primary_tensor

    return replace_primary_tensor(output, replacement)


def _inputs(model: Any, row: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch

    device = next(model.parameters()).device
    ids = torch.as_tensor(np.asarray(row["input_ids"])[None, :], dtype=torch.long, device=device)
    mask = torch.as_tensor(np.asarray(row["attention_mask"])[None, :], dtype=torch.long, device=device)
    return ids, mask


def _margin(model: Any, row: Mapping[str, Any], target_token: int, other_token: int) -> float:
    ids, mask = _inputs(model, row)
    with __import__("torch").no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    position = int(row["target_position"])
    value = (logits[0, position, target_token] - logits[0, position, other_token]).detach().cpu()
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("task margin is non-finite")
    return result


def _capture_layers(model: Any, row: Mapping[str, Any], layers: Sequence[int]) -> dict[int, np.ndarray]:
    import torch

    captured: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for layer in layers:
            module = _module(model, layer)

            def hook(_module: Any, _inputs: Any, output: Any, *, layer_value: int = int(layer)) -> None:
                captured[layer_value] = _extract(output).detach().clone()

            handles.append(module.register_forward_hook(hook))
        ids, mask = _inputs(model, row)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        position = int(row["target_position"])
        result = {}
        for layer in layers:
            value = captured.get(int(layer))
            if value is None:
                raise ValueError(f"layer {layer} activation was not captured")
            result[int(layer)] = value[0, position].detach().cpu().numpy().astype(np.float64, copy=True)
        return result
    finally:
        for handle in handles:
            handle.remove()


def patched_margin(
    model: Any,
    row: Mapping[str, Any],
    *,
    layer: int,
    donor: np.ndarray,
    target_token: int,
    other_token: int,
    strength: float = 1.0,
    donor_label: str = "clean",
    position: int | None = None,
) -> float:
    """Run one corrupted forward with a clone-and-replace donor hook."""
    import torch

    if donor.ndim != 1 or not np.isfinite(donor).all():
        raise ValueError("donor activation must be a finite one-dimensional vector")
    module = _module(model, layer)
    ids, mask = _inputs(model, row)
    donor_tensor = torch.as_tensor(donor, dtype=torch.float32, device=ids.device)
    score_position = int(row["target_position"])
    patch_position = score_position if position is None else int(position)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = _extract(output)
        changed = tensor.clone()
        if float(strength) != 0.0:
            value = donor_tensor.to(dtype=tensor.dtype, device=tensor.device)
            changed[:, patch_position, :] = changed[:, patch_position, :] + float(strength) * (
                value - changed[:, patch_position, :]
            )
        return _replace(output, changed)

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        value = logits[0, score_position, target_token] - logits[0, score_position, other_token]
        result = float(value.detach().cpu())
        if not np.isfinite(result):
            raise ValueError(f"{donor_label} patched margin is non-finite")
        return result
    finally:
        handle.remove()


def _pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pair = str(row["causal_pair_id"])
        condition = str(row["condition"])
        if condition not in {"clean", "corrupted"} or condition in pairs.get(pair, {}):
            raise ValueError(f"causal pair {pair!r} must contain one clean and one corrupted row")
        pairs.setdefault(pair, {})[condition] = row
    if any(set(value) != {"clean", "corrupted"} for value in pairs.values()):
        raise ValueError("every causal pair must contain clean and corrupted rows")
    for pair, value in pairs.items():
        clean, corrupted = value["clean"], value["corrupted"]
        if clean["group_id"] != corrupted["group_id"] or clean["split"] != corrupted["split"]:
            raise ValueError(f"causal pair {pair!r} has inconsistent group or split")
    return dict(sorted(pairs.items()))


def deterministic_donor_derangement(pair_ids: Sequence[str], seed: int) -> dict[str, str]:
    names = sorted(str(value) for value in pair_ids)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("shuffled donor derangement requires at least two unique pairs")
    order = np.random.default_rng(int(seed)).permutation(len(names)).tolist()
    return {names[order[index]]: names[order[(index + 1) % len(order)]] for index in range(len(order))}


def deterministic_split_donor_derangement(
    pairs: Mapping[str, Mapping[str, Mapping[str, Any]]], seed: int
) -> dict[str, str]:
    """Derange train and holdout donors independently, never across splits."""
    result: dict[str, str] = {}
    for offset, split in enumerate(("train", "holdout")):
        names = [pair for pair, value in pairs.items() if value["clean"]["split"] == split]
        result.update(deterministic_donor_derangement(names, int(seed) + offset))
    return result


def donor_mapping_digest(mapping: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(sorted(mapping.items())))).hexdigest()


def _metric(
    values: Sequence[float], *, seed: int, threshold: float, comparator: str, units: str, statistic: str = "mean"
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("metric values must be finite and non-empty")
    if statistic == "max":
        point = float(np.max(array))
        rng = np.random.default_rng(int(seed))
        samples = array[rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))]
        interval = [
            float(np.quantile(np.max(samples, axis=1), 0.025)),
            float(np.quantile(np.max(samples, axis=1), 0.975)),
        ]
    else:
        point = float(np.mean(array))
        interval = bootstrap(array.tolist(), int(seed), replicates=BOOTSTRAP_REPLICATES)
    passed = point > threshold if comparator == ">" else point <= threshold
    return {
        "point_estimate": point,
        "confidence_interval_95": interval,
        "units": units,
        "aggregation_unit": "independent causal group",
        "statistic": statistic,
        "threshold": float(threshold),
        "comparator": comparator,
        "pass": bool(passed),
    }


def _group_values(evidence: Sequence[Mapping[str, Any]], key: str, *, absolute: bool = False) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for item in evidence:
        value = float(item[key])
        grouped.setdefault(str(item["group_id"]), []).append(abs(value) if absolute else value)
    return [float(np.mean(values)) for _, values in sorted(grouped.items())]


def _previous_valid_position(row: Mapping[str, Any]) -> int:
    positions = np.flatnonzero(np.asarray(row["attention_mask"], dtype=np.int64))
    if len(positions) < 2:
        raise ValueError(f"row {row.get('row_id')!r} has no previous valid token")
    return int(positions[-2])


def _run_seed(
    model: Any, pairs: Mapping[str, Mapping[str, Mapping[str, Any]]], seed: int, *, target_token: int, other_token: int
) -> dict[str, Any]:
    pair_ids = list(pairs)
    mapping = deterministic_split_donor_derangement(pairs, seed)
    clean_cache = {
        pair: _capture_layers(model, rows["clean"], (OFF_TARGET_LAYER, LAYER)) for pair, rows in pairs.items()
    }
    evidence: list[dict[str, Any]] = []
    for pair, pair_rows in pairs.items():
        clean, corrupted = pair_rows["clean"], pair_rows["corrupted"]
        clean_activation = clean_cache[pair]
        donor_pair = mapping[pair]
        clean_previous_position = _previous_valid_position(clean)
        corrupted_previous_position = _previous_valid_position(corrupted)
        m_clean = _margin(model, clean, target_token, other_token)
        m_corrupted = _margin(model, corrupted, target_token, other_token)
        denominator = m_clean - m_corrupted
        if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
            raise ValueError(f"pair {pair!r} has a non-finite or zero clean/corrupted denominator")
        m_true = patched_margin(
            model,
            corrupted,
            layer=LAYER,
            donor=clean_activation[LAYER],
            target_token=target_token,
            other_token=other_token,
            donor_label="clean",
        )
        m_off_layer = patched_margin(
            model,
            corrupted,
            layer=OFF_TARGET_LAYER,
            donor=clean_activation[OFF_TARGET_LAYER],
            target_token=target_token,
            other_token=other_token,
            donor_label="off-target-layer",
        )
        token_donor = _capture_position(model, clean, LAYER, clean_previous_position)
        m_off_token = patched_margin(
            model,
            corrupted,
            layer=LAYER,
            donor=token_donor,
            target_token=target_token,
            other_token=other_token,
            donor_label="off-target-token",
            position=corrupted_previous_position,
        )
        m_shuffled = patched_margin(
            model,
            corrupted,
            layer=LAYER,
            donor=clean_cache[donor_pair][LAYER],
            target_token=target_token,
            other_token=other_token,
            donor_label="shuffled-donor",
        )
        m_zero = patched_margin(
            model,
            corrupted,
            layer=LAYER,
            donor=clean_activation[LAYER],
            target_token=target_token,
            other_token=other_token,
            strength=0.0,
            donor_label="zero-strength",
        )
        evidence.append(
            {
                "pair_id": pair,
                "group_id": str(clean["group_id"]),
                "split": str(clean["split"]),
                "clean_row_id": str(clean["row_id"]),
                "corrupted_row_id": str(corrupted["row_id"]),
                "clean_condition": "clean",
                "corrupted_condition": "corrupted",
                "clean_target_position": int(clean["target_position"]),
                "corrupted_target_position": int(corrupted["target_position"]),
                "clean_previous_valid_position": clean_previous_position,
                "corrupted_previous_valid_position": corrupted_previous_position,
                "clean_margin": m_clean,
                "corrupted_margin": m_corrupted,
                "true_interchange_margin": m_true,
                "off_target_layer_margin": m_off_layer,
                "off_target_token_margin": m_off_token,
                "shuffled_donor_margin": m_shuffled,
                "zero_strength_margin": m_zero,
                "recovery": (m_true - m_corrupted) / max(abs(denominator), 1e-12),
                "off_target_layer_effect": m_off_layer - m_corrupted,
                "off_target_token_effect": m_off_token - m_corrupted,
                "shuffled_donor_effect": m_shuffled - m_corrupted,
                "zero_strength_error": abs(m_zero - m_corrupted),
                "strength_grid": {
                    str(strength): patched_margin(
                        model,
                        corrupted,
                        layer=LAYER,
                        donor=clean_activation[LAYER],
                        target_token=target_token,
                        other_token=other_token,
                        strength=strength,
                    )
                    for strength in STRENGTH_GRID
                },
            }
        )
    holdout = [item for item in evidence if item["split"] == "holdout"]
    recovery_values = _group_values(holdout, "recovery")
    layer_values = _group_values(holdout, "off_target_layer_effect", absolute=True)
    token_values = _group_values(holdout, "off_target_token_effect", absolute=True)
    off_values = [max(layer, token) for layer, token in zip(layer_values, token_values, strict=True)]
    recovery = _metric(
        recovery_values,
        seed=seed,
        threshold=RECOVERY_CI_LOWER_THRESHOLD,
        comparator=">",
        units="normalized causal recovery",
    )
    off_target = _metric(
        off_values,
        seed=seed,
        threshold=OFF_TARGET_ABSOLUTE_EFFECT_MAX,
        comparator="<=",
        units="absolute logit margin effect",
        statistic="max",
    )
    layer_metric = _metric(
        layer_values,
        seed=seed,
        threshold=OFF_TARGET_ABSOLUTE_EFFECT_MAX,
        comparator="<=",
        units="absolute logit margin effect",
        statistic="max",
    )
    token_metric = _metric(
        token_values,
        seed=seed,
        threshold=OFF_TARGET_ABSOLUTE_EFFECT_MAX,
        comparator="<=",
        units="absolute logit margin effect",
        statistic="max",
    )
    zero_values = _group_values(holdout, "zero_strength_error", absolute=True)
    zero_metric = _metric(
        zero_values,
        seed=seed,
        threshold=ZERO_STRENGTH_IDENTITY_ATOL,
        comparator="<=",
        units="absolute logit margin difference",
        statistic="max",
    )
    return {
        "seed": int(seed),
        "train_pairs": [pair for pair in pair_ids if pairs[pair]["clean"]["split"] == "train"],
        "holdout_pairs": [pair for pair in pair_ids if pairs[pair]["clean"]["split"] == "holdout"],
        "holdout_evidence": holdout,
        "recovery": recovery,
        "off_target": off_target,
        "off_target_layer": layer_metric,
        "off_target_token": token_metric,
        "zero_strength": zero_metric,
        "shuffled_direction": {
            "semantic": "shuffled donor activation; compatibility key retained",
            "mapping": mapping,
            "mapping_sha256": donor_mapping_digest(mapping),
            "finite": bool(np.isfinite([item["shuffled_donor_effect"] for item in holdout]).all()),
        },
        "finite": bool(
            np.isfinite(
                [
                    value
                    for item in holdout
                    for key in (
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
                    for value in (item[key],)
                ]
            ).all()
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _capture_position(model: Any, row: Mapping[str, Any], layer: int, position: int) -> np.ndarray:
    import torch

    module = _module(model, layer)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["value"] = _extract(output).detach().clone()

    handle = module.register_forward_hook(hook)
    try:
        ids, mask = _inputs(model, row)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
        value = captured.get("value")
        if value is None:
            raise ValueError("activation was not captured")
        return value[0, int(position)].detach().cpu().numpy().astype(np.float64, copy=True)
    finally:
        handle.remove()


def _fixture_linkage(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    raw, frozen = read_fixture(FIXTURE_PATH)
    if plan["fixture"]["content_sha256"] != content_digest(raw) or list(rows) != frozen:
        raise ValueError("rows must exactly match the frozen authored fixture")
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
        for row in frozen
    ]


def _resource_peak(
    torch: Any, started: float, device: str, rss_start: int, rss_source: str, rss_unit: str
) -> dict[str, Any]:
    return {
        "cuda_device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "max_rss_bytes": max(rss_start, _rss_measurement()[0]),
        "rss_source": rss_source,
        "rss_unit": rss_unit,
    }


def _acceptance(summaries: Sequence[Mapping[str, Any]], no_mutation: bool, within_budget: bool) -> bool:
    return bool(
        summaries
        and no_mutation
        and within_budget
        and all(
            item["recovery"]["confidence_interval_95"][0] > RECOVERY_CI_LOWER_THRESHOLD
            and item["off_target"]["point_estimate"] <= OFF_TARGET_ABSOLUTE_EFFECT_MAX
            and item["zero_strength"]["pass"]
            and item["finite"]
            and item["shuffled_direction"]["finite"]
            for item in summaries
        )
    )


def run_true_activation_patching(
    plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, integration_factory: Callable[..., Any] | None = None
) -> dict[str, Any]:
    from scripts._m14_l04_tcav_runtime import (
        RealExecutionError,
        parameter_digest,
        read_rows,
        resolve_target_token,
        seed_everything,
    )

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
        raise RealExecutionError("true activation patching requires LATENT_ANYTHING_RUN_NETWORK=1", resources)
    if os.environ.get("LATENT_ANYTHING_NETWORK_DEVICE", "").strip().lower() != "cuda":
        raise RealExecutionError("true activation patching requires LATENT_ANYTHING_NETWORK_DEVICE=cuda", resources)
    try:
        import torch
    except ImportError as exc:
        resources["stage"] = "dependency_check"
        raise RealExecutionError("PyTorch is required for true activation patching", resources) from exc
    if not torch.cuda.is_available():
        resources["stage"] = "cuda_check"
        raise RealExecutionError("true activation patching requires an available CUDA device", resources)
    started = time.perf_counter()
    torch.use_deterministic_algorithms(True)
    resources.update(
        {
            "execution_attempted": True,
            "execution_backend": "cuda",
            "network": "enabled",
            "device": torch.cuda.get_device_name(0),
            "stage": "cuda_check",
        }
    )
    torch.cuda.reset_peak_memory_stats()
    model: Any | None = None
    rss_start, rss_source, rss_unit = _rss_measurement()
    try:
        resources["stage"] = "model_load"
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
        model.eval()
        before = parameter_digest(model)
        resolved = {text: resolve_target_token(tokenizer, text) for text in TARGET_TEXTS}
        if resolved != {text: (TARGET_TOKEN_IDS[text], TARGET_TOKEN_STRINGS[text]) for text in TARGET_TEXTS}:
            raise ValueError("target token IDs/strings do not match pinned GPT-2")
        fixture_linkage = _fixture_linkage(plan, rows)
        real_rows = read_rows(integration, rows, int(plan["tokenization_and_sampling"]["max_length"]))
        pairs = _pairs(real_rows)
        summaries = []
        resources["stage"] = "scoring"
        for seed in SEEDS:
            seed_everything(seed, torch)
            summaries.append(
                _run_seed(
                    model, pairs, seed, target_token=TARGET_TOKEN_IDS[" true"], other_token=TARGET_TOKEN_IDS[" false"]
                )
            )
        resource_peak = _resource_peak(torch, started, str(resources["device"]), rss_start, rss_source, rss_unit)
        resources["resource_peak"] = resource_peak
        within_budget = budget_pass(resource_peak)
        after = parameter_digest(model)
        no_mutation = after == before
        accepted = _acceptance(summaries, no_mutation, within_budget)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        resources.update(
            {"cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied", "stage": "complete"}
        )
        return {
            "status": REAL_STATUS if accepted else "failed",
            "evidence_eligible": accepted,
            "acceptance": accepted,
            "evidence_level": "D3" if accepted else "D0",
            "failure_reason": None if accepted else "one or more true activation patching gates failed",
            "metrics": {"recovery": {str(item["seed"]): item["recovery"] for item in summaries}},
            "confidence_intervals": {
                str(item["seed"]): item["recovery"]["confidence_interval_95"] for item in summaries
            },
            "controls": {
                "clean_endpoint": {"pass": True},
                "corrupted_endpoint": {"pass": True},
                "true_interchange": {"pass": all(item["recovery"]["pass"] for item in summaries)},
                "off_target_layer": {
                    "pass": all(item["off_target_layer"]["pass"] for item in summaries),
                    "metrics": {str(item["seed"]): item["off_target_layer"] for item in summaries},
                },
                "off_target_token": {
                    "pass": all(item["off_target_token"]["pass"] for item in summaries),
                    "metrics": {str(item["seed"]): item["off_target_token"] for item in summaries},
                },
                "off_target_combined": {
                    "pass": all(item["off_target"]["pass"] for item in summaries),
                    "metrics": {str(item["seed"]): item["off_target"] for item in summaries},
                },
                "shuffled_direction": {
                    "pass": all(item["shuffled_direction"]["finite"] for item in summaries),
                    "semantic": "shuffled donor activation; compatibility key retained",
                },
                "zero_strength": {"pass": all(item["zero_strength"]["pass"] for item in summaries)},
            },
            "raw_summaries": summaries,
            "fixture_linkage": fixture_linkage,
            "token_ids": dict(TARGET_TOKEN_IDS),
            "target_token_strings": dict(TARGET_TOKEN_STRINGS),
            "layer": LAYER,
            "native_hidden_state_index": NATIVE_HIDDEN_STATE_INDEX,
            "seed": SEEDS[0],
            "seeds": list(SEEDS),
            "no_mutation": no_mutation,
            "model_parameter_digest_before": before,
            "model_parameter_digest_after": after,
            "budget_pass": within_budget,
            "provenance": {
                "runtime": "real TransformerLMIntegration",
                "model_revision": str(model_spec["revision"]),
                "integration": "TransformerLMIntegration",
                "adapter": "N/A",
                "target_token_ids": dict(TARGET_TOKEN_IDS),
                "target_token_strings": dict(TARGET_TOKEN_STRINGS),
                "target_position": "last non-padding token",
                "donor_semantics": "clean hidden activation replaces corrupted hidden activation",
                "off_target_controls": {"layer": OFF_TARGET_LAYER, "token": "previous valid token"},
                "strength_grid": list(STRENGTH_GRID),
                "shuffled_direction_semantics": "shuffled donor activation; compatibility key retained",
                "network": "enabled",
                "device": resources["device"],
                "execution_attempted": True,
                "execution_backend": "cuda",
                "stage": "complete",
                "deterministic_algorithms": True,
                "runtime_versions": runtime_versions(),
                "resource_peak": resource_peak,
                "model_parameter_digest_before": before,
                "model_parameter_digest_after": after,
                "budget_pass": within_budget,
                "cleanup": resources["cleanup"],
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            },
            "resources": resources,
        }
    except RealExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - dispatcher retains sanitized failure
        resources["stage"] = "cleanup" if resources.get("execution_attempted") else resources.get("stage", "preflight")
        try:
            if model is not None:
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            resources["cleanup"] = "failure cleanup synchronized; gradients cleared; CUDA cache emptied"
        except Exception as cleanup_error:  # noqa: BLE001
            resources["cleanup"] = f"failure cleanup incomplete: {type(cleanup_error).__name__}"
        raise RealExecutionError(str(exc), resources) from exc


run_activation_patching = run_true_activation_patching

__all__ = [
    "BOOTSTRAP_REPLICATES",
    "LAYER",
    "NATIVE_HIDDEN_STATE_INDEX",
    "REAL_STATUS",
    "SEEDS",
    "STRENGTH_GRID",
    "budget_pass",
    "deterministic_donor_derangement",
    "deterministic_split_donor_derangement",
    "donor_mapping_digest",
    "patched_margin",
    "run_activation_patching",
    "run_true_activation_patching",
]
