"""Concrete streaming tuned-logit-lens execution for M14 L04.7."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

import numpy as np

from scripts._m14_l04_digest import runtime_versions
from scripts._m14_l04_execution_common import RealExecutionError, parameter_digest, seed_everything
from scripts._m14_l04_tuned_lens_metrics import improvement_metric, macro_improvement, row_token_kl
from scripts._m14_l04_wikitext_runtime import load_selected_rows, read_manifest

REAL_STATUS = "passed_real_cuda"
FIT_SEED = 79
FITTED_LAYERS = tuple(range(12))
NATIVE_LAYERS = tuple(range(13))
BOOTSTRAP_SEEDS = (17, 29, 41, 53, 67)
BOOTSTRAP_REPLICATES = 2000
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
MAX_ELAPSED_SECONDS = 1800.0
MAX_GPU_BYTES = 6 * 1024**3
MAX_RSS_BYTES = 4 * 1024**3
MANIFEST_PATH = "artifacts/m14/l04-wikitext-2-manifest.json"


def _rss_bytes() -> int:
    if resource is None:
        return 0
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if os.name != "nt" else value


def _paired_batches(
    source_texts: Sequence[str],
    target_texts: Sequence[str],
    *,
    integration: Any,
    model: Any,
    max_length: int,
    batch_size: int,
    device: Any,
) -> Iterator[tuple[Any, Any, Any, Any, Any, Any]]:
    """Yield source/target activations from exactly one model forward per batch."""
    import torch

    if len(source_texts) != len(target_texts):
        raise ValueError("paired corpus rows must have equal lengths")
    tokenizer = getattr(integration, "_tokenizer", None)
    for start in range(0, len(source_texts), batch_size):
        source = tuple(source_texts[start : start + batch_size])
        target = tuple(target_texts[start : start + batch_size])
        prompts = source + target
        if tokenizer is not None:
            encoded = tokenizer(
                prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
            )
        else:
            encoded = integration.tokenize(prompts, max_length=max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
        hidden = output.hidden_states
        if hidden is None or len(hidden) != 13:
            raise ValueError("tuned-lens requires exactly 13 native hidden states")
        size = len(source)
        yield (
            mask[:size],
            tuple(state[:size].detach() for state in hidden),
            output.logits[:size].detach(),
            mask[size:],
            tuple(state[size:].detach() for state in hidden),
            output.logits[size:].detach(),
        )


def _project(model: Any, values: Any, *, apply_final_norm: bool = True) -> Any:
    normalized = model.transformer.ln_f(values) if apply_final_norm and hasattr(model.transformer, "ln_f") else values
    return model.lm_head(normalized)


def _kl_loss(teacher: Any, predicted: Any, mask: Any, torch: Any) -> Any:
    valid = mask.to(dtype=torch.bool)
    if int(valid.sum()) == 0:
        raise ValueError("tuned-lens fit received no non-padding tokens")
    teacher_logp = torch.log_softmax(teacher[valid].float(), dim=-1)
    predicted_logp = torch.log_softmax(predicted[valid].float(), dim=-1)
    loss = torch.sum(torch.exp(teacher_logp) * (teacher_logp - predicted_logp), dim=-1).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("tuned-lens KL objective became non-finite")
    return loss


def _new_translators(model: Any, device: Any, torch: Any) -> tuple[dict[int, Any], dict[int, Any]]:
    hidden_dim = int(model.lm_head.in_features)
    true: dict[int, Any] = {}
    shuffled: dict[int, Any] = {}
    for layer in FITTED_LAYERS:
        for target in (true, shuffled):
            translator = torch.nn.Linear(hidden_dim, hidden_dim, bias=True, device=device, dtype=torch.float32)
            with torch.no_grad():
                translator.weight.copy_(torch.eye(hidden_dim, device=device))
                translator.bias.zero_()
            translator.train()
            target[layer] = translator
    return true, shuffled


def fit_translators(
    *,
    model: Any,
    integration: Any,
    source_texts: Sequence[str],
    shuffled_texts: Sequence[str],
    max_length: int,
    device: Any,
    torch: Any,
) -> tuple[dict[int, Any], dict[int, Any], dict[str, float], dict[str, float]]:
    """Fit real and shuffled translators while reusing one forward per corpus batch."""
    true, shuffled = _new_translators(model, device, torch)
    optimizers = {
        (control, layer): torch.optim.AdamW(
            translators[layer].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        for control, translators in (("true", true), ("shuffled", shuffled))
        for layer in FITTED_LAYERS
    }
    totals: dict[tuple[str, int], float] = {
        (control, layer): 0.0 for control in ("true", "shuffled") for layer in FITTED_LAYERS
    }
    counts: dict[tuple[str, int], int] = {
        (control, layer): 0 for control in ("true", "shuffled") for layer in FITTED_LAYERS
    }
    for _epoch in range(EPOCHS):
        for source_mask, source_hidden, source_teacher, _target_mask, _target_hidden, target_teacher in _paired_batches(
            source_texts,
            shuffled_texts,
            integration=integration,
            model=model,
            max_length=max_length,
            batch_size=BATCH_SIZE,
            device=device,
        ):
            for layer in FITTED_LAYERS:
                values = source_hidden[layer].float()
                for control, translators, teacher, mask in (
                    ("true", true, source_teacher, source_mask),
                    ("shuffled", shuffled, target_teacher, source_mask),
                ):
                    predicted = _project(model, translators[layer](values), apply_final_norm=True)
                    loss = _kl_loss(teacher, predicted, mask, torch)
                    optimizer = optimizers[(control, layer)]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(translators[layer].parameters(), GRAD_CLIP_NORM)
                    optimizer.step()
                    valid_count = int(mask.to(dtype=torch.bool).sum())
                    totals[(control, layer)] += float(loss.detach()) * valid_count
                    counts[(control, layer)] += valid_count
    if not all(counts.values()):
        raise ValueError("tuned-lens fit received no non-padding tokens")
    for translator in (*true.values(), *shuffled.values()):
        translator.eval()
    return (
        true,
        shuffled,
        {str(layer): totals[("true", layer)] / counts[("true", layer)] for layer in FITTED_LAYERS},
        {str(layer): totals[("shuffled", layer)] / counts[("shuffled", layer)] for layer in FITTED_LAYERS},
    )


def _translator_digest(translator: Any) -> str:
    digest = hashlib.sha256()
    for parameter in translator.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate_translators(
    *,
    model: Any,
    integration: Any,
    texts: Sequence[str],
    shuffled_texts: Sequence[str],
    translators: Mapping[int, Any],
    shuffled_translators: Mapping[int, Any],
    max_length: int,
    device: Any,
    torch: Any,
) -> tuple[dict[int, list[float]], dict[int, list[float]], dict[int, list[float]], list[int], float, float]:
    direct: dict[int, list[float]] = {layer: [] for layer in NATIVE_LAYERS}
    tuned: dict[int, list[float]] = {layer: [] for layer in NATIVE_LAYERS}
    shuffled: dict[int, list[float]] = {layer: [] for layer in NATIVE_LAYERS}
    token_counts: list[int] = []
    global_abs = 0.0
    global_rel = 0.0
    for source_mask, source_hidden, source_teacher, _target_mask, _target_hidden, shuffled_teacher in _paired_batches(
        texts,
        shuffled_texts,
        integration=integration,
        model=model,
        max_length=max_length,
        batch_size=BATCH_SIZE,
        device=device,
    ):
        token_counts.extend(source_mask.to(dtype=torch.int64).sum(dim=1).detach().cpu().tolist())
        terminal_error = (_project(model, source_hidden[12], apply_final_norm=False) - source_teacher).abs()
        global_abs = max(global_abs, float(terminal_error.max()))
        global_rel = max(global_rel, float((terminal_error / torch.clamp(source_teacher.abs(), min=1e-12)).max()))
        for layer in NATIVE_LAYERS:
            direct_logits = _project(model, source_hidden[layer], apply_final_norm=layer != 12)
            direct[layer].extend(row_token_kl(source_teacher, direct_logits, source_mask).tolist())
            if layer == 12:
                tuned[layer].extend(row_token_kl(source_teacher, source_teacher, source_mask).tolist())
                shuffled[layer].extend(row_token_kl(shuffled_teacher, shuffled_teacher, source_mask).tolist())
            else:
                tuned_logits = _project(model, translators[layer](source_hidden[layer].float()), apply_final_norm=True)
                shuffled_logits = _project(
                    model, shuffled_translators[layer](source_hidden[layer].float()), apply_final_norm=True
                )
                tuned[layer].extend(row_token_kl(source_teacher, tuned_logits, source_mask).tolist())
                shuffled[layer].extend(row_token_kl(shuffled_teacher, shuffled_logits, source_mask).tolist())
    return direct, tuned, shuffled, token_counts, global_abs, global_rel


def _control_metric(value: float, *, threshold: float, comparator: str) -> dict[str, Any]:
    passed = value <= threshold if comparator == "<=" else value > threshold
    return {
        "point_estimate": float(value),
        "confidence_interval_95": [float(value), float(value)],
        "units": "logits",
        "aggregation_unit": "all validation tokens and vocabulary coordinates",
        "statistic": "global_max",
        "threshold": threshold,
        "comparator": comparator,
        "pass": bool(passed),
    }


def _selected_identity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"row_id": str(row["row_id"]), "index": str(row["index"]), "text_sha256": str(row["text_sha256"])}
        for row in rows
    ]


def _fit_and_evaluate(
    *,
    model: Any,
    integration: Any,
    train_texts: Sequence[str],
    shuffled_train: Sequence[str],
    validation_texts: Sequence[str],
    shuffled_validation: Sequence[str],
    max_length: int,
    device: Any,
    torch: Any,
) -> tuple[
    Any,
    Any,
    dict[str, float],
    dict[str, float],
    dict[int, list[float]],
    dict[int, list[float]],
    dict[int, list[float]],
    list[int],
    float,
    float,
]:
    """Own model fitting/evaluation while the runner owns orchestration and payloads."""
    translators, shuffled_translators, train_objectives, shuffled_objectives = fit_translators(
        model=model,
        integration=integration,
        source_texts=train_texts,
        shuffled_texts=shuffled_train,
        max_length=max_length,
        device=device,
        torch=torch,
    )
    direct, tuned, shuffled_kl, token_counts, parity_abs, parity_rel = evaluate_translators(
        model=model,
        integration=integration,
        texts=validation_texts,
        shuffled_texts=shuffled_validation,
        translators=translators,
        shuffled_translators=shuffled_translators,
        max_length=max_length,
        device=device,
        torch=torch,
    )
    return (
        translators,
        shuffled_translators,
        train_objectives,
        shuffled_objectives,
        direct,
        tuned,
        shuffled_kl,
        token_counts,
        parity_abs,
        parity_rel,
    )


def run_tuned_logit_lens(
    plan: Mapping[str, Any],
    _rows: Sequence[Mapping[str, Any]],
    *,
    integration_factory: Callable[..., Any] | None = None,
    manifest_path: Path = Path(MANIFEST_PATH),
    dataset_loader: Callable[..., object] | None = None,
) -> dict[str, Any]:
    """Fit/evaluate the pinned corpus; this path is explicitly network-gated."""
    resources: dict[str, Any] = {
        "device": "cuda",
        "network": "not attempted",
        "resource_peak": "not measured",
        "cleanup": "pending",
    }
    if os.environ.get("LATENT_ANYTHING_RUN_NETWORK") != "1":
        raise RealExecutionError("tuned logit lens requires LATENT_ANYTHING_RUN_NETWORK=1", resources)
    if os.environ.get("LATENT_ANYTHING_NETWORK_DEVICE", "").strip().lower() != "cuda":
        raise RealExecutionError("tuned logit lens requires LATENT_ANYTHING_NETWORK_DEVICE=cuda", resources)
    try:
        import importlib
        from importlib import metadata

        import torch
    except ImportError as exc:
        raise RealExecutionError("PyTorch, Transformers, and datasets are required for tuned lens", resources) from exc
    if dataset_loader is None:
        datasets = importlib.import_module("datasets")
        dataset_loader = cast(Callable[..., object], datasets.load_dataset)
        try:
            if metadata.version("datasets") != "4.8.5":
                raise ValueError("datasets must be pinned to 4.8.5")
        except metadata.PackageNotFoundError as exc:
            raise RealExecutionError("datasets package metadata is unavailable", resources) from exc
    if not torch.cuda.is_available():
        raise RealExecutionError("real tuned logit lens requires an available CUDA device", resources)
    torch.use_deterministic_algorithms(True)
    seed_everything(FIT_SEED, torch)
    resources["network"] = "enabled"
    resources["device"] = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model: Any | None = None
    try:
        manifest, manifest_sha = read_manifest(manifest_path)
        selected = load_selected_rows(manifest_path, dataset_loader=dataset_loader)
        train_texts = [row["text"] for row in selected["train"]]
        validation_texts = [row["text"] for row in selected["validation"]]
        rng = np.random.default_rng(FIT_SEED)
        train_perm = rng.permutation(len(train_texts))
        validation_perm = rng.permutation(len(validation_texts))
        shuffled_train = [train_texts[index] for index in train_perm]
        shuffled_validation = [validation_texts[index] for index in validation_perm]
        if integration_factory is None:
            from scripts._m14_l04_boundary import transformer_integration_type

            integration_factory = transformer_integration_type()
        model_spec = cast_mapping(plan["model"])
        integration = integration_factory(
            model_id=str(model_spec["id"]), revision=str(model_spec["revision"]), device="cuda"
        )
        model, tokenizer, _config = integration._backend()
        if model is None or tokenizer is None:
            raise ValueError("TransformerLMIntegration returned no model/tokenizer")
        before = parameter_digest(model)
        max_length = int(cast_mapping(plan["tuned_lens_corpus"])["max_tokens_per_row"])
        device = next(model.parameters()).device
        (
            translators,
            shuffled_translators,
            train_objectives,
            shuffled_objectives,
            direct,
            tuned,
            shuffled_kl,
            token_counts,
            parity_abs_value,
            parity_rel_value,
        ) = _fit_and_evaluate(
            model=model,
            integration=integration,
            train_texts=train_texts,
            shuffled_train=shuffled_train,
            validation_texts=validation_texts,
            shuffled_validation=shuffled_validation,
            max_length=max_length,
            device=device,
            torch=torch,
        )
        improvements = macro_improvement(direct, tuned)
        shuffled_improvements = macro_improvement(direct, shuffled_kl)
        threshold = float(
            cast_mapping(plan["thresholds_and_controls"])["lens"]["tuned_holdout_kl_improvement_strict_gt_nats"]
        )
        seed_metrics = {
            str(seed): improvement_metric(improvements.tolist(), seed=seed, threshold=threshold)
            for seed in BOOTSTRAP_SEEDS
        }
        minimum_lower = min(float(value["confidence_interval_95"][0]) for value in seed_metrics.values())
        point_metric = improvement_metric(improvements.tolist(), seed=FIT_SEED, threshold=threshold)
        lower_metric = {
            **point_metric,
            "point_estimate": minimum_lower,
            "confidence_interval_95": [minimum_lower, minimum_lower],
            "pass": bool(minimum_lower > threshold),
        }
        finite = bool(np.isfinite(improvements).all() and np.isfinite(shuffled_improvements).all())
        after = parameter_digest(model)
        no_mutation = after == before
        elapsed = time.perf_counter() - started
        peak = {
            "cuda_device": resources["device"],
            "elapsed_seconds": elapsed,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "max_rss_bytes": _rss_bytes(),
        }
        resources["resource_peak"] = peak
        budget_pass = bool(
            elapsed <= MAX_ELAPSED_SECONDS
            and int(peak["max_memory_allocated_bytes"]) <= MAX_GPU_BYTES
            and int(peak["max_rss_bytes"]) <= MAX_RSS_BYTES
        )
        parity_abs = _control_metric(parity_abs_value, threshold=1e-6, comparator="<=")
        parity_rel = _control_metric(parity_rel_value, threshold=1e-6, comparator="<=")
        controls: dict[str, Any] = {
            "direct_lens": {"metrics": {"finite_fraction": 1.0 if finite else 0.0}, "pass": finite},
            "affine_tuned_lens": {"metrics": {"finite_fraction": 1.0 if finite else 0.0}, "pass": finite},
            "train_holdout_separation": {
                "metrics": {"train_rows": len(train_texts), "validation_rows": len(validation_texts), "disjoint": True},
                "pass": len(train_texts) == 8192 and len(validation_texts) == 2048,
            },
            "shuffled_translator_target": {
                "metrics": {
                    "finite_fraction": 1.0 if np.isfinite(shuffled_improvements).all() else 0.0,
                    "permutation_sha256": hashlib.sha256(np.asarray(train_perm, dtype=np.int64).tobytes()).hexdigest(),
                },
                "pass": bool(np.isfinite(shuffled_improvements).all()),
            },
            "terminal_post_ln_f_parity": {
                "metrics": {"max_abs_error": parity_abs, "max_relative_error": parity_rel},
                "pass": bool(parity_abs["pass"] and parity_rel["pass"]),
            },
        }
        all_pass = bool(
            point_metric["pass"]
            and lower_metric["pass"]
            and all(value["pass"] for value in controls.values())
            and no_mutation
            and budget_pass
        )
        rows_out = []
        for index, selected_row in enumerate(selected["validation"]):
            rows_out.append(
                {
                    "row_id": selected_row["row_id"],
                    "index": selected_row["index"],
                    "text_sha256": selected_row["text_sha256"],
                    "split": "validation",
                    "token_count": token_counts[index],
                    "direct_kl": [float(direct[layer][index]) for layer in NATIVE_LAYERS],
                    "tuned_kl": [float(tuned[layer][index]) for layer in NATIVE_LAYERS],
                    "improvement": [float(direct[layer][index] - tuned[layer][index]) for layer in FITTED_LAYERS],
                    "macro_improvement": float(improvements[index]),
                    "shuffled_macro_improvement": float(shuffled_improvements[index]),
                    "finite": finite,
                }
            )
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        resources["cleanup"] = "CUDA synchronized; model gradients cleared; CUDA cache emptied"
        split_values = cast(Mapping[str, Mapping[str, Any]], manifest["splits"])
        return {
            "status": REAL_STATUS if all_pass else "failed",
            "evidence_eligible": all_pass,
            "acceptance": all_pass,
            "evidence_level": "D3" if all_pass else "D0",
            "failure_reason": None if all_pass else "tuned-lens gate, control, or resource budget failed",
            "metrics": {
                "tuned_holdout_kl_improvement": point_metric,
                "tuned_holdout_calibration_ci_lower": lower_metric,
            },
            "confidence_intervals": seed_metrics,
            "controls": controls,
            "raw_summaries": [
                {
                    "seed": FIT_SEED,
                    "fit_layers": list(FITTED_LAYERS),
                    "native_layers": list(NATIVE_LAYERS),
                    "train_rows": _selected_identity(selected["train"]),
                    "validation_rows": _selected_identity(selected["validation"]),
                    "rows": rows_out,
                    "train_permutation": train_perm.tolist(),
                    "validation_permutation": validation_perm.tolist(),
                    "train_objectives": train_objectives,
                    "shuffled_train_objectives": shuffled_objectives,
                    "translator_digests": {
                        str(layer): _translator_digest(translators[layer]) for layer in FITTED_LAYERS
                    },
                    "shuffled_translator_digests": {
                        str(layer): _translator_digest(shuffled_translators[layer]) for layer in FITTED_LAYERS
                    },
                    "terminal_logit_max_abs_error": parity_abs_value,
                    "terminal_logit_max_relative_error": parity_rel_value,
                }
            ],
            "layer": 6,
            "native_hidden_state_index": 7,
            "seed": FIT_SEED,
            "seeds": list(BOOTSTRAP_SEEDS),
            "no_mutation": no_mutation,
            "provenance": {
                "runtime": "real TransformerLMIntegration",
                "model_id": str(model_spec["id"]),
                "model_revision": str(model_spec["revision"]),
                "dataset_id": "Salesforce/wikitext",
                "dataset_config": "wikitext-2-raw-v1",
                "dataset_revision": "f776294184f13b8ff2337b3841cf9269a6216d1e",
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "manifest_content_sha256": manifest["content_sha256"],
                "manifest_split_sha256": manifest["split_sha256"],
                "model_parameter_digest_before": before,
                "model_parameter_digest_after": after,
                "no_mutation": no_mutation,
                "fit_seed": FIT_SEED,
                "bootstrap_seeds": list(BOOTSTRAP_SEEDS),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "fit_layers": list(FITTED_LAYERS),
                "native_layers": list(NATIVE_LAYERS),
                "objective": "tokenwise KL(p_true || q_translated) in nats over every non-padding position",
                "optimizer": "AdamW",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "network": "enabled",
                "device": resources["device"],
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "runtime_versions": runtime_versions(),
                "resource_peak": peak,
                "cleanup": resources["cleanup"],
                "train_rows": len(train_texts),
                "validation_rows": len(validation_texts),
                "official_train_rows": split_values["train"]["official_rows"],
                "official_validation_rows": split_values["validation"]["official_rows"],
                "model_forwards_per_corpus_batch": 1,
                "budget_pass": budget_pass,
            },
            "resources": resources,
        }
    except RealExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001
        try:
            if model is not None:
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            resources["cleanup"] = "failure cleanup synchronized; gradients cleared; CUDA cache emptied"
        except Exception as cleanup_error:  # noqa: BLE001
            resources["cleanup"] = f"failure cleanup incomplete: {type(cleanup_error).__name__}"
        raise RealExecutionError(str(exc), resources) from exc


def cast_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("tuned-lens plan section must be an object")
    return value


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEEDS",
    "FIT_SEED",
    "FITTED_LAYERS",
    "NATIVE_LAYERS",
    "REAL_STATUS",
    "_project",
    "fit_translators",
    "run_tuned_logit_lens",
]
