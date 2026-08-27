"""Thin L03 orchestrator; ``--check`` is always side-effect free."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from typing import Any

from latent_anything.mlp_probe import MLPProbeConfig, nonlinear_memorization_test
from latent_anything.probes import LinearProbeConfig
from scripts.m14_l03_data import array_digest, grouped_digit_split
from scripts.m14_l03_envelope import (
    apply_dependency_blocking,
    build_artifact,
    build_run_record,
    failure_envelope,
    git_sha,
    runtime_versions,
    source_digests,
    validate_artifact,
)
from scripts.m14_l03_features import extract_batched
from scripts.m14_l03_finalize import build_report, validate_report
from scripts.m14_l03_metrics import compression_ok, evaluate_linear, evaluate_mlp, fit_train_only_pca
from scripts.m14_l03_plan import load_plan, section


def check() -> dict[str, Any]:
    """Validate the immutable plan and cheap split contract only."""
    plan = load_plan()
    split = grouped_digit_split(int(section(plan, "split")["seed"]))
    return {"plan_sha256": str(plan["plan_sha256"]), "split": split["metadata"]}


def run_real() -> dict[str, Any]:
    """Run the remote-only real model lane and write evidence after validation."""
    plan = load_plan()
    run_resources: dict[str, Any] = {}
    try:
        split = grouped_digit_split(int(section(plan, "split")["seed"]))
        from latent_anything.clustering import KMeans, KMeansConfig
        from latent_anything.integrations.transformer_lm import TransformerGenerationRequest, TransformerLMIntegration

        model_spec = section(plan, "model")
        model = TransformerLMIntegration(str(model_spec["model_id"]), str(model_spec["revision"]), device="cuda")
        prompts = tuple(str(prompt) for prompt in split["prompts"])
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real L03 lane")
        torch.cuda.reset_peak_memory_stats()

        def request_factory(batch: tuple[str, ...], *, max_length: int, layers: tuple[int, ...]) -> Any:
            return TransformerGenerationRequest(
                prompt=batch,
                max_length=max_length,
                seed=79,
                capture_hidden_states=True,
                capture_layers=layers,
                top_k_logit_lens=0,
            )

        hidden, inference = extract_batched(
            model,
            prompts,
            layers=(0, 4, 8, 12),
            max_length=64,
            batch_size=int(model_spec["inference_batch_size"]),
            request_factory=request_factory,
        )
        inference["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        inference["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        inference["gpu_name"] = str(torch.cuda.get_device_name())
        run_resources = {**inference, "host_rss": "not measured by runner"}
        masks = split["partitions"]
        linear_cfg = LinearProbeConfig(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            test_size=0.2,
            val_size=0.25,
            random_state=79,
            standardize=True,
            class_weight="balanced",
            fit_intercept=True,
        )
        primary = hidden[12]
        clustering = KMeans(
            KMeansConfig(n_clusters=10, n_init=10, max_iter=300, random_state=79, standardize=True)
        ).fit_predict(primary[masks["train"]], provenance={"role": "diagnostic-only", "fit_scope": "train rows only"})
        layer_metrics = {
            str(layer): evaluate_linear(hidden[layer], split["labels"], masks, linear_cfg) for layer in (0, 4, 8, 12)
        }
        linear = layer_metrics["12"]
        pca_features, pca_meta = fit_train_only_pca(primary, masks, components=32)
        pca = evaluate_linear(pca_features, split["labels"], masks, linear_cfg)
        raw_features = split["images"].reshape(len(split["images"]), -1)
        raw_glyph = evaluate_linear(raw_features, split["labels"], masks, linear_cfg)
        mlp_cfg = MLPProbeConfig(
            hidden_sizes=[16],
            activation="relu",
            max_epochs=80,
            early_stopping_patience=10,
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=64,
            test_size=0.2,
            val_size=0.25,
            random_state=79,
            standardize=True,
        )
        mlp = evaluate_mlp(pca_features, split["labels"], masks, mlp_cfg)
        capacity = {
            str(width): evaluate_mlp(
                pca_features,
                split["labels"],
                masks,
                mlp_cfg.model_copy(update={"hidden_sizes": [width]}),
            )
            for width in (4, 64)
        }
        api_diagnostic = nonlinear_memorization_test(
            pca_features, split["labels"], config=mlp_cfg, selectivity_threshold=2.0
        )
        linear_ok = (
            linear["test_accuracy"] >= 0.20 and linear["bootstrap"]["lower"] > 0.05 and linear["wilson_95"][0] > 0.15
        )
        pca_ok = (
            linear["test_accuracy"] >= 0.20
            and pca["test_accuracy"] >= 0.20
            and pca["bootstrap"]["lower"] > 0.05
            and pca["wilson_95"][0] > 0.15
            and compression_ok(pca["test_accuracy"], linear["test_accuracy"])
        )
        mlp_ok = (
            mlp["test_accuracy"] >= 0.20
            and mlp["bootstrap"]["lower"] > 0.05
            and mlp["control_accuracy"] <= 0.15
            and abs(mlp["val_accuracy"] - mlp["test_accuracy"]) <= 0.15
            and mlp["n_params"] / int(masks["train"].sum()) <= 4
        )
        records = apply_dependency_blocking(
            [
                {
                    "record_id": plan["records"][0]["record_id"],
                    "gap_id": plan["records"][0]["gap_id"],
                    "accepted": pca_ok,
                    "metrics": {
                        "full_hidden": linear,
                        "declared_layer_diagnostics": layer_metrics,
                        "pca32": pca,
                        "pca": pca_meta,
                        "raw_glyph_diagnostic": raw_glyph,
                        "kmeans_diagnostic": clustering.to_dict(),
                    },
                },
                {
                    "record_id": plan["records"][1]["record_id"],
                    "gap_id": plan["records"][1]["gap_id"],
                    "accepted": linear_ok,
                    "metrics": linear,
                },
                {
                    "record_id": plan["records"][2]["record_id"],
                    "gap_id": plan["records"][2]["gap_id"],
                    "accepted": mlp_ok,
                    "metrics": {
                        "primary": mlp,
                        "capacity_diagnostics": capacity,
                        "api_shuffled_label_diagnostic": asdict(api_diagnostic),
                    },
                },
            ]
        )
        resources = run_resources
        _backend_model, tokenizer, model_config = model._backend()  # pyright: ignore[reportPrivateUsage]
        provenance = {
            "git_sha": git_sha(),
            **source_digests(),
            "model_id": model.model_id,
            "model_revision": model.revision,
            "model": model.provenance,
            "model_url": f"https://huggingface.co/{model.model_id}/tree/{model.revision}",
            "model_license_url": f"https://huggingface.co/{model.model_id}/blob/{model.revision}/LICENSE",
            "dataset_source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
            "dataset_license": "BSD-3-Clause",
            "dataset_license_url": "https://github.com/scikit-learn/scikit-learn/blob/main/COPYING",
            "tokenizer": {
                "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
                "max_length": 64,
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
                "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            },
            "model_config": {
                "class": f"{model_config.__class__.__module__}.{model_config.__class__.__name__}",
                "n_layer": getattr(model_config, "n_layer", None),
                "n_head": getattr(model_config, "n_head", None),
                "n_embd": getattr(model_config, "n_embd", None),
                "vocab_size": getattr(model_config, "vocab_size", None),
                "max_position_embeddings": getattr(model_config, "n_positions", None),
            },
            "runtime_versions": runtime_versions(),
            "resources": resources,
            "cleanup": "pending external wrapper cleanup; runner makes no completion claim",
            "wrapper_cleanup_evidence": "reported by remote-cuda wrapper separately",
            "network": "model download/network access required only during remote run",
            "credentials": "none used by runner",
            "data_digests": {
                key: split["metadata"][key]
                for key in ("content_sha256", "label_sha256", "prompt_sha256", "prompt_digest_sha256", "fold_sha256")
            },
            "hidden_layers": sorted(hidden),
            "feature_digest": hashlib.sha256(
                "".join(array_digest(hidden[layer]) for layer in sorted(hidden)).encode("ascii")
            ).hexdigest(),
        }
        artifact = build_artifact(plan, records, split["metadata"], provenance)
        errors = validate_artifact(artifact, plan, source_digests())
        if errors:
            raise ValueError("invalid L03 artifact: " + "; ".join(errors))
        run_record = build_run_record(plan, artifact, resources)
        report = build_report(plan, artifact, run_record)
        errors = validate_report(report, plan)
        if errors:
            raise ValueError("invalid L03 report: " + "; ".join(errors))
        return report
    except Exception as error:
        return failure_envelope(plan, error, model_attempt=section(plan, "model"), resources=run_resources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run-real", action="store_true")
    args = parser.parse_args()
    if args.run_real:
        print(json.dumps(run_real(), ensure_ascii=True, sort_keys=True))
    else:
        print(check())


if __name__ == "__main__":
    main()
