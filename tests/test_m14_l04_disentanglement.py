"""Offline protocol and envelope tests for M14 L04.8."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts import _m14_l04_disentanglement as disentanglement_handler
from scripts._m14_l04_disentanglement_metrics import (
    FACTORS,
    bootstrap,
    brier_quality,
    deterministic_group_derangement,
    fixture_row_summary,
    group_factor_quality,
    mapping_digest,
    metric,
    shuffled_labels,
)
from scripts._m14_l04_disentanglement_runtime import binary_token_bow, excluded_columns_digest, fit_logistic_probe
from scripts._m14_l04_fixture_contract import FIXTURE_PATH, read_fixture
from scripts._m14_l04_tcav_runtime import read_rows
from scripts._m14_l04_validate import validate_artifact, validate_failure, validate_run_record
from scripts._m14_l04_validate_disentanglement import validate_real_disentanglement_execution
from scripts.m14_l04_contract import load_plan
from scripts.m14_l04_explanations import run_real


def _rows() -> list[dict[str, Any]]:
    return [
        {"group_id": "g01", "input_ids": np.array([1, 2, 0]), "attention_mask": np.array([1, 1, 0])},
        {"group_id": "g01", "input_ids": np.array([3, 2, 0]), "attention_mask": np.array([1, 1, 0])},
        {"group_id": "g02", "input_ids": np.array([1, 4, 0]), "attention_mask": np.array([1, 1, 0])},
        {"group_id": "g02", "input_ids": np.array([3, 4, 0]), "attention_mask": np.array([1, 1, 0])},
    ]


def test_brier_quality_and_group_unit_are_finite_for_imbalanced_labels() -> None:
    assert brier_quality([0.99, 0.01, 0.01], [1, 0, 0]) > 0.99
    values = group_factor_quality(
        _rows(),
        [0.9, 0.1, 0.2, 0.8],
        {"animal_cat": [1, 1, 0, 0], "tone_positive": [1, 0, 1, 0]},
    )
    assert set(values) == {"g01", "g02"}
    assert values["g01"]["animal_cat"] == pytest.approx(0.59)


def test_degenerate_bootstrap_is_finite_and_deterministic() -> None:
    assert bootstrap([0.2, 0.2, 0.2, 0.2], seed=17) == [0.2, 0.2]
    assert metric([0.2, 0.2, 0.2, 0.2], seed=17, point_threshold=0.1, ci_lower_threshold=0.05)[
        "confidence_interval_95"
    ] == [0.2, 0.2]


def test_shuffle_is_group_block_derangement_with_slot_reversal() -> None:
    rows = _rows()
    labels = {"animal_cat": [1, 1, 0, 0], "tone_positive": [1, 0, 1, 0]}
    mapping = deterministic_group_derangement(["g01", "g02"], 17)
    assert all(item["source_group"] != item["target_group"] for item in mapping)
    assert all(item["slot_reversal"] is True for item in mapping)
    assert mapping_digest(mapping) == mapping_digest(deterministic_group_derangement(["g01", "g02"], 17))
    shuffled = shuffled_labels(rows, labels, mapping)
    assert shuffled["animal_cat"] == [0, 0, 1, 1]
    assert shuffled["tone_positive"] == [0, 1, 0, 1]
    assert {sum(labels[factor]) for factor in FACTORS} == {sum(shuffled[factor]) for factor in FACTORS}


def test_probe_uses_train_only_standardization_and_is_repeatable() -> None:
    features = np.array([[0.0, 5.0], [0.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
    labels = np.array([0, 0, 1, 1])
    first = fit_logistic_probe(features, labels, torch=torch)
    second = fit_logistic_probe(features, labels, torch=torch)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.predict_proba(features), second.predict_proba(features))
    assert first.scale[1] == 1.0  # zero variance handling is explicit and finite


def test_actual_tcav_reader_preserves_authored_condition_and_rejects_missing_or_tampered() -> None:
    _raw, authored = read_fixture(FIXTURE_PATH)

    class Integration:
        def tokenize(self, prompts: tuple[str, ...], *, max_length: int, return_tensors: str) -> dict[str, Any]:
            assert len(prompts) in {1, len(authored)}
            assert max_length == 32
            assert return_tensors == "pt"
            return {
                "input_ids": torch.ones((len(prompts), 3), dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]] * len(prompts), dtype=torch.long),
            }

    real_rows = read_rows(Integration(), authored, 32)
    assert [row["condition"] for row in real_rows] == [row["condition"] for row in authored]

    missing = dict(authored[0])
    missing.pop("condition")
    with pytest.raises(ValueError, match="invalid or missing condition"):
        read_rows(Integration(), [missing], 32)
    tampered = dict(authored[0])
    tampered["condition"] = "not-a-condition"
    with pytest.raises(ValueError, match="invalid or missing condition"):
        read_rows(Integration(), [tampered], 32)


def test_linux_ru_maxrss_is_normalized_to_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    resource = pytest.importorskip("resource")
    monkeypatch.setattr(disentanglement_handler.sys, "platform", "linux")
    monkeypatch.setattr(resource, "getrusage", lambda _who: type("Usage", (), {"ru_maxrss": 1234})())
    assert disentanglement_handler.__dict__["_rss_measurement"]() == (
        1234 * 1024,
        "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "bytes",
    )


def test_handler_scoring_deduplicates_eight_train_group_ids() -> None:
    train_rows = [
        {
            "row_id": f"l04-g{group:02d}-{'clean' if slot == 0 else 'corrupted'}",
            "group_id": f"g{group:02d}",
            "causal_pair_id": f"p{group:02d}",
            "condition": "clean" if slot == 0 else "corrupted",
            "split": "train",
            "task": "task",
            "prompt": "prompt",
            "target_text": " true" if slot == 0 else " false",
            "factor_labels": {"animal_cat": 1 if group == 1 else 0, "tone_positive": 1 if slot == 0 else 0},
        }
        for group in range(1, 9)
        for slot in range(2)
    ]
    holdout_rows = [
        {
            "row_id": f"l04-g{group:02d}-{'clean' if slot == 0 else 'corrupted'}",
            "group_id": f"g{group:02d}",
            "causal_pair_id": f"p{group:02d}",
            "condition": "clean" if slot == 0 else "corrupted",
            "split": "holdout",
            "task": "task",
            "prompt": "prompt",
            "target_text": " true" if slot == 0 else " false",
            "factor_labels": {"animal_cat": 1 if group == 9 else 0, "tone_positive": 1 if slot == 0 else 0},
        }
        for group in range(9, 13)
        for slot in range(2)
    ]
    train_labels = {
        "animal_cat": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "tone_positive": [1, 0] * 8,
    }
    holdout_labels = {"animal_cat": [1, 1, 0, 0, 0, 0, 0, 0], "tone_positive": [1, 0] * 4}
    train_features = np.asarray([[float(index % 4), float(index // 4)] for index in range(16)])
    holdout_features = np.asarray([[float(index % 4), float(index // 4)] for index in range(8)])
    summary = disentanglement_handler.__dict__["_run_seed"](
        17,
        train_rows,
        holdout_rows,
        train_features,
        holdout_features,
        train_labels,
        holdout_labels,
        train_features,
        holdout_features,
        torch=torch,
    )
    assert [item["source_group"] for item in summary["shuffled_group_mapping"]] == [
        f"g{group:02d}" for group in range(1, 9)
    ]
    assert len(summary["shuffled_group_mapping"]) == 8
    assert len(summary["holdout_evidence"]) == 8


def test_actual_handler_path_reaches_scoring_with_duplicate_pair_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = load_plan()
    _raw, authored = read_fixture(FIXTURE_PATH)
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")

    class FakeTokenizer:
        vocab_size = 16

        def decode(self, values: list[int]) -> str:
            return " true" if values[0] == 2081 else " false"

    class FakeIntegration:
        def __init__(self, **_kwargs: object) -> None:
            self.tokenizer = FakeTokenizer()

        def tokenize(self, prompts: tuple[str, ...], *, max_length: int, return_tensors: str) -> dict[str, Any]:
            assert len(prompts) in {1, len(authored)}
            assert max_length == 32
            assert return_tensors == "pt"
            return {
                "input_ids": torch.ones((len(prompts), 3), dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]] * len(prompts), dtype=torch.long),
            }

        def _backend(self) -> tuple[object, object, object]:
            return torch.nn.Linear(4, 4), self.tokenizer, type("Config", (), {"vocab_size": 16})()

    def fake_capture(_model: object, source_rows: list[dict[str, Any]], _layer: int) -> np.ndarray:
        return np.asarray([[float(index % 4), float(index // 4), 0.0, 1.0] for index in range(len(source_rows))])

    monkeypatch.setattr(disentanglement_handler, "capture_activations", fake_capture)
    monkeypatch.setattr(disentanglement_handler, "parameter_digest", lambda _model: "digest")
    monkeypatch.setattr(disentanglement_handler, "_rss_bytes", lambda: 1)
    monkeypatch.setattr(disentanglement_handler, "tokenizer_vocab_size", lambda _tokenizer, _config: 16)
    monkeypatch.setattr(disentanglement_handler, "GPT2_VOCAB_SIZE", 16)
    monkeypatch.setattr(
        disentanglement_handler,
        "resolve_target_token",
        lambda _tokenizer, text: (2081, text) if text == " true" else (3991, text),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "fake-cuda")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = disentanglement_handler.run_disentanglement(plan, authored, integration_factory=FakeIntegration)
    assert len(result["raw_summaries"]) == 5
    assert all(len(summary["shuffled_group_mapping"]) == 8 for summary in result["raw_summaries"])
    expected_conditions = {row["row_id"]: row["condition"] for row in authored if row["split"] == "holdout"}
    assert all(
        item["condition"] == expected_conditions[item["row_id"]]
        for summary in result["raw_summaries"]
        for item in summary["holdout_evidence"]
    )


def test_raw_token_bow_excludes_padding_and_never_builds_vocab_from_holdout() -> None:
    values = binary_token_bow(_rows(), vocab_size=8)
    assert values.shape == (4, 8)
    assert values[0, 0] == 0.0
    assert values[0, 2] == 1.0
    assert values[0, 7] == 0.0  # padding slot is not a feature
    with pytest.raises(ValueError, match="outside"):
        binary_token_bow([{"input_ids": np.array([8]), "attention_mask": np.array([1])}], 8)
    no_target = binary_token_bow(_rows(), vocab_size=8, excluded_token_ids={2})
    assert no_target[0, 2] == 0.0


def _successful_result(plan: dict[str, Any]) -> dict[str, Any]:
    _raw, authored = read_fixture(FIXTURE_PATH)
    holdout = [row for row in authored if row["split"] == "holdout"]
    summaries = []
    train_groups = list(plan["fixture"]["split"]["train_groups"])
    holdout_groups = list(plan["fixture"]["split"]["holdout_groups"])
    for seed in (17, 29, 41, 53, 67):
        mapping = deterministic_group_derangement(train_groups, seed)
        train_fixture = [row for row in authored if row["split"] == "train"]
        train_labels = {factor: [int(row["factor_labels"][factor]) for row in train_fixture] for factor in FACTORS}
        evidence = []
        methods = ("real", "shuffled", "factor_permutation", "raw_token")
        for row in holdout:
            labels = {factor: int(row["factor_labels"][factor]) for factor in FACTORS}
            probabilities = {
                "real": {factor: 0.9 if labels[factor] else 0.1 for factor in FACTORS},
                "shuffled": {factor: 0.5 for factor in FACTORS},
                "factor_permutation": {factor: 0.8 if labels[factor] else 0.2 for factor in FACTORS},
                "raw_token": {factor: 0.6 for factor in FACTORS},
            }
            evidence.append(
                {
                    "row_id": row["row_id"],
                    "group_id": row["group_id"],
                    "causal_pair_id": row["causal_pair_id"],
                    "condition": row["condition"],
                    "true_labels": labels,
                    "predicted_probabilities": probabilities,
                    "fixture_row_linkage": fixture_row_summary(row),
                }
            )

        def qualities(method: str, evidence_rows: list[dict[str, Any]] = evidence) -> dict[str, dict[str, float]]:
            return {
                group: {
                    factor: float(
                        np.mean(
                            [
                                brier_quality(
                                    [item["predicted_probabilities"][method][factor]],
                                    [item["true_labels"][factor]],
                                )
                                for item in evidence_rows
                                if item["group_id"] == group
                            ]
                        )
                    )
                    for factor in FACTORS
                }
                for group in holdout_groups
            }

        by_method = {method: qualities(method) for method in methods}
        macros = {
            method: {group: float(np.mean(list(values.values()))) for group, values in grouped.items()}
            for method, grouped in by_method.items()
        }
        gains = {group: macros["real"][group] - macros["shuffled"][group] for group in holdout_groups}
        heldout = metric(list(gains.values()), seed=seed, point_threshold=0.1, ci_lower_threshold=0.05)
        counts = {factor: [8, 8] for factor in FACTORS}
        counts["animal_cat"] = [14, 2]
        fit_metadata = {
            name: {
                factor: {
                    "class_counts": counts[factor]
                    if name != "factor_permutation"
                    else counts[FACTORS[1] if factor == FACTORS[0] else FACTORS[0]],
                    "feature_dim": 50257 if name == "raw_token" else 4,
                    "standardization_sha256": "0" * 64,
                    "probe_sha256": "1" * 64,
                }
                for factor in FACTORS
            }
            for name in ("real", "shuffled", "factor_permutation", "raw_token")
        }
        summaries.append(
            {
                "seed": seed,
                "train_groups": train_groups,
                "holdout_groups": holdout_groups,
                "real_group_factor_quality": by_method["real"],
                "shuffled_group_factor_quality": {group: values for group, values in by_method["shuffled"].items()},
                "factor_permutation_group_factor_quality": {
                    group: values for group, values in by_method["factor_permutation"].items()
                },
                "raw_token_group_factor_quality": {group: values for group, values in by_method["raw_token"].items()},
                "real_group_quality": macros["real"],
                "shuffled_group_quality": macros["shuffled"],
                "factor_permutation_group_quality": macros["factor_permutation"],
                "raw_token_group_quality": macros["raw_token"],
                "gain_by_group": gains,
                "heldout_gain": heldout,
                "shuffled_group_mapping": mapping,
                "shuffled_mapping_sha256": mapping_digest(mapping),
                "factor_permutation": {"swapped_factors": list(FACTORS)},
                "factor_permutation_supervision": disentanglement_handler.factor_permutation_supervision(
                    train_fixture, train_labels
                ),
                "seeded_repeat_exact": True,
                "finite": True,
                "factor_count_preserved": True,
                "bootstrap_replicates": 2000,
                "holdout_evidence": evidence,
                "seeded_repeat_probabilities": {
                    factor: [item["predicted_probabilities"]["real"][factor] for item in evidence] for factor in FACTORS
                },
                "fit_metadata": fit_metadata,
                "model_parameter_digest_before": "2" * 64,
                "model_parameter_digest_after": "2" * 64,
            }
        )
    raw_token_linkage = {
        "row_order": [str(row["row_id"]) for row in authored],
        "tokenizer": "pinned-gpt2",
        "vocab_size": 50257,
        "padding": "attention_mask; excluded padding tokens",
        "tokenization_digest": disentanglement_handler.tokenization_digest(authored, vocab_size=50257),
        "feature_matrix": {
            "digest": "a" * 64,
            "shape": [len(authored), 50257],
            "dtype": "float64",
            "order": "C",
            "config": "binary token presence; attention_mask; no padding; target IDs excluded",
        },
        "excluded_columns": {
            "token_ids": [2081, 3991],
            "digest": excluded_columns_digest(len(authored), [2081, 3991]),
            "shape": [len(authored), 2],
            "dtype": "float64",
            "order": "C",
            "all_zero": True,
        },
    }
    return {
        "status": "passed_real_cuda",
        "evidence_eligible": True,
        "acceptance": True,
        "evidence_level": "D2",
        "metrics": {"heldout_gain_over_shuffled": {str(item["seed"]): item["heldout_gain"] for item in summaries}},
        "confidence_intervals": {
            str(item["seed"]): item["heldout_gain"]["confidence_interval_95"] for item in summaries
        },
        "controls": {
            **{name: {"pass": True} for name in ("group_preserving_shuffle", "factor_permutation", "seeded_repeat")},
            "raw_token_baseline": {"pass": True, "vocab_size": 50257},
        },
        "raw_summaries": summaries,
        "raw_token_linkage": raw_token_linkage,
        "fixture_linkage": [fixture_row_summary(row) for row in authored],
        "token_ids": {" true": 2081, " false": 3991},
        "target_token_strings": {" true": " true", " false": " false"},
        "layer": 6,
        "native_hidden_state_index": 7,
        "seeds": [17, 29, 41, 53, 67],
        "no_mutation": True,
        "model_parameter_digest_before": "2" * 64,
        "model_parameter_digest_after": "2" * 64,
        "budget_pass": True,
        "provenance": {
            "network": "enabled",
            "device": "fake-cuda",
            "execution_attempted": True,
            "execution_backend": "cuda",
            "target_token_ids": {" true": 2081, " false": 3991},
            "target_token_strings": {" true": " true", " false": " false"},
            "deterministic_algorithms": True,
            "resource_peak": {
                "cuda_device": "fake-cuda",
                "elapsed_seconds": 1.0,
                "max_memory_allocated_bytes": 1,
                "max_memory_reserved_bytes": 1,
                "max_rss_bytes": 1,
                "rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "rss_unit": "bytes",
            },
            "budget_pass": True,
            "stage": "complete",
            "model_parameter_digest_before": "2" * 64,
            "model_parameter_digest_after": "2" * 64,
            "model_parameter_digest_algorithm": "sha256/canonical-ordered-named-parameters-v1",
            "raw_token_excluded_ids": [2081, 3991],
            "probe": {
                "dtype": "float64 CPU",
                "optimizer": "torch.optim.LBFGS strong_wolfe",
                "max_iter": 100,
                "tolerance_grad": 1e-9,
                "tolerance_change": 1e-12,
                "convergence_grad_tol": 1e-6,
                "l2_c": 1.0,
                "class_weight": "balanced",
                "standardization": "train-only; zero variance scale=1",
            },
        },
        "resources": {
            "device": "fake-cuda",
            "network": "enabled",
            "execution_attempted": True,
            "execution_backend": "cuda",
            "stage": "complete",
            "resource_peak": {
                "cuda_device": "fake-cuda",
                "elapsed_seconds": 1.0,
                "max_memory_allocated_bytes": 1,
                "max_memory_reserved_bytes": 1,
                "max_rss_bytes": 1,
                "rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "rss_unit": "bytes",
            },
            "cleanup": "fake cleanup",
        },
    }


def test_real_like_runner_artifact_validator_and_tamper_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = load_plan()
    from scripts import _m14_l04_disentanglement as handler_module

    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(handler_module, "run_disentanglement", lambda _plan, _rows: _successful_result(plan))
    result = run_real(use_case="Disentanglement", output_dir=tmp_path)
    assert result["status"] == "passed_real_cuda"
    assert validate_artifact(result["artifact"], plan) == []
    assert (
        validate_real_disentanglement_execution(
            next(item for item in result["artifact"]["executions"] if item["use_case"] == "Disentanglement"),
            result["artifact"],
            plan,
        )
        == []
    )
    tampered = json.loads(json.dumps(result["artifact"]))
    active = next(item for item in tampered["executions"] if item["use_case"] == "Disentanglement")
    active["controls"]["seeded_repeat"]["pass"] = False
    assert validate_artifact(tampered, plan)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("budget_pass", False),
        ("no_mutation", False),
        ("model_parameter_digest_after", "3" * 64),
    ),
)
def test_resource_and_mutation_flags_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: bool
) -> None:
    plan = load_plan()
    from scripts import _m14_l04_disentanglement as handler_module

    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(handler_module, "run_disentanglement", lambda _plan, _rows: _successful_result(plan))
    result = run_real(use_case="Disentanglement", output_dir=tmp_path)
    active = next(item for item in result["artifact"]["executions"] if item["use_case"] == "Disentanglement")
    active[field] = value
    assert validate_real_disentanglement_execution(active, result["artifact"], plan)


@pytest.mark.parametrize(
    "resource_field,value",
    (
        ("elapsed_seconds", 1801.0),
        ("max_memory_allocated_bytes", 6 * 1024**3 + 1),
        ("max_memory_reserved_bytes", 6 * 1024**3 + 1),
        ("max_rss_bytes", float("nan")),
        ("rss_source", "psutil.Process.memory_info().rss"),
        ("rss_unit", "KiB"),
    ),
)
def test_adversarial_resource_evidence_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resource_field: str, value: float
) -> None:
    plan = load_plan()
    from scripts import _m14_l04_disentanglement as handler_module

    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(handler_module, "run_disentanglement", lambda _plan, _rows: _successful_result(plan))
    result = run_real(use_case="Disentanglement", output_dir=tmp_path)
    active = next(item for item in result["artifact"]["executions"] if item["use_case"] == "Disentanglement")
    active["provenance"]["resource_peak"][resource_field] = value
    assert validate_real_disentanglement_execution(active, result["artifact"], plan)


def test_missing_resource_evidence_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = load_plan()
    from scripts import _m14_l04_disentanglement as handler_module

    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(handler_module, "run_disentanglement", lambda _plan, _rows: _successful_result(plan))
    result = run_real(use_case="Disentanglement", output_dir=tmp_path)
    active = next(item for item in result["artifact"]["executions"] if item["use_case"] == "Disentanglement")
    del active["provenance"]["resource_peak"]
    assert validate_real_disentanglement_execution(active, result["artifact"], plan)


def test_factor_and_raw_token_linkage_tampering_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = load_plan()
    from scripts import _m14_l04_disentanglement as handler_module

    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(handler_module, "run_disentanglement", lambda _plan, _rows: _successful_result(plan))
    result = run_real(use_case="Disentanglement", output_dir=tmp_path)
    active = next(item for item in result["artifact"]["executions"] if item["use_case"] == "Disentanglement")
    active["raw_token_linkage"]["excluded_columns"]["digest"] = "0" * 64
    assert validate_real_disentanglement_execution(active, result["artifact"], plan)
    active["raw_token_linkage"]["excluded_columns"]["digest"] = excluded_columns_digest(24, [2081, 3991])
    active["raw_summaries"][0]["factor_permutation_supervision"]["rows"][0]["swapped_labels"]["animal_cat"] = 9
    assert validate_real_disentanglement_execution(active, result["artifact"], plan)


def test_real_preflight_cuda_failure_is_truthful_nonpromoting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = run_real(use_case="Disentanglement", output_dir=tmp_path)
    plan = load_plan()
    assert result["status"] == "failed"
    assert result["artifact"]["evidence_level"] == "D0"
    assert result["artifact"]["provenance"]["execution_attempted"] is False
    assert result["artifact"]["provenance"]["execution_backend"] == "none"
    assert result["artifact"]["provenance"]["stage"] == "preflight"
    assert validate_artifact(result["artifact"], plan) == []
    assert validate_run_record(result["run_record"], result["artifact"], plan) == []
    assert validate_failure(result["failure"], plan, result["artifact"]) == []
    assert (
        validate_real_disentanglement_execution(
            next(item for item in result["artifact"]["executions"] if item["use_case"] == "Disentanglement"),
            result["artifact"],
            plan,
        )
        == []
    )


def test_partial_cuda_failure_preserves_device_in_artifact_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import _m14_l04_boundary as boundary_module

    def failing_factory() -> object:
        raise RuntimeError("model factory failure")

    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "fake-cuda")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(boundary_module, "transformer_integration_type", failing_factory)
    result = run_real(use_case="Disentanglement", output_dir=tmp_path)
    plan = load_plan()
    assert result["status"] == "failed"
    assert result["artifact"]["provenance"]["device"] == "fake-cuda"
    assert result["artifact"]["provenance"]["execution_attempted"] is True
    assert result["artifact"]["provenance"]["execution_backend"] == "cuda"
    assert result["artifact"]["provenance"]["stage"] == "cleanup"
    assert validate_artifact(result["artifact"], plan) == []
    assert validate_run_record(result["run_record"], result["artifact"], plan) == []
    assert validate_failure(result["failure"], plan, result["artifact"]) == []


@pytest.mark.parametrize("device", (None, "", "not used", "not attempted", "cpu", "cuda"))
def test_failure_validator_rejects_placeholder_attempted_cuda_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: object
) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = run_real(use_case="Disentanglement", output_dir=tmp_path)
    resource = result["failure"]["resource"]
    resource.update(
        {
            "device": device,
            "network": "enabled",
            "execution_attempted": True,
            "execution_backend": "cuda",
            "stage": "cuda_check",
        }
    )
    assert validate_failure(result["failure"], load_plan(), result["artifact"])


@pytest.mark.parametrize("device", (None, "", "not used", "not attempted", "cpu", "cuda"))
def test_artifact_validator_rejects_placeholder_attempted_cuda_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: object
) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = run_real(use_case="Disentanglement", output_dir=tmp_path)
    provenance = result["artifact"]["provenance"]
    provenance.update(
        {
            "device": device,
            "network": "enabled",
            "execution_attempted": True,
            "execution_backend": "cuda",
            "stage": "cuda_check",
        }
    )
    assert validate_artifact(result["artifact"], load_plan())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("elapsed_seconds", 1800.001),
        ("max_memory_allocated_bytes", 6 * 1024**3 + 1),
        ("max_memory_reserved_bytes", 6 * 1024**3 + 1),
        ("max_rss_bytes", 4 * 1024**3 + 1),
    ),
)
def test_budget_pass_rejects_each_frozen_overrun(field: str, value: float) -> None:
    resource_peak = {
        "elapsed_seconds": 1.0,
        "max_memory_allocated_bytes": 1,
        "max_memory_reserved_bytes": 1,
        "max_rss_bytes": 1,
    }
    resource_peak[field] = value
    assert disentanglement_handler.budget_pass(resource_peak) is False


@pytest.mark.parametrize("tamper", ("missing_probabilities", "wrong_probabilities_type", "wrong_metrics_type"))
def test_validator_malformed_nested_payload_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str
) -> None:
    plan = load_plan()
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(disentanglement_handler, "run_disentanglement", lambda _plan, _rows: _successful_result(plan))
    result = run_real(use_case="Disentanglement", output_dir=tmp_path)
    active = next(item for item in result["artifact"]["executions"] if item["use_case"] == "Disentanglement")
    if tamper == "missing_probabilities":
        del active["raw_summaries"][0]["holdout_evidence"][0]["predicted_probabilities"]
    elif tamper == "wrong_probabilities_type":
        active["raw_summaries"][0]["holdout_evidence"][0]["predicted_probabilities"] = []
    else:
        active["metrics"] = []
    errors = validate_real_disentanglement_execution(active, result["artifact"], plan)
    assert errors
    assert all(isinstance(error, str) and error for error in errors)
