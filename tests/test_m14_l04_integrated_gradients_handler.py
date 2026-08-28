"""Offline contract tests for the L04.4 Integrated Gradients handler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

from scripts._m14_l04_artifact import build_artifact
from scripts._m14_l04_data import fixture_metadata
from scripts._m14_l04_envelope import (
    build_run_record,
    failure_envelope,
    validate_artifact,
    validate_failure,
    validate_run_record,
)
from scripts._m14_l04_fixture_contract import read_fixture
from scripts._m14_l04_ig_metrics import bootstrap, group_means, metric
from scripts._m14_l04_ig_runtime import Row, parameter_digest, target_attribution
from scripts._m14_l04_integrated_gradients import RealExecutionError, run_integrated_gradients
from scripts.m14_l04_contract import FIXTURE_PATH, load_plan


class _LinearTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 3)
        self.transformer: Any = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Identity() for _ in range(7)])
        self.lm_head = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(48, dtype=torch.float32).reshape(16, 3) / 20.0)
            self.lm_head.weight.zero_()
            self.lm_head.weight[1].copy_(torch.tensor([1.0, -2.0, 0.5]))
            self.lm_head.weight[2].copy_(torch.tensor([-0.5, 0.25, 1.0]))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, output_hidden_states: bool = False
    ) -> object:
        del attention_mask, output_hidden_states
        hidden = self.embedding(input_ids)
        for block in self.transformer.h:
            hidden = block(hidden)
        return type("Output", (), {"logits": self.lm_head(hidden)})()


def _complete_real_result() -> dict[str, Any]:
    main_metrics = {
        "completeness_relative_error": metric([0.0001], seed=17, threshold=0.001, comparator="<="),
        "step_16_vs_64_attribution_cosine": metric([0.99], seed=18, threshold=0.95, comparator=">", statistic="median"),
    }
    fixture_rows = read_fixture(FIXTURE_PATH)[1]
    raw_summaries = []
    for seed in (17, 29, 41, 53, 67):
        zero_baseline = [
            {
                "row_id": row["row_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "completeness_relative_error_16": 0.0001,
                "completeness_relative_error_64": 0.0001,
                "step_16_vs_64_attribution_cosine": 0.99,
                "randomized_target_attribution_cosine": 0.1,
                "seeded_repeat_cosine": 1.0,
                "finite": True,
                "no_mutation": True,
                "target_token_id": 1,
                "other_token_id": 2,
                "target_position": 3,
            }
            for row in fixture_rows
        ]
        batch_baseline = [
            {
                "row_id": row["row_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "completeness_relative_error": 0.0001,
            }
            for row in fixture_rows
        ]
        raw_summaries.append({"seed": seed, "zero_baseline": zero_baseline, "batch_mean_baseline": batch_baseline})
    controls = {
        "zero_baseline": {"metrics": main_metrics, "pass": True},
        "batch_mean_baseline": {
            "metrics": {"completeness_relative_error": metric([0.0001], seed=19, threshold=0.001, comparator="<=")},
            "pass": True,
        },
        "random_target": {
            "metrics": {"attribution_cosine": metric([0.1], seed=20, threshold=0.25, comparator="<=")},
            "pass": True,
        },
        "seeded_repeat": {
            "metrics": {"attribution_cosine": metric([1.0], seed=21, threshold=1.0 - 1e-8, comparator=">")},
            "repeat_count": 2,
            "seeds": [17, 29, 41, 53, 67],
            "pass": True,
        },
        "finite/no-mutation": {
            "metrics": {"finite_fraction": metric([1.0], seed=22, threshold=1.0, comparator=">=")},
            "finite_rows": 120,
            "mutated": False,
            "pass": True,
        },
    }
    return {
        "status": "passed_real_cuda",
        "evidence_eligible": True,
        "acceptance": True,
        "metrics": main_metrics,
        "controls": controls,
        "raw_summaries": raw_summaries,
        "token_ids": {" true": 1, " false": 2},
        "seeds": [17, 29, 41, 53, 67],
        "layer": 6,
        "native_hidden_state_index": 7,
        "provenance": {
            "device": "NVIDIA test GPU",
            "network": "enabled",
            "deterministic_algorithms": True,
            "runtime_versions": {"torch": "test"},
            "resource_peak": {"max_memory_allocated_bytes": 1},
        },
        "resources": {"device": "NVIDIA test GPU", "network": "enabled", "cleanup": "done"},
    }


def _complete_artifact() -> dict[str, Any]:
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = _complete_real_result()
    return build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "IntegratedGradients",
        "passed_real_cuda",
        "failure.json",
        execution_result=result,
        resources=result["resources"],
    )


def test_real_handler_requires_explicit_cuda_network_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cpu")

    with pytest.raises(RealExecutionError, match="NETWORK_DEVICE=cuda"):
        run_integrated_gradients(load_plan(), [])


def test_production_ig_selection_is_not_marked_as_injected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from scripts.m14_l04_explanations import run_real

    result = run_real(use_case="IntegratedGradients", output_dir=tmp_path)

    assert result["status"] == "failed"
    assert result["artifact"]["provenance"]["evidence_origin"] == "real-cuda"
    assert result["artifact"]["provenance"]["evidence_origin"] != "dependency-injected-offline"


def test_median_metric_and_control_threshold_have_frozen_semantics() -> None:
    robust = metric([1.0, 1.0, 10.0], seed=17, threshold=2.0, comparator=">", statistic="median")
    strict_max = metric([0.2, 0.3], seed=17, threshold=0.1, comparator="<=")
    inclusive_min = metric([1.0], seed=17, threshold=1.0, comparator=">=")

    assert robust["point_estimate"] == 1.0
    assert robust["pass"] is False
    assert strict_max["pass"] is False
    assert inclusive_min["pass"] is True
    assert len(robust["confidence_interval_95"]) == 2


def test_fake_transformer_ig_is_complete_and_batch_baseline_is_reproducible() -> None:
    model = _LinearTransformer()
    ids = np.asarray([[1, 2, 3], [4, 5, 0]], dtype=np.int64)
    masks = np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.int64)
    row = Row("r1", "g1", "train", ids[0], masks[0])
    before = parameter_digest(model)

    attr, delta, error = target_attribution(
        model,
        row,
        target_token=1,
        other_token=2,
        steps=16,
        baseline="zero",
        seed=17,
        source_model_version="fake-v1",
        batch_ids=ids,
        batch_mask=masks,
        batch_index=0,
    )
    repeat, repeat_delta, repeat_error = target_attribution(
        model,
        row,
        target_token=1,
        other_token=2,
        steps=16,
        baseline="batch_mean",
        seed=17,
        source_model_version="fake-v1",
        batch_ids=ids,
        batch_mask=masks,
        batch_index=0,
    )

    assert np.isfinite(attr).all() and np.isfinite(repeat).all()
    assert attr.sum() == pytest.approx(delta, abs=1e-6)
    assert error == pytest.approx(0.0, abs=1e-6)
    assert repeat.sum() == pytest.approx(repeat_delta, abs=1e-6)
    assert repeat_error == pytest.approx(0.0, abs=1e-6)
    assert parameter_digest(model) == before


def test_handler_emits_validator_schema_for_seeded_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA fake CUDA")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    import scripts._m14_l04_integrated_gradients as ig_module

    calls: list[tuple[int, int, str]] = []

    def fake_target_attribution(*_args: object, **kwargs: object) -> tuple[np.ndarray, float, float]:
        target_token = int(cast(Any, kwargs["target_token"]))
        batch_ids = cast(Any, kwargs["batch_ids"])
        calls.append((int(cast(Any, kwargs["steps"])), len(batch_ids), str(kwargs["baseline"])))
        vector = np.asarray([1.0, 0.0]) if target_token in {1, 2} else np.asarray([0.0, 1.0])
        return vector, 1.0, 0.0

    monkeypatch.setattr(ig_module, "_target_attribution", fake_target_attribution)
    model: Any = _LinearTransformer()
    model.config = type("Config", (), {"vocab_size": 16})()

    class FakeIntegration:
        provenance = "fake-integration"

        def __init__(self, **_kwargs: object) -> None:
            pass

        def _backend(self) -> tuple[Any, Any, Any]:
            return model, self, model.config

        def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
            return {"input_ids": [1 if text == " true" else 2]}

        def tokenize(self, prompts: tuple[str, ...], **_kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.ones((len(prompts), 4), dtype=torch.long),
                "attention_mask": torch.ones((len(prompts), 4), dtype=torch.long),
            }

    plan = load_plan()
    rows = read_fixture(FIXTURE_PATH)[1]
    result = run_integrated_gradients(plan, rows, integration_factory=FakeIntegration)

    assert result["status"] == "passed_real_cuda"
    seeded = result["controls"]["seeded_repeat"]
    assert set(seeded) == {"metrics", "repeat_count", "seeds", "pass"}
    assert set(seeded["metrics"]) == {"attribution_cosine"}
    assert seeded["repeat_count"] == 2
    assert seeded["seeds"] == [17, 29, 41, 53, 67]
    assert len(calls) == 312
    assert sum(batch_size == 24 for _steps, batch_size, _baseline in calls) == 24
    assert sum(steps == 16 for steps, _batch_size, _baseline in calls) == 48
    assert sum(steps == 64 for steps, _batch_size, _baseline in calls) == 264

    raw, fixture_rows = read_fixture(FIXTURE_PATH)
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, fixture_rows),
        "IntegratedGradients",
        result["status"],
        "l04-explanations.IntegratedGradients.attempt1.failure.json",
        execution_result=result,
        resources=result["resources"],
    )
    run_record = build_run_record(
        plan,
        artifact,
        "IntegratedGradients",
        result["status"],
        result["resources"],
        artifact_name="l04-explanations.IntegratedGradients.attempt1.partial.json",
    )
    failure = failure_envelope(
        plan,
        "IntegratedGradients",
        result["status"],
        failure_ref="l04-explanations.IntegratedGradients.attempt1.failure.json",
        run_record=run_record,
    )
    assert validate_artifact(artifact, plan) == []
    assert validate_run_record(run_record, artifact, plan) == []
    assert validate_failure(failure, plan, artifact) == []


def test_group_bootstrap_uses_one_estimate_per_group() -> None:
    rows = [
        {"group_id": "g1", "value": 1.0},
        {"group_id": "g1", "value": 3.0},
        {"group_id": "g2", "value": 5.0},
    ]

    assert group_means(rows, "value") == [2.0, 5.0]
    assert bootstrap(group_means(rows, "value"), seed=17) == bootstrap(group_means(rows, "value"), seed=17)


def test_real_validator_rejects_incomplete_support_only_metrics() -> None:
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = {
        "status": "passed_real_cuda",
        "evidence_eligible": True,
        "acceptance": True,
        "metrics": {},
        "controls": {},
        "raw_summaries": [{"seed": 17, "groups": []}],
        "token_ids": {" true": 1, " false": 2},
        "seeds": [17, 29, 41, 53, 67],
        "provenance": {"device": "CUDA", "network": "enabled"},
    }
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "IntegratedGradients",
        "passed_real_cuda",
        "failure.json",
        execution_result=result,
        resources={"device": "CUDA", "network": "enabled", "cleanup": "done"},
    )

    errors = validate_artifact(artifact, plan)

    assert any("metrics have the wrong schema" in error for error in errors)
    assert any("controls have the wrong schema" in error for error in errors)
    assert artifact["accepted_record_ids"] == []
    assert artifact["accepted_gap_ids"] == []
    assert "prompt" not in artifact.get("raw_summaries", [{}])[0]


def test_real_validator_links_acceptance_to_all_metric_and_control_verdicts() -> None:
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = _complete_real_result()
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "IntegratedGradients",
        "passed_real_cuda",
        "failure.json",
        execution_result=result,
        resources=result["resources"],
    )

    assert validate_artifact(artifact, plan) == []
    artifact["executions"][0]["metrics"]["completeness_relative_error"]["pass"] = False
    artifact["artifact_sha256"] = "tampered"
    assert validate_artifact(artifact, plan)


def test_metric_and_bootstrap_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="unsupported statistic"):
        bootstrap([1.0], seed=17, statistic="trimmed")
    with pytest.raises(ValueError, match="finite"):
        bootstrap([float("nan")], seed=17)
    with pytest.raises(ValueError, match="at least"):
        metric([], seed=17, threshold=0.1, comparator="<=")
    with pytest.raises(ValueError, match="finite number"):
        metric([1.0], seed=17, threshold=float("inf"), comparator="<=")
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap([1.0], seed=17, replicates=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap([1.0], seed=17, replicates="2")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda artifact: artifact["executions"][0]["metrics"]["completeness_relative_error"].pop("units"), "units"),
        (
            lambda artifact: artifact["executions"][0]["metrics"]["completeness_relative_error"].update(
                {"point_estimate": float("nan")}
            ),
            "point estimate",
        ),
        (
            lambda artifact: artifact["executions"][0]["metrics"]["completeness_relative_error"].update(
                {"confidence_interval_95": [1.0, 0.0]}
            ),
            "CI",
        ),
        (
            lambda artifact: artifact["executions"][0]["metrics"]["completeness_relative_error"].update(
                {"aggregation_unit": "row"}
            ),
            "aggregation",
        ),
        (
            lambda artifact: artifact["executions"][0]["metrics"]["completeness_relative_error"].update(
                {"comparator": "="}
            ),
            "comparator",
        ),
        (
            lambda artifact: artifact["executions"][0]["metrics"]["completeness_relative_error"].update(
                {"threshold": float("inf")}
            ),
            "threshold",
        ),
        (
            lambda artifact: artifact["executions"][0]["controls"]["batch_mean_baseline"].update({"pass": False}),
            "control batch_mean_baseline pass",
        ),
        (
            lambda artifact: artifact["executions"][0]["controls"]["random_target"]["metrics"].update(
                {"wrong": {"pass": True}}
            ),
            "wrong schema",
        ),
        (
            lambda artifact: artifact["executions"][0]["controls"]["seeded_repeat"].update({"seeds": [17]}),
            "seed/count",
        ),
        (
            lambda artifact: artifact["executions"][0]["controls"]["finite/no-mutation"].update({"finite_rows": 999}),
            "rows",
        ),
    ],
)
def test_real_schema_and_control_mutations_fail_closed(mutate: Any, needle: str) -> None:
    artifact = _complete_artifact()
    mutate(artifact)

    assert any(needle.lower() in error.lower() for error in validate_artifact(artifact, load_plan()))


def test_raw_summary_missing_row_fails_closed() -> None:
    artifact = _complete_artifact()
    del artifact["raw_summaries"][0]["zero_baseline"][-1]

    errors = validate_artifact(artifact, load_plan())

    assert any("row count is invalid" in error for error in errors)


def test_raw_summary_duplicate_row_fails_closed() -> None:
    artifact = _complete_artifact()
    rows = artifact["raw_summaries"][0]["batch_mean_baseline"]
    rows[-1]["row_id"] = rows[0]["row_id"]

    errors = validate_artifact(artifact, load_plan())

    assert any("row ids are not unique" in error for error in errors)
    assert any("group coverage is invalid" in error for error in errors)


def test_production_dispatch_success_keeps_real_result_and_validates_triads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    import scripts._m14_l04_integrated_gradients as ig_module
    from scripts.m14_l04_explanations import main, run_real

    monkeypatch.setattr(ig_module, "run_integrated_gradients", lambda _plan, _rows: _complete_real_result())
    result = run_real(use_case="IntegratedGradients", output_dir=tmp_path)

    assert result["status"] == "passed_real_cuda"
    assert result["artifact"]["provenance"]["evidence_origin"] == "real-cuda"
    assert "execution_result_digest" in result["artifact"]["provenance"]
    assert "injected_handler_result_digest" not in result["artifact"]["provenance"]
    assert validate_artifact(result["artifact"], load_plan()) == []
    assert validate_run_record(result["run_record"], result["artifact"], load_plan()) == []
    assert validate_failure(result["failure"], load_plan(), result["artifact"]) == []
    assert result["artifact"]["accepted_record_ids"] == []
    assert result["artifact"]["accepted_gap_ids"] == []

    main(["--run-real", "--use-case", "IntegratedGradients", "--output-dir", str(tmp_path)])
