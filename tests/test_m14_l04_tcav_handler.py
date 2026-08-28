"""Focused offline tests for the M14 L04 TCAV lane."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from scripts._m14_l04_artifact import build_artifact
from scripts._m14_l04_data import fixture_metadata
from scripts._m14_l04_envelope import build_run_record, failure_envelope
from scripts._m14_l04_fixture_contract import read_fixture
from scripts._m14_l04_tcav import run_tcav
from scripts._m14_l04_tcav_controls import assemble_controls
from scripts._m14_l04_tcav_metrics import corrected_empirical_p, group_means, wilson_lower
from scripts._m14_l04_validate import validate_artifact, validate_failure, validate_run_record
from scripts.m14_l04_contract import FIXTURE_PATH, load_plan


def test_tcav_corrected_empirical_p_uses_plus_one_correction() -> None:
    assert corrected_empirical_p(0.8, [0.8] + [0.2] * 98) == 2.0 / 100.0


def test_tcav_group_means_keep_pairs_from_being_independent_rows() -> None:
    rows = [{"group_id": "g1", "score": 0.0}, {"group_id": "g1", "score": 1.0}, {"group_id": "g2", "score": 1.0}]
    assert group_means(rows, "score") == [0.5, 1.0]


def test_tcav_controls_recompute_all_five_control_verdicts() -> None:
    controls = assemble_controls(
        shuffled_scores=[0.0, 0.0, 0.0, 0.0],
        random_scores=[0.0, 0.0, 0.0, 0.0],
        matched_scores=[0.0, 0.0, 0.0, 0.0],
        off_target_scores=[0.0, 0.0, 0.0, 0.0],
        zero_differences=[0.0, 0.0, 0.0, 0.0],
        seed=17,
        sensitivity_reference=0.6,
    )
    assert set(controls) == {
        "shuffled_concept_labels",
        "random_concept_directions",
        "matched_norm_null",
        "off_target_target_token",
        "zero_strength_identity",
    }
    assert all(control["pass"] is True for control in controls.values())


def test_tcav_wilson_lower_is_finite_for_group_accuracy() -> None:
    assert 0.0 < wilson_lower(3, 4) < 1.0


class _ContextBlock(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + values.mean(dim=1, keepdim=True)


class _CountingTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 4)
        self.transformer: nn.Module = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Identity() for _ in range(6)] + [_ContextBlock()])  # type: ignore[attr-defined]
        self.lm_head = nn.Linear(4, 128, bias=False)
        self.calls = 0
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            self.lm_head.weight[2] = torch.tensor([-1.0, 0.0, 0.0, 0.0])

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, output_hidden_states: bool = False
    ) -> object:
        del attention_mask, output_hidden_states
        self.calls += 1
        hidden = self.embedding(input_ids)
        for block in self.transformer.h:  # type: ignore[attr-defined]
            hidden = block(hidden)
        return type("Output", (), {"logits": self.lm_head(hidden)})()


class _FakeIntegration:
    provenance = "fake-gpt2"
    last_instance: _FakeIntegration | None = None

    def __init__(self, **_kwargs: object) -> None:
        self.model = _CountingTransformer()
        self.config = type("Config", (), {"vocab_size": 128})()
        _FakeIntegration.last_instance = self

    def _backend(self) -> tuple[object, object, object]:
        return self.model, self, self.config

    def tokenize(self, prompts: tuple[str, ...], **_kwargs: object) -> dict[str, torch.Tensor]:
        ids = [[index + 3, index + 4, 7, 8] for index, _prompt in enumerate(prompts)]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones((len(ids), 4), dtype=torch.long)}

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": [1] if text == " true" else [2]}

    def decode(self, ids: list[int]) -> str:
        return " true" if ids[0] == 1 else " false"


def test_tcav_handler_uses_real_hook_path_with_bounded_fake_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "fake CUDA")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    plan = load_plan()
    rows = read_fixture(FIXTURE_PATH)[1]
    result = run_tcav(plan, rows, integration_factory=_FakeIntegration)
    assert result["status"] == "failed"
    assert result["seeds"] == [17, 29, 41, 53, 67]
    assert result["provenance"]["null_count"] == 99
    assert result["provenance"]["bootstrap_replicates"] == 2000
    assert len(result["raw_summaries"]) == 5
    assert all(len(summary["holdout_groups"]) == 4 for summary in result["raw_summaries"])
    assert [len(summary["null_group_scores"]) for summary in result["raw_summaries"]] == [20, 20, 20, 20, 19]
    assert [len(summary["null_families"]) for summary in result["raw_summaries"]] == [20, 20, 20, 20, 19]
    assert result["provenance"]["null_family_counts"] == {"shuffled": 33, "random": 33, "matched": 33}
    assert set(result["controls"]) == {
        "shuffled_concept_labels",
        "random_concept_directions",
        "matched_norm_null",
        "off_target_target_token",
        "zero_strength_identity",
    }
    assert all(
        set(control["metrics"]) == {"absolute_margin_difference"}
        if name == "zero_strength_identity"
        else set(control["metrics"]) == {"tcav_sensitivity"}
        for name, control in result["controls"].items()
    )
    assert all(
        abs(float(norm) - 1.0) < 1e-9
        for summary in result["raw_summaries"]
        for family, norm in zip(summary["null_families"], summary["null_direction_norms"], strict=True)
        if family == "matched"
    )
    assert any(
        abs(float(norm) - 1.0) > 1e-6
        for summary in result["raw_summaries"]
        for family, norm in zip(summary["null_families"], summary["null_direction_norms"], strict=True)
        if family == "random"
    )
    assert all("prompt" not in summary for summary in result["raw_summaries"])
    assert result["resources"]["network"] == "enabled"
    assert _FakeIntegration.last_instance is not None
    assert _FakeIntegration.last_instance.model.calls < 200


def test_tcav_semantic_failure_builds_valid_nonpromoting_triads() -> None:
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    resources: dict[str, Any] = {"device": "CUDA", "network": "enabled", "cleanup": "done"}
    result: dict[str, Any] = {
        "status": "failed",
        "evidence_eligible": False,
        "acceptance": False,
        "metrics": {},
        "controls": {},
        "raw_summaries": [],
        "resources": resources,
    }
    failure_ref = "l04-explanations.TCAV.attempt1.failure.json"
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "TCAV",
        "failed",
        failure_ref,
        execution_result=result,
        resources=resources,
    )
    run = build_run_record(
        plan,
        artifact,
        "TCAV",
        "failed",
        resources,
        artifact_name="l04-explanations.TCAV.attempt1.partial.json",
    )
    failure = failure_envelope(
        plan,
        "TCAV",
        "failed",
        error=RuntimeError("semantic miss"),
        failure_ref=failure_ref,
        run_record=run,
        resources=resources,
    )
    assert validate_artifact(artifact, plan) == []
    assert validate_run_record(run, artifact, plan) == []
    assert validate_failure(failure, plan, artifact) == []
    assert artifact["accepted_record_ids"] == []
    assert artifact["evidence_level"] == "D0"


def _accepted_metric(point: float, threshold: float, comparator: str) -> dict[str, object]:
    passed = {"<=": point <= threshold, "<": point < threshold, ">=": point >= threshold, ">": point > threshold}[
        comparator
    ]
    return {
        "point_estimate": point,
        "confidence_interval_95": [point, point],
        "units": "dimensionless",
        "aggregation_unit": "independent causal group",
        "statistic": "mean",
        "threshold": threshold,
        "comparator": comparator,
        "pass": passed,
    }


def _accepted_result() -> dict[str, Any]:
    seeds = [17, 29, 41, 53, 67]
    summaries = []
    null_offset = 0
    for seed in seeds:
        null_count = 20 if seed != 67 else 19
        families = [
            ("shuffled", "random", "matched")[index % 3] for index in range(null_offset, null_offset + null_count)
        ]
        summaries.append(
            {
                "seed": seed,
                "train_groups": [f"g{i:02d}" for i in range(1, 9)],
                "holdout_groups": [f"g{i:02d}" for i in range(9, 13)],
                "heldout_group_accuracy": [1.0] * 4,
                "heldout_group_tcav": [1.0] * 4,
                "null_group_scores": [0.0] * null_count,
                "null_direction_norms": [2.0 if family == "random" else 1.0 for family in families],
                "null_families": families,
                "heldout_row_correct": [1.0] * 8,
            }
        )
        null_offset += null_count
    controls = assemble_controls(
        shuffled_scores=[0.0] * 5,
        random_scores=[0.0] * 5,
        matched_scores=[0.0] * 5,
        off_target_scores=[0.0] * 4,
        zero_differences=[0.0] * 4,
        seed=17,
        sensitivity_reference=1.0,
    )
    control_raw = {
        "group_ids": ["g09", "g10", "g11", "g12"],
        "intervention_agreement": [1.0] * 4,
        "off_target_target_token": [0.0] * 4,
        "zero_strength_identity": [0.0] * 4,
    }
    return {
        "status": "passed_real_cuda",
        "evidence_eligible": True,
        "acceptance": True,
        "evidence_level": "D3",
        "metrics": {
            "heldout_accuracy": _accepted_metric(1.0, 0.6, ">"),
            "heldout_accuracy_wilson_lower": _accepted_metric(wilson_lower(8, 8), 0.55, ">"),
            "bootstrap_ci_lower": _accepted_metric(1.0, 0.5, ">"),
            "corrected_empirical_p": _accepted_metric(1.0 / 100.0, 0.05, "<="),
            "intervention_agreement": _accepted_metric(1.0, 0.8, ">"),
        },
        "controls": controls,
        "control_raw": control_raw,
        "raw_summaries": summaries,
        "token_ids": {" true": 1, " false": 2},
        "target_token_strings": {" true": " true", " false": " false"},
        "seeds": seeds,
        "layer": 6,
        "native_hidden_state_index": 7,
        "provenance": {
            "runtime": "real TransformerLMIntegration",
            "model_id": "openai-community/gpt2",
            "model_revision": "e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
            "network": "enabled",
            "device": "fake CUDA",
            "deterministic_algorithms": True,
            "runtime_versions": {"torch": "test"},
            "resource_peak": {"max_memory_allocated_bytes": 1},
            "null_count": 99,
            "bootstrap_replicates": 2000,
            "direction_fit": "train groups only",
            "concept_factor": "tone_positive",
            "primary_sensitivity": 1.0,
            "concept_direction_norm": 1.0,
            "null_family_counts": {"shuffled": 33, "random": 33, "matched": 33},
            "off_target_token_id": 2,
            "target_token_strings": {" true": " true", " false": " false"},
            "target_position": "last non-padding token",
        },
        "resources": {"device": "fake CUDA", "network": "enabled", "cleanup": "done"},
    }


def test_tcav_passing_triads_and_acceptance_linkage_fail_closed_on_mutation() -> None:
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = _accepted_result()
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "TCAV",
        "passed_real_cuda",
        "l04-explanations.TCAV.attempt1.failure.json",
        execution_result=result,
        resources=result["resources"],
    )
    run = build_run_record(
        plan,
        artifact,
        "TCAV",
        "passed_real_cuda",
        result["resources"],
        artifact_name="l04-explanations.TCAV.attempt1.partial.json",
    )
    failure = failure_envelope(
        plan,
        "TCAV",
        "passed_real_cuda",
        failure_ref="l04-explanations.TCAV.attempt1.failure.json",
        run_record=run,
        resources=result["resources"],
    )
    assert validate_artifact(artifact, plan) == []
    assert validate_run_record(run, artifact, plan) == []
    assert validate_failure(failure, plan, artifact) == []
    assert artifact["evidence_level"] == "D3"
    assert artifact["accepted_record_ids"] == ["t05_tcav"]
    assert artifact["accepted_gap_ids"] == ["THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018"]
    artifact["accepted_record_ids"] = []
    assert validate_artifact(artifact, plan)
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "TCAV",
        "passed_real_cuda",
        "l04-explanations.TCAV.attempt1.failure.json",
        execution_result=result,
        resources=result["resources"],
    )
    artifact["accepted_gap_ids"] = ["THY-T05-LINEAR-PROBING"]
    assert validate_artifact(artifact, plan)
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "TCAV",
        "passed_real_cuda",
        "l04-explanations.TCAV.attempt1.failure.json",
        execution_result=result,
        resources=result["resources"],
    )
    artifact["executions"][1]["control_raw"]["off_target_target_token"] = [1.0] * 4
    assert validate_artifact(artifact, plan)


@pytest.mark.parametrize("field, value", [("evidence_level", "D3"), ("acceptance", True)])
def test_tcav_failed_record_cannot_forge_promotion(field: str, value: object) -> None:
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = _accepted_result()
    result.update({"status": "failed", "evidence_eligible": False, "acceptance": False, "evidence_level": "D0"})
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "TCAV",
        "failed",
        "l04-explanations.TCAV.attempt1.failure.json",
        execution_result=result,
        resources={"device": "fake CUDA", "network": "enabled", "cleanup": "done"},
    )
    active_record = next(
        record
        for record in artifact["records"]
        if record["record_id"] == "THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018"
    )
    active_record[field] = value
    assert validate_artifact(artifact, plan)


def test_tcav_production_dispatch_selects_real_handler_without_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from scripts.m14_l04_explanations import run_real

    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setattr(
        "scripts._m14_l04_tcav.run_tcav",
        lambda _plan, _rows: {
            "status": "failed",
            "failure_reason": "deterministic test failure",
            "resources": {"device": "cuda", "network": "enabled", "cleanup": "done"},
        },
    )
    result = run_real(use_case="TCAV", output_dir=tmp_path)
    assert result["status"] == "failed"
    assert result["artifact"]["provenance"]["evidence_origin"] == "real-cuda"


def test_tcav_injected_dispatch_is_explicitly_non_eligible(tmp_path: Any) -> None:
    from scripts.m14_l04_explanations import run_real

    result = run_real(
        use_case="TCAV",
        output_dir=tmp_path,
        handlers={"TCAV": lambda _plan, _rows: {"status": "passed_real_cuda", "acceptance": True}},
    )
    assert result["status"] == "injected_offline_non_eligible"
    assert result["artifact"]["provenance"]["evidence_origin"] == "dependency-injected-offline"
    assert result["artifact"]["accepted_record_ids"] == []
