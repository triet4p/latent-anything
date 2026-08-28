"""Offline contract tests for the M14 L04.6 direct logit-lens handler."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from scripts._m14_l04_artifact import build_artifact
from scripts._m14_l04_data import fixture_metadata
from scripts._m14_l04_direct_lens import run_direct_logit_lens
from scripts._m14_l04_envelope import build_run_record, failure_envelope
from scripts._m14_l04_fixture_contract import read_fixture
from scripts._m14_l04_validate import validate_artifact, validate_failure, validate_run_record
from scripts.m14_l04_contract import FIXTURE_PATH, load_plan


class _FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [1 if text == " true" else 2]}

    def decode(self, ids: list[int]) -> str:
        return " true" if ids[0] == 1 else " false"


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class _FakeIntegration:
    provenance = "fake-direct-lens"
    last_instance: _FakeIntegration | None = None

    def __init__(self, **_kwargs: object) -> None:
        self.model = _FakeModel()
        self.tokenizer = _FakeTokenizer()
        _FakeIntegration.last_instance = self

    def _backend(self) -> tuple[object, object, object]:
        return self.model, self.tokenizer, SimpleNamespace(num_hidden_layers=12)

    def generate(self, request: Any) -> Any:
        prompts = request.prompt
        batch = len(prompts)
        seq_len = 4
        vocab = 4
        input_ids = np.tile(np.arange(seq_len, dtype=np.int64), (batch, 1))
        mask = np.ones((batch, seq_len), dtype=np.int64)
        hidden_states = []
        lens_results = []
        for layer in range(13):
            hidden = np.full((batch, seq_len, 3), layer + 1.0, dtype=np.float32)
            logits = np.zeros((batch, seq_len, vocab), dtype=np.float32)
            logits[:, :, 1] = layer + 1.0
            logits[:, :, 2] = -(layer + 1.0)
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
            hidden_states.append(SimpleNamespace(layer=layer, values=hidden))
            lens_results.append(SimpleNamespace(layer=layer, logits=logits, probabilities=probabilities))
        return SimpleNamespace(
            input_ids=input_ids,
            attention_mask=mask,
            logits=lens_results[-1].logits.copy(),
            hidden_states=tuple(hidden_states),
            lens_results=tuple(lens_results),
        )


def _fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_direct_lens_handler_uses_batched_integration_and_native_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    rows = read_fixture(FIXTURE_PATH)[1]
    result = run_direct_logit_lens(plan, rows, integration_factory=_FakeIntegration)

    assert result["status"] == "passed_real_cuda"
    assert result["evidence_eligible"] is True
    assert result["acceptance"] is True
    assert result["evidence_level"] == "D0"
    assert result["seeds"] == [17, 29, 41, 53, 67]
    assert len(result["raw_summaries"]) == 5
    assert all(summary["layer_indices"] == list(range(13)) for summary in result["raw_summaries"])
    assert all(len(summary["rows"]) == 24 for summary in result["raw_summaries"])
    assert len(result["control_raw"]["holdout_group_margins"]) == 20
    assert set(result["controls"]) == {
        "target_non_target_selectivity",
        "shuffled_target_labels",
        "randomized_target_tokens",
        "terminal_post_ln_f_parity",
    }
    assert result["token_ids"] == {" true": 1, " false": 2}
    assert result["provenance"]["native_layer_indices"] == list(range(13))
    assert result["provenance"]["aggregation_unit"] == "independent causal group"


def test_direct_lens_triads_validate_and_cannot_promote(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_direct_logit_lens(plan, rows, integration_factory=_FakeIntegration)
    resources = result["resources"]
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "DirectLogitLens",
        result["status"],
        "l04-explanations.DirectLogitLens.attempt1.failure.json",
        execution_result=result,
        resources=resources,
    )
    run = build_run_record(
        plan,
        artifact,
        "DirectLogitLens",
        result["status"],
        resources,
        artifact_name="l04-explanations.DirectLogitLens.attempt1.partial.json",
    )
    failure = failure_envelope(
        plan,
        "DirectLogitLens",
        result["status"],
        failure_ref="l04-explanations.DirectLogitLens.attempt1.failure.json",
        run_record=run,
        resources=resources,
    )
    assert validate_artifact(artifact, plan) == []
    assert validate_run_record(run, artifact, plan) == []
    assert validate_failure(failure, plan, artifact) == []
    assert artifact["accepted_record_ids"] == []
    assert artifact["accepted_gap_ids"] == []
    assert artifact["evidence_level"] == "D0"

    artifact["raw_summaries"][0]["rows"][0]["target_margin"] += 0.1
    assert validate_artifact(artifact, plan)


def test_direct_lens_semantic_failure_emits_valid_nonpromoting_triads(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    original_generate = _FakeIntegration.generate

    def mismatched_generate(self: _FakeIntegration, request: Any) -> Any:
        result = original_generate(self, request)
        result.logits[:, :, 0] += 1.0
        return result

    monkeypatch.setattr(_FakeIntegration, "generate", mismatched_generate)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_direct_logit_lens(plan, rows, integration_factory=_FakeIntegration)

    assert result["status"] == "failed"
    assert result["evidence_eligible"] is False
    assert result["acceptance"] is False
    resources = result["resources"]
    failure_ref = "l04-explanations.DirectLogitLens.attempt2.failure.json"
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "DirectLogitLens",
        result["status"],
        failure_ref,
        execution_result=result,
        resources=resources,
    )
    run = build_run_record(
        plan,
        artifact,
        "DirectLogitLens",
        result["status"],
        resources,
        artifact_name="l04-explanations.DirectLogitLens.attempt2.partial.json",
    )
    failure = failure_envelope(
        plan,
        "DirectLogitLens",
        result["status"],
        error=RuntimeError("direct parity gate failed"),
        failure_ref=failure_ref,
        run_record=run,
        resources=resources,
    )
    assert validate_artifact(artifact, plan) == []
    assert validate_run_record(run, artifact, plan) == []
    assert validate_failure(failure, plan, artifact) == []
    assert artifact["accepted_record_ids"] == []
    assert artifact["accepted_gap_ids"] == []
    active = next(item for item in artifact["executions"] if item["use_case"] == "DirectLogitLens")
    active["evidence_eligible"] = True
    assert validate_artifact(artifact, plan)


def test_direct_lens_requires_cuda_network_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cpu")
    with pytest.raises(Exception, match="NETWORK_DEVICE=cuda"):
        run_direct_logit_lens(load_plan(), [], integration_factory=_FakeIntegration)


def test_direct_lens_injected_dispatch_is_explicitly_non_eligible(tmp_path: Any) -> None:
    from scripts.m14_l04_explanations import run_real

    result = run_real(
        use_case="DirectLogitLens",
        output_dir=tmp_path,
        handlers={"DirectLogitLens": lambda _plan, _rows: {"status": "passed_real_cuda", "acceptance": True}},
    )
    assert result["status"] == "injected_offline_non_eligible"
    assert result["artifact"]["provenance"]["evidence_origin"] == "dependency-injected-offline"
    assert result["artifact"]["accepted_record_ids"] == []
