"""Focused offline tests for the M14 L04.10 additive steering lane."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import scripts.m14_l04_explanations as explanations
from scripts._m14_l04_artifact import build_artifact
from scripts._m14_l04_contract_common import canonical_json_bytes
from scripts._m14_l04_data import fixture_metadata
from scripts._m14_l04_digest import canonical_digest
from scripts._m14_l04_envelope import build_run_record, failure_envelope
from scripts._m14_l04_fixture_contract import read_fixture
from scripts._m14_l04_steering import (
    LAYER,
    NATIVE_HIDDEN_STATE_INDEX,
    SEEDS,
    STRENGTH_GRID,
    TARGET_TOKEN_IDS,
    TARGET_TOKEN_STRINGS,
    RealExecutionError,
    _shuffled_label_direction,  # pyright: ignore[reportPrivateUsage]
    apply_additive_intervention,
    budget_pass,
    capture_activation,
    normalized_direction,
    paired_absolute_changes,
    run_additive_steering,
)
from scripts._m14_l04_validate import validate_artifact, validate_failure, validate_run_record
from scripts.m14_l04_contract import FIXTURE_PATH, PLAN_PATH, load_plan
from scripts.m14_l04_explanations import run_real


def _unsigned_execution_result_digest(value: dict[str, Any]) -> str:
    unsigned = deepcopy(value)
    provenance = unsigned.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("execution_result_digest", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


class _TokenBlock(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([_TokenBlock() for _ in range(7)])  # type: ignore[attr-defined]
        self.lm_head = nn.Linear(4, 4096, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[11, 0] = 1.0
            self.embedding.weight[3, 0] = -1.0
            self.lm_head.weight.zero_()
            self.lm_head.weight[TARGET_TOKEN_IDS[" true"], 0] = 1.0
            self.lm_head.weight[TARGET_TOKEN_IDS[" false"], 0] = -1.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> object:
        del attention_mask
        hidden = self.embedding(input_ids)
        for block in self.transformer.h:  # type: ignore[attr-defined]
            hidden = block(hidden)
        return type("Output", (), {"logits": self.lm_head(hidden)})()


class _FakeTokenizer:
    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": [TARGET_TOKEN_IDS[text]]}

    def decode(self, ids: list[int]) -> str:
        return TARGET_TOKEN_STRINGS[" true"] if ids[0] == TARGET_TOKEN_IDS[" true"] else TARGET_TOKEN_STRINGS[" false"]


class _FakeIntegration:
    last: _FakeIntegration | None = None

    def __init__(self, **_kwargs: object) -> None:
        self.model = _FakeModel()
        self.tokenizer = _FakeTokenizer()
        self.config = type("Config", (), {"vocab_size": 4096})()
        _FakeIntegration.last = self

    def _backend(self) -> tuple[object, object, object]:
        return self.model, self.tokenizer, self.config

    def tokenize(self, prompts: tuple[str, ...], **_kwargs: object) -> dict[str, torch.Tensor]:
        rows = []
        for prompt in prompts:
            condition_id = 11 if any(word in prompt for word in ("calm", "friendly", "quiet", "helpful")) else 3
            rows.append([5, 6, 4, condition_id])
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 4), dtype=torch.long),
        }


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


def test_normalized_direction_rejects_degenerate_and_preserves_requested_norm() -> None:
    with pytest.raises(ValueError, match="positive"):
        normalized_direction(np.zeros(3))
    direction = normalized_direction(np.asarray([3.0, 4.0]), norm=2.0)
    assert np.allclose(direction, [1.2, 1.6])
    assert np.isclose(np.linalg.norm(direction), 2.0)


def test_additive_intervention_math_is_exact_and_zero_is_identity() -> None:
    model = _FakeModel()
    row = {"input_ids": np.asarray([5, 6, 4, 3]), "attention_mask": np.ones(4), "target_position": 3}
    baseline = apply_additive_intervention(model, row, np.asarray([1.0, 0.0, 0.0, 0.0]), strength=0.0)
    steered = apply_additive_intervention(model, row, np.asarray([1.0, 0.0, 0.0, 0.0]), strength=0.25)
    assert baseline == -2.0
    assert steered == -1.5
    assert not model.transformer.h[LAYER]._forward_hooks  # type: ignore[attr-defined]


def test_capture_activation_removes_hook_after_success_and_failure() -> None:
    model = _FakeModel()
    row = {"input_ids": np.asarray([5, 6, 4, 11]), "attention_mask": np.ones(4), "target_position": 3}
    assert np.array_equal(capture_activation(model, row), np.asarray([1.0, 0.0, 0.0, 0.0]))
    assert not model.transformer.h[LAYER]._forward_hooks  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="dimension"):
        apply_additive_intervention(model, row, np.ones(3), strength=1.0)
    assert not model.transformer.h[LAYER]._forward_hooks  # type: ignore[attr-defined]


def test_budget_pass_is_fail_closed_for_missing_or_over_limit_measurements() -> None:
    valid = {
        "elapsed_seconds": 1.0,
        "max_memory_allocated_bytes": 1,
        "max_memory_reserved_bytes": 2,
        "max_rss_bytes": 3,
    }
    assert budget_pass(valid)
    assert not budget_pass({**valid, "max_rss_bytes": 4 * 1024**3 + 1})
    assert not budget_pass({**valid, "elapsed_seconds": float("nan")})
    assert not budget_pass({"elapsed_seconds": 1.0})


def test_off_target_pair_metric_does_not_cancel_opposite_endpoint_changes() -> None:
    rows = [
        {"group_id": "g01", "causal_pair_id": "p01", "condition": "clean"},
        {"group_id": "g01", "causal_pair_id": "p01", "condition": "corrupted"},
    ]
    # A signed mean would report zero; the frozen locality diagnostic retains
    # the paired absolute change instead.
    assert paired_absolute_changes(rows, [1.0, -1.0]) == [pytest.approx(2.0)]


def test_real_handler_is_network_and_cuda_gated_without_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LATENT_ANYTHING_RUN_NETWORK", raising=False)
    with pytest.raises(RealExecutionError) as raised:
        run_additive_steering(load_plan(), read_fixture(FIXTURE_PATH)[1], integration_factory=_FakeIntegration)
    assert raised.value.resources["stage"] == "preflight"
    assert _FakeIntegration.last is None


def test_real_handler_returns_sanitized_five_seed_strength_and_control_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    _FakeIntegration.last = None
    result = run_additive_steering(load_plan(), read_fixture(FIXTURE_PATH)[1], integration_factory=_FakeIntegration)
    assert result["status"] == "completed_real_cuda_d0"
    assert result["layer"] == LAYER
    assert result["native_hidden_state_index"] == NATIVE_HIDDEN_STATE_INDEX
    assert result["seeds"] == list(SEEDS)
    assert result["strength_grid"] == list(STRENGTH_GRID)
    assert result["token_ids"] == TARGET_TOKEN_IDS
    assert result["target_token_strings"] == TARGET_TOKEN_STRINGS
    assert set(result["controls"]) == {
        "zero_strength",
        "randomized_direction",
        "shuffled_labels",
        "off_target_token",
        "matched_norm_direction",
    }
    assert len(result["raw_summaries"]) == 5
    assert all(
        set(summary["strength_curve"]) == {str(value) for value in STRENGTH_GRID} for summary in result["raw_summaries"]
    )
    assert all(len(summary["holdout_groups"]) == 4 for summary in result["raw_summaries"])
    assert set(result["train_groups"]).isdisjoint(result["holdout_groups"])
    assert result["train_groups"] == [f"g{index:02d}" for index in range(1, 9)]
    assert result["holdout_groups"] == [f"g{index:02d}" for index in range(9, 13)]
    assert all(set(result["controls"][name]["by_seed"]) == {str(seed) for seed in SEEDS} for name in result["controls"])
    assert all(
        summary["control_direction_norms"]["matched_norm"] == pytest.approx(result["direction_norm"])
        for summary in result["raw_summaries"]
    )
    assert all(item["split"] == "holdout" for item in result["holdout_evidence"])
    assert result["resources"]["cleanup"].startswith("CUDA synchronized")
    forbidden = {"prompt", "input_ids", "attention_mask", "weights", "path"}
    assert not forbidden.intersection(result)
    assert all(not forbidden.intersection(item) for item in result["holdout_evidence"])
    assert _FakeIntegration.last is not None
    assert not _FakeIntegration.last.model.transformer.h[LAYER]._forward_hooks  # type: ignore[attr-defined]


def test_cleanup_failure_fails_closed_and_preserves_parameter_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)

    def fail_empty_cache() -> None:
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(torch.cuda, "empty_cache", fail_empty_cache)
    result = run_additive_steering(load_plan(), read_fixture(FIXTURE_PATH)[1], integration_factory=_FakeIntegration)
    assert result["status"] == "failed"
    assert result["evidence_eligible"] is False
    assert result["acceptance"] is False
    assert result["evidence_level"] == "D0"
    assert result["semantic_candidate"] is False
    assert result["criteria"]["cleanup_complete"] is False
    assert result["resources"]["cleanup_complete"] is False
    assert result["resources"]["cleanup_error"] == "RuntimeError"
    assert result["model_parameter_digest_before"] == result["model_parameter_digest_after"]
    assert result["provenance"]["model_parameter_digest_before"] == result["model_parameter_digest_after"]


def test_hook_cleanup_failure_is_explicit_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel()
    original_remove = torch.utils.hooks.RemovableHandle.remove

    def fail_remove(handle: torch.utils.hooks.RemovableHandle) -> None:
        original_remove(handle)
        raise RuntimeError("injected hook cleanup failure")

    monkeypatch.setattr(torch.utils.hooks.RemovableHandle, "remove", fail_remove)
    row = {"input_ids": np.asarray([5, 6, 4, 11]), "attention_mask": np.ones(4), "target_position": 3}
    with pytest.raises(RuntimeError, match="hook cleanup failed"):
        capture_activation(model, row)
    assert not model.transformer.h[LAYER]._forward_hooks  # type: ignore[attr-defined]


def test_successful_phase_a_runtime_builds_validator_clean_d0_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "AdditiveSteering",
        result["status"],
        "l04-explanations.AdditiveSteering.attempt1.failure.json",
        execution_result=result,
        resources=result["resources"],
    )
    active = next(item for item in artifact["executions"] if item["use_case"] == "AdditiveSteering")
    assert artifact["evidence_level"] == "D0"
    assert artifact["accepted_record_ids"] == []
    assert artifact["accepted_gap_ids"] == []
    assert active["evidence_level"] == "D0"
    assert active["evidence_eligible"] is False
    assert active["acceptance"] is False
    assert validate_artifact(artifact, plan) == []


def test_malformed_fixture_fails_before_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    _FakeIntegration.last = None
    rows = read_fixture(FIXTURE_PATH)[1]
    rows[0] = {**rows[0], "split": "holdout"}
    with pytest.raises(RealExecutionError, match="additive steering failed"):
        run_additive_steering(load_plan(), rows, integration_factory=_FakeIntegration)
    assert _FakeIntegration.last is None


def test_dispatcher_accepts_injected_additive_steering_handler(tmp_path: Path) -> None:
    def handler(received_plan: dict[str, Any], received_rows: list[dict[str, Any]]) -> dict[str, Any]:
        assert received_plan["lane"] == "L04"
        assert len(received_rows) == 24
        return {"status": "failed", "failure_reason": "offline test"}

    result = run_real(
        plan_path=PLAN_PATH,
        fixture_path=FIXTURE_PATH,
        use_case="AdditiveSteering",
        output_dir=tmp_path,
        handlers={"AdditiveSteering": handler},
    )
    assert result["status"] == "injected_offline_non_eligible"
    assert result["use_case"] == "AdditiveSteering"
    assert result["paths"]["partial"].endswith(".partial.json")


def test_additive_builder_rejects_handler_raw_output_and_forged_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    result["raw_output"] = "do not retain"
    with pytest.raises(ValueError, match="unexpected fields"):
        build_artifact(
            plan,
            fixture_metadata(plan, raw, rows),
            "AdditiveSteering",
            result["status"],
            "failure.json",
            execution_result=result,
            resources=result["resources"],
        )
    result.pop("raw_output")
    result["evidence_level"] = "D3"
    with pytest.raises(ValueError, match="non-promoting D0"):
        build_artifact(
            plan,
            fixture_metadata(plan, raw, rows),
            "AdditiveSteering",
            result["status"],
            "failure.json",
            execution_result=result,
            resources=result["resources"],
        )


def test_additive_builder_rejects_malformed_resource_and_digest_linkage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    result["resources"]["resource_peak"]["elapsed_seconds"] = True
    with pytest.raises(ValueError, match="resource|elapsed_seconds"):
        build_artifact(
            plan,
            fixture_metadata(plan, raw, rows),
            "AdditiveSteering",
            result["status"],
            "failure.json",
            execution_result=result,
            resources=result["resources"],
        )
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    result["provenance"]["model_parameter_digest_after"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        build_artifact(
            plan,
            fixture_metadata(plan, raw, rows),
            "AdditiveSteering",
            result["status"],
            "failure.json",
            execution_result=result,
            resources=result["resources"],
        )


def test_pre_cuda_failure_dispatcher_and_main_emit_validator_clean_d0_envelopes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = run_real(use_case="AdditiveSteering", output_dir=tmp_path)
    plan = load_plan()
    assert result["artifact"]["provenance"]["execution_attempted"] is False
    assert result["artifact"]["provenance"]["execution_backend"] == "none"
    assert validate_artifact(result["artifact"], plan) == []
    assert validate_run_record(result["run_record"], result["artifact"], plan) == []
    assert validate_failure(result["failure"], plan, result["artifact"]) == []

    monkeypatch.setattr(explanations, "run_real", lambda **_kwargs: result)
    with pytest.raises(SystemExit) as raised:
        explanations.main(["--run-real", "--use-case", "AdditiveSteering", "--output-dir", str(tmp_path)])
    assert raised.value.code == 1


def test_cuda_preflight_exception_stays_pre_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")

    def fail_cuda_probe() -> bool:
        raise RuntimeError("driver unavailable")

    monkeypatch.setattr(torch.cuda, "is_available", fail_cuda_probe)
    with pytest.raises(RealExecutionError) as raised:
        run_additive_steering(load_plan(), read_fixture(FIXTURE_PATH)[1], integration_factory=_FakeIntegration)
    assert raised.value.resources == {
        "device": "not used",
        "network": "not attempted",
        "resource_peak": "not measured",
        "cleanup": "pending",
        "execution_attempted": False,
        "execution_backend": "none",
        "stage": "cuda_check",
    }


def test_malformed_additive_handler_result_is_published_as_failed_d0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    monkeypatch.setenv("LATENT_ANYTHING_NETWORK_DEVICE", "cuda")
    import scripts._m14_l04_steering as steering

    monkeypatch.setattr(steering, "run_additive_steering", lambda _plan, _rows: {"status": "completed_real_cuda_d0"})
    result = run_real(use_case="AdditiveSteering", output_dir=tmp_path)
    plan = load_plan()
    assert result["status"] == "failed"
    assert result["artifact"]["provenance"]["stage"] == "preflight"
    assert validate_artifact(result["artifact"], plan) == []
    assert validate_run_record(result["run_record"], result["artifact"], plan) == []
    assert validate_failure(result["failure"], plan, result["artifact"]) == []


def test_additive_handler_constants_are_rejected_instead_of_canonicalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    mutations = (
        ("seeds", [17, 29, 41, 53, 68]),
        ("strength_grid", [0.0, 0.25, 0.5, 2.0]),
        ("train_groups", ["g01"]),
        ("holdout_groups", ["g09"]),
        ("layer", 5),
        ("native_hidden_state_index", 6),
        ("token_ids", {" true": 1, " false": 2}),
        ("target_token_strings", {" true": "yes", " false": "no"}),
        ("direction_norm", 0.0),
        ("direction_norm", float("nan")),
    )
    for field, value in mutations:
        forged = deepcopy(result)
        forged[field] = value
        with pytest.raises(ValueError, match="additive"):
            build_artifact(
                plan,
                fixture_metadata(plan, raw, rows),
                "AdditiveSteering",
                result["status"],
                "failure.json",
                execution_result=forged,
                resources=forged["resources"],
            )


def test_additive_rehashed_divergence_and_recursive_unknowns_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "AdditiveSteering",
        result["status"],
        "failure.json",
        execution_result=result,
        resources=result["resources"],
    )
    mutations = []
    divergent = deepcopy(artifact)
    divergent["metrics"]["target_effect"]["17"]["point_estimate"] += 1.0
    mutations.append(divergent)
    sensitive = deepcopy(artifact)
    sensitive["raw_summaries"][0]["prompt"] = "must not be retained"
    mutations.append(sensitive)
    unknown_fixture = deepcopy(artifact)
    unknown_fixture["fixture"]["unknown"] = True
    mutations.append(unknown_fixture)
    unknown_top_level = deepcopy(artifact)
    unknown_top_level["raw_output"] = "must not be retained"
    mutations.append(unknown_top_level)
    for forged in mutations:
        forged["artifact_sha256"] = canonical_digest(forged, "artifact_sha256")
        assert validate_artifact(forged, plan)


def test_completed_additive_status_links_through_failure_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "AdditiveSteering",
        result["status"],
        "l04-explanations.AdditiveSteering.attempt1.failure.json",
        execution_result=result,
        resources=result["resources"],
    )
    run = build_run_record(
        plan,
        artifact,
        "AdditiveSteering",
        result["status"],
        result["resources"],
        artifact_name="l04-explanations.AdditiveSteering.attempt1.partial.json",
    )
    failure = failure_envelope(
        plan,
        "AdditiveSteering",
        result["status"],
        failure_ref="l04-explanations.AdditiveSteering.attempt1.failure.json",
        run_record=run,
        resources=result["resources"],
    )
    assert validate_run_record(run, artifact, plan) == []
    assert validate_failure(failure, plan, artifact) == []
    completed_result = {"status": result["status"], "artifact": artifact, "run_record": run, "failure": failure}
    monkeypatch.setattr(explanations, "run_real", lambda **_kwargs: completed_result)
    explanations.main(["--run-real", "--use-case", "AdditiveSteering"])


def test_completed_additive_requires_complete_contract_at_all_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "AdditiveSteering",
        result["status"],
        "l04-explanations.AdditiveSteering.attempt1.failure.json",
        execution_result=result,
        resources=result["resources"],
    )
    run = build_run_record(plan, artifact, "AdditiveSteering", result["status"], result["resources"])
    failure = failure_envelope(
        plan,
        "AdditiveSteering",
        result["status"],
        failure_ref="l04-explanations.AdditiveSteering.attempt1.failure.json",
        run_record=run,
        resources=result["resources"],
    )
    for field in ("metrics", "controls", "raw_summaries", "criteria", "direction_norm", "provenance"):
        forged = deepcopy(artifact)
        active = next(item for item in forged["executions"] if item["use_case"] == "AdditiveSteering")
        active.pop(field, None)
        forged.pop(field, None)
        forged["artifact_sha256"] = canonical_digest(forged, "artifact_sha256")
        assert validate_artifact(forged, plan)
        assert validate_run_record(run, forged, plan)
        assert validate_failure(failure, plan, forged)


def test_every_additive_retained_location_is_linked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A self-rehashed divergence cannot survive one envelope boundary."""
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "AdditiveSteering",
        result["status"],
        "failure.json",
        execution_result=result,
        resources=result["resources"],
    )
    active = next(item for item in artifact["executions"] if item["use_case"] == "AdditiveSteering")
    record = next(item for item in artifact["records"] if item["record_id"].startswith("THY-T05-STEERING"))

    mutations: list[dict[str, Any]] = []
    top_level = deepcopy(artifact)
    top_level["direction_norm"] += 1.0
    mutations.append(top_level)
    execution = deepcopy(artifact)
    execution_active = next(item for item in execution["executions"] if item["use_case"] == "AdditiveSteering")
    execution_active["criteria"]["finite"] = not execution_active["criteria"]["finite"]
    mutations.append(execution)
    record_mutation = deepcopy(artifact)
    record_active = next(
        item for item in record_mutation["records"] if item["record_id"].startswith("THY-T05-STEERING")
    )
    record_active["target_token_strings"] = {" true": "tampered", " false": "tampered"}
    mutations.append(record_mutation)
    provenance = deepcopy(artifact)
    provenance["provenance"]["direction_fit"] = "tampered"
    mutations.append(provenance)
    execution_provenance = deepcopy(artifact)
    next(item for item in execution_provenance["executions"] if item["use_case"] == "AdditiveSteering")["provenance"][
        "direction_fit"
    ] = "tampered"
    mutations.append(execution_provenance)
    record_provenance = deepcopy(artifact)
    next(item for item in record_provenance["records"] if item["record_id"].startswith("THY-T05-STEERING"))[
        "provenance"
    ]["direction_fit"] = "tampered"
    mutations.append(record_provenance)
    assert active["direction_norm"] == record["direction_norm"]
    for forged in mutations:
        forged["artifact_sha256"] = canonical_digest(forged, "artifact_sha256")
        assert validate_artifact(forged, plan)


def test_pre_cuda_resources_reject_recursive_unknown_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "AdditiveSteering",
        "failed",
        "failure.json",
        resources={
            "device": "not used",
            "network": "not attempted",
            "resource_peak": "not measured",
            "cleanup": "pending",
            "execution_attempted": False,
            "execution_backend": "none",
            "stage": "cuda_check",
        },
    )
    artifact["resources"]["unknown"] = True
    artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    assert validate_artifact(artifact, plan)


def test_additive_holdout_requires_canonical_pairs_and_unique_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    for mutate in (
        lambda value: value["holdout_evidence"].__setitem__(0, {**value["holdout_evidence"][0], "group_id": "g01"}),
        lambda value: value["holdout_evidence"].__setitem__(
            1, {**value["holdout_evidence"][1], "row_id": "l04-g09-clean"}
        ),
        lambda value: value["holdout_evidence"].__setitem__(
            0, {**value["holdout_evidence"][0], "causal_pair_id": "p10"}
        ),
    ):
        forged = deepcopy(result)
        mutate(forged)
        with pytest.raises(ValueError, match="holdout"):
            build_artifact(
                plan,
                fixture_metadata(plan, raw, rows),
                "AdditiveSteering",
                forged["status"],
                "failure.json",
                execution_result=forged,
                resources=forged["resources"],
            )


def test_additive_fixture_metadata_rejects_forged_self_rehash(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    forged_fixture = fixture_metadata(plan, raw, rows)
    forged_fixture["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixture"):
        build_artifact(
            plan,
            forged_fixture,
            "AdditiveSteering",
            result["status"],
            "failure.json",
            execution_result=result,
            resources=result["resources"],
        )


def test_additive_result_cross_links_reject_independent_rehash_substitutions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    for mutate in (
        lambda value: value["metrics"]["target_effect"]["17"].__setitem__(
            "point_estimate", value["metrics"]["target_effect"]["17"]["point_estimate"] + 1.0
        ),
        lambda value: value["controls"]["shuffled_labels"]["by_seed"]["17"].__setitem__(
            "point_estimate", value["controls"]["shuffled_labels"]["by_seed"]["17"]["point_estimate"] + 1.0
        ),
        lambda value: value["raw_summaries"][0]["control_direction_norms"].__setitem__(
            "matched_norm", value["raw_summaries"][0]["control_direction_norms"]["matched_norm"] + 1.0
        ),
    ):
        forged = deepcopy(result)
        mutate(forged)
        with pytest.raises(ValueError, match="link|norm|summary"):
            build_artifact(
                plan,
                fixture_metadata(plan, raw, rows),
                "AdditiveSteering",
                forged["status"],
                "failure.json",
                execution_result=forged,
                resources=forged["resources"],
            )


def test_post_cuda_real_execution_failure_keeps_scoring_stage_and_validates(
    tmp_path: Path,
) -> None:
    def handler(_plan: dict[str, Any], _rows: list[dict[str, Any]]) -> dict[str, Any]:
        raise RealExecutionError(
            "scoring failed",
            {
                "device": "synthetic CUDA",
                "network": "enabled",
                "resource_peak": "not measured",
                "cleanup": "pending",
                "execution_attempted": True,
                "execution_backend": "cuda",
                "stage": "scoring",
            },
        )

    result = run_real(
        use_case="AdditiveSteering",
        output_dir=tmp_path,
        handlers={"AdditiveSteering": handler},
    )
    plan = load_plan()
    assert result["artifact"]["provenance"]["stage"] == "scoring"
    assert result["failure"]["stage"] == "scoring"
    assert result["artifact"]["resources"]["resource_peak"] == "not measured"
    assert validate_artifact(result["artifact"], plan) == []
    assert validate_run_record(result["run_record"], result["artifact"], plan) == []
    assert validate_failure(result["failure"], plan, result["artifact"]) == []


def test_attempted_generic_stage_is_preserved_without_echoing_unsafe_device(tmp_path: Path) -> None:
    def handler(_plan: dict[str, Any], _rows: list[dict[str, Any]]) -> dict[str, Any]:
        raise RealExecutionError(
            "execution failed",
            {
                "device": r"C:\secret-token",
                "network": "enabled",
                "resource_peak": "not measured",
                "cleanup": "pending",
                "execution_attempted": True,
                "execution_backend": "cuda",
                "stage": "execution",
            },
        )

    result = run_real(
        use_case="AdditiveSteering",
        output_dir=tmp_path,
        handlers={"AdditiveSteering": handler},
    )
    plan = load_plan()
    serialized = str(result)
    assert r"C:\secret-token" not in serialized
    assert result["artifact"]["provenance"]["stage"] == "execution"
    assert result["artifact"]["provenance"]["execution_attempted"] is True
    assert result["artifact"]["provenance"]["execution_backend"] == "cuda"
    assert result["failure"]["stage"] == "execution"
    assert result["failure"]["execution_attempted"] is True
    assert validate_artifact(result["artifact"], plan) == []
    assert validate_run_record(result["run_record"], result["artifact"], plan) == []
    assert validate_failure(result["failure"], plan, result["artifact"]) == []


def test_shuffled_label_digest_covers_actual_ordered_assignment() -> None:
    rows = [{"condition": "clean" if index < 8 else "corrupted"} for index in range(16)]
    activations = [np.asarray([float(index), 1.0]) for index in range(16)]

    class _Rng:
        def permutation(self, size: int) -> np.ndarray:
            permutation = np.arange(size)
            permutation[[0, 8]] = permutation[[8, 0]]
            return permutation

    _direction, provenance = _shuffled_label_direction(rows, activations, _Rng())  # type: ignore[arg-type]
    assignment = [-1] + [1] * 8 + [-1] * 7
    expected = hashlib.sha256(canonical_json_bytes({"ordered_labels": assignment})).hexdigest()
    assert provenance["identity_permutation"] is False
    assert provenance["permutation_digest"] == expected


def test_rehashed_additive_threshold_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    result["metrics"]["target_effect"]["17"]["threshold"] = 999
    result["provenance"]["execution_result_digest"] = _unsigned_execution_result_digest(result)
    with pytest.raises(ValueError, match="pinned|declaration"):
        build_artifact(
            plan,
            fixture_metadata(plan, raw, rows),
            "AdditiveSteering",
            result["status"],
            "failure.json",
            execution_result=result,
            resources=result["resources"],
        )


def test_additive_resource_tuple_must_match_run_and_failure_envelopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda(monkeypatch)
    plan = load_plan()
    raw, rows = read_fixture(FIXTURE_PATH)
    result = run_additive_steering(plan, rows, integration_factory=_FakeIntegration)
    artifact = build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "AdditiveSteering",
        result["status"],
        "l04-explanations.AdditiveSteering.attempt1.failure.json",
        execution_result=result,
        resources=result["resources"],
    )
    run = build_run_record(plan, artifact, "AdditiveSteering", result["status"], result["resources"])
    forged_run = deepcopy(run)
    forged_run["device"] = "other CUDA"
    forged_run["run_record_sha256"] = canonical_digest(forged_run, "run_record_sha256")
    assert validate_run_record(forged_run, artifact, plan)
    failure = failure_envelope(
        plan,
        "AdditiveSteering",
        result["status"],
        failure_ref="l04-explanations.AdditiveSteering.attempt1.failure.json",
        run_record=run,
        resources=result["resources"],
    )
    forged_failure = deepcopy(failure)
    forged_failure["resource"]["resource_peak"] = "not measured"
    forged_failure["failure_sha256"] = canonical_digest(forged_failure, "failure_sha256")
    assert validate_failure(forged_failure, plan, artifact)
