"""Offline contract tests for the L04.9 true-interchange boundary."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from scripts._m14_l04_activation_patching import (
    BOOTSTRAP_REPLICATES,
    LAYER,
    NATIVE_HIDDEN_STATE_INDEX,
    OFF_TARGET_LAYER,
    RECOVERY_CI_LOWER_THRESHOLD,
    SEEDS,
    TARGET_TOKEN_IDS,
    TARGET_TOKEN_STRINGS,
    _capture_layers,  # pyright: ignore[reportPrivateUsage]
    _metric,  # pyright: ignore[reportPrivateUsage]
    _pairs,  # pyright: ignore[reportPrivateUsage]
    deterministic_donor_derangement,
    deterministic_split_donor_derangement,
    donor_mapping_digest,
    patched_margin,
)
from scripts._m14_l04_artifact import build_artifact
from scripts._m14_l04_contract_common import canonical_json_bytes
from scripts._m14_l04_data import fixture_metadata
from scripts._m14_l04_digest import canonical_digest
from scripts._m14_l04_fixture_contract import read_fixture
from scripts._m14_l04_validate import validate_artifact, validate_failure, validate_run_record
from scripts._m14_l04_validate_activation_patching import (
    _expected_linkage,  # pyright: ignore[reportPrivateUsage]
    validate_real_true_activation_patching_execution,
)
from scripts.m14_l04_contract import FIXTURE_PATH, PLAN_PATH, load_plan
from scripts.m14_l04_explanations import run_real

SIDECAR_PATH = Path(__file__).resolve().parents[1] / (
    "artifacts/m14/l04-explanations.ssh.TrueActivationPatching.3d5d6720ee5fe155ff4a1f1c25814225b66170f3.sidecar.json"
)


class _FakeBlock(nn.Module):
    def __init__(self, output_kind: str) -> None:
        super().__init__()
        self.output_kind = output_kind
        self.aux = object()

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, object] | list[object]:
        if self.output_kind == "tuple":
            return values, self.aux
        return [values, self.aux]


class _FakeTransformerModel(nn.Module):
    def __init__(self, output_kind: str) -> None:
        super().__init__()
        transformer = nn.Module()
        transformer.h = nn.ModuleList([_FakeBlock(output_kind) for _ in range(7)])  # type: ignore[attr-defined]
        self.transformer = transformer
        self.embedding = nn.Embedding(8, 2)
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(8, 2))
            self.head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        self.fail = False
        self.aux_seen: list[object] = []

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        del attention_mask
        if self.fail:
            raise RuntimeError("intentional fake forward failure")
        values = self.embedding(input_ids)
        for block in self.transformer.h:  # type: ignore[attr-defined]
            output = block(values)
            self.aux_seen.append(output[1])
            values = output[0]  # type: ignore[assignment]
        return type("FakeOutput", (), {"logits": self.head(values)})()


def _row(pair: str, condition: str, *, group: str = "g09", split: str = "holdout") -> dict[str, object]:
    return {
        "row_id": f"{pair}-{condition}",
        "group_id": group,
        "causal_pair_id": pair,
        "condition": condition,
        "split": split,
        "prompt": "A prompt",
        "target_text": " true" if condition == "clean" else " false",
        "factor_labels": {"animal_cat": 0, "tone_positive": 0},
    }


def test_target_token_contract_is_frozen() -> None:
    assert TARGET_TOKEN_IDS == {" true": 2081, " false": 3991}


def test_donor_mapping_is_deterministic_and_non_self() -> None:
    first = deterministic_donor_derangement(["p01", "p02", "p03", "p04"], 17)
    assert first == {"p01": "p03", "p02": "p04", "p03": "p02", "p04": "p01"}
    assert all(source != target for source, target in first.items())
    expected_digest = hashlib.sha256(canonical_json_bytes(dict(sorted(first.items())))).hexdigest()
    assert donor_mapping_digest(first) == expected_digest


def test_split_donor_mapping_never_crosses_split() -> None:
    pairs = {
        f"p{index:02d}": {
            "clean": {"split": split},
            "corrupted": {"split": split},
        }
        for index, split in enumerate(
            ("train", "train", "train", "train", "holdout", "holdout", "holdout", "holdout"), 1
        )
    }
    mapping = deterministic_split_donor_derangement(pairs, 17)
    assert set(mapping) == set(pairs)
    assert set(mapping.values()) == set(pairs)
    assert all(source != target for source, target in mapping.items())
    assert all(pairs[source]["clean"]["split"] == pairs[target]["clean"]["split"] for source, target in mapping.items())


def test_pairs_require_one_clean_and_corrupted_row() -> None:
    pairs = _pairs([_row("p01", "clean"), _row("p01", "corrupted")])
    assert set(pairs["p01"]) == {"clean", "corrupted"}
    with pytest.raises(ValueError, match="one clean"):
        _pairs([_row("p01", "clean"), _row("p01", "clean")])


def test_pairs_reject_group_split_mismatch() -> None:
    with pytest.raises(ValueError, match="inconsistent group or split"):
        _pairs([_row("p01", "clean"), _row("p01", "corrupted", group="g10")])


@pytest.mark.parametrize("output_kind", ["tuple", "list"])
def test_patching_fake_transformer_replaces_at_exact_position_and_cleans_hooks(output_kind: str) -> None:
    model = _FakeTransformerModel(output_kind)
    row = {
        "input_ids": [2, 2, 2],
        "attention_mask": [1, 1, 1],
        "target_position": 2,
    }
    clean = {"input_ids": [1, 1, 1], "attention_mask": [1, 1, 1], "target_position": 2}
    donor = _capture_layers(model, clean, (6,))[6]
    baseline = patched_margin(model, row, layer=6, donor=donor, target_token=0, other_token=1, strength=0.0)
    replaced = patched_margin(model, row, layer=6, donor=donor, target_token=0, other_token=1, strength=1.0)
    half = patched_margin(model, row, layer=6, donor=donor, target_token=0, other_token=1, strength=0.5)
    off_position = patched_margin(
        model, row, layer=6, donor=donor, target_token=0, other_token=1, strength=1.0, position=1
    )
    assert baseline == 4.0
    assert replaced == 2.0
    assert half == 3.0
    assert off_position == baseline
    assert model.aux_seen[-1] is model.transformer.h[6].aux  # type: ignore[attr-defined]
    assert not model.transformer.h[6]._forward_hooks  # type: ignore[attr-defined]

    model.fail = True
    with pytest.raises(RuntimeError, match="intentional fake"):
        patched_margin(model, row, layer=6, donor=donor, target_token=0, other_token=1)
    assert not model.transformer.h[6]._forward_hooks  # type: ignore[attr-defined]


def test_malformed_nested_validator_is_fail_closed() -> None:
    result = validate_real_true_activation_patching_execution(
        {"status": "passed_real_cuda", "raw_summaries": [{"holdout_evidence": [{"hidden": object()}]}]},
        {},
        {"fixture": {"split": {"holdout_groups": []}}},
    )
    assert result and all(isinstance(message, str) for message in result)


def test_injected_activation_failure_is_sanitized_without_secret_payload(tmp_path: Path) -> None:
    plan = load_plan(PLAN_PATH)

    def failing_handler(_plan: dict[str, Any], _rows: list[dict[str, Any]]) -> dict[str, Any]:
        raise RuntimeError("PROMPT SECRET hidden_states tensor payload")

    result = run_real(
        plan_path=PLAN_PATH,
        fixture_path=FIXTURE_PATH,
        use_case="TrueActivationPatching",
        output_dir=tmp_path,
        handlers={"TrueActivationPatching": failing_handler},
    )
    serialized = json.dumps(result, sort_keys=True)
    assert "PROMPT SECRET" not in serialized
    assert result["failure"]["exception"] == "true_activation_patching_failed:execution"
    assert plan["model"]["id"] in serialized


def test_historical_activation_sidecar_is_canonical_and_sanitized() -> None:
    sidecar = json.loads(SIDECAR_PATH.read_bytes())
    assert (
        canonical_digest(sidecar, "sidecar_sha256")
        == sidecar["sidecar_sha256"]
        == ("41ba907d9101f5e4ce3a038c240520ac51db5fc481b069dc0c5d5ef3fe11e463")
    )
    assert sidecar["source_sha"] == "3d5d6720ee5fe155ff4a1f1c25814225b66170f3"
    assert sidecar["plan_sha256"] == "f3c315e356af0ee54d4196cc365ee22bd997b069d18a3e72c6b479f94e0b3e1a"
    assert sidecar["use_case"] == "TrueActivationPatching"
    assert sidecar["semantic_status"] == sidecar["validator_status"] == "failed"
    assert sidecar["reason"] == "failure_envelope_stage_mismatch"
    assert sidecar["raw_retention_status"] == "deleted_by_owner_exception"
    assert sidecar["standard_finalize"] is False
    assert sidecar["owner_exception"] == {
        "deletion_verification": {
            "absent_after_delete": True,
            "pre_delete_bytes": 154600,
            "pre_delete_sha256": "137bb19ceb983a286ce178553d581a7ba5b37f68bc4f58cecb56c1ed93fcdb3e",
        },
        "previous_sidecar_sha256": "d3c283912be5ee6e5b3d2fce6b0452b7e1e11e08a5078855ab8357981e01b920",
        "reason": (
            "Historical raw contains an invalid embedded failed+complete execution triad; "
            "standard finalization is fail-closed and cannot process it."
        ),
        "standard_finalize": False,
    }
    assert sidecar["archive"] == {
        "bytes": 19697,
        "sha256": "379ae82f5f71ce7bce457187534cc4da3f8d3689b293b7b1b8caf7110e4f3385",
        "source": "embedded in raw capture",
    }
    assert sidecar["bundle_members"] == [
        {
            "bytes": 13067,
            "path": "artifacts/m14/l04-explanations.TrueActivationPatching.attempt1.failure.json",
            "sha256": "b42b50a08f1dba7a01dbf7d843f6c8343ccad5a3180288a9296c59b3e3b1357b",
        },
        {
            "bytes": 102114,
            "path": "artifacts/m14/l04-explanations.TrueActivationPatching.attempt1.partial.json",
            "sha256": "f7fd29470850c397da02f7d6518e90f1c152cea76ac9c50f742376db0e11a4d5",
        },
        {
            "bytes": 6476,
            "path": "artifacts/m14/l04-explanations.TrueActivationPatching.attempt1.run.json",
            "sha256": "673d6a01af94c5b99aa46a92353f4b79155522278a175350beb249028c21b241",
        },
    ]
    assert sidecar["raw_capture"] == {
        "bytes": 154600,
        "path": (
            "artifacts/m14/l04-explanations.ssh.TrueActivationPatching.3d5d6720ee5fe155ff4a1f1c25814225b66170f3.raw.txt"
        ),
        "sha256": "137bb19ceb983a286ce178553d581a7ba5b37f68bc4f58cecb56c1ed93fcdb3e",
    }
    assert sidecar["raw_markers"] == {
        "bundle_status": 0,
        "cli_status": 1,
        "decode_match": "PASS",
        "decode_status": 0,
        "final_status": 1,
        "remote_cleanup": "PASS",
        "transport_cleanup": "PASS",
    }
    assert sidecar["validator_errors"] == ["failure: ['failed execution stage is not a truthful partial stage']"]
    assert sidecar["repository_promotion"] is False
    serialized = SIDECAR_PATH.read_text(encoding="utf-8").lower()
    assert not any(token in serialized for token in ("prompt", "tensor", "stdout", "stderr", "/tmp/", "192.168."))


def _completed_activation_execution() -> dict[str, Any]:
    execution = copy.deepcopy(
        next(
            item
            for item in _accepted_activation_artifact()["executions"]
            if item["use_case"] == "TrueActivationPatching"
        )
    )
    execution["confidence_intervals"] = {
        str(item["seed"]): item["recovery"]["confidence_interval_95"] for item in execution["raw_summaries"]
    }
    return execution


def test_completed_activation_success_remains_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts import _m14_l04_activation_patching as activation_module

    execution = _completed_activation_execution()
    monkeypatch.setattr(activation_module, "run_true_activation_patching", lambda _plan, _rows: execution)
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    result = run_real(
        plan_path=PLAN_PATH,
        fixture_path=FIXTURE_PATH,
        use_case="TrueActivationPatching",
        output_dir=tmp_path,
    )

    active = next(item for item in result["artifact"]["executions"] if item["use_case"] == "TrueActivationPatching")
    assert result["status"] == "passed_real_cuda"
    assert active["resources"]["stage"] == "complete"
    assert active["provenance"]["stage"] == "complete"
    assert result["run_record"]["stage"] == "complete"
    assert result["failure"]["stage"] == "complete"
    assert validate_artifact(result["artifact"], load_plan(PLAN_PATH)) == []


@pytest.mark.parametrize("stage", ["preflight", "cuda_check", "model_load"])
def test_activation_early_failure_stage_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    from scripts import _m14_l04_activation_patching as activation_module
    from scripts._m14_l04_tcav_runtime import RealExecutionError

    execution_attempted = stage != "preflight"
    resources = {
        "stage": stage,
        "execution_attempted": execution_attempted,
        "execution_backend": "cuda" if execution_attempted else "none",
        "network": "enabled" if execution_attempted else "not attempted",
        "cleanup": "pending",
    }

    def fail(_plan: dict[str, Any], _rows: list[dict[str, Any]]) -> dict[str, Any]:
        raise RealExecutionError("injected early failure", resources)

    monkeypatch.setattr(activation_module, "run_true_activation_patching", fail)
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    result = run_real(
        plan_path=PLAN_PATH,
        fixture_path=FIXTURE_PATH,
        use_case="TrueActivationPatching",
        output_dir=tmp_path,
    )

    assert result["status"] == "failed"
    assert result["artifact"]["provenance"]["stage"] == stage
    assert result["run_record"]["stage"] == stage
    assert result["failure"]["stage"] == stage


def test_malformed_completed_activation_failure_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import _m14_l04_activation_patching as activation_module

    execution = _completed_activation_execution()
    execution.update({"status": "failed", "evidence_eligible": False, "acceptance": False, "evidence_level": "D0"})
    execution["controls"]["true_interchange"]["pass"] = False
    execution["resources"].update(
        {"stage": "complete", "cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied"}
    )
    execution["provenance"].update(
        {"stage": "complete", "cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied"}
    )
    execution["raw_summaries"] = [{}]
    execution["fixture_linkage"] = [{}]
    execution["resources"]["cleanup"] = "synchronized cleared emptied"
    execution["provenance"]["cleanup"] = "synchronized cleared emptied"

    monkeypatch.setattr(activation_module, "run_true_activation_patching", lambda _plan, _rows: execution)
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    result = run_real(
        plan_path=PLAN_PATH,
        fixture_path=FIXTURE_PATH,
        use_case="TrueActivationPatching",
        output_dir=tmp_path,
    )

    assert result["artifact"]["provenance"]["stage"] == "complete"
    assert result["run_record"]["stage"] == "complete"
    errors = validate_failure(result["failure"], load_plan(PLAN_PATH), result["artifact"])
    assert any("failed execution stage is not a truthful partial stage" in error for error in errors)


@pytest.mark.parametrize("field", ["raw_summaries", "fixture_linkage"])
def test_malformed_activation_list_entries_remain_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    from scripts import _m14_l04_activation_patching as activation_module

    execution = _completed_activation_execution()
    execution.update({"status": "failed", "evidence_eligible": False, "acceptance": False, "evidence_level": "D0"})
    execution["controls"]["true_interchange"]["pass"] = False
    execution["resources"].update(
        {"stage": "complete", "cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied"}
    )
    execution["provenance"].update(
        {"stage": "complete", "cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied"}
    )
    execution[field] = [None]

    monkeypatch.setattr(activation_module, "run_true_activation_patching", lambda _plan, _rows: execution)
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    result = run_real(
        plan_path=PLAN_PATH,
        fixture_path=FIXTURE_PATH,
        use_case="TrueActivationPatching",
        output_dir=tmp_path,
    )

    assert result["artifact"]["provenance"]["stage"] == "complete"
    assert result["run_record"]["stage"] == "complete"


def test_failed_complete_with_all_recomputed_gates_passing_is_not_normalized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import _m14_l04_activation_patching as activation_module

    execution = _completed_activation_execution()
    execution.update({"status": "failed", "evidence_eligible": False, "acceptance": False, "evidence_level": "D0"})
    execution["resources"].update(
        {"stage": "complete", "cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied"}
    )
    execution["provenance"].update(
        {"stage": "complete", "cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied"}
    )

    monkeypatch.setattr(activation_module, "run_true_activation_patching", lambda _plan, _rows: execution)
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    result = run_real(
        plan_path=PLAN_PATH,
        fixture_path=FIXTURE_PATH,
        use_case="TrueActivationPatching",
        output_dir=tmp_path,
    )

    assert result["artifact"]["provenance"]["stage"] == "complete"
    assert result["run_record"]["stage"] == "complete"
    errors = validate_failure(result["failure"], load_plan(PLAN_PATH), result["artifact"])
    assert any("failed execution stage is not a truthful partial stage" in error for error in errors)


def test_declared_recovery_forgery_cannot_trigger_normalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import _m14_l04_activation_patching as activation_module

    execution = _completed_activation_execution()
    execution.update({"status": "failed", "evidence_eligible": False, "acceptance": False, "evidence_level": "D0"})
    for summary in execution["raw_summaries"]:
        for evidence in summary["holdout_evidence"]:
            evidence["recovery"] = 0.0
        summary["recovery"] = _metric(
            [0.0, 0.0, 0.0, 0.0],
            seed=summary["seed"],
            threshold=RECOVERY_CI_LOWER_THRESHOLD,
            comparator=">",
            units="normalized causal recovery",
        )
    execution["metrics"]["recovery"] = {
        str(summary["seed"]): summary["recovery"] for summary in execution["raw_summaries"]
    }
    execution["confidence_intervals"] = {
        str(summary["seed"]): summary["recovery"]["confidence_interval_95"] for summary in execution["raw_summaries"]
    }
    execution["resources"].update(
        {"stage": "complete", "cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied"}
    )
    execution["provenance"].update(
        {"stage": "complete", "cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied"}
    )

    monkeypatch.setattr(activation_module, "run_true_activation_patching", lambda _plan, _rows: execution)
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    result = run_real(
        plan_path=PLAN_PATH,
        fixture_path=FIXTURE_PATH,
        use_case="TrueActivationPatching",
        output_dir=tmp_path,
    )

    assert result["artifact"]["provenance"]["stage"] == "complete"
    assert result["run_record"]["stage"] == "complete"
    errors = validate_failure(result["failure"], load_plan(PLAN_PATH), result["artifact"])
    assert any("failed execution stage is not a truthful partial stage" in error for error in errors)


def test_completed_activation_gate_failure_is_cleanup_stage_d0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts import _m14_l04_activation_patching as activation_module

    execution = _completed_activation_execution()
    execution.update(
        {
            "status": "failed",
            "evidence_eligible": False,
            "acceptance": False,
            "evidence_level": "D0",
            "failure_reason": "one or more true activation patching gates failed",
        }
    )
    execution["controls"]["true_interchange"]["pass"] = False
    for summary in execution["raw_summaries"]:
        for evidence in summary["holdout_evidence"]:
            evidence["true_interchange_margin"] = evidence["corrupted_margin"]
            evidence["recovery"] = 0.0
        summary["recovery"] = _metric(
            [0.0, 0.0, 0.0, 0.0],
            seed=summary["seed"],
            threshold=RECOVERY_CI_LOWER_THRESHOLD,
            comparator=">",
            units="normalized causal recovery",
        )
    execution["metrics"]["recovery"] = {
        str(summary["seed"]): summary["recovery"] for summary in execution["raw_summaries"]
    }
    execution["confidence_intervals"] = {
        str(summary["seed"]): summary["recovery"]["confidence_interval_95"] for summary in execution["raw_summaries"]
    }
    execution["resources"].update(
        {
            "stage": "complete",
            "cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied",
        }
    )
    execution["provenance"].update(
        {"stage": "complete", "cleanup": "CUDA synchronized; model gradients cleared; CUDA cache emptied"}
    )

    monkeypatch.setattr(activation_module, "run_true_activation_patching", lambda _plan, _rows: execution)
    monkeypatch.setenv("LATENT_ANYTHING_RUN_NETWORK", "1")
    result = run_real(
        plan_path=PLAN_PATH,
        fixture_path=FIXTURE_PATH,
        use_case="TrueActivationPatching",
        output_dir=tmp_path,
    )

    active = next(item for item in result["artifact"]["executions"] if item["use_case"] == "TrueActivationPatching")
    assert result["status"] == "failed"
    assert result["artifact"]["evidence_level"] == "D0"
    assert result["artifact"]["provenance"]["stage"] == "cleanup"
    assert active["resources"]["stage"] == "cleanup"
    assert active["provenance"]["stage"] == "cleanup"
    assert result["run_record"]["stage"] == "cleanup"
    assert result["failure"]["stage"] == "cleanup"
    assert validate_artifact(result["artifact"], load_plan(PLAN_PATH)) == []
    assert validate_run_record(result["run_record"], result["artifact"], load_plan(PLAN_PATH)) == []
    assert validate_failure(result["failure"], load_plan(PLAN_PATH), result["artifact"]) == []


def _accepted_activation_artifact() -> dict[str, Any]:
    plan = load_plan(PLAN_PATH)
    raw, rows = read_fixture(FIXTURE_PATH)
    pairs = _pairs(rows)
    holdout_pairs = [pair for pair, value in pairs.items() if value["clean"]["split"] == "holdout"]
    evidence_template = [
        {
            "pair_id": pair,
            "group_id": str(pairs[pair]["clean"]["group_id"]),
            "split": "holdout",
            "clean_row_id": str(pairs[pair]["clean"]["row_id"]),
            "corrupted_row_id": str(pairs[pair]["corrupted"]["row_id"]),
            "clean_condition": "clean",
            "corrupted_condition": "corrupted",
            "clean_target_position": 2,
            "corrupted_target_position": 2,
            "clean_previous_valid_position": 1,
            "corrupted_previous_valid_position": 1,
            "clean_margin": 2.0,
            "corrupted_margin": 1.0,
            "true_interchange_margin": 2.0,
            "off_target_layer_margin": 1.0,
            "off_target_token_margin": 1.0,
            "shuffled_donor_margin": 1.0,
            "zero_strength_margin": 1.0,
            "recovery": 1.0,
            "off_target_layer_effect": 0.0,
            "off_target_token_effect": 0.0,
            "shuffled_donor_effect": 0.0,
            "zero_strength_error": 0.0,
            "strength_grid": {"0.0": 1.0, "0.25": 1.25, "0.5": 1.5, "1.0": 2.0},
        }
        for pair in holdout_pairs
    ]
    summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        mapping = deterministic_split_donor_derangement(pairs, seed)
        summaries.append(
            {
                "seed": seed,
                "train_pairs": [pair for pair, value in pairs.items() if value["clean"]["split"] == "train"],
                "holdout_pairs": holdout_pairs,
                "holdout_evidence": evidence_template,
                "recovery": _metric(
                    [1.0, 1.0, 1.0, 1.0],
                    seed=seed,
                    threshold=RECOVERY_CI_LOWER_THRESHOLD,
                    comparator=">",
                    units="normalized causal recovery",
                ),
                "off_target": _metric(
                    [0.0, 0.0, 0.0, 0.0],
                    seed=seed,
                    threshold=0.1,
                    comparator="<=",
                    units="absolute logit margin effect",
                    statistic="max",
                ),
                "off_target_layer": _metric(
                    [0.0, 0.0, 0.0, 0.0],
                    seed=seed,
                    threshold=0.1,
                    comparator="<=",
                    units="absolute logit margin effect",
                    statistic="max",
                ),
                "off_target_token": _metric(
                    [0.0, 0.0, 0.0, 0.0],
                    seed=seed,
                    threshold=0.1,
                    comparator="<=",
                    units="absolute logit margin effect",
                    statistic="max",
                ),
                "zero_strength": _metric(
                    [0.0, 0.0, 0.0, 0.0],
                    seed=seed,
                    threshold=1e-6,
                    comparator="<=",
                    units="absolute logit margin difference",
                    statistic="max",
                ),
                "shuffled_direction": {
                    "semantic": "shuffled donor activation; compatibility key retained",
                    "mapping": mapping,
                    "mapping_sha256": donor_mapping_digest(mapping),
                    "finite": True,
                },
                "finite": True,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    digest = "a" * 64
    resource_peak = {
        "cuda_device": "fake-cuda",
        "elapsed_seconds": 1.0,
        "max_memory_allocated_bytes": 1,
        "max_memory_reserved_bytes": 1,
        "max_rss_bytes": 1,
        "rss_source": "fake",
        "rss_unit": "bytes",
    }
    execution: dict[str, Any] = {
        "status": "passed_real_cuda",
        "evidence_eligible": True,
        "acceptance": True,
        "evidence_level": "D3",
        "metrics": {
            "recovery": {str(seed): summary["recovery"] for seed, summary in zip(SEEDS, summaries, strict=True)}
        },
        "controls": {
            "clean_endpoint": {"pass": True},
            "corrupted_endpoint": {"pass": True},
            "true_interchange": {"pass": True},
            "off_target_layer": {
                "pass": True,
                "metrics": {
                    str(seed): summary["off_target_layer"] for seed, summary in zip(SEEDS, summaries, strict=True)
                },
            },
            "off_target_token": {
                "pass": True,
                "metrics": {
                    str(seed): summary["off_target_token"] for seed, summary in zip(SEEDS, summaries, strict=True)
                },
            },
            "off_target_combined": {
                "pass": True,
                "metrics": {str(seed): summary["off_target"] for seed, summary in zip(SEEDS, summaries, strict=True)},
            },
            "shuffled_direction": {"pass": True, "semantic": "shuffled donor activation; compatibility key retained"},
            "zero_strength": {"pass": True},
        },
        "raw_summaries": summaries,
        "fixture_linkage": _expected_linkage(rows),
        "token_ids": dict(TARGET_TOKEN_IDS),
        "target_token_strings": dict(TARGET_TOKEN_STRINGS),
        "layer": LAYER,
        "native_hidden_state_index": NATIVE_HIDDEN_STATE_INDEX,
        "seed": SEEDS[0],
        "seeds": list(SEEDS),
        "no_mutation": True,
        "model_parameter_digest_before": digest,
        "model_parameter_digest_after": digest,
        "budget_pass": True,
        "provenance": {
            "runtime": "real TransformerLMIntegration",
            "model_revision": plan["model"]["revision"],
            "integration": "TransformerLMIntegration",
            "adapter": "N/A",
            "target_token_ids": dict(TARGET_TOKEN_IDS),
            "target_token_strings": dict(TARGET_TOKEN_STRINGS),
            "target_position": "last non-padding token",
            "donor_semantics": "clean hidden activation replaces corrupted hidden activation",
            "off_target_controls": {"layer": OFF_TARGET_LAYER, "token": "previous valid token"},
            "strength_grid": [0.0, 0.25, 0.5, 1.0],
            "shuffled_direction_semantics": "shuffled donor activation; compatibility key retained",
            "network": "enabled",
            "device": "fake-cuda",
            "execution_attempted": True,
            "execution_backend": "cuda",
            "stage": "complete",
            "deterministic_algorithms": True,
            "resource_peak": resource_peak,
            "model_parameter_digest_before": digest,
            "model_parameter_digest_after": digest,
            "budget_pass": True,
        },
        "resources": {
            "execution_attempted": True,
            "execution_backend": "cuda",
            "network": "enabled",
            "stage": "complete",
            "device": "fake-cuda",
            "resource_peak": resource_peak,
        },
    }
    return build_artifact(
        plan,
        fixture_metadata(plan, raw, rows),
        "TrueActivationPatching",
        "passed_real_cuda",
        "failure.json",
        execution_result=execution,
        resources=execution["resources"],
    )


def test_accepted_activation_artifact_baseline_validates() -> None:
    artifact = _accepted_activation_artifact()
    active = next(item for item in artifact["executions"] if item["use_case"] == "TrueActivationPatching")
    assert validate_real_true_activation_patching_execution(active, artifact, load_plan(PLAN_PATH)) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda artifact: artifact["executions"][-2]["provenance"]["resource_peak"].update(
                {"elapsed_seconds": 9999.0}
            ),
            "budget",
        ),
        (
            lambda artifact: artifact["executions"][-2].update({"model_parameter_digest_before": "bad"}),
            "parameter digests",
        ),
        (
            lambda artifact: artifact["executions"][-2].pop("model_parameter_digest_after"),
            "parameter digests",
        ),
        (
            lambda artifact: artifact["executions"][-2]["provenance"].update(
                {"model_parameter_digest_after": "b" * 64}
            ),
            "parameter digests",
        ),
        (
            lambda artifact: artifact["raw_summaries"].__setitem__(0, {**artifact["raw_summaries"][0], "seed": 29}),
            "seeds",
        ),
        (
            lambda artifact: artifact["raw_summaries"].reverse(),
            "seeds",
        ),
        (lambda artifact: artifact["raw_summaries"][0].update({"finite": False}), "finite flag"),
        (
            lambda artifact: artifact["raw_summaries"][0]["holdout_evidence"][0].update({"clean_margin": float("nan")}),
            "non-finite metrics",
        ),
        (
            lambda artifact: artifact["raw_summaries"][0]["holdout_evidence"][0].update({"zero_strength_error": 1.0}),
            "zero-strength error",
        ),
        (
            lambda artifact: artifact["raw_summaries"][0]["holdout_evidence"][0]["strength_grid"].update({"0.0": 9.0}),
            "strength endpoints",
        ),
        (
            lambda artifact: artifact["fixture_linkage"].__setitem__(
                0, {**artifact["fixture_linkage"][0], "group_id": "tampered"}
            ),
            "row linkage",
        ),
        (lambda artifact: artifact["fixture"].update({"content_sha256": "f" * 64}), "fixture digests"),
        (
            lambda artifact: artifact["executions"][-2]["controls"]["true_interchange"].update({"pass": False}),
            "pass flag",
        ),
    ],
)
def test_accepted_activation_artifact_tampering_is_rejected(mutation: Any, message: str) -> None:
    artifact = _accepted_activation_artifact()
    mutation(artifact)
    active = next(item for item in artifact["executions"] if item["use_case"] == "TrueActivationPatching")
    errors = validate_real_true_activation_patching_execution(active, artifact, load_plan(PLAN_PATH))
    assert any(message in error for error in errors)


def test_offline_dispatch_keeps_activation_unpromoted(tmp_path: Path) -> None:
    plan = load_plan(PLAN_PATH)
    result = run_real(
        plan_path=PLAN_PATH, fixture_path=FIXTURE_PATH, use_case="TrueActivationPatching", output_dir=tmp_path
    )
    assert result["status"] == "not_implemented_pending_L04.9"
    assert validate_artifact(result["artifact"], plan) == []
    assert validate_run_record(result["run_record"], result["artifact"], plan) == []
    assert validate_failure(result["failure"], plan, result["artifact"]) == []
