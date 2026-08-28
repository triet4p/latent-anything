"""Offline contract tests for the frozen M14 L04 explanation lane."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.m14_l04_contract import (
    FIXTURE_PATH,
    PLAN_PATH,
    ContractValidationError,
    load_and_validate,
    plan_digest,
    validate_plan,
    validate_target_tokens,
)


def test_frozen_plan_and_fixture_have_independent_canonical_digests() -> None:
    result = load_and_validate()

    assert result["plan_sha256"] == json.loads(PLAN_PATH.read_text(encoding="utf-8"))["plan_sha256"]
    assert result["content_sha256"] == "f5c66f6d947c23f25d41e6aaf8982481feabc92bbff600bd929d27772fb62c0f"
    assert result["split_sha256"] == "7d788c18212bb1d7e345528c68af6f2bf3e0f745ca77e2d115d74ac3e964121b"
    assert result["pair_sha256"] == "7225e73c1238b23f6521718c8401331e59653a90499f4b2d75f32dddfe6c1c9c"


def test_contract_facade_reexports_each_private_responsibility() -> None:
    import scripts._m14_l04_contract_common as common
    import scripts._m14_l04_fixture_contract as fixture_contract
    import scripts._m14_l04_plan_contract as plan_contract
    import scripts._m14_l04_token_contract as token_contract
    import scripts.m14_l04_contract as facade

    assert facade.ContractValidationError is common.ContractValidationError
    assert facade.plan_digest is plan_contract.plan_digest
    assert facade.validate_plan is plan_contract.validate_plan
    assert facade.validate_fixture is fixture_contract.validate_fixture
    assert facade.fixture_digests is fixture_contract.fixture_digests
    assert facade.validate_target_tokens is token_contract.validate_target_tokens
    assert facade.PLAN_PATH == plan_contract.PLAN_PATH
    assert facade.FIXTURE_PATH == fixture_contract.FIXTURE_PATH


def test_plan_digest_does_not_mutate_plan() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    before = plan["plan_sha256"]

    assert plan_digest(plan) == before
    assert plan["plan_sha256"] == before


def test_tcav_plan_keeps_authoritative_gap_id_for_record_linkage() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert plan["record_order"][0]["record_id"] == "THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018"
    assert plan["real_use_case_checklist"][1]["record_id"] == "THY-T05-CONCEPT-ACTIVATION-VECTORS-TCAV-KIM-ET-AL-2018"


def test_malformed_plan_schema_fails_closed(tmp_path: Path) -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    del plan["remote_cuda_workflow"]
    plan_path = tmp_path / "bad-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ContractValidationError, match="plan schema mismatch"):
        load_and_validate(plan_path=plan_path)


def test_record_order_semantic_mutation_fails_after_digest_recompute() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan["record_order"][0], plan["record_order"][1] = plan["record_order"][1], plan["record_order"][0]
    plan["plan_sha256"] = plan_digest(plan)

    errors = validate_plan(plan)

    assert any("record_order" in error for error in errors)


def test_frozen_threshold_semantic_mutation_fails_after_digest_recompute() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan["thresholds_and_controls"]["tcav"]["heldout_accuracy_min"] = 0.61
    plan["plan_sha256"] = plan_digest(plan)

    errors = validate_plan(plan)

    assert any("heldout_accuracy_min is not frozen" in error for error in errors)


def test_remote_protocol_marker_semantic_mutation_fails_after_digest_recompute() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    commands = plan["remote_cuda_workflow"]["powershell_commands"]
    plan["remote_cuda_workflow"]["powershell_commands"] = [
        command.replace("base64", "encoded", 1) for command in commands
    ]
    plan["plan_sha256"] = plan_digest(plan)

    errors = validate_plan(plan)

    assert any("required marker 'base64'" in error for error in errors)


def test_duplicate_row_fails_closed(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]
    rows[-1]["row_id"] = rows[0]["row_id"]
    fixture = tmp_path / "duplicate.jsonl"
    fixture.write_bytes("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8"))

    with pytest.raises(ContractValidationError, match="row_id values must be unique"):
        load_and_validate(fixture_path=fixture)


def test_pair_with_two_clean_rows_fails_closed(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]
    rows[1]["condition"] = "clean"
    fixture = tmp_path / "bad-pair.jsonl"
    fixture.write_bytes("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8"))

    with pytest.raises(ContractValidationError, match="exactly one clean and one corrupted"):
        load_and_validate(fixture_path=fixture)


def test_split_leak_fails_closed(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]
    rows[0]["split"] = "holdout"
    fixture = tmp_path / "leak.jsonl"
    fixture.write_bytes("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8"))

    with pytest.raises(ContractValidationError, match="group .* crosses split"):
        load_and_validate(fixture_path=fixture)


def test_target_mismatch_fails_closed(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]
    rows[0]["target_text"] = " false"
    fixture = tmp_path / "target-mismatch.jsonl"
    fixture.write_bytes("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8"))

    with pytest.raises(ContractValidationError, match="condition/target mismatch"):
        load_and_validate(fixture_path=fixture)


def test_target_token_validation_is_dependency_injected() -> None:
    calls: list[str] = []

    def fake_tokenizer(text: str) -> dict[str, list[int]]:
        calls.append(text)
        return {"input_ids": [101]}

    validate_target_tokens(fake_tokenizer, (" true", " false"))

    assert calls == [" true", " false"]


def test_target_token_validation_rejects_non_single_token_fake() -> None:
    with pytest.raises(ContractValidationError, match="resolves to 2 tokens"):
        validate_target_tokens(lambda _text: {"input_ids": [1, 2]}, (" true",))


def test_target_token_validation_rejects_tokenizer_failure() -> None:
    def broken_tokenizer(_text: str) -> object:
        raise RuntimeError("network should not be needed")

    with pytest.raises(ContractValidationError, match="tokenizer failed"):
        validate_target_tokens(broken_tokenizer, (" true",))


def test_offline_cli_import_isolation_excludes_model_and_network_modules() -> None:
    code = """
import sys
sys.argv = ['scripts/m14_l04_explanations.py', '--check']
from scripts.m14_l04_explanations import main
main()
for name in ('transformers', 'huggingface_hub', 'requests', 'httpx', 'aiohttp'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"plan_sha256"' in result.stdout
