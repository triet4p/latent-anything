"""Offline tests for L04.3 dispatch and failure-preserving infrastructure."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts._m14_l04_boundary import INTEGRATION_FACTORY, transformer_integration_type
from scripts._m14_l04_digest import code_sha
from scripts._m14_l04_envelope import source_digests, validate_artifact, validate_failure, validate_run_record
from scripts.m14_l04_contract import FIXTURE_PATH, PLAN_PATH, load_plan
from scripts.m14_l04_explanations import PENDING, USE_CASES, run_real


@pytest.mark.parametrize("use_case", USE_CASES)
def test_each_use_case_is_dispatch_only_and_isolated(use_case: str, tmp_path: Path) -> None:
    result = run_real(use_case=use_case, output_dir=tmp_path)

    assert result["status"] == PENDING[use_case]
    assert result["artifact"]["evidence_level"] == "D0"
    assert result["artifact"]["accepted_record_ids"] == []
    assert len(result["artifact"]["executions"]) == 7
    assert result["artifact"]["integration"] == "TransformerLMIntegration"
    assert result["artifact"]["adapter"] == "N/A"
    assert result["paths"]["partial"].startswith(f"l04-explanations.{use_case}.attempt")
    assert result["paths"]["run"].endswith(".run.json")
    assert result["paths"]["failure"].endswith(".failure.json")
    assert not (tmp_path / "l04-explanations.json").exists()
    assert validate_artifact(result["artifact"], load_plan()) == []
    assert validate_run_record(result["run_record"], result["artifact"], load_plan()) == []
    assert validate_failure(result["failure"], load_plan(), result["artifact"]) == []
    assert (
        result["artifact"]["provenance"]["implementation_source_files"]
        == source_digests()["implementation_source_files"]
    )


def test_injected_handler_can_exercise_envelopes_but_never_promotes(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_handler(plan: dict[str, object], rows: list[dict[str, object]]) -> dict[str, object]:
        calls.append(str(plan["lane"]))
        assert rows
        return {"accepted": True, "metrics": {"pretend": 1.0}}

    result = run_real(use_case="TCAV", output_dir=tmp_path, handlers={"TCAV": fake_handler})

    assert calls == ["L04"]
    assert result["status"] == "injected_offline_non_eligible"
    assert result["artifact"]["provenance"]["evidence_origin"] == "dependency-injected-offline"
    assert result["artifact"]["accepted_record_ids"] == []
    assert result["artifact"]["evidence_level"] == "D0"
    assert all(not entry["evidence_eligible"] for entry in result["artifact"]["executions"])


def test_injected_handler_exception_is_retained_as_failed(tmp_path: Path) -> None:
    def broken_handler(_plan: dict[str, object], _rows: list[dict[str, object]]) -> dict[str, object]:
        raise ValueError("offline handler failure")

    result = run_real(use_case="TCAV", output_dir=tmp_path, handlers={"TCAV": broken_handler})

    assert result["status"] == "failed"
    assert result["failure"]["exception_type"] == "ValueError"
    assert result["failure"]["exception"] == "offline handler failure"
    assert validate_failure(result["failure"], load_plan(), result["artifact"]) == []


def test_non_json_injected_result_is_retained_as_failed(tmp_path: Path) -> None:
    def malformed_handler(_plan: dict[str, object], _rows: list[dict[str, object]]) -> dict[str, object]:
        return {"not_json": object()}

    result = run_real(use_case="TCAV", output_dir=tmp_path, handlers={"TCAV": malformed_handler})

    assert result["status"] == "failed"
    assert result["failure"]["exception_type"] == "TypeError"
    assert validate_failure(result["failure"], load_plan(), result["artifact"]) == []


def test_tampered_artifact_and_failure_linkage_fail_closed(tmp_path: Path) -> None:
    result = run_real(use_case="DirectLogitLens", output_dir=tmp_path)
    artifact = json.loads(json.dumps(result["artifact"]))
    artifact["accepted_record_ids"] = ["unexpected"]
    assert validate_artifact(artifact, load_plan())

    failure = json.loads(json.dumps(result["failure"]))
    failure["run_record"]["artifact_sha256"] = "0" * 64
    assert any("self-digest" in error for error in validate_failure(failure, load_plan(), result["artifact"]))

    remapped = json.loads(json.dumps(result["artifact"]))
    remapped["executions"][0]["record_id"] = "wrong"
    assert validate_artifact(remapped, load_plan())

    model_tampered = json.loads(json.dumps(result["artifact"]))
    model_tampered["provenance"]["model_id"] = "wrong"
    assert any("model identity" in error for error in validate_artifact(model_tampered, load_plan()))

    provenance = json.loads(json.dumps(result["artifact"]))
    provenance["provenance"]["implementation_source_files"]["future_module.py"] = "0" * 64
    assert any("aggregate" in error for error in validate_artifact(provenance, load_plan()))

    linkage = json.loads(json.dumps(result["failure"]))
    linkage["run_record"]["status"] = "failed"
    assert validate_failure(linkage, load_plan(), result["artifact"])


def test_run_and_failure_identity_status_model_tampering_fails_closed(tmp_path: Path) -> None:
    result = run_real(use_case="TCAV", output_dir=tmp_path)
    plan = load_plan()

    bad_run = json.loads(json.dumps(result["run_record"]))
    bad_run["schema_version"] = "wrong"
    bad_run["model"] = {"id": "wrong"}
    bad_run["status"] = "failed"
    errors = validate_run_record(bad_run, result["artifact"], plan)
    assert any("identity" in error for error in errors)
    assert any("model" in error for error in errors)
    assert any("status" in error for error in errors)

    bad_failure = json.loads(json.dumps(result["failure"]))
    bad_failure["lane"] = "wrong"
    bad_failure["status"] = "failed"
    bad_failure["model"] = {"id": "wrong"}
    errors = validate_failure(bad_failure, plan, result["artifact"])
    assert any("identity" in error for error in errors)
    assert any("model" in error for error in errors)
    assert any("status" in error for error in errors)


def test_repeated_invocations_get_distinct_use_case_files(tmp_path: Path) -> None:
    first = run_real(use_case="TCAV", output_dir=tmp_path)
    second = run_real(use_case="TCAV", output_dir=tmp_path)

    assert first["paths"]["partial"] != second["paths"]["partial"]
    assert (tmp_path / first["paths"]["partial"]).exists()
    assert len(list(tmp_path.glob("*.partial.json"))) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_dispatch_never_overwrites_finalizer_output(tmp_path: Path) -> None:
    final = tmp_path / "l04-explanations.json"
    final.write_text("finalizer-owned\n", encoding="utf-8")

    run_real(use_case="IntegratedGradients", output_dir=tmp_path)

    assert final.read_text(encoding="utf-8") == "finalizer-owned\n"


def test_tampered_fixture_fails_before_handler_or_output(tmp_path: Path) -> None:
    fixture = tmp_path / "tampered.jsonl"
    rows = [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]
    rows[0]["target_text"] = " false"
    fixture.write_bytes("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8"))
    calls: list[str] = []

    def handler(_plan: dict[str, object], _rows: list[dict[str, object]]) -> dict[str, object]:
        calls.append("called")
        return {}

    with pytest.raises(Exception, match="condition/target mismatch"):
        run_real(use_case="TCAV", fixture_path=fixture, output_dir=tmp_path, handlers={"TCAV": handler})

    assert calls == []
    assert list(tmp_path.glob("l04-explanations.*.json")) == []


def test_code_sha_fails_closed_without_zero_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_git(*_args: object, **_kwargs: object) -> object:
        raise OSError("git unavailable")

    monkeypatch.setattr("scripts._m14_l04_digest.subprocess.run", broken_git)

    with pytest.raises(RuntimeError, match="SHA is unavailable"):
        code_sha()


def test_check_cli_does_not_import_model_or_network_modules() -> None:
    code = """
import sys
sys.argv = ['scripts/m14_l04_explanations.py', '--check']
from scripts.m14_l04_explanations import main
main()
for name in ('torch', 'transformers', 'huggingface_hub', 'requests', 'httpx', 'aiohttp'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PLAN_PATH.parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "plan_sha256" in result.stdout


def test_private_real_factory_identity_is_concrete_transformer_integration() -> None:
    integration = transformer_integration_type()

    assert f"{integration.__module__}.{integration.__name__}" == INTEGRATION_FACTORY
