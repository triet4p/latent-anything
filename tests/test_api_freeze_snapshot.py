"""Strict drift checks for the reviewed API-freeze inventory."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "artifacts" / "api_freeze_snapshot_0.1.0b1.json"


def test_api_freeze_snapshot_has_no_unreviewed_drift() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/api_freeze_snapshot.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_api_freeze_snapshot_separates_canonical_and_beta_surfaces() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    public = snapshot["sections"]["A_public_surface"]
    aliases = snapshot["sections"]["B_beta_compatibility"]["symbol_aliases"]
    assert public["current_count"] == 205
    assert public["canonical_stable_count"] == 202
    stable_names = {item["name"] for item in public["canonical_stable_surface"]}
    assert not stable_names.intersection({"Method", "BMethod", "ManipulationPipeline"})
    assert {item["canonical"] for item in aliases} == {
        "AnalysisMethod",
        "Intervention",
        "InterventionPipeline",
    }
    assert all(item["deadline"] == "0.9.0" for item in aliases)
    assert snapshot["sections"]["D_registry"]["count"] == 32
    assert snapshot["sections"]["E_plugin_groups"]["count"] == 5
    assert snapshot["sections"]["F_optional_profiles"]["count"] == 12
    assert snapshot["sections"]["G_config_schemas"]["count"] == 28
    assert snapshot["sections"]["K_sync_async"]["count"] == 9
    assert set(snapshot["section_digests"]) == set(snapshot["sections"])


def test_api_freeze_snapshot_records_runtime_observations() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    aliases = snapshot["sections"]["B_beta_compatibility"]
    assert all(item["observed_identity"] for item in aliases["symbol_aliases"])
    assert all(item["observed_same_parser"] for item in aliases["cli_aliases"])
    assert all(item["observed_warning"] for item in aliases["registry_kind_aliases"])
    assert aliases["config_aliases"][0]["observed_values"] == [2.0, 2.0]
    assert all(item["observed_presence"] for item in aliases["result_property_aliases"])

    cli = snapshot["sections"]["I_cli"]
    assert cli["parser_malformed_exit"] == 2
    assert cli["main_success_exit"] == 0
    assert {item["name"] for item in cli["commands"]} == {
        "capture-points",
        "inspect-dataset",
        "inspect-policy",
        "replay-run",
        "compare-runs",
    }
    serialization = snapshot["sections"]["J_serialization"]
    assert serialization["disk_cache"]["version"] == "disk-cache-v1"
    assert (
        serialization["result_envelope"]["fixture_digest"] == serialization["result_envelope"]["golden_fixture_digest"]
    )
    assert len(serialization["result_envelope"]["observed_fixture_digest"]) == 64
    assert all(
        item["async_is_coroutine"] or item["async_is_async_generator"]
        for item in snapshot["sections"]["K_sync_async"]["pairs"]
    )
    assert snapshot["sections"]["L_exceptions"]["count"] == len(snapshot["sections"]["L_exceptions"]["entries"])


def test_api_freeze_comparator_reports_nested_runtime_drift_without_writing() -> None:
    from scripts.api_freeze_snapshot import compare_snapshot

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = copy.deepcopy(expected)
    sections = cast(dict[str, object], current["sections"])
    cli = cast(dict[str, object], sections["I_cli"])
    commands = cast(list[dict[str, object]], cli["commands"])
    commands[0]["name"] = "unexpected-command"
    serialization = cast(dict[str, object], sections["J_serialization"])
    disk_cache = cast(dict[str, object], serialization["disk_cache"])
    disk_cache["version"] = "wrong-version"
    differences = compare_snapshot(expected, current)
    assert "snapshot.sections.I_cli.commands[0].name" in differences
    assert "snapshot.sections.J_serialization.disk_cache.version" in differences


def test_api_freeze_runtime_helpers_are_live_contract_sources() -> None:
    from latent_anything._api_freeze_runtime import aliases, async_pairs, cli_contract, exceptions, serialization

    cli = cast(dict[str, object], cli_contract())
    serialized = cast(dict[str, object], serialization())
    alias_rows = cast(list[dict[str, object]], cast(dict[str, object], aliases())["symbol_aliases"])
    assert cli["parser_malformed_exit"] == 2
    assert cast(dict[str, object], serialized["disk_cache"])["version"] == "disk-cache-v1"
    assert alias_rows[1]["observed_identity"] is True
    assert async_pairs()["count"] == 9
    assert exceptions()["count"] == 7


def test_api_freeze_cli_runtime_failure_keeps_domain_error_contract(tmp_path: Path) -> None:
    from latent_anything.cli import main

    try:
        main(["compare-runs", "--record-root", str(tmp_path)])
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("compare-runs without records must fail with its domain ValueError")
