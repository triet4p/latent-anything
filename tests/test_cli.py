"""Offline tests for the Sprint 62 CLI surface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from latent_anything.cli import main
from latent_anything.run_record import FileSystemRunRecorder


def test_capture_points_command_lists_supported_seams(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["capture-points", "--policy", "smolvla"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in payload["capture_points"]} == {
        "vision_context",
        "language_context",
        "state_context",
        "action_expert",
    }


def test_policy_inspection_command_is_lazy_and_serializable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect-policy", "--policy", "act"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint"]["policy_repo_id"] == "lerobot/act_aloha_sim_insertion_human"


def test_cli_import_isolation_does_not_eagerly_load_lerobot() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import latent_anything.cli; "
                "assert not any(name == 'lerobot' or name.startswith('lerobot.') "
                "for name in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""


def test_inspect_dataset_missing_lerobot_has_actionable_extra() -> None:
    code = """
import builtins
from latent_anything.cli import main

original_import = builtins.__import__

def block_lerobot(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "lerobot" or name.startswith("lerobot."):
        error = ModuleNotFoundError("blocked optional backend")
        error.name = "lerobot"
        raise error
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_lerobot
try:
    main(["inspect-dataset", "fixture/repo"])
except ImportError as error:
    assert str(error) == (
        "Optional backend 'lerobot' is unavailable. "
        "Install with: uv sync --extra lerobot"
    )
else:
    raise AssertionError("blocked LeRobot import unexpectedly succeeded")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_replay_command_materializes_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recorder = FileSystemRunRecorder(tmp_path)
    record = recorder.start("fixture", config={"seed": 4}, seeds=(4,))
    output = tmp_path / "replay.json"

    assert main(["replay-run", record.run_id, "--record-root", str(tmp_path), "--output", str(output)]) == 0

    assert json.loads(output.read_text(encoding="utf-8"))["config"] == {"seed": 4}
    assert "Wrote" in capsys.readouterr().out
