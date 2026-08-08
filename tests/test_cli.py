"""Offline tests for the Sprint 62 CLI surface."""

from __future__ import annotations

import json
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


def test_replay_command_materializes_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recorder = FileSystemRunRecorder(tmp_path)
    record = recorder.start("fixture", config={"seed": 4}, seeds=(4,))
    output = tmp_path / "replay.json"

    assert main(["replay-run", record.run_id, "--record-root", str(tmp_path), "--output", str(output)]) == 0

    assert json.loads(output.read_text(encoding="utf-8"))["config"] == {"seed": 4}
    assert "Wrote" in capsys.readouterr().out
