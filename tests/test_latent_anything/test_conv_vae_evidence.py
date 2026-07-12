"""Smoke the real-image ConvVAE evidence composition path."""

from __future__ import annotations

from scripts.conv_vae_digits_evidence import main


def test_digits_evidence_script_writes_metrics(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    main()
    assert (tmp_path / "artifacts" / "conv_vae_digits_evidence.json").is_file()
