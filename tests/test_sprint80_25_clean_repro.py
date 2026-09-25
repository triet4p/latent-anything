"""Focused behavior checks for the committed Sprint 80.25 clean replay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "sprint80_task80_25_clean_repro.py"
ENCODER_REFERENCE = Path("artifacts/diagnostics/proof-80-25-clean-encoder-v3-20260924-035357-6996")
TRANSFORMER_REFERENCE = Path("artifacts/diagnostics/proof-80-25-clean-transformer-v1-target-v2-20260924-035357-6996")


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("sprint80_task80_25_clean_repro", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repro = _load_script()


def _make_fake_source(root: Path) -> None:
    """Build a miniature project tree with frozen inputs and generated junk."""
    files = {
        "pyproject.toml": "[project]\n",
        "uv.lock": "[[package]]\n",
        ".python-version": "3.13\n",
        "README.md": "# read\n",
        "LICENSE": "MIT\n",
        "src/pkg/mod.py": "VALUE = 1\n",
        "src/pkg/__pycache__/mod.cpython-313.pyc": "bytecode",
        "src/latent_anything.egg-info/PKG-INFO": "Name: latent-anything\n",
        "scripts/proof.py": "main = 1\n",
        "scripts/__pycache__/proof.pyc": "bytecode",
        "scripts/.git/config": "hidden",
        "artifacts/manifest.json": "{}",
        "artifacts/m14/l04-wikitext-2-manifest.json": "{}",
        "tests/test_thing.py": "def test_x(): pass\n",
        "docs/plan.md": "not an input",
        "graphify-out/graph.json": "generated",
        "head_init_tmp.py": "stray",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_fresh_root_copies_declared_inputs_without_generated_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "clean"
    source.mkdir()
    destination.mkdir()
    _make_fake_source(source)

    copied = repro.copy_fresh_root(source, destination)
    copied_files = set(copied["files"])
    assert copied_files == {
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "README.md",
        "LICENSE",
        "src/pkg/mod.py",
        "scripts/proof.py",
        "artifacts/manifest.json",
        "artifacts/m14/l04-wikitext-2-manifest.json",
        "tests/test_thing.py",
    }
    for relative in copied_files:
        assert (destination / relative).read_bytes() == (source / relative).read_bytes()
    for absent in (
        "src/pkg/__pycache__",
        "src/latent_anything.egg-info",
        "scripts/.git",
        "docs",
        "graphify-out",
        "head_init_tmp.py",
    ):
        assert not (destination / absent).exists()
    with pytest.raises(AssertionError, match="not empty"):
        repro.copy_fresh_root(source, destination)


def test_pins_cover_accepted_cases_and_independent_stability_evidence() -> None:
    verified = repro.verify_pinned_inputs(REPO)
    required = {
        "artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v3_model_weight_lesion.json",
        "artifacts/benchmark_model_sprint80_encoder_autoencoder_v3_baseline.npz",
        "artifacts/benchmark_manifest_sprint80_transformer_hidden_state_v1.json",
        "artifacts/benchmark_manifest_sprint80_transformer_explanation_stability_v1.json",
        "scripts/sprint80_task80_23_v3_proof.py",
        "scripts/sprint80_task80_24_proof.py",
        "scripts/sprint80_task80_24_stability_proof.py",
        "artifacts/task_80.25_evidence_clean_20260924/summary.json",
        "artifacts/task_80.25_evidence_clean_20260924/commands.json",
        ENCODER_REFERENCE.joinpath("proof-run.json").as_posix(),
        TRANSFORMER_REFERENCE.joinpath("runs/c4012406e89254ea.json").as_posix(),
        TRANSFORMER_REFERENCE.joinpath("diagnostic-report").as_posix(),
        "artifacts/task_80.25_evidence/summary.json",
        "artifacts/task_80.25_evidence_attempt1_failed/summary.json",
        "src/latent_anything/**/*.py",
    }
    assert required <= verified.keys()
    assert verified == repro.verify_pinned_inputs(REPO)


def test_pin_mismatch_fails_with_the_exact_input_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = tmp_path / "uv.lock"
    pinned.write_text("original\n", encoding="utf-8")
    expected = hashlib.sha256(pinned.read_bytes()).hexdigest()
    package_source = tmp_path / "src" / "latent_anything"
    shutil.copytree(
        REPO / "src" / "latent_anything",
        package_source,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    monkeypatch.setattr(repro, "PINNED_INPUT_SHA256", {"uv.lock": expected})
    assert repro.verify_pinned_inputs(tmp_path) == {
        "uv.lock": expected,
        "src/latent_anything/**/*.py": repro.PINNED_SOURCE_TREE_SHA256,
    }

    pinned.write_text("changed\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="uv.lock"):
        repro.verify_pinned_inputs(tmp_path)


def test_seeded_hub_cache_contains_only_verified_pinned_snapshot_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "models--example--toy"
    revision = "a" * 40
    payload = b"immutable snapshot bytes"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        repro,
        "PINNED_HF_SNAPSHOT",
        {repo_id: {"revision": revision, "files": {"config.json": digest}}},
    )
    source_home = tmp_path / "external"
    fresh_home = tmp_path / "fresh"
    source_snapshot = source_home / "hub" / repo_id / "snapshots" / revision
    source_snapshot.mkdir(parents=True)
    (source_snapshot / "config.json").write_bytes(payload)
    (source_snapshot / "unlisted.bin").write_bytes(b"not declared")

    seeded = repro.seed_hf_snapshot(source_home, fresh_home)
    copied_snapshot = fresh_home / "hub" / repo_id / "snapshots" / revision
    assert seeded["file_count"] == 1
    assert (copied_snapshot / "config.json").read_bytes() == payload
    assert not (copied_snapshot / "unlisted.bin").exists()
    assert repro.verify_hf_snapshot(fresh_home)[repo_id]["revision"] == revision

    with pytest.raises(AssertionError, match="not empty"):
        repro.seed_hf_snapshot(source_home, fresh_home)


def test_child_environment_strips_developer_state_and_uses_per_proof_caches(tmp_path: Path) -> None:
    base = {
        "PATH": "/usr/bin",
        "PYTHONPATH": "F:/dev/src",
        "HF_TOKEN": "secret",
        "HF_HOME": "F:/dev/hf",
        "HF_HUB_CACHE": "F:/dev/hub",
        "TMP": "C:/dev/temp",
        "VIRTUAL_ENV": "F:/dev/.venv",
        "KEEP_ME": "kept",
    }
    work = tmp_path / "work"
    cache_home = work / "runtime" / "case-a" / "hf-home"
    env = repro.build_child_env(work, "case-a", cache_home, base=base)

    for stripped in ("PYTHONPATH", "HF_TOKEN", "HF_HOME", "VIRTUAL_ENV"):
        assert stripped not in env
    assert env["KEEP_ME"] == "kept"
    assert env["PATH"] == "/usr/bin"
    assert env["TMP"] != base["TMP"]
    assert env["PYTHONNOUSERSITE"] == "1"
    assert Path(env["TMP"]) == work / "runtime" / "case-a" / "tmp"
    assert env["TEMP"] == env["TMPDIR"] == env["TMP"]
    assert Path(env["HF_HUB_CACHE"]) == cache_home / "hub"
    assert Path(env["HF_DATASETS_CACHE"]).is_relative_to(work / "runtime" / "case-a" / "generated")
    assert Path(env["HF_MODULES_CACHE"]).is_relative_to(work / "runtime" / "case-a" / "generated")
    assert repro._cache_empty(repro._cache_state(work, "case-a"))
    (Path(env["HF_DATASETS_CACHE"]) / "data.arrow").write_bytes(b"generated")
    assert not repro._cache_empty(repro._cache_state(work, "case-a"))


def test_encoder_restoration_effect_and_comparison_delta_keep_distinct_signs(tmp_path: Path) -> None:
    reference = REPO / ENCODER_REFERENCE
    validated = repro._encoder_output(reference)
    assert validated["intervention_effect_restoration_increase"] == pytest.approx(0.4638888888888889)
    assert validated["comparison_candidate_minus_baseline_delta"] == pytest.approx(-0.4638888888888889)

    relative_output = ENCODER_REFERENCE
    copied_output = tmp_path / relative_output
    shutil.copytree(reference, copied_output)
    record_path = copied_output / "proof-run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["acceptance"] = "not passed"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(AssertionError, match="did not pass"):
        repro._encoder_output(copied_output)


def test_transformer_validator_accepts_v2_output_and_rejects_historical_failure_marker() -> None:
    reference = REPO / TRANSFORMER_REFERENCE
    stdout = "\n".join(
        (
            "ACCEPTANCE: PASSED",
            f"PASS deterministic-replay: pooled hidden states sha "
            f"{repro.EXPECTED_POOL_SHA256} reproduced bit-identically",
            f"PASS capture: capture_identity {repro.EXPECTED_CAPTURE_IDENTITY} shape (360, 4)",
            "PASS determinism: detect payload sha "
            + "0" * 64
            + f", localize payload sha {repro.EXPECTED_LOCALIZATION_SHA256}, capture identity stable",
        )
    )
    accepted = repro._transformer_output(reference, stdout)
    assert accepted["evidence_contract"] == "diagnostic-evidence-v2"
    assert accepted["acceptance"] == "passed"

    old_failure_stdout = stdout.replace("ACCEPTANCE: PASSED", "ACCEPTANCE: NOT PASSED")
    with pytest.raises(AssertionError, match="ACCEPTANCE: PASSED"):
        repro._transformer_output(reference, old_failure_stdout)
