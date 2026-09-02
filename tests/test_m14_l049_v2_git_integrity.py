"""Regression tests for byte-stable L04.9 v2 evidence across Git clones."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import scripts._m14_l049_v2_promotion as promotion
from scripts._m14_l049_v2_inputs import validate_canonical_stage_b_inputs

ROOT = Path(__file__).resolve().parents[1]
D3_PATH = Path(
    "artifacts/m14/l04-explanations.L049V2StageB.6af20749b305f591d2c90d868cb09e71f623bdd0.d3-promotion-real-v2.json"
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit_files(repo: Path, relative_paths: list[str]) -> None:
    _git(repo, "init", "--quiet")
    _git(repo, "add", "--", *relative_paths)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=L04.9 Git test",
            "-c",
            "user.email=l049-git-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "byte-stable evidence",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_real_canonical_paths_have_explicit_binary_attributes() -> None:
    expected_paths = [*promotion.REAL_CANONICAL_PATHS.values(), D3_PATH]

    assert len(promotion.REAL_CANONICAL_PATHS) == 18
    for relative in expected_paths:
        assert _git(ROOT, "check-attr", "text", "--", relative.as_posix()) == (f"{relative.as_posix()}: text: unset")


def test_real_canonical_would_be_index_blobs_match_current_bytes() -> None:
    for relative in promotion.REAL_CANONICAL_PATHS.values():
        value = relative.as_posix()
        index_line = _git(ROOT, "ls-files", "--stage", "--", value)
        index_oid = index_line.split()[1]
        path_oid = _git(ROOT, "hash-object", f"--path={value}", "--", value)
        raw_oid = _git(ROOT, "hash-object", "--no-filters", "--", value)
        assert path_oid == raw_oid == index_oid

    d3_oid = _git(ROOT, "hash-object", "--no-filters", "--", D3_PATH.as_posix())
    assert d3_oid == _git(ROOT, "hash-object", f"--path={D3_PATH.as_posix()}", "--", D3_PATH.as_posix())


def test_text_auto_is_rejected_before_stage_b_input_reads(tmp_path: Path) -> None:
    repo = tmp_path / "auto-text-repo"
    destination = repo / "artifacts" / "m14"
    destination.mkdir(parents=True)
    (repo / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    relative_paths: list[str] = []
    for source in promotion.REAL_CANONICAL_PATHS.values():
        if source.name not in {
            "l04-explanations.v2-authoring-manifest.json",
            "l04-explanations.v2-holdout.jsonl",
            "l04-explanations.v2-holdout.seed",
            "l04-explanations.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.candidate.json",
        }:
            continue
        shutil.copyfile(ROOT / source, destination / source.name)
        relative_paths.append((Path("artifacts/m14") / source.name).as_posix())
    _commit_files(repo, [".gitattributes", *relative_paths])

    assert validate_canonical_stage_b_inputs(repo, require_tracked=True) == ["canonical_input_text_attribute"]


def test_core_autocrlf_true_local_clone_preserves_canonical_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = source / "artifacts" / "m14"
    destination.mkdir(parents=True)
    shutil.copyfile(ROOT / ".gitattributes", source / ".gitattributes")
    relative_paths = [".gitattributes"]
    expected: dict[str, str] = {}
    for relative in [*promotion.REAL_CANONICAL_PATHS.values(), D3_PATH]:
        source_path = ROOT / relative
        destination_path = destination / relative.name
        shutil.copyfile(source_path, destination_path)
        relative_value = (Path("artifacts/m14") / relative.name).as_posix()
        relative_paths.append(relative_value)
        expected[relative_value] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    _commit_files(source, relative_paths)

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "-c", "core.autocrlf=true", "clone", "--quiet", "--local", str(source), str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    for relative_value, digest in expected.items():
        assert hashlib.sha256((clone / relative_value).read_bytes()).hexdigest() == digest
        assert _git(clone, "check-attr", "text", "--", relative_value) == f"{relative_value}: text: unset"
