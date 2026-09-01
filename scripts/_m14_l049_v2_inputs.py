"""Canonical, fail-closed Stage B input provisioning and validation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from scripts._m14_l049_v2_fixture import (
    TRAIN_FIXTURE_PATH,
    authoring_manifest_digest,
    read_rows,
    validate_fixture,
    validate_rows,
)
from scripts._m14_l049_v2_schema import (
    AUTHORING_MANIFEST_FILE_SHA256,
    EXPECTED_AUTHORING_MANIFEST_SHA256,
    EXPECTED_HOLDOUT_CONTENT_SHA256,
    EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256,
    V2_ADDENDUM_PATH,
    canonical_digest,
    fixture_digest,
)
from scripts._m14_l049_v2_validate import validate_stage_a

CANONICAL_STAGE_B_MANIFEST = Path("artifacts/m14/l04-explanations.v2-authoring-manifest.json")
CANONICAL_STAGE_B_HOLDOUT = Path("artifacts/m14/l04-explanations.v2-holdout.jsonl")
CANONICAL_STAGE_B_SEED = Path("artifacts/m14/l04-explanations.v2-holdout.seed")
SOURCE_KEYED_STAGE_B_CANDIDATE = Path(
    "artifacts/m14/l04-explanations.L049V2StageA.76a45ea74fbb2843b7d109855c2c387ab98b3e47.candidate.json"
)
EXPECTED_STAGE_B_CANDIDATE_FILE_SHA256 = "29bcd20ab494092abbb074bff5d99d091ec288d261a0399f97f2e2fb4f092aa2"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_stage_b_paths(root: Path) -> dict[str, Path]:
    return {
        "manifest": root / CANONICAL_STAGE_B_MANIFEST,
        "holdout": root / CANONICAL_STAGE_B_HOLDOUT,
        "seed": root / CANONICAL_STAGE_B_SEED,
        "candidate": root / SOURCE_KEYED_STAGE_B_CANDIDATE,
    }


def _safe_regular_under(root: Path, path: Path) -> bool:
    try:
        root_real = root.resolve(strict=True)
        path_real = path.resolve(strict=True)
        return path.is_file() and not path.is_symlink() and path_real.parent == root_real / "artifacts" / "m14"
    except OSError:
        return False


def _is_tracked(root: Path, path: Path) -> bool:
    """Check index membership without invoking a shell or exposing git output."""
    try:
        relative = path.relative_to(root).as_posix()
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return False
    return result.returncode == 0 and result.stdout.strip() == relative


def _repo_root_matches(root: Path) -> bool:
    """Require ``root`` to be the actual Git worktree root without leaking output."""
    try:
        requested_root = root.resolve(strict=True)
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        reported_text = result.stdout.strip()
        if result.returncode != 0 or not reported_text or "\n" in reported_text or "\r" in reported_text:
            return False
        reported_root = Path(reported_text).resolve(strict=True)
        return os.path.normcase(str(reported_root)) == os.path.normcase(str(requested_root))
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return False


def validate_canonical_stage_b_inputs(root: Path, *, require_tracked: bool = False) -> list[str]:
    """Return fixed diagnostic codes; never include paths, values, or content."""
    if require_tracked and not _repo_root_matches(root):
        return ["canonical_repo_root_mismatch"]
    paths = canonical_stage_b_paths(root)
    errors: list[str] = []
    if any(not _safe_regular_under(root, path) for path in paths.values()):
        errors.append("canonical_input_shape")
        return errors
    if require_tracked and any(not _is_tracked(root, path) for path in paths.values()):
        errors.append("canonical_input_untracked")
        return errors
    expected_files = {
        "manifest": AUTHORING_MANIFEST_FILE_SHA256,
        "holdout": EXPECTED_HOLDOUT_CONTENT_SHA256,
        "seed": EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256,
    }
    for name, expected in expected_files.items():
        try:
            if _digest(paths[name]) != expected:
                errors.append(f"canonical_{name}_digest")
        except OSError:
            errors.append(f"canonical_{name}_unreadable")
    try:
        manifest = json.loads(paths["manifest"].read_bytes())
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("manifest_sha256") != EXPECTED_AUTHORING_MANIFEST_SHA256
            or authoring_manifest_digest(manifest) != EXPECTED_AUTHORING_MANIFEST_SHA256
        ):
            errors.append("canonical_manifest_commitment")
    except (OSError, json.JSONDecodeError, TypeError):
        errors.append("canonical_manifest_schema")
        manifest = {}
    try:
        _raw, holdout_rows = read_rows(paths["holdout"])
        errors.extend(f"holdout_{error}" for error in validate_rows(holdout_rows, "holdout", 24))
        if fixture_digest(holdout_rows) != EXPECTED_HOLDOUT_CONTENT_SHA256:
            errors.append("holdout_fixture_commitment")
        _train_raw, train_rows = read_rows(TRAIN_FIXTURE_PATH)
        errors.extend(f"cross_fixture_{error}" for error in validate_fixture(train_rows, holdout_rows))
    except (OSError, ValueError, UnicodeError):
        errors.append("canonical_holdout_schema")
        holdout_rows = []
    try:
        seed = paths["seed"].read_bytes()
        if len(seed) != 32 or hashlib.sha256(seed).hexdigest() != EXPECTED_HOLDOUT_SEED_COMMITMENT_SHA256:
            errors.append("holdout_seed_commitment")
    except OSError:
        errors.append("canonical_seed_unreadable")
    try:
        candidate = json.loads(paths["candidate"].read_bytes())
        addendum = json.loads(V2_ADDENDUM_PATH.read_bytes())
        _train_raw, train_rows = read_rows(TRAIN_FIXTURE_PATH)
        if _digest(paths["candidate"]) != EXPECTED_STAGE_B_CANDIDATE_FILE_SHA256:
            errors.append("candidate_file_commitment")
        elif not isinstance(candidate, Mapping) or validate_stage_a(candidate, train_rows, addendum):
            errors.append("candidate_stage_a_validation")
        elif candidate.get("selection", {}).get("consensus_candidate") != {"layer": 10, "offset": 0}:
            errors.append("candidate_selection_binding")
        elif candidate.get("artifact_sha256") != canonical_digest(candidate, "artifact_sha256"):
            errors.append("candidate_digest")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        errors.append("candidate_schema")
    return errors


__all__ = [
    "CANONICAL_STAGE_B_HOLDOUT",
    "CANONICAL_STAGE_B_MANIFEST",
    "CANONICAL_STAGE_B_SEED",
    "SOURCE_KEYED_STAGE_B_CANDIDATE",
    "canonical_stage_b_paths",
    "validate_canonical_stage_b_inputs",
]
