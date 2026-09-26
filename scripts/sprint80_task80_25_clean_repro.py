"""Committed clean replay for the accepted Sprint 80 core proofs.

Run from the repository root (the external snapshots must already be present)::

    uv run python scripts/sprint80_task80_25_clean_repro.py --hf-root F:/llms/hf/models

The driver pins the prospective encoder v3 and transformer-v1/target-evidence-v2
inputs, proof entrypoints, core implementation sources, supplemental stability
protocol and its immutable precedent records, and the external GPT-2/WikiText
snapshot files. It copies the repository into a new temporary root, installs a
non-editable locked environment, gives each proof fresh runtime caches seeded
only from the hash-verified external snapshots, and independently inspects the
produced content-addressed outputs. The 80.24 stability supplement is a
separate proof and output root; it is not folded into the seven-stage v2 run.

Historical encoder-v1 and pre-amendment transformer failures remain in their
original evidence directories and are never executed or reclassified here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import psutil

REPO = Path(__file__).resolve().parents[1]

# Frozen acceptance inputs, proof sources, the complete installed Python
# package source tree, critical implementation modules, and historical
# evidence references. The earlier accepted replay used uncommitted proof
# source bytes; those digests are recorded separately.
PINNED_INPUT_SHA256: dict[str, str] = {
    ".python-version": "a7b9da5e355ce2ea4508cc53756849f44c960d87a38e5a76d0c734d7512c5371",
    "pyproject.toml": "7050e09260e2f11dcd12b46faf65ddd7f526563d3a78846187b2b1898653913d",
    "uv.lock": "cb6a23d95b73954f212a29e3e0360fc20773ead147c7709c460e4c6e79eb1379",
    "artifacts/benchmark_manifest_schema_v1.json": "de24e80e267484df99f880927885fef836f2d70e4bd750e4a7a61d0c979364fc",
    "artifacts/diagnostic_report_schema_v1.json": "a76580e58a672ac283fa0a313b34f6148375dc5181dcb395ae1061ce42928d07",
    "artifacts/m14/l04-wikitext-2-manifest.json": "0908f843efd72ce93c628e34cdb27f56e37e764d196ef91be7eff6d7757b78f3",
    "scripts/sprint80_task80_23_proof.py": "4b574d04a1a58c9508798197da3b6cda4909ba203b5f74f97b383e89a57f2922",
    "scripts/sprint80_task80_23_v3_proof.py": "87891b0c75d4ddcfe890792e421ac4108801d08aa49a9d3a234da764481cb53f",
    "scripts/sprint80_task80_24_proof.py": "dee2d016bf9a22dfd5ef9089ca7bea74bc28bf86d6f36f78ae28db505f8d6f7a",
    "scripts/sprint80_task80_24_stability_proof.py": "a1c08fff6cb45f8a768ae57d2f727a4a89392b04f0b3bd26f4807e0972dc8ac5",
    "src/latent_anything/_artifact_path.py": "75fcd5ec39287a25e827193b12cedb3041da5f00680488e97b24bde05a82dc5a",
    "src/latent_anything/_benchmark_manifest.py": "2016ced9d323aee717d44c6888c582f553704304fe7d965d4b27401c24621a49",
    "src/latent_anything/_capture_binding.py": "1e266ba8bb6d39d522751b93bb05e6d6fa396a1df29e3ffb3b9a3fdb877a06a6",
    "src/latent_anything/_collapse_detection.py": "6187350691082b396d4254ce118f5793a059ec4aa2c1c19afeedbd3372b36d19",
    "src/latent_anything/_diagnostic_artifact.py": "6ffe0410625adc10fd6082f576df1fa3137c5e4a82befe0a53a118555e175952",
    "src/latent_anything/_diagnostic_report.py": "6314d7914392783748193d4bf3e92f3b257d33e80f29d0a2c0dc9c7b420cae74",
    "src/latent_anything/_diagnostic_validator.py": "196ed217ff57d5ae4b6777614287cd887f7c3ff21d19b203a7852c6462385247",
    "src/latent_anything/_diagnostic_workflow.py": "86778e9ea0b3b037f2941daed89913c5bb2fa454a463b8c9a501a0fd575c2be4",
    "src/latent_anything/_intervention_trials.py": "1d26a9d0c17f23f5e12c736997431e2a9b39205785e6a119d9ebedd5ca3e1a37",
    "src/latent_anything/_portable_contract.py": "a937904258761f15584d5f9730b9a33644126df5793112709a7c6d2986dfcefa",
    "src/latent_anything/_report_renderer.py": "eb358345cad8973e7dd04fc63fafd550acb4287e12b19654441d39f288c5994e",
    "src/latent_anything/_run_comparison.py": "2ed52f653090103e82ae2b337b60ec4038e961906f02f300d844d95dc4afc11b",
    "src/latent_anything/_run_record_codec.py": "5916cd3c64a6627a5a64d5028bdc9bd3331e007f51ccc16eec0c82f3131d9826",
    "src/latent_anything/_run_record_schema.py": "01157070b79762fb43a03e3d8fd2e8c8742a84affa2ccc53e7b5d97a9d234962",
    "src/latent_anything/_target_evidence.py": "c213f857d5e2172772b2efbda970b1a44ef8c59db6933a8e3a54667f5fc567dd",
    "src/latent_anything/capture.py": "fc22bb957bfedf2a874ee21127312354d4760cd72697e607c06c218801845269",
    "src/latent_anything/diagnostics.py": "f6d722a5dbd52b325abd328d2eb0c2a83216a8f61d8cda3837bbe0c058312f7d",
    "src/latent_anything/latent_space.py": "7bf85b9fa1d0473ae9ab8efe83b1b1f66fd32ea9b9cde3b08ef950d2882f0aa9",
    "src/latent_anything/latent_value.py": "8981be158fe12b7e1a06ead84a4e7daaad7201c757047e540c92c9b6f9b1b8a1",
    "src/latent_anything/probes.py": "78bf42fee77748fe44427e1771bb4808f0e25b48a57c7eb78dc93dd988bd1aab",
    "artifacts/task_80.25_evidence/summary.json": "49a6ed1ad34ebe2f889daea38a4f0cdb8ae85b11f1b5388f877d80596cd541a5",
    **dict(
        [
            (
                "artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v1.json",
                "e2935cd821a1605c719b63c748a9f93c2375cd6087c51739862d480a18c56959",
            ),
            (
                "artifacts/benchmark_manifest_sprint80_transformer_hidden_state_v1.json",
                "339c5c302def9d8b859e1914101803c84905347eec3caa1d596a5eadc015d0a2",
            ),
            (
                "artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v3_model_weight_lesion.json",
                "e5bcd341b1b746f05e4ac02f908bcd27f23bace39737c86c8ac38888e7451b82",
            ),
            (
                "artifacts/benchmark_model_sprint80_encoder_autoencoder_v3_baseline.npz",
                "92829e57aaaebf52d5446d1a9c9db5de4e12c1657fad5ecc986329ed8752bb2a",
            ),
            (
                "artifacts/benchmark_manifest_sprint80_transformer_explanation_stability_v1.json",
                "a69b7152a800ac543ac76fec4d699c16681517708a95b353e5d241241c8acf0a",
            ),
            (
                "src/latent_anything/_layer_slice_localization.py",
                "abf50c74f5ac675576642745435ea252a06c7330ba36a9fe91cb48c2716b14d0",
            ),
            (
                "src/latent_anything/_probe_tcav_ig_explanation.py",
                "0d114dfeb884c5ddbbabbee9f077ac284e547936a249423b65f517b456c75f93",
            ),
            (
                "src/latent_anything/_redundancy_separability_detection.py",
                "961a49dbe0f59a9bfd7f0bdbda49e6d0cfc08217bd093e8ce81161fdc7dd1b76",
            ),
            (
                "src/latent_anything/_representation_taxonomy.py",
                "9227b6266d62b8c94f6c5f5e9cf397fa5e26ba606e2386265d3ceb83f04c1e19",
            ),
            (
                (
                    "artifacts/diagnostics/proof-80-25-clean-transformer-v1-target-v2-20260924-"
                    "035357-6996/artifacts/"
                    "3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19"
                ),
                "3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19",
            ),
            (
                "artifacts/diagnostics/proof-80-24-v2-rule-bound/artifacts/"
                "e2ebd582ac9fa0defe03f1b221290067d931a9b99b606b661da75fccab7eee7a",
                "e2ebd582ac9fa0defe03f1b221290067d931a9b99b606b661da75fccab7eee7a",
            ),
            (
                "artifacts/diagnostics/proof-80-24-v2-rule-bound/artifacts/"
                "d629ecc3d0562bdd71292a332f58778b6ab70444ee7fc1d74766e32707650b1c",
                "d629ecc3d0562bdd71292a332f58778b6ab70444ee7fc1d74766e32707650b1c",
            ),
            (
                "artifacts/diagnostics/proof-80-24-v2-rule-bound/artifacts/"
                "c670acea7b27665581844916f17fb03aa38b5aa6d7740583dea34a5a6f467bdb",
                "c670acea7b27665581844916f17fb03aa38b5aa6d7740583dea34a5a6f467bdb",
            ),
            (
                "artifacts/diagnostics/proof-80-24-v2-rule-bound/diagnostic-report",
                "788b9a316ff3ec14922b7a1515fd8aee2b0d2b4b4d0dc884f321eb0e68b8693a",
            ),
            (
                "artifacts/diagnostics/proof-80-24-v2-rule-bound/runs/421578bb1c06fa15.json",
                "d67afbb637f7f6752b042274d06cb26a7cc1f1223c2f62a8c3681e2cce4a5211",
            ),
            (
                (
                    "artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-complete-"
                    "20260923/artifacts/"
                    "e5c6db0ab5080807138e091bc332a5d3e788889a382917e93775b15f562aef0f"
                ),
                "e5c6db0ab5080807138e091bc332a5d3e788889a382917e93775b15f562aef0f",
            ),
            (
                "artifacts/diagnostics/proof-80-23-v3-model-weight-lesion-complete-20260923/proof-run.json",
                "deb38278b6442db366d09a2b8b5c0271eada8162f3913efd65f37648b47ba7f1",
            ),
            (
                "artifacts/task_80.25_evidence_attempt1_failed/summary.json",
                "80b4be9e049ba6324e8dc46431fc1bbf3ba2d76f748325bd8d44a8d4657aa140",
            ),
            (
                "artifacts/task_80.25_evidence_clean_20260924/summary.json",
                "ae5d291cb78237017134ca968b34738153e71b0f5fd11fa1e15babb868bdd382",
            ),
            (
                "artifacts/task_80.25_evidence_clean_20260924/commands.json",
                "739638711298932d17232ea1c7845a48dd9baec9effcf4c3c6e07030380f28f9",
            ),
            (
                "artifacts/diagnostics/proof-80-25-clean-encoder-v3-20260924-035357-6996/proof-run.json",
                "209d83b16d15f9322126efcf5451201189b69c18b097df80f74c66a3842412bd",
            ),
            (
                (
                    "artifacts/diagnostics/proof-80-25-clean-transformer-v1-target-v2-20260924-"
                    "035357-6996/runs/c4012406e89254ea.json"
                ),
                "37b80651f380eaf4dbbae456a5f74bb9a0f6eddda5ab6cd7c2b897991a984bbb",
            ),
            (
                (
                    "artifacts/diagnostics/proof-80-25-clean-transformer-v1-target-v2-20260924-"
                    "035357-6996/diagnostic-report"
                ),
                "930a2082d9fa297c1b4f90f5c7858dbea65d4f8f6085fc6048be5f4294360aad",
            ),
        ]
    ),
}

PINNED_SOURCE_TREE_FILE_COUNT = 162
PINNED_SOURCE_TREE_SHA256 = "8df33742de6c3072969aff823b514500c4b067592fc4d0ae303638f994fed3b6"

PINNED_HF_SNAPSHOT: dict[str, dict[str, object]] = {
    "models--openai-community--gpt2": {
        "revision": "e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "files": {
            "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
            "generation_config.json": "ed0b32ac72c0f5f44a719abb2d7786ea5146c871f83717b7f2018065954de02b",
            "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
            "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
            "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
            "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
        },
    },
    "datasets--Salesforce--wikitext": {
        "revision": "f776294184f13b8ff2337b3841cf9269a6216d1e",
        "files": {
            "README.md": "b7c15c5f44ca6e49d92f1843652287e48f5aaaace3d51af499aed3a3f2fba691",
            **dict(
                [
                    (
                        "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                        "3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925",
                    ),
                    (
                        "wikitext-2-raw-v1/train-00000-of-00001.parquet",
                        "49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701",
                    ),
                    (
                        "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
                        "a10eb5a42c4264c45dd38bfef4da6a74ddef5fc22cb419e1a89d54c4cc6d748f",
                    ),
                ]
            ),
        },
    },
}

PINNED_VERSIONS = {
    "numpy": "2.4.6",
    "scikit-learn": "1.9.0",
    "torch": "2.10.0",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "huggingface-hub": "0.35.3",
    "datasets": "3.6.0",
    "fsspec": "2025.3.0",
}
PINNED_TORCH_MODULE_VERSION = "2.10.0+cpu"
OVERLAY_PACKAGES = ("datasets==3.6.0", "huggingface-hub==0.35.3")
OVERLAY_ALLOWED_ADJUSTMENTS = {"fsspec": "2025.3.0"}
PROOF_STAGES = ("capture", "detect", "localize", "explain", "intervene", "compare", "report")
MAX_PROOF_SECONDS = 1800
MAX_PROCESS_TREE_RSS_BYTES = 16 * (1 << 30)
MIN_FREE_DISK_BYTES = 12 * (1 << 30)
EXPECTED_POOL_SHA256 = "b0e60bc54412e50e917ec8b9dfb3c4c2af467b691b27953e9a5feb5d1021a33b"
EXPECTED_CAPTURE_IDENTITY = "28f4fbd2c47231fec8908044d335d06a87b2fda2f18bde1e3f0567dcb8c05d89"
EXPECTED_LOCALIZATION_SHA256 = "1129ac27a96fdbf79fff47e985779a7dcbd9c591e70319bedb0d2caceb00c2bc"
EXPECTED_TARGET_RECORD_SHA256 = "3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19"
EXPECTED_TARGET_RULE_SHA256 = "9b12485635decf738c4904702f75d7d5ae8cb6663555e2af5ee66476269f025b"

# The older successful clean replay was run by an uncommitted inline Python
# driver against these source bytes. It is historical provenance, not this
# committed-driver run or a substitute for its verification.
HISTORICAL_TEMP_DRIVER_SOURCE_SHA256 = {
    "scripts/sprint80_task80_23_v3_proof.py": "f9f25df0e746c1423b116ea5c3e481ab8e10837a856d4fad0618297689c888be",
    "scripts/sprint80_task80_24_proof.py": "b9e036eba0fb6831dbfa3ba6ba960f51843c8cce0e24f887b635146c2488a31a",
}

COPY_INCLUDE = (
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
    "LICENSE",
    "src",
    "scripts",
    "artifacts",
    "tests",
)
EXCLUDED_DIR_NAMES = frozenset(
    {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".pyright"}
)
STRIP_ENV_NAMES = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONCASEOK",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_MODULES_CACHE",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "TMP",
        "TEMP",
        "TMPDIR",
        "UV_CACHE_DIR",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
    }
)


def say(message: str) -> None:
    print(f"[80.25] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(repo: Path) -> tuple[int, str]:
    """Fingerprint every Python source file installed from the package tree."""
    source_root = repo / "src" / "latent_anything"
    if not source_root.is_dir():
        raise AssertionError(f"pinned package source tree missing: {source_root}")
    files = sorted(source_root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(repo).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return len(files), digest.hexdigest()


def verify_pinned_inputs(repo: Path) -> dict[str, str]:
    """Hash every frozen repository and source input, failing on any drift."""
    verified: dict[str, str] = {}
    for relative, expected in PINNED_INPUT_SHA256.items():
        path = repo / relative
        if not path.is_file():
            raise AssertionError(f"pinned input missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise AssertionError(f"pinned input drifted: {relative} sha256 {actual} != {expected}")
        verified[relative] = actual
    source_count, source_digest = source_tree_sha256(repo)
    if source_count != PINNED_SOURCE_TREE_FILE_COUNT or source_digest != PINNED_SOURCE_TREE_SHA256:
        raise AssertionError(
            "pinned package source tree drifted: "
            f"files={source_count} sha256={source_digest} "
            f"!= files={PINNED_SOURCE_TREE_FILE_COUNT} sha256={PINNED_SOURCE_TREE_SHA256}"
        )
    verified["src/latent_anything/**/*.py"] = source_digest
    return verified


def _hf_snapshot_path(hub_root: Path, repo_id: str, revision: str) -> Path:
    return hub_root / repo_id / "snapshots" / revision


def verify_hf_snapshot(hf_home: Path) -> dict[str, dict[str, object]]:
    """Verify all immutable external or freshly seeded snapshot files."""
    verified: dict[str, dict[str, object]] = {}
    hub_root = hf_home / "hub"
    for repo_id, spec in PINNED_HF_SNAPSHOT.items():
        revision = str(spec["revision"])
        files = cast(Mapping[str, str], spec["files"])
        snapshot = _hf_snapshot_path(hub_root, repo_id, revision)
        if not snapshot.is_dir():
            raise AssertionError(f"declared snapshot missing: {snapshot}")
        rows: dict[str, dict[str, object]] = {}
        for relative, expected in files.items():
            path = snapshot / relative
            if not path.is_file():
                raise AssertionError(f"declared snapshot file missing: {repo_id}@{revision}/{relative}")
            actual = sha256_file(path)
            if actual != expected:
                raise AssertionError(
                    f"declared snapshot drifted: {repo_id}@{revision}/{relative} {actual} != {expected}"
                )
            rows[relative] = {"sha256": actual, "size_bytes": path.stat().st_size}
        verified[repo_id] = {"revision": revision, "files": rows}
    return verified


def seed_hf_snapshot(source_hf_home: Path, fresh_hf_home: Path) -> dict[str, object]:
    """Copy only declared immutable model/data files into a fresh Hub cache."""
    if fresh_hf_home.exists() and any(fresh_hf_home.iterdir()):
        raise AssertionError(f"fresh Hugging Face cache is not empty: {fresh_hf_home}")
    fresh_hf_home.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, object]] = []
    for repo_id, spec in PINNED_HF_SNAPSHOT.items():
        revision = str(spec["revision"])
        files = cast(Mapping[str, str], spec["files"])
        source = _hf_snapshot_path(source_hf_home / "hub", repo_id, revision)
        destination = _hf_snapshot_path(fresh_hf_home / "hub", repo_id, revision)
        destination.mkdir(parents=True, exist_ok=True)
        for relative in files:
            source_file = source / relative
            destination_file = destination / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_file)
            copied.append({"repo_id": repo_id, "revision": revision, "path": relative})
    verified = verify_hf_snapshot(fresh_hf_home)
    return {"file_count": len(copied), "files": copied, "verified": verified}


def _excluded_relative(relative: Path) -> bool:
    return any(
        part in EXCLUDED_DIR_NAMES or part.endswith(".egg-info") or part.endswith(".pyc") for part in relative.parts
    )


def copy_fresh_root(source: Path, destination: Path) -> dict[str, object]:
    """Copy only the declared project tree, excluding developer/generated state."""
    if destination.exists() and any(destination.iterdir()):
        raise AssertionError(f"fresh root is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    total_bytes = 0
    for entry in COPY_INCLUDE:
        source_path = source / entry
        target_path = destination / entry
        if not source_path.exists():
            raise AssertionError(f"required project input is missing: {entry}")
        if source_path.is_file():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied.append(entry)
            total_bytes += target_path.stat().st_size
            continue
        for current, directories, filenames in os.walk(source_path):
            current_path = Path(current)
            relative_dir = current_path.relative_to(source)
            directories[:] = sorted(name for name in directories if not _excluded_relative(relative_dir / name))
            for filename in sorted(filenames):
                source_file = current_path / filename
                relative = source_file.relative_to(source)
                if _excluded_relative(relative):
                    continue
                target_file = destination / relative
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)
                copied.append(relative.as_posix())
                total_bytes += target_file.stat().st_size
    return {"files": sorted(copied), "count": len(copied), "bytes": total_bytes}


def sha256_tree(root: Path) -> dict[str, str]:
    """Content inventory excluding venvs, Python bytecode, and generated caches."""
    inventory: dict[str, str] = {}
    for current, directories, filenames in os.walk(root):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root)
        directories[:] = sorted(name for name in directories if not _excluded_relative(relative_dir / name))
        for filename in sorted(filenames):
            path = current_path / filename
            relative = path.relative_to(root)
            if _excluded_relative(relative):
                continue
            inventory[relative.as_posix()] = sha256_file(path)
    return inventory


def diff_inventories(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, list[str]]:
    before_keys, after_keys = set(before), set(after)
    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "modified": sorted(path for path in before_keys & after_keys if before[path] != after[path]),
    }


def _is_build_metadata(path: str) -> bool:
    return path.startswith("build/") or ".egg-info/" in path


def _inside_any(path: str, prefixes: Sequence[str]) -> bool:
    parts = Path(path).parts
    return any(parts[: len(Path(prefix).parts)] == Path(prefix).parts for prefix in prefixes)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return cast(dict[str, object], value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def record_check(
    checks: list[dict[str, object]], name: str, ok: bool, detail: object, *, case: str | None = None
) -> None:
    row: dict[str, object] = {"check": name, "ok": bool(ok), "detail": detail}
    if case is not None:
        row["case"] = case
    checks.append(row)
    if not ok:
        raise AssertionError(f"{name} failed: {json.dumps(detail, ensure_ascii=False, sort_keys=True)}")


def build_child_env(
    work: Path,
    label: str,
    cache_home: Path,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Use fresh per-proof temp/data/model caches and strip developer secrets."""
    source = dict(os.environ if base is None else base)
    env = {key: value for key, value in source.items() if key not in STRIP_ENV_NAMES}
    temp_dir = work / "runtime" / label / "tmp"
    generated = work / "runtime" / label / "generated"
    (generated / "datasets").mkdir(parents=True, exist_ok=True)
    (generated / "modules").mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    cache_home.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "TMP": str(temp_dir),
            "TEMP": str(temp_dir),
            "TMPDIR": str(temp_dir),
            "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "HF_HUB_CACHE": str(cache_home / "hub"),
            "HUGGINGFACE_HUB_CACHE": str(cache_home / "hub"),
            "HF_DATASETS_CACHE": str(generated / "datasets"),
            "HF_MODULES_CACHE": str(generated / "modules"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
        }
    )
    return env


def _process_tree(process: psutil.Process) -> list[psutil.Process]:
    result = [process]
    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
        result.extend(process.children(recursive=True))
    return result


def _sample_tree(process: psutil.Process) -> int:
    total = 0
    for child in _process_tree(process):
        try:
            info = child.memory_info()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        total += max(info.rss, getattr(info, "peak_wset", 0))
    return total


def _kill_tree(process: psutil.Process, popen: subprocess.Popen[str]) -> None:
    for child in reversed(_process_tree(process)):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            child.kill()
    with contextlib.suppress(ProcessLookupError):
        popen.kill()


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: int,
    label: str,
    log_path: Path,
    memory_limit: int = MAX_PROCESS_TREE_RSS_BYTES,
) -> dict[str, object]:
    """Run one fixed argv while measuring its process tree and retaining its log."""
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise RuntimeError(f"could not launch {label}: {exc}") from exc
    assert process.stdout is not None
    ps_process = psutil.Process(process.pid)
    output: list[str] = []

    def pump() -> None:
        for line in process.stdout:  # type: ignore[union-attr]
            output.append(line)

    reader = threading.Thread(target=pump, name=f"{label}-output", daemon=True)
    reader.start()
    peak_rss = 0
    timed_out = False
    memory_limit_exceeded = False
    while process.poll() is None:
        peak_rss = max(peak_rss, _sample_tree(ps_process))
        elapsed = time.perf_counter() - started
        if peak_rss > memory_limit:
            memory_limit_exceeded = True
            _kill_tree(ps_process, process)
            break
        if elapsed > timeout:
            timed_out = True
            _kill_tree(ps_process, process)
            break
        time.sleep(0.2)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)
    peak_rss = max(peak_rss, _sample_tree(ps_process))
    reader.join(timeout=10)
    text = "".join(output)
    log_path.write_text(text, encoding="utf-8")
    return {
        "argv": list(argv),
        "cwd": str(cwd),
        "exit": process.returncode,
        "label": label,
        "log": log_path.relative_to(REPO).as_posix() if log_path.is_relative_to(REPO) else str(log_path),
        "memory_limit_bytes": memory_limit,
        "memory_limit_exceeded": memory_limit_exceeded,
        "output_bytes": len(text.encode("utf-8")),
        "output_tail": text[-3000:],
        "peak_process_tree_rss_bytes": peak_rss,
        "timed_out": timed_out,
        "timeout_s": timeout,
        "wall_s": round(time.perf_counter() - started, 3),
        "text": text,
    }


def _run_tracked(
    commands: list[dict[str, object]],
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: int,
    label: str,
    log_dir: Path,
) -> dict[str, object]:
    result = run_command(
        argv,
        cwd=cwd,
        env=env,
        timeout=timeout,
        label=label,
        log_path=log_dir / f"{label}.log",
    )
    commands.append({key: value for key, value in result.items() if key != "text"})
    say(
        f"{label}: exit={result['exit']} wall={result['wall_s']}s "
        f"peak={int(result['peak_process_tree_rss_bytes']) / (1 << 20):.1f} MiB"
    )
    if result["timed_out"] or result["memory_limit_exceeded"] or result["exit"] != 0:
        raise AssertionError(
            f"{label} failed: exit={result['exit']} timeout={result['timed_out']} "
            f"rss_limit={result['memory_limit_exceeded']} output={str(result['output_tail'])[-1200:]}"
        )
    return result


def _parse_uv_list(output: str) -> dict[str, str]:
    start = output.find("[")
    if start < 0:
        raise AssertionError(f"uv pip list produced no JSON: {output[-1000:]}")
    value = json.loads(output[start:])
    return {str(row["name"]): str(row["version"]) for row in value}


def _isolation_probe(
    python_exe: Path,
    fresh_root: Path,
    *,
    commands: list[dict[str, object]],
    env: Mapping[str, str],
    log_dir: Path,
    label: str,
) -> dict[str, object]:
    names = sorted(PINNED_VERSIONS)
    snippet = (
        "import json, sys; from importlib.metadata import distribution, version; "
        "import latent_anything, torch; "
        f"names={names!r}; dist=distribution('latent-anything'); raw=dist.read_text('direct_url.json'); "
        "editable=bool(json.loads(raw).get('dir_info', {}).get('editable')) if raw else False; "
        "print('@@JSON@@'+json.dumps({'python':sys.version.split()[0], 'executable':sys.executable, "
        "'prefix':sys.prefix, 'latent_file':latent_anything.__file__, 'sys_path':sys.path, "
        "'editable':editable, 'versions':{n:version(n) for n in names}, 'torch_module':torch.__version__}))"
    )
    result = _run_tracked(
        commands,
        [str(python_exe), "-c", snippet],
        cwd=fresh_root,
        env=env,
        timeout=300,
        label=label,
        log_dir=log_dir,
    )
    text = str(result["output_tail"])
    marker = "@@JSON@@"
    if marker not in text:
        raise AssertionError(f"clean install probe had no structured output: {text[-1000:]}")
    record = json.loads(text.split(marker, 1)[1].splitlines()[0])
    root = str(fresh_root.resolve()).replace("\\", "/").casefold()
    prefix = str(Path(str(record["prefix"])).resolve()).replace("\\", "/").casefold()
    installed_file = str(Path(str(record["latent_file"])).resolve()).replace("\\", "/").casefold()
    if prefix != root + "/.venv" or not installed_file.startswith(root + "/.venv/"):
        raise AssertionError(f"clean package did not load from the fresh environment: {record}")
    if record["editable"]:
        raise AssertionError("latent-anything was installed editable")
    developer_root = str(REPO.resolve()).replace("\\", "/").casefold()
    leaked = [
        entry
        for entry in record["sys_path"]
        if (candidate := str(entry).replace("\\", "/").casefold()) == developer_root
        or candidate.startswith(developer_root + "/")
    ]
    if leaked:
        raise AssertionError(f"developer checkout leaked into clean sys.path: {leaked}")
    versions = cast(Mapping[str, str], record["versions"])
    for name, expected in PINNED_VERSIONS.items():
        if versions.get(name) != expected:
            raise AssertionError(f"clean dependency drift: {name} {versions.get(name)} != {expected}")
    if record["torch_module"] != PINNED_TORCH_MODULE_VERSION:
        raise AssertionError(f"unexpected torch module build: {record['torch_module']}")
    record["editable"] = False
    return cast(dict[str, object], record)


def _content_addressed_files(root: Path) -> dict[str, str]:
    directory = root / "artifacts"
    if not directory.is_dir():
        raise AssertionError(f"content-addressed artifact directory is missing: {directory}")
    result: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if re.fullmatch(r"[0-9a-f]{64}", path.name) is None:
            raise AssertionError(f"non-content-addressed file under artifacts/: {path}")
        actual = sha256_file(path)
        if actual != path.name:
            raise AssertionError(f"content-addressed artifact digest mismatch: {path.name} != {actual}")
        result[path.name] = actual
    return result


def _encoder_output(root: Path) -> dict[str, object]:
    proof = _read_json(root / "proof-run.json")
    if proof.get("acceptance") != "passed":
        raise AssertionError(f"encoder v3 proof did not pass: {proof.get('acceptance')}")
    if proof.get("manifest_id") != "sprint80-core-encoder-autoencoder-collapse-model-lesion-v3":
        raise AssertionError("encoder run record names the wrong prospective manifest")
    if (
        proof.get("manifest_raw_sha256")
        != PINNED_INPUT_SHA256["artifacts/benchmark_manifest_sprint80_encoder_autoencoder_v3_model_weight_lesion.json"]
    ):
        raise AssertionError("encoder run record does not bind the pinned v3 manifest")
    if proof.get("manifest_digest") != "f823efc1b016ca9ad62c0f207e17d138947040a39f5d1c1f42ab6d0d65f5954f":
        raise AssertionError("encoder manifest commitment changed")
    if (
        proof.get("baseline_checkpoint_sha256")
        != PINNED_INPUT_SHA256["artifacts/benchmark_model_sprint80_encoder_autoencoder_v3_baseline.npz"]
    ):
        raise AssertionError("encoder baseline checkpoint differs from the pinned input")
    if proof.get("workflow_status") != "completed" or proof.get("workflow_stages") != list(PROOF_STAGES):
        raise AssertionError("encoder proof did not complete the shared seven-stage workflow")
    artifact = cast(Mapping[str, object], proof["artifact"])
    validator = cast(Mapping[str, object], artifact["independent_validator"])
    if validator.get("status") != "passed":
        raise AssertionError("encoder artifact was not independently validator-clean")
    if artifact.get("render_deterministic") is not True or artifact.get("render_sha256") != artifact.get(
        "persisted_render_sha256"
    ):
        raise AssertionError("encoder report render is not byte-identical to the persisted report")
    intervention = cast(Mapping[str, object], proof["intervention"])
    comparison = cast(Mapping[str, object], proof["comparison"])
    effect = float(intervention["effect"])
    comparison_delta = float(cast(Mapping[str, object], comparison["task"])["signed_delta"])
    if (
        effect <= 0
        or intervention.get("expected_effect") != "increase"
        or intervention.get("conclusion") != "supported"
    ):
        raise AssertionError(f"encoder restoration effect has wrong direction or conclusion: {intervention}")
    if comparison_delta >= 0:
        raise AssertionError(f"encoder candidate-minus-baseline comparison delta has wrong sign: {comparison_delta}")
    if abs(effect - 0.4638888888888889) > 1e-12 or abs(comparison_delta + 0.4638888888888889) > 1e-12:
        raise AssertionError(f"encoder restoration/comparison measurements drifted: {effect}, {comparison_delta}")
    if not proof.get("manifest_raw_unchanged_during_run"):
        raise AssertionError("encoder proof observed a changed frozen manifest")
    blobs = _content_addressed_files(root)
    expected_digests = {
        str(artifact["artifact_digest"]),
        str(artifact["report_digest"]),
        str(cast(Mapping[str, object], artifact["rendered_report_artifact"])["digest"]),
    }
    if not expected_digests.issubset(blobs):
        raise AssertionError(
            f"encoder run record references missing content-addressed blobs: {expected_digests - set(blobs)}"
        )
    report_path_value = str(artifact["rendered_report_path"]).replace("\\", "/")
    rendered_path = Path(report_path_value)
    if not rendered_path.is_absolute():
        rendered_path = root.parent.parent.parent / rendered_path
    if not rendered_path.is_file() or sha256_file(rendered_path) != artifact["render_sha256"]:
        raise AssertionError("encoder rendered report file does not match its content-addressed digest")
    lesion_path = root / "model" / "lesioned_encoder.npz"
    lesion_record_path = root / "model-lesion.json"
    lesion_sha = sha256_file(lesion_path)
    lesion_record_sha = sha256_file(lesion_record_path)
    if lesion_sha != proof.get("lesioned_checkpoint_sha256") or lesion_record_sha != proof.get("lesion_record_sha256"):
        raise AssertionError("generated encoder lesion files do not match their run record")
    if lesion_sha != "81a29cc8d92b38164b108958764a6f1ffe07e5b822701077bd45faeffea91008":
        raise AssertionError("generated encoder lesion checkpoint differs from accepted v3 reference")
    if lesion_record_sha != "d1b0763ee791b0069dc94e13b698edb51f4fcd7711ff169ce22264c3130d0570":
        raise AssertionError("generated encoder lesion record differs from accepted v3 reference")
    return {
        "acceptance": proof["acceptance"],
        "artifact_digest": artifact["artifact_digest"],
        "content_addressed_blob_count": len(blobs),
        "independent_validator": dict(validator),
        "intervention_effect_restoration_increase": effect,
        "comparison_candidate_minus_baseline_delta": comparison_delta,
        "lesion_record_sha256": lesion_record_sha,
        "lesioned_checkpoint_sha256": lesion_sha,
        "manifest_commitment": proof["manifest_digest"],
        "manifest_sha256": proof["manifest_raw_sha256"],
        "report_digest": artifact["report_digest"],
        "render_sha256": artifact["render_sha256"],
        "run_id": artifact["run_id"],
        "workflow_stages": list(PROOF_STAGES),
        "workflow_status": proof["workflow_status"],
    }


def _transformer_output(root: Path, stdout: str) -> dict[str, object]:
    if re.search(r"(?m)^ACCEPTANCE: PASSED\b", stdout) is None:
        raise AssertionError("transformer proof output did not report ACCEPTANCE: PASSED")
    run_paths = sorted((root / "runs").glob("*.json"))
    if len(run_paths) != 1:
        raise AssertionError(f"expected one fresh transformer run record, found {len(run_paths)}")
    run_record = _read_json(run_paths[0])
    if run_record.get("status") != "completed" or run_record.get("run_id") != run_paths[0].stem:
        raise AssertionError("transformer persisted run record is not completed or content-addressed by its run id")
    if run_record.get("model_revisions") != {"openai-community/gpt2": "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"}:
        raise AssertionError("transformer output does not bind the pinned GPT-2 revision")
    if run_record.get("dataset_revisions") != {"Salesforce/wikitext": "f776294184f13b8ff2337b3841cf9269a6216d1e"}:
        raise AssertionError("transformer output does not bind the pinned WikiText revision")
    artifact_rows = cast(Sequence[Mapping[str, object]], run_record["artifacts"])
    by_name = {str(row["name"]): row for row in artifact_rows}
    required_names = {
        "diagnostic-artifact",
        "diagnostic-report",
        "diagnostic-report.rendered",
        "target-section-header-attribute-record",
        "stage-record-detect",
        "stage-record-localize",
    }
    if not required_names.issubset(by_name):
        raise AssertionError(
            f"transformer output omitted required registered artifacts: {required_names - set(by_name)}"
        )
    verified_artifacts: dict[str, str] = {}
    for name, row in by_name.items():
        relative = str(row["relative_path"]).replace("\\", "/")
        path = root / relative
        if not path.is_file():
            raise AssertionError(f"transformer artifact reference is missing: {name} -> {relative}")
        actual = sha256_file(path)
        if actual != row["digest"]:
            raise AssertionError(f"transformer artifact hash mismatch for {name}: {actual} != {row['digest']}")
        verified_artifacts[name] = actual
    artifact_document = _read_json(root / str(by_name["diagnostic-artifact"]["relative_path"]))
    validator = cast(Mapping[str, object], artifact_document["validator_result"])
    if validator.get("status") != "passed":
        raise AssertionError("transformer output did not pass its independent validator")
    report = _read_json(root / str(by_name["diagnostic-report"]["relative_path"]))
    if report.get("evidence_contract") != "diagnostic-evidence-v2":
        raise AssertionError("transformer output did not use target-evidence-v2")
    target = cast(Mapping[str, object], report["target_evidence"])
    if (
        target.get("target_id") != "section-header-attribute"
        or target.get("record_digest") != EXPECTED_TARGET_RECORD_SHA256
        or target.get("rule_digest") != EXPECTED_TARGET_RULE_SHA256
    ):
        raise AssertionError(f"transformer target evidence differs from the accepted v2 input: {target}")
    target_record = _read_json(root / str(by_name["target-section-header-attribute-record"]["relative_path"]))
    if target_record.get("target_id") != "section-header-attribute":
        raise AssertionError("transformer target record has the wrong target identity")
    if sha256_file(root / "diagnostic-report") != by_name["diagnostic-report.rendered"]["digest"]:
        raise AssertionError("transformer rendered report differs from its content-addressed render")
    detect_record = _read_json(root / str(by_name["stage-record-detect"]["relative_path"]))
    payload = cast(Mapping[str, object], detect_record["payload"])
    measurements = cast(Mapping[str, object], payload["measurements"])
    separability = cast(Mapping[str, object], measurements["separability"])
    if float(separability["heldout_accuracy"]) < 0.7 or float(separability["leakage_gap"]) < 0.15:
        raise AssertionError(f"transformer declared positive gates did not pass: {separability}")
    localization_record = _read_json(root / str(by_name["stage-record-localize"]["relative_path"]))
    pooled_match = re.search(r"pooled hidden states sha ([0-9a-f]{64})", stdout)
    capture_match = re.search(r"capture_identity ([0-9a-f]{64}) shape", stdout)
    localization_match = re.search(r"localize payload sha ([0-9a-f]{64})", stdout)
    if pooled_match is None or pooled_match.group(1) != EXPECTED_POOL_SHA256:
        raise AssertionError("transformer cold extraction did not reproduce the pinned pooled-state digest")
    if capture_match is None or capture_match.group(1) != EXPECTED_CAPTURE_IDENTITY:
        raise AssertionError("transformer capture identity differs from accepted evidence")
    if localization_match is None or localization_match.group(1) != EXPECTED_LOCALIZATION_SHA256:
        raise AssertionError("transformer localization payload differs from accepted evidence")
    artifacts = _content_addressed_files(root)
    if set(verified_artifacts.values()) - set(artifacts.values()):
        raise AssertionError("transformer run references blobs absent from its output root")
    return {
        "acceptance": "passed",
        "artifact_digest": verified_artifacts["diagnostic-artifact"],
        "content_addressed_blob_count": len(verified_artifacts),
        "detect_leakage_gap": float(separability["leakage_gap"]),
        "detect_heldout_accuracy": float(separability["heldout_accuracy"]),
        "detect_randomized_accuracy": float(
            cast(Mapping[str, object], payload["control_metrics"])["control-label-randomization"][
                "heldout-probe-accuracy"
            ]
        ),
        "evidence_contract": report["evidence_contract"],
        "independent_validator": dict(validator),
        "localize_outcome": localization_record.get("outcome"),
        "manifest_commitment": cast(Mapping[str, object], artifact_document["workflow"])["manifest_digest"],
        "pooled_hidden_states_sha256": pooled_match.group(1),
        "report_digest": verified_artifacts["diagnostic-report"],
        "render_sha256": verified_artifacts["diagnostic-report.rendered"],
        "run_id": run_record["run_id"],
        "target_record_sha256": target["record_digest"],
        "target_rule_sha256": target["rule_digest"],
        "workflow_status": run_record["status"],
    }


def _stability_output(root: Path, stdout: str) -> dict[str, object]:
    if re.search(r"(?m)^ACCEPTANCE: PASSED\b", stdout) is None:
        raise AssertionError("supplemental stability proof did not report ACCEPTANCE: PASSED")
    run_paths = sorted((root / "runs").glob("*.json"))
    report_paths = sorted((root / "reports").glob("*.md"))
    if len(run_paths) != 1 or len(report_paths) != 1:
        raise AssertionError(f"stability output must contain one run and one report: {run_paths}, {report_paths}")
    record_path, report_path = run_paths[0], report_paths[0]
    record_digest = sha256_file(record_path)
    report_digest = sha256_file(report_path)
    if record_path.stem != record_digest or report_path.stem != report_digest:
        raise AssertionError("supplemental stability artifacts are not content-addressed")
    record = _read_json(record_path)
    if (
        cast(Mapping[str, object], record["protocol"]).get("sha256")
        != PINNED_INPUT_SHA256["artifacts/benchmark_manifest_sprint80_transformer_explanation_stability_v1.json"]
    ):
        raise AssertionError("supplemental stability output did not use the pinned protocol")
    acceptance = cast(Mapping[str, object], record["acceptance"])
    if acceptance.get("outcome") != "supported" or not acceptance.get("claim_allowed"):
        raise AssertionError("supplemental stability output did not independently support its claim")
    evidence = cast(Mapping[str, object], cast(Mapping[str, object], record["explanation"])["evidence"])
    gates = {
        "fidelity": cast(Mapping[str, object], evidence["fidelity"]),
        "stability": cast(Mapping[str, object], evidence["stability"]),
        "selectivity": cast(Mapping[str, object], evidence["selectivity"]),
    }
    if not acceptance.get("all_three_gates_passed") or not acceptance.get("controls_passed"):
        raise AssertionError("supplemental stability did not pass all three gates and controls")
    for name, gate in gates.items():
        if gate.get("status") != "passed":
            raise AssertionError(f"supplemental {name} gate did not pass: {gate}")
    stability = gates["stability"]
    if stability.get("threshold") != [">=", 0.8] or float(stability["observed"]) < 0.8:
        raise AssertionError(f"supplemental grouped stability gate is not the frozen >=0.8 gate: {stability}")
    runtime = cast(Mapping[str, object], record["runtime"])
    if runtime.get("fresh_replay_bit_identical") is not True:
        raise AssertionError("supplemental full model replay was not bit-identical")
    return {
        "acceptance": dict(acceptance),
        "content_addressed_report_sha256": report_digest,
        "content_addressed_run_sha256": record_digest,
        "fidelity": dict(gates["fidelity"]),
        "protocol_sha256": cast(Mapping[str, object], record["protocol"])["sha256"],
        "runtime_seconds": runtime["total_seconds"],
        "stability": dict(stability),
        "selectivity": dict(gates["selectivity"]),
        "supplemental_not_part_of_transformer_v2_seven_stage_run": True,
    }


def _cache_state(work: Path, label: str) -> dict[str, object]:
    runtime = work / "runtime" / label
    names = {
        "tmp": runtime / "tmp",
        "datasets": runtime / "generated" / "datasets",
        "modules": runtime / "generated" / "modules",
    }
    result: dict[str, object] = {}
    for name, path in names.items():
        files = (
            sorted(item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file())
            if path.exists()
            else []
        )
        result[name] = {"path": str(path), "files": files}
    return result


def _cache_empty(cache: Mapping[str, object]) -> bool:
    return all(cast(Sequence[object], cast(Mapping[str, object], value)["files"]) == [] for value in cache.values())


def _has_pooled_cache(cache: Mapping[str, object]) -> bool:
    files = cast(Sequence[str], cast(Mapping[str, object], cache["tmp"])["files"])
    return any(path.startswith("latent-anything-80-24-") and path.endswith(".npz") for path in files)


def _output_copy(source: Path, destination: Path) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite prior evidence output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    source_inventory = sha256_tree(source)
    destination_inventory = sha256_tree(destination)
    if source_inventory != destination_inventory:
        raise AssertionError(f"copied output differs from the validated fresh-root output: {destination}")
    return {
        "path": destination.relative_to(REPO).as_posix(),
        "files": len(destination_inventory),
        "tree_sha256": hashlib.sha256(
            json.dumps(destination_inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _proof_snippet(case: str, output_location: str) -> str:
    if case == "encoder-v3":
        return (
            "from scripts import sprint80_task80_23_v3_proof as p; "
            f"p.OUTPUT_LOCATION={output_location!r}; p.OUTPUT_ROOT=p.REPO / p.OUTPUT_LOCATION; "
            "raise SystemExit(p.main())"
        )
    if case == "transformer-v1-target-v2":
        return (
            "from scripts import sprint80_task80_24_proof as p; "
            f"p.OUTPUT_LOCATION={output_location!r}; raise SystemExit(p.main())"
        )
    if case == "transformer-stability-supplement":
        return (
            "from scripts import sprint80_task80_24_stability_proof as p; "
            f"p.PROOF_ROOT=p.REPO / {output_location!r}; raise SystemExit(p.main())"
        )
    raise ValueError(f"unknown proof case: {case}")


def _case_output_location(case: str, run_id: str) -> str:
    slug = {
        "encoder-v3": "encoder-v3",
        "transformer-v1-target-v2": "transformer-v1-target-v2",
        "transformer-stability-supplement": "transformer-stability-supplement",
    }[case]
    return f"artifacts/diagnostics/proof-80-25-committed-{slug}-{run_id}"


def default_hf_home() -> Path:
    raw = os.environ.get("HF_HOME")
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "huggingface"


def _execute_case(
    case: str,
    *,
    run_id: str,
    source_hf_home: Path,
    work: Path,
    fresh_root: Path,
    python_exe: Path,
    evidence_dir: Path,
    commands: list[dict[str, object]],
    checks: list[dict[str, object]],
) -> tuple[dict[str, object], str, str]:
    label = case.replace("-", "_")
    cache_home = work / "runtime" / label / "hf-home"
    if case != "encoder-v3":
        cache_seed = seed_hf_snapshot(source_hf_home, cache_home)
    else:
        cache_home.mkdir(parents=True, exist_ok=False)
        cache_seed = {"file_count": 0, "files": [], "verified": {}}
    env = build_child_env(work, label, cache_home)
    initial_cache = _cache_state(work, label)
    if not _cache_empty(initial_cache):
        raise AssertionError(f"proof caches were not fresh before {case}: {initial_cache}")
    output_location = _case_output_location(case, run_id)
    output_root = fresh_root / Path(output_location)
    if output_root.exists():
        raise AssertionError(f"clean proof output path already exists: {output_root}")
    snippet = _proof_snippet(case, output_location)
    command = [str(python_exe), "-c", snippet]
    result = _run_tracked(
        commands,
        command,
        cwd=fresh_root,
        env=env,
        timeout=MAX_PROOF_SECONDS,
        label=case,
        log_dir=evidence_dir / "logs",
    )
    stdout = str(result["text"])
    cache_after = _cache_state(work, label)
    if case == "encoder-v3":
        if _has_pooled_cache(cache_after):
            raise AssertionError("encoder run unexpectedly reused/wrote a transformer pooled-state cache")
        parsed = _encoder_output(output_root)
    elif case == "transformer-v1-target-v2":
        if not _has_pooled_cache(cache_after):
            raise AssertionError("transformer run did not materialize the cold pooled-state cache in isolated TEMP")
        parsed = _transformer_output(output_root, stdout)
    else:
        if not _has_pooled_cache(cache_after):
            raise AssertionError(
                "stability supplement did not materialize its cold pooled-state cache in isolated TEMP"
            )
        parsed = _stability_output(output_root, stdout)
    external_snapshot_post = verify_hf_snapshot(source_hf_home) if case != "encoder-v3" else {}
    fresh_snapshot_post = verify_hf_snapshot(cache_home) if case != "encoder-v3" else {}
    record_check(checks, "accepted_case_output_validated", True, parsed, case=case)
    record_check(checks, "fresh_generated_caches_started_empty", _cache_empty(initial_cache), initial_cache, case=case)
    if case != "encoder-v3":
        record_check(checks, "fresh_hub_snapshot_still_matches_external_pins", True, fresh_snapshot_post, case=case)
        record_check(checks, "external_hub_snapshot_unchanged_after_run", True, external_snapshot_post, case=case)
    execution = {
        **parsed,
        "case": case,
        "cache_seed": cache_seed,
        "cache_files_after": cache_after,
        "external_snapshot_verified_after": external_snapshot_post,
        "fresh_snapshot_verified_after": fresh_snapshot_post,
        "output_location": output_location,
        "peak_process_tree_rss_bytes": result["peak_process_tree_rss_bytes"],
        "timeout_s": result["timeout_s"],
        "wall_s": result["wall_s"],
    }
    return execution, output_location, stdout


def _environment_record(
    *,
    fresh_root: Path,
    python_exe: Path,
    commands: list[dict[str, object]],
    isolation: Mapping[str, object],
    versions: Mapping[str, str],
    uv_version: str,
    copy_record: Mapping[str, object],
    hf_home: Path,
) -> dict[str, object]:
    return {
        "clean_install": dict(isolation),
        "clean_root": str(fresh_root),
        "copy": {"count": copy_record["count"], "bytes": copy_record["bytes"]},
        "dependency_versions": dict(versions),
        "editable_install": False,
        "external_hf_home": str(hf_home),
        "external_environment_tokens_passed": False,
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version,
            "processor": platform.processor(),
            "uv": uv_version,
        },
        "fresh_python": str(python_exe),
        "runtime_cache_policy": (
            "fresh per-proof TEMP, datasets, modules, and HF hub cache seeded from pinned snapshot bytes"
        ),
        "uv_package_cache": "uv's normal package download cache; not used for model/data state",
        "commands": [dict(command) for command in commands],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="verify pins and build an isolated non-editable environment only"
    )
    parser.add_argument(
        "--hf-root", type=Path, default=None, help="immutable pinned Hugging Face snapshot root (default: HF_HOME)"
    )
    parser.add_argument(
        "--keep-work", action="store_true", help="retain the temporary clean root and caches for inspection"
    )
    parser.add_argument(
        "--evidence-dir", type=Path, default=None, help="new evidence directory (must not already exist)"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    started_at_utc = datetime.now(UTC)
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f") + f"-{os.getpid()}"
    hf_home = (args.hf_root or default_hf_home()).resolve()
    evidence_dir = (args.evidence_dir or REPO / "artifacts" / f"task_80.25_evidence_committed_{run_id}").resolve()
    if not args.dry_run and evidence_dir.exists():
        raise FileExistsError(f"evidence directory already exists: {evidence_dir}")

    commands: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    case_results: list[dict[str, object]] = []
    work: Path | None = None
    fresh_root: Path | None = None
    error: str | None = None
    environment: dict[str, object] = {}
    cleanup: dict[str, object] = {"removed": False, "kept_by_flag": args.keep_work}
    baseline_inventory: dict[str, str] = {}
    inventory_delta: dict[str, list[str]] = {"added": [], "removed": [], "modified": []}
    pre_pins: dict[str, str] = {}
    post_pins: dict[str, str] = {}
    external_pre: dict[str, dict[str, object]] = {}
    external_post: dict[str, dict[str, object]] = {}
    copy_record: dict[str, object] = {}
    isolation: dict[str, object] = {}
    package_versions: dict[str, str] = {}
    uv_version = ""

    if args.dry_run:
        work = Path(tempfile.mkdtemp(prefix="sprint80-80-25-dry-"))
        evidence_dir = work / "evidence"
    else:
        evidence_dir.mkdir(parents=True, exist_ok=False)
        (evidence_dir / "logs").mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f"sprint80-80-25-{run_id}-"))

    try:
        say(f"preflight run_id={run_id} hf_home={hf_home} clean_driver_sha256={sha256_file(Path(__file__).resolve())}")
        pre_pins = verify_pinned_inputs(REPO)
        record_check(
            checks,
            "accepted_repository_inputs_verified_pre",
            len(pre_pins) == len(PINNED_INPUT_SHA256) + 1,
            {"count": len(pre_pins), "hashes": pre_pins},
        )
        external_pre = verify_hf_snapshot(hf_home)
        record_check(checks, "immutable_external_snapshots_verified_pre", True, external_pre)
        for directory in (
            REPO / "artifacts/task_80.25_evidence",
            REPO / "artifacts/task_80.25_evidence_attempt1_failed",
        ):
            if not directory.is_dir():
                raise AssertionError(f"historical negative-evidence directory missing: {directory}")
        record_check(
            checks,
            "historical_failed_v1_evidence_preserved",
            True,
            {
                "paths": ["artifacts/task_80.25_evidence", "artifacts/task_80.25_evidence_attempt1_failed"],
                "summary_sha256": {
                    path: pre_pins[path]
                    for path in (
                        "artifacts/task_80.25_evidence/summary.json",
                        "artifacts/task_80.25_evidence_attempt1_failed/summary.json",
                    )
                },
            },
        )
        free_bytes = shutil.disk_usage(tempfile.gettempdir()).free
        record_check(
            checks, "disk_headroom_at_least_12_gib", free_bytes >= MIN_FREE_DISK_BYTES, {"free_bytes": free_bytes}
        )
        uv = _run_tracked(
            commands,
            ["uv", "--version"],
            cwd=REPO,
            env={key: value for key, value in os.environ.items() if key not in STRIP_ENV_NAMES},
            timeout=60,
            label="uv-version",
            log_dir=evidence_dir / "logs",
        )
        uv_version = str(uv["text"]).strip()
        assert work is not None
        fresh_root = work / "root"
        fresh_root.mkdir()
        copy_record = copy_fresh_root(REPO, fresh_root)
        fresh_pins = verify_pinned_inputs(fresh_root)
        record_check(
            checks,
            "fresh_root_copied_all_pinned_inputs",
            fresh_pins == pre_pins,
            {"copied_files": copy_record["count"], "bytes": copy_record["bytes"], "pins": len(fresh_pins)},
        )
        baseline_inventory = sha256_tree(fresh_root)

        install_label = "install"
        install_cache = work / "runtime" / install_label / "hf-home"
        install_env = build_child_env(work, install_label, install_cache)
        _run_tracked(
            commands,
            ["uv", "sync", "--frozen", "--no-dev", "--extra", "transformers", "--no-editable"],
            cwd=fresh_root,
            env=install_env,
            timeout=1800,
            label="uv-sync-non-editable",
            log_dir=evidence_dir / "logs",
        )
        python_exe = fresh_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python_exe.is_file():
            raise AssertionError(f"fresh locked interpreter missing: {python_exe}")
        before_overlay_result = _run_tracked(
            commands,
            ["uv", "pip", "list", "--python", str(python_exe), "--format", "json"],
            cwd=fresh_root,
            env=install_env,
            timeout=300,
            label="packages-before-overlay",
            log_dir=evidence_dir / "logs",
        )
        before_overlay = _parse_uv_list(str(before_overlay_result["text"]))
        _run_tracked(
            commands,
            ["uv", "pip", "install", "--python", str(python_exe), *OVERLAY_PACKAGES],
            cwd=fresh_root,
            env=install_env,
            timeout=900,
            label="declared-datasets-overlay",
            log_dir=evidence_dir / "logs",
        )
        pip_check = _run_tracked(
            commands,
            ["uv", "pip", "check", "--python", str(python_exe)],
            cwd=fresh_root,
            env=install_env,
            timeout=300,
            label="uv-pip-check",
            log_dir=evidence_dir / "logs",
        )
        after_overlay_result = _run_tracked(
            commands,
            ["uv", "pip", "list", "--python", str(python_exe), "--format", "json"],
            cwd=fresh_root,
            env=install_env,
            timeout=300,
            label="packages-after-overlay",
            log_dir=evidence_dir / "logs",
        )
        package_versions = _parse_uv_list(str(after_overlay_result["text"]))
        for name, expected in PINNED_VERSIONS.items():
            if package_versions.get(name) != expected:
                raise AssertionError(
                    f"clean package version mismatch: {name} {package_versions.get(name)} != {expected}"
                )
        changed_locked = sorted(name for name in before_overlay if package_versions.get(name) != before_overlay[name])
        disallowed = sorted(set(changed_locked) - set(OVERLAY_ALLOWED_ADJUSTMENTS))
        adjustment_failures = {
            name: {"actual": package_versions.get(name), "expected": expected}
            for name, expected in OVERLAY_ALLOWED_ADJUSTMENTS.items()
            if package_versions.get(name) != expected
        }
        record_check(
            checks,
            "overlay_changes_only_declared_locked_adjustments",
            not disallowed and not adjustment_failures,
            {
                "changed_locked_packages": changed_locked,
                "disallowed": disallowed,
                "adjustment_failures": adjustment_failures,
            },
        )
        record_check(
            checks,
            "uv_pip_check_clean",
            "All installed packages are compatible" in str(pip_check["text"]),
            str(pip_check["text"]).strip(),
        )
        isolation = _isolation_probe(
            python_exe,
            fresh_root,
            commands=commands,
            env=install_env,
            log_dir=evidence_dir / "logs",
            label="non-editable-install-isolation-probe",
        )
        record_check(checks, "non_editable_install_import_isolated", not bool(isolation["editable"]), isolation)
        if args.dry_run:
            delta = diff_inventories(baseline_inventory, sha256_tree(fresh_root))
            unexpected = (
                [path for path in delta["added"] if not _is_build_metadata(path)] + delta["removed"] + delta["modified"]
            )
            record_check(checks, "dry_run_tree_unchanged_except_build_metadata", not unexpected, delta)
            say(f"dry-run passed: {len(checks)} checks; work={work}")
            return 0

        current_source_hashes = {
            relative: pre_pins[relative]
            for relative in PINNED_INPUT_SHA256
            if relative.startswith("scripts/") or relative.startswith("src/")
        }
        environment = _environment_record(
            fresh_root=fresh_root,
            python_exe=python_exe,
            commands=commands,
            isolation=isolation,
            versions=package_versions,
            uv_version=uv_version,
            copy_record=copy_record,
            hf_home=hf_home,
        )
        environment["source_hashes_used"] = current_source_hashes
        environment["historical_temporary_driver_source_hashes"] = HISTORICAL_TEMP_DRIVER_SOURCE_SHA256

        output_locations: list[str] = []
        for case in ("encoder-v3", "transformer-v1-target-v2", "transformer-stability-supplement"):
            execution, output_location, stdout = _execute_case(
                case,
                run_id=run_id,
                source_hf_home=hf_home,
                work=work,
                fresh_root=fresh_root,
                python_exe=python_exe,
                evidence_dir=evidence_dir,
                commands=commands,
                checks=checks,
            )
            case_results.append(execution)
            output_locations.append(output_location)
            say(f"validated {case}: {json.dumps(execution, ensure_ascii=False, sort_keys=True)}")

        final_inventory = sha256_tree(fresh_root)
        inventory_delta = diff_inventories(baseline_inventory, final_inventory)
        fresh_post_pins = verify_pinned_inputs(fresh_root)
        unexpected_added = [
            path
            for path in inventory_delta["added"]
            if not _is_build_metadata(path) and not _inside_any(path, output_locations)
        ]
        unexpected_removed = inventory_delta["removed"]
        unexpected_modified = inventory_delta["modified"]
        record_check(
            checks,
            "fresh_root_only_gained_declared_output_roots_and_build_metadata",
            not unexpected_added and not unexpected_removed and not unexpected_modified,
            {"delta": inventory_delta, "unexpected_added": unexpected_added},
        )
        record_check(
            checks,
            "historical_negative_evidence_unchanged_in_fresh_root",
            all(
                not _inside_any(path, output_locations)
                for path in inventory_delta["modified"] + inventory_delta["removed"]
            ),
            {
                "historical_summaries": {
                    path: fresh_post_pins[path]
                    for path in (
                        "artifacts/task_80.25_evidence/summary.json",
                        "artifacts/task_80.25_evidence_attempt1_failed/summary.json",
                    )
                }
            },
        )

        # Preserve validated outputs byte-for-byte at their matching relative
        # paths. Each destination is unique and overwrite is refused.
        for output_location in output_locations:
            copied = _output_copy(fresh_root / Path(output_location), REPO / Path(output_location))
            outputs.append(copied)
        record_check(checks, "validated_outputs_copied_byte_exact", len(outputs) == len(case_results), outputs)
        post_pins = verify_pinned_inputs(REPO)
        external_post = verify_hf_snapshot(hf_home)
        record_check(
            checks,
            "accepted_repository_inputs_verified_post",
            post_pins == pre_pins,
            {"count": len(post_pins), "hashes": post_pins},
        )
        record_check(
            checks, "fresh_root_pins_verified_post", fresh_post_pins == pre_pins, {"count": len(fresh_post_pins)}
        )
        record_check(checks, "immutable_external_snapshots_verified_post", external_post == external_pre, external_post)

        environment["commands"] = [
            {key: value for key, value in command.items() if key != "text"} for command in commands
        ]
        environment["repository_pins_pre"] = pre_pins
        environment["repository_pins_post"] = post_pins
        environment["external_snapshots_pre"] = external_pre
        environment["external_snapshots_post"] = external_post
        environment["packages_before_overlay"] = before_overlay
        environment["overlay_changes"] = {
            "changed_locked_packages": changed_locked,
            "expected_adjustments": OVERLAY_ALLOWED_ADJUSTMENTS,
        }
        environment["pip_check"] = str(pip_check["text"]).strip()
        environment["copy_inventory"] = copy_record
        environment["fresh_root_inventory_delta"] = inventory_delta

    except Exception as exc:  # retain a truthful evidence record on any fail-closed outcome
        error = f"{type(exc).__name__}: {exc}"
        say(f"FAILED: {error}")
    finally:
        if work is not None:
            if args.keep_work:
                cleanup = {"work": str(work), "removed": False, "kept_by_flag": True}
            else:
                shutil.rmtree(work, ignore_errors=True)
                cleanup = {"work": str(work), "removed": not work.exists(), "kept_by_flag": False}
        if not args.dry_run:
            failed_checks = [item for item in checks if not item["ok"]]
            summary = {
                "acceptance": "PASSED" if error is None and not failed_checks else "NOT PASSED",
                "case_results": case_results,
                "checks": checks,
                "checks_failed": len(failed_checks),
                "checks_total": len(checks),
                "cleanup": cleanup,
                "commands": [{key: value for key, value in command.items() if key != "text"} for command in commands],
                "driver_sha256": sha256_file(Path(__file__).resolve()),
                "error": error,
                "external_snapshots_pre": external_pre,
                "external_snapshots_post": external_post,
                "fresh_root_inventory_delta": inventory_delta,
                "historical_negative_evidence_preserved": all(
                    (REPO / path).is_file() and sha256_file(REPO / path) == expected
                    for path, expected in (
                        (
                            "artifacts/task_80.25_evidence/summary.json",
                            PINNED_INPUT_SHA256["artifacts/task_80.25_evidence/summary.json"],
                        ),
                        (
                            "artifacts/task_80.25_evidence_attempt1_failed/summary.json",
                            PINNED_INPUT_SHA256["artifacts/task_80.25_evidence_attempt1_failed/summary.json"],
                        ),
                    )
                ),
                "outputs": outputs,
                "pinned_repository_inputs_pre": pre_pins,
                "pinned_repository_inputs_post": post_pins,
                "resource_bounds": {
                    "max_proof_seconds": MAX_PROOF_SECONDS,
                    "max_process_tree_rss_bytes": MAX_PROCESS_TREE_RSS_BYTES,
                },
                "run_id": run_id,
                "source_drift_chronology": {
                    "prior_temporary_driver_run": (
                        "2026-09-24 accepted replay, executed through uncommitted "
                        "inline python -c; retained as historical evidence only"
                    ),
                    "prior_source_sha256": HISTORICAL_TEMP_DRIVER_SOURCE_SHA256,
                    "committed_driver_source_sha256_at_replay": {
                        path: pre_pins.get(path)
                        for path in (
                            "scripts/sprint80_task80_23_v3_proof.py",
                            "scripts/sprint80_task80_24_proof.py",
                            "scripts/sprint80_task80_24_stability_proof.py",
                        )
                    },
                    "interpretation": (
                        "the new committed-driver replay uses currently pinned "
                        "entrypoint bytes; earlier output hashes do not establish this replay"
                    ),
                },
                "started_at_utc": started_at_utc.isoformat(),
                "status": "PASSED" if error is None and not failed_checks else "NOT PASSED",
                "task": "80.25",
                "total_wall_s": round(time.perf_counter() - started, 3),
            }
            if environment:
                write_json(evidence_dir / "environment.json", environment)
            if baseline_inventory:
                write_json(evidence_dir / "inventory_baseline.json", baseline_inventory)
                write_json(evidence_dir / "inventory_delta.json", inventory_delta)
            write_json(evidence_dir / "runs.json", case_results)
            write_json(evidence_dir / "checks.json", checks)
            write_json(
                evidence_dir / "pins.json",
                {
                    "repository_inputs": pre_pins,
                    "external_snapshots": external_pre,
                    "historical_temporary_driver_source_sha256": HISTORICAL_TEMP_DRIVER_SOURCE_SHA256,
                },
            )
            write_json(evidence_dir / "summary.json", summary)
    if args.dry_run:
        return 0 if error is None else 1
    say(
        f"status={summary['status']} checks={summary['checks_total']} "
        f"failed={summary['checks_failed']} wall={summary['total_wall_s']}s "
        f"evidence={evidence_dir}"
    )
    return 0 if summary["status"] == "PASSED" and cleanup.get("removed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
