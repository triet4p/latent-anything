"""Persist the M14 L14 bounded tokenized-world-model evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import psutil

from tokenized_world_model_benchmark import main as benchmark_main

RUN_COMMAND = "uv run python scripts/m14_l14_tokenized.py"
EVIDENCE_PATH = Path("artifacts/tokenized_world_model_evidence.json")
CONFIG_PATH = Path("artifacts/tokenized_world_model_evidence_config.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    benchmark_main()
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    process = psutil.Process()
    report = {
        "schema_version": "m14-l14-run-v1",
        "source_sha": source_sha,
        "command": RUN_COMMAND,
        "target": {
            "dataset": evidence["dataset"],
            "dataset_revision": evidence["dataset_revision"],
            "model_revision": "compact-tokenized-world-model-v1",
            "codebook_version": evidence["codebook"]["codebook_version"],
            "license": "BSD-3-Clause (scikit-learn digits)",
            "device": "cpu",
        },
        "split": {
            "train_episodes": 128,
            "train_horizon": 6,
            "heldout_episodes": 32,
            "heldout_horizon": 8,
            "train_seed": 72,
            "heldout_seed": 1702,
        },
        "metrics": {
            "teacher_forced": evidence["teacher_forced"],
            "free_running": evidence["free_running"],
            "codebook": evidence["codebook"],
            "seeded_rollout_bit_exact": evidence["seeded_rollout_bit_exact"],
        },
        "acceptance": {
            "teacher_forced_perplexity_finite": evidence["acceptance"]["teacher_forced_perplexity_finite"],
            "free_running_horizon_complete": evidence["acceptance"]["free_running_horizon_complete"],
            "decoded_consistency_available": evidence["acceptance"]["decoded_consistency_available"],
            "task_proxy_available": evidence["acceptance"]["task_proxy_available"],
            "seeded_rollout_reproducible": evidence["acceptance"]["seeded_rollout_reproducible"],
            "tokenizer_used_for_fit": evidence["acceptance"]["tokenizer_used_for_fit"],
            "heldout_observations_encoded_before_evaluation": evidence["acceptance"][
                "heldout_observations_encoded_before_evaluation"
            ],
            "nontrivial_token_usage": evidence["acceptance"]["nontrivial_token_usage"],
        },
        "thresholds": {
            "teacher_forced_perplexity": "finite",
            "free_running_horizon": 8,
            "codebook_perplexity_min": 1.0,
            "dead_code_rate_max_exclusive": 1.0,
            "active_codes_min": 2,
        },
        "artifacts": [
            {"path": str(EVIDENCE_PATH).replace("\\", "/"), "sha256": _sha256(EVIDENCE_PATH)},
            {"path": str(CONFIG_PATH).replace("\\", "/"), "sha256": _sha256(CONFIG_PATH)},
        ],
        "environment": {
            "device": "cpu",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "rss_peak_bytes": process.memory_info().rss,
            "network_policy": "offline; no model or dataset download",
        },
        "cleanup": (
            "No temporary files; generated evidence/config are retained as immutable "
            "artifacts and no scratch state remains."
        ),
    }
    output = Path("artifacts/m14/l14-tokenized-run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
