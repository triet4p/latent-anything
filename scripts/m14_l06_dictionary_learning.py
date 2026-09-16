"""Run the bounded M14 L06 dictionary-learning comparison."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from latent_anything.dictionary_learning import DictionaryLearning, DictionaryLearningConfig

RECORD_ID = "THY-T05-DICTIONARY-LEARNING"
SEED = 79
N_SAMPLES = 600
N_FEATURES = 12
N_COMPONENTS = 8
TRAIN_FRACTION = 0.8
ARTIFACT_PATH = Path("artifacts/m14/l06-dictionary-learning.json")
RUN_RECORD_PATH = Path("artifacts/m14/l06-dictionary-learning.run.json")


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_fixture() -> tuple[np.ndarray, str]:
    """Create the deterministic bounded sparse dictionary corpus."""
    rng = np.random.default_rng(SEED)
    dictionary = rng.normal(size=(N_COMPONENTS, N_FEATURES))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    codes = np.zeros((N_SAMPLES, N_COMPONENTS), dtype=np.float64)
    for row in codes:
        active = rng.choice(N_COMPONENTS, size=2, replace=False)
        row[active] = rng.uniform(0.4, 1.0, size=2)
    data = codes @ dictionary + rng.normal(scale=0.01, size=(N_SAMPLES, N_FEATURES))
    return data, _array_digest(data)


def run_benchmark() -> dict[str, Any]:
    data, data_digest = build_fixture()
    before = data.copy()
    config = DictionaryLearningConfig(
        n_components=N_COMPONENTS,
        alpha=0.05,
        max_iter=500,
        tol=1e-6,
        random_state=SEED,
        val_fraction=1.0 - TRAIN_FRACTION,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=2,
    )
    learner = DictionaryLearning(config)
    evaluation = learner.fit(data)
    assert learner.train_indices_ is not None
    assert learner.validation_indices_ is not None
    after_digest = _array_digest(data)
    checks = {
        "finite_metrics": bool(
            np.isfinite(
                [
                    evaluation.train_reconstruction_mse,
                    evaluation.val_reconstruction_mse,
                    evaluation.train_mean_l0,
                    evaluation.val_mean_l0,
                ]
            ).all()
        ),
        "heldout_reconstruction_beats_mean_baseline": evaluation.val_reconstruction_mse
        < evaluation.val_baseline_mse * 0.5,
        "no_input_mutation": data_digest == after_digest,
        "train_heldout_disjoint": bool(
            set(learner.train_indices_.tolist()).isdisjoint(set(learner.validation_indices_.tolist()))
        ),
        "dictionary_shape": learner.components_.shape == (N_COMPONENTS, N_FEATURES),
    }
    accepted = all(checks.values())
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload: dict[str, Any] = {
        "schema_version": "m14-l06-dictionary-learning-v1",
        "lane": "M14-L06",
        "capability_id": RECORD_ID,
        "evidence_level": "D2",
        "accepted_gap_ids": [RECORD_ID] if accepted else [],
        "accepted": accepted,
        "seed": SEED,
        "dataset": {
            "name": "deterministic sparse dictionary fixture",
            "revision": "sparse-dictionary-v1",
            "license": "original deterministic fixture",
            "samples": N_SAMPLES,
            "features": N_FEATURES,
            "content_sha256": data_digest,
        },
        "split": {
            "algorithm": "numpy.default_rng(seed).permutation; first floor(80%) train, remainder held-out",
            "train_fraction": TRAIN_FRACTION,
            "train_samples": evaluation.n_train,
            "heldout_samples": evaluation.n_val,
            "train_index_sha256": _array_digest(learner.train_indices_),
            "heldout_index_sha256": _array_digest(learner.validation_indices_),
        },
        "model": {
            "implementation": "latent_anything.dictionary_learning.DictionaryLearning",
            "backend": "scikit-learn.DictionaryLearning",
            "config": {
                "n_components": N_COMPONENTS,
                "alpha": 0.05,
                "max_iter": 500,
                "tol": 1e-6,
                "transform_algorithm": "omp",
                "transform_n_nonzero_coefs": 2,
            },
        },
        "metrics": {
            "train_reconstruction_mse": evaluation.train_reconstruction_mse,
            "heldout_reconstruction_mse": evaluation.val_reconstruction_mse,
            "train_mean_l0": evaluation.train_mean_l0,
            "heldout_mean_l0": evaluation.val_mean_l0,
            "train_mean_baseline_mse": evaluation.train_baseline_mse,
            "heldout_mean_baseline_mse": evaluation.val_baseline_mse,
            "dictionary_shape": list(learner.components_.shape),
        },
        "thresholds": {
            "heldout_reconstruction_relative_to_mean_baseline_strict_lt": 0.5,
            "heldout_mean_l0_max": 2.0,
        },
        "controls": checks,
        "provenance": {
            "source_sha": source_sha,
            "runner": "scripts/m14_l06_dictionary_learning.py",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "network_policy": "offline",
            "cleanup": "no temporary files created",
        },
        "artifact_sha256": "",
    }
    payload["artifact_sha256"] = _digest(payload)
    return payload


def main() -> None:
    payload = run_benchmark()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_record = {
        "schema_version": "m14-l06-dictionary-learning-run-v1",
        "lane": "M14-L06",
        "record_id": RECORD_ID,
        "artifact": str(ARTIFACT_PATH).replace("\\", "/"),
        "artifact_sha256": payload["artifact_sha256"],
        "runner": "scripts/m14_l06_dictionary_learning.py",
        "command": "uv run python scripts/m14_l06_dictionary_learning.py",
        "status": "accepted" if payload["accepted"] else "failed",
        "cleanup": payload["provenance"]["cleanup"],
    }
    RUN_RECORD_PATH.write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
