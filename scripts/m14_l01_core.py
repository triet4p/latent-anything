"""Produce the M14 L01 D2 evidence artifact for core latent contracts.

The runner deliberately uses the existing ConvVAE adapter and AnalysisPipeline
on the real sklearn digits data.  It records a train-only fit, a deterministic
held-out reconstruction score, and the LatentSpace/LatentValue invariants that
the L01 gap requires.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.adapters.conv_vae import ConvVAE
from latent_anything.analysis_pipeline import AnalysisPipeline
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.methods import PCA
from latent_anything.pipeline_models import PipelineResult

SEED = 42
TRAIN_FRACTION = 0.8
LATENT_DIM = 4
EPOCHS = 8
MIN_ZERO_BASELINE_IMPROVEMENT = 0.10
ARTIFACT_NAME = "l01-core.json"
RUN_RECORD_NAME = "l01-core.run.json"

_REQUIRED_FIELDS = (
    "schema_version",
    "lane",
    "capability_id",
    "evidence_level",
    "seed",
    "dataset",
    "split",
    "model",
    "backend",
    "contracts",
    "metrics",
    "controls",
    "acceptance",
    "provenance",
    "artifact_sha256",
)


class _DigitsDataset(Protocol):
    images: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class _BenchmarkResult:
    """Private typed hand-off between execution, payload, and output stages."""

    images: np.ndarray
    labels: np.ndarray
    train: np.ndarray
    heldout: np.ndarray
    train_labels: np.ndarray
    heldout_labels: np.ndarray
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    train_latents: np.ndarray
    heldout_latents: np.ndarray
    heldout_pca: np.ndarray
    input_digest_before: str
    input_digest_after: str
    space: LatentSpace
    train_value: LatentValue
    heldout_value: LatentValue
    pipeline_result: PipelineResult
    metrics: dict[str, float]
    acceptance: dict[str, bool]


def split_digits(images: np.ndarray, labels: np.ndarray, *, seed: int = SEED) -> tuple[np.ndarray, ...]:
    """Return disjoint deterministic train/held-out arrays and source indices."""

    if images.ndim != 4 or images.shape[1:] != (1, 8, 8):
        raise ValueError(f"images must have shape (n, 1, 8, 8), got {images.shape}")
    if labels.ndim != 1 or len(images) != len(labels):
        raise ValueError("labels must be a matching one-dimensional array")
    permutation = np.random.default_rng(seed).permutation(len(images))
    split = int(len(images) * TRAIN_FRACTION)
    train_indices, heldout_indices = permutation[:split], permutation[split:]
    return (
        images[train_indices],
        images[heldout_indices],
        labels[train_indices],
        labels[heldout_indices],
        train_indices,
        heldout_indices,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    """Hash dtype, shape, and bytes for exact dataset/split provenance."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _payload_digest(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    return _sha256_bytes(_canonical_bytes(unsigned))


def _git_sha() -> str:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runner_source_digest() -> str:
    """Hash the exact runner source when the lane is executed pre-commit."""

    return _sha256_bytes(Path(__file__).read_bytes())


def _version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "unavailable"


def validate_l01_artifact(payload: Mapping[str, object]) -> list[str]:
    """Return schema and digest errors without mutating or executing a run."""

    errors = [f"missing required field: {field}" for field in _REQUIRED_FIELDS if field not in payload]
    if errors:
        return errors
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("lane") != "M14-L01":
        errors.append("lane must be M14-L01")
    if payload.get("capability_id") != "THY-T01-METRIC-SPACE-VA-VECTOR-SPACE":
        errors.append("capability_id does not identify the L01 gap")
    if payload.get("evidence_level") != "D2":
        errors.append("accepted L01 evidence must be D2")
    artifact_sha = payload.get("artifact_sha256")
    if not isinstance(artifact_sha, str) or artifact_sha != _payload_digest(payload):
        errors.append("artifact_sha256 does not match canonical unsigned payload")
    return errors


def _build_acceptance(
    *,
    train: np.ndarray,
    heldout: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    train_latents: np.ndarray,
    heldout_latents: np.ndarray,
    train_value: LatentValue,
    heldout_value: LatentValue,
    pipeline_result: PipelineResult,
    heldout_pca: np.ndarray,
    input_digest_before: str,
    input_digest_after: str,
    metrics: Mapping[str, float],
    midpoint: np.ndarray,
) -> dict[str, bool]:
    """Evaluate predeclared L01 controls independently of payload writing."""

    arithmetic_input = heldout_value[0:1].to_numpy()
    arithmetic_result = heldout_value[0:1].add_scaled(heldout_value[1:2], 0.25)
    distance = metrics["latent_distance_first_pair"]
    finite_metrics = bool(np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all())
    return {
        "finite_metrics": finite_metrics,
        "heldout_beats_zero_baseline_sanity": (
            metrics["improvement_over_zero_baseline"] >= MIN_ZERO_BASELINE_IMPROVEMENT
        ),
        "train_mean_baseline_is_stronger_diagnostic": (
            metrics["train_mean_baseline_mse"] < metrics["heldout_reconstruction_mse"]
        ),
        "split_is_disjoint": bool(np.intersect1d(train_indices, heldout_indices).size == 0),
        "input_not_mutated": input_digest_before == input_digest_after,
        "latent_shapes_dtypes_finite": (
            train_latents.shape == (len(train), LATENT_DIM)
            and heldout_latents.shape == (len(heldout), LATENT_DIM)
            and train_latents.dtype == np.float64
            and heldout_latents.dtype == np.float64
            and bool(np.isfinite(train_latents).all())
            and bool(np.isfinite(heldout_latents).all())
        ),
        "space_operations_finite_and_shape_safe": (
            np.isfinite(distance)
            and distance > 0.0
            and midpoint.shape == (LATENT_DIM,)
            and midpoint.dtype == np.float64
            and bool(np.isfinite(midpoint).all())
        ),
        "latent_value_immutable_arithmetic": (
            train_value.shape == (len(train), LATENT_DIM)
            and heldout_value.shape == (len(heldout), LATENT_DIM)
            and arithmetic_result.shape == (1, LATENT_DIM)
            and arithmetic_result.to_numpy().dtype == np.float64
            and bool(np.array_equal(arithmetic_input, heldout_value[0:1].to_numpy()))
            and bool(np.isfinite(arithmetic_result.to_numpy()).all())
        ),
        "pipeline_result_contract": (
            type(pipeline_result) is PipelineResult
            and pipeline_result.latents.shape == (len(train), LATENT_DIM)
            and pipeline_result.transformed.shape == (len(train), 2)
            and heldout_pca.shape == (len(heldout), 2)
            and pipeline_result.latent_space.dim == LATENT_DIM
            and bool(np.isfinite(pipeline_result.transformed).all())
            and bool(np.isfinite(heldout_pca).all())
        ),
    }


def _run_benchmark(git_sha: str) -> _BenchmarkResult:
    """Execute the real-data model, pipeline, and latent contract checks."""

    digits = cast(_DigitsDataset, load_digits())
    images = (digits.images / 16.0).astype(np.float64)[:, None, :, :]
    labels = digits.target.astype(np.int64)
    train, heldout, train_labels, heldout_labels, train_indices, heldout_indices = split_digits(images, labels)
    if np.intersect1d(train_indices, heldout_indices).size != 0:
        raise RuntimeError("train and held-out indices overlap")
    input_digest_before = _array_digest(train, heldout, train_labels, heldout_labels)

    adapter = ConvVAE(latent_dim=LATENT_DIM, random_state=SEED, n_epochs=EPOCHS)
    adapter.fit(train)
    train_latents = adapter.encode(train)
    heldout_latents = adapter.encode(heldout)
    heldout_reconstruction = adapter.decode(heldout_latents)
    zero_mse = float(np.mean((heldout - np.zeros_like(heldout)) ** 2))
    train_mean_mse = float(
        np.mean((heldout - np.broadcast_to(np.mean(train, axis=0, keepdims=True), heldout.shape)) ** 2)
    )
    heldout_mse = float(np.mean((heldout - heldout_reconstruction) ** 2))
    metrics: dict[str, float] = {
        "heldout_reconstruction_mse": heldout_mse,
        "zero_baseline_mse": zero_mse,
        "train_mean_baseline_mse": train_mean_mse,
        "improvement_over_zero_baseline": float(1.0 - heldout_mse / zero_mse),
        "latent_utilization_train": float(adapter.metrics_["latent_utilization"]),
    }

    # Fit the existing pipeline only on train representations; held-out PCA is
    # transformed after fitting and is never used to fit the method.
    pipeline = AnalysisPipeline(adapter=adapter, method=PCA(n_components=2))
    pipeline_result = pipeline.run(train)
    heldout_pca = pipeline.method.transform(heldout_latents)
    space = adapter.latent_space
    value_metadata = {"source_representation_identity": "sklearn-digits::convvae", "revision": git_sha}
    train_value = LatentValue(train_latents, space, metadata=value_metadata)
    heldout_value = LatentValue(heldout_latents, space, metadata=value_metadata)
    distance = space.distance(heldout_latents[0], heldout_latents[1])
    midpoint = space.interpolate(heldout_latents[0], heldout_latents[1], 0.5)
    metrics["latent_distance_first_pair"] = distance
    input_digest_after = _array_digest(train, heldout, train_labels, heldout_labels)
    acceptance = _build_acceptance(
        train=train,
        heldout=heldout,
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        train_latents=train_latents,
        heldout_latents=heldout_latents,
        train_value=train_value,
        heldout_value=heldout_value,
        pipeline_result=pipeline_result,
        heldout_pca=heldout_pca,
        input_digest_before=input_digest_before,
        input_digest_after=input_digest_after,
        metrics=metrics,
        midpoint=midpoint,
    )
    return _BenchmarkResult(
        images=images,
        labels=labels,
        train=train,
        heldout=heldout,
        train_labels=train_labels,
        heldout_labels=heldout_labels,
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        train_latents=train_latents,
        heldout_latents=heldout_latents,
        heldout_pca=heldout_pca,
        input_digest_before=input_digest_before,
        input_digest_after=input_digest_after,
        space=space,
        train_value=train_value,
        heldout_value=heldout_value,
        pipeline_result=pipeline_result,
        metrics=metrics,
        acceptance=acceptance,
    )


def _build_payload(result: _BenchmarkResult, git_sha: str) -> dict[str, object]:
    """Construct the stable unsigned evidence payload from one run result."""

    accepted = all(result.acceptance.values())
    return {
        "schema_version": 1,
        "lane": "M14-L01",
        "capability_id": "THY-T01-METRIC-SPACE-VA-VECTOR-SPACE",
        "evidence_level": "D2" if accepted else "D1",
        "claim_scope": (
            "D2 verifies real held-out metric/vector/pipeline contract evidence; "
            "it does not claim model-quality superiority."
        ),
        "seed": SEED,
        "dataset": {
            "name": "sklearn.datasets.load_digits",
            "package_revision": f"scikit-learn=={_version('scikit-learn')}",
            "license": "BSD-3-Clause (scikit-learn bundled digits dataset)",
            "samples": len(result.images),
            "raw_shape": list(result.images.shape),
            "content_sha256": _array_digest(result.images, result.labels),
        },
        "split": {
            "algorithm": "numpy.default_rng(seed).permutation; first floor(80%) train, remainder held-out",
            "train_fraction": TRAIN_FRACTION,
            "train_samples": len(result.train),
            "heldout_samples": len(result.heldout),
            "train_index_digest": _array_digest(result.train_indices),
            "heldout_index_digest": _array_digest(result.heldout_indices),
            "train_indices": result.train_indices.tolist(),
            "heldout_indices": result.heldout_indices.tolist(),
            "heldout_label_counts": np.bincount(result.heldout_labels, minlength=10).tolist(),
        },
        "model": {
            "adapter": "latent_anything.adapters.conv_vae.ConvVAE",
            "architecture": "8x8 grayscale Conv2d(1,4,3,padding=1)-ReLU-Flatten -> mu/logvar -> Linear(4,64)-Sigmoid",
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "fit_partition": "train_only",
            "random_state": SEED,
            "source_revision": git_sha,
        },
        "backend": {
            "device": "cpu",
            "numpy": _version("numpy"),
            "scikit_learn": _version("scikit-learn"),
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        },
        "contracts": {
            "latent_space": {
                "geometry": result.space.geometry,
                "dim": result.space.dim,
                "source_model": result.space.source_model,
            },
            "latent_value": {
                "train_shape": list(result.train_value.shape),
                "heldout_shape": list(result.heldout_value.shape),
            },
            "pipeline": {"type": "AnalysisPipeline", "protocol": "PipelineContract"},
            "pipeline_result": {
                "type": "PipelineResult",
                "train_latents_shape": list(result.pipeline_result.latents.shape),
                "train_transformed_shape": list(result.pipeline_result.transformed.shape),
                "heldout_transformed_shape": list(result.heldout_pca.shape),
            },
        },
        "metrics": {
            **result.metrics,
            "interpolation_midpoint_l2_norm": float(
                np.linalg.norm(result.space.interpolate(result.heldout_latents[0], result.heldout_latents[1], 0.5))
            ),
        },
        "controls": {
            "input_digest_before": result.input_digest_before,
            "input_digest_after": result.input_digest_after,
            "input_digest_algorithm": "sha256(dtype + shape + contiguous bytes) for train/heldout images and labels",
            "train_vs_heldout_fit_separation": True,
            "no_network": True,
            "no_mock_or_synthetic_data": True,
            "strong_baseline_is_diagnostic_only": True,
        },
        "thresholds": {"min_zero_baseline_improvement_sanity": MIN_ZERO_BASELINE_IMPROVEMENT},
        "acceptance": result.acceptance,
        "accepted": accepted,
        "provenance": {
            "git_sha": git_sha,
            "runner": "scripts/m14_l01_core.py",
            "runner_source_sha256": _runner_source_digest(),
            "command": "uv run python scripts/m14_l01_core.py",
            "environment": {"platform": platform.platform(), "network": "offline"},
            "license_access": "scikit-learn BSD-3-Clause; bundled digits data; no model download or credentials",
            "credentials": "none",
            "resource_peak": "not measured; M14 estimate only",
            "cleanup": "no persistent temporary files; output artifact and run record retained",
        },
    }


def _write_outputs(payload: dict[str, object], output_dir: Path, accepted: bool) -> None:
    """Persist the artifact and its immutable time-bearing run record."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / ARTIFACT_NAME
    artifact_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    run_record = {
        "schema_version": 1,
        "lane": "M14-L01",
        "artifact": str(artifact_path).replace("\\", "/"),
        "artifact_sha256": payload["artifact_sha256"],
        "runner": "scripts/m14_l01_core.py",
        "command": "uv run python scripts/m14_l01_core.py",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "accepted" if accepted else "failed",
        "resource_peak": "not measured; M14 estimate only",
        "cleanup": "completed; no temporary files created",
    }
    (output_dir / RUN_RECORD_NAME).write_text(
        json.dumps(run_record, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def main(output_dir: Path = Path("artifacts/m14")) -> dict[str, object]:
    """Run the bounded CPU benchmark and write the artifact plus run record."""

    git_sha = _git_sha()
    result = _run_benchmark(git_sha)
    payload = _build_payload(result, git_sha)
    payload["artifact_sha256"] = _payload_digest(payload)
    errors = validate_l01_artifact(payload)
    if errors:
        raise RuntimeError(f"L01 artifact validation failed: {errors}")
    _write_outputs(payload, output_dir, all(result.acceptance.values()))
    if not all(result.acceptance.values()):
        raise RuntimeError(f"L01 acceptance failed: {result.acceptance}")
    return payload


if __name__ == "__main__":
    main()
