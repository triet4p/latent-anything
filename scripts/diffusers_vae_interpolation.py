"""Produce a local-only, revision-pinned Diffusers VAE interpolation artifact.

The script consumes the already verified checkpoint snapshot. It never
downloads model files and denies socket connections while the adapter loads.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]

try:
    import scripts.diffusers_vae_fidelity as fidelity
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    import diffusers_vae_fidelity as fidelity  # type: ignore[import-not-found]
from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter

MODEL_ID = fidelity.MODEL_ID
MODEL_REVISION = fidelity.MODEL_REVISION
SAFE_WEIGHTS_SHA256 = fidelity.SAFE_WEIGHTS_SHA256
SAFE_WEIGHTS_SIZE = fidelity.SAFE_WEIGHTS_SIZE
ARTIFACT_DIR = Path("artifacts")
DEFAULT_JSON = ARTIFACT_DIR / "diffusers_vae_digits_interpolation.json"
DEFAULT_PNG = ARTIFACT_DIR / "diffusers_vae_digits_interpolation.png"
WEIGHTS = np.linspace(0.0, 1.0, 7, dtype=np.float32)
MAX_RUNTIME_SECONDS = 60.0
MAX_PEAK_RSS_BYTES = 2 * 1024**3
MIN_ENDPOINT_LATENT_L2 = 1e-3
MIN_ADJACENT_LATENT_L2 = 1e-4
MIN_ENDPOINT_DECODED_L2 = 1e-3
MIN_ADJACENT_DECODED_L2 = 1e-4
PNG_WIDTH = 2100
PNG_HEIGHT = 360


class _DigitsDataset(Protocol):
    images: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inputs() -> np.ndarray:
    digits = cast(_DigitsDataset, load_digits())
    images = np.stack([digits.images[0], digits.images[1]])
    nchw = np.repeat(np.kron(images, np.ones((4, 4)))[:, None, :, :], 3, axis=1).astype(np.float32)
    return nchw / 8.0 - 1.0


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
    }


def validate_arrays(
    weights: np.ndarray,
    endpoint_latent: np.ndarray,
    interpolation_latent: np.ndarray,
    endpoint_decoded: np.ndarray,
    interpolation_decoded: np.ndarray,
) -> dict[str, float]:
    if weights.shape != (7,) or not np.all(np.diff(weights) > 0) or weights[0] != 0 or weights[-1] != 1:
        raise ValueError("interpolation coefficients must be seven strictly increasing values from 0 to 1")
    arrays = (endpoint_latent, interpolation_latent, endpoint_decoded, interpolation_decoded)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("interpolation arrays must be finite")
    if interpolation_latent.shape[0] != len(weights) or interpolation_decoded.shape[0] != len(weights):
        raise ValueError("interpolation batch length must match coefficient count")
    if not np.array_equal(interpolation_latent[0], endpoint_latent[0]) or not np.array_equal(
        interpolation_latent[-1], endpoint_latent[-1]
    ):
        raise ValueError("latent interpolation endpoints are out of order or not preserved")
    if endpoint_decoded.shape != (2, *interpolation_decoded.shape[1:]):
        raise ValueError("decoded endpoint shape does not match interpolation outputs")
    endpoint_latent_l2 = float(np.linalg.norm(endpoint_latent[1] - endpoint_latent[0]))
    adjacent_latent_l2 = float(np.min(np.linalg.norm(np.diff(interpolation_latent, axis=0).reshape(6, -1), axis=1)))
    endpoint_decoded_l2 = float(np.linalg.norm(endpoint_decoded[1] - endpoint_decoded[0]))
    adjacent_decoded_l2 = float(np.min(np.linalg.norm(np.diff(interpolation_decoded, axis=0).reshape(6, -1), axis=1)))
    thresholds = {
        "endpoint_latent_l2": endpoint_latent_l2,
        "min_adjacent_latent_l2": adjacent_latent_l2,
        "endpoint_decoded_l2": endpoint_decoded_l2,
        "min_adjacent_decoded_l2": adjacent_decoded_l2,
    }
    if endpoint_latent_l2 <= MIN_ENDPOINT_LATENT_L2 or adjacent_latent_l2 <= MIN_ADJACENT_LATENT_L2:
        raise ValueError("latent interpolation is degenerate")
    if endpoint_decoded_l2 <= MIN_ENDPOINT_DECODED_L2 or adjacent_decoded_l2 <= MIN_ADJACENT_DECODED_L2:
        raise ValueError("decoded interpolation is degenerate")
    return thresholds


def validate_payload(payload: dict[str, Any], png_path: Path) -> None:
    """Validate provenance, artifact bytes, dimensions, and declared gates."""

    if payload.get("model_id") != MODEL_ID or payload.get("model_revision") != MODEL_REVISION:
        raise ValueError("interpolation provenance does not match the pinned checkpoint")
    if payload.get("license") != "mit" or payload.get("weights_sha256") != SAFE_WEIGHTS_SHA256:
        raise ValueError("interpolation artifact has invalid license or checkpoint hash")
    if payload.get("coefficients") != [float(value) for value in WEIGHTS]:
        raise ValueError("interpolation coefficients are not canonical")
    if not png_path.is_file() or payload.get("png_sha256") != _sha256(png_path):
        raise ValueError("interpolation PNG is missing or does not match the JSON artifact")
    image = mpimg.imread(png_path)
    if image.shape[1] != PNG_WIDTH or image.shape[0] != PNG_HEIGHT:
        raise ValueError("interpolation PNG dimensions do not match the declared artifact")
    if payload.get("png_width") != PNG_WIDTH or payload.get("png_height") != PNG_HEIGHT:
        raise ValueError("interpolation PNG dimensions are missing or incorrect")
    if payload.get("network_attempts") != [] or payload.get("remote_code") is not False:
        raise ValueError("interpolation lane was not local-only")


def run_evidence(
    snapshot: Path,
    json_path: Path = DEFAULT_JSON,
    png_path: Path = DEFAULT_PNG,
) -> dict[str, Any]:
    """Run the bounded local interpolation lane and write JSON/PNG evidence."""

    expected_files = {"config.json", "README.md", "diffusion_pytorch_model.safetensors"}
    if not snapshot.is_dir() or {path.name for path in snapshot.iterdir()} != expected_files:
        raise FileNotFoundError(f"expected isolated checkpoint files under {snapshot}")
    weights_file = snapshot / "diffusion_pytorch_model.safetensors"
    if weights_file.stat().st_size != SAFE_WEIGHTS_SIZE or _sha256(weights_file) != SAFE_WEIGHTS_SHA256:
        raise ValueError("safetensors size or SHA-256 does not match the pinned checkpoint")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") != "AutoencoderKL" or "auto_map" in config:
        raise ValueError("checkpoint config is not the standard local AutoencoderKL contract")

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        if "after parallel work has started" not in str(error):
            raise
    torch.use_deterministic_algorithms(True)
    memory_sample = cast(Any, fidelity)._memory_sample
    memory_samples: list[dict[str, int | None]] = [memory_sample()]
    started = time.perf_counter()
    network_attempts: list[str] = []
    original_connect = socket.socket.connect

    def deny_connect(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        network_attempts.append("socket.connect")
        raise AssertionError("network connection attempted during local-only interpolation evidence")

    socket.socket.connect = deny_connect
    try:
        inputs = _inputs()
        adapter = DiffusersAutoencoderKLAdapter(str(snapshot), MODEL_REVISION, dtype=np.float32)
        endpoint_latent = adapter.encode(inputs)
        interpolation_latent = np.stack(
            [(1.0 - weight) * endpoint_latent[0] + weight * endpoint_latent[1] for weight in WEIGHTS]
        )
        endpoint_decoded = adapter.decode(endpoint_latent)
        interpolation_decoded = adapter.decode(interpolation_latent)
        movement = validate_arrays(
            WEIGHTS, endpoint_latent, interpolation_latent, endpoint_decoded, interpolation_decoded
        )
        memory_samples.append(memory_sample())
        ARTIFACT_DIR.mkdir(exist_ok=True)
        figure, axes = plt.subplots(1, len(WEIGHTS), figsize=(14, 2.4))
        for axis, image, weight in zip(axes, interpolation_decoded, WEIGHTS, strict=True):
            axis.imshow(np.moveaxis((image + 1.0) / 2.0, 0, -1).clip(0, 1))
            axis.set_title(f"t={weight:.2f}")
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(png_path, dpi=150, metadata={"Software": "latent-anything-sprint35"})
        plt.close(figure)
    finally:
        socket.socket.connect = original_connect
    memory_samples.extend((memory_sample(), memory_sample()))
    runtime_seconds = time.perf_counter() - started
    rss_values = [sample["peak_rss_bytes"] for sample in memory_samples if sample["peak_rss_bytes"] is not None]
    peak_rss = max(rss_values) if rss_values else None
    if runtime_seconds > MAX_RUNTIME_SECONDS or (peak_rss is not None and peak_rss > MAX_PEAK_RSS_BYTES):
        raise ValueError("interpolation exceeded CPU runtime or RSS bounds")
    image = mpimg.imread(png_path)
    body: dict[str, Any] = {
        "evidence_level": "D2",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license": "mit",
        "weights_sha256": SAFE_WEIGHTS_SHA256,
        "input_layout": "NCHW",
        "input_range": [-1.0, 1.0],
        "coefficients": [float(value) for value in WEIGHTS],
        "endpoint_order": ["digit_0", "digit_1"],
        "latent_shape": list(interpolation_latent.shape),
        "decoded_shape": list(interpolation_decoded.shape),
        "latent_dtype": str(interpolation_latent.dtype),
        "decoded_dtype": str(interpolation_decoded.dtype),
        "latent_range": [float(np.min(interpolation_latent)), float(np.max(interpolation_latent))],
        "decoded_range": [float(np.min(interpolation_decoded)), float(np.max(interpolation_decoded))],
        "movement": movement,
        "endpoint_reconstruction": _metrics(endpoint_decoded, interpolation_decoded[[0, -1]]),
        "thresholds": {
            "min_endpoint_latent_l2": MIN_ENDPOINT_LATENT_L2,
            "min_adjacent_latent_l2": MIN_ADJACENT_LATENT_L2,
            "min_endpoint_decoded_l2": MIN_ENDPOINT_DECODED_L2,
            "min_adjacent_decoded_l2": MIN_ADJACENT_DECODED_L2,
        },
        "remote_code": False,
    }
    deterministic_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = {
        **body,
        "deterministic_content_sha256": hashlib.sha256(deterministic_bytes).hexdigest(),
        "png_sha256": _sha256(png_path),
        "png_width": int(image.shape[1]),
        "png_height": int(image.shape[0]),
        "png_channels": int(image.shape[2]) if image.ndim == 3 else 1,
        "network_attempts": network_attempts,
        "runtime_seconds": runtime_seconds,
        "max_peak_rss_bytes": peak_rss,
        "memory_samples": memory_samples,
        "bounded_cpu": {"max_runtime_seconds": MAX_RUNTIME_SECONDS, "max_peak_rss_bytes": MAX_PEAK_RSS_BYTES},
        "json_path": json_path.name,
        "png_path": png_path.name,
    }
    validate_payload(payload, png_path)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    snapshot = root / ".cache" / f"hf-sd-vae-ft-mse-{MODEL_REVISION}"
    payload = run_evidence(snapshot, root / DEFAULT_JSON, root / DEFAULT_PNG)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
