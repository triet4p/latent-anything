"""Measure fidelity of the public Diffusers VAE adapter against AutoencoderKL.

This lane is deliberately local-only.  It consumes the already acquired,
revision-pinned snapshot and never acquires model files itself.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import gc
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from latent_anything.integrations.diffusers_vae import DiffusersAutoencoderKLAdapter

MODEL_ID = "stabilityai/sd-vae-ft-mse"
MODEL_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
SAFE_WEIGHTS_SHA256 = "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"
SAFE_WEIGHTS_SIZE = 334_643_276
RTOL = 1e-5
ATOL = 1e-6
MAX_RUNTIME_SECONDS = 60.0
MAX_PEAK_RSS_BYTES = 2 * 1024**3
EXPECTED_FILES = {"config.json", "README.md", "diffusion_pytorch_model.safetensors"}


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _memory_sample() -> dict[str, int | None]:
    """Return current-process RSS/peak RSS and available physical RAM."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    got_rss = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), ctypes.sizeof(counters))
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    got_available = kernel.GlobalMemoryStatusEx(ctypes.byref(status))
    return {
        "rss_bytes": int(counters.WorkingSetSize) if got_rss else None,
        "peak_rss_bytes": int(counters.PeakWorkingSetSize) if got_rss else None,
        "available_physical_bytes": int(status.ullAvailPhys) if got_available else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_snapshot_label(snapshot: Path) -> str:
    """Return a repository-relative checkpoint label for portable evidence."""

    return f".cache/{snapshot.name}"


def _inputs() -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    first = np.stack((xx, yy, (xx + yy) / 2.0), axis=0)
    second = np.stack((np.sin(np.pi * xx), np.cos(np.pi * yy), xx * yy), axis=0).astype(np.float32)
    return np.stack((first, second), axis=0).astype(np.float32)


def _direct_encode(model: Any, images: np.ndarray, mode: str, seed: int) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(images)
    with torch.no_grad():
        distribution = model.encode(tensor).latent_dist
        if mode == "mean":
            raw = distribution.mean
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            raw = distribution.sample(generator=generator)
        return (raw * float(model.config.scaling_factor)).detach().cpu().numpy()


def _direct_decode(model: Any, latent: np.ndarray) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(latent)
    with torch.no_grad():
        output = model.decode(tensor / float(model.config.scaling_factor)).sample
        return output.detach().cpu().numpy()


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    denominator = np.maximum(np.abs(np.asarray(reference, dtype=np.float64)), 1e-8)
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "max_relative_error": float(np.max(np.abs(delta) / denominator)),
    }


def run_evidence(snapshot: Path, artifact_path: Path | None = None) -> dict[str, Any]:
    """Run and optionally persist the direct-versus-adapter fidelity lane."""
    if os.name != "nt":
        raise RuntimeError("Sprint35 fidelity evidence currently requires Windows RSS APIs")
    if not snapshot.is_dir() or {path.name for path in snapshot.iterdir()} != EXPECTED_FILES:
        raise FileNotFoundError(f"expected isolated checkpoint files under {snapshot}")
    weights = snapshot / "diffusion_pytorch_model.safetensors"
    if weights.stat().st_size != SAFE_WEIGHTS_SIZE or _sha256(weights) != SAFE_WEIGHTS_SHA256:
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
    import diffusers
    import safetensors
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    samples: list[dict[str, int | None]] = [_memory_sample()]
    started = time.perf_counter()
    network_attempts: list[str] = []
    original_connect = socket.socket.connect

    def deny_connect(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        network_attempts.append("socket.connect")
        raise AssertionError("network connection attempted during local-only fidelity evidence")

    socket.socket.connect = deny_connect
    try:
        autoencoder = cast(Any, diffusers).AutoencoderKL
        direct = autoencoder.from_pretrained(
            snapshot,
            local_files_only=True,
            use_safetensors=True,
        ).to(device="cpu", dtype=torch.float32)
        direct.eval()
        samples.append(_memory_sample())
        images = _inputs()
        mode_results: dict[str, Any] = {}
        modes: tuple[tuple[Literal["mean", "sample"], int], ...] = (("mean", 1234), ("sample", 5678))
        for mode, seed in modes:
            adapter = DiffusersAutoencoderKLAdapter(str(snapshot), MODEL_REVISION, latent_mode=mode, dtype=np.float32)
            direct_latent = _direct_encode(direct, images, mode, seed)
            adapter_latent = adapter.encode(images, seed=seed if mode == "sample" else None)
            latent_metrics = metrics(direct_latent, adapter_latent)
            np.testing.assert_allclose(adapter_latent, direct_latent, rtol=RTOL, atol=ATOL)
            direct_decoded = _direct_decode(direct, direct_latent)
            adapter_decoded = adapter.decode(adapter_latent)
            decode_metrics = metrics(direct_decoded, adapter_decoded)
            np.testing.assert_allclose(adapter_decoded, direct_decoded, rtol=RTOL, atol=ATOL)
            if mode == "sample":
                repeat = adapter.encode(images, seed=seed)
                np.testing.assert_array_equal(repeat, adapter_latent)
            mode_results[mode] = {
                "seed": seed,
                "latent_shape": list(adapter_latent.shape),
                "decoded_shape": list(adapter_decoded.shape),
                "latent_dtype": str(adapter_latent.dtype),
                "decoded_dtype": str(adapter_decoded.dtype),
                "latent_metrics": latent_metrics,
                "decode_metrics": decode_metrics,
                "adapter_latent_space_dim": adapter.latent_space.dim,
                "adapter_metadata": dict(adapter.latent_space.metadata),
                "finite": bool(np.isfinite(adapter_latent).all() and np.isfinite(adapter_decoded).all()),
            }
            samples.append(_memory_sample())
            del adapter
            gc.collect()
    finally:
        socket.socket.connect = original_connect
    elapsed = time.perf_counter() - started
    samples.append(_memory_sample())
    del direct
    gc.collect()
    samples.append(_memory_sample())
    rss_values = [sample["peak_rss_bytes"] for sample in samples if sample["peak_rss_bytes"] is not None]
    minimum_available = [
        sample["available_physical_bytes"] for sample in samples if sample["available_physical_bytes"] is not None
    ]
    peak_rss = max(rss_values) if rss_values else None
    payload: dict[str, Any] = {
        "evidence_level": "D2",
        "accepted": bool(
            not network_attempts
            and elapsed <= MAX_RUNTIME_SECONDS
            and (peak_rss is None or peak_rss <= MAX_PEAK_RSS_BYTES)
            and all(result["finite"] for result in mode_results.values())
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "snapshot": portable_snapshot_label(snapshot),
        "license": "mit",
        "weights": {"path": weights.name, "bytes": weights.stat().st_size, "sha256": _sha256(weights)},
        "config": {
            "class_name": config["_class_name"],
            "diffusers_version": config.get("_diffusers_version"),
            "scaling_factor": float(config.get("scaling_factor", 0.18215)),
            "auto_map": False,
        },
        "versions": {
            "diffusers": diffusers.__version__,
            "safetensors": safetensors.__version__,
            "torch": torch.__version__,
        },
        "input": {"shape": list(_inputs().shape), "dtype": "float32", "range": [-1.0, 1.0], "layout": "NCHW"},
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "modes": mode_results,
        "network_attempts": network_attempts,
        "runtime_seconds": elapsed,
        "max_peak_rss_bytes": peak_rss,
        "minimum_available_physical_bytes": min(minimum_available) if minimum_available else None,
        "memory_samples": samples,
        "bounded_cpu": {"max_runtime_seconds": MAX_RUNTIME_SECONDS, "max_peak_rss_bytes": MAX_PEAK_RSS_BYTES},
    }
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    snapshot = root / ".cache" / f"hf-sd-vae-ft-mse-{MODEL_REVISION}"
    artifact = root / "artifacts" / "diffusers_vae_fidelity.json"
    payload = run_evidence(snapshot, artifact)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["accepted"]:
        raise SystemExit("fidelity acceptance failed")


if __name__ == "__main__":
    main()
