"""Producer-side structured runtime attestation for the L04.9 v2 lane."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts._m14_l049_v2_schema import (
    EXPECTED_RUNTIME_DTYPE,
    EXPECTED_RUNTIME_INTEGRATION,
    EXPECTED_RUNTIME_MODEL,
    RUNTIME_ATTESTATION_SCHEMA,
    RUNTIME_EVENT_CODES,
    canonical_digest,
    canonical_json_bytes,
    digest_bytes,
    top_level_cli_sha256,
)


def build_runtime_attestation(
    *,
    stage: str,
    mode: str,
    group_count: int,
    pair_count: int,
    candidate_count: int,
    seed_count: int,
    fixture_sha256: str,
    candidate_sha256: str,
    source_sha256: str,
    addendum_sha256: str,
    cli_sha256: str,
    resources: Mapping[str, Any],
    operation_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Create a deterministic attestation from declared primitive counts.

    The producer records facts and hashes only; the independent validators
    recompute every field rather than trusting this helper or its digest.
    """
    if min(group_count, pair_count, candidate_count, seed_count) <= 0:
        raise ValueError("runtime attestation counts must be positive")
    if mode not in {"synthetic", "real"}:
        raise ValueError("runtime attestation mode is invalid")
    execution_backend = "cuda" if mode == "real" else "synthetic"
    expected_cli_sha = top_level_cli_sha256(stage)
    if expected_cli_sha is not None and cli_sha256 != expected_cli_sha:
        raise ValueError("attestation must bind the top-level stage CLI")
    if mode == "real" and resources.get("execution_backend") != "cuda":
        raise ValueError("real attestation requires CUDA resources")
    events = [{"ordinal": index, "code": code} for index, code in enumerate(RUNTIME_EVENT_CODES)]
    if operation_counts is None:
        # Synthetic protocol fixtures have no model operations.  Keep their
        # declared protocol counts deterministic; real runs must supply the
        # counters collected by the runtime seam below.
        capture_count = group_count * 2 * seed_count
        hook_count = group_count * seed_count
        patch_count = group_count * seed_count
        control_count = group_count * 4 * seed_count
        forward_count = capture_count + patch_count + control_count
    else:
        required_counts = {"candidate_evaluations", "hooks", "captures", "patches", "controls", "forwards"}
        if set(operation_counts) != required_counts or any(value < 0 for value in operation_counts.values()):
            raise ValueError("runtime operation counters are malformed")
        capture_count = operation_counts["captures"]
        hook_count = operation_counts["hooks"]
        patch_count = operation_counts["patches"]
        control_count = operation_counts["controls"]
        forward_count = operation_counts["forwards"]
    hash_chain = [
        {"name": "fixture", "sha256": fixture_sha256},
        {"name": "candidate", "sha256": candidate_sha256},
        {"name": "source", "sha256": source_sha256},
        {"name": "addendum", "sha256": addendum_sha256},
    ]
    zero_digest = "0" * 64
    peak = resources.get("resource_peak")
    if mode == "real" and isinstance(peak, Mapping):
        peak_resources = {
            "peak_cpu_bytes": peak.get("peak_cpu_bytes"),
            "peak_gpu_bytes": peak.get("peak_gpu_bytes"),
            "unit": peak.get("unit"),
            "source": "torch.cuda.max_memory_allocated",
            "budget_cpu_bytes": peak.get("budget_cpu_bytes"),
            "budget_gpu_bytes": peak.get("budget_gpu_bytes"),
        }
    else:
        peak_resources = {
            "peak_cpu_bytes": 0,
            "peak_gpu_bytes": 0,
            "unit": "bytes",
            "source": "synthetic_fixture",
            "budget_cpu_bytes": 0,
            "budget_gpu_bytes": 0,
        }
    cleanup = resources.get("cleanup")
    cleanup_hook_count = (
        int(cleanup["hooks_remaining"])
        if isinstance(cleanup, Mapping)
        and cleanup.get("attempted") is True
        and cleanup.get("completed") is False
        and isinstance(cleanup.get("hooks_remaining"), int)
        else 0
    )
    attestation: dict[str, Any] = {
        "schema_version": RUNTIME_ATTESTATION_SCHEMA,
        "stage": stage,
        "mode": mode,
        "model": {
            "name": EXPECTED_RUNTIME_MODEL,
            "revision": EXPECTED_RUNTIME_MODEL,
            "integration": EXPECTED_RUNTIME_INTEGRATION,
            "model_adapter": "N/A",
            "device": "cuda" if mode == "real" else "cpu",
            "backend": execution_backend,
            "dtype": EXPECTED_RUNTIME_DTYPE,
        },
        "commitments": {
            "fixture_sha256": fixture_sha256,
            "candidate_sha256": candidate_sha256,
            "source_sha256": source_sha256,
            "addendum_sha256": addendum_sha256,
        },
        "events": events,
        "counts": {
            "groups": group_count,
            "pairs": pair_count,
            "candidates": candidate_count,
            "seeds": seed_count,
            "candidate_evaluations": operation_counts["candidate_evaluations"]
            if operation_counts is not None
            else group_count * 2 * candidate_count * seed_count,
            "hooks": hook_count,
            "captures": capture_count,
            "patches": patch_count,
            "controls": control_count,
            "forwards": forward_count,
        },
        "parameters": {"before_sha256": zero_digest, "after_sha256": zero_digest},
        "cleanup_hook_count": cleanup_hook_count,
        "resources": peak_resources,
        "cli_sha256": cli_sha256,
        "runtime_module_digests": {"producer": source_sha256, "integration": source_sha256},
        "hash_chain": hash_chain,
        "hash_chain_sha256": digest_bytes(canonical_json_bytes({"items": hash_chain})),
        "transcript_sha256": digest_bytes(canonical_json_bytes({"items": events})),
        "execution_backend": execution_backend,
    }
    attestation["attestation_sha256"] = canonical_digest(attestation, "attestation_sha256")
    return attestation


__all__ = ["build_runtime_attestation"]
