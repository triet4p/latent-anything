"""Future M14 L02 artifact/run envelopes and provenance validation."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from latent_anything.adapters.conv_vae import ConvVAE
from latent_anything.density import GaussianMixtureDensity
from scripts.m14_l02_plan import EXPECTED_RECORD_IDS, plan_digest, section

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def artifact_digest(artifact: Mapping[str, Any]) -> str:
    """Return canonical artifact digest excluding its self-reference."""
    unsigned = dict(artifact)
    unsigned.pop("artifact_sha256", None)
    canonical = json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _package_versions() -> dict[str, str]:
    packages = ("latent-anything", "numpy", "scikit-learn", "scipy", "torch")
    versions: dict[str, str] = {"python": platform.python_version(), "platform": platform.platform()}
    for package in packages:
        try:
            versions[package] = package_version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def source_digests() -> dict[str, str]:
    """Hash the orchestrator, plan, and supporting lane implementation."""
    root = Path(__file__).resolve().parent
    runner_names = ("m14_l02_geometry.py", "m14_l02_data.py", "m14_l02_metrics.py", "m14_l02_envelope.py")
    runner_bytes = b"".join((root / name).read_bytes() for name in runner_names)
    return {
        "runner_source_sha256": hashlib.sha256((root / "m14_l02_geometry.py").read_bytes()).hexdigest(),
        "contract_source_sha256": hashlib.sha256((root / "m14_l02_plan.py").read_bytes()).hexdigest(),
        "implementation_source_sha256": hashlib.sha256(runner_bytes).hexdigest(),
    }


def input_digests(named_arrays: Mapping[str, Any]) -> dict[str, str]:
    """Hash scientific arrays in stable key order for mutation checks."""
    output: dict[str, str] = {}
    for name in sorted(named_arrays):
        value = named_arrays[name]
        contiguous = value if value.flags.c_contiguous else value.copy(order="C")
        header = f"{contiguous.dtype.str}:{contiguous.shape}".encode("ascii")
        output[name] = hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()
    return output


def validate_run_record(
    run_record: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
    current_sources: Mapping[str, str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if run_record.get("schema_version") != "m14-l02-geometry-run-v1":
        errors.append("run record schema is not M14-L02")
    if run_record.get("lane") != "M14-L02":
        errors.append("run record lane must be M14-L02")
    if run_record.get("artifact_name") != "artifacts/m14/l02-geometry.json":
        errors.append("run record must point to the shared L02 artifact")
    for field in ("artifact_sha256", "plan_sha256", "runner_source_sha256", "contract_source_sha256"):
        if not isinstance(run_record.get(field), str) or not SHA256_RE.fullmatch(run_record[field]):
            errors.append(f"run record {field} must be a SHA-256 hex digest")
    if not isinstance(run_record.get("git_sha"), str) or not SHA1_RE.fullmatch(run_record["git_sha"]):
        errors.append("run record git_sha must be an exact 40-hex committed SHA")
    timestamp = run_record.get("timestamp_utc")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        errors.append("run record timestamp_utc must be UTC and end in Z")
    else:
        try:
            datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except ValueError:
            errors.append("run record timestamp_utc must be ISO-8601")
    if run_record.get("status") != "completed":
        errors.append("run record status must be completed")
    for field in ("command", "network", "credentials", "cleanup", "resource_measurement"):
        if not isinstance(run_record.get(field), str) or not run_record[field].strip():
            errors.append(f"run record {field} wording is required")
    if not isinstance(run_record.get("accepted_record_ids"), list) or not isinstance(
        run_record.get("accepted_gap_ids"), list
    ):
        errors.append("run record accepted IDs must be lists")
    if artifact is not None:
        if run_record.get("artifact_sha256") != artifact.get("artifact_sha256"):
            errors.append("run record artifact digest does not match artifact self-digest")
        if run_record.get("plan_sha256") != artifact.get("plan_sha256"):
            errors.append("run record plan digest does not match artifact")
        if run_record.get("accepted_record_ids") != artifact.get("accepted_record_ids") or run_record.get(
            "accepted_gap_ids"
        ) != artifact.get("accepted_gap_ids"):
            errors.append("run record accepted IDs do not match artifact")
    if plan is not None and run_record.get("plan_sha256") != plan_digest(plan):
        errors.append("run record plan digest does not match the plan")
    if current_sources is not None:
        for field in ("runner_source_sha256", "contract_source_sha256"):
            if run_record.get(field) != current_sources.get(field):
                errors.append(f"run record {field} does not match current source")
    return errors


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None = None,
    current_sources: Mapping[str, str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if artifact.get("schema_version") != "m14-l02-geometry-artifact-v1" or artifact.get("lane") != "M14-L02":
        errors.append("artifact identity is not M14-L02")
    if artifact.get("partial_promotion") is not True or artifact.get("evidence_level") not in {"D1", "D2"}:
        errors.append("artifact promotion/evidence envelope is invalid")
    records = artifact.get("records")
    objects = [item for item in records if isinstance(item, Mapping)] if isinstance(records, list) else []
    if (
        not isinstance(records, list)
        or len(objects) != len(records)
        or tuple(item.get("record_id") for item in objects) != EXPECTED_RECORD_IDS
    ):
        errors.append("artifact must contain six independent records in plan order")
    else:
        accepted = [str(item["record_id"]) for item in objects if item.get("accepted") is True]
        gaps = [str(gap) for item in objects if item.get("accepted") is True for gap in item.get("gap_ids", [])]
        if artifact.get("accepted_record_ids") != accepted or artifact.get("accepted_gap_ids") != gaps:
            errors.append("artifact accepted records/gaps do not match independent verdicts")
        if artifact.get("evidence_level") != ("D2" if accepted else "D1"):
            errors.append("artifact evidence level does not match accepted records")
        for item in objects:
            expected = "accepted" if item.get("accepted") is True else "failed"
            if not isinstance(item.get("accepted"), bool) or item.get("verdict") != expected:
                errors.append(f"artifact record {item.get('record_id')!r} has invalid verdict fields")
    digest = artifact.get("artifact_sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or digest != artifact_digest(artifact):
        errors.append("artifact_sha256 does not match canonical artifact")
    if not isinstance(artifact.get("plan_sha256"), str) or not SHA256_RE.fullmatch(artifact["plan_sha256"]):
        errors.append("artifact plan_sha256 must be a SHA-256 hex digest")
    if plan is not None and artifact.get("plan_sha256") != plan_digest(plan):
        errors.append("artifact plan digest does not match the plan")
    provenance = artifact.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or not isinstance(provenance.get("git_sha"), str)
        or not SHA1_RE.fullmatch(provenance["git_sha"])
    ):
        errors.append("artifact provenance git_sha must be exact 40-hex")
    elif current_sources is not None:
        for field in ("runner_source_sha256", "contract_source_sha256"):
            if provenance.get(field) != current_sources.get(field):
                errors.append(f"artifact provenance {field} does not match current source")
    dataset = artifact.get("dataset")
    required = {
        "dataset",
        "license",
        "content_sha256",
        "total_samples",
        "train_samples",
        "heldout_samples",
        "train_index_sha256",
        "heldout_index_sha256",
    }
    if not isinstance(dataset, Mapping) or not required.issubset(dataset):
        errors.append("artifact dataset provenance is incomplete")
    elif (
        not all(
            isinstance(dataset.get(field), str) and SHA256_RE.fullmatch(dataset[field])
            for field in ("content_sha256", "train_index_sha256", "heldout_index_sha256")
        )
        or not all(
            isinstance(dataset.get(field), int) for field in ("total_samples", "train_samples", "heldout_samples")
        )
        or dataset["total_samples"] != dataset["train_samples"] + dataset["heldout_samples"]
    ):
        errors.append("artifact dataset counts or digests are invalid")
    model = artifact.get("model")
    density = artifact.get("density")
    if (
        not isinstance(model, Mapping)
        or model.get("fit_scope") != "train images only"
        or not isinstance(model.get("config"), Mapping)
    ):
        errors.append("artifact model provenance is incomplete")
    if (
        not isinstance(density, Mapping)
        or density.get("fit_scope") != "train latents only"
        or not isinstance(density.get("config"), Mapping)
    ):
        errors.append("artifact density provenance is incomplete")
    if not isinstance(artifact.get("backend_versions"), Mapping):
        errors.append("artifact backend/package versions are missing")
    inputs = artifact.get("input_digests")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(inputs.get("before"), Mapping)
        or not isinstance(inputs.get("after"), Mapping)
        or inputs["before"].keys() != inputs["after"].keys()
    ):
        errors.append("artifact before/after input digests are missing or mismatched")
    elif not all(
        isinstance(value, str) and SHA256_RE.fullmatch(value)
        for side in ("before", "after")
        for value in inputs[side].values()
    ):
        errors.append("artifact input digests must be SHA-256 maps")
    return errors


def build_payload(
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    adapter: ConvVAE,
    density: GaussianMixtureDensity,
    split_metadata: Mapping[str, Any],
    before: Mapping[str, str],
    after: Mapping[str, str],
    latent_digests: Mapping[str, str],
) -> dict[str, Any]:
    accepted = [record for record in records if bool(record["accepted"])]
    payload: dict[str, Any] = {
        "schema_version": str(section(plan, "artifact_schema")["schema_version"]),
        "lane": "M14-L02",
        "evidence_level": "D2" if accepted else "D1",
        "accepted_record_ids": [str(r["record_id"]) for r in accepted],
        "accepted_gap_ids": [gap for r in accepted for gap in r["gap_ids"]],
        "partial_promotion": True,
        "records": [dict(record) for record in records],
        "dataset": dict(split_metadata),
        "model": {
            "config": dict(section(plan, "model")),
            "metrics": dict(adapter.metrics_),
            "fit_scope": "train images only",
        },
        "density": {
            "config": dict(section(plan, "density")),
            "state_digest": density.state_digest(),
            "fit_scope": "train latents only",
        },
        "backend_versions": _package_versions(),
        "input_digests": {"before": dict(before), "after": dict(after)},
        "derived_latent_digests": dict(latent_digests),
        "provenance": {"git_sha": _git_sha(), **source_digests()},
        "plan_sha256": plan_digest(plan),
    }
    payload["artifact_sha256"] = artifact_digest(payload)
    return payload


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def write_artifact_and_run_record(
    plan: Mapping[str, Any], payload: Mapping[str, Any], root: Path | None = None
) -> None:
    """Write future outputs only after payload and provenance validation."""
    output_root = root or Path(__file__).resolve().parents[1] / "artifacts/m14"
    artifact_path, run_path = output_root / "l02-geometry.json", output_root / "l02-geometry.run.json"
    artifact_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    sources = source_digests()
    errors = validate_artifact(payload, plan=plan, current_sources=sources)
    if errors:
        raise ValueError("invalid L02 artifact payload: " + "; ".join(errors))
    run_record = {
        "schema_version": str(section(plan, "run_record_schema")["schema_version"]),
        "lane": "M14-L02",
        "artifact_name": "artifacts/m14/l02-geometry.json",
        "artifact_sha256": str(payload["artifact_sha256"]),
        "git_sha": str(payload["provenance"]["git_sha"]),
        "runner_source_sha256": sources["runner_source_sha256"],
        "contract_source_sha256": sources["contract_source_sha256"],
        "plan_sha256": plan_digest(plan),
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "network": str(section(plan, "resource_contract")["network"]),
        "credentials": "none used; no credentials required",
        "cleanup": str(section(plan, "resource_contract")["cleanup"]),
        "resource_measurement": str(section(plan, "resource_contract")["peak_rss"]),
        "accepted_record_ids": list(payload["accepted_record_ids"]),
        "accepted_gap_ids": list(payload["accepted_gap_ids"]),
    }
    errors = validate_run_record(run_record, artifact=payload, plan=plan, current_sources=sources)
    if errors:
        raise ValueError("invalid L02 run record: " + "; ".join(errors))
    artifact_path.write_bytes(artifact_bytes)
    run_path.write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
