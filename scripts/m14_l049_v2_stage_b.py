"""CLI for the one-shot L04.9 v2 Stage B holdout evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from scripts._m14_l049_v2_fixture import read_rows, validate_rows
from scripts._m14_l049_v2_schema import V2_ADDENDUM_PATH, canonical_json_bytes, validation_rejection_codes
from scripts._m14_l049_v2_stage_b import build_stage_b_validation_rejected_artifact
from scripts._m14_l049_v2_validate import validate_stage_b


def _source_commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None or value == "0" * 40:
        raise argparse.ArgumentTypeError("source commit must be a nonzero lowercase 40-hex SHA")
    return value


def _write_attempt_triad(output: Path, artifact: dict[str, object], source_commit_sha: str) -> None:
    pattern = re.compile(r"^l04-explanations\.L049V2StageB\.attempt([0-9]+)\.(?:partial|run|failure)\.json$")
    attempts = [
        int(match.group(1))
        for path in output.parent.glob("l04-explanations.L049V2StageB.attempt*.json")
        if (match := pattern.fullmatch(path.name))
    ]
    attempt = max(attempts, default=0) + 1
    prefix = output.parent / f"l04-explanations.L049V2StageB.attempt{attempt}"
    common = {
        "schema_version": "m14-l04.9-v2-transport-envelope-v1",
        "use_case": "L049V2StageB",
        "attempt": f"attempt{attempt}",
        "source_commit_sha": source_commit_sha,
        "stage": artifact["stage"],
        "status": artifact["status"],
        "artifact_sha256": artifact["artifact_sha256"],
    }
    values = (
        ("partial", {**common, "kind": "partial", "artifact": artifact}),
        (
            "run",
            {**common, "kind": "run", "artifact_name": f"l04-explanations.L049V2StageB.attempt{attempt}.partial.json"},
        ),
        (
            "failure",
            {
                **common,
                "kind": "failure",
                "failure_ref": f"l04-explanations.L049V2StageB.attempt{attempt}.failure.json",
            },
        ),
    )
    for kind, value in values:
        (Path(f"{prefix}.{kind}.json")).write_bytes(canonical_json_bytes(value) + b"\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-fixture", type=Path, required=True)
    parser.add_argument("--holdout-seed", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--observations", type=Path, help="Offline observations; omitted for --run-real")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit-sha", required=True, type=_source_commit)
    parser.add_argument("--run-real", action="store_true", help="Run the pinned CUDA holdout boundary")
    args = parser.parse_args(argv)
    _raw, rows = read_rows(args.holdout_fixture)
    errors = validate_rows(rows, "holdout", 24)
    if errors:
        raise SystemExit("; ".join(errors))
    addendum = json.loads(V2_ADDENDUM_PATH.read_bytes())
    candidate = json.loads(args.candidate_manifest.read_bytes())
    cli_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    seed = args.holdout_seed.read_bytes()
    if args.run_real:
        from scripts._m14_l049_v2_stage_b import build_stage_b_failure_artifact

        try:
            import torch
        except Exception as error:  # noqa: BLE001 - no CUDA attempt started
            raise SystemExit("real Stage B dependencies are unavailable") from error
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as error:  # noqa: BLE001 - CUDA preflight failed
            raise SystemExit("CUDA preflight failed") from error
        if not cuda_available:
            raise SystemExit("CUDA is unavailable")
        resources: dict[str, object] | None = None
        real_runtime_error_type: type[BaseException] | tuple[type[BaseException], ...] = ()
        try:
            from scripts._m14_l049_v2_real_runtime import (
                RealRuntimeError,
                build_stage_b_runtime,
            )

            real_runtime_error_type = RealRuntimeError
            observations, resources = build_stage_b_runtime(rows, candidate)
            from scripts._m14_l049_v2_stage_b import evaluate_stage_b

            artifact = evaluate_stage_b(
                rows, observations, candidate, addendum, seed, resources=resources, cli_sha256=cli_sha
            )
        except Exception as error:  # noqa: BLE001 - all post-CUDA failures are attempted-real
            runtime_error = error.original_error if isinstance(error, real_runtime_error_type) else error
            runtime_resources = error.resources if isinstance(error, real_runtime_error_type) else resources
            if not isinstance(runtime_resources, dict):
                from scripts._m14_l049_v2_stage_a import attempted_real_resources

                runtime_resources = attempted_real_resources()
            artifact = build_stage_b_failure_artifact(
                rows,
                candidate,
                addendum,
                seed,
                source_sha256=str(candidate.get("source_sha256", "")),
                error=runtime_error,
                resources=runtime_resources,
                cli_sha256=cli_sha,
            )
    else:
        if args.observations is None:
            raise SystemExit("--observations is required unless --run-real is used")
        from scripts._m14_l049_v2_stage_b import evaluate_stage_b

        observations = json.loads(args.observations.read_bytes())
        artifact = evaluate_stage_b(rows, observations, candidate, addendum, seed, resources=None, cli_sha256=cli_sha)
    validation = validate_stage_b(artifact, rows, seed, candidate, addendum)
    if validation and args.run_real:
        artifact = build_stage_b_validation_rejected_artifact(
            rows,
            candidate,
            addendum,
            seed,
            source_sha256=str(candidate.get("source_sha256", "")),
            resources=artifact.get("resources") if isinstance(artifact.get("resources"), Mapping) else None,
            validation_codes=validation_rejection_codes(validation),
            cli_sha256=cli_sha,
        )
        validation = validate_stage_b(artifact, rows, seed, candidate, addendum)
    if validation:
        raise SystemExit("real Stage B artifact validation failed" if args.run_real else "; ".join(validation))
    try:
        args.output.write_bytes(canonical_json_bytes(artifact) + b"\n")
        _write_attempt_triad(args.output, artifact, args.source_commit_sha)
    except Exception as error:  # noqa: BLE001 - no recursive artifact fabrication
        raise SystemExit("Stage B triad serialization failed") from error
    print(
        json.dumps(
            {
                "stage": "stage_b_holdout_evaluation",
                "status": artifact["status"],
                "evidence_level": artifact["evidence_level"],
                "artifact_sha256": artifact["artifact_sha256"],
                "cli_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
