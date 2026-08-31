"""CLI for an offline, train-only L04.9 v2 Stage A protocol fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from scripts._m14_l049_v2_fixture import read_rows, validate_rows
from scripts._m14_l049_v2_schema import V2_ADDENDUM_PATH, canonical_json_bytes
from scripts._m14_l049_v2_stage_a import attempted_real_resources, build_stage_a_artifact
from scripts._m14_l049_v2_validate import validate_stage_a


def _source_commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None or value == "0" * 40:
        raise argparse.ArgumentTypeError("source commit must be a nonzero lowercase 40-hex SHA")
    return value


def _write_attempt_triad(output: Path, artifact: dict[str, object], source_commit_sha: str) -> None:
    pattern = re.compile(r"^l04-explanations\.L049V2StageA\.attempt([0-9]+)\.(?:partial|run|failure)\.json$")
    attempts = [
        int(match.group(1))
        for path in output.parent.glob("l04-explanations.L049V2StageA.attempt*.json")
        if (match := pattern.fullmatch(path.name))
    ]
    attempt = max(attempts, default=0) + 1
    prefix = output.parent / f"l04-explanations.L049V2StageA.attempt{attempt}"
    common = {
        "schema_version": "m14-l04.9-v2-transport-envelope-v1",
        "use_case": "L049V2StageA",
        "attempt": f"attempt{attempt}",
        "source_commit_sha": source_commit_sha,
        "stage": artifact["stage"],
        "status": artifact["status"],
        "artifact_sha256": artifact["artifact_sha256"],
    }
    partial = {**common, "kind": "partial", "artifact": artifact}
    run = {**common, "kind": "run", "artifact_name": f"l04-explanations.L049V2StageA.attempt{attempt}.partial.json"}
    failure = {
        **common,
        "kind": "failure",
        "failure_ref": f"l04-explanations.L049V2StageA.attempt{attempt}.failure.json",
    }
    for kind, value in (("partial", partial), ("run", run), ("failure", failure)):
        (Path(f"{prefix}.{kind}.json")).write_bytes(canonical_json_bytes(value) + b"\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit-sha", required=True, type=_source_commit)
    parser.add_argument(
        "--run-real", action="store_true", help="Run the pinned CUDA-only train boundary when available"
    )
    args = parser.parse_args(argv)
    _raw, rows = read_rows(args.train_fixture)
    errors = validate_rows(rows, "train", 36)
    if errors:
        raise SystemExit("; ".join(errors))
    addendum = json.loads(V2_ADDENDUM_PATH.read_bytes())
    source_sha = hashlib.sha256(Path(__file__).with_name("_m14_l049_v2_stage_a.py").read_bytes()).hexdigest()
    cli_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.run_real:
        runtime: dict[str, object] | None = None
        try:
            import torch
        except Exception as error:  # noqa: BLE001 - no CUDA attempt started
            runtime = {
                "error": error,
                "resources": {
                    "stage": "preflight",
                    "execution_attempted": False,
                    "execution_backend": "none",
                    "device": "not used",
                    "network": "not attempted",
                    "no_mutation": True,
                },
            }
        else:
            try:
                cuda_available = bool(torch.cuda.is_available())
            except Exception as error:  # noqa: BLE001 - CUDA preflight failed
                cuda_available = False
                runtime = {
                    "error": error,
                    "resources": {
                        "stage": "preflight",
                        "execution_attempted": False,
                        "execution_backend": "none",
                        "device": "not used",
                        "network": "not attempted",
                        "no_mutation": True,
                    },
                }
            if not cuda_available and runtime is None:
                runtime = {
                    "error": RuntimeError("CUDA is unavailable"),
                    "resources": {
                        "stage": "preflight",
                        "execution_attempted": False,
                        "execution_backend": "none",
                        "device": "not used",
                        "network": "not attempted",
                        "no_mutation": True,
                    },
                }
            elif cuda_available:
                try:
                    from scripts._m14_l049_v2_real_runtime import (
                        RealRuntimeError,
                        build_stage_a_runtime,
                    )

                    try:
                        scorer, resources = build_stage_a_runtime(rows)
                        runtime = {"score": scorer, "resources": resources}
                    except RealRuntimeError as error:
                        runtime = {"error": error.original_error, "resources": error.resources}
                    except Exception as error:  # noqa: BLE001 - post-CUDA failures are attempted-real
                        runtime = {"error": error, "resources": attempted_real_resources()}
                except Exception as error:  # noqa: BLE001 - helper import/resolution is attempted-real
                    runtime = {"error": error, "resources": attempted_real_resources()}
        assert runtime is not None
        try:
            from scripts._m14_l049_v2_stage_a import run_real_stage_a

            artifact = run_real_stage_a(rows, addendum, source_sha256=source_sha, runtime=runtime, cli_sha256=cli_sha)
        except Exception as error:  # noqa: BLE001 - every post-CUDA helper failure is attempted-real
            from scripts._m14_l049_v2_stage_a import build_stage_a_failure_artifact

            runtime_resources = runtime.get("resources")
            if not isinstance(runtime_resources, Mapping):
                runtime_resources = attempted_real_resources()
            artifact = build_stage_a_failure_artifact(
                rows,
                addendum,
                source_sha256=source_sha,
                error=error,
                resources=runtime_resources,
                cli_sha256=cli_sha,
            )
    else:
        artifact = build_stage_a_artifact(rows, addendum, source_sha256=source_sha, cli_sha256=cli_sha)
    validation = validate_stage_a(artifact, rows, addendum)
    if validation:
        raise SystemExit("; ".join(validation))
    args.output.write_bytes(canonical_json_bytes(artifact) + b"\n")
    _write_attempt_triad(args.output, artifact, args.source_commit_sha)
    print(
        json.dumps(
            {
                "stage": (
                    "stage_a_real_train_selection"
                    if args.run_real and artifact.get("resources", {}).get("execution_backend") == "cuda"
                    else "stage_a_train_selection_protocol_fixture"
                ),
                "status": artifact["status"],
                "artifact_sha256": artifact["artifact_sha256"],
                "cli_sha256": cli_sha,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
