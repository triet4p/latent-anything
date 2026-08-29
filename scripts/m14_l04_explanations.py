"""M14 L04 dispatcher and failure-preserving artifact writer.

Computation handlers are intentionally supplied by L04.4--L04.10. This lane
only validates the frozen inputs, dispatches one use case, and writes honest
non-promoting envelopes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts._m14_l04_data import fixture_metadata
from scripts._m14_l04_envelope import (
    build_artifact,
    build_run_record,
    canonical_digest,
    failure_envelope,
    safe_write,
)
from scripts._m14_l04_fixture_contract import read_fixture
from scripts._m14_l04_validate import validate_artifact, validate_failure, validate_run_record
from scripts.m14_l04_contract import FIXTURE_PATH, PLAN_PATH, load_and_validate, load_plan

__all__ = [
    "check",
    "run_real",
    "run_one_use_case",
    "failure_envelope",
    "validate_artifact",
    "validate_failure",
    "validate_run_record",
]

USE_CASES = (
    "IntegratedGradients",
    "TCAV",
    "DirectLogitLens",
    "TunedLogitLens",
    "Disentanglement",
    "TrueActivationPatching",
    "AdditiveSteering",
)
PENDING = {
    "IntegratedGradients": "not_implemented_pending_L04.4",
    "TCAV": "not_implemented_pending_L04.5",
    "DirectLogitLens": "not_implemented_pending_L04.6",
    "TunedLogitLens": "blocked_missing_corpus",
    "Disentanglement": "not_implemented_pending_L04.8",
    "TrueActivationPatching": "not_implemented_pending_L04.9",
    "AdditiveSteering": "not_implemented_pending_L04.10",
}

Handler = Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]]


def check(plan_path: Path = PLAN_PATH, fixture_path: Path = FIXTURE_PATH) -> dict[str, str]:
    """Perform the existing side-effect-free offline contract check."""
    return load_and_validate(plan_path, fixture_path)


def _attempt(output_dir: Path, use_case: str) -> int:
    prefix = f"l04-explanations.{use_case}.attempt"
    numbers = []
    for path in output_dir.glob(f"{prefix}*.partial.json"):
        try:
            numbers.append(int(path.name[len(prefix) :].split(".", 1)[0]))
        except ValueError:
            continue
    return max(numbers, default=0) + 1


def run_real(
    *,
    plan_path: Path = PLAN_PATH,
    fixture_path: Path = FIXTURE_PATH,
    use_case: str,
    output_dir: Path = Path("artifacts/m14"),
    handlers: dict[str, Handler] | None = None,
) -> dict[str, Any]:
    """Dispatch one use case without loading a model in this infrastructure task."""
    if use_case not in USE_CASES:
        raise ValueError(f"unknown use case {use_case!r}")
    # Full contract validation is the preflight barrier: no handler, attempt
    # allocation, or output write is allowed when plan/fixture bytes are bad.
    load_and_validate(plan_path, fixture_path)
    plan = load_plan(plan_path)
    raw, rows = read_fixture(fixture_path)
    output_dir = Path(output_dir)
    attempt = _attempt(output_dir, use_case)
    stem = f"l04-explanations.{use_case}.attempt{attempt}"
    partial_name = f"{stem}.partial.json"
    run_name = f"{stem}.run.json"
    failure_name = f"{stem}.failure.json"
    handler = None if handlers is None else handlers.get(use_case)
    # Preserve the offline dispatcher behavior unless the caller has
    # explicitly opted into the real CUDA/network lane.
    if handlers is None and os.environ.get("LATENT_ANYTHING_RUN_NETWORK") == "1":
        if use_case == "IntegratedGradients":
            from scripts._m14_l04_integrated_gradients import run_integrated_gradients

            handler = run_integrated_gradients
        elif use_case == "TCAV":
            from scripts._m14_l04_tcav import run_tcav

            handler = run_tcav
        elif use_case == "DirectLogitLens":
            from scripts._m14_l04_direct_lens import run_direct_logit_lens

            handler = run_direct_logit_lens
        elif use_case == "TunedLogitLens":
            from scripts._m14_l04_tuned_lens import run_tuned_logit_lens

            handler = run_tuned_logit_lens
        elif use_case == "Disentanglement":
            from scripts._m14_l04_disentanglement import run_disentanglement

            handler = run_disentanglement
    status = PENDING[use_case]
    injected = handlers is not None
    error: BaseException | None = None
    handler_result: dict[str, Any] = {}
    handler_result_digest: str | None = None
    resources: dict[str, Any] = {
        "device": "not used",
        "network": "not attempted",
        "resource_peak": "not measured",
        "cleanup": "not applicable; no model was loaded",
        "execution_attempted": bool(handler is not None and not injected),
        "execution_backend": "cuda" if handler is not None and not injected else "none",
        "stage": "dispatch",
    }
    if handler is not None:
        try:
            handler_result = dict(handler(plan, rows))
            if handler_result:
                handler_result_digest = hashlib.sha256(
                    json.dumps(handler_result, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            status = (
                "injected_offline_non_eligible" if handlers is not None else str(handler_result.get("status", "failed"))
            )
            if status == "failed":
                error = RuntimeError(str(handler_result.get("failure_reason", "real execution failed")))
            if isinstance(handler_result.get("resources"), dict):
                resources.update(handler_result["resources"])
        except Exception as exc:  # noqa: BLE001 - retain every injected failure
            error = exc
            status = "failed"
            handler_result = {}
            error_resources = getattr(exc, "resources", None)
            if isinstance(error_resources, dict):
                resources.update(error_resources)
            if resources.get("stage") == "dispatch" and resources.get("execution_attempted") is True:
                resources["stage"] = "execution"
    fixture = fixture_metadata(plan, raw, rows)
    artifact = build_artifact(
        plan,
        fixture,
        use_case,
        status,
        failure_name,
        injected=injected,
        execution_result=handler_result if handler_result and not injected else None,
        resources=resources,
    )
    provenance = artifact["provenance"]
    if handler_result_digest is not None:
        provenance["injected_handler_result_digest" if injected else "execution_result_digest"] = handler_result_digest
        artifact["artifact_sha256"] = canonical_digest(artifact, "artifact_sha256")
    safe_write(output_dir / partial_name, artifact)
    run = build_run_record(plan, artifact, use_case, status, resources, artifact_name=partial_name)
    safe_write(output_dir / run_name, run)
    failure = failure_envelope(
        plan,
        use_case,
        status,
        error=error,
        failure_ref=failure_name,
        run_record=run,
        resources=resources,
    )
    safe_write(output_dir / failure_name, failure)
    return {
        "status": status,
        "use_case": use_case,
        "artifact": artifact,
        "run_record": run,
        "failure": failure,
        "paths": {"partial": partial_name, "run": run_name, "failure": failure_name},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--use-case", choices=USE_CASES)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/m14"))
    args = parser.parse_args(argv)
    if args.check and args.run_real:
        parser.error("choose only --check or --run-real")
    if args.check:
        print(json.dumps(check(args.plan, args.fixture), sort_keys=True, separators=(",", ":")))
        return
    if not args.run_real:
        parser.error("choose --check or --run-real")
    if args.use_case is None:
        parser.error("--run-real requires --use-case")
    result = run_real(
        plan_path=args.plan, fixture_path=args.fixture, use_case=args.use_case, output_dir=args.output_dir
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    plan = load_plan(args.plan)
    valid = (
        validate_artifact(result["artifact"], plan)
        + validate_run_record(result["run_record"], result["artifact"], plan)
        + validate_failure(result["failure"], plan, result["artifact"])
    )
    if valid or result["status"] != "passed_real_cuda":
        raise SystemExit(1)


run_one_use_case = run_real


if __name__ == "__main__":
    main()
