"""Run the accepted Sprint 80 transformer v1 / target-evidence-v2 diagnosis.

The case composition uses the shared DiagnosticRequest/DiagnosticWorkflow
chain, then persists, independently reloads/validates, and renders a
content-addressed artifact. This entry point delegates that case-specific
composition instead of copying its capture, statistics, intervention,
comparison, or artifact plumbing.

Run from the repository root after downloading the pinned GPT-2 snapshot.
By default a unique output directory is kept under artifacts/diagnostics;
--output may select another unused, repository-relative directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import cast

from latent_anything._diagnostic_artifact import load_diagnostic_artifact
from latent_anything._report_renderer import render_diagnostic_report

REPO = Path(__file__).resolve().parents[1]
PROOF_NAME = "sprint80_task80_24_proof"
STAGES = ("capture", "detect", "localize", "explain", "intervene", "compare", "report")
MANIFEST_COMMITMENT = "c4773238ddf4d8854cb40ad26811b7b204cab8210ff9fe45f79e83e12da4675a"
TARGET_RECORD_SHA256 = "3e470ef055e2a546408a6f51717bbc5a4d8bf2478410555862045a516955bc19"
TARGET_RULE_SHA256 = "9b12485635decf738c4904702f75d7d5ae8cb6663555e2af5ee66476269f025b"


def _load_proof() -> ModuleType:
    """Load the committed frozen transformer-v1/v2 composition."""
    path = REPO / "scripts" / f"{PROOF_NAME}.py"
    spec = importlib.util.spec_from_file_location(PROOF_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the frozen transformer composition from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"persisted {name} must be an object")
    return value


def _verify_persisted_output(
    root: Path, manifest: Mapping[str, object], rendered_path: Path
) -> tuple[str, str, str, str, int]:
    run_files = sorted((root / "runs").glob("*.json"))
    if len(run_files) != 1:
        raise RuntimeError(f"expected one completed run record, found {len(run_files)}")
    loaded = load_diagnostic_artifact(root, run_id=run_files[0].stem, manifest=manifest)
    document = loaded.document
    result = _mapping(document.get("result"), name="workflow result")
    completed_stages = result.get("completed_stages")
    if result.get("status") != "completed" or completed_stages != list(STAGES):
        raise RuntimeError("persisted artifact does not contain the completed seven-stage workflow")
    validator = _mapping(document.get("validator_result"), name="validator result")
    if validator.get("status") != "passed":
        raise RuntimeError("independent diagnostic-report validation did not pass")
    validator_input = _mapping(document.get("validator_input"), name="validator input")
    controls = _mapping(validator_input.get("control_outcomes"), name="control outcomes")
    if not controls or any(status != "passed" for status in controls.values()):
        raise RuntimeError("one or more declared controls did not pass")
    report = loaded.report
    target = _mapping(report.get("target_evidence"), name="target evidence")
    if (
        target.get("target_id") != "section-header-attribute"
        or target.get("record_digest") != TARGET_RECORD_SHA256
        or target.get("rule_digest") != TARGET_RULE_SHA256
    ):
        raise RuntimeError("report is missing the frozen versioned target evidence")
    hypotheses = report.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise RuntimeError("validated report has no hypothesis")
    if _mapping(hypotheses[0], name="hypothesis").get("status") != "supported":
        raise RuntimeError("validated report does not support the declared separability finding")
    report_digest = document.get("report_digest")
    if not isinstance(report_digest, str):
        raise RuntimeError("persisted report digest is missing")

    rendered = render_diagnostic_report(root, run_id=loaded.run_id, manifest=manifest)
    if rendered != render_diagnostic_report(root, run_id=loaded.run_id, manifest=manifest):
        raise RuntimeError("rendered report is not deterministic")
    if not rendered_path.is_file() or rendered_path.read_bytes() != rendered:
        raise RuntimeError("materialized report differs from the validated deterministic render")
    rendered_digest = sha256(rendered).hexdigest()
    rendered_blob = root / "artifacts" / rendered_digest
    if not rendered_blob.is_file() or rendered_blob.read_bytes() != rendered:
        raise RuntimeError("content-addressed rendered-report blob is missing or mis-hashed")
    blobs = document.get("blobs")
    if not isinstance(blobs, list):
        raise RuntimeError("persisted artifact blob inventory is missing")
    return loaded.run_id, loaded.artifact_digest, report_digest, rendered_digest, len(blobs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_location = (
        f"artifacts/diagnostics/ai-engineer-transformer-v1-evidence-v2-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    parser.add_argument(
        "--output",
        default=default_location,
        help="unused repository-relative output directory (default: a unique timestamped directory)",
    )
    arguments = parser.parse_args()
    if Path.cwd().resolve() != REPO:
        parser.error("run this example from the repository root")
    output_path = Path(arguments.output)
    if not arguments.output.strip() or output_path.is_absolute() or ".." in output_path.parts:
        parser.error("--output must be a non-empty repository-relative path without '..'")
    output_location = output_path.as_posix()
    output_root = REPO / output_path
    if output_root.exists():
        print(
            f"EXAMPLE OUTCOME: REFUSED — output root already exists; choose an unused --output path: {output_location}"
        )
        return 2

    proof = _load_proof()
    proof.__dict__["OUTPUT_LOCATION"] = output_location
    proof_main = cast(Callable[[], int], proof.__dict__["main"])
    captured = StringIO()
    with redirect_stdout(captured):
        status = proof_main()
    proof_output = captured.getvalue()
    print(proof_output, end="")
    if status != 0 or "ACCEPTANCE: PASSED —" not in proof_output:
        print("EXAMPLE OUTCOME: NOT PASSED — inspect the proof output; any partial artifact is retained")
        return 2

    manifest_loader = cast(Callable[[], dict[str, object]], proof.__dict__["load_manifest"])
    manifest = manifest_loader()
    run_id, artifact_digest, report_digest, rendered_digest, blob_count = _verify_persisted_output(
        output_root, manifest, output_root / "diagnostic-report"
    )
    commitment = _mapping(manifest.get("commitment"), name="manifest commitment")
    if commitment.get("manifest_sha256") != MANIFEST_COMMITMENT:
        raise RuntimeError("loaded transformer manifest differs from the frozen commitment")
    print("EXAMPLE OUTCOME: ACCEPTANCE PASSED — all seven stages, controls, v2 validation, and rendering verified")
    print(f"  manifest: {manifest.get('manifest_id')} ({MANIFEST_COMMITMENT})")
    print(f"  output  : {output_location}")
    print(f"  run id  : {run_id}")
    print(f"  artifact SHA-256: {artifact_digest}")
    print(f"  report SHA-256: {report_digest}")
    print(f"  rendered report SHA-256: {rendered_digest}")
    print(f"  independently reloaded content-addressed blobs: {blob_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
