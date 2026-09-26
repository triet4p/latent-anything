"""Consumer-observable tests for the Sprint 80.22 deterministic report renderer."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_diagnostic_persistence import (  # noqa: E402 - sibling helper lives outside the package
    _completed_run,
    completed_run_full_outcomes,
    failed_control_run,
)

from latent_anything._diagnostic_artifact import (  # noqa: E402 - after test-dir path shim
    DiagnosticArtifactError,
    persist_diagnostic_artifact,
)
from latent_anything._report_renderer import (  # noqa: E402 - after test-dir path shim
    RENDERED_REPORT_SCHEMA,
    RenderedReportHandle,
    ReportRenderingError,
    persist_rendered_report,
    render_diagnostic_report,
)
from latent_anything.diagnostics import OutputSelection  # noqa: E402 - after test-dir path shim

_SECTION_HEADERS = [
    "## Identifiers",
    "## Symptom",
    "## Location",
    "## Evidence",
    "## Causal",
    "## Comparison",
    "## Limitations",
    "## Next action",
]


def _persist(tmp_path: Path, chain: tuple) -> tuple[str, dict[str, object]]:
    request, manifest, result, checkpoint, report = chain
    handle = persist_diagnostic_artifact(
        tmp_path / "root",
        request=request,
        manifest=manifest,
        result=result,
        checkpoint=checkpoint,
        report=report,
    )
    return handle.run_id, manifest


def _render(tmp_path: Path, chain: tuple) -> str:
    run_id, manifest = _persist(tmp_path, chain)
    return render_diagnostic_report(tmp_path / "root", run_id=run_id, manifest=manifest).decode("utf-8")


def test_sections_have_exact_order_and_full_identifiers(tmp_path: Path) -> None:
    text = _render(tmp_path, _completed_run())
    lines = text.splitlines()
    assert lines[0].startswith("# Diagnostic Report ")
    headers = [line for line in lines if line.startswith("## ")]
    assert headers == _SECTION_HEADERS
    assert f"render schema: {RENDERED_REPORT_SCHEMA}" in lines
    for key in (
        "artifact schema: diagnostic-artifact-v1",
        "report schema: diagnostic-report-schema-v1",
        "validator: passed",
    ):
        assert key in lines, key
    digest_lines = [
        line
        for line in lines
        if re.match(
            r"^(artifact|report|manifest|taxonomy|report schema|workflow) (digest|sha256|identity): [0-9a-f]{64}$", line
        )
    ]
    assert len(digest_lines) == 6, digest_lines
    assert any(line.startswith("run id: ") for line in lines)
    assert any(line.startswith("manifest: sprint80-core-encoder") for line in lines)
    assert any(line.startswith("request id: ") for line in lines)


def test_all_four_outcomes_render_without_promotion(tmp_path: Path) -> None:
    text = _render(tmp_path, completed_run_full_outcomes())
    assert "iv-ablate-record, conclusion supported" in text
    assert "iv-inert-record, conclusion falsified" in text
    assert "iv-blocked-record, conclusion unsupported" in text
    assert "comparison cmp-inconclusive: inconclusive (classification inconclusive)" in text
    assert "comparison cmp-both: observed (classification both)" in text
    # Claim promotion flags stay truthful per outcome.
    assert "claim intervention-iv-ablate: supported (claim allowed yes)" in text
    assert "claim intervention-iv-inert: falsified (claim allowed yes)" in text
    assert "claim intervention-iv-blocked: unsupported (claim allowed no)" in text
    # Negative results are explicit, never omitted.
    assert "negative result: intervention intervention-iv-inert-record — falsified" in text
    assert "negative result: intervention intervention-iv-blocked-record — unsupported" in text
    assert "negative result: comparison cmp-inconclusive — inconclusive" in text
    # Aligned representation-vs-task comparison with signed deltas.
    assert "representation bottleneck-effective-rank: signed delta " in text
    assert "task bottleneck-singular-spread: signed delta " in text
    assert "tolerance 0.1" in text and "tolerance 0.05" in text


def test_failed_control_never_renders_as_success(tmp_path: Path) -> None:
    text = _render(tmp_path, failed_control_run())
    assert "control control-healthy-counterexample: failed" in text
    assert "control control-benign-low-variance: passed" in text
    assert "bounded conclusion: inconclusive" in text
    assert "bounded conclusion: supported" not in text
    assert "causal promotion blocked: required control(s) control-healthy-counterexample failed" in text
    assert "no success is claimed" in text


def test_symptom_location_evidence_and_next_action_sections(tmp_path: Path) -> None:
    text = _render(tmp_path, _completed_run())
    assert "family: collapse_rank_loss" in text
    assert "bounded conclusion: supported" in text
    assert "description: bottleneck rank collapse" in text
    assert "axes: sample, feature" in text
    assert "location: slice slice-back (confidence 0.9, status observed)" in text
    assert "metric bottleneck-effective-rank: estimate 2.0, interval [1.9, 2.1], status observed" in text
    assert "family evidence collapse_rank_loss:" in text
    assert "explanation h-collapse-probe: probe outcome supported" in text
    assert "action: render the AI-engineer report" in text
    assert "rationale: 80.22 owns rendering" in text
    assert "requires evidence e-1" in text
    assert "requires evidence o-1" in text
    assert "limitation lim-1:" in text and "blocking no" in text


def test_supported_location_status_is_rendered(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    localization = report["localization"]
    assert isinstance(localization, list)
    location = localization[0]
    assert isinstance(location, dict)
    location["status"] = "supported"

    text = _render(tmp_path, (request, manifest, result, checkpoint, report))

    assert "location: slice slice-back (confidence 0.9, status supported)" in text


def test_evidence_links_resolve_and_rehash(tmp_path: Path) -> None:
    run_id, _ = _persist(tmp_path, _completed_run())
    text = render_diagnostic_report(tmp_path / "root", run_id=run_id, manifest=_completed_run()[1]).decode("utf-8")
    links = re.findall(r"^evidence (.+): ([0-9a-f]{64})$", text, flags=re.MULTILINE)
    assert links, "rendered report must index evidence links"
    for name, digest in links:
        blob = tmp_path / "root" / "artifacts" / digest
        assert blob.is_file(), name
        data = blob.read_bytes()
        assert sha256(data).hexdigest() == digest, name
        assert json.loads(data) is not None


def test_malicious_labels_cannot_alter_structure(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = _completed_run()
    hostile = copy.deepcopy(report)
    hostile["symptoms"][0]["description"] = "evil\n## Fake section\n\x1b[31mtrailer\r\tend"
    hostile["hypotheses"][0]["statement"] = "inject\r\n## Other fake"
    handle = persist_diagnostic_artifact(
        tmp_path / "root",
        request=request,
        manifest=manifest,
        result=result,
        checkpoint=checkpoint,
        report=hostile,
    )
    rendered = render_diagnostic_report(tmp_path / "root", run_id=handle.run_id, manifest=manifest)
    text = rendered.decode("utf-8")
    lines = text.splitlines()
    headers = [line for line in lines if line.startswith("## ")]
    assert headers == _SECTION_HEADERS
    assert not any(line.startswith("## Fake") for line in lines)
    assert not any(line.startswith("## Other fake") for line in lines)
    assert "\\n" in text  # escaped, literal
    assert "\x1b" not in text  # raw escape byte never survives
    assert "evil\\n## Fake section\\n\\x1b[31mtrailer\\r\\tend" in text


def test_contradictory_comparison_status_refuses_rendering(tmp_path: Path) -> None:
    request, manifest, result, checkpoint, report = completed_run_full_outcomes()
    contradictory = copy.deepcopy(report)
    row = next(row for row in contradictory["comparisons"] if row["id"] == "cmp-inconclusive")
    row["status"] = "observed"  # disagrees with the recorded inconclusive classification
    handle = persist_diagnostic_artifact(
        tmp_path / "root",
        request=request,
        manifest=manifest,
        result=result,
        checkpoint=checkpoint,
        report=contradictory,
    )
    with pytest.raises(ReportRenderingError, match="disagrees with its recorded"):
        render_diagnostic_report(tmp_path / "root", run_id=handle.run_id, manifest=manifest)


def test_invalid_or_tampered_artifact_refuses_rendering(tmp_path: Path) -> None:
    run_id, manifest = _persist(tmp_path, _completed_run())
    root = tmp_path / "root"
    document = json.loads((root / "artifacts" / _artifact_digest(root, run_id)).read_bytes())
    detect_ref = next(blob for blob in document["blobs"] if blob["name"] == "stage-record-detect")
    (root / "artifacts" / detect_ref["digest"]).write_bytes(b"tampered\n")
    with pytest.raises(DiagnosticArtifactError, match="rejected"):
        render_diagnostic_report(root, run_id=run_id, manifest=manifest)

    # A manifest that disagrees with the persisted hash refuses rendering.
    clean = tmp_path / "clean"
    clean_handle = persist_diagnostic_artifact(clean, **_chain_kwargs(_completed_run()))
    other_manifest = copy.deepcopy(manifest)
    other_manifest["manifest_id"] = "other-manifest"
    with pytest.raises(DiagnosticArtifactError, match="manifest hash disagrees"):
        render_diagnostic_report(clean, run_id=clean_handle.run_id, manifest=other_manifest)


def _artifact_digest(root: Path, run_id: str) -> str:
    record = json.loads((root / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))
    return str(record["metadata"]["diagnostic_artifact_digest"])


def _chain_kwargs(chain: tuple) -> dict[str, object]:
    request, manifest, result, checkpoint, report = chain
    return {
        "request": request,
        "manifest": manifest,
        "result": result,
        "checkpoint": checkpoint,
        "report": report,
    }


def test_deterministic_replay_and_idempotent_persist(tmp_path: Path) -> None:
    run_id, manifest = _persist(tmp_path, _completed_run())
    root = tmp_path / "root"
    first = render_diagnostic_report(root, run_id=run_id, manifest=manifest)
    second = render_diagnostic_report(root, run_id=run_id, manifest=manifest)
    assert first == second

    output_dir = "artifacts/diagnostics/test-80-22-render"
    output = OutputSelection(output_location=output_dir, artifact_name="diagnostic-report", include_report=True)
    try:
        handle_one = persist_rendered_report(root, run_id=run_id, manifest=manifest, output=output)
        declared = Path(output_dir) / "diagnostic-report"
        assert declared.read_bytes() == first
        blob = root / "artifacts" / handle_one.digest
        assert blob.read_bytes() == first
        handle_two = persist_rendered_report(root, run_id=run_id, manifest=manifest, output=output)
        assert handle_two.digest == handle_one.digest
        assert isinstance(handle_one, RenderedReportHandle)
        assert handle_one.relative_path == f"artifacts/{handle_one.digest}"
        assert handle_one.artifact_name == "diagnostic-report.rendered"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_output_selection_and_hygiene_fail_closed(tmp_path: Path) -> None:
    run_id, manifest = _persist(tmp_path, _completed_run())
    root = tmp_path / "root"
    no_report = OutputSelection(
        output_location="artifacts/diagnostics/test-80-22-none",
        artifact_name="diagnostic-report",
        include_report=False,
    )
    with pytest.raises(ReportRenderingError, match="does not request a report"):
        persist_rendered_report(root, run_id=run_id, manifest=manifest, output=no_report)
    nested = OutputSelection(
        output_location="artifacts/diagnostics/test-80-22-nested",
        artifact_name="nested/name",
        include_report=True,
    )
    with pytest.raises(ReportRenderingError, match="simple file name"):
        persist_rendered_report(root, run_id=run_id, manifest=manifest, output=nested)

    text = render_diagnostic_report(root, run_id=run_id, manifest=manifest).decode("utf-8")
    for pattern in ("F:\\\\", "C:\\\\", "/home/", "/Users/"):
        assert pattern not in text, pattern
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text)
    for key, value in os.environ.items():
        if any(token in key for token in ("PATH", "SECRET", "TOKEN", "KEY", "HOME")) and len(value) >= 8:
            assert value not in text, key


def test_public_surface_unchanged() -> None:
    import latent_anything as la

    expected = {
        "CaptureSelection",
        "ComparisonRequest",
        "ControlSelection",
        "DiagnosticRequest",
        "DiagnosticRequestError",
        "DiagnosticResult",
        "DiagnosticSelection",
        "InterventionRequest",
        "OutputSelection",
    }
    assert len([name for name in la.__all__ if name in expected]) == 9
    for leaked in (
        "render_diagnostic_report",
        "persist_rendered_report",
        "ReportRenderingError",
        "RenderedReportHandle",
    ):
        assert not hasattr(la, leaked), leaked


def _tamper(
    root: Path,
    run_id: str,
    *,
    blob_edits: dict[str, bytes] | None = None,
    report_edits: Callable[[dict[str, object]], None] | None = None,
) -> None:
    """Rewrite a persisted artifact consistently after content tampering.

    Updates blob bytes, the inventory, report and validator digests, the
    artifact document, and the run metadata (with a valid identity) so the
    loader accepts the tampered artifact and only the renderer's new
    record-content cross-checks can refuse it.
    """
    from test_diagnostic_persistence import _rewrite_run_metadata

    from latent_anything._run_record_codec import canonical_json

    artifacts = root / "artifacts"
    document_path = artifacts / _artifact_digest(root, run_id)
    document = json.loads(document_path.read_text(encoding="utf-8"))
    inventory: list[dict[str, object]] = document["blobs"]
    by_name = {str(row["name"]): row for row in inventory}

    def _store(data: bytes) -> str:
        digest = sha256(data).hexdigest()
        target = artifacts / digest
        if not target.exists():
            target.write_bytes(data)
        return digest

    for name, data in (blob_edits or {}).items():
        digest = _store(data)
        row = by_name[name]
        row["digest"] = digest
        row["size_bytes"] = len(data)
        row["relative_path"] = f"artifacts/{digest}"
    if report_edits is not None:
        report_row = by_name["diagnostic-report"]
        report = json.loads((artifacts / str(report_row["digest"])).read_bytes())
        report_edits(report)
        report_bytes = canonical_json(report) + b"\n"
        digest = _store(report_bytes)
        report_row["digest"] = digest
        report_row["size_bytes"] = len(report_bytes)
        report_row["relative_path"] = f"artifacts/{digest}"
        document["report_digest"] = digest
        document["validator_result"]["report_digest"] = digest  # type: ignore[index]
    document["validator_input"]["artifact_digests"] = {  # type: ignore[index]
        str(row["name"]): str(row["digest"]) for row in inventory
    }
    document["validator_result"]["inputs_digest"] = sha256(  # type: ignore[index]
        canonical_json(
            {
                "applicability": document["validator_input"]["applicability"],  # type: ignore[index]
                "artifact_digests": document["validator_input"]["artifact_digests"],  # type: ignore[index]
                "control_outcomes": document["validator_input"]["control_outcomes"],  # type: ignore[index]
                "family_evidence": document["validator_input"]["family_evidence"],  # type: ignore[index]
            }
        )
        + b"\n"
    ).hexdigest()
    document_bytes = canonical_json(document) + b"\n"
    document_digest = _store(document_bytes)
    _rewrite_run_metadata(
        root / "runs" / f"{run_id}.json",
        digest=document_digest,
        size=len(document_bytes),
    )


def test_supported_row_vs_falsified_record_refuses_rendering(tmp_path: Path) -> None:
    from hashlib import sha256 as _sha256

    from latent_anything._run_record_codec import canonical_json

    run_id, manifest = _persist(tmp_path, _completed_run())
    root = tmp_path / "root"
    document = json.loads((root / "artifacts" / _artifact_digest(root, run_id)).read_bytes())
    trial_ref = next(row for row in document["blobs"] if row["name"] == "stage-record-intervene-iv-ablate")
    trial = json.loads((root / "artifacts" / trial_ref["digest"]).read_bytes().decode("utf-8"))
    trial["conclusion"] = "falsified"
    tampered_trial_bytes = canonical_json(trial) + b"\n"
    tampered_digest = _sha256(tampered_trial_bytes).hexdigest()

    def _edit_report(report: dict[str, object]) -> None:
        row = next(item for item in report["interventions"] if item["id"] == "intervention-iv-ablate-record")
        # Row still claims supported; only the record now says falsified.
        row["outcome"]["record_digest"] = tampered_digest  # type: ignore[index]

    _tamper(
        root,
        run_id,
        blob_edits={"stage-record-intervene-iv-ablate": tampered_trial_bytes},
        report_edits=_edit_report,
    )
    with pytest.raises(ReportRenderingError, match="disagrees with record conclusion"):
        render_diagnostic_report(root, run_id=run_id, manifest=manifest)

    # Refusal happens before any output materialization.
    output_dir = "artifacts/diagnostics/test-80-22-refused"
    output = OutputSelection(output_location=output_dir, artifact_name="diagnostic-report", include_report=True)
    try:
        with pytest.raises(ReportRenderingError, match="disagrees with record conclusion"):
            persist_rendered_report(root, run_id=run_id, manifest=manifest, output=output)
        assert not Path(output_dir).exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_wrong_intervention_id_in_record_refuses_rendering(tmp_path: Path) -> None:
    run_id, manifest = _persist(tmp_path, _completed_run())
    root = tmp_path / "root"

    def _edit_report(report: dict[str, object]) -> None:
        row = next(item for item in report["interventions"] if item["id"] == "intervention-iv-ablate-record")
        row["id"] = "intervention-other-record"
        claim = next(item for item in report["claims"] if item["id"] == "intervention-iv-ablate")
        claim["evidence_refs"] = ["intervention-other-record"]

    _tamper(root, run_id, report_edits=_edit_report)
    with pytest.raises(ReportRenderingError, match="record names intervention"):
        render_diagnostic_report(root, run_id=run_id, manifest=manifest)


def test_unrelated_registered_blob_refuses_rendering(tmp_path: Path) -> None:
    run_id, manifest = _persist(tmp_path, _completed_run())
    root = tmp_path / "root"

    def _edit_report(report: dict[str, object]) -> None:
        row = next(item for item in report["interventions"] if item["id"] == "intervention-iv-ablate-record")
        row["evidence_refs"] = ["stage-record-detect"]

    _tamper(root, run_id, report_edits=_edit_report)
    # The unrelated blob is refused: its parsed content names no such
    # intervention (identity cross-check fires before digest comparison).
    with pytest.raises(ReportRenderingError, match="record names intervention"):
        render_diagnostic_report(root, run_id=run_id, manifest=manifest)
