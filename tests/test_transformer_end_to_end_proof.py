"""Focused behavioral coverage for the Sprint80.24 transformer end-to-end proof contract.

These tests defend the real core-proof behavior on the frozen transformer
manifest: the pinned selection and leakage-safe grouping reproduce before any
evidence, the declared positive/counterexample/negative controls record
truthfully, and layer-axis localization names the earliest passing layer.
The seven-stage workflow completes through the generic v2 target-evidence
contract, persists and independently revalidates a content-addressed artifact,
and keeps the historical v1 axes mismatch fail-closed. The tests also reject
forged capture axes, target-label tampering, and sample-order misalignment.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PROOF_PATH = REPO / "scripts" / "sprint80_task80_24_proof.py"


def _load_proof() -> object:
    spec = importlib.util.spec_from_file_location("sprint80_task80_24_proof", PROOF_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


proof = _load_proof()


def test_frozen_inputs_selection_and_grouping_reproduce() -> None:
    manifest = proof.load_manifest()  # type: ignore[attr-defined]
    assert proof.manifest_digest(manifest) == proof.PINNED_MANIFEST_DIGEST  # type: ignore[attr-defined]
    before = proof.frozen_digests()  # type: ignore[attr-defined]
    assert len(before) == 6

    data = proof.build_data()  # type: ignore[attr-defined]
    l04 = data["l04"]
    assert l04["source"]["revision"] == proof.DATASET_REVISION  # type: ignore[index]
    assert l04["splits"]["validation"]["official_rows"] == 3760
    assert l04["splits"]["validation"]["nonblank_rows"] == 2461
    assert l04["splits"]["validation"]["selected_rows"] == 2048
    assert len(np.asarray(data["selected_indices"])) == 2048  # type: ignore[arg-type]
    labels = np.asarray(data["labels"])
    assert np.array_equal(
        labels,
        np.asarray(
            [int(proof.HEADER_PATTERN.match(text) is not None) for text in data["text"]]  # type: ignore[union-attr]
        ),
    )
    train = np.asarray(data["train_positions"])
    evaluation = np.asarray(data["eval_positions"])
    assert len(train) >= 770
    assert set(train) & set(evaluation) == set()
    assert len(np.unique(labels[train])) == 2 and len(np.unique(labels[evaluation])) == 2
    selected = np.asarray(data["selected_indices"])
    train_groups = {int(i) // 8 for i in selected[train]}
    eval_groups = {int(i) // 8 for i in selected[evaluation]}
    assert train_groups and eval_groups and not train_groups & eval_groups
    train_identity, eval_identity = proof.split_identities()  # type: ignore[attr-defined]
    assert train_identity != eval_identity
    assert proof.frozen_digests() == before  # type: ignore[attr-defined]


def test_capture_identity_and_declared_control_cases_record_truthfully() -> None:
    concrete, provenance = proof._capture()  # type: ignore[attr-defined]
    assert concrete.capture_identity == proof._capture()[0].capture_identity  # type: ignore[attr-defined]
    assert tuple(concrete.shape) == (2048, 768)
    assert provenance["capture_id"] == proof.CAPTURE_ID  # type: ignore[attr-defined]
    assert provenance["representation_identity"] == proof.REPRESENTATION  # type: ignore[attr-defined]
    assert provenance["model_revision"] == proof.MODEL_REVISION  # type: ignore[attr-defined]
    assert provenance["split_identity"] == proof.SPLIT_IDENTITY  # type: ignore[attr-defined]
    assert provenance["axes"] == ["slice", "feature"]
    assert "layer" in provenance["absent_axes"]  # selection axis, not captured as an array axis

    detect = proof.detect_case()  # type: ignore[attr-defined]
    family = detect["family"]
    payload = detect["payload"]
    separability = payload["measurements"]["separability"]
    # The frozen positive claim, executed as written.
    assert separability["heldout_accuracy"] >= 0.7
    assert separability["leakage_gap"] >= 0.15
    assert dict(family.threshold_pass) == {
        "heldout-probe-accuracy": True,
        "probe-leakage-gap": True,
    }
    assert family.outcome == "supported" and family.claim_allowed is True
    assert family.missing_evidence == ()
    assert dict(family.control_outcomes) == {
        "control-capacity": "passed",
        "control-label-randomization": "passed",
        "control-nonseparable-negative": "passed",
        "control-split-swap": "passed",
    }
    negative = payload["control_metrics"]["control-nonseparable-negative"]
    assert negative["heldout-probe-accuracy"] < 0.7
    uncertainty = separability["uncertainty"]["heldout-probe-accuracy"]
    assert uncertainty["repetitions"] == 200
    assert uncertainty["lower"] <= uncertainty["mean"] <= uncertainty["upper"]


def test_layer_axis_localization_names_earliest_passing_layer() -> None:
    localize = proof.localize_case()  # type: ignore[attr-defined]
    result = localize["result"]
    payload = localize["payload"]
    accuracies = proof.per_layer_probes()["accuracies"]  # type: ignore[attr-defined]
    earliest = next(layer for layer, value in enumerate(accuracies) if value >= 0.7)
    assert result.verdict == "localized"
    assert payload["earliest_layer"] == f"transformer.h.{earliest}"  # type: ignore[index]
    assert payload["affected_layers"] == [
        f"transformer.h.{layer}" for layer in range(len(accuracies)) if accuracies[layer] >= 0.7
    ]  # type: ignore[index]
    assert payload["metric_id"] == "heldout-probe-accuracy"  # type: ignore[index]
    config = payload["config"]  # type: ignore[index]
    assert config["affected_when"] == "threshold_pass"
    assert config["threshold_value"] == 0.7
    assert config["tolerance"] == 0.02
    assert config["direction"] == "higher_is_better"
    rows = payload["report_localization"]  # type: ignore[index]
    layer_rows = [row for row in rows if row["axis"] == "layer"]
    slice_rows = [row for row in rows if row["axis"] == "slice"]
    assert layer_rows[0]["selection"] == f"transformer.h.{earliest}"
    assert {row["selection"] for row in slice_rows} == {proof.EVAL_SLICE_ID}  # type: ignore[attr-defined]
    assert len(payload["sample_cells_by_sample_id"]) == 499  # type: ignore[index]
    slice_mean = payload["slice_means"][proof.EVAL_SLICE_ID]  # type: ignore[index]
    assert slice_mean == pytest.approx(accuracies[earliest])

    _result_again, rerun = proof.evaluate_localization(localize["source"])  # type: ignore[attr-defined]
    assert proof._payload_sha(rerun) == proof._payload_sha(payload)  # type: ignore[attr-defined]


def test_workflow_persists_and_revalidates_versioned_target_evidence(tmp_path: Path) -> None:
    from copy import deepcopy
    from hashlib import sha256

    from latent_anything._diagnostic_artifact import (
        DiagnosticArtifactError,
        load_diagnostic_artifact,
        persist_diagnostic_artifact,
        registered_artifact_bytes,
    )
    from latent_anything._diagnostic_report import validate_report_shape
    from latent_anything._diagnostic_validator import ReportValidationError, validate_diagnostic_report
    from latent_anything._diagnostic_workflow import WORKFLOW_STAGES
    from latent_anything._intervention_trials import intervention_report_items
    from latent_anything._run_comparison import comparison_report_items
    from latent_anything._target_evidence import target_record_digest, target_sample_digest

    manifest = proof.load_manifest()
    workflow, request = proof.build_workflow()  # type: ignore[attr-defined]
    result, checkpoint = workflow.run(request, manifest)
    assert result.status == "completed", checkpoint.failure
    assert tuple(checkpoint.completed_stages) == WORKFLOW_STAGES
    payloads = {item.stage: dict(item.payload) for item in checkpoint.outputs}
    assert payloads["detect"]["families"][0]["family_id"] == proof.FAMILY  # type: ignore[attr-defined]
    assert payloads["localize"]["verdict"] == "localized"  # type: ignore[index]
    assert payloads["report"]["report_id"] == proof.REPORT_ID  # type: ignore[attr-defined]

    interventions = payloads["intervene"]["trials"]
    comparisons = payloads["compare"]["comparisons"]
    report = proof.build_report(  # type: ignore[attr-defined]
        checkpoint.outputs,
        intervention_claims=intervention_report_items(interventions),
        comparison_rows=comparison_report_items(comparisons),
    )
    validate_report_shape(report)
    assert report["evidence_contract"] == "diagnostic-evidence-v2"
    target_block = report["target_evidence"]
    assert isinstance(target_block, dict)
    assert target_block["target_id"] == "section-header-attribute"
    assert target_block["rule"] == f"section-header attribute {proof.HEADER_PATTERN.pattern!r}"  # type: ignore[attr-defined]

    trial = interventions[0]
    assert trial["conclusion"] in ("supported", "falsified", "inconclusive", "unsupported")
    assert trial["target"] == payloads["localize"]["earliest_layer"]  # type: ignore[index]
    assert set(trial["control_outcomes"]) == set(proof.TRIAL_CONTROLS)  # type: ignore[attr-defined]
    assert comparisons[0]["classification"] in (
        "representation_only",
        "task_only",
        "both",
        "neither",
        "inconclusive",
        "unsupported",
    )

    root = tmp_path / "v2-root"
    handle = persist_diagnostic_artifact(
        root,
        request=request,
        manifest=manifest,
        result=result,
        checkpoint=checkpoint,
        report=report,  # type: ignore[arg-type]
    )
    loaded = load_diagnostic_artifact(root, run_id=handle.run_id, manifest=manifest)
    assert loaded.artifact_digest == handle.artifact_digest
    assert loaded.document["validator_result"]["status"] == "passed"  # type: ignore[index]
    blobs = registered_artifact_bytes(loaded, root)
    loaded_block = loaded.report["target_evidence"]  # type: ignore[index]
    assert isinstance(loaded_block, dict)
    target_ref = loaded_block["evidence_refs"][0]
    target_bytes = blobs[target_ref]
    target_record = json.loads(target_bytes.decode("utf-8"))
    target_record["record_digest"] = sha256(target_bytes).hexdigest()
    detect_stage = json.loads(blobs["stage-record-detect"].decode("utf-8"))
    persisted_detect = detect_stage["payload"]
    target_provenance = persisted_detect["target_provenance"]
    assert isinstance(target_provenance, dict)
    assert target_provenance["target_id"] == proof.TARGET_ID
    assert target_provenance["rule"] == proof.TARGET_RULE
    assert target_provenance["rule_kind"] == proof.TARGET_RULE_KIND
    assert target_provenance == payloads["detect"]["target_provenance"]
    assert target_record["labels"] == target_provenance["labels"]
    assert target_record["train_indices"] == target_provenance["train_indices"]
    assert target_record["eval_indices"] == target_provenance["eval_indices"]
    assert target_record["dataset_split_identity"] == manifest["dataset"]["split_identity"]  # type: ignore[index]
    assert target_record["rule_digest"] == sha256(target_record["rule"].encode("utf-8")).hexdigest()
    assert target_record["sample_digest"] == target_sample_digest(target_record["sample_ids"])
    assert target_record["record_digest"] == target_record_digest(
        {key: value for key, value in target_record.items() if key != "record_digest"}
    )

    validator_input = loaded.document["validator_input"]  # type: ignore[index]

    def validate_v2(report_to_validate: dict[str, object], target_to_validate: dict[str, object]) -> None:
        validate_diagnostic_report(
            report_to_validate,
            manifest,
            applicability=validator_input["applicability"],  # type: ignore[index]
            family_evidence=validator_input["family_evidence"],  # type: ignore[index]
            control_outcomes=validator_input["control_outcomes"],  # type: ignore[index]
            artifacts=blobs,
            artifact_digests=validator_input["artifact_digests"],  # type: ignore[index]
            target_evidence=target_to_validate,
            target_provenance=target_provenance,
        )

    clean_report = deepcopy(dict(loaded.report))
    validate_v2(clean_report, target_record)
    forged_rule_record = deepcopy(target_record)
    forged_rule_record["rule"] = "forged target rule"
    forged_rule_record["rule_digest"] = sha256(b"forged target rule").hexdigest()
    forged_rule_body = {key: value for key, value in forged_rule_record.items() if key != "record_digest"}
    forged_rule_record["record_digest"] = target_record_digest(forged_rule_body)
    forged_rule_report = deepcopy(clean_report)
    forged_rule_block = forged_rule_report["target_evidence"]
    forged_rule_block["rule"] = forged_rule_record["rule"]  # type: ignore[index]
    forged_rule_block["rule_digest"] = forged_rule_record["rule_digest"]  # type: ignore[index]
    forged_rule_block["record_digest"] = forged_rule_record["record_digest"]  # type: ignore[index]
    with pytest.raises(ReportValidationError, match="evaluated target provenance at rule"):
        validate_v2(forged_rule_report, forged_rule_record)

    # "layer" is a manifest-declared activation axis, but is not present on
    # this concrete capture. The validator must compare against its registered
    # bound-capture payload, not merely accept an allowed axis name.
    forged_axis_report = deepcopy(clean_report)
    captures = forged_axis_report["capture_provenance"]["captures"]  # type: ignore[index]
    captures[0]["axes"].append("layer")  # type: ignore[index,union-attr]
    with pytest.raises(ReportValidationError, match="capture axes disagree with the registered bound capture"):
        validate_v2(forged_axis_report, target_record)

    tampered_record = deepcopy(target_record)
    tampered_record["labels"][0] = 1 - tampered_record["labels"][0]  # type: ignore[index,operator]
    with pytest.raises(ReportValidationError, match="target evidence is invalid"):
        validate_v2(clean_report, tampered_record)

    misaligned_record = {key: value for key, value in target_record.items() if key != "record_digest"}
    sample_ids = list(misaligned_record["sample_ids"])  # type: ignore[arg-type]
    sample_ids[0], sample_ids[1] = sample_ids[1], sample_ids[0]
    misaligned_record["sample_ids"] = sample_ids
    misaligned_record["sample_digest"] = target_sample_digest(sample_ids)
    misaligned_digest = target_record_digest(misaligned_record)
    misaligned_input = {**misaligned_record, "record_digest": misaligned_digest}
    misaligned_report = deepcopy(clean_report)
    misaligned_block = misaligned_report["target_evidence"]  # type: ignore[index]
    misaligned_block["sample_digest"] = misaligned_record["sample_digest"]  # type: ignore[index]
    misaligned_block["record_digest"] = misaligned_digest  # type: ignore[index]
    with pytest.raises(ReportValidationError, match="does not match evaluated target provenance at sample_digest"):
        validate_v2(misaligned_report, misaligned_input)

    # Historical v1 has no target-evidence side channel and therefore retains
    # the original taxonomy applicability failure without writing any files.
    legacy_report = deepcopy(clean_report)
    legacy_report.pop("evidence_contract")
    legacy_report.pop("target_evidence")
    legacy_root = tmp_path / "legacy-v1-root"
    with pytest.raises(DiagnosticArtifactError, match="taxonomy applicability mismatch"):
        persist_diagnostic_artifact(
            legacy_root,
            request=request,
            manifest=manifest,
            result=result,
            checkpoint=checkpoint,
            report=legacy_report,  # type: ignore[arg-type]
        )
    assert not (legacy_root / "runs").exists()
    assert not (legacy_root / "artifacts").exists()


def test_fail_closed_tamper_leakage_and_failed_control_cases() -> None:
    from latent_anything._benchmark_manifest import BenchmarkManifestValidationError, validate_manifest
    from latent_anything._capture_binding import CaptureBindingError, bind_selection
    from latent_anything._diagnostic_workflow import WorkflowError
    from latent_anything._redundancy_separability_detection import DECLARED_CAPACITY, DetectionError, LabeledBatch
    from latent_anything.diagnostics import CaptureSelection, OutputSelection

    manifest = proof.load_manifest()
    mutated = json.loads(
        (REPO / "artifacts" / "benchmark_manifest_sprint80_transformer_hidden_state_v1.json").read_text(
            encoding="utf-8"
        )
    )
    mutated["thresholds"][0]["value"] = 0.01  # type: ignore[index]
    with pytest.raises(BenchmarkManifestValidationError, match="manifest digest"):
        validate_manifest(mutated)

    with pytest.raises(CaptureBindingError, match="provenance mismatch"):
        bind_selection(
            CaptureSelection(
                capture_id=proof.CAPTURE_ID,  # type: ignore[attr-defined]
                representation_identity="openai-community-gpt2:hidden-states:layers0-10:dim768",
                axes=("slice", "feature"),
            ),
            manifest=manifest,
            request_id=proof.REQUEST_ID,  # type: ignore[attr-defined]
        )

    data = proof.build_data()  # type: ignore[attr-defined]
    labels = tuple(int(item) for item in np.asarray(data["labels"]))
    sample_ids = tuple(f"validation-{int(i)}" for i in np.asarray(data["selected_indices"]))
    value = proof.LatentValue(proof._batch(0), proof._space())  # type: ignore[attr-defined]
    with pytest.raises(DetectionError, match="split leaks"):
        LabeledBatch(
            value,
            labels,
            sample_ids,
            tuple(range(2048 - 1)),
            tuple(range(10, 2048)),
            "leak-train",
            "leak-eval",
            DECLARED_CAPACITY,
        )

    # A self-comparable negative control must block the supported conclusion.
    from latent_anything._redundancy_separability_detection import (
        detection_config_from_manifest,
        evaluate_detection,
    )

    config = detection_config_from_manifest(proof._detect_request(), manifest)  # type: ignore[attr-defined]
    blocked_detections, _ = evaluate_detection(
        proof._labeled(11),  # type: ignore[attr-defined]
        config,
        controls={"control-nonseparable-negative": proof._labeled(11)},  # type: ignore[attr-defined]
    )
    blocked = blocked_detections[0]
    assert blocked.outcome == "inconclusive" and blocked.claim_allowed is False
    assert any(item.startswith("failed-control:") for item in blocked.missing_evidence)

    # Tampered resume identity rejects before any stage runs.
    workflow, request = proof.build_workflow()  # type: ignore[attr-defined]
    _, checkpoint = workflow.run(request, manifest, stop_after="compare")
    tampered = dataclasses.replace(
        request,
        output=OutputSelection(output_location="artifacts/diagnostics/proof-80-24-tampered"),
    )
    with pytest.raises(WorkflowError, match="request_digest"):
        workflow.resume(checkpoint, tampered, manifest)

    # Persistence refuses an incomplete workflow without writing files.
    import tempfile as _tempfile

    from latent_anything._diagnostic_artifact import DiagnosticArtifactError, persist_diagnostic_artifact
    from latent_anything.diagnostics import DiagnosticResult

    incomplete = DiagnosticResult(
        request_id=request.request_id,
        manifest_id=request.manifest_id,
        status="running",
        completed_stages=tuple(checkpoint.completed_stages),
    )
    with _tempfile.TemporaryDirectory(prefix="proof-80-24-partial-") as tmp:
        root = Path(tmp) / "root"
        with pytest.raises(DiagnosticArtifactError, match="completed"):
            persist_diagnostic_artifact(
                root,
                request=request,
                manifest=manifest,
                result=incomplete,
                checkpoint=checkpoint,
                report={},
            )
        assert not (root / "runs").exists()
        assert not (root / "artifacts").exists()


@pytest.mark.large_download
def test_deterministic_replay_of_hidden_state_extraction() -> None:
    pooled = proof.pooled_features()  # type: ignore[attr-defined]
    digest = proof.pooled_digest(pooled)  # type: ignore[attr-defined]
    assert digest == proof.pooled_digest(proof.pooled_features())  # type: ignore[attr-defined]
    # Fresh re-extraction of a fixed prefix must be bit-identical (no cache).
    fresh_prefix = proof._extract_pooled(rows=8)  # type: ignore[attr-defined]
    assert np.array_equal(fresh_prefix, pooled[:, :8, :])
