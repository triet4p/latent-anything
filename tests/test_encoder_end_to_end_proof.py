"""Focused behavioral coverage for the Sprint80.23 encoder end-to-end proof contract.

These tests defend the real core-proof behavior on the frozen encoder
manifest: the declared control cases execute with truthful recorded
results, localization is truthfully negative under the frozen estimator,
the seven-stage workflow fails closed (never fabricates a location) when
the explain-stage localization binding cannot resolve, persistence of the
incomplete workflow is refused without writing files, tampered identities
and failed required controls reject fail-closed, downstream measurements
are real and manifest-bound, and the frozen artifacts stay byte-identical
with deterministic replay at the declared seeds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PROOF_PATH = REPO / "scripts" / "sprint80_task80_23_proof.py"


def _load_proof() -> object:
    spec = importlib.util.spec_from_file_location("sprint80_task80_23_proof", PROOF_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


proof = _load_proof()


def test_frozen_inputs_verify_and_replay_is_deterministic() -> None:
    manifest = proof.load_manifest()  # type: ignore[attr-defined]
    assert proof.manifest_digest(manifest) == proof.PINNED_MANIFEST_DIGEST  # type: ignore[attr-defined]
    before = proof.frozen_digests()  # type: ignore[attr-defined]
    assert set(before) == {
        "benchmark_manifest_schema_v1.json",
        "benchmark_manifest_sprint80_encoder_autoencoder_v1.json",
        "benchmark_manifest_sprint80_transformer_hidden_state_v1.json",
        "diagnostic_report_schema_v1.json",
        "representation_problem_taxonomy_v1.json",
    }

    data = proof.build_data()  # type: ignore[attr-defined]
    train_idx = np.asarray(data["train_indices"])
    heldout_idx = np.asarray(data["heldout_indices"])
    assert (len(train_idx), len(heldout_idx)) == (1437, 360)
    assert np.intersect1d(train_idx, heldout_idx).size == 0
    digest = lambda values: hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()  # noqa: E731
    assert digest(train_idx) == proof.PINNED_TRAIN_INDEX_DIGEST  # type: ignore[attr-defined]
    assert digest(heldout_idx) == proof.PINNED_HELDOUT_INDEX_DIGEST  # type: ignore[attr-defined]

    case = proof.build_model_case()  # type: ignore[attr-defined]
    assert np.array_equal(case["heldout_latents"], case["replay_latents"])
    assert case["heldout_latents"].shape == (360, 4)
    assert proof.frozen_digests() == before  # type: ignore[attr-defined]


def test_real_capture_and_declared_control_cases_record_truthfully() -> None:
    concrete, provenance = proof._capture()
    assert concrete.capture_identity == proof._capture()[0].capture_identity
    assert len(concrete.capture_identity) == 64
    assert tuple(concrete.shape) == (360, 4)
    assert concrete.dtype == "float64"
    assert provenance["capture_id"] == proof.CAPTURE_ID  # type: ignore[attr-defined]
    assert provenance["representation_identity"] == proof.REPRESENTATION  # type: ignore[attr-defined]
    assert provenance["model_revision"] == proof.MODEL_REVISION  # type: ignore[attr-defined]
    assert provenance["axes"] == ["sample", "feature", "slice"]
    assert provenance["split_identity"] == "default_rng-42-permutation-train1437-heldout360"

    detect = proof.detect_case()  # type: ignore[attr-defined]
    family = detect["family"]
    payload = detect["payload"]
    # The declared positive case is recorded exactly as observed: both
    # frozen metrics pass on the pinned revision, so the known defect is
    # NOT observed. Thresholds are never adjusted.
    assert dict(family.threshold_pass) == {
        "bottleneck-effective-rank": True,
        "bottleneck-singular-spread": True,
    }
    assert family.outcome == "supported" and family.claim_allowed is True
    assert family.missing_evidence == ()
    assert dict(family.control_outcomes) == {
        "control-healthy-counterexample": "passed",
        "control-benign-low-variance": "passed",
        "control-null-shuffle": "passed",
    }
    measurements = payload["measurements"]
    assert measurements["effective_rank"] >= 3.0
    assert measurements["singular_spread"] >= 0.25
    control_metrics = payload["control_metrics"]
    healthy = control_metrics["control-healthy-counterexample"]
    benign = control_metrics["control-benign-low-variance"]
    assert healthy["bottleneck-effective-rank"] >= 3.0
    assert healthy["bottleneck-singular-spread"] >= 0.25
    assert benign["bottleneck-effective-rank"] >= 3.0
    assert benign["bottleneck-singular-spread"] >= 0.25
    assert benign["bottleneck-min-variance"] < 1e-6
    uncertainty = measurements["uncertainty"]["bottleneck-effective-rank"]
    assert uncertainty["repetitions"] == 200
    assert uncertainty["lower"] <= uncertainty["mean"] <= uncertainty["upper"]

    # Byte-deterministic re-evaluation at the declared seeds.
    config = detect["config"]
    _, rerun = proof.evaluate_detection(  # type: ignore[attr-defined]
        proof._value(np.asarray(proof.build_model_case()["heldout_latents"])),  # type: ignore[attr-defined]
        config,
        controls=proof._control_batches(),  # type: ignore[attr-defined]
    )
    assert proof._payload_sha(rerun) == proof._payload_sha(payload)  # type: ignore[attr-defined]


def test_truthful_localization_is_negative_and_binds_manifest_wiring() -> None:
    localize = proof.localize_case()  # type: ignore[attr-defined]
    result = localize["result"]
    payload = localize["payload"]
    assert result.verdict == "negative"
    assert result.earliest_layer is None
    assert result.affected_layers == ()
    assert result.affected_samples == ()
    assert result.affected_slices == ()
    assert payload["report_localization"] == []
    assert payload["metric_id"] == "bottleneck-effective-rank"
    assert payload["family_id"] == "collapse_rank_loss"
    assert payload["representation_identity"] == proof.REPRESENTATION  # type: ignore[attr-defined]
    config = payload["config"]
    assert config["comparator"] == ">="
    assert config["threshold_value"] == 3.0
    assert config["tolerance"] == 0.1
    assert config["direction"] == "higher_is_better"
    assert config["affected_when"] == "threshold_fail"
    assert set(config["control_outcomes"]) >= {
        "control-healthy-counterexample",
        "control-benign-low-variance",
    }
    assert len(payload["sample_cells_by_sample_id"]) == 360
    assert payload["declared_slice_ids"] == [proof.SLICE_ID]  # type: ignore[attr-defined]
    assert all(row["metric_value"] > 3.1 for row in payload["layer_cells_in_declared_order"])

    _result_again, rerun = proof.evaluate_localization(localize["source"])  # type: ignore[attr-defined]
    assert proof._payload_sha(rerun) == proof._payload_sha(payload)  # type: ignore[attr-defined]


def test_workflow_fails_closed_at_explain_and_persistence_refuses(tmp_path: Path) -> None:
    from latent_anything._diagnostic_workflow import WORKFLOW_STAGES

    manifest = proof.load_manifest()
    workflow, request = proof.build_workflow()  # type: ignore[attr-defined]
    running_result, checkpoint = workflow.run(request, manifest, stop_after="localize")
    assert running_result.status == "running"
    assert checkpoint.status == "running"
    assert tuple(checkpoint.completed_stages) == WORKFLOW_STAGES[:3]
    localize_output = next(item for item in checkpoint.outputs if item.stage == "localize")
    assert localize_output.payload["verdict"] == "negative"

    result, failed = workflow.resume(checkpoint, request, manifest)
    assert result.status == "failed"
    assert failed.status == "failed"
    assert failed.failure is not None
    assert failed.failure.stage == "explain"
    assert failed.failure.error_type == "StageContractError"
    assert "localization" in failed.failure.message
    # The chain never fabricates a location to get past the gate.
    assert localize_output.payload["report_localization"] == []

    from latent_anything._diagnostic_artifact import DiagnosticArtifactError, persist_diagnostic_artifact

    root = tmp_path / "root"
    with pytest.raises(DiagnosticArtifactError, match="must be completed"):
        persist_diagnostic_artifact(
            root,
            request=request,
            manifest=manifest,
            result=result,
            checkpoint=failed,
            report={},
        )
    assert not (root / "runs").exists()
    assert not (root / "artifacts").exists()


def test_tampered_manifest_capture_and_resume_identities_fail_closed() -> None:
    from latent_anything._benchmark_manifest import BenchmarkManifestValidationError, validate_manifest
    from latent_anything._capture_binding import CaptureBindingError, bind_selection
    from latent_anything._diagnostic_workflow import WorkflowError
    from latent_anything.diagnostics import CaptureSelection, OutputSelection

    manifest = proof.load_manifest()
    mutated = json.loads(
        (REPO / "artifacts" / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json").read_text(encoding="utf-8")
    )
    mutated["thresholds"][0]["value"] = 0.01
    with pytest.raises(BenchmarkManifestValidationError, match="manifest digest"):
        validate_manifest(mutated)

    with pytest.raises(CaptureBindingError, match="provenance mismatch"):
        bind_selection(
            CaptureSelection(
                capture_id=proof.CAPTURE_ID,  # type: ignore[attr-defined]
                representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=99",
                axes=("sample", "feature", "slice"),
            ),
            manifest=manifest,
            request_id=proof.REQUEST_ID,  # type: ignore[attr-defined]
        )

    workflow, request = proof.build_workflow()  # type: ignore[attr-defined]
    _, checkpoint = workflow.run(request, manifest, stop_after="localize")
    tampered = dataclasses.replace(
        request,
        output=OutputSelection(output_location="artifacts/diagnostics/tampered-request"),
    )
    with pytest.raises(WorkflowError, match="request_digest"):
        workflow.resume(checkpoint, tampered, manifest)


def test_failed_required_control_blocks_the_conclusion() -> None:
    manifest = proof.load_manifest()
    config = proof.detection_config_from_manifest(proof._detect_request(), manifest)  # type: ignore[attr-defined]
    detections, _payload = proof.evaluate_detection(  # type: ignore[attr-defined]
        proof._value(np.asarray(proof.build_model_case()["heldout_latents"])),  # type: ignore[attr-defined]
        config,
        controls={
            "control-healthy-counterexample": proof._value(proof._failing_counterexample_batch()),  # type: ignore[attr-defined]
            "control-benign-low-variance": proof._value(proof._negative_batch()),  # type: ignore[attr-defined]
        },
    )
    blocked = detections[0]
    assert blocked.outcome == "inconclusive"
    assert blocked.claim_allowed is False
    assert any(item.startswith("failed-control:") for item in blocked.missing_evidence)


def test_downstream_measurements_are_real_and_manifest_bound() -> None:
    manifest = proof.load_manifest()
    metric_ids = {str(metric["id"]) for metric in manifest["metrics"]}  # type: ignore[index]
    assert metric_ids == {"bottleneck-effective-rank", "bottleneck-singular-spread"}

    detect = proof.detect_case()  # type: ignore[attr-defined]
    case = proof.build_model_case()  # type: ignore[attr-defined]
    previews = proof.downstream_previews()  # type: ignore[attr-defined]

    # Cross-consistency: the direct baseline equals the detector's observation.
    assert previews["baseline_effective_rank"] == detect["payload"]["measurements"]["effective_rank"]
    per_direction = previews["per_direction"]
    assert len(per_direction) == 4
    assert {int(row["dim"]) for row in per_direction} == {0, 1, 2, 3}
    target_dim = int(case["argmin_dim"])
    assert previews["declared_trial_target"] == f"bottleneck-mu-dim{target_dim}"
    assert target_dim == int(np.argmin(np.var(np.asarray(case["heldout_latents"]), axis=0)))
    for row in per_direction:
        assert np.isfinite(row["effective_rank"])
        assert np.isfinite(row["heldout_reconstruction_mse"])
    # Ablating the declared target direction must change the frozen metric
    # beyond the manifest tolerance (the measurement itself is real).
    target_row = per_direction[target_dim]
    assert abs(target_row["effective_rank_delta"]) > 0.1
    # The manifest falsification rule2 arithmetic: the off-target direction
    # moves the metric at least as much as the diagnosed direction.
    assert previews["off_target_specificity_rule_violated_by_arithmetic"] is True

    points = previews["comparison_points"]
    assert points[proof.BASELINE_RUN] == points[proof.CANDIDATE_RUN]  # type: ignore[attr-defined]
    assert previews["comparison_absolute_delta"] == {
        "bottleneck-effective-rank": 0.0,
        "bottleneck-singular-spread": 0.0,
    }
