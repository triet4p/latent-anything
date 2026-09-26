"""Consumer-observable tests for Sprint 80.9 collapse/anisotropy detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latent_anything._benchmark_manifest import manifest_digest
from latent_anything._collapse_detection import (
    DetectionError,
    FamilyThreshold,
    detect_families,
    detection_config_from_manifest,
    evaluate_detection,
    make_detect_executor,
)
from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageInvocation,
    StageOutput,
)
from latent_anything.diagnostics import (
    CaptureSelection,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticSelection,
    OutputSelection,
)
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _manifest() -> dict[str, object]:
    path = ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _manifest_v2() -> dict[str, object]:
    path = ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v2_injected.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _request(families: tuple[str, ...] = ("collapse_rank_loss",)) -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id="request-80-9",
        manifest_id="sprint80-core-encoder-autoencoder-collapse-v1",
        capture=CaptureSelection(
            capture_id="capture-encoder",
            representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
            axes=("sample", "feature"),
        ),
        diagnostics=DiagnosticSelection(family_ids=families),
        controls=ControlSelection(
            control_ids=("control-healthy-counterexample", "control-benign-low-variance"),
            metric_ids=("bottleneck-effective-rank", "bottleneck-singular-spread"),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-9"),
    )


def _request_v2() -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id="request-80-23-v2",
        manifest_id="sprint80-core-encoder-autoencoder-collapse-injected-v2",
        capture=CaptureSelection(
            capture_id="capture-encoder-v2",
            representation_identity="conv_vae_8x8:bottleneck-mu:latent_dim=4",
            axes=("sample", "feature", "slice"),
        ),
        diagnostics=DiagnosticSelection(family_ids=("collapse_rank_loss",)),
        controls=ControlSelection(
            control_ids=("control-healthy-counterexample", "control-benign-low-variance"),
            metric_ids=(
                "bottleneck-effective-rank",
                "bottleneck-singular-spread",
                "bottleneck-feature-variance-ratio",
            ),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-23-v2"),
    )


def _space() -> LatentSpace:
    return LatentSpace(
        dim=4,
        source_model="probe-80-9",
        metadata={
            "source_representation_identity": "conv_vae_8x8:bottleneck-mu:latent_dim=4",
            "model_version": "conv-vae-8x8-latent4-seed0-epochs5",
        },
    )


def _value(data: np.ndarray) -> LatentValue:
    return LatentValue(np.asarray(data, dtype=np.float64), _space())


def _batches(seed: int = 0) -> dict[str, LatentValue]:
    rng = np.random.default_rng(seed)
    healthy = rng.normal(size=(360, 4))
    collapsed = np.column_stack(
        [
            rng.normal(size=360),
            np.full(360, 2.0),
            rng.normal(size=360) * 1e-9,
            rng.normal(size=360),
        ]
    )
    anisotropic = np.column_stack(
        [
            rng.normal(size=360) * 3.0,
            rng.normal(size=360),
            rng.normal(size=360) * 0.3,
            rng.normal(size=360) * 0.05,
        ]
    )
    scaled = rng.normal(size=(360, 4)) * 0.05
    return {
        "healthy": _value(healthy),
        "collapsed": _value(collapsed),
        "anisotropic": _value(anisotropic),
        "scaled": _value(scaled),
    }


def _controls(batches: dict[str, LatentValue]) -> dict[str, LatentValue]:
    return {
        "control-healthy-counterexample": batches["healthy"],
        "control-benign-low-variance": batches["scaled"],
    }


def test_rank_collapse_positive_flags_defect_with_controls_passed() -> None:
    batches = _batches()
    config = detection_config_from_manifest(_request(), _manifest())
    (detection,) = detect_families(batches["collapsed"], config, controls=_controls(batches))

    assert detection.family_id == "collapse_rank_loss"
    assert detection.outcome == "supported"
    assert detection.claim_allowed is True
    assert detection.threshold_pass == {"bottleneck-effective-rank": False, "bottleneck-singular-spread": False}
    assert detection.control_outcomes == {
        "control-healthy-counterexample": "passed",
        "control-benign-low-variance": "passed",
        "control-null-shuffle": "passed",
    }
    assert detection.missing_evidence == ()
    assert detection.evidence_status["collapse_rank_profile"] == "observed"
    assert detection.observed_metrics["bottleneck-effective-rank"] < 3.0


def test_anisotropy_positive_flags_defect_through_directional_lens() -> None:
    batches = _batches()
    config = detection_config_from_manifest(
        _request(("collapse_rank_loss", "anisotropy_inactive_dimensions")), _manifest()
    )
    detections = detect_families(batches["anisotropic"], config, controls=_controls(batches))
    by_family = {detection.family_id: detection for detection in detections}

    assert set(by_family) == {"collapse_rank_loss", "anisotropy_inactive_dimensions"}
    anisotropy = by_family["anisotropy_inactive_dimensions"]
    assert anisotropy.outcome == "supported"
    assert anisotropy.claim_allowed is True
    assert anisotropy.evidence_status["anisotropy_activity_spectrum"] == "observed"
    assert anisotropy.evidence_status["anisotropy_predeclared_threshold"] == "observed"
    assert anisotropy.threshold_pass["bottleneck-singular-spread"] is False
    assert anisotropy.observed_metrics["bottleneck-singular-spread"] < 0.25


def test_benign_scaled_low_variance_is_negative_not_collapse() -> None:
    batches = _batches()
    config = detection_config_from_manifest(_request(), _manifest())
    (detection,) = detect_families(batches["scaled"], config, controls=_controls(batches))

    assert detection.outcome == "supported"
    assert detection.claim_allowed is True
    # Scale-invariant rank/spread stay healthy: no defect is promoted.
    assert detection.threshold_pass == {"bottleneck-effective-rank": True, "bottleneck-singular-spread": True}
    assert detection.control_outcomes["control-benign-low-variance"] == "passed"


def test_healthy_counterexample_and_null_control_pass() -> None:
    batches = _batches()
    config = detection_config_from_manifest(_request(), _manifest())
    (detection,), payload = evaluate_detection(batches["healthy"], config, _controls(batches))
    assert detection.control_outcomes["control-healthy-counterexample"] == "passed"
    null_metrics = payload["measurements"]["null_shuffled"]
    assert isinstance(null_metrics, dict)
    assert set(null_metrics) >= {"bottleneck-effective-rank", "bottleneck-singular-spread"}
    uncertainty = payload["measurements"]["uncertainty"]["bottleneck-effective-rank"]
    assert uncertainty["repetitions"] == 200
    assert uncertainty["method"] == "bootstrap"
    assert uncertainty["lower"] <= uncertainty["mean"] <= uncertainty["upper"]
    assert payload["family_evidence"]["collapse_rank_loss"]["collapse_rank_profile"] == "observed"


def test_predeclared_feature_variance_ratio_detects_injected_axis_collapse() -> None:
    rng = np.random.default_rng(23)
    healthy = rng.normal(size=(360, 4))
    injected = healthy.copy()
    injected[:, 0] = float(np.mean(healthy[:, 0]))
    scaled = healthy * 1e-3
    controls = {
        "control-healthy-counterexample": _value(healthy),
        "control-benign-low-variance": _value(scaled),
    }
    config = detection_config_from_manifest(_request_v2(), _manifest_v2())
    (detection,), payload = evaluate_detection(_value(injected), config, controls)

    assert detection.outcome == "supported"
    assert detection.claim_allowed is True
    assert detection.threshold_pass["bottleneck-feature-variance-ratio"] is False
    assert detection.observed_metrics["bottleneck-feature-variance-ratio"] < 1e-20
    assert detection.control_outcomes["control-healthy-counterexample"] == "passed"
    assert detection.control_outcomes["control-benign-low-variance"] == "passed"
    assert payload["control_metrics"]["control-healthy-counterexample"]["bottleneck-feature-variance-ratio"] >= 0.1
    assert payload["control_metrics"]["control-benign-low-variance"]["bottleneck-feature-variance-ratio"] >= 0.1
    feature_ratios = payload["measurements"]["feature_variance_ratios"]
    assert isinstance(feature_ratios, list)
    assert feature_ratios[0] < 1e-20
    uncertainty = payload["measurements"]["uncertainty"]["bottleneck-feature-variance-ratio"]
    assert uncertainty["repetitions"] == 200
    assert uncertainty["lower"] <= uncertainty["mean"] <= uncertainty["upper"] < 0.1


def test_unselected_task_metric_threshold_is_excluded_from_detector_config() -> None:
    manifest = _manifest_v2()
    task_metric_id = "heldout-brightness-bin-accuracy"
    metrics = manifest["metrics"]
    thresholds = manifest["thresholds"]
    assert isinstance(metrics, list)
    assert isinstance(thresholds, list)
    metrics.append(
        {
            "aggregation": "mean-with-interval",
            "direction": "higher_is_better",
            "estimator": "fixed-training-median-global-brightness-bin-accuracy",
            "id": task_metric_id,
            "taxonomy_family_id": "collapse_rank_loss",
            "unit": "accuracy",
        }
    )
    thresholds.append(
        {
            "comparator": ">=",
            "metric_id": task_metric_id,
            "predeclared": True,
            "tolerance": 0.02,
            "value": 0.9,
        }
    )
    commitment = manifest["commitment"]
    assert isinstance(commitment, dict)
    commitment["manifest_sha256"] = manifest_digest(manifest)

    request = _request_v2()
    config = detection_config_from_manifest(request, manifest)

    assert config.metric_ids == request.controls.metric_ids
    assert {item.metric_id for item in config.thresholds} == set(request.controls.metric_ids)
    assert task_metric_id not in config.metric_ids


def test_failed_required_control_blocks_supported_conclusion() -> None:
    batches = _batches()
    config = detection_config_from_manifest(_request(), _manifest())
    (detection,) = detect_families(
        batches["healthy"],
        config,
        controls={
            "control-healthy-counterexample": batches["collapsed"],
            "control-benign-low-variance": batches["scaled"],
        },
    )

    assert detection.outcome == "inconclusive"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("failed-control:control-healthy-counterexample",)
    assert "blocks the conclusion" in detection.reason


def test_results_are_deterministic_across_repeated_evaluation() -> None:
    batches = _batches(seed=7)
    config = detection_config_from_manifest(_request(), _manifest())
    _, first = evaluate_detection(batches["healthy"], config, _controls(batches))
    _, second = evaluate_detection(batches["healthy"], config, _controls(batches))
    assert first == second


def test_executor_evaluates_each_batch_bootstrap_and_control_exactly_once() -> None:
    """Instrumented proof: exactly ``repetitions`` resamples per metric, one control pass."""
    import latent_anything._collapse_detection as detector

    batches = _batches()
    config = detection_config_from_manifest(_request(), _manifest())
    controls = _controls(batches)
    calls = {"health": 0, "svd": 0}
    real_health = detector.compute_latent_health
    real_svd = detector.np.linalg.svd

    def _counting_health(block: np.ndarray) -> object:
        calls["health"] += 1
        return real_health(block)

    def _counting_svd(matrix: np.ndarray, **kwargs: object) -> object:
        calls["svd"] += 1
        return real_svd(matrix, **kwargs)

    detector.compute_latent_health = _counting_health  # type: ignore[method-assign]
    detector.np.linalg.svd = _counting_svd  # type: ignore[method-assign]
    try:
        detect = make_detect_executor(batches["healthy"], config, controls=controls)

        def _only_detect(invocation: StageInvocation) -> StageOutput:
            return detect(invocation)

        invocation = StageInvocation(
            stage="detect",
            request=_request(),
            manifest=_manifest(),
            prior=tuple(
                StageOutput(stage=stage, outcome="completed", payload={}, artifact_refs=()) for stage in ("capture",)
            ),
            workflow_identity="0" * 64,
            request_digest="1" * 64,
            manifest_digest="2" * 64,
            config_digest="3" * 64,
        )
        output = _only_detect(invocation)
    finally:
        detector.compute_latent_health = real_health  # type: ignore[method-assign]
        detector.np.linalg.svd = real_svd  # type: ignore[method-assign]

    # One health+SVD per evaluated batch: target + 2 controls + 1 central
    # null = 4 health evaluations; one SVD per _evaluate_batch plus one per
    # resample draw (repetitions), since each draw records both statistics.
    # Central scheduling shares one draw schedule across both metric streams.
    assert calls["health"] == 4 + config.repetitions
    assert calls["svd"] == 4 + config.repetitions
    uncertainty = output.payload["measurements"]["uncertainty"]  # type: ignore[index]
    assert uncertainty["bottleneck-effective-rank"]["repetitions"] == 200
    assert uncertainty["bottleneck-singular-spread"]["repetitions"] == 200
    # Naive composition (detect_families + detection_payload + control table)
    # would cost 2x bootstrap loops plus duplicated control/health passes; the
    # executor must stay at exactly one pass per stream, observable here as
    # 404, still far below a doubled 808+.
    assert calls["health"] < 2 * (4 + 2 * config.repetitions)


def test_detect_executor_integrates_with_workflow_without_coordinator_algorithms() -> None:
    batches = _batches()
    config = detection_config_from_manifest(_request(), _manifest())
    detect = make_detect_executor(batches["collapsed"], config, controls=_controls(batches))

    def _passthrough(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "detect":
            return detect(invocation)
        payload: dict[str, object] = {"stage": invocation.stage}
        if invocation.stage == "report":
            payload["report_id"] = "report-80-9"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    executors = {stage: _passthrough for stage in WORKFLOW_STAGES}
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    workflow = DiagnosticWorkflow(executors, versions)  # type: ignore[arg-type]
    result, _ = workflow.run(_request(), _manifest())

    assert result.status == "completed"
    detect_stage = result.stage_results["stages"]["detect"]
    assert detect_stage["outcome"] == "completed"
    assert detect_stage["payload"]["families"][0]["family_id"] == "collapse_rank_loss"
    assert detect_stage["payload"]["config"]["manifest_id"] == "sprint80-core-encoder-autoencoder-collapse-v1"
    assert "DiagnosticWorkflow" not in str(type(detect))
    assert detect.detector_version == "collapse-anisotropy-detector-v1"  # type: ignore[attr-defined]


def test_fail_closed_inputs_reject() -> None:
    batches = _batches()
    config = detection_config_from_manifest(_request(), _manifest())

    non_finite = _value(np.full((360, 4), np.inf))
    with pytest.raises(DetectionError, match="non-finite"):
        detect_families(non_finite, config, controls=_controls(batches))

    undersampled = _value(np.zeros((4, 4)))
    with pytest.raises(DetectionError, match="undersampled"):
        detect_families(undersampled, config, controls=_controls(batches))

    flat = LatentValue(np.zeros(4), LatentSpace(dim=4))
    with pytest.raises(DetectionError, match="incompatible rank"):
        detect_families(flat, config, controls=_controls(batches))

    with pytest.raises(DetectionError, match="missing batch data"):
        detect_families(batches["healthy"], config, controls={})

    with pytest.raises(DetectionError, match="unknown control"):
        detect_families(
            batches["healthy"], config, controls={**_controls(batches), "control-unknown": batches["healthy"]}
        )

    mutated = dict(_manifest())
    thresholds = [dict(item) for item in mutated["thresholds"]]  # type: ignore[union-attr]
    mutated["thresholds"] = [item for item in thresholds if item["metric_id"] != "bottleneck-effective-rank"]
    with pytest.raises(DetectionError, match="digest does not match|threshold"):
        detection_config_from_manifest(_request(), mutated)

    request = _request()
    other_controls = ControlSelection(control_ids=("control-undeclared",), metric_ids=request.controls.metric_ids)
    tampered = DiagnosticRequest(
        request_id=request.request_id,
        manifest_id=request.manifest_id,
        capture=request.capture,
        diagnostics=request.diagnostics,
        controls=other_controls,
        output=request.output,
    )
    with pytest.raises(DetectionError, match="not declared by the manifest"):
        detection_config_from_manifest(tampered, _manifest())

    with pytest.raises(DetectionError, match="not supported by this detector"):
        detection_config_from_manifest(_request(("separability_probe_leakage",)), _manifest())

    with pytest.raises(DetectionError, match="unsupported threshold comparator"):
        FamilyThreshold(metric_id="m", comparator="!=", value=1.0, tolerance=0.1)
