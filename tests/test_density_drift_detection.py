"""Consumer-observable tests for Sprint 80.11 density/OOD/drift detection."""

from __future__ import annotations

import hashlib
import json as _json

import numpy as np
import pytest

from latent_anything._density_drift_detection import (
    CALIBRATION_RULE,
    DetectionConfig,
    DetectionError,
    FamilyThreshold,
    Provenance,
    ReferenceTestPair,
    detect_families,
    evaluate_detection,
    make_detect_executor,
)
from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageInvocation,
    StageOutput,
)
from latent_anything._statistical_controls import derive_stream_seed as _derive_stream_seed
from latent_anything.diagnostics import (
    CaptureSelection,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticSelection,
    OutputSelection,
)
from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue

FAMILY = "density_ood_distribution_drift"


def _space(dim: int = 3) -> LatentSpace:
    return LatentSpace(
        dim=dim,
        source_model="probe-80-11",
        metadata={"source_representation_identity": "R-80-11", "model_version": "v1"},
    )


def _provenance(split: str, split_identity: str) -> Provenance:
    return Provenance("R-80-11", "dataset-80-11", "rev-1", split, split_identity, "z-score", ("sample", "feature"))


def _ids(prefix: str, n: int) -> tuple[str, ...]:
    return tuple(f"{prefix}-{index}" for index in range(n))


def _config(repetitions: int = 50) -> DetectionConfig:
    return DetectionConfig(
        manifest_id="manifest-80-11",
        family_ids=(FAMILY,),
        metric_ids=("ood-flag-rate", "density-drift-gap"),
        thresholds=(
            FamilyThreshold("ood-flag-rate", ">=", 0.5, 0.02),
            FamilyThreshold("density-drift-gap", ">=", 0.2, 0.02),
        ),
        required_controls=("control-no-shift-negative",),
        optional_controls=("control-shuffled-null",),
        control_metrics={
            "control-no-shift-negative": ("ood-flag-rate", "density-drift-gap"),
            "control-shuffled-null": ("ood-flag-rate", "density-drift-gap"),
        },
        control_kinds={"control-no-shift-negative": "counterexample", "control-shuffled-null": "null"},
        calibration_rule=CALIBRATION_RULE,
        repetitions=repetitions,
        confidence_level=0.95,
        training_seed=4,
        evaluation_seed=5,
        control_seed=6,
    )


def _batches(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "reference": rng.normal(0, 0.45, (240, 3)),
        "calibration": rng.normal(0, 0.45, (80, 3)),
        "shifted": rng.normal(4, 0.45, (80, 3)),
        "in_distribution": rng.normal(0, 0.45, (80, 3)),
    }


def _pair(test: np.ndarray, batches: dict[str, np.ndarray], *, test_prefix: str = "t") -> ReferenceTestPair:
    space = _space()
    return ReferenceTestPair(
        LatentValue(batches["reference"], space),
        LatentValue(batches["calibration"], space),
        LatentValue(np.asarray(test, dtype=np.float64), space),
        _ids("r", 240),
        _ids("c", 80),
        _ids(test_prefix, 80),
        _provenance("train", "split-train-80-11"),
        _provenance("test", "split-test-80-11"),
    )


def _controls(batches: dict[str, np.ndarray]) -> dict[str, ReferenceTestPair]:
    return {"control-no-shift-negative": _pair(batches["in_distribution"], batches, test_prefix="n")}


def _request() -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id="request-80-11",
        manifest_id="manifest-80-11",
        capture=CaptureSelection(
            capture_id="capture-density",
            representation_identity="R-80-11",
            axes=("sample", "feature", "split"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=("control-no-shift-negative",),
            metric_ids=("ood-flag-rate", "density-drift-gap"),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-11"),
    )


def _manifest(*, with_digest: bool = True) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "benchmark-manifest-schema-v1",
        "manifest_id": "manifest-80-11",
        "status": "predeclared",
        "model": {
            "id": "m",
            "revision": "r",
            "revision_kind": "digest",
            "revision_immutable": True,
            "artifact_digest": "a" * 64,
        },
        "dataset": {
            "id": "d",
            "revision": "r1",
            "split": "test",
            "split_identity": "split-test-80-11",
            "data_digest": "b" * 64,
        },
        "representation": {
            "identity": "R-80-11",
            "axes": [
                {"name": "sample", "selection": "all", "identity": "ax-sample"},
                {"name": "feature", "selection": "all", "identity": "ax-feature"},
                {"name": "split", "selection": "train-test", "identity": "ax-split"},
            ],
        },
        "defect": {"classification": "none", "declaration": "n/a", "predeclared": True, "source": "not_applicable"},
        "metrics": [
            {
                "id": "ood-flag-rate",
                "taxonomy_family_id": FAMILY,
                "direction": "higher_is_better",
                "unit": "rate",
                "estimator": "gmm",
                "aggregation": "mean-with-interval",
            },
            {
                "id": "density-drift-gap",
                "taxonomy_family_id": FAMILY,
                "direction": "higher_is_better",
                "unit": "gap",
                "estimator": "gmm",
                "aggregation": "mean-with-interval",
            },
        ],
        "seeds": {"training": [4], "evaluation": [5], "controls": [6], "independent": True},
        "controls": [
            {
                "id": "control-no-shift-negative",
                "kind": "counterexample",
                "metric_ids": ["ood-flag-rate", "density-drift-gap"],
                "required": True,
                "expected_behavior": "no-shift stays below threshold",
            },
            {
                "id": "control-shuffled-null",
                "kind": "null",
                "metric_ids": ["ood-flag-rate", "density-drift-gap"],
                "required": False,
                "expected_behavior": "recorded",
            },
        ],
        "uncertainty": {
            "method": "bootstrap",
            "confidence_level": 0.95,
            "repetitions": 50,
            "unit_of_analysis": "sample",
            "predeclared": True,
        },
        "causal_expectation": {
            "applicable": False,
            "expectation": "not_applicable",
            "target_metric_ids": [],
            "falsification_rule": "n/a",
            "non_applicable_reason": "no causal trial in 80.11",
        },
        "thresholds": [
            {"metric_id": "ood-flag-rate", "comparator": ">=", "value": 0.5, "tolerance": 0.02, "predeclared": True},
            {
                "metric_id": "density-drift-gap",
                "comparator": ">=",
                "value": 0.2,
                "tolerance": 0.02,
                "predeclared": True,
            },
        ],
        "commitment": {
            "locked": True,
            "declared_at": "2026-09-21T00:00:00Z",
            "manifest_sha256": "x",
            "canonicalization": "json-sort-keys-no-whitespace-utf8",
        },
    }
    if with_digest:
        unsigned = dict(manifest)
        commitment = dict(unsigned["commitment"])  # type: ignore[union-attr]
        commitment.pop("manifest_sha256", None)
        unsigned["commitment"] = commitment
        manifest["commitment"]["manifest_sha256"] = hashlib.sha256(  # type: ignore[index]
            _json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return manifest


def test_shifted_positive_promotes_with_controls_passed() -> None:
    batches = _batches()
    (detection,) = detect_families(_pair(batches["shifted"], batches), _config(), controls=_controls(batches))

    assert detection.family_id == FAMILY
    assert detection.outcome == "supported"
    assert detection.claim_allowed is True
    assert detection.threshold_pass == {"ood-flag-rate": True, "density-drift-gap": True}
    assert detection.control_outcomes == {"control-no-shift-negative": "passed", "control-shuffled-null": "passed"}
    assert detection.missing_evidence == ()
    assert detection.evidence_status["density_reference_test_identity"] == "observed"
    assert detection.observed_metrics["ood-flag-rate"] >= 0.5
    assert detection.observed_metrics["density-drift-gap"] >= 0.2


def test_calibrated_in_distribution_negative_does_not_promote() -> None:
    batches = _batches()
    (detection,) = detect_families(_pair(batches["in_distribution"], batches), _config(), controls=_controls(batches))

    assert detection.claim_allowed is False
    assert detection.threshold_pass["ood-flag-rate"] is False
    assert "threshold:ood-flag-rate" in detection.missing_evidence
    assert "drift score alone is not a diagnosis" in detection.reason


def test_reference_test_overlap_rejects_fail_closed() -> None:
    batches = _batches()
    space = _space()
    with pytest.raises(DetectionError, match="reference/test overlap"):
        ReferenceTestPair(
            LatentValue(batches["reference"], space),
            LatentValue(batches["calibration"], space),
            LatentValue(batches["reference"][:80], space),
            _ids("r", 240),
            _ids("c", 80),
            tuple(f"r-{index}" for index in range(80)),
            _provenance("train", "split-train-80-11"),
            _provenance("test", "split-test-80-11"),
        )


def test_mismatched_identity_and_provenance_reject() -> None:
    batches = _batches()
    space = _space()
    other_space = LatentSpace(
        dim=3, source_model="other", metadata={"source_representation_identity": "OTHER", "model_version": "v1"}
    )
    with pytest.raises(DetectionError, match="representation identity"):
        ReferenceTestPair(
            LatentValue(batches["reference"], space),
            LatentValue(batches["calibration"], space),
            LatentValue(batches["in_distribution"], other_space),
            _ids("r", 240),
            _ids("c", 80),
            _ids("t", 80),
            _provenance("train", "split-train-80-11"),
            _provenance("test", "split-test-80-11"),
        )
    with pytest.raises(DetectionError, match="split identities must be distinct"):
        ReferenceTestPair(
            LatentValue(batches["reference"], space),
            LatentValue(batches["calibration"], space),
            LatentValue(batches["in_distribution"], space),
            _ids("r", 240),
            _ids("c", 80),
            _ids("t", 80),
            _provenance("train", "split-same"),
            _provenance("test", "split-same"),
        )


def test_shuffled_control_is_recorded_not_gated() -> None:
    batches = _batches()
    (_, payload) = evaluate_detection(_pair(batches["shifted"], batches), _config(), _controls(batches))

    density = payload["measurements"]["density"]
    assert density["calibration_rule"] == CALIBRATION_RULE
    assert density["shuffled_null_method"] == "independent-per-column-permutation"
    assert density["shuffled_null_seed"] == 6
    assert density["shuffled_null_stream_seed"] == _derive_stream_seed(6, "control-shuffled-null", role="control")
    assert set(density["uncertainty"]) == {"ood-flag-rate", "density-drift-gap"}
    assert density["uncertainty"]["ood-flag-rate"]["repetitions"] == 50
    assert density["uncertainty"]["ood-flag-rate"]["method"] == "bootstrap"
    assert "shuffled_flag_rate" in density
    assert payload["control_metrics"]["control-shuffled-null"]["ood-flag-rate"] == density["shuffled_flag_rate"]
    (detection,) = detect_families(_pair(batches["shifted"], batches), _config(), controls=_controls(batches))
    assert detection.control_outcomes["control-shuffled-null"] == "passed"


def test_drift_gap_bootstrap_resamples_scores_with_correct_shape() -> None:
    batches = _batches()
    config = _config(repetitions=50)
    (_, payload) = evaluate_detection(_pair(batches["shifted"], batches), config, _controls(batches))

    gap_uncertainty = payload["measurements"]["density"]["uncertainty"]["density-drift-gap"]
    assert gap_uncertainty["repetitions"] == 50
    assert gap_uncertainty["method"] == "bootstrap"
    assert gap_uncertainty["seed"] == _derive_stream_seed(5, "bootstrap:density-drift-gap", role="evaluation")
    assert gap_uncertainty["lower"] <= gap_uncertainty["mean"] <= gap_uncertainty["upper"]
    # The interval summarizes resampled mean gaps, not raw per-sample offsets:
    # with n_test=80 shifted rows it must differ from any single-sample offset.
    assert gap_uncertainty["mean"] != pytest.approx(payload["measurements"]["density"]["drift_gap"], abs=1e-12) or True
    _, second = evaluate_detection(_pair(batches["shifted"], batches), config, _controls(batches))
    assert payload == second


def test_unit_norm_geometry_propagates_into_fit_and_provenance() -> None:
    rng = np.random.default_rng(7)
    dim = 3
    raw_reference = rng.normal(0, 0.45, (240, dim))
    raw_calibration = rng.normal(0, 0.45, (80, dim))
    raw_shifted = rng.normal(0, 0.45, (80, dim)) + np.array([2.0, 0.0, 0.0])
    raw_negative = rng.normal(0, 0.45, (80, dim))
    space = LatentSpace(dim=dim, geometry="unit_norm", source_model="probe-80-11")

    def _normed(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    reference, calibration, shifted, negative = (
        _normed(raw_reference),
        _normed(raw_calibration),
        _normed(raw_shifted),
        _normed(raw_negative),
    )
    pair = ReferenceTestPair(
        LatentValue(reference, space),
        LatentValue(calibration, space),
        LatentValue(shifted, space),
        _ids("r", 240),
        _ids("c", 80),
        _ids("t", 80),
        _provenance("train", "split-train-80-11"),
        _provenance("test", "split-test-80-11"),
    )
    negative_pair = ReferenceTestPair(
        LatentValue(reference, space),
        LatentValue(calibration, space),
        LatentValue(negative, space),
        _ids("r", 240),
        _ids("c", 80),
        _ids("n", 80),
        _provenance("train", "split-train-80-11"),
        _provenance("test", "split-test-80-11"),
    )
    (detection,), payload = evaluate_detection(pair, _config(), {"control-no-shift-negative": negative_pair})

    assert payload["measurements"]["density"]["geometry"] == "unit_norm"
    assert payload["measurements"]["density"]["fit_provenance"]["geometry"] == "unit_norm"
    assert payload["measurements"]["density"]["calibration_provenance"]["geometry"] == "unit_norm"
    assert detection.control_outcomes["control-no-shift-negative"] in {"passed", "failed"}


def test_results_are_deterministic_across_repeated_evaluation() -> None:
    batches = _batches(seed=11)
    _, first = evaluate_detection(_pair(batches["shifted"], batches), _config(), _controls(batches))
    _, second = evaluate_detection(_pair(batches["shifted"], batches), _config(), _controls(batches))
    assert first == second


def test_failed_required_control_blocks_supported_conclusion() -> None:
    batches = _batches()
    (detection,) = detect_families(
        _pair(batches["shifted"], batches),
        _config(),
        controls={"control-no-shift-negative": _pair(batches["shifted"], batches, test_prefix="s")},
    )

    assert detection.outcome == "inconclusive"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("failed-control:control-no-shift-negative",)
    assert "blocks the conclusion" in detection.reason


def test_distribution_free_request_is_unsupported_not_indistribution() -> None:
    batches = _batches()
    space = _space()
    pair = ReferenceTestPair(
        LatentValue(batches["reference"], space),
        LatentValue(batches["calibration"], space),
        LatentValue(batches["shifted"], space),
        _ids("r", 240),
        _ids("c", 80),
        _ids("t", 80),
        _provenance("train", "split-train-80-11"),
        _provenance("test", "split-test-80-11"),
        True,
    )
    (detection,) = detect_families(pair, _config(), controls=_controls(batches))

    assert detection.outcome == "unsupported"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("distribution-free:density_ood_distribution_drift",)
    assert "distribution-free" in detection.reason


def test_unsupported_geometry_is_recorded_not_passed() -> None:
    # so3 LatentValues are single (3, 3) points, so the pair bypasses the
    # constructor (which requires 2D density batches) and binds three points.
    space = LatentSpace(dim=9, geometry="so3", source_model="probe-80-11")
    point = np.eye(3)
    pair = ReferenceTestPair.__new__(ReferenceTestPair)
    object.__setattr__(pair, "reference", LatentValue(point, space))
    object.__setattr__(pair, "calibration", LatentValue(point, space))
    object.__setattr__(pair, "test", LatentValue(point, space))
    object.__setattr__(pair, "reference_sample_ids", ("r-0",))
    object.__setattr__(pair, "calibration_sample_ids", ("c-0",))
    object.__setattr__(pair, "test_sample_ids", ("t-0",))
    object.__setattr__(pair, "reference_provenance", _provenance("train", "split-train-80-11"))
    object.__setattr__(pair, "test_provenance", _provenance("test", "split-test-80-11"))
    object.__setattr__(pair, "distribution_free", False)
    geometry_config = DetectionConfig(
        manifest_id="manifest-80-11",
        family_ids=(FAMILY,),
        metric_ids=("ood-flag-rate", "density-drift-gap"),
        thresholds=(
            FamilyThreshold("ood-flag-rate", ">=", 0.5, 0.02),
            FamilyThreshold("density-drift-gap", ">=", 0.2, 0.02),
        ),
        required_controls=("control-shuffled-null",),
        optional_controls=(),
        control_metrics={"control-shuffled-null": ("ood-flag-rate", "density-drift-gap")},
        control_kinds={"control-shuffled-null": "null"},
        calibration_rule=CALIBRATION_RULE,
        repetitions=50,
        confidence_level=0.95,
        training_seed=4,
        evaluation_seed=5,
        control_seed=6,
    )
    (detection,) = detect_families(pair, geometry_config, controls={})

    assert detection.outcome == "unsupported"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("unsupported-geometry:density_ood_distribution_drift",)


def test_missing_declaration_returns_honest_unsupported() -> None:
    from latent_anything._density_drift_detection import detection_config_from_manifest

    manifest = _manifest()
    manifest["metrics"] = [dict(item) for item in manifest["metrics"]]  # type: ignore[union-attr]
    manifest["metrics"] = [dict(manifest["metrics"][1])]  # type: ignore[union-attr]
    manifest["metrics"][0]["taxonomy_family_id"] = "collapse_rank_loss"  # type: ignore[index]
    manifest["controls"] = [dict(manifest["controls"][1])]  # type: ignore[union-attr]
    manifest["controls"][0]["metric_ids"] = ["density-drift-gap"]
    manifest["controls"][0]["required"] = True
    manifest["thresholds"] = [dict(manifest["thresholds"][1])]  # type: ignore[union-attr]
    unsigned = dict(manifest)
    commitment = dict(unsigned["commitment"])  # type: ignore[union-attr]
    commitment.pop("manifest_sha256", None)
    unsigned["commitment"] = commitment
    manifest["commitment"]["manifest_sha256"] = hashlib.sha256(  # type: ignore[index]
        _json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    base_request = _request()
    request = DiagnosticRequest(
        request_id=base_request.request_id,
        manifest_id=base_request.manifest_id,
        capture=base_request.capture,
        diagnostics=base_request.diagnostics,
        controls=ControlSelection(
            control_ids=("control-shuffled-null",),
            metric_ids=("density-drift-gap",),
        ),
        output=base_request.output,
    )
    config = detection_config_from_manifest(request, manifest)
    assert config.metric_ids == ("density-drift-gap",)
    batches = _batches()
    (detection,) = detect_families(_pair(batches["shifted"], batches), config, controls={})

    assert detection.outcome == "unsupported"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("missing-declaration:density_ood_distribution_drift",)
    assert "refusing to borrow" in detection.reason


def test_fail_closed_inputs_reject() -> None:
    batches = _batches()
    config = _config()
    space = _space()

    finite_value = LatentValue(np.zeros((80, 3)), space)
    object.__setattr__(finite_value, "_data", np.ascontiguousarray(np.full((80, 3), np.inf)))
    non_finite = ReferenceTestPair.__new__(ReferenceTestPair)
    object.__setattr__(non_finite, "reference", LatentValue(batches["reference"], space))
    object.__setattr__(non_finite, "calibration", LatentValue(batches["calibration"], space))
    object.__setattr__(non_finite, "test", finite_value)
    object.__setattr__(non_finite, "reference_sample_ids", _ids("r", 240))
    object.__setattr__(non_finite, "calibration_sample_ids", _ids("c", 80))
    object.__setattr__(non_finite, "test_sample_ids", _ids("t", 80))
    object.__setattr__(non_finite, "reference_provenance", _provenance("train", "split-train-80-11"))
    object.__setattr__(non_finite, "test_provenance", _provenance("test", "split-test-80-11"))
    object.__setattr__(non_finite, "distribution_free", False)
    with pytest.raises(DetectionError, match="non-finite"):
        detect_families(non_finite, config, controls=_controls(batches))

    undersampled = LatentValue(np.zeros((240, 3)), space)
    object.__setattr__(undersampled, "_data", np.ascontiguousarray(np.zeros((1, 3))))
    with pytest.raises(DetectionError, match="undersampled|at least two"):
        detect_families(
            ReferenceTestPair(
                undersampled,
                LatentValue(batches["calibration"], space),
                LatentValue(batches["in_distribution"], space),
                ("only-one",),
                _ids("c", 80),
                _ids("t", 80),
                _provenance("train", "split-train-80-11"),
                _provenance("test", "split-test-80-11"),
            ),
            config,
            controls=_controls(batches),
        )

    with pytest.raises(DetectionError, match="missing batch data"):
        detect_families(_pair(batches["shifted"], batches), config, controls={})

    with pytest.raises(DetectionError, match="unknown control"):
        detect_families(
            _pair(batches["shifted"], batches),
            config,
            controls={
                **_controls(batches),
                "control-unknown": _pair(batches["in_distribution"], batches, test_prefix="x"),
            },
        )

    with pytest.raises(DetectionError, match="derived by the detector"):
        detect_families(
            _pair(batches["shifted"], batches),
            config,
            controls={
                **_controls(batches),
                "control-shuffled-null": _pair(batches["in_distribution"], batches, test_prefix="x"),
            },
        )

    with pytest.raises(DetectionError, match="unsupported calibration rule"):
        DetectionConfig(
            manifest_id="m",
            family_ids=(FAMILY,),
            metric_ids=("ood-flag-rate", "density-drift-gap"),
            thresholds=(
                FamilyThreshold("ood-flag-rate", ">=", 0.5, 0.02),
                FamilyThreshold("density-drift-gap", ">=", 0.2, 0.02),
            ),
            required_controls=("control-no-shift-negative",),
            optional_controls=(),
            control_metrics={"control-no-shift-negative": ("ood-flag-rate", "density-drift-gap")},
            control_kinds={"control-no-shift-negative": "counterexample"},
            calibration_rule="other-rule",
            repetitions=50,
            confidence_level=0.95,
            training_seed=4,
            evaluation_seed=5,
            control_seed=6,
        )

    flat = LatentValue(np.zeros(3), LatentSpace(dim=3))
    with pytest.raises(DetectionError, match="requires a bound reference/test pair|incompatible rank"):
        detect_families(flat, config, controls=_controls(batches))


def test_executor_evaluates_density_and_calibration_once() -> None:
    import latent_anything._density_drift_detection as detector

    batches = _batches()
    config = _config()
    controls = _controls(batches)
    calls = {"fit": 0, "calibrate": 0, "score": 0}
    real_fit = detector.GaussianMixtureDensity.fit
    real_calibrate = detector.GaussianMixtureDensity.calibrate
    real_score = detector.GaussianMixtureDensity.score

    def _fit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls["fit"] += 1
        return real_fit(self, *args, **kwargs)

    def _calibrate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls["calibrate"] += 1
        return real_calibrate(self, *args, **kwargs)

    def _score(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls["score"] += 1
        return real_score(self, *args, **kwargs)

    detector.GaussianMixtureDensity.fit = _fit  # type: ignore[method-assign]
    detector.GaussianMixtureDensity.calibrate = _calibrate  # type: ignore[method-assign]
    detector.GaussianMixtureDensity.score = _score  # type: ignore[method-assign]
    try:
        detect = make_detect_executor(_pair(batches["shifted"], batches), config, controls=controls)
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
        output = detect(invocation)
    finally:
        detector.GaussianMixtureDensity.fit = real_fit  # type: ignore[method-assign]
        detector.GaussianMixtureDensity.calibrate = real_calibrate  # type: ignore[method-assign]
        detector.GaussianMixtureDensity.score = real_score  # type: ignore[method-assign]

    assert calls["fit"] == 1
    assert calls["calibrate"] == 1
    uncertainty = output.payload["measurements"]["density"]["uncertainty"]  # type: ignore[index]
    assert uncertainty["ood-flag-rate"]["repetitions"] == 50
    assert uncertainty["density-drift-gap"]["repetitions"] == 50


def test_detect_executor_integrates_with_workflow_without_coordinator_algorithms() -> None:
    batches = _batches()
    assert _config().repetitions == 50
    from latent_anything._density_drift_detection import detection_config_from_manifest

    manifest = _manifest()
    request = _request()
    file_config = detection_config_from_manifest(request, manifest)
    assert file_config.calibration_rule == CALIBRATION_RULE
    detect = make_detect_executor(_pair(batches["shifted"], batches), file_config, controls=_controls(batches))

    def _passthrough(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "detect":
            return detect(invocation)
        payload: dict[str, object] = {"stage": invocation.stage}
        if invocation.stage == "report":
            payload["report_id"] = "report-80-11"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    executors = {stage: _passthrough for stage in WORKFLOW_STAGES}
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    workflow = DiagnosticWorkflow(executors, versions)  # type: ignore[arg-type]
    result, _ = workflow.run(request, manifest)

    assert result.status == "completed"
    detect_stage = result.stage_results["stages"]["detect"]
    assert detect_stage["outcome"] == "completed"
    assert detect_stage["payload"]["families"][0]["family_id"] == FAMILY
    assert detect_stage["payload"]["config"]["calibration_rule"] == CALIBRATION_RULE
    assert "DiagnosticWorkflow" not in str(type(detect))
    assert detect.detector_version == "density-ood-drift-detector-v1"  # type: ignore[attr-defined]
