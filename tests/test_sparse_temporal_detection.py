"""Consumer-observable tests for Sprint 80.12 sparse/temporal detection."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageInvocation,
    StageOutput,
)
from latent_anything._sparse_temporal_detection import (
    DetectionConfig,
    DetectionError,
    FamilyThreshold,
    SparseInput,
    TemporalInput,
    detect_families,
    detection_config_from_manifest,
    evaluate_detection,
    make_detect_executor,
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
from latent_anything.trajectory import Trajectory

SPARSE = "sparse_feature_instability"
TEMPORAL = "sequence_trajectory_drift"


def _space(dim: int = 4) -> LatentSpace:
    return LatentSpace(
        dim=dim,
        source_model="probe-80-12",
        metadata={"source_representation_identity": "R-80-12", "model_version": "v1"},
    )


def _sparse_config() -> DetectionConfig:
    return DetectionConfig(
        manifest_id="manifest-80-12",
        family_ids=(SPARSE,),
        metric_ids=("sparse-cross-seed-stability", "sparse-reconstruction-quality"),
        thresholds=(
            FamilyThreshold("sparse-cross-seed-stability", ">=", 0.5, 0.02),
            FamilyThreshold("sparse-reconstruction-quality", ">=", 0.5, 0.02),
        ),
        required_controls=("control-stable-negative",),
        optional_controls=("control-seed-noise",),
        control_metrics={
            "control-stable-negative": ("sparse-cross-seed-stability",),
            "control-seed-noise": ("sparse-cross-seed-stability",),
        },
        control_kinds={"control-stable-negative": "counterexample", "control-seed-noise": "seed"},
        seed_axis=(11, 12),
        repetitions=20,
        confidence_level=0.95,
        training_seed=11,
        evaluation_seed=5,
        control_seed=99,
    )


def _temporal_config() -> DetectionConfig:
    return DetectionConfig(
        manifest_id="manifest-80-12",
        family_ids=(TEMPORAL,),
        metric_ids=("trajectory-stepwise-drift", "trajectory-repeatability"),
        thresholds=(
            FamilyThreshold("trajectory-stepwise-drift", ">=", 0.2, 0.02),
            FamilyThreshold("trajectory-repeatability", ">=", 0.1, 0.02),
        ),
        required_controls=("control-no-drift-negative",),
        optional_controls=("control-shuffled-sequence",),
        control_metrics={
            "control-no-drift-negative": ("trajectory-stepwise-drift",),
            "control-shuffled-sequence": ("trajectory-stepwise-drift",),
        },
        control_kinds={"control-no-drift-negative": "counterexample", "control-shuffled-sequence": "shuffled"},
        seed_axis=(11, 12),
        repetitions=20,
        confidence_level=0.95,
        training_seed=11,
        evaluation_seed=5,
        control_seed=6,
    )


def _sparse_batches(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dictionary = rng.normal(size=(4, 8))
    dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True)
    codes = np.zeros((200, 8))
    for i in range(200):
        chosen = rng.choice(8, 2, replace=False)
        codes[i, chosen] = rng.normal(size=2) + np.array([2.0, -2.0])
    sparse = codes @ dictionary.T + rng.normal(scale=0.05, size=(200, 4))
    noise = rng.normal(size=(200, 4))
    return sparse, noise


def _sparse_input(matrix: np.ndarray, negative: np.ndarray, *, permuted: bool = False) -> SparseInput:
    # Permuted-but-stable: the same samples in a different row order. The
    # detector must treat row order as a sample artifact, not a feature
    # difference, so stability is permutation-invariant by construction.
    rng = np.random.default_rng(0)
    space = _space()
    ids = tuple(f"s-{i}" for i in range(matrix.shape[0]))
    if permuted:
        perm = rng.permutation(matrix.shape[0])
        second = matrix[perm]
        second_ids = tuple(ids[i] for i in perm)
    else:
        second, second_ids = matrix, ids
    return SparseInput(
        LatentValue(matrix, space),
        ids,
        (11, 12),
        {"11": LatentValue(matrix, space), "12": LatentValue(np.asarray(second, dtype=np.float64), space)},
        {"11": ids, "12": second_ids},
        {"neg": LatentValue(np.asarray(negative, dtype=np.float64), space)},
        "R-80-12",
    )


def _trajectories() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(6)
    steps = 20
    reference = np.column_stack([np.linspace(0, 1, steps), np.zeros(steps), np.zeros(steps), np.zeros(steps)])
    drift = reference + np.column_stack([np.zeros(steps), np.linspace(0, 1, steps), np.zeros(steps), np.zeros(steps)])
    negative = reference + rng.normal(scale=0.02, size=(steps, 4))
    return reference, drift, negative


def _temporal_input(drift: np.ndarray, reference: np.ndarray, negative: np.ndarray) -> TemporalInput:
    space = LatentSpace(dim=4, source_model="probe-80-12")
    return TemporalInput(
        Trajectory(reference),
        Trajectory(np.asarray(drift, dtype=np.float64)),
        Trajectory(np.asarray(negative, dtype=np.float64)),
        tuple(f"step-{i}" for i in range(drift.shape[0])),
        space,
        "R-80-12",
    )


def test_permuted_but_stable_sparse_positive_passes() -> None:
    sparse, noise = _sparse_batches()
    (detection,) = detect_families(_sparse_input(sparse, noise, permuted=True), _sparse_config(), controls={})

    assert detection.family_id == SPARSE
    assert detection.outcome == "supported"
    assert detection.claim_allowed is True
    assert detection.threshold_pass == {"sparse-cross-seed-stability": True, "sparse-reconstruction-quality": True}
    assert detection.control_outcomes == {"control-stable-negative": "passed", "control-seed-noise": "passed"}
    assert detection.missing_evidence == ()
    assert detection.evidence_status["sparse_cross_seed_stability"] == "observed"
    # Permutation-invariant matching: row order must not matter.
    assert detection.observed_metrics["sparse-cross-seed-stability"] >= 0.5


def test_unstable_sparse_input_cannot_promote_without_separation() -> None:
    # A noise batch is cross-seed stable under permutation-invariant matching,
    # so the stability threshold alone cannot separate it: the negative gate
    # (same noise family) fails and blocks promotion. Only the sparse positive
    # with a noise negative promotes.
    sparse, noise = _sparse_batches()
    # Noise-on-noise aligns to nothing at cosine >= 0.85: the detector fails
    # closed with no manufactured stability rather than promoting or passing
    # a vacuous control.
    with pytest.raises(DetectionError, match="no matched features"):
        detect_families(_sparse_input(noise, noise, permuted=True), _sparse_config(), controls={})


def test_stable_negative_control_blocks_sparse_promotion() -> None:
    sparse, _ = _sparse_batches()
    (detection,) = detect_families(_sparse_input(sparse, sparse, permuted=True), _sparse_config(), controls={})

    assert detection.outcome == "inconclusive"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("failed-control:control-stable-negative",)
    assert "blocks the conclusion" in detection.reason


def test_ordered_temporal_drift_promotes_with_controls() -> None:
    reference, drift, negative = _trajectories()
    (detection,) = detect_families(_temporal_input(drift, reference, negative), _temporal_config(), controls={})

    assert detection.family_id == TEMPORAL
    assert detection.outcome == "supported"
    assert detection.claim_allowed is True
    assert detection.threshold_pass == {"trajectory-stepwise-drift": True, "trajectory-repeatability": True}
    assert detection.control_outcomes == {"control-no-drift-negative": "passed", "control-shuffled-sequence": "passed"}
    assert detection.observed_metrics["trajectory-stepwise-drift"] >= 0.2


def test_no_drift_temporal_negative_cannot_promote() -> None:
    reference, _, negative = _trajectories()
    (detection,) = detect_families(_temporal_input(negative, reference, negative), _temporal_config(), controls={})

    assert detection.outcome == "inconclusive"
    assert detection.claim_allowed is False
    assert "threshold:trajectory-stepwise-drift" in detection.missing_evidence


def test_failed_temporal_control_blocks_promotion() -> None:
    # A negative that itself drifts (copy of the drift trajectory) must fail
    # its control and block promotion even though headline thresholds pass.
    reference, drift, _ = _trajectories()
    (detection,) = detect_families(_temporal_input(drift, reference, drift), _temporal_config(), controls={})

    assert detection.outcome == "inconclusive"
    assert detection.claim_allowed is False
    assert detection.threshold_pass["trajectory-stepwise-drift"] is True
    assert detection.missing_evidence == ("failed-control:control-no-drift-negative",)
    assert "blocks the conclusion" in detection.reason


def test_malformed_axis_rejects_and_non_applicable_omits() -> None:
    reference, drift, negative = _trajectories()
    space = LatentSpace(dim=4, source_model="probe-80-12")
    with pytest.raises(DetectionError, match="step_ids must cover"):
        TemporalInput(Trajectory(reference), Trajectory(drift), Trajectory(negative), ("only-one",), space, "R-80-12")
    with pytest.raises(DetectionError, match="at least two steps"):
        TemporalInput(
            Trajectory(reference[:1]), Trajectory(drift[:1]), Trajectory(negative[:1]), ("step-0",), space, "R-80-12"
        )
    # A bare LatentValue carries no sequence_or_time axis: temporal evaluation
    # must fail closed, not manufacture a pass.
    with pytest.raises(DetectionError, match="sequence_or_time|TemporalInput"):
        detect_families(LatentValue(np.zeros((20, 4)), _space()), _temporal_config(), controls={})


def test_missing_declaration_returns_honest_unsupported() -> None:
    sparse, noise = _sparse_batches()
    config = _sparse_config()
    object.__setattr__(config, "manifest_metric_families", ("collapse_rank_loss",))
    (detection,) = detect_families(_sparse_input(sparse, noise, permuted=True), config, controls={})

    assert detection.outcome == "unsupported"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("missing-declaration:sparse_feature_instability",)
    assert "refusing to borrow" in detection.reason


def test_results_are_deterministic_across_repeated_evaluation() -> None:
    reference, drift, negative = _trajectories()
    _, first = evaluate_detection(_temporal_input(drift, reference, negative), _temporal_config(), {})
    _, second = evaluate_detection(_temporal_input(drift, reference, negative), _temporal_config(), {})
    assert first == second


def test_fail_closed_inputs_reject() -> None:
    sparse, noise = _sparse_batches()
    config = _sparse_config()
    with pytest.raises(DetectionError, match="at least two seeds"):
        SparseInput(
            LatentValue(sparse, _space()),
            tuple(f"s-{i}" for i in range(200)),
            (11,),
            {"11": LatentValue(sparse, _space())},
            {"neg": LatentValue(noise, _space())},
            "R-80-12",
        )
    with pytest.raises(DetectionError, match="no external control batches"):
        detect_families(_sparse_input(sparse, noise, permuted=True), config, controls={"control-unknown": 1})  # type: ignore[dict-item]
    with pytest.raises(DetectionError, match="not declared by the manifest"):
        detection_config_from_manifest(
            DiagnosticRequest(
                request_id="r",
                manifest_id="manifest-80-12",
                capture=CaptureSelection(capture_id="c", representation_identity="R", axes=("sample", "feature")),
                diagnostics=DiagnosticSelection(family_ids=("sparse_feature_instability",)),
                controls=ControlSelection(control_ids=("control-nope",), metric_ids=("sparse-cross-seed-stability",)),
                output=OutputSelection(output_location="artifacts/diagnostics/r"),
            ),
            _manifest_shape(),
        )
    with pytest.raises(DetectionError, match="not supported by this detector"):
        detection_config_from_manifest(
            DiagnosticRequest(
                request_id="r",
                manifest_id="manifest-80-12",
                capture=CaptureSelection(capture_id="c", representation_identity="R", axes=("sample", "feature")),
                diagnostics=DiagnosticSelection(family_ids=("collapse_rank_loss",)),
                controls=ControlSelection(control_ids=("a",), metric_ids=("b",)),
                output=OutputSelection(output_location="artifacts/diagnostics/r"),
            ),
            _manifest_shape(),
        )
    with pytest.raises(DetectionError, match="unsupported threshold comparator"):
        FamilyThreshold(metric_id="m", comparator="!=", value=1.0, tolerance=0.1)


def _manifest_shape() -> dict[str, object]:
    import hashlib
    import json as _json

    manifest: dict[str, object] = {
        "schema_version": "benchmark-manifest-schema-v1",
        "manifest_id": "manifest-80-12",
        "status": "predeclared",
        "model": {
            "id": "m",
            "revision": "r",
            "revision_kind": "digest",
            "revision_immutable": True,
            "artifact_digest": "a" * 64,
        },
        "dataset": {"id": "d", "revision": "r1", "split": "test", "split_identity": "s", "data_digest": "b" * 64},
        "representation": {
            "identity": "R-80-12",
            "axes": [
                {"name": "sample", "selection": "all", "identity": "ax-s"},
                {"name": "feature", "selection": "all", "identity": "ax-f"},
                {"name": "seed", "selection": "11-12", "identity": "ax-seed"},
            ],
        },
        "defect": {"classification": "none", "declaration": "n/a", "predeclared": True, "source": "not_applicable"},
        "metrics": [
            {
                "id": "sparse-cross-seed-stability",
                "taxonomy_family_id": SPARSE,
                "direction": "higher_is_better",
                "unit": "rate",
                "estimator": "dict",
                "aggregation": "mean-with-interval",
            },
            {
                "id": "sparse-reconstruction-quality",
                "taxonomy_family_id": SPARSE,
                "direction": "higher_is_better",
                "unit": "gain",
                "estimator": "dict",
                "aggregation": "mean-with-interval",
            },
        ],
        "seeds": {"training": [11], "evaluation": [5], "controls": [99], "independent": True},
        "controls": [
            {
                "id": "control-stable-negative",
                "kind": "counterexample",
                "metric_ids": ["sparse-cross-seed-stability"],
                "required": True,
                "expected_behavior": "stable negative fails",
            },
            {
                "id": "control-seed-noise",
                "kind": "seed",
                "metric_ids": ["sparse-cross-seed-stability"],
                "required": False,
                "expected_behavior": "recorded",
            },
        ],
        "uncertainty": {
            "method": "bootstrap",
            "confidence_level": 0.95,
            "repetitions": 20,
            "unit_of_analysis": "sample",
            "predeclared": True,
        },
        "causal_expectation": {
            "applicable": False,
            "expectation": "not_applicable",
            "target_metric_ids": [],
            "falsification_rule": "n/a",
            "non_applicable_reason": "no causal trial in 80.12",
        },
        "thresholds": [
            {
                "metric_id": "sparse-cross-seed-stability",
                "comparator": ">=",
                "value": 0.5,
                "tolerance": 0.02,
                "predeclared": True,
            },
            {
                "metric_id": "sparse-reconstruction-quality",
                "comparator": ">=",
                "value": 0.5,
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
    unsigned = dict(manifest)
    commitment = dict(unsigned["commitment"])  # type: ignore[union-attr]
    commitment.pop("manifest_sha256", None)
    unsigned["commitment"] = commitment
    manifest["commitment"]["manifest_sha256"] = hashlib.sha256(  # type: ignore[index]
        _json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest


def test_manifest_shape_parses_and_workflow_integrates() -> None:
    manifest = _manifest_shape()
    request = DiagnosticRequest(
        request_id="request-80-12",
        manifest_id="manifest-80-12",
        capture=CaptureSelection(
            capture_id="capture-sparse", representation_identity="R-80-12", axes=("sample", "feature", "seed")
        ),
        diagnostics=DiagnosticSelection(family_ids=(SPARSE,)),
        controls=ControlSelection(
            control_ids=("control-stable-negative",),
            metric_ids=("sparse-cross-seed-stability", "sparse-reconstruction-quality"),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-12"),
    )
    config = detection_config_from_manifest(request, manifest)
    assert config.seed_axis == (11, 12)
    sparse, noise = _sparse_batches()
    target = _sparse_input(sparse, noise, permuted=True)
    (pre,) = detect_families(target, config, controls={})
    detect = make_detect_executor(target, config, controls={})

    def _passthrough(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "detect":
            return detect(invocation)
        payload: dict[str, object] = {"stage": invocation.stage}
        if invocation.stage == "report":
            payload["report_id"] = "report-80-12"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    executors = {stage: _passthrough for stage in WORKFLOW_STAGES}
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    workflow = DiagnosticWorkflow(executors, versions)  # type: ignore[arg-type]
    result, _ = workflow.run(request, manifest)

    assert result.status == "completed"
    detect_stage = result.stage_results["stages"]["detect"]
    assert detect_stage["outcome"] == "completed"
    assert detect_stage["payload"]["families"][0]["family_id"] == SPARSE
    assert "DiagnosticWorkflow" not in str(type(detect))
    assert detect.detector_version == "sparse-temporal-detector-v1"  # type: ignore[attr-defined]
