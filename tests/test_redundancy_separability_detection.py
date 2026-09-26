"""Consumer-observable tests for Sprint 80.10 redundancy/separability detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageInvocation,
    StageOutput,
)
from latent_anything._redundancy_separability_detection import (
    DECLARED_CAPACITY,
    DetectionConfig,
    DetectionError,
    FamilyThreshold,
    LabeledBatch,
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

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _manifest() -> dict[str, object]:
    path = ARTIFACTS / "benchmark_manifest_sprint80_transformer_hidden_state_v1.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _request(families: tuple[str, ...] = ("separability_probe_leakage",)) -> DiagnosticRequest:
    return DiagnosticRequest(
        request_id="request-80-10",
        manifest_id="sprint80-core-transformer-hidden-state-probe-v1",
        capture=CaptureSelection(
            capture_id="capture-transformer",
            representation_identity="openai-community-gpt2:hidden-states:layers0-11:dim768",
            axes=("sample", "feature", "label"),
        ),
        diagnostics=DiagnosticSelection(family_ids=families),
        controls=ControlSelection(
            control_ids=("control-capacity", "control-label-randomization", "control-nonseparable-negative"),
            metric_ids=("heldout-probe-accuracy", "probe-leakage-gap"),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-10"),
    )


def _redundancy_config() -> DetectionConfig:
    return DetectionConfig(
        manifest_id="request-80-10-redundancy",
        family_ids=("redundancy_superposition",),
        metric_ids=("feature-max-abs-correlation", "sparse-feature-sharing"),
        thresholds=(
            FamilyThreshold("feature-max-abs-correlation", ">=", 0.5, 0.02),
            FamilyThreshold("sparse-feature-sharing", ">=", 0.75, 0.02),
        ),
        required_controls=("control-redundant-counterexample",),
        optional_controls=("control-redundant-null-shuffle",),
        control_metrics={
            "control-redundant-counterexample": ("feature-max-abs-correlation", "sparse-feature-sharing"),
            "control-redundant-null-shuffle": ("feature-max-abs-correlation", "sparse-feature-sharing"),
        },
        control_kinds={
            "control-redundant-counterexample": "counterexample",
            "control-redundant-null-shuffle": "randomized",
        },
        repetitions=20,
        confidence_level=0.95,
        training_seed=79,
        evaluation_seed=80,
        control_seed=81,
    )


def _space(dim: int = 6) -> LatentSpace:
    return LatentSpace(dim=dim, source_model="probe-80-10")


def _labeled(
    matrix: np.ndarray,
    labels: np.ndarray,
    train: tuple[int, ...],
    evaluation: tuple[int, ...],
    *,
    capacity: str = DECLARED_CAPACITY,
    train_identity: str = "split-train-A",
    eval_identity: str = "split-eval-A",
) -> LabeledBatch:
    n = int(matrix.shape[0])
    return LabeledBatch(
        LatentValue(np.asarray(matrix, dtype=np.float64), _space(int(matrix.shape[1]))),
        tuple(int(item) for item in labels),
        tuple(f"sample-{index}" for index in range(n)),
        train,
        evaluation,
        train_identity,
        eval_identity,
        capacity,
    )


def _separable_batches(seed: int = 0) -> tuple[LabeledBatch, LabeledBatch]:
    rng = np.random.default_rng(seed)
    n, dim = 200, 6
    target_matrix = rng.normal(size=(n, dim))
    target_matrix[:100] += 1.5
    target_matrix[100:] -= 1.5
    labels = np.array([0] * 100 + [1] * 100)
    perm = np.random.default_rng(79).permutation(n)
    train = tuple(int(item) for item in perm[:140])
    evaluation = tuple(int(item) for item in perm[140:])
    negative_matrix = rng.normal(size=(n, dim))
    return (
        _labeled(target_matrix, labels, train, evaluation),
        _labeled(negative_matrix, labels, train, evaluation),
    )


def _negative_controls(negative: LabeledBatch) -> dict[str, LabeledBatch]:
    return {"control-nonseparable-negative": negative}


def test_leakage_safe_separable_positive_with_all_controls_passed() -> None:
    target, negative = _separable_batches()
    config = detection_config_from_manifest(_request(), _manifest())
    (detection,) = detect_families(target, config, controls=_negative_controls(negative))

    assert detection.family_id == "separability_probe_leakage"
    assert detection.outcome == "supported"
    assert detection.claim_allowed is True
    assert detection.threshold_pass == {"heldout-probe-accuracy": True, "probe-leakage-gap": True}
    assert detection.control_outcomes == {
        "control-capacity": "passed",
        "control-label-randomization": "passed",
        "control-nonseparable-negative": "passed",
        "control-split-swap": "passed",
    }
    assert detection.missing_evidence == ()
    assert detection.evidence_status["probe_split_provenance"] == "observed"
    assert detection.evidence_status["probe_nonseparable_negative_control"] == "observed"
    assert detection.observed_metrics["heldout-probe-accuracy"] >= 0.7
    assert detection.observed_metrics["probe-leakage-gap"] >= 0.15


def test_label_leakage_overlap_rejects_fail_closed() -> None:
    target, _ = _separable_batches()
    with pytest.raises(DetectionError, match="split leaks"):
        LabeledBatch(
            target.value,
            target.labels,
            target.sample_ids,
            target.train_indices,
            (*target.eval_indices, target.train_indices[0]),
            target.train_split_identity,
            target.eval_split_identity,
            target.capacity,
        )


def test_duplicate_sample_identity_rejects_fail_closed() -> None:
    target, _ = _separable_batches()
    duplicated = (*target.sample_ids[:1], *target.sample_ids[:1], *target.sample_ids[2:])
    with pytest.raises(DetectionError, match="sample identities must be unique"):
        LabeledBatch(
            target.value,
            target.labels,
            duplicated,
            target.train_indices,
            target.eval_indices,
            target.train_split_identity,
            target.eval_split_identity,
            target.capacity,
        )


def test_identical_split_identity_rejects_fail_closed() -> None:
    target, _ = _separable_batches()
    with pytest.raises(DetectionError, match="split identities must be distinct"):
        LabeledBatch(
            target.value,
            target.labels,
            target.sample_ids,
            target.train_indices,
            target.eval_indices,
            "split-same",
            "split-same",
            target.capacity,
        )


def test_capacity_violation_blocks_promotion() -> None:
    target, negative = _separable_batches()
    config = detection_config_from_manifest(_request(), _manifest())
    over_capacity = LabeledBatch(
        target.value,
        target.labels,
        target.sample_ids,
        target.train_indices,
        target.eval_indices,
        target.train_split_identity,
        target.eval_split_identity,
        "mlp-hidden-512-unbounded",
    )
    (detection,) = detect_families(over_capacity, config, controls=_negative_controls(negative))

    assert detection.outcome == "inconclusive"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("failed-control:control-capacity",)
    assert "blocks the conclusion" in detection.reason


def test_memorization_via_label_randomization_blocks_promotion() -> None:
    seed = 11
    rng = np.random.default_rng(seed)
    n, dim = 200, 6
    noise = rng.normal(size=(n, dim))
    labels = np.array([0] * 100 + [1] * 100)
    perm = np.random.default_rng(79).permutation(n)
    train = tuple(int(item) for item in perm[:140])
    evaluation = tuple(int(item) for item in perm[140:])
    memorizer = _labeled(noise, labels, train, evaluation)
    negative = _labeled(rng.normal(size=(n, dim)), labels, train, evaluation)
    config = detection_config_from_manifest(_request(), _manifest())
    (detection,) = detect_families(memorizer, config, controls=_negative_controls(negative))

    # Held-out noise cannot separate: the unmet headline threshold blocks
    # promotion first, and accuracy alone never promotes.
    assert detection.outcome == "inconclusive"
    assert detection.claim_allowed is False
    assert detection.threshold_pass["heldout-probe-accuracy"] is False
    assert detection.missing_evidence[0] == "threshold:heldout-probe-accuracy"
    assert "probe accuracy alone is not a diagnosis" in detection.reason


def test_nonseparable_negative_blocks_when_it_separates() -> None:
    target, _ = _separable_batches()
    config = detection_config_from_manifest(_request(), _manifest())
    # A "negative" that is itself separable must fail its control and block.
    (detection,) = detect_families(target, config, controls=_negative_controls(target))

    assert detection.outcome == "inconclusive"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("failed-control:control-nonseparable-negative",)


def test_high_accuracy_alone_cannot_promote_without_gap() -> None:
    # Weak-signal fixture: held-out accuracy clears 0.7 while the seeded
    # label-randomization gap stays below 0.15, so accuracy alone must not
    # promote even though the headline threshold passes.
    rng = np.random.default_rng(48)
    n, dim = 200, 6
    matrix = rng.normal(size=(n, dim))
    matrix[:100] += 0.3
    matrix[100:] -= 0.3
    labels = np.array([0] * 100 + [1] * 100)
    perm = np.random.default_rng(79).permutation(n)
    train = tuple(int(item) for item in perm[:140])
    evaluation = tuple(int(item) for item in perm[140:])
    target = _labeled(matrix, labels, train, evaluation)
    negative = _labeled(rng.normal(size=(n, dim)), labels, train, evaluation)
    config = detection_config_from_manifest(_request(), _manifest())
    (detection,) = detect_families(target, config, controls=_negative_controls(negative))

    assert detection.threshold_pass["heldout-probe-accuracy"] is True
    assert detection.threshold_pass["probe-leakage-gap"] is False
    assert detection.outcome == "inconclusive"
    assert detection.claim_allowed is False
    assert "threshold:probe-leakage-gap" in detection.missing_evidence
    assert "probe accuracy alone is not a diagnosis" in detection.reason


def test_redundancy_positive_reports_both_lenses_with_counterexample() -> None:
    rng = np.random.default_rng(3)
    n, dim = 200, 4
    base = rng.normal(size=(n, 1))
    redundant = np.hstack(
        [
            base + rng.normal(scale=0.05, size=(n, 1)),
            base * 0.9 + rng.normal(scale=0.05, size=(n, 1)),
            rng.normal(size=(n, 2)),
        ]
    )
    healthy = rng.normal(size=(n, dim))
    space = LatentSpace(dim=dim, source_model="probe-80-10")
    config = _redundancy_config()
    (detection,) = detect_families(
        LatentValue(redundant, space),
        config,
        controls={"control-redundant-counterexample": LatentValue(healthy, space)},
    )

    assert detection.family_id == "redundancy_superposition"
    assert detection.outcome == "supported"
    assert detection.claim_allowed is True
    assert detection.threshold_pass == {"feature-max-abs-correlation": True, "sparse-feature-sharing": True}
    assert detection.control_outcomes == {
        "control-redundant-counterexample": "passed",
        "control-redundant-null-shuffle": "passed",
    }
    assert detection.observed_metrics["feature-max-abs-correlation"] >= 0.5
    assert detection.observed_metrics["sparse-feature-sharing"] >= 0.75


def test_redundancy_correlation_alone_cannot_promote() -> None:
    rng = np.random.default_rng(3)
    n, dim = 200, 4
    base = rng.normal(size=(n, 1))
    redundant = np.hstack(
        [
            base + rng.normal(scale=0.05, size=(n, 1)),
            base * 0.9 + rng.normal(scale=0.05, size=(n, 1)),
            rng.normal(size=(n, 2)),
        ]
    )
    healthy = rng.normal(size=(n, dim))
    space = LatentSpace(dim=dim, source_model="probe-80-10")
    strict = DetectionConfig(
        manifest_id="request-80-10-redundancy",
        family_ids=("redundancy_superposition",),
        metric_ids=("feature-max-abs-correlation", "sparse-feature-sharing"),
        thresholds=(
            FamilyThreshold("feature-max-abs-correlation", ">=", 0.5, 0.02),
            FamilyThreshold("sparse-feature-sharing", ">=", 0.99, 0.001),
        ),
        required_controls=("control-redundant-counterexample",),
        optional_controls=(),
        control_metrics={
            "control-redundant-counterexample": ("feature-max-abs-correlation", "sparse-feature-sharing"),
        },
        control_kinds={"control-redundant-counterexample": "counterexample"},
        repetitions=20,
        confidence_level=0.95,
        training_seed=79,
        evaluation_seed=80,
        control_seed=81,
    )
    (detection,) = detect_families(
        LatentValue(redundant, space),
        strict,
        controls={"control-redundant-counterexample": LatentValue(healthy, space)},
    )

    assert detection.outcome == "inconclusive"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("inconsistent-evidence:redundancy_superposition",)
    assert "Correlation alone" not in detection.reason
    assert "correlation without sparse-comparison support" in detection.reason


def test_redundancy_unsupported_without_predeclared_metrics() -> None:
    target, negative = _separable_batches()
    config = detection_config_from_manifest(_request(), _manifest())
    detections = detect_families(
        LabeledBatch(
            target.value,
            target.labels,
            target.sample_ids,
            target.train_indices,
            target.eval_indices,
            target.train_split_identity,
            target.eval_split_identity,
            target.capacity,
        ),
        DetectionConfig(
            manifest_id=config.manifest_id,
            family_ids=("redundancy_superposition",),
            metric_ids=config.metric_ids,
            thresholds=config.thresholds,
            required_controls=config.required_controls,
            optional_controls=config.optional_controls,
            control_metrics=dict(config.control_metrics),
            control_kinds=dict(config.control_kinds),
            repetitions=config.repetitions,
            confidence_level=config.confidence_level,
            training_seed=config.training_seed,
            evaluation_seed=config.evaluation_seed,
            control_seed=config.control_seed,
        ),
        controls=_negative_controls(negative),
    )
    (detection,) = detections
    assert detection.outcome == "unsupported"
    assert detection.claim_allowed is False
    assert detection.missing_evidence == ("missing-declaration:redundancy_superposition",)
    assert "refusing to borrow" in detection.reason


def test_results_are_deterministic_across_repeated_evaluation() -> None:
    target, negative = _separable_batches(seed=7)
    config = detection_config_from_manifest(_request(), _manifest())
    _, first = evaluate_detection(target, config, _negative_controls(negative))
    _, second = evaluate_detection(target, config, _negative_controls(negative))
    assert first == second


def test_fail_closed_inputs_reject() -> None:
    target, negative = _separable_batches()
    config = detection_config_from_manifest(_request(), _manifest())

    finite_value = LatentValue(np.zeros((200, 6)), _space(6))
    object.__setattr__(finite_value, "_data", np.ascontiguousarray(np.full((200, 6), np.inf)))
    non_finite = LabeledBatch.__new__(LabeledBatch)
    object.__setattr__(non_finite, "value", finite_value)
    object.__setattr__(non_finite, "labels", tuple(int(item) for item in np.array([0] * 100 + [1] * 100)))
    object.__setattr__(non_finite, "sample_ids", tuple(f"sample-{index}" for index in range(200)))
    object.__setattr__(non_finite, "train_indices", target.train_indices)
    object.__setattr__(non_finite, "eval_indices", target.eval_indices)
    object.__setattr__(non_finite, "train_split_identity", target.train_split_identity)
    object.__setattr__(non_finite, "eval_split_identity", target.eval_split_identity)
    object.__setattr__(non_finite, "capacity", target.capacity)
    with pytest.raises(DetectionError, match="non-finite"):
        detect_families(non_finite, config, controls=_negative_controls(negative))

    with pytest.raises(DetectionError, match="missing batch data"):
        detect_families(target, config, controls={})

    with pytest.raises(DetectionError, match="unknown control"):
        detect_families(target, config, controls={**_negative_controls(negative), "control-unknown": negative})

    with pytest.raises(DetectionError, match="derived by the detector"):
        detect_families(
            target,
            config,
            controls={**_negative_controls(negative), "control-capacity": negative},
        )

    with pytest.raises(DetectionError, match="at least two classes"):
        _labeled(
            np.asarray(target.value.to_numpy()),
            np.zeros(200, dtype=int),
            target.train_indices,
            target.eval_indices,
        )

    mutated = dict(_manifest())
    thresholds = [dict(item) for item in mutated["thresholds"]]  # type: ignore[union-attr]
    mutated["thresholds"] = [item for item in thresholds if item["metric_id"] != "heldout-probe-accuracy"]
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
        detection_config_from_manifest(_request(("collapse_rank_loss",)), _manifest())

    with pytest.raises(DetectionError, match="unsupported threshold comparator"):
        FamilyThreshold(metric_id="m", comparator="!=", value=1.0, tolerance=0.1)


def test_detect_executor_integrates_with_workflow_without_coordinator_algorithms() -> None:
    target, negative = _separable_batches()
    config = detection_config_from_manifest(_request(), _manifest())
    detect = make_detect_executor(target, config, controls=_negative_controls(negative))

    def _passthrough(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "detect":
            return detect(invocation)
        payload: dict[str, object] = {"stage": invocation.stage}
        if invocation.stage == "report":
            payload["report_id"] = "report-80-10"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    executors = {stage: _passthrough for stage in WORKFLOW_STAGES}
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    workflow = DiagnosticWorkflow(executors, versions)  # type: ignore[arg-type]
    result, _ = workflow.run(_request(), _manifest())

    assert result.status == "completed"
    detect_stage = result.stage_results["stages"]["detect"]
    assert detect_stage["outcome"] == "completed"
    assert detect_stage["payload"]["families"][0]["family_id"] == "separability_probe_leakage"
    assert detect_stage["payload"]["config"]["manifest_id"] == "sprint80-core-transformer-hidden-state-probe-v1"
    assert "DiagnosticWorkflow" not in str(type(detect))
    assert detect.detector_version == "redundancy-separability-detector-v1"  # type: ignore[attr-defined]
