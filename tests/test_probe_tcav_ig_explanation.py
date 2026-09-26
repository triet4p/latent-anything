"""Consumer-observable tests for Sprint 80.16 probe/TCAV/IG explanation evidence."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything._probe_tcav_ig_explanation import (
    DECLARED_PROBE_CAPACITY,
    SUPPORTED_METHODS,
    ExplanationError,
    ExplanationHypothesis,
    MethodInputs,
    evaluate_explanations,
    explanation_payload,
    explanation_report_items,
    make_explain_executor,
)


def _hypothesis(method: str, hypothesis_id: str = "hyp-80-16", **overrides: object) -> ExplanationHypothesis:
    base: dict[str, object] = {
        "hypothesis_id": hypothesis_id,
        "symptom_id": "symptom-1",
        "family_id": "separability_probe_leakage",
        "target_id": "target-1",
        "representation_id": "rep-80-16",
        "layer_id": "layer-1",
        "slice_id": "slice-A",
        "method": method,
        "expected_direction": "higher",
        "dataset_id": "dataset-80-16",
        "train_split_identity": "split-train-A",
        "eval_split_identity": "split-eval-A",
        "seeds": (7, 8),
        "control_ids": (
            "capacity:control-capacity",
            "randomized:control-label-randomization",
            "negative:control-nonseparable-negative",
        ),
        "metric_ids": ("heldout_accuracy", "leakage_gap"),
        "thresholds": (("heldout_accuracy", ">=", 0.7), ("leakage_gap", ">=", 0.15)),
        "localization_bindings": (("layer", "layer-1"), ("slice", "slice-A")),
    }
    if method == "tcav":
        base.update(
            {
                "concept_id": "concept-stripes",
                "control_ids": (
                    "fidelity:control-concept-fidelity",
                    "stability:control-cav-stability",
                    "selectivity:control-random-concept",
                ),
                "metric_ids": ("concept_separability", "selectivity_margin"),
                "thresholds": (
                    ("concept_separability", ">=", 0.7),
                    ("cav_stability", ">=", 0.9),
                    ("selectivity_margin", ">=", 0.4),
                ),
                "negative_concept_ids": ("concept-dots",),
            }
        )
    if method == "integrated_gradients":
        base.update(
            {
                "baseline_policy": "zero",
                "control_ids": (
                    "fidelity:control-ig-completeness",
                    "stability:control-ig-stability",
                    "selectivity:control-ig-selectivity",
                ),
                "metric_ids": ("completeness_error", "selectivity_margin"),
                "thresholds": (
                    ("completeness_error", "<=", 0.05),
                    ("attribution_stability", ">=", 0.9),
                    ("selectivity_margin", ">=", 0.3),
                ),
            }
        )
    base.update(overrides)
    return ExplanationHypothesis(**base)  # type: ignore[arg-type]


def _probe_bundle(shift: float = 1.5, seed: int = 0, capacity: str = DECLARED_PROBE_CAPACITY) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n, dim = 200, 6
    matrix = rng.normal(size=(n, dim))
    matrix[:100] += shift
    matrix[100:] -= shift
    labels = np.array([0] * 100 + [1] * 100)
    perm = np.random.default_rng(79).permutation(n)
    train = np.array([int(item) for item in perm[:140]])
    evaluation = np.array([int(item) for item in perm[140:]])
    neg_matrix = rng.normal(size=(n, dim))
    from latent_anything.probes import _fast_probe

    negative = float(
        _fast_probe(neg_matrix[train], labels[train], neg_matrix[evaluation], labels[evaluation], 7).accuracy
    )
    return {
        "train_matrix": matrix[train],
        "eval_matrix": matrix[evaluation],
        "train_labels": labels[train],
        "eval_labels": labels[evaluation],
        "train_ids": [f"sample-{index}" for index in range(n) if index in set(int(i) for i in train)],
        "eval_ids": [f"sample-{index}" for index in range(n) if index in set(int(i) for i in evaluation)],
        "capacity": capacity,
        "negative_accuracy": negative,
    }


def _tcav_bundle(seed: int = 3) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    dim = 8
    direction = np.zeros(dim)
    direction[0] = 1.0
    concept = rng.normal(size=(60, dim)) + np.array([3.0] + [0.0] * (dim - 1))
    reference = rng.normal(size=(60, dim))
    grad_rng = np.random.default_rng(9)
    gradients = grad_rng.normal(size=(40, dim)) * 0.3 + np.array([2.0] + [0.0] * (dim - 1))
    neg_concept = rng.normal(size=(60, dim)) + np.array([0.0, 3.0] + [0.0] * (dim - 2))
    neg_reference = rng.normal(size=(60, dim))
    return {
        "gradients": gradients,
        "concept_matrix": concept,
        "reference_matrix": reference,
        "direction_method": "mean_diff",
        "target_identity": "target-1",
        "concept_ids": [f"concept-{index}" for index in range(60)],
        "reference_ids": [f"reference-{index}" for index in range(60)],
        "negative_concepts": {
            "concept-dots": {"concept_matrix": neg_concept, "reference_matrix": neg_reference},
        },
    }


def _ig_bundle() -> dict[str, object]:
    rng = np.random.default_rng(11)
    dim = 8
    on_target = rng.normal(size=dim) + np.array([2.0] + [0.0] * (dim - 1))
    off_target = rng.normal(size=dim)
    random_baseline = rng.normal(size=dim)
    variant = on_target + rng.normal(size=dim) * 0.01
    return {
        "attributions": on_target,
        "baseline_attributions": {"zero-variant": variant},
        "off_target_attributions": off_target,
        "random_attributions": random_baseline,
        "completeness_error": 0.001,
        "completeness_delta": 1.5,
        "target_identity": "target-1",
        "baseline_identity": "zero-baseline",
        "input_identity": "input-activation-A",
    }


def test_undeclared_method_never_invokes_method_code() -> None:
    hypothesis = _hypothesis("probe")
    calls = {"count": 0}

    def _callback() -> dict[str, object]:
        calls["count"] += 1
        return _probe_bundle()

    # No declared input bundle: the callback must remain uncalled.
    (record,), payload = evaluate_explanations(hypothesis if False else (hypothesis,), MethodInputs())
    assert calls["count"] == 0
    assert record.outcome == "omitted"
    assert record.claim_allowed is False
    assert record.missing_evidence == ("omitted:undeclared-hypothesis-or-method",)
    assert payload["evidence"][0]["outcome"] == "omitted"


def test_hypothesis_serialization_preserves_target_binding() -> None:
    hypothesis = _hypothesis("probe", target_id="target-stability-frozen")
    assert hypothesis.to_dict()["target_id"] == "target-stability-frozen"


def test_undeclared_hypothesis_method_pair_stays_omitted() -> None:
    probe = _hypothesis("probe", hypothesis_id="hyp-probe")
    tcav = _hypothesis("tcav", hypothesis_id="hyp-tcav")
    (probe_record, tcav_record), _ = evaluate_explanations((probe, tcav), MethodInputs(probe=_probe_bundle()))
    assert probe_record.outcome == "supported"
    assert tcav_record.outcome == "omitted"
    assert tcav_record.claim_allowed is False


def test_leakage_safe_selective_stable_probe_supports() -> None:
    hypothesis = _hypothesis("probe")
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(probe=_probe_bundle()))
    assert record.outcome == "supported"
    assert record.claim_allowed is True
    assert record.missing_evidence == ()
    assert record.observed_effect["heldout_accuracy"] >= 0.7
    assert record.observed_effect["leakage_gap"] >= 0.15
    assert record.fidelity["status"] == "passed"
    assert record.stability["status"] == "passed"
    assert record.selectivity["status"] == "passed"
    assert record.leakage["status"] == "passed"
    assert record.leakage["capacity_passed"] is True
    assert record.uncertainty["heldout_accuracy"]["repetitions"] == 20
    assert record.central_outcomes["control-capacity"]["status"] == "passed"


def test_train_only_accuracy_is_never_promoted() -> None:
    hypothesis = _hypothesis("probe")
    bundle = _probe_bundle()
    # Even a perfect train-only score cannot substitute for held-out fidelity:
    # corrupt the eval split into pure noise so held-out fidelity fails.
    rng = np.random.default_rng(0)
    bundle["eval_matrix"] = rng.normal(size=np.asarray(bundle["eval_matrix"]).shape)
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(probe=bundle))
    assert record.outcome == "inconclusive"
    assert record.claim_allowed is False
    assert "failed-fidelity:heldout_accuracy" in record.missing_evidence


def test_leaky_probe_split_blocks_promotion() -> None:
    hypothesis = _hypothesis("probe")
    bundle = _probe_bundle()
    bundle["eval_ids"] = list(bundle["train_ids"])[:60]
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(probe=bundle))
    assert record.outcome == "inconclusive"
    assert record.claim_allowed is False
    assert any("leak" in item for item in record.missing_evidence)


def test_random_label_probe_cannot_support() -> None:
    hypothesis = _hypothesis("probe")
    bundle = _probe_bundle(shift=0.0)
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(probe=bundle))
    assert record.outcome == "inconclusive"
    assert record.claim_allowed is False
    assert record.missing_evidence != ()


def test_nonseparable_negative_blocks_when_it_separates() -> None:
    hypothesis = _hypothesis("probe")
    bundle = _probe_bundle()
    bundle["negative_accuracy"] = 0.99
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(probe=bundle))
    assert record.outcome == "inconclusive"
    assert record.claim_allowed is False
    assert "failed-control:control-nonseparable-negative" in record.missing_evidence


def _group_resample_bundle() -> dict[str, object]:
    rng = np.random.default_rng(4)
    train_y = np.tile(np.array([0, 1]), 100)
    train_sign = 2.0 * train_y - 1.0
    train_x = rng.normal(scale=0.1, size=(200, 2))
    train_x[:100, 1] += 2.0 * train_sign[:100]
    train_x[100:, 0] += 2.0 * train_sign[100:]
    eval_y = np.tile(np.array([0, 1]), 30)
    eval_sign = 2.0 * eval_y - 1.0
    eval_x = rng.normal(scale=0.1, size=(60, 2))
    eval_x[:, 0] += 2.0 * eval_sign
    eval_x[:, 1] += 2.0 * eval_sign
    return {
        "train_matrix": train_x,
        "eval_matrix": eval_x,
        "train_labels": train_y,
        "eval_labels": eval_y,
        "train_ids": [f"train-{index}" for index in range(200)],
        "eval_ids": [f"eval-{index}" for index in range(60)],
        "capacity": DECLARED_PROBE_CAPACITY,
        "negative_accuracy": 0.5,
        "stability_replicates": [
            {
                "train_matrix": train_x[:100],
                "train_labels": train_y[:100],
                "train_ids": [f"train-{index}" for index in range(100)],
                "fit_seed": 31,
            },
            {
                "train_matrix": train_x[100:],
                "train_labels": train_y[100:],
                "train_ids": [f"train-{index}" for index in range(100, 200)],
                "fit_seed": 37,
            },
        ],
    }


def test_grouped_training_replicates_can_fail_the_stability_gate() -> None:
    hypothesis = _hypothesis(
        "probe",
        thresholds=(
            ("heldout_accuracy", ">=", 0.7),
            ("coef_stability", ">=", 0.8),
            ("leakage_gap", ">=", 0.15),
        ),
    )
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(probe=_group_resample_bundle()))

    assert record.fidelity["status"] == "passed"
    assert record.selectivity["status"] == "passed"
    assert record.stability["status"] == "failed"
    assert record.stability["observed"] < 0.8
    assert record.stability["basis"] == "identity-bound-training-set-replicates"
    assert len(record.stability["replicates"]) == 2
    assert record.outcome == "inconclusive"
    assert record.claim_allowed is False
    assert "failed-stability:coef_stability" in record.missing_evidence


def test_training_stability_replicate_cannot_reuse_heldout_identities() -> None:
    bundle = _group_resample_bundle()
    bundle["stability_replicates"] = [
        {
            "train_matrix": np.asarray(bundle["train_matrix"])[:100],
            "train_labels": np.asarray(bundle["train_labels"])[:100],
            "train_ids": [f"eval-{index}" for index in range(60)] + [f"train-{index}" for index in range(40)],
            "fit_seed": 31,
        },
        {
            "train_matrix": np.asarray(bundle["train_matrix"])[100:],
            "train_labels": np.asarray(bundle["train_labels"])[100:],
            "train_ids": [f"train-{index}" for index in range(100, 200)],
            "fit_seed": 37,
        },
    ]
    hypothesis = _hypothesis(
        "probe",
        thresholds=(
            ("heldout_accuracy", ">=", 0.7),
            ("coef_stability", ">=", 0.8),
            ("leakage_gap", ">=", 0.15),
        ),
    )
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(probe=bundle))

    assert record.outcome == "inconclusive"
    assert record.claim_allowed is False
    assert "failed-stability:stability-replicates" in record.missing_evidence


def test_synthetic_known_concept_tcav_supports() -> None:
    hypothesis = _hypothesis("tcav")
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(tcav=_tcav_bundle()))
    assert record.outcome == "supported"
    assert record.claim_allowed is True
    assert record.observed_effect["tcav_score"] == 1.0
    assert record.fidelity["status"] == "passed"
    assert record.stability["status"] == "passed"
    assert record.selectivity["status"] == "passed"
    assert record.leakage["status"] == "passed"
    assert record.uncertainty["tcav_score"]["repetitions"] == 20


def test_unstable_random_concept_tcav_cannot_support() -> None:
    hypothesis = _hypothesis("tcav")
    bundle = _tcav_bundle()
    rng = np.random.default_rng(4)
    noise_concept = rng.normal(size=(60, 8))
    noise_reference = rng.normal(size=(60, 8))
    bundle["concept_matrix"] = noise_concept
    bundle["reference_matrix"] = noise_reference
    # Gradients orthogonal to any stable direction: zero-mean noise.
    bundle["gradients"] = rng.normal(size=(40, 8)) * 0.3
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(tcav=bundle))
    assert record.outcome == "inconclusive"
    assert record.claim_allowed is False
    assert record.missing_evidence != ()


def test_simple_differentiable_model_ig_supports() -> None:
    hypothesis = _hypothesis("integrated_gradients")
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(integrated_gradients=_ig_bundle()))
    assert record.outcome == "supported"
    assert record.claim_allowed is True
    assert record.fidelity["status"] == "passed"
    assert record.stability["status"] == "passed"
    assert record.selectivity["status"] == "passed"
    assert record.leakage["status"] == "passed"
    assert record.uncertainty["attribution_mass"]["repetitions"] == 20


def test_bad_baseline_ig_fails() -> None:
    hypothesis = _hypothesis("integrated_gradients")
    bundle = _ig_bundle()
    bundle["completeness_error"] = 2.5
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(integrated_gradients=bundle))
    assert record.outcome == "inconclusive"
    assert record.claim_allowed is False
    assert "failed-fidelity:completeness_error" in record.missing_evidence


def test_disconnected_target_ig_is_unsupported() -> None:
    hypothesis = _hypothesis("integrated_gradients")
    bundle = _ig_bundle()
    bundle["disconnected"] = True
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(integrated_gradients=bundle))
    assert record.outcome == "unsupported"
    assert record.claim_allowed is False
    assert record.missing_evidence == ("unsupported:disconnected-target",)


def test_each_method_missing_evidence_blocks() -> None:
    probe = _hypothesis("probe")
    (probe_record,), _ = evaluate_explanations((probe,), MethodInputs(probe={"train_matrix": [[1.0]]}))
    assert probe_record.outcome == "inconclusive"
    assert probe_record.claim_allowed is False

    tcav = _hypothesis("tcav")
    (tcav_record,), _ = evaluate_explanations((tcav,), MethodInputs(tcav={"gradients": [[1.0]]}))
    assert tcav_record.outcome in ("inconclusive", "unsupported")
    assert tcav_record.claim_allowed is False

    ig = _hypothesis("integrated_gradients")
    (ig_record,), _ = evaluate_explanations((ig,), MethodInputs(integrated_gradients={"attributions": [1.0]}))
    assert ig_record.outcome in ("inconclusive", "unsupported")
    assert ig_record.claim_allowed is False


def test_deterministic_central_streams_and_uncertainty() -> None:
    hypothesis = _hypothesis("probe")
    bundle = _probe_bundle()
    (first,), _ = evaluate_explanations((hypothesis,), MethodInputs(probe=dict(bundle)))
    (second,), _ = evaluate_explanations((hypothesis,), MethodInputs(probe=dict(bundle)))
    assert first.to_dict() == second.to_dict()
    assert first.uncertainty["heldout_accuracy"]["seed"] == second.uncertainty["heldout_accuracy"]["seed"]
    assert first.uncertainty["heldout_accuracy"]["seed"] != 0
    assert (
        first.central_outcomes["control-label-randomization"]["stream_seed"]
        == (second.central_outcomes["control-label-randomization"]["stream_seed"])
    )


def test_common_output_is_report_compatible_but_non_causal() -> None:
    from latent_anything._diagnostic_report import validate_report_shape

    probe = _hypothesis("probe", hypothesis_id="hyp-probe")
    tcav = _hypothesis("tcav", hypothesis_id="hyp-tcav")
    ig = _hypothesis("integrated_gradients", hypothesis_id="hyp-ig")
    (records, _) = evaluate_explanations(
        (probe, tcav, ig),
        MethodInputs(probe=_probe_bundle(), tcav=_tcav_bundle(), integrated_gradients=_ig_bundle()),
    )
    items = explanation_report_items(records)
    assert {item["kind"] for item in items} == {"explanation"}
    assert all(item["causal"] is False for item in items)
    report = {
        "schema_version": "diagnostic-report-schema-v1",
        "report_id": "report-80-16",
        "status": "declared",
        "capture_provenance": {
            "captures": [
                {
                    "capture_id": "capture-1",
                    "model_revision": "model@rev",
                    "dataset_split": "split@A",
                    "representation_identity": "rep-80-16",
                    "axes": ["sample", "feature"],
                    "artifact_refs": ["capture-artifact-1"],
                }
            ]
        },
        "symptoms": [
            {
                "id": "symptom-1",
                "description": "d",
                "metric_ids": ["m-1"],
                "evidence_refs": ["obs-1"],
                "status": "observed",
            }
        ],
        "localization": [
            {
                "id": "loc-1",
                "axis": "layer",
                "selection": "layer-1",
                "evidence_refs": ["obs-1"],
                "confidence": 0.8,
                "status": "observed",
            }
        ],
        "hypotheses": [
            {
                "id": "hyp-probe",
                "statement": "s",
                "evidence_refs": ["obs-1"],
                "alternatives": ["a"],
                "status": "supported",
            }
        ],
        "statistical_evidence": [
            {
                "id": "stats-1",
                "metric_id": "m-1",
                "estimate": 0.2,
                "uncertainty": {"kind": "interval", "lower": 0.1, "upper": 0.3},
                "control_refs": ["control-1"],
                "evidence_refs": ["stats-artifact-1"],
                "status": "supported",
            }
        ],
        "interventions": [
            {
                "id": "intervention-1",
                "intervention": "i",
                "target": "t",
                "control_refs": ["control-1"],
                "outcome": {},
                "evidence_refs": ["a-1"],
                "status": "supported",
            }
        ],
        "comparisons": [
            {
                "id": "comparison-1",
                "baseline": "a",
                "candidate": "b",
                "alignment": {},
                "metric_ids": ["m-1"],
                "evidence_refs": ["a-1"],
                "status": "supported",
            }
        ],
        "limitations": [{"id": "lim-1", "description": "d", "affects": ["x"], "blocking": False}],
        "next_action": {"action": "a", "rationale": "r", "required_evidence_refs": []},
        "claims": [
            {
                "id": "obs-1",
                "kind": "observation",
                "status": "observed",
                "claim": "c",
                "evidence_refs": ["capture-artifact-1"],
                "control_refs": [],
                "causal": False,
                "claim_allowed": True,
                "missing_evidence": [],
            },
            *items,
        ],
    }
    validate_report_shape(report)


def _real_prior() -> tuple[object, object, object, object]:
    import json
    from pathlib import Path

    import numpy as np

    from latent_anything._diagnostic_workflow import StageOutput as _RealStage
    from latent_anything._redundancy_separability_detection import (
        LabeledBatch as _RealBatch,
    )
    from latent_anything._redundancy_separability_detection import (
        detection_config_from_manifest as _real_config,
    )
    from latent_anything._redundancy_separability_detection import (
        evaluate_detection as _real_detect,
    )
    from latent_anything.diagnostics import (
        CaptureSelection as _Capture,
    )
    from latent_anything.diagnostics import (
        ControlSelection as _Controls,
    )
    from latent_anything.diagnostics import (
        DiagnosticRequest as _Request,
    )
    from latent_anything.diagnostics import (
        DiagnosticSelection as _Diag,
    )
    from latent_anything.diagnostics import (
        OutputSelection as _Output,
    )
    from latent_anything.latent_space import LatentSpace as _RealSpace
    from latent_anything.latent_value import LatentValue as _RealValue

    manifest = dict(
        json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "artifacts/benchmark_manifest_sprint80_transformer_hidden_state_v1.json"
            ).read_text(encoding="utf-8")
        )
    )
    request = _Request(
        request_id="request-80-16-real",
        manifest_id=str(manifest["manifest_id"]),
        capture=_Capture(
            capture_id="c",
            representation_identity="openai-community-gpt2:hidden-states:layers0-11:dim768",
            axes=("sample", "feature"),
        ),
        diagnostics=_Diag(family_ids=("separability_probe_leakage",)),
        controls=_Controls(
            control_ids=("control-capacity", "control-label-randomization", "control-nonseparable-negative"),
            metric_ids=("heldout-probe-accuracy", "probe-leakage-gap"),
        ),
        output=_Output(output_location="artifacts/diagnostics/request-80-16"),
    )
    config = _real_config(request, manifest)
    rng = np.random.default_rng(0)
    n, dim = 200, 6
    matrix = rng.normal(size=(n, dim))
    matrix[:100] += 1.5
    matrix[100:] -= 1.5
    labels = np.array([0] * 100 + [1] * 100)
    space = _RealSpace(dim=dim, source_model="probe-80-16-real")
    perm = np.random.default_rng(79).permutation(n)
    train = tuple(int(item) for item in perm[:140])
    evaluation = tuple(int(item) for item in perm[140:])
    target = _RealBatch(
        _RealValue(matrix, space),
        tuple(int(item) for item in labels),
        tuple(f"sample-{index}" for index in range(n)),
        train,
        evaluation,
        "split-train-A",
        "split-eval-A",
        "linear-logreg-C1.0-standardized",
    )
    negative = _RealBatch(
        _RealValue(rng.normal(size=(n, dim)), space),
        tuple(int(item) for item in labels),
        tuple(f"negative-{index}" for index in range(n)),
        train,
        evaluation,
        "split-train-A",
        "split-eval-A",
        "linear-logreg-C1.0-standardized",
    )
    _, detect_payload = _real_detect(target, config, {"control-nonseparable-negative": negative})
    assert "stage" not in str(dict(detect_payload))
    return manifest, request, detect_payload, _RealStage


def test_explain_executor_rejects_mismatched_identities() -> None:
    import dataclasses
    import json
    from pathlib import Path

    from latent_anything._diagnostic_workflow import StageContractError, StageInvocation
    from latent_anything._diagnostic_workflow import StageOutput as _StageOutput3
    from latent_anything.diagnostics import (
        CaptureSelection,
        ControlSelection,
        DiagnosticRequest,
        DiagnosticSelection,
        OutputSelection,
    )

    manifest = dict(
        json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "artifacts/benchmark_manifest_sprint80_transformer_hidden_state_v1.json"
            ).read_text(encoding="utf-8")
        )
    )
    manifest_id = str(manifest["manifest_id"])
    manifest_rep = "openai-community-gpt2:hidden-states:layers0-11:dim768"

    def _invocation(hypotheses: object, manifest_id_override: str | None = None) -> StageInvocation:
        executor_probe = make_explain_executor(hypotheses, MethodInputs(probe=_probe_bundle()))  # type: ignore[arg-type]
        request = DiagnosticRequest(
            request_id="request-80-16",
            manifest_id=manifest_id_override or manifest_id,
            capture=CaptureSelection(capture_id="c", representation_identity=manifest_rep, axes=("sample", "feature")),
            diagnostics=DiagnosticSelection(family_ids=("separability_probe_leakage",)),
            controls=ControlSelection(
                control_ids=("control-capacity", "control-label-randomization", "control-nonseparable-negative"),
                metric_ids=("heldout-probe-accuracy", "probe-leakage-gap"),
            ),
            output=OutputSelection(output_location="artifacts/diagnostics/request-80-16"),
        )
        prior = tuple(
            _StageOutput3(stage=stage, outcome="completed", payload={"stage": stage}, artifact_refs=())
            for stage in ("capture", "detect", "localize")
        )
        invocation = StageInvocation(
            stage="explain",
            request=request,
            manifest=manifest,
            prior=prior,
            workflow_identity="w" * 64,
            request_digest="r" * 64,
            manifest_digest="m" * 64,
            config_digest="c" * 64,
        )
        executor_probe(invocation)
        raise AssertionError("expected StageContractError")

    probe = dataclasses.replace(
        _hypothesis("probe", hypothesis_id="hyp-probe"),
        representation_id=manifest_rep,
        dataset_id="wikitext-2-raw-v1-validation",
        manifest_id=manifest_id,
    )
    # Mismatched run manifest.
    with __import__("pytest").raises(StageContractError, match="manifest"):
        _invocation((probe,), manifest_id_override="other-manifest")
    # Mismatched representation.
    wrong_rep = dataclasses.replace(probe, representation_id="SOME-OTHER-REP")
    with __import__("pytest").raises(StageContractError, match="representation"):
        _invocation((wrong_rep,))
    # Mismatched hypothesis manifest declaration.
    wrong_manifest = dataclasses.replace(probe, manifest_id="other-manifest")
    with __import__("pytest").raises(StageContractError, match="manifest"):
        _invocation((wrong_manifest,))


def test_tcav_payload_controls_use_declaration_bound_set() -> None:
    hypothesis = _hypothesis("tcav")
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(tcav=_tcav_bundle()))
    declared = {"control-concept-fidelity", "control-cav-stability", "control-random-concept"}
    assert set(record.control_outcomes) == declared
    assert set(record.central_outcomes) == declared
    assert "control-tcav-random-concept-stream" not in record.central_outcomes
    items = explanation_report_items((record,))
    assert sorted(items[0]["control_refs"]) == sorted(declared)


def test_explain_executor_binds_real_family_metric_location() -> None:
    import dataclasses

    from latent_anything._diagnostic_workflow import StageContractError, StageInvocation
    from latent_anything._layer_slice_localization import (
        DetectionEvidence as _Evidence,
    )
    from latent_anything._layer_slice_localization import (
        LayerCell as _Layer,
    )
    from latent_anything._layer_slice_localization import (
        LocalizationInput as _LocInput,
    )
    from latent_anything._layer_slice_localization import (
        SampleCell as _Sample,
    )
    from latent_anything._layer_slice_localization import (
        SliceDefinition as _Slice,
    )
    from latent_anything._layer_slice_localization import (
        evaluate_localization as _eval_loc,
    )

    manifest_real, request_real, detect_payload, _real_stage = _real_prior()
    assert _real_stage is not None
    manifest_id = str(manifest_real["manifest_id"])
    manifest_rep = "openai-community-gpt2:hidden-states:layers0-11:dim768"
    evidence = _Evidence(
        manifest_id=manifest_id,
        family_id="separability_probe_leakage",
        metric_id="heldout-probe-accuracy",
        comparator=">=",
        threshold_value=0.7,
        tolerance=0.02,
        direction="higher_is_better",
        affected_when="threshold_pass",
        required_controls=("control-a",),
        control_outcomes={"control-a": "passed"},
        outcome="supported",
        claim_allowed=True,
        representation_identity="R-80-16-real",
    )
    source = _LocInput(
        manifest_id=manifest_id,
        evidence=evidence,
        layer_order=("layer-0", "layer-1", "layer-2"),
        layer_cells=(
            _Layer("layer-0", 0.2, "R-80-16-real"),
            _Layer("layer-1", 3.5, "R-80-16-real"),
            _Layer("layer-2", 0.8, "R-80-16-real"),
        ),
        sample_cells=(
            _Sample("s0", 3.8, "R-80-16-real"),
            _Sample("s1", 3.9, "R-80-16-real"),
            _Sample("s2", 3.7, "R-80-16-real"),
            _Sample("s3", 1.0, "R-80-16-real"),
            _Sample("s4", 1.1, "R-80-16-real"),
            _Sample("s5", 3.8, "R-80-16-real"),
        ),
        declared_slice_ids=("slice-A", "slice-B"),
        slices=(
            _Slice("slice-A", "samples s0-s2", ("s0", "s1", "s2")),
            _Slice("slice-B", "samples s3-s5", ("s3", "s4", "s5")),
        ),
    )
    _, localize_payload = _eval_loc(source)
    assert str(dict(localize_payload)).find("stage") == -1

    def _run(hypotheses: object) -> str:
        from latent_anything._diagnostic_workflow import StageOutput as _Out

        executor = make_explain_executor(hypotheses, MethodInputs(probe=_probe_bundle()))  # type: ignore[arg-type]
        prior = (
            _Out(stage="capture", outcome="completed", payload={"capture_id": "c"}, artifact_refs=()),
            _Out(stage="detect", outcome="completed", payload=dict(detect_payload), artifact_refs=()),
            _Out(stage="localize", outcome="completed", payload=dict(localize_payload), artifact_refs=()),
        )
        invocation = StageInvocation(
            stage="explain",
            request=request_real,
            manifest=manifest_real,
            prior=prior,
            workflow_identity="w" * 64,
            request_digest="r" * 64,
            manifest_digest="m" * 64,
            config_digest="c" * 64,
        )
        produced = executor(invocation)
        (record,) = [item for item in produced.payload["evidence"]]  # type: ignore[index]
        return str(record.get("outcome"))

    matching = dataclasses.replace(
        _hypothesis("probe", hypothesis_id="hyp-match"),
        representation_id=manifest_rep,
        dataset_id="wikitext-2-raw-v1-validation",
        manifest_id=manifest_id,
        family_id="separability_probe_leakage",
        layer_id="layer-1",
        slice_id="slice-A",
        localization_bindings=(("layer", "layer-1"), ("slice", "slice-A")),
        metric_ids=("heldout-probe-accuracy", "probe-leakage-gap"),
        thresholds=(("heldout-probe-accuracy", ">=", 0.7), ("probe-leakage-gap", ">=", 0.15)),
    )
    fabricated_family = dataclasses.replace(matching, hypothesis_id="hyp-bad-family", family_id="family-FABRICATED")
    with __import__("pytest").raises(StageContractError, match="family"):
        _run((fabricated_family,))
    fabricated_metric = dataclasses.replace(
        matching,
        hypothesis_id="hyp-bad-metric",
        metric_ids=("metric-FABRICATED", "probe-leakage-gap"),
        thresholds=(("metric-FABRICATED", ">=", 0.7), ("probe-leakage-gap", ">=", 0.15)),
    )
    with __import__("pytest").raises(StageContractError, match="metric"):
        _run((fabricated_metric,))
    # Status words are not detect families: the real payload carries
    # ``evidence_status`` values of ``"observed"``, which must not bind.
    observed_family = dataclasses.replace(matching, hypothesis_id="hyp-observed", family_id="observed")
    with __import__("pytest").raises(StageContractError, match="family"):
        _run((observed_family,))
    # Control-kind words are not detect metrics: ``"null"`` names a control
    # kind in the real payload config, not a declared metric identity.
    null_metric = dataclasses.replace(
        matching,
        hypothesis_id="hyp-null-metric",
        metric_ids=("null",),
        thresholds=(("null", ">=", 0.7),),
    )
    with __import__("pytest").raises(StageContractError, match="metric"):
        _run((null_metric,))
    missing_binding = dataclasses.replace(matching, hypothesis_id="hyp-no-binding", localization_bindings=())
    with __import__("pytest").raises(StageContractError, match="[Ll]ocalization"):
        _run((missing_binding,))
    fabricated_layer = dataclasses.replace(
        matching,
        hypothesis_id="hyp-bad-layer",
        localization_bindings=(("layer", "layer-FABRICATED"), ("slice", "slice-A")),
    )
    with __import__("pytest").raises(StageContractError, match="[Ll]ayer|[Ll]ocalization"):
        _run((fabricated_layer,))
    fabricated_slice = dataclasses.replace(
        matching,
        hypothesis_id="hyp-bad-slice",
        localization_bindings=(("layer", "layer-1"), ("slice", "slice-FABRICATED")),
    )
    with __import__("pytest").raises(StageContractError, match="[Ss]lice|[Ll]ocalization"):
        _run((fabricated_slice,))
    # Fabricated symptom/target with a valid family still executes: they are
    # hypothesis-local identities bound by method inputs (probe labels), not
    fabricated_local = dataclasses.replace(
        matching, hypothesis_id="hyp-local", symptom_id="symptom-FABRICATED", target_id="target-FABRICATED"
    )
    assert _run((fabricated_local,)) in ("supported", "inconclusive")


def test_explain_executor_binds_real_axial_location() -> None:
    import dataclasses

    from latent_anything._diagnostic_workflow import StageContractError, StageInvocation
    from latent_anything._diagnostic_workflow import StageOutput as _AxialStage
    from latent_anything._layer_slice_localization import evaluate_axial_localization

    manifest_real, request_real, detect_payload, _ = _real_prior()
    manifest_id = str(manifest_real["manifest_id"])
    manifest_rep = "openai-community-gpt2:hidden-states:layers0-11:dim768"
    import tests.test_checkpoint_token_time_localization as _axial_tests

    _, axial_payload = evaluate_axial_localization(_axial_tests._checkpoint_source())
    assert "stage" not in str(dict(axial_payload))
    _, token_payload = evaluate_axial_localization(_axial_tests._token_source())
    _, time_payload = evaluate_axial_localization(_axial_tests._time_source())
    _, feature_payload = evaluate_axial_localization(_axial_tests._feature_source())

    def _run_axial(hypotheses: object, payload: object) -> str:
        executor = make_explain_executor(hypotheses, MethodInputs(probe=_probe_bundle()))  # type: ignore[arg-type]
        prior = (
            _AxialStage(stage="capture", outcome="completed", payload={"capture_id": "c"}, artifact_refs=()),
            _AxialStage(stage="detect", outcome="completed", payload=dict(detect_payload), artifact_refs=()),
            _AxialStage(stage="localize", outcome="completed", payload=dict(payload), artifact_refs=()),  # type: ignore[arg-type]
        )
        invocation = StageInvocation(
            stage="explain",
            request=request_real,
            manifest=manifest_real,
            prior=prior,
            workflow_identity="w" * 64,
            request_digest="r" * 64,
            manifest_digest="m" * 64,
            config_digest="c" * 64,
        )
        produced = executor(invocation)
        (record,) = [item for item in produced.payload["evidence"]]  # type: ignore[index]
        return str(record.get("outcome"))

    def _axial_hypothesis(
        hypothesis_id: str,
        bindings: object,
        *,
        family_id: str = "sequence_trajectory_drift",
        metric_ids: object = ("axial-health",),
    ) -> ExplanationHypothesis:
        return dataclasses.replace(
            _hypothesis("probe", hypothesis_id=hypothesis_id),
            representation_id=manifest_rep,
            dataset_id="wikitext-2-raw-v1-validation",
            manifest_id=manifest_id,
            family_id=family_id,
            metric_ids=metric_ids,  # type: ignore[arg-type]
            thresholds=tuple((metric, ">=", 0.1) for metric in metric_ids),  # type: ignore[union-attr]
            localization_bindings=bindings,  # type: ignore[arg-type]
        )

    def _axial_detect_payload() -> dict[str, object]:
        """Relabel a copy of the real detect payload to the axial family.

        Keeps every structured identity field (config metric identities,
        observed/threshold metric keys, family-evidence keys) intact while
        renaming the family and its metric vocabulary to the axial
        ``sequence_trajectory_drift``/``axial-health`` composition, so the
        full capture/detect/localize prior shares one exact family/metric.
        """
        import copy as _copy

        relabeled = _copy.deepcopy(dict(detect_payload))
        relabeled_families = []
        for entry in relabeled.get("families", []):
            entry = dict(entry)
            entry["family_id"] = "sequence_trajectory_drift"
            entry["observed_metrics"] = {"axial-health": 3.6}
            entry["threshold_pass"] = {"axial-health": True}
            relabeled_families.append(entry)
        relabeled["families"] = relabeled_families
        relabeled["family_evidence"] = {"sequence_trajectory_drift": {"axial_metric": "observed"}}
        config = dict(relabeled.get("config", {}))
        config["family_ids"] = ["sequence_trajectory_drift"]
        config["metric_ids"] = ["axial-health"]
        config["thresholds"] = [{"comparator": ">=", "metric_id": "axial-health", "tolerance": 0.1, "value": 3.0}]
        relabeled["config"] = config
        return relabeled

    def _run_composed(hypotheses: object, payload: object) -> str:
        """Run the axial composition through the full prior stages.

        The shared localize gate compares exact declared localize
        ``family_id``/``metric_id`` whenever the payload declares them, so the
        detect prior must share the axial family/metric vocabulary.
        """
        executor = make_explain_executor(hypotheses, MethodInputs(probe=_probe_bundle()))  # type: ignore[arg-type]
        prior = (
            _AxialStage(stage="capture", outcome="completed", payload={"capture_id": "c"}, artifact_refs=()),
            _AxialStage(stage="detect", outcome="completed", payload=_axial_detect_payload(), artifact_refs=()),
            _AxialStage(stage="localize", outcome="completed", payload=dict(payload), artifact_refs=()),  # type: ignore[arg-type]
        )
        invocation = StageInvocation(
            stage="explain",
            request=request_real,
            manifest=manifest_real,
            prior=prior,
            workflow_identity="w" * 64,
            request_digest="r" * 64,
            manifest_digest="m" * 64,
            config_digest="c" * 64,
        )
        produced = executor(invocation)
        (record,) = [item for item in produced.payload["evidence"]]  # type: ignore[index]
        return str(record.get("outcome"))

    assert _run_composed((_axial_hypothesis("hyp-ckpt", (("checkpoint", "ckpt-1"),)),), dict(axial_payload)) in (
        "supported",
        "inconclusive",
    )
    assert _run_composed((_axial_hypothesis("hyp-tok", (("token", "tok-2"),)),), dict(token_payload)) in (
        "supported",
        "inconclusive",
    )
    assert _run_composed((_axial_hypothesis("hyp-time", (("time", "step-2"),)),), dict(time_payload)) in (
        "supported",
        "inconclusive",
    )
    assert _run_composed((_axial_hypothesis("hyp-feature", (("feature", "feature-0"),)),), dict(feature_payload)) in (
        "supported",
        "inconclusive",
    )
    # A hypothesis declaring the wrong family fails the exact detect family
    # gate before any method callback, even with a satisfied binding.
    with __import__("pytest").raises(StageContractError, match="detect family"):
        _run_composed(
            (_axial_hypothesis("hyp-ckpt", (("checkpoint", "ckpt-1"),), family_id="separability_probe_leakage"),),
            dict(axial_payload),
        )
    with __import__("pytest").raises(StageContractError, match="[Cc]heckpoint|[Ll]ocalization"):
        _run_composed((_axial_hypothesis("hyp-bad-ckpt", (("checkpoint", "ckpt-FABRICATED"),)),), dict(axial_payload))
    with __import__("pytest").raises(StageContractError, match="[Tt]oken|[Ll]ocalization"):
        _run_composed((_axial_hypothesis("hyp-bad-tok", (("token", "tok-FABRICATED"),)),), dict(token_payload))
    with __import__("pytest").raises(StageContractError, match="[Ff]eature|[Ll]ocalization"):
        _run_composed(
            (_axial_hypothesis("hyp-bad-feature", (("feature", "feature-FABRICATED"),)),),
            dict(feature_payload),
        )
    with __import__("pytest").raises(StageContractError, match="[Tt]oken|[Ll]ocalization"):
        _run_composed((_axial_hypothesis("hyp-absent", (("token", "tok-2"),)),), dict(axial_payload))
    not_applicable = dict(axial_payload)
    not_applicable["axes"] = [
        {"axis": "checkpoint", "status": "not_applicable", "affected": [], "earliest": None, "report_rows": []}
    ]
    with __import__("pytest").raises(StageContractError, match="[Ll]ocalization"):
        _run_composed((_axial_hypothesis("hyp-na", (("checkpoint", "ckpt-1"),)),), not_applicable)
    # Wrong localize family/metric fail even with a satisfied binding. The
    # family mismatch trips the detect gate first (same shared identities).
    with __import__("pytest").raises(StageContractError, match="detect family"):
        _run_composed(
            (_axial_hypothesis("hyp-wrong-fam", (("checkpoint", "ckpt-1"),), family_id="separability_probe_leakage"),),
            dict(axial_payload),
        )
    # The metric mismatch likewise trips the detect gate first.
    with __import__("pytest").raises(StageContractError, match="detect metric"):
        _run_composed(
            (_axial_hypothesis("hyp-wrong-met", (("checkpoint", "ckpt-1"),), metric_ids=("heldout-probe-accuracy",)),),
            dict(axial_payload),
        )
    # The localize gate itself is covered by a localize-only composition: a
    # relabeled axial payload declaring a different family fails the localize
    # family comparison even though detect linkage passes.
    _relabeled_axial = dict(axial_payload)
    _relabeled_axial["family_id"] = "separability_probe_leakage"
    with __import__("pytest").raises(StageContractError, match="localize family"):
        _run_composed((_axial_hypothesis("hyp-loc-fam", (("checkpoint", "ckpt-1"),)),), _relabeled_axial)
    executor = make_explain_executor(
        (_axial_hypothesis("hyp-unrelated", (("checkpoint", "ckpt-1"),)),),
        MethodInputs(tcav=_tcav_bundle()),
    )
    prior = (
        _AxialStage(stage="capture", outcome="completed", payload={"capture_id": "c"}, artifact_refs=()),
        _AxialStage(stage="detect", outcome="completed", payload=_axial_detect_payload(), artifact_refs=()),
        _AxialStage(stage="localize", outcome="completed", payload=dict(axial_payload), artifact_refs=()),
    )
    invocation = StageInvocation(
        stage="explain",
        request=request_real,
        manifest=manifest_real,
        prior=prior,
        workflow_identity="w" * 64,
        request_digest="r" * 64,
        manifest_digest="m" * 64,
        config_digest="c" * 64,
    )
    produced = executor(invocation)
    (record,) = [item for item in produced.payload["evidence"]]  # type: ignore[index]
    assert record.get("outcome") == "omitted"

    # Payload control keys equal the declaration suffixes.
    hypothesis = _hypothesis("probe")
    (record,), _ = evaluate_explanations((hypothesis,), MethodInputs(probe=_probe_bundle()))
    assert record.outcome == "supported"
    assert sorted(record.control_outcomes) == [
        "control-capacity",
        "control-label-randomization",
        "control-nonseparable-negative",
    ]
    assert sorted(record.central_outcomes) == [
        "control-capacity",
        "control-label-randomization",
        "control-nonseparable-negative",
    ]
    # Missing role declaration blocks before method work.
    bad = dataclasses.replace(
        hypothesis, control_ids=("capacity:control-capacity", "randomized:control-label-randomization")
    )
    (blocked,), _ = evaluate_explanations((bad,), MethodInputs(probe=_probe_bundle()))
    assert blocked.outcome == "inconclusive"
    assert "missing-evidence:control-declaration" in blocked.missing_evidence
    # Wrong role prefix blocks before method work.
    swapped = dataclasses.replace(
        hypothesis,
        control_ids=(
            "randomized:control-capacity",
            "capacity:control-label-randomization",
            "negative:control-nonseparable-negative",
        ),
    )
    (blocked2,), _ = evaluate_explanations((swapped,), MethodInputs(probe=_probe_bundle()))
    assert blocked2.outcome == "inconclusive"
    assert "missing-evidence:control-declaration" in blocked2.missing_evidence


def test_feature_method_rejected_before_ig_callback() -> None:
    calls = {"count": 0}
    hypothesis = _hypothesis("sae_sparse", hypothesis_id="hyp-sae")
    bundle = {"attributions": [1.0, 2.0], "symptom_relationship": "x"}
    with __import__("pytest").raises(ExplanationError, match="not an 80.16 explanation method"):
        evaluate_explanations((hypothesis,), MethodInputs(integrated_gradients=bundle))
    assert calls["count"] == 0


def test_nine_name_surface_and_frozen_artifacts_remain() -> None:
    from pathlib import Path

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
    assert expected <= set(la.__all__)
    assert len([name for name in la.__all__ if name in expected]) == 9
    for leaked in (
        "ExplanationHypothesis",
        "ExplanationEvidence",
        "evaluate_explanations",
        "make_explain_executor",
        "ControlPlan",
        "execute_plan",
        "evaluate_feature_explanations",
        "make_feature_explain_executor",
    ):
        assert not hasattr(la, leaked)
    assert SUPPORTED_METHODS == ("probe", "tcav", "integrated_gradients")
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    for name in (
        "benchmark_manifest_schema_v1.json",
        "diagnostic_report_schema_v1.json",
        "representation_problem_taxonomy_v1.json",
    ):
        assert (artifacts / name).exists()


def test_fail_closed_inputs_reject() -> None:
    with pytest.raises(ExplanationError, match="hypotheses must declare"):
        evaluate_explanations((), MethodInputs(probe=_probe_bundle()))
    with pytest.raises(
        (ExplanationError, TypeError), match="unsupported explanation method|unexpected keyword|missing"
    ):
        ExplanationHypothesis(**{**_hypothesis("probe").to_dict(), "method": "sae", "hypothesis_id": "x"})  # type: ignore[arg-type]
    with pytest.raises(ExplanationError, match="distinct"):
        _hypothesis("probe", train_split_identity="same", eval_split_identity="same")
    with pytest.raises(ExplanationError, match="inputs must be"):
        evaluate_explanations((_hypothesis("probe"),), object())  # type: ignore[arg-type]
    with pytest.raises(ExplanationError, match="context must be"):
        explanation_payload(object())  # type: ignore[arg-type]
    with pytest.raises(ExplanationError, match="confidence_level"):
        evaluate_explanations((_hypothesis("probe"),), MethodInputs(), confidence_level=True)
