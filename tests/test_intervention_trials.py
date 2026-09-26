"""Consumer-observable tests for the Sprint 80.18 intervention-trial executor."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from latent_anything._benchmark_manifest import manifest_digest
from latent_anything._capture_binding import bind_selection
from latent_anything._diagnostic_report import validate_report_shape
from latent_anything._diagnostic_workflow import (
    WORKFLOW_STAGES,
    DiagnosticWorkflow,
    StageContractError,
    StageInvocation,
    StageOutput,
)
from latent_anything._intervention_trials import (
    INTERVENTION_CONTROL_CLASSES,
    INTERVENTION_KINDS,
    INTERVENTION_VERSION,
    InterventionError,
    Measurement,
    TrialApplication,
    TrialControl,
    TrialSpec,
    intervention_report_items,
    make_intervene_executor,
)
from latent_anything._probe_tcav_ig_explanation import (
    ExplanationHypothesis,
    MethodInputs,
    evaluate_explanations,
)
from latent_anything.diagnostics import (
    CaptureSelection,
    ComparisonRequest,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticSelection,
    InterventionRequest,
    OutputSelection,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MANIFEST_ID = "sprint80-core-encoder-autoencoder-collapse-v1"
REPRESENTATION = "conv_vae_8x8:bottleneck-mu:latent_dim=4"
FAMILY = "collapse_rank_loss"
METRIC = "bottleneck-effective-rank"
TASK_METRIC = "heldout-task-utility"
HYPOTHESIS_ID = "h-collapse-probe"
_BASELINE = 2.0


def _manifest() -> dict[str, object]:
    path = ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _control_ids(prefix: str) -> tuple[str, ...]:
    return (f"{prefix}-zero", f"{prefix}-random", f"{prefix}-shuffled", f"{prefix}-off")


def _intervention(intervention_id: str, *, target: str = "bottleneck-feature-mu-dim0") -> InterventionRequest:
    return InterventionRequest(
        intervention_id=intervention_id,
        target=target,
        control_ids=_control_ids(intervention_id),
    )


def _request(
    interventions: tuple[InterventionRequest, ...] | None = None,
    *,
    metric_ids: tuple[str, ...] = (METRIC,),
    comparisons: tuple[ComparisonRequest, ...] = (),
) -> DiagnosticRequest:
    declared = interventions or (_intervention("iv-trial"),)
    control_ids: list[str] = []
    for entry in declared:
        control_ids.extend(entry.control_ids)
    return DiagnosticRequest(
        request_id="request-80-18",
        manifest_id=MANIFEST_ID,
        capture=CaptureSelection(
            capture_id="capture-encoder",
            representation_identity=REPRESENTATION,
            axes=("sample", "feature"),
        ),
        diagnostics=DiagnosticSelection(family_ids=(FAMILY,)),
        controls=ControlSelection(
            control_ids=tuple(dict.fromkeys(control_ids)),
            metric_ids=metric_ids,
        ),
        interventions=declared,
        comparisons=comparisons,
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-18"),
    )


def _controls(prefix: str, *, off_target: str = "bottleneck-feature-mu-dim3") -> tuple[TrialControl, ...]:
    return (
        TrialControl(f"{prefix}-zero", "zero_strength", "identity rerun reproduces the baseline"),
        TrialControl(f"{prefix}-random", "random", "random intervention stays below the effect"),
        TrialControl(f"{prefix}-shuffled", "shuffled", "shuffled assignment stays below the effect"),
        TrialControl(
            f"{prefix}-off",
            "off_target",
            "off-target intervention stays below the effect",
            target=off_target,
        ),
    )


def _trial(
    intervention_id: str = "iv-trial",
    *,
    kind: str = "ablate",
    expected_effect: str = "decrease",
    target: str = "bottleneck-feature-mu-dim0",
    metric_id: str = METRIC,
) -> TrialSpec:
    semantics = {
        "patch": "fraction-of-donor-activation-patched",
        "ablate": "fraction-of-target-activation-zeroed",
        "remove": "fraction-of-target-direction-removed",
    }[kind]
    return TrialSpec(
        intervention_id=intervention_id,
        kind=kind,
        target=target,
        metric_id=metric_id,
        hypothesis_id=HYPOTHESIS_ID,
        strength=1.0,
        strength_semantics=semantics,
        expected_effect=expected_effect,
        controls=_controls(intervention_id),
        provenance={"method": kind, "carrier": "test-measure-seam-v1"},
    )


def _probe_hypothesis() -> ExplanationHypothesis:
    return ExplanationHypothesis(
        hypothesis_id=HYPOTHESIS_ID,
        symptom_id="symptom-collapse",
        family_id=FAMILY,
        target_id="bottleneck-mu",
        representation_id=REPRESENTATION,
        layer_id="bottleneck-mu",
        slice_id="slice-all",
        method="probe",
        expected_direction="higher",
        dataset_id="digits-heldout",
        train_split_identity="split-train-A",
        eval_split_identity="split-eval-A",
        seeds=(7, 8),
        control_ids=(
            "capacity:control-probe-capacity",
            "randomized:control-probe-randomized",
            "negative:control-probe-negative",
        ),
        metric_ids=(METRIC,),
        thresholds=(("heldout_accuracy", ">=", 0.7), ("leakage_gap", ">=", 0.15)),
        manifest_id=MANIFEST_ID,
        localization_bindings=(("slice", "slice-all"),),
    )


def _probe_bundle() -> dict[str, object]:
    rng = np.random.default_rng(0)
    n, dim = 200, 6
    matrix = rng.normal(size=(n, dim))
    matrix[:100] += 1.5
    matrix[100:] -= 1.5
    labels = np.array([0] * 100 + [1] * 100)
    perm = np.random.default_rng(79).permutation(n)
    train = np.array([int(item) for item in perm[:140]])
    evaluation = np.array([int(item) for item in perm[140:]])
    neg = rng.normal(size=(n, dim))
    from latent_anything.probes import _fast_probe

    negative = float(_fast_probe(neg[train], labels[train], neg[evaluation], labels[evaluation], 7).accuracy)
    train_set = set(int(i) for i in train)
    eval_set = set(int(i) for i in evaluation)
    return {
        "train_matrix": matrix[train],
        "eval_matrix": matrix[evaluation],
        "train_labels": labels[train],
        "eval_labels": labels[evaluation],
        "train_ids": [f"sample-{i}" for i in range(n) if i in train_set],
        "eval_ids": [f"sample-{i}" for i in range(n) if i in eval_set],
        "capacity": "linear-logreg-C1.0-standardized",
        "negative_accuracy": negative,
    }


_EXPLAIN_CACHE: dict[str, dict[str, object]] = {}


def _explain_payload() -> dict[str, object]:
    """One real explain-stage payload from the 80.16 evaluator, computed once."""
    if "payload" not in _EXPLAIN_CACHE:
        _, payload = evaluate_explanations((_probe_hypothesis(),), MethodInputs(probe=_probe_bundle()))
        _EXPLAIN_CACHE["payload"] = payload
    return copy.deepcopy(_EXPLAIN_CACHE["payload"])


def _capture_payload(request: DiagnosticRequest, manifest: dict[str, object]) -> dict[str, object]:
    bound = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    return {"capture": bound.provenance()}


def _localize_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "family_id": FAMILY,
        "manifest_id": MANIFEST_ID,
        "metric_id": METRIC,
        "representation_identity": REPRESENTATION,
        "verdict": "localized",
    }
    payload.update(overrides)
    return payload


def _prior(
    request: DiagnosticRequest,
    manifest: dict[str, object],
    *,
    capture_payload: dict[str, object] | None = None,
    localize_payload: dict[str, object] | None = None,
    explain: dict[str, object] | None = None,
) -> tuple[StageOutput, ...]:
    return (
        StageOutput(
            stage="capture",
            outcome="completed",
            payload=capture_payload if capture_payload is not None else _capture_payload(request, manifest),
            artifact_refs=(),
        ),
        StageOutput(stage="detect", outcome="completed", payload={"stage": "detect"}, artifact_refs=()),
        StageOutput(
            stage="localize",
            outcome="completed",
            payload=localize_payload if localize_payload is not None else _localize_payload(),
            artifact_refs=(),
        ),
        StageOutput(
            stage="explain",
            outcome="completed",
            payload=explain if explain is not None else _explain_payload(),
            artifact_refs=(),
        ),
    )


def _invocation(
    request: DiagnosticRequest,
    manifest: dict[str, object],
    prior: tuple[StageOutput, ...],
    *,
    stage: str = "intervene",
    prior_override: tuple[StageOutput, ...] | None = None,
) -> StageInvocation:
    return StageInvocation(
        stage=stage,
        request=request,
        manifest=manifest,
        prior=prior_override if prior_override is not None else prior,
        workflow_identity="a" * 64,
        request_digest="b" * 64,
        manifest_digest="c" * 64,
        config_digest="d" * 64,
    )


def _measure_factory(
    values: dict[str, float],
    *,
    record: list[TrialApplication] | None = None,
    draw: Callable[[TrialApplication], float] | None = None,
) -> Callable[[TrialApplication], Measurement]:
    def _measure(app: TrialApplication) -> Measurement:
        if record is not None:
            record.append(app)
        slot = (
            "baseline"
            if app.role == "baseline"
            else "intervened"
            if app.role == "intervened"
            else (app.control_class or "control")
        )
        if app.role == "intervened" and app.rng is not None and draw is not None:
            value = float(draw(app))
        else:
            specific = f"{app.intervention_id}:{slot}"
            value = float(values.get(specific, values.get(slot, _BASELINE)))
        return Measurement(app.metric_id, value, app.inputs_digest)

    return _measure


def _run(
    trials: tuple[TrialSpec, ...],
    measure: Callable[[TrialApplication], Measurement],
    *,
    request: DiagnosticRequest | None = None,
    manifest: dict[str, object] | None = None,
) -> StageOutput:
    manifest = manifest if manifest is not None else _manifest()
    request = (
        request if request is not None else _request(tuple(_intervention(spec.intervention_id) for spec in trials))
    )
    executor = make_intervene_executor(trials, measure)
    return executor(_invocation(request, manifest, _prior(request, manifest)))


def test_positive_patch_ablate_remove_cases_share_captured_inputs_and_metric() -> None:
    trials = (
        _trial("iv-patch", kind="patch", expected_effect="increase"),
        _trial("iv-ablate", kind="ablate", expected_effect="decrease"),
        _trial("iv-remove", kind="remove", expected_effect="decrease"),
    )
    values = {
        "iv-patch:intervened": 3.0,
        "iv-ablate:intervened": 1.0,
        "iv-remove:intervened": 1.2,
    }
    record: list[TrialApplication] = []
    request = _request(tuple(_intervention(spec.intervention_id) for spec in trials))
    manifest = _manifest()
    output = _run(trials, _measure_factory(values, record=record), request=request, manifest=manifest)

    assert output.stage == "intervene"
    assert output.outcome == "completed"
    assert output.payload["intervention_version"] == INTERVENTION_VERSION
    conclusions = dict(output.payload["conclusions"])  # type: ignore[arg-type]
    assert conclusions == {"iv-patch": "supported", "iv-ablate": "supported", "iv-remove": "supported"}
    trials_payload = list(output.payload["trials"])  # type: ignore[arg-type]
    assert [row["kind"] for row in trials_payload] == list(INTERVENTION_KINDS)

    bound = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    # Every callback across all three trials saw the exact same captured
    # inputs digest and the exact same downstream metric.
    assert {app.inputs_digest for app in record} == {bound.capture_identity}
    assert {app.metric_id for app in record} == {METRIC}
    for intervention_id in ("iv-patch", "iv-ablate", "iv-remove"):
        per_trial = [app for app in record if app.intervention_id == intervention_id]
        # 1 baseline + 1 intervened point + 4 controls + 200 uncertainty draws.
        assert len(per_trial) == 206
        assert sum(1 for app in per_trial if app.role == "baseline") == 1
        assert sum(1 for app in per_trial if app.role == "intervened" and app.rng is None) == 1
        assert sum(1 for app in per_trial if app.role == "intervened" and app.rng is not None) == 200
        controls = [app for app in per_trial if app.role == "control"]
        assert len(controls) == 4
        assert {app.control_class for app in controls} == set(INTERVENTION_CONTROL_CLASSES)
        assert all(app.rng is not None for app in controls)

    for row in trials_payload:
        assert row["conclusion"] == "supported"
        assert row["inputs_digest"] == bound.capture_identity
        assert row["metric_id"] == METRIC
        assert set(row["control_outcomes"]) == set(_control_ids(str(row["intervention_id"])))
        outcomes = row["control_outcomes"]
        assert all(item["status"] == "passed" for item in outcomes.values())  # type: ignore[union-attr]
        assert row["uncertainty"]["repetitions"] == 200  # type: ignore[index]
        assert row["uncertainty"]["seed"] == row["provenance"]["seeds"]["uncertainty_stream"]  # type: ignore[index]
        provenance = row["provenance"]
        assert provenance["method"] in INTERVENTION_KINDS  # type: ignore[index]
        assert provenance["carrier"] == "test-measure-seam-v1"  # type: ignore[index]
        assert provenance["capture_identity"] == bound.capture_identity  # type: ignore[index]
        assert provenance["localize_family_id"] == FAMILY  # type: ignore[index]
        assert row["strength_semantics"]
        assert row["strength"] == 1.0
        assert set(row["control_thresholds"]) == set(_control_ids(str(row["intervention_id"])))

    items = intervention_report_items(list(output.payload["trials"]))  # type: ignore[arg-type]
    assert [item["status"] for item in items] == ["supported"] * 3
    assert all(item["kind"] == "causal_result" and item["causal"] is True for item in items)
    assert all(item["claim_allowed"] is True for item in items)


def test_failed_controls_block_causal_promotion() -> None:
    zero_fail = _run(
        (_trial("iv-zero-fail"),),
        _measure_factory({"intervened": 3.0, "zero_strength": 2.3}),
    )
    zero_conclusions = dict(zero_fail.payload["conclusions"])  # type: ignore[arg-type]
    assert zero_conclusions == {"iv-zero-fail": "unsupported"}
    zero_items = intervention_report_items(list(zero_fail.payload["trials"]))  # type: ignore[arg-type]
    assert zero_items[0]["status"] == "unsupported"
    assert zero_items[0]["claim_allowed"] is False
    assert zero_items[0]["missing_evidence"]

    specificity = _run(
        (_trial("iv-random-reproduces"),),
        _measure_factory({"intervened": 3.0, "random": 3.0}),
    )
    conclusions = dict(specificity.payload["conclusions"])  # type: ignore[arg-type]
    assert conclusions == {"iv-random-reproduces": "falsified"}
    first = list(specificity.payload["trials"])[0]  # type: ignore[arg-type]
    outcomes = first["control_outcomes"]
    assert outcomes["iv-random-reproduces-random"]["status"] == "failed"  # type: ignore[index]
    items = intervention_report_items([first])
    assert items[0]["status"] == "falsified"
    assert "target specificity is falsified" in str(items[0]["claim"])

    all_unsupported = _run(
        (_trial("iv-blocked"),),
        _measure_factory({"intervened": 3.0, "zero_strength": 3.4}),
    )
    assert all_unsupported.outcome == "unsupported"


def test_inert_and_wrong_direction_interventions_are_falsified() -> None:
    inert = _run(
        (_trial("iv-inert"),),
        _measure_factory({"intervened": 2.05, "random": 2.01}),
    )
    inert_payload = list(inert.payload["trials"])[0]  # type: ignore[arg-type]
    assert inert_payload["conclusion"] == "falsified"
    assert "inert beyond noise" in str(inert_payload["reason"])
    assert inert.outcome == "completed"

    wrong_direction = _run(
        (_trial("iv-backwards", kind="ablate", expected_effect="decrease"),),
        _measure_factory({"intervened": 3.0}),
    )
    payload = list(wrong_direction.payload["trials"])[0]  # type: ignore[arg-type]
    assert payload["conclusion"] == "falsified"
    assert "opposite the declared decrease direction" in str(payload["reason"])


def test_uncertainty_spanning_baseline_is_inconclusive() -> None:
    def _draw(app: TrialApplication) -> float:
        assert app.intervention_id == "iv-uncertain"
        assert app.rng is not None
        return 3.0 + float(app.rng.normal(0.0, 0.6))

    output = _run(
        (_trial("iv-uncertain", kind="patch", expected_effect="increase"),),
        _measure_factory({"intervened": 3.0}, draw=_draw),
    )
    payload = list(output.payload["trials"])[0]  # type: ignore[arg-type]
    assert payload["conclusion"] == "inconclusive"
    assert "spans the baseline" in str(payload["reason"])
    items = intervention_report_items([payload])
    assert items[0]["claim_allowed"] is False
    assert items[0]["missing_evidence"]


def test_all_conclusions_recorded_in_one_workflow_run_with_truthful_report() -> None:
    ids = ("iv-patch", "iv-ablate", "iv-remove", "iv-blocked", "iv-uncertain")
    interventions = tuple(_intervention(intervention_id) for intervention_id in ids)
    trials = (
        _trial("iv-patch", kind="patch", expected_effect="increase"),
        _trial("iv-ablate", kind="ablate", expected_effect="decrease"),
        _trial("iv-remove", kind="remove", expected_effect="decrease"),
        _trial("iv-blocked", kind="patch", expected_effect="increase"),
        _trial("iv-uncertain", kind="patch", expected_effect="increase"),
    )
    values = {
        "iv-patch:intervened": 3.0,
        "iv-ablate:intervened": 1.0,
        "iv-remove:intervened": 2.05,
        "iv-blocked:intervened": 3.0,
        "iv-blocked:zero_strength": 2.3,
        "iv-uncertain:intervened": 3.0,
        "random": 2.01,
        "shuffled": 2.0,
        "off_target": 2.0,
    }

    def _draw(app: TrialApplication) -> float:
        assert app.rng is not None
        if app.intervention_id == "iv-uncertain":
            return 3.0 + float(app.rng.normal(0.0, 0.6))
        return float(values.get(f"{app.intervention_id}:intervened", 1.0))

    request = _request(interventions)
    manifest = _manifest()
    prior = _prior(request, manifest)
    payload_by_stage = {output.stage: dict(output.payload) for output in prior}
    executor = make_intervene_executor(trials, _measure_factory(values, draw=_draw))

    def _stage(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "intervene":
            return executor(invocation)
        payload = dict(payload_by_stage.get(invocation.stage, {"stage": invocation.stage}))
        if invocation.stage == "report":
            payload["report_id"] = "report-80-18"
        return StageOutput(stage=invocation.stage, outcome="completed", payload=payload, artifact_refs=())

    executors = {stage: _stage for stage in WORKFLOW_STAGES}
    versions = {stage: "stage-v1" for stage in WORKFLOW_STAGES}
    workflow = DiagnosticWorkflow(executors, versions)  # type: ignore[arg-type]
    result, checkpoint = workflow.run(request, manifest)

    assert result.status == "completed"
    assert checkpoint.status == "completed"
    intervene_stage = result.stage_results["stages"]["intervene"]  # type: ignore[index]
    assert intervene_stage["outcome"] == "completed"  # type: ignore[index]
    assert intervene_stage["payload"]["conclusions"] == {  # type: ignore[index]
        "iv-patch": "supported",
        "iv-ablate": "supported",
        "iv-remove": "falsified",
        "iv-blocked": "unsupported",
        "iv-uncertain": "inconclusive",
    }

    trials_payload = list(intervene_stage["payload"]["trials"])  # type: ignore[index]
    items = intervention_report_items(trials_payload)
    statuses = {item["id"]: item["status"] for item in items}
    assert statuses == {
        "intervention-iv-patch": "supported",
        "intervention-iv-ablate": "supported",
        "intervention-iv-remove": "falsified",
        "intervention-iv-blocked": "unsupported",
        "intervention-iv-uncertain": "inconclusive",
    }
    allowed = {item["id"]: item["claim_allowed"] for item in items}
    assert allowed["intervention-iv-patch"] is True
    assert allowed["intervention-iv-remove"] is True
    assert allowed["intervention-iv-blocked"] is False
    assert allowed["intervention-iv-uncertain"] is False

    validate_report_shape(_report(items))


def _report(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "diagnostic-report-schema-v1",
        "report_id": "report-80-18",
        "status": "declared",
        "capture_provenance": {
            "captures": [
                {
                    "artifact_refs": ["cap-1"],
                    "axes": ["sample", "feature"],
                    "capture_id": "capture-encoder",
                    "dataset_split": "heldout",
                    "model_revision": "conv-vae-8x8-latent4-seed0-epochs5",
                    "representation_identity": REPRESENTATION,
                }
            ]
        },
        "symptoms": [
            {
                "description": "bottleneck rank collapse",
                "evidence_refs": ["o-1"],
                "id": "s-1",
                "metric_ids": [METRIC],
                "status": "observed",
            }
        ],
        "localization": [
            {
                "axis": "slice",
                "confidence": 0.9,
                "evidence_refs": ["o-1"],
                "id": "l-1",
                "selection": "slice-all",
                "status": "observed",
            }
        ],
        "hypotheses": [
            {
                "alternatives": ["benign-low-variance"],
                "evidence_refs": ["o-1"],
                "id": HYPOTHESIS_ID,
                "statement": "the collapsed bottleneck direction causes the defect",
                "status": "supported",
            }
        ],
        "statistical_evidence": [
            {
                "control_refs": [],
                "estimate": 0.5,
                "evidence_refs": ["o-1"],
                "id": "e-1",
                "metric_id": METRIC,
                "status": "observed",
                "uncertainty": {"kind": "interval", "lower": 0.4, "upper": 0.6},
            }
        ],
        "interventions": [
            {
                "control_refs": list(item["control_refs"]),  # type: ignore[arg-type]
                "evidence_refs": list(item["evidence_refs"]),  # type: ignore[arg-type]
                "id": str(item["id"]).removeprefix("intervention-"),
                "intervention": str(item["id"]),
                "outcome": {"conclusion": str(item["status"])},
                "status": str(item["status"]),
                "target": "bottleneck-feature-mu-dim0",
            }
            for item in items
        ],
        "comparisons": [],
        "limitations": [
            {
                "affects": ["causal"],
                "blocking": False,
                "description": "intervention carriers are caller-supplied in 80.18",
                "id": "lim-1",
            }
        ],
        "next_action": {
            "action": "run dose-response trials",
            "rationale": "80.19 owns strength sweeps",
            "required_evidence_refs": [],
        },
        "claims": [
            {
                "causal": False,
                "claim": "the bottleneck shows a rank-collapse symptom",
                "claim_allowed": True,
                "control_refs": [],
                "evidence_refs": ["cap-1"],
                "id": "o-1",
                "kind": "observation",
                "missing_evidence": [],
                "status": "observed",
            },
            *items,
        ],
    }


def test_invalid_prior_and_stage_identity_rejects_before_callbacks() -> None:
    manifest = _manifest()
    request = _request()
    prior = _prior(request, manifest)
    record: list[TrialApplication] = []
    executor = make_intervene_executor((_trial(),), _measure_factory({}, record=record))

    capture = _capture_payload(request, manifest)
    capture_without_identity = {
        "capture": {
            key: value
            for key, value in dict(capture["capture"]).items()  # type: ignore[arg-type]
            if key != "capture_identity"
        }
    }
    capture_representation = _capture_payload(request, manifest)
    capture_inner = capture_representation["capture"]
    assert isinstance(capture_inner, dict)
    capture_inner["representation_identity"] = "other:representation"

    omitted_explain = _explain_payload()
    family_evidence = omitted_explain["family_evidence"]
    assert isinstance(family_evidence, dict)
    evidence_row = family_evidence[HYPOTHESIS_ID]
    assert isinstance(evidence_row, dict)
    evidence_row["outcome"] = "omitted"
    no_row_explain = _explain_payload()
    no_row_explain["hypotheses"] = []
    no_evidence_explain = _explain_payload()
    no_evidence_explain.pop("family_evidence")

    for tampered in (
        _prior(request, manifest, capture_payload=capture_without_identity),
        _prior(request, manifest, capture_payload={"stage": "capture"}),
        _prior(request, manifest, capture_payload=capture_representation),
        _prior(request, manifest, localize_payload=_localize_payload(metric_id="bottleneck-singular-spread")),
        _prior(request, manifest, localize_payload=_localize_payload(manifest_id="other-manifest")),
        _prior(request, manifest, explain=omitted_explain),
        _prior(request, manifest, explain=no_row_explain),
        _prior(request, manifest, explain=no_evidence_explain),
    ):
        with pytest.raises(StageContractError):
            executor(_invocation(request, manifest, tampered))

    inapplicable = copy.deepcopy(manifest)
    causal = inapplicable["causal_expectation"]
    assert isinstance(causal, dict)
    causal["applicable"] = False
    with pytest.raises(StageContractError, match="not applicable"):
        executor(_invocation(request, inapplicable, prior))

    mismatched_request = _request(
        (
            InterventionRequest(
                intervention_id="iv-trial",
                target="bottleneck-feature-mu-dim9",
                control_ids=_control_ids("iv-trial"),
            ),
        )
    )
    with pytest.raises(StageContractError, match="target"):
        executor(_invocation(mismatched_request, manifest, _prior(mismatched_request, manifest)))

    missing_control_request = _request(
        (
            InterventionRequest(
                intervention_id="iv-trial",
                target="bottleneck-feature-mu-dim0",
                control_ids=("iv-trial-zero",),
            ),
        )
    )
    with pytest.raises(StageContractError, match="controls"):
        executor(_invocation(missing_control_request, manifest, _prior(missing_control_request, manifest)))

    extra = _request((_intervention("iv-trial"), _intervention("iv-undeclared")))
    with pytest.raises(StageContractError, match="match exactly"):
        executor(_invocation(extra, manifest, _prior(extra, manifest)))

    off_metric_request = _request(metric_ids=(METRIC, "bottleneck-singular-spread"))
    off_metric_executor = make_intervene_executor(
        (_trial(metric_id="bottleneck-singular-spread"),), _measure_factory({}, record=record)
    )
    with pytest.raises(StageContractError, match="causal-expectation target"):
        off_metric_executor(_invocation(off_metric_request, manifest, _prior(off_metric_request, manifest)))

    assert record == []


def test_wrong_stage_rejects_before_callbacks() -> None:
    manifest = _manifest()
    request = _request()
    prior = _prior(request, manifest)
    record: list[TrialApplication] = []
    executor = make_intervene_executor((_trial(),), _measure_factory({}, record=record))
    compare_prior = prior + (StageOutput(stage="intervene", outcome="completed", payload={}, artifact_refs=()),)
    with pytest.raises(StageContractError, match="received stage"):
        executor(_invocation(request, manifest, prior, stage="compare", prior_override=compare_prior))
    assert record == []


def test_trial_declaration_validation_fails_closed() -> None:
    with pytest.raises(InterventionError, match="kind"):
        TrialSpec(
            intervention_id="iv-x",
            kind="steer",
            target="bottleneck-feature-mu-dim0",
            metric_id=METRIC,
            hypothesis_id=HYPOTHESIS_ID,
            strength=1.0,
            strength_semantics="steer-along-direction",
            expected_effect="increase",
            controls=_controls("iv-x"),
            provenance={"method": "steer", "carrier": "seam"},
        )
    with pytest.raises(InterventionError, match="expected_effect"):
        TrialSpec(
            intervention_id="iv-x",
            kind="ablate",
            target="bottleneck-feature-mu-dim0",
            metric_id=METRIC,
            hypothesis_id=HYPOTHESIS_ID,
            strength=1.0,
            strength_semantics="fraction-of-target-activation-zeroed",
            expected_effect="sideways",
            controls=_controls("iv-x"),
            provenance={"method": "ablate", "carrier": "seam"},
        )
    with pytest.raises(InterventionError, match="finite"):
        TrialSpec(
            intervention_id="iv-x",
            kind="ablate",
            target="bottleneck-feature-mu-dim0",
            metric_id=METRIC,
            hypothesis_id=HYPOTHESIS_ID,
            strength=float("nan"),
            strength_semantics="fraction-of-target-activation-zeroed",
            expected_effect="decrease",
            controls=_controls("iv-x"),
            provenance={"method": "ablate", "carrier": "seam"},
        )
    with pytest.raises(InterventionError, match="each required class"):
        TrialSpec(
            intervention_id="iv-x",
            kind="ablate",
            target="bottleneck-feature-mu-dim0",
            metric_id=METRIC,
            hypothesis_id=HYPOTHESIS_ID,
            strength=1.0,
            strength_semantics="fraction-of-target-activation-zeroed",
            expected_effect="decrease",
            controls=_controls("iv-x")[:3],
            provenance={"method": "ablate", "carrier": "seam"},
        )
    with pytest.raises(InterventionError, match="different target"):
        TrialSpec(
            intervention_id="iv-x",
            kind="ablate",
            target="bottleneck-feature-mu-dim0",
            metric_id=METRIC,
            hypothesis_id=HYPOTHESIS_ID,
            strength=1.0,
            strength_semantics="fraction-of-target-activation-zeroed",
            expected_effect="decrease",
            controls=_controls("iv-x", off_target="bottleneck-feature-mu-dim0"),
            provenance={"method": "ablate", "carrier": "seam"},
        )
    with pytest.raises(InterventionError, match="carrier"):
        TrialSpec(
            intervention_id="iv-x",
            kind="ablate",
            target="bottleneck-feature-mu-dim0",
            metric_id=METRIC,
            hypothesis_id=HYPOTHESIS_ID,
            strength=1.0,
            strength_semantics="fraction-of-target-activation-zeroed",
            expected_effect="decrease",
            controls=_controls("iv-x"),
            provenance={"method": "ablate"},
        )
    with pytest.raises(InterventionError, match="at least one trial"):
        make_intervene_executor((), _measure_factory({}))
    with pytest.raises(InterventionError, match="unique"):
        make_intervene_executor((_trial(), _trial()), _measure_factory({}))


def test_measurement_contract_violations_fail_closed() -> None:
    manifest = _manifest()
    request = _request()
    prior = _prior(request, manifest)
    invocation = _invocation(request, manifest, prior)

    wrong_metric = make_intervene_executor(
        (_trial(),), lambda _app: Measurement("bottleneck-singular-spread", 1.0, _app.inputs_digest)
    )
    with pytest.raises(InterventionError, match="domain mismatch"):
        wrong_metric(invocation)

    wrong_digest = make_intervene_executor((_trial(),), lambda _app: Measurement(_app.metric_id, 1.0, "f" * 64))
    with pytest.raises(InterventionError, match="inputs mismatch"):
        wrong_digest(invocation)

    non_finite = make_intervene_executor(
        (_trial(),), lambda _app: Measurement(_app.metric_id, float("inf"), _app.inputs_digest)
    )
    with pytest.raises(InterventionError, match="finite"):
        non_finite(invocation)

    not_measurement = make_intervene_executor(
        (_trial(),),
        lambda _app: {"value": 1.0},  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(InterventionError, match="must return a Measurement"):
        not_measurement(invocation)

    crashing = make_intervene_executor((_trial(),), _boom)
    with pytest.raises(RuntimeError, match="model exploded"):
        crashing(invocation)

    control_path_violation = make_intervene_executor((_trial(),), _control_path_violation)
    with pytest.raises(InterventionError, match="measurement failed"):
        control_path_violation(invocation)


def _boom(_app: TrialApplication) -> Measurement:
    raise RuntimeError("model exploded")


def _control_path_violation(_app: TrialApplication) -> Measurement:
    if _app.role == "control":
        raise InterventionError("broken control harness")
    return Measurement(_app.metric_id, 3.0 if _app.role == "intervened" else _BASELINE, _app.inputs_digest)


def test_deterministic_replay_reproduces_identical_payload_and_digest() -> None:
    manifest = _manifest()
    request = _request((_intervention("iv-patch"),))
    trials = (_trial("iv-patch", kind="patch", expected_effect="increase"),)
    values = {"intervened": 3.0}

    first = _run(trials, _measure_factory(values), request=request, manifest=manifest)
    second = _run(trials, _measure_factory(values), request=request, manifest=manifest)

    assert dict(first.payload) == dict(second.payload)  # type: ignore[arg-type]
    workflow_identity = "a" * 64
    assert first.digest(workflow_identity) == second.digest(workflow_identity)
    assert first.digest(workflow_identity) != first.digest("e" * 64)

    config = dict(first.payload["config"])  # type: ignore[arg-type]
    assert config["evaluation_seed"] == 42
    assert config["control_seed"] == 17
    assert config["repetitions"] == 200
    assert config["workflow_identity"] == workflow_identity
    causal = dict(config["causal_expectation"])  # type: ignore[arg-type]
    assert causal["applicable"] is True
    assert causal["target_metric_ids"] == [METRIC]
    assert causal["falsification_rule"]


def test_comparison_only_task_metric_can_drive_a_causal_intervention() -> None:
    manifest = _manifest()
    task_metric = TASK_METRIC
    metrics = manifest["metrics"]
    thresholds = manifest["thresholds"]
    assert isinstance(metrics, list)
    assert isinstance(thresholds, list)
    metrics.append(
        {
            "aggregation": "mean-with-interval",
            "direction": "higher_is_better",
            "estimator": "predeclared-heldout-task-accuracy",
            "id": task_metric,
            "taxonomy_family_id": FAMILY,
            "unit": "accuracy",
        }
    )
    thresholds.append(
        {
            "comparator": ">=",
            "metric_id": task_metric,
            "predeclared": True,
            "tolerance": 0.02,
            "value": 0.9,
        }
    )
    causal = manifest["causal_expectation"]
    assert isinstance(causal, dict)
    causal["target_metric_ids"] = [task_metric]
    commitment = manifest["commitment"]
    assert isinstance(commitment, dict)
    commitment["manifest_sha256"] = manifest_digest(manifest)

    comparison = ComparisonRequest(
        comparison_id="cmp-task-utility",
        baseline_run="healthy",
        candidate_run="lesioned",
        metric_ids=(METRIC, task_metric),
    )
    request = _request(comparisons=(comparison,))
    trial = _trial("iv-trial", kind="patch", expected_effect="increase", metric_id=task_metric)

    def measure(application: TrialApplication) -> Measurement:
        value = 0.8 if application.role == "intervened" else 0.5
        return Measurement(application.metric_id, value, application.inputs_digest)

    output = _run((trial,), measure, request=request, manifest=manifest)
    records = output.payload["trials"]  # type: ignore[index]
    record = records[0]  # type: ignore[index]

    assert record["metric_id"] == task_metric  # type: ignore[index]
    assert record["conclusion"] == "supported"  # type: ignore[index]
