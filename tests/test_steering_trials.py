"""Consumer-observable tests for the Sprint 80.19 steering dose-response executor."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

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
    STEERING_CONTROL_CLASSES,
    STEERING_KIND,
    InterventionError,
    Measurement,
    SteeringTrialSpec,
    TrialApplication,
    TrialControl,
    intervention_report_items,
    make_steering_intervene_executor,
)
from latent_anything._probe_tcav_ig_explanation import (
    ExplanationHypothesis,
    MethodInputs,
    evaluate_explanations,
)
from latent_anything.diagnostics import (
    CaptureSelection,
    ControlSelection,
    DiagnosticRequest,
    DiagnosticSelection,
    InterventionRequest,
    OutputSelection,
)
from latent_anything.methods.steering import SteeringVector

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MANIFEST_ID = "sprint80-core-encoder-autoencoder-collapse-v1"
REPRESENTATION = "conv_vae_8x8:bottleneck-mu:latent_dim=4"
FAMILY = "collapse_rank_loss"
METRIC = "bottleneck-effective-rank"
HYPOTHESIS_ID = "h-collapse-probe"
ON_TARGET = "slice-steer-target"
OFF_TARGET = "slice-steer-off"
_BASELINE = 2.0


def _manifest() -> dict[str, object]:
    path = ARTIFACTS / "benchmark_manifest_sprint80_encoder_autoencoder_v1.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _control_ids(prefix: str) -> tuple[str, ...]:
    return (f"{prefix}-zero", f"{prefix}-random", f"{prefix}-off")


def _controls(prefix: str, *, off_target: str = OFF_TARGET) -> tuple[TrialControl, ...]:
    return (
        TrialControl(f"{prefix}-zero", "zero_strength", "identity rerun reproduces the baseline"),
        TrialControl(
            f"{prefix}-random",
            "random_direction",
            "rng-drawn direction at the endpoint dose does not reproduce the declared response",
        ),
        TrialControl(
            f"{prefix}-off",
            "off_target",
            "steering the off-target selection stays below the on-target response",
            target=off_target,
        ),
    )


def _spec(
    intervention_id: str = "iv-steer",
    *,
    strengths: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    expected_effect: str = "decrease",
    target: str = ON_TARGET,
    metric_id: str = METRIC,
    direction: np.ndarray | None = None,
    off_target: str = OFF_TARGET,
    provenance: dict[str, str] | None = None,
) -> SteeringTrialSpec:
    return SteeringTrialSpec(
        intervention_id=intervention_id,
        target=target,
        metric_id=metric_id,
        hypothesis_id=HYPOTHESIS_ID,
        strengths=strengths,
        strength_semantics="additive-latent-offset-along-unit-direction",
        expected_effect=expected_effect,
        direction=np.array([1.0, 0.0, 0.0, 0.0]) if direction is None else direction,
        provenance=provenance
        or {
            "method": "steering-vector",
            "carrier": "test-steer-seam-v1",
            "source": "prototype-contrast-e0",
        },
        controls=_controls(intervention_id, off_target=off_target),
    )


def _intervention(spec: SteeringTrialSpec) -> InterventionRequest:
    return InterventionRequest(
        intervention_id=spec.intervention_id,
        target=spec.target,
        control_ids=_control_ids(spec.intervention_id),
    )


def _request(
    interventions: tuple[InterventionRequest, ...] | None = None,
    *,
    metric_ids: tuple[str, ...] = (METRIC,),
) -> DiagnosticRequest:
    declared = interventions or (_intervention(_spec()),)
    control_ids: list[str] = []
    for entry in declared:
        control_ids.extend(entry.control_ids)
    return DiagnosticRequest(
        request_id="request-80-19",
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
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-19"),
    )


def _probe_hypothesis() -> ExplanationHypothesis:
    return ExplanationHypothesis(
        hypothesis_id=HYPOTHESIS_ID,
        symptom_id="symptom-collapse",
        family_id=FAMILY,
        target_id="bottleneck-mu",
        representation_id=REPRESENTATION,
        layer_id="bottleneck-mu",
        slice_id=ON_TARGET,
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
        localization_bindings=(("slice", ON_TARGET),),
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
    if "payload" not in _EXPLAIN_CACHE:
        _, payload = evaluate_explanations((_probe_hypothesis(),), MethodInputs(probe=_probe_bundle()))
        _EXPLAIN_CACHE["payload"] = payload
    return copy.deepcopy(_EXPLAIN_CACHE["payload"])


def _capture_payload(request: DiagnosticRequest, manifest: dict[str, object]) -> dict[str, object]:
    bound = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    return {"capture": bound.provenance()}


def _localize_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "affected_layers": [],
        "affected_slices": [ON_TARGET],
        "declared_slice_ids": [ON_TARGET, OFF_TARGET],
        "earliest_layer": None,
        "family_id": FAMILY,
        "layer_order": [],
        "manifest_id": MANIFEST_ID,
        "metric_id": METRIC,
        "report_localization": [],
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


_Curve = dict[float, float]


def _steer_measure_factory(
    curves: dict[str, _Curve],
    control_values: dict[str, dict[str, float]],
    *,
    record: list[TrialApplication] | None = None,
    draw: Callable[[TrialApplication], float] | None = None,
) -> Callable[[TrialApplication], Measurement]:
    def _measure(app: TrialApplication) -> Measurement:
        if record is not None:
            record.append(app)
        curve = curves[app.intervention_id]
        if app.role == "baseline":
            value = float(curve[0.0])
        elif app.role == "intervened":
            value = float(draw(app)) if app.rng is not None and draw is not None else float(curve[float(app.strength)])
        else:
            value = float(control_values.get(app.intervention_id, {}).get(app.control_class or "", float(curve[0.0])))
        return Measurement(app.metric_id, value, app.inputs_digest)

    return _measure


_DEFAULT_CONTROLS = {
    "zero_strength": _BASELINE,
    "random_direction": 3.0,  # opposite the declared decrease: recorded, never hidden
    "off_target": 1.9,
}


def _run(
    specs: tuple[SteeringTrialSpec, ...],
    measure: Callable[[TrialApplication], Measurement],
    *,
    request: DiagnosticRequest | None = None,
    manifest: dict[str, object] | None = None,
) -> StageOutput:
    manifest = manifest if manifest is not None else _manifest()
    request = request if request is not None else _request(tuple(_intervention(spec) for spec in specs))
    executor = make_steering_intervene_executor(specs, measure)
    return executor(_invocation(request, manifest, _prior(request, manifest)))


def test_supported_dose_response_records_strengths_directions_and_controls() -> None:
    spec = _spec()
    record: list[TrialApplication] = []
    curves = {"iv-steer": {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0}}
    request = _request((_intervention(spec),))
    manifest = _manifest()
    output = _run(
        (spec,),
        _steer_measure_factory(curves, {"iv-steer": dict(_DEFAULT_CONTROLS)}, record=record),
        request=request,
        manifest=manifest,
    )

    assert output.stage == "intervene"
    assert output.outcome == "completed"
    conclusions = dict(output.payload["conclusions"])  # type: ignore[arg-type]
    assert conclusions == {"iv-steer": "supported"}
    row = list(output.payload["trials"])[0]  # type: ignore[arg-type]
    assert row["kind"] == STEERING_KIND
    assert row["response_classification"] == "monotonic_decrease"
    assert row["strengths"] == [0.0, 0.5, 1.0, 2.0]
    assert row["endpoint_strength"] == 2.0
    assert row["effect"] == pytest.approx(-1.0)
    assert row["task_metric_change"] == pytest.approx(-1.0)
    assert row["strength_semantics"] == "additive-latent-offset-along-unit-direction"
    assert row["expected_effect"] == "decrease"
    assert row["conclusion"] == "supported"

    bound = bind_selection(request.capture, manifest=manifest, request_id=request.request_id)
    doses = list(row["doses"])
    assert [dose["strength"] for dose in doses] == [0.5, 1.0, 2.0]
    for dose in doses:
        assert dose["measurement"]["inputs_digest"] == bound.capture_identity
        assert dose["measurement"]["metric_id"] == METRIC
        assert dose["uncertainty"]["repetitions"] == 200  # type: ignore[index]
        assert isinstance(dose["effect"], float)
    assert row["measurements"]["baseline"]["value"] == pytest.approx(_BASELINE)  # type: ignore[index]
    assert row["uncertainty"]["repetitions"] == 200  # type: ignore[index]
    assert row["uncertainty"]["seed"] == row["doses"][-1]["uncertainty"]["seed"]  # type: ignore[index]

    direction = row["direction"]
    assert isinstance(direction["digest"], str) and len(direction["digest"]) == 64  # type: ignore[index]
    assert direction["norm"] == pytest.approx(1.0)  # type: ignore[index]
    assert direction["provenance"] == {  # type: ignore[index]
        "method": "steering-vector",
        "carrier": "test-steer-seam-v1",
        "source": "prototype-contrast-e0",
    }
    provenance = row["provenance"]
    assert provenance["capture_identity"] == bound.capture_identity  # type: ignore[index]
    assert provenance["localize_family_id"] == FAMILY  # type: ignore[index]
    seeds = provenance["seeds"]
    assert seeds["evaluation"] == 42 and seeds["controls"] == 17  # type: ignore[index]
    assert set(seeds["dose_streams"]) == {"0.5", "1.0", "2.0"}  # type: ignore[index]

    outcomes = row["control_outcomes"]
    assert set(outcomes) == set(_control_ids("iv-steer"))  # type: ignore[arg-type]
    assert all(item["status"] == "passed" for item in outcomes.values())  # type: ignore[union-attr]
    controls_payload = row["measurements"]["controls"]
    # Random-direction behavior and off-target effects are recorded, not hidden:
    # the rng-drawn direction moved the metric opposite the declared response.
    assert controls_payload["iv-steer-random"]["value"] == pytest.approx(3.0)  # type: ignore[index]
    assert controls_payload["iv-steer-off"]["value"] == pytest.approx(1.9)  # type: ignore[index]
    assert controls_payload["iv-steer-zero"]["value"] == pytest.approx(_BASELINE)  # type: ignore[index]

    # Every callback saw the same captured inputs and downstream metric.
    assert {app.inputs_digest for app in record} == {bound.capture_identity}
    assert {app.metric_id for app in record} == {METRIC}
    # 1 baseline + 3 nonzero doses + 3x200 dose draws + 3 controls.
    assert len(record) == 607
    dose_applications = [app for app in record if app.role == "intervened" and app.rng is None]
    assert sorted(app.strength for app in dose_applications) == [0.5, 1.0, 2.0]
    control_apps = [app for app in record if app.role == "control"]
    assert {app.control_class for app in control_apps} == set(STEERING_CONTROL_CLASSES)
    zero_app = next(app for app in control_apps if app.control_class == "zero_strength")
    assert zero_app.strength == 0.0
    endpoint_apps = [app for app in control_apps if app.control_class in ("random_direction", "off_target")]
    assert all(app.strength == 2.0 for app in endpoint_apps)

    items = intervention_report_items([row])
    assert items[0]["status"] == "supported"
    assert items[0]["kind"] == "causal_result" and items[0]["causal"] is True
    assert items[0]["claim_allowed"] is True
    assert items[0]["control_refs"] == sorted(_control_ids("iv-steer"))
    assert "steer intervention" in str(items[0]["claim"])


def test_non_monotonic_inert_and_wrong_direction_responses_are_falsified() -> None:
    non_monotonic = _run(
        (_spec("iv-wave"),),
        _steer_measure_factory(
            {"iv-wave": {0.0: _BASELINE, 0.5: 1.5, 1.0: 2.6, 2.0: 1.4}},
            {"iv-wave": dict(_DEFAULT_CONTROLS)},
        ),
    )
    row = list(non_monotonic.payload["trials"])[0]  # type: ignore[arg-type]
    assert row["response_classification"] == "non_monotonic"
    assert row["conclusion"] == "falsified"
    assert "non-monotonic" in str(row["reason"])
    assert non_monotonic.outcome == "completed"

    inert = _run(
        (_spec("iv-inert"),),
        _steer_measure_factory(
            {"iv-inert": {0.0: _BASELINE, 0.5: 2.05, 1.0: 1.98, 2.0: 2.02}},
            {
                "iv-inert": {
                    "zero_strength": _BASELINE,
                    "random_direction": _BASELINE,
                    "off_target": _BASELINE,
                }
            },
        ),
    )
    row = list(inert.payload["trials"])[0]  # type: ignore[arg-type]
    assert row["response_classification"] == "inert"
    assert row["conclusion"] == "falsified"
    assert "inert beyond tolerance" in str(row["reason"])

    backwards = _run(
        (_spec("iv-backwards"),),
        _steer_measure_factory(
            {"iv-backwards": {0.0: _BASELINE, 0.5: 2.2, 1.0: 2.6, 2.0: 3.2}},
            {"iv-backwards": dict(_DEFAULT_CONTROLS)},
        ),
    )
    row = list(backwards.payload["trials"])[0]  # type: ignore[arg-type]
    assert row["response_classification"] == "monotonic_increase"
    assert row["conclusion"] == "falsified"
    assert "contradicts the declared decrease" in str(row["reason"])


def test_failed_identity_random_and_off_target_controls_block_causal_promotion() -> None:
    random_blocked = _run(
        (_spec("iv-random-reproduces"),),
        _steer_measure_factory(
            {"iv-random-reproduces": {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0}},
            {
                "iv-random-reproduces": {
                    "zero_strength": _BASELINE,
                    "random_direction": 0.5,  # same declared sign, at least the endpoint change
                    "off_target": 1.9,
                }
            },
        ),
    )
    conclusions = dict(random_blocked.payload["conclusions"])  # type: ignore[arg-type]
    assert conclusions == {"iv-random-reproduces": "falsified"}
    row = list(random_blocked.payload["trials"])[0]  # type: ignore[arg-type]
    outcomes = row["control_outcomes"]
    assert outcomes["iv-random-reproduces-random"]["status"] == "failed"  # type: ignore[index]
    assert "random-direction control" in str(row["reason"])
    items = intervention_report_items([row])
    assert items[0]["status"] == "falsified"
    assert items[0]["claim_allowed"] is True

    off_blocked = _run(
        (_spec("iv-off-reproduces"),),
        _steer_measure_factory(
            {"iv-off-reproduces": {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0}},
            {
                "iv-off-reproduces": {
                    "zero_strength": _BASELINE,
                    "random_direction": 3.0,
                    "off_target": 0.5,  # off-target moves the metric as much as on-target
                }
            },
        ),
    )
    row = list(off_blocked.payload["trials"])[0]  # type: ignore[arg-type]
    assert row["conclusion"] == "falsified"
    assert "selectivity fails" in str(row["reason"])
    outcomes = row["control_outcomes"]
    assert outcomes["iv-off-reproduces-off"]["status"] == "failed"  # type: ignore[index]

    zero_blocked = _run(
        (_spec("iv-zero-broken"),),
        _steer_measure_factory(
            {"iv-zero-broken": {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0}},
            {
                "iv-zero-broken": {
                    "zero_strength": 2.5,
                    "random_direction": 3.0,
                    "off_target": 1.9,
                }
            },
        ),
    )
    assert zero_blocked.outcome == "unsupported"
    conclusions = dict(zero_blocked.payload["conclusions"])  # type: ignore[arg-type]
    assert conclusions == {"iv-zero-broken": "unsupported"}
    row = list(zero_blocked.payload["trials"])[0]  # type: ignore[arg-type]
    assert row["response_classification"] == "unsupported"
    items = intervention_report_items([row])
    assert items[0]["claim_allowed"] is False
    assert items[0]["missing_evidence"]


def test_uncertainty_spanning_baseline_is_inconclusive() -> None:
    curve = {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0}

    def _draw(app: TrialApplication) -> float:
        assert app.rng is not None
        if app.strength == 2.0:
            return 1.0 + float(app.rng.normal(0.0, 0.6))
        return float(curve[float(app.strength)])

    output = _run(
        (_spec("iv-uncertain"),),
        _steer_measure_factory({"iv-uncertain": dict(curve)}, {"iv-uncertain": dict(_DEFAULT_CONTROLS)}, draw=_draw),
    )
    row = list(output.payload["trials"])[0]  # type: ignore[arg-type]
    assert row["conclusion"] == "inconclusive"
    assert "spans the baseline" in str(row["reason"])
    items = intervention_report_items([row])
    assert items[0]["claim_allowed"] is False
    assert items[0]["missing_evidence"]


def test_direction_dose_plan_and_identity_reject_before_callbacks() -> None:
    manifest = _manifest()
    request = _request()
    prior = _prior(request, manifest)
    record: list[TrialApplication] = []
    executor = make_steering_intervene_executor((_spec(),), _steer_measure_factory({}, {}, record=record))

    # Declaration-level failures raise InterventionError before anything runs.
    with pytest.raises(InterventionError, match="zero/identity dose"):
        _spec(strengths=(1.0, 2.0, 3.0))
    with pytest.raises(InterventionError, match="at least two nonzero doses"):
        _spec(strengths=(0.0, 1.0))
    with pytest.raises(InterventionError, match="finite"):
        _spec(strengths=(0.0, 1.0, float("inf")))
    with pytest.raises(InterventionError, match="unit vector"):
        _spec(direction=np.array([2.0, 0.0, 0.0, 0.0]))
    with pytest.raises(InterventionError, match="1-D"):
        _spec(direction=np.eye(4))
    with pytest.raises(InterventionError, match="finite"):
        _spec(direction=np.array([0.5, 0.5, float("nan"), 0.0]))
    with pytest.raises(InterventionError, match="source"):
        _spec(provenance={"method": "steering-vector", "carrier": "seam"})
    with pytest.raises(InterventionError, match="each required class"):
        SteeringTrialSpec(
            intervention_id="iv-x",
            target=ON_TARGET,
            metric_id=METRIC,
            hypothesis_id=HYPOTHESIS_ID,
            strengths=(0.0, 1.0, 2.0),
            strength_semantics="offset",
            expected_effect="decrease",
            direction=np.array([1.0, 0.0, 0.0, 0.0]),
            provenance={"method": "m", "carrier": "c", "source": "s"},
            controls=_controls("iv-x")[:2],
        )
    with pytest.raises(InterventionError, match="different target"):
        SteeringTrialSpec(
            intervention_id="iv-x",
            target=ON_TARGET,
            metric_id=METRIC,
            hypothesis_id=HYPOTHESIS_ID,
            strengths=(0.0, 1.0, 2.0),
            strength_semantics="offset",
            expected_effect="decrease",
            direction=np.array([1.0, 0.0, 0.0, 0.0]),
            provenance={"method": "m", "carrier": "c", "source": "s"},
            controls=_controls("iv-x", off_target=ON_TARGET),
        )
    with pytest.raises(InterventionError, match="at least one trial"):
        make_steering_intervene_executor((), _steer_measure_factory({}, {}))
    with pytest.raises(InterventionError, match="unique"):
        make_steering_intervene_executor((_spec(), _spec()), _steer_measure_factory({}, {}))
    with pytest.raises(InterventionError, match="SteeringTrialSpec"):
        make_steering_intervene_executor(
            ("not-a-spec",),
            _steer_measure_factory({}, {}),  # type: ignore[arg-type]
        )

    # Binding-level failures raise StageContractError before any callback.
    capture = _capture_payload(request, manifest)
    capture_without_identity = {
        "capture": {
            key: value
            for key, value in dict(capture["capture"]).items()  # type: ignore[arg-type]
            if key != "capture_identity"
        }
    }
    omitted_explain = _explain_payload()
    family_evidence = omitted_explain["family_evidence"]
    assert isinstance(family_evidence, dict)
    evidence_row = family_evidence[HYPOTHESIS_ID]
    assert isinstance(evidence_row, dict)
    evidence_row["outcome"] = "omitted"
    no_slice_localize = _localize_payload(declared_slice_ids=[], affected_slices=[])

    for tampered in (
        _prior(request, manifest, capture_payload={"stage": "capture"}),
        _prior(request, manifest, capture_payload=capture_without_identity),
        _prior(request, manifest, localize_payload=_localize_payload(metric_id="other-metric")),
        _prior(request, manifest, localize_payload=no_slice_localize),
        _prior(request, manifest, explain=omitted_explain),
    ):
        with pytest.raises(StageContractError):
            executor(_invocation(request, manifest, tampered))

    inapplicable = copy.deepcopy(manifest)
    causal = inapplicable["causal_expectation"]
    assert isinstance(causal, dict)
    causal["applicable"] = False
    with pytest.raises(StageContractError, match="not applicable"):
        executor(_invocation(request, inapplicable, prior))

    mismatched_target = _request(
        (
            InterventionRequest(
                intervention_id="iv-steer",
                target="slice-somewhere-else",
                control_ids=_control_ids("iv-steer"),
            ),
        )
    )
    with pytest.raises(StageContractError, match="target"):
        executor(_invocation(mismatched_target, manifest, _prior(mismatched_target, manifest)))

    missing_control = _request(
        (
            InterventionRequest(
                intervention_id="iv-steer",
                target=ON_TARGET,
                control_ids=("iv-steer-zero",),
            ),
        )
    )
    with pytest.raises(StageContractError, match="controls"):
        executor(_invocation(missing_control, manifest, _prior(missing_control, manifest)))

    extra = _request((_intervention(_spec()), _intervention(_spec("iv-undeclared"))))
    with pytest.raises(StageContractError, match="match exactly"):
        executor(_invocation(extra, manifest, _prior(extra, manifest)))

    off_metric_request = _request(metric_ids=(METRIC, "bottleneck-singular-spread"))
    off_metric_executor = make_steering_intervene_executor(
        (_spec(metric_id="bottleneck-singular-spread"),),
        _steer_measure_factory({}, {}, record=record),
    )
    with pytest.raises(StageContractError, match="causal-expectation target"):
        off_metric_executor(_invocation(off_metric_request, manifest, _prior(off_metric_request, manifest)))

    compare_prior = prior + (StageOutput(stage="intervene", outcome="completed", payload={}, artifact_refs=()),)
    with pytest.raises(StageContractError, match="received stage"):
        executor(_invocation(request, manifest, prior, stage="compare", prior_override=compare_prior))

    assert record == []


def test_measurement_contract_violations_fail_closed() -> None:
    manifest = _manifest()
    request = _request()
    prior = _prior(request, manifest)
    invocation = _invocation(request, manifest, prior)

    wrong_metric = make_steering_intervene_executor(
        (_spec(),), lambda _app: Measurement("bottleneck-singular-spread", 1.0, _app.inputs_digest)
    )
    with pytest.raises(InterventionError, match="domain mismatch"):
        wrong_metric(invocation)

    wrong_digest = make_steering_intervene_executor((_spec(),), lambda _app: Measurement(_app.metric_id, 1.0, "f" * 64))
    with pytest.raises(InterventionError, match="inputs mismatch"):
        wrong_digest(invocation)

    non_finite = make_steering_intervene_executor(
        (_spec(),), lambda _app: Measurement(_app.metric_id, float("nan"), _app.inputs_digest)
    )
    with pytest.raises(InterventionError, match="finite"):
        non_finite(invocation)

    crashing = make_steering_intervene_executor((_spec(),), _boom)
    with pytest.raises(RuntimeError, match="steering model exploded"):
        crashing(invocation)

    control_crash = make_steering_intervene_executor((_spec(),), _control_crash)
    with pytest.raises(InterventionError, match="measurement failed"):
        control_crash(invocation)


def _boom(_app: TrialApplication) -> Measurement:
    raise RuntimeError("steering model exploded")


def _control_crash(_app: TrialApplication) -> Measurement:
    if _app.role == "control":
        raise RuntimeError("control harness broke")
    curve = {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0}
    return Measurement(_app.metric_id, curve[float(_app.strength)], _app.inputs_digest)


def test_deterministic_replay_reproduces_identical_payload_direction_and_digest() -> None:
    manifest = _manifest()
    request = _request()
    curves = {"iv-steer": {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0}}
    controls = {"iv-steer": dict(_DEFAULT_CONTROLS)}

    first = _run((_spec(),), _steer_measure_factory(curves, controls), request=request, manifest=manifest)
    second = _run((_spec(),), _steer_measure_factory(curves, controls), request=request, manifest=manifest)

    assert dict(first.payload) == dict(second.payload)  # type: ignore[arg-type]
    workflow_identity = "a" * 64
    assert first.digest(workflow_identity) == second.digest(workflow_identity)
    assert first.digest(workflow_identity) != first.digest("e" * 64)

    row = list(first.payload["trials"])[0]  # type: ignore[arg-type]
    assert row["direction"]["digest"] == _spec().direction_digest  # type: ignore[index]
    config = dict(first.payload["config"])  # type: ignore[arg-type]
    assert config["evaluation_seed"] == 42
    assert config["control_seed"] == 17
    assert config["repetitions"] == 200
    causal = dict(config["causal_expectation"])  # type: ignore[arg-type]
    assert causal["applicable"] is True
    assert causal["target_metric_ids"] == [METRIC]


def test_steering_vector_direction_is_reused_immutably() -> None:
    rng = np.random.default_rng(11)
    positives = rng.normal(size=(40, 4)) + np.array([2.0, 0.0, 0.0, 0.0])
    negatives = rng.normal(size=(40, 4))
    steering = SteeringVector()
    steering.fit(positives, negatives)
    direction = steering.direction

    assert direction.shape == (4,)
    assert float(np.linalg.norm(direction)) == pytest.approx(1.0)
    spec = _spec(
        direction=direction,
        provenance={
            "method": "steering-vector",
            "carrier": "SteeringVector.fit",
            "source": "synthetic-contrast-positives-negatives",
        },
    )
    assert not spec.direction.flags.writeable
    assert (
        spec.direction_digest
        == _spec(
            direction=direction,
            provenance={
                "method": "steering-vector",
                "carrier": "SteeringVector.fit",
                "source": "synthetic-contrast-positives-negatives",
            },
        ).direction_digest
    )

    row_point = steering(np.array([1.0, 1.0, 1.0, 1.0]), 2.5)
    manual = np.array([1.0, 1.0, 1.0, 1.0]) + 2.5 * direction
    assert np.allclose(row_point, manual)

    output = _run(
        (spec,),
        _steer_measure_factory(
            {"iv-steer": {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0}},
            {"iv-steer": dict(_DEFAULT_CONTROLS)},
        ),
    )
    row = list(output.payload["trials"])[0]  # type: ignore[arg-type]
    assert row["direction"]["digest"] == spec.direction_digest  # type: ignore[index]
    assert row["direction"]["provenance"]["carrier"] == "SteeringVector.fit"  # type: ignore[index]
    assert row["conclusion"] == "supported"


def test_workflow_end_to_end_records_every_steer_conclusion_truthfully() -> None:
    ids = ("iv-supported", "iv-wave", "iv-blocked", "iv-uncertain")
    interventions = tuple(_intervention(_spec(intervention_id)) for intervention_id in ids)
    specs = (
        _spec("iv-supported"),
        _spec("iv-wave"),
        _spec("iv-blocked"),
        _spec("iv-uncertain"),
    )
    curves = {
        "iv-supported": {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0},
        "iv-wave": {0.0: _BASELINE, 0.5: 1.5, 1.0: 2.6, 2.0: 1.4},
        "iv-blocked": {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0},
        "iv-uncertain": {0.0: _BASELINE, 0.5: 1.9, 1.0: 1.5, 2.0: 1.0},
    }
    controls = {
        "iv-supported": dict(_DEFAULT_CONTROLS),
        "iv-wave": dict(_DEFAULT_CONTROLS),
        "iv-blocked": {
            "zero_strength": 2.5,
            "random_direction": 3.0,
            "off_target": 1.9,
        },
        "iv-uncertain": dict(_DEFAULT_CONTROLS),
    }

    def _draw(app: TrialApplication) -> float:
        assert app.rng is not None
        if app.intervention_id == "iv-uncertain" and app.strength == 2.0:
            return 1.0 + float(app.rng.normal(0.0, 0.6))
        return float(curves[app.intervention_id][float(app.strength)])

    request = _request(interventions)
    manifest = _manifest()
    prior = _prior(request, manifest)
    payload_by_stage = {output.stage: dict(output.payload) for output in prior}
    executor = make_steering_intervene_executor(specs, _steer_measure_factory(curves, controls, draw=_draw))

    def _stage(invocation: StageInvocation) -> StageOutput:
        if invocation.stage == "intervene":
            return executor(invocation)
        payload = dict(payload_by_stage.get(invocation.stage, {"stage": invocation.stage}))
        if invocation.stage == "report":
            payload["report_id"] = "report-80-19"
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
        "iv-supported": "supported",
        "iv-wave": "falsified",
        "iv-blocked": "unsupported",
        "iv-uncertain": "inconclusive",
    }

    trials_payload = list(intervene_stage["payload"]["trials"])  # type: ignore[index]
    items = intervention_report_items(trials_payload)
    statuses = {item["id"]: item["status"] for item in items}
    assert statuses == {
        "intervention-iv-supported": "supported",
        "intervention-iv-wave": "falsified",
        "intervention-iv-blocked": "unsupported",
        "intervention-iv-uncertain": "inconclusive",
    }
    allowed = {item["id"]: item["claim_allowed"] for item in items}
    assert allowed["intervention-iv-supported"] is True
    assert allowed["intervention-iv-wave"] is True
    assert allowed["intervention-iv-blocked"] is False
    assert allowed["intervention-iv-uncertain"] is False
    assert all(item["causal"] is True for item in items)
    assert all(str(item["claim"]).startswith("steer intervention") for item in items)

    validate_report_shape(_report(items))


def _report(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "diagnostic-report-schema-v1",
        "report_id": "report-80-19",
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
                "selection": ON_TARGET,
                "status": "observed",
            }
        ],
        "hypotheses": [
            {
                "alternatives": ["benign-low-variance"],
                "evidence_refs": ["o-1"],
                "id": HYPOTHESIS_ID,
                "statement": "the collapsed bottleneck direction causes the rank defect",
                "status": "supported",
            }
        ],
        "statistical_evidence": [
            {
                "control_refs": [],
                "estimate": 2.0,
                "evidence_refs": ["o-1"],
                "id": "e-1",
                "metric_id": METRIC,
                "status": "observed",
                "uncertainty": {"kind": "interval", "lower": 1.9, "upper": 2.1},
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
                "target": ON_TARGET,
            }
            for item in items
        ],
        "comparisons": [],
        "limitations": [
            {
                "affects": ["causal"],
                "blocking": False,
                "description": "steering dose curves are caller-supplied in 80.19",
                "id": "lim-1",
            }
        ],
        "next_action": {
            "action": "run aligned checkpoint comparison",
            "rationale": "80.20 owns comparison",
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
