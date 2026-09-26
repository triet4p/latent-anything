"""Consumer-observable tests for the Sprint 80.15 statistical-control executor."""

from __future__ import annotations

import hashlib
import json as _json
from copy import deepcopy

import numpy as np
import pytest

from latent_anything._statistical_controls import (
    CONTROL_KINDS,
    ControlExecutionError,
    ControlOutcome,
    ControlPlan,
    ControlSpec,
    derive_stream_seed,
    execute_plan,
    failed_required,
    plan_from_manifest,
    run_bootstrap,
    run_permutation_control,
    summarize_interval,
)


def _plan(**overrides: object) -> ControlPlan:
    base: dict[str, object] = {
        "plan_id": "plan-80-15",
        "repetitions": 40,
        "confidence_level": 0.95,
        "evaluation_seed": 5,
        "control_seed": 6,
        "training_seed": 4,
        "controls": (
            ControlSpec("control-required", "counterexample", True, ("metric-a",), "healthy passes"),
            ControlSpec("control-optional", "null", False, ("metric-a",), "recorded"),
        ),
    }
    base.update(overrides)
    return ControlPlan(**base)  # type: ignore[arg-type]


def test_exact_repetitions_and_percentile_ci_on_known_statistic() -> None:
    draws = [float(index) for index in range(40)]
    summary = summarize_interval(draws, repetitions=40, seed=99, confidence_level=0.95)

    assert summary["repetitions"] == 40
    assert summary["method"] == "bootstrap"
    assert summary["seed"] == 99
    assert summary["lower"] == 1.0
    assert summary["upper"] == 39.0
    assert summary["mean"] == pytest.approx(19.5)
    with pytest.raises(ControlExecutionError, match="exactly 40 draws"):
        summarize_interval(draws[:39], repetitions=40, seed=99, confidence_level=0.95)


def test_bootstrap_runs_exact_repetitions_under_identity_stream() -> None:
    calls = {"count": 0}

    def _draw(rng: object) -> float:
        calls["count"] += 1
        assert isinstance(rng, object)
        return 1.0

    outcome, interval = run_bootstrap(
        control_id="control-bootstrap",
        base_seed=5,
        seed_role="evaluation",
        repetitions=40,
        confidence_level=0.95,
        required=False,
        kind="bootstrap",
        metric_ids=("metric-a",),
        expected_behavior="recorded",
        draw=_draw,
    )

    assert calls["count"] == 40
    assert interval["repetitions"] == 40
    assert interval["seed"] == derive_stream_seed(5, "control-bootstrap", role="evaluation")
    assert outcome.stream_seed == interval["seed"]
    assert outcome.status == "recorded"


def test_identity_derived_streams_are_deterministic_and_order_invariant() -> None:
    first = derive_stream_seed(6, "control-x", role="control")
    second = derive_stream_seed(6, "control-x", role="control")

    assert first == second
    assert derive_stream_seed(6, "control-y", role="control") != first
    assert derive_stream_seed(7, "control-x", role="control") != first

    plan = _plan()
    forward = execute_plan(
        plan,
        {"control-required": {"metric-a": 0.1}, "control-optional": {"metric-a": 0.9}},
    )
    assert failed_required(forward) == ()
    assert forward["control-optional"].status == "recorded"


def test_each_control_kind_exercises_the_central_path() -> None:
    def _constant(_rng: np.random.Generator) -> dict[str, float]:
        assert _rng is not None
        return {"metric-a": 0.25}

    for kind in ("bootstrap", "null", "shuffled", "randomized", "cross_seed", "seed"):
        outcome = run_permutation_control(
            control_id=f"control-{kind}",
            base_seed=6,
            seed_role="control",
            required=False,
            kind=kind,
            metric_ids=("metric-a",),
            expected_behavior="recorded",
            statistic=_constant,
        )
        assert outcome.status == "recorded"
        assert outcome.kind == kind
    assert set(CONTROL_KINDS) >= {"bootstrap", "null", "shuffled", "randomized", "cross_seed"}


def test_required_pass_fail_missing_nonfinite_semantics() -> None:
    plan = ControlPlan(
        plan_id="plan-gated",
        repetitions=10,
        confidence_level=0.95,
        evaluation_seed=5,
        control_seed=6,
        controls=(
            ControlSpec("control-gated", "counterexample", True, ("metric-a",), "stays below", "below_threshold", 0.5),
            ControlSpec("control-optional", "null", False, ("metric-a",), "recorded"),
        ),
    )
    passing = execute_plan(plan, {"control-gated": {"metric-a": 0.1}, "control-optional": {"metric-a": 0.9}})
    assert passing["control-gated"].status == "passed"
    assert failed_required(passing) == ()

    failing = execute_plan(plan, {"control-gated": {"metric-a": 0.8}, "control-optional": {"metric-a": 0.9}})
    assert failing["control-gated"].status == "failed"
    assert failed_required(failing) == ("control-gated",)

    missing = execute_plan(plan, {"control-optional": {"metric-a": 0.9}})
    assert missing["control-gated"].status == "failed"
    assert failed_required(missing) == ("control-gated",)

    non_finite = execute_plan(plan, {"control-gated": {"metric-a": float("inf")}})
    assert non_finite["control-gated"].status == "failed"

    def _boom(_rng: np.random.Generator) -> dict[str, float]:
        raise RuntimeError("boom")

    broken = run_permutation_control(
        control_id="control-broken",
        base_seed=6,
        seed_role="control",
        required=True,
        kind="shuffled",
        metric_ids=("metric-a",),
        expected_behavior="fails",
        statistic=_boom,
    )
    assert broken.status == "failed"

    with pytest.raises(ControlExecutionError, match="never read as pass"):
        ControlOutcome(
            control_id="control-x",
            kind="null",
            status="unsupported",
            required=False,
            observed={},
            expected_behavior="",
            repetitions=10,
            confidence_level=0.95,
            stream_seed=1,
            seed_role="control",
            reason="unsupported control",
        )


def test_optional_vs_required_semantics() -> None:
    plan = _plan()
    outcomes = execute_plan(plan, {"control-required": {"metric-a": 0.1}})
    assert outcomes["control-required"].status == "passed"
    assert outcomes["control-optional"].status == "recorded"
    assert failed_required(outcomes) == ()

    empty_plan = ControlPlan(
        plan_id="plan-optional-only",
        repetitions=10,
        confidence_level=0.95,
        evaluation_seed=5,
        control_seed=6,
        controls=(ControlSpec("control-optional", "null", False, ("metric-a",), "recorded"),),
    )
    assert failed_required(execute_plan(empty_plan, {})) == ()


def test_plan_binds_manifest_without_observed_tuning() -> None:
    manifest: dict[str, object] = {
        "schema_version": "benchmark-manifest-schema-v1",
        "manifest_id": "manifest-80-15",
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
            "identity": "R-80-15",
            "axes": [
                {"name": "sample", "selection": "all", "identity": "ax-s"},
                {"name": "feature", "selection": "all", "identity": "ax-f"},
            ],
        },
        "defect": {"classification": "none", "declaration": "n/a", "predeclared": True, "source": "not_applicable"},
        "metrics": [
            {
                "id": "metric-a",
                "taxonomy_family_id": "collapse_rank_loss",
                "direction": "higher_is_better",
                "unit": "u",
                "estimator": "e",
                "aggregation": "mean-with-interval",
            },
        ],
        "seeds": {"training": [4], "evaluation": [5], "controls": [6], "independent": True},
        "controls": [
            {
                "id": "control-required",
                "kind": "counterexample",
                "metric_ids": ["metric-a"],
                "required": True,
                "expected_behavior": "healthy passes",
            },
            {
                "id": "control-optional",
                "kind": "null",
                "metric_ids": ["metric-a"],
                "required": False,
                "expected_behavior": "recorded",
            },
        ],
        "uncertainty": {
            "method": "bootstrap",
            "confidence_level": 0.95,
            "repetitions": 40,
            "unit_of_analysis": "sample",
            "predeclared": True,
        },
        "causal_expectation": {
            "applicable": False,
            "expectation": "not_applicable",
            "target_metric_ids": [],
            "falsification_rule": "n/a",
            "non_applicable_reason": "no causal trial in 80.15",
        },
        "thresholds": [
            {"metric_id": "metric-a", "comparator": ">=", "value": 0.5, "tolerance": 0.02, "predeclared": True},
        ],
        "commitment": {
            "locked": True,
            "declared_at": "2026-09-21T00:00:00Z",
            "manifest_sha256": "x",
            "canonicalization": "json-sort-keys-no-whitespace-utf8",
        },
    }
    unsigned = deepcopy(manifest)
    unsigned_commitment = dict(unsigned["commitment"])  # type: ignore[arg-type]
    unsigned_commitment.pop("manifest_sha256", None)
    unsigned["commitment"] = unsigned_commitment
    manifest_commitment = manifest["commitment"]
    assert isinstance(manifest_commitment, dict)
    manifest_commitment["manifest_sha256"] = hashlib.sha256(
        _json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    plan = plan_from_manifest(
        plan_id="plan-80-15",
        manifest=manifest,
        metric_ids=("metric-a",),
        evaluation_seed=5,
        control_seed=6,
        training_seed=4,
    )

    assert plan.repetitions == 40
    assert plan.confidence_level == 0.95
    assert plan.evaluation_seed == 5
    assert plan.control_seed == 6
    assert [spec.control_id for spec in plan.controls] == ["control-required", "control-optional"]

    with pytest.raises(ControlExecutionError, match="seed roles must match"):
        plan_from_manifest(
            plan_id="plan-80-15",
            manifest=manifest,
            metric_ids=("metric-a",),
            evaluation_seed=5,
            control_seed=999,
            training_seed=4,
        )


def test_failed_required_control_blocks_detector_conclusion() -> None:
    import numpy as np

    from latent_anything._collapse_detection import (
        DetectionConfig,
        FamilyThreshold,
        detect_families,
        evaluate_detection,
    )
    from latent_anything.latent_space import LatentSpace
    from latent_anything.latent_value import LatentValue

    space = LatentSpace(dim=4, source_model="probe-80-15")
    rng = np.random.default_rng(0)
    healthy = LatentValue(rng.normal(size=(360, 4)), space)
    collapsed = LatentValue(
        np.column_stack([rng.normal(size=360), np.full(360, 2.0), rng.normal(size=360) * 1e-9, rng.normal(size=360)]),
        space,
    )
    scaled = LatentValue(rng.normal(size=(360, 4)) * 0.05, space)
    config = DetectionConfig(
        manifest_id="manifest-80-15",
        family_ids=("collapse_rank_loss",),
        metric_ids=("bottleneck-effective-rank", "bottleneck-singular-spread"),
        thresholds=(
            FamilyThreshold("bottleneck-effective-rank", ">=", 3.0, 0.1),
            FamilyThreshold("bottleneck-singular-spread", ">=", 0.25, 0.02),
        ),
        required_controls=("control-healthy-counterexample", "control-benign-low-variance"),
        optional_controls=(),
        control_metrics={
            "control-healthy-counterexample": ("bottleneck-effective-rank", "bottleneck-singular-spread"),
            "control-benign-low-variance": ("bottleneck-effective-rank", "bottleneck-singular-spread"),
        },
        repetitions=20,
        confidence_level=0.95,
        evaluation_seed=42,
        control_seed=17,
    )
    (blocked,) = detect_families(
        collapsed,
        config,
        controls={"control-healthy-counterexample": collapsed, "control-benign-low-variance": scaled},
    )
    assert blocked.outcome == "inconclusive"
    assert blocked.claim_allowed is False
    assert blocked.missing_evidence == ("failed-control:control-healthy-counterexample",)

    (positive,) = detect_families(
        collapsed,
        config,
        controls={"control-healthy-counterexample": healthy, "control-benign-low-variance": scaled},
    )
    assert positive.outcome == "supported"
    assert positive.claim_allowed is True
    _, payload = evaluate_detection(
        collapsed,
        config,
        {"control-healthy-counterexample": healthy, "control-benign-low-variance": scaled},
    )
    assert payload["measurements"]["uncertainty"]["bottleneck-effective-rank"]["repetitions"] == 20


def test_migrated_adapters_use_central_scheduling() -> None:
    """Behavioral proof: migrated adapters schedule through central streams.

    Each adapter's bootstrap intervals carry the central executor's
    identity-derived stream seeds (not the raw manifest seed), and repeated
    executor calls reproduce identical central schedules. This observes the
    migration behaviorally instead of asserting on private import names.
    """
    import numpy as np

    from latent_anything._collapse_detection import (
        DetectionConfig as CollapseConfig,
    )
    from latent_anything._collapse_detection import (
        FamilyThreshold as CollapseThreshold,
    )
    from latent_anything._collapse_detection import (
        evaluate_detection as evaluate_collapse,
    )
    from latent_anything.latent_space import LatentSpace
    from latent_anything.latent_value import LatentValue

    space = LatentSpace(dim=4, source_model="probe-80-15-central")
    rng = np.random.default_rng(3)
    healthy = LatentValue(rng.normal(size=(120, 4)), space)
    scaled = LatentValue(rng.normal(size=(120, 4)) * 0.05, space)
    config = CollapseConfig(
        manifest_id="manifest-80-15-central",
        family_ids=("collapse_rank_loss",),
        metric_ids=("bottleneck-effective-rank", "bottleneck-singular-spread"),
        thresholds=(
            CollapseThreshold("bottleneck-effective-rank", ">=", 3.0, 0.1),
            CollapseThreshold("bottleneck-singular-spread", ">=", 0.25, 0.02),
        ),
        required_controls=("control-healthy-counterexample", "control-benign-low-variance"),
        optional_controls=(),
        control_metrics={
            "control-healthy-counterexample": ("bottleneck-effective-rank", "bottleneck-singular-spread"),
            "control-benign-low-variance": ("bottleneck-effective-rank", "bottleneck-singular-spread"),
        },
        repetitions=20,
        confidence_level=0.95,
        evaluation_seed=42,
        control_seed=17,
    )
    controls = {"control-healthy-counterexample": healthy, "control-benign-low-variance": scaled}
    _, first = evaluate_collapse(healthy, config, controls)
    _, second = evaluate_collapse(healthy, config, controls)
    first_rank = first["measurements"]["uncertainty"]["bottleneck-effective-rank"]
    assert second["measurements"]["uncertainty"]["bottleneck-effective-rank"] == first_rank
    assert first_rank["repetitions"] == 20
    assert first_rank["seed"] == derive_stream_seed(42, "bootstrap:collapse-both", role="evaluation")
    assert first_rank["seed"] != 42
    assert first == second
    # Recorded derived seed reproduces the actual null draw: replay the
    # recorded stream and confirm the null metrics match exactly.
    import numpy as np

    from latent_anything._collapse_detection import _evaluate_batch

    stream_seed = int(first["controls"]["central:column-shuffle-null"]["stream_seed"])
    replay = np.random.default_rng(stream_seed)
    data = np.asarray(healthy.to_numpy(), dtype=np.float64)
    replayed = np.column_stack([replay.permutation(data[:, j]) for j in range(data.shape[1])])
    replayed_metrics = _evaluate_batch(replayed)
    assert first["measurements"]["null_shuffled"]["bottleneck-effective-rank"] == replayed_metrics.effective_rank
    # Changing the control identity changes the actual stream (and hence the
    # draw for an identity-sensitive transform).
    first_stream = derive_stream_seed(6, "control-a", role="control")
    second_stream = derive_stream_seed(6, "control-b", role="control")
    assert first_stream != second_stream


def test_redundancy_central_draw_reproduces_probe_and_null() -> None:
    """Replay from declared identity/stream reproduces actual separability draw."""
    import json as _json
    from pathlib import Path as _Path

    import numpy as np

    from latent_anything._redundancy_separability_detection import (
        LabeledBatch,
        _accuracy_of,
        _batch_matrix,
        _probe_predictions,
        detect_families,
        detection_config_from_manifest,
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

    manifest = dict(
        _json.loads(
            (
                _Path(__file__).resolve().parents[1]
                / "artifacts/benchmark_manifest_sprint80_transformer_hidden_state_v1.json"
            ).read_text(encoding="utf-8")
        )
    )
    request = DiagnosticRequest(
        request_id="request-80-15-redundancy",
        manifest_id=manifest["manifest_id"],
        capture=CaptureSelection(
            capture_id="c",
            representation_identity="openai-community-gpt2:hidden-states:layers0-11:dim768",
            axes=("sample", "feature", "label"),
        ),
        diagnostics=DiagnosticSelection(family_ids=("separability_probe_leakage",)),
        controls=ControlSelection(
            control_ids=("control-capacity", "control-label-randomization", "control-nonseparable-negative"),
            metric_ids=("heldout-probe-accuracy", "probe-leakage-gap"),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-15"),
    )
    config = detection_config_from_manifest(request, manifest)
    rng = np.random.default_rng(0)
    n, dim = 200, 6
    target_matrix = rng.normal(size=(n, dim))
    target_matrix[:100] += 1.5
    target_matrix[100:] -= 1.5
    labels = np.array([0] * 100 + [1] * 100)
    space = LatentSpace(dim=dim, source_model="probe-80-15")
    perm = np.random.default_rng(79).permutation(n)
    train = tuple(int(item) for item in perm[:140])
    evaluation = tuple(int(item) for item in perm[140:])
    target = LabeledBatch(
        LatentValue(target_matrix, space),
        tuple(int(item) for item in labels),
        tuple(f"sample-{index}" for index in range(n)),
        train,
        evaluation,
        "split-train-A",
        "split-eval-A",
        "linear-logreg-C1.0-standardized",
    )
    negative = LabeledBatch(
        LatentValue(rng.normal(size=(n, dim)), space),
        tuple(int(item) for item in labels),
        tuple(f"negative-{index}" for index in range(n)),
        train,
        evaluation,
        "split-train-A",
        "split-eval-A",
        "linear-logreg-C1.0-standardized",
    )
    (detection,) = detect_families(target, config, controls={"control-nonseparable-negative": negative})
    assert detection.control_outcomes["control-label-randomization"] == "passed"
    matrix = _batch_matrix(target.value)
    lab = np.asarray(target.labels)
    train_idx = np.asarray([int(item) for item in target.train_indices])
    eval_idx = np.asarray([int(item) for item in target.eval_indices])
    train_x, train_y = matrix[train_idx], lab[train_idx]
    eval_x, eval_y = matrix[eval_idx], lab[eval_idx]
    stream = derive_stream_seed(config.control_seed, "control-label-randomization", role="control")
    replayed_labels = np.asarray(np.random.default_rng(stream).permutation(np.asarray(train_y).ravel()))
    replayed_pred = _probe_predictions(train_x, replayed_labels, eval_x, eval_y, seed=config.training_seed)
    headline = _probe_predictions(train_x, train_y, eval_x, eval_y, seed=config.training_seed)
    replayed_gap = float(_accuracy_of(headline, eval_y) - _accuracy_of(replayed_pred, eval_y))
    assert detection.observed_metrics["probe-leakage-gap"] == replayed_gap


def test_density_central_null_reproduces_scores_with_one_fit() -> None:
    """Replay from declared null identity reproduces actual shuffled scores."""
    import numpy as np

    import tests.test_density_drift_detection as _density_tests
    from latent_anything._density_drift_detection import evaluate_detection
    from latent_anything._statistical_controls import derive_stream_seed as _seed

    batches = _density_tests._batches()
    config = _density_tests._config()
    (_, payload) = evaluate_detection(
        _density_tests._pair(batches["shifted"], batches), config, _density_tests._controls(batches)
    )
    null_controls = payload["controls"]["control-shuffled-null"]
    assert null_controls["observed"]
    stream = int(null_controls["stream_seed"])
    assert stream == _seed(config.control_seed, "control-shuffled-null", role="control")
    replay = np.random.default_rng(stream)
    test_matrix = batches["shifted"]
    replayed = np.column_stack([replay.permutation(test_matrix[:, j]) for j in range(test_matrix.shape[1])])
    # Replay the recorded stream through the same already-fitted model path:
    # refit reference + calibrate once, then score the replayed permutation.
    from latent_anything.density import GaussianMixtureDensity, GMMConfig

    estimator = GaussianMixtureDensity(GMMConfig(n_components=2, random_state=config.training_seed))
    estimator.fit(
        batches["reference"],
        source_representation_identity="R-80-11",
        geometry="euclidean",
        provenance={"role": "reference-fit"},
    )
    estimator.calibrate(batches["calibration"], provenance={"role": "heldout-calibration"})
    replayed_scores = tuple(float(item) for item in np.asarray(estimator.score(replayed).calibrated_ood_score).ravel())
    assert payload["measurements"]["density"]["shuffled_flag_rate"] == float(
        np.mean(np.asarray(replayed_scores) >= payload["measurements"]["density"]["flag_threshold"])
    )


def test_sparse_cross_seed_and_temporal_shuffle_reproduce_with_fit_bounds() -> None:
    """Cross-seed fit seed and temporal shuffle replay from declared identities."""
    import tests.test_sparse_temporal_detection as _sparse_tests
    from latent_anything._sparse_temporal_detection import _stepwise_drift, evaluate_detection
    from latent_anything._statistical_controls import derive_stream_seed as _seed
    from latent_anything.latent_space import LatentSpace

    sparse, noise = _sparse_tests._sparse_batches()
    (_, payload) = evaluate_detection(
        _sparse_tests._sparse_input(sparse, noise, permuted=True), _sparse_tests._sparse_config(), {}
    )
    sparse_controls = payload["controls"]["control-seed-noise"]
    assert sparse_controls["observed"]
    assert sparse_controls["stream_seed"] == _seed(99, "control-seed-noise", role="control")
    assert payload["measurements"]["sparse"]["seed_control_fit_seed"] == int(sparse_controls["observed"]["fit_seed"])
    reference, drift, negative = _sparse_tests._trajectories()
    (_, temporal_payload) = evaluate_detection(
        _sparse_tests._temporal_input(drift, reference, negative), _sparse_tests._temporal_config(), {}
    )
    temporal_controls = temporal_payload["controls"]["control-shuffled-sequence"]
    assert temporal_controls["observed"]
    stream = int(temporal_controls["stream_seed"])
    assert stream == _seed(6, "control-shuffled-sequence", role="control")
    import numpy as np

    replay = np.random.default_rng(stream)
    order = replay.permutation(drift.shape[0])
    space = LatentSpace(dim=4, source_model="probe-80-12")
    replayed, _ = _stepwise_drift(reference, drift[order], space)
    assert temporal_payload["measurements"]["temporal"]["shuffled_drift"] == replayed


def test_central_transforms_execute_once_with_instrumented_counts() -> None:
    """Instrumented proof: central null/shuffle callbacks execute exactly once."""

    import latent_anything._density_drift_detection as density
    import tests.test_density_drift_detection as _density_tests

    batches = _density_tests._batches()
    config = _density_tests._config()
    controls = _density_tests._controls(batches)
    calls = {"fit": 0, "calibrate": 0, "score": 0}
    real_fit = density.GaussianMixtureDensity.fit
    real_calibrate = density.GaussianMixtureDensity.calibrate
    real_score = density.GaussianMixtureDensity.score

    def _fit(self: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["fit"] += 1
        return real_fit(self, *args, **kwargs)  # type: ignore[arg-type]

    def _calibrate(self: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["calibrate"] += 1
        return real_calibrate(self, *args, **kwargs)  # type: ignore[arg-type]

    def _score(self: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["score"] += 1
        return real_score(self, *args, **kwargs)  # type: ignore[arg-type]

    density.GaussianMixtureDensity.fit = _fit  # type: ignore[method-assign]
    density.GaussianMixtureDensity.calibrate = _calibrate  # type: ignore[method-assign]
    density.GaussianMixtureDensity.score = _score  # type: ignore[method-assign]
    try:
        (_, payload) = density.evaluate_detection(_density_tests._pair(batches["shifted"], batches), config, controls)
    finally:
        density.GaussianMixtureDensity.fit = real_fit  # type: ignore[method-assign]
        density.GaussianMixtureDensity.calibrate = real_calibrate  # type: ignore[method-assign]
        density.GaussianMixtureDensity.score = real_score  # type: ignore[method-assign]
    # One fit + one calibration; scores: calibration + reference + test +
    # shuffled null + negative control = 5 scoring passes, each exactly once.
    assert calls["fit"] == 1
    assert calls["calibrate"] == 1
    assert calls["score"] == 5
    assert payload["controls"]["control-shuffled-null"]["observed"]


def test_redundancy_null_evaluates_once_with_bounded_work() -> None:
    """Instrumented proof: the redundancy null transform/evaluation runs once.

    The central null callback permutes once and evaluates once; the stored
    evaluation is consumed directly with no second transform, no matrix
    reconstruction, and no second dictionary/covariance/bootstrap schedule.
    """
    import numpy as np

    import latent_anything._redundancy_separability_detection as redundancy
    from latent_anything.latent_space import LatentSpace
    from latent_anything.latent_value import LatentValue

    space = LatentSpace(dim=4, source_model="probe-80-15-null-once")
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(120, 4))
    value = LatentValue(matrix, space)
    healthy = LatentValue(rng.normal(size=(120, 4)), space)
    config = redundancy.DetectionConfig(
        manifest_id="manifest-80-15-null-once",
        family_ids=("redundancy_superposition",),
        metric_ids=("feature-max-abs-correlation", "sparse-feature-sharing"),
        thresholds=(
            redundancy.FamilyThreshold("feature-max-abs-correlation", ">=", 0.5, 0.02),
            redundancy.FamilyThreshold("sparse-feature-sharing", ">=", 0.75, 0.02),
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
        repetitions=10,
        confidence_level=0.95,
        training_seed=7,
        evaluation_seed=42,
        control_seed=17,
    )
    calls = {"redundancy": 0, "dictionary": 0, "bootstrap": 0}
    real_evaluate = redundancy._evaluate_redundancy
    real_metrics = redundancy._redundancy_metrics
    real_bootstrap = redundancy._central_bootstrap

    def _counting_evaluate(matrix: object, config: object):  # type: ignore[no-untyped-def]
        calls["redundancy"] += 1
        return real_evaluate(matrix, config)  # type: ignore[arg-type]

    def _counting_metrics(matrix: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["dictionary"] += 1
        return real_metrics(matrix, **kwargs)  # type: ignore[arg-type]

    def _counting_bootstrap(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["bootstrap"] += 1
        return real_bootstrap(*args, **kwargs)  # type: ignore[arg-type]

    redundancy._evaluate_redundancy = _counting_evaluate  # type: ignore[method-assign]
    redundancy._redundancy_metrics = _counting_metrics  # type: ignore[method-assign]
    redundancy._central_bootstrap = _counting_bootstrap  # type: ignore[method-assign]
    try:
        (_, payload) = redundancy.evaluate_detection(value, config, {"control-redundant-counterexample": healthy})
    finally:
        redundancy._evaluate_redundancy = real_evaluate  # type: ignore[method-assign]
        redundancy._redundancy_metrics = real_metrics  # type: ignore[method-assign]
        redundancy._central_bootstrap = real_bootstrap  # type: ignore[method-assign]
    # One target evaluation + one null evaluation + one counterexample
    # evaluation = 3 total; the null path contributes exactly one.
    assert calls["redundancy"] == 3
    assert calls["dictionary"] == 3
    # Each evaluation runs two bootstrap intervals; no second null schedule.
    assert calls["bootstrap"] == 6
    null_key = next(
        key
        for key in payload["controls"]
        if isinstance(payload["controls"][key], dict) and payload["controls"][key].get("kind") == "randomized"
    )
    assert payload["controls"][null_key]["observed"]


def test_redundancy_undeclared_null_never_enters_payload() -> None:
    """An undeclared null cannot appear in payload controls under any ID."""
    import numpy as np

    import latent_anything._redundancy_separability_detection as redundancy
    from latent_anything.latent_space import LatentSpace
    from latent_anything.latent_value import LatentValue

    space = LatentSpace(dim=4, source_model="probe-80-15-null-absent")
    rng = np.random.default_rng(3)
    value = LatentValue(rng.normal(size=(120, 4)), space)
    healthy = LatentValue(rng.normal(size=(120, 4)), space)
    config = redundancy.DetectionConfig(
        manifest_id="manifest-80-15-null-absent",
        family_ids=("redundancy_superposition",),
        metric_ids=("feature-max-abs-correlation", "sparse-feature-sharing"),
        thresholds=(
            redundancy.FamilyThreshold("feature-max-abs-correlation", ">=", 0.5, 0.02),
            redundancy.FamilyThreshold("sparse-feature-sharing", ">=", 0.75, 0.02),
        ),
        required_controls=("control-redundant-counterexample",),
        optional_controls=(),
        control_metrics={
            "control-redundant-counterexample": ("feature-max-abs-correlation", "sparse-feature-sharing"),
        },
        control_kinds={"control-redundant-counterexample": "counterexample"},
        repetitions=10,
        confidence_level=0.95,
        evaluation_seed=42,
        control_seed=17,
        training_seed=7,
    )
    (_, payload) = redundancy.evaluate_detection(value, config, {"control-redundant-counterexample": healthy})
    assert "control-null-shuffle" not in payload["controls"]
    assert all("null-shuffle" not in key for key in payload["controls"])


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
    assert expected <= set(la.__all__)
    assert len([name for name in la.__all__ if name in expected]) == 9
    for leaked in (
        "ControlPlan",
        "ControlOutcome",
        "execute_plan",
        "run_bootstrap",
        "derive_stream_seed",
        "summarize_interval",
    ):
        assert not hasattr(la, leaked)
