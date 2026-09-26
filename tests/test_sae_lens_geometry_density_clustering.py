"""Consumer-observable tests for Sprint 80.17 feature explanation evidence."""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything._diagnostic_workflow import StageInvocation, StageOutput
from latent_anything._probe_tcav_ig_explanation import (
    ExplanationError,
    ExplanationHypothesis,
    MethodInputs,
    explanation_report_items,
)
from latent_anything._sae_lens_geometry_density_clustering import (
    FEATURE_EXPLAINER_VERSION,
    SUPPORTED_FEATURE_METHODS,
    evaluate_feature_explanations,
    feature_explanation_payload,
    make_feature_explain_executor,
)


def _hypothesis(method: str, hypothesis_id: str = "hyp-80-17", **overrides: object) -> ExplanationHypothesis:
    bases: dict[str, dict[str, object]] = {
        "probe": {
            "metric_ids": ("heldout_accuracy",),
            "thresholds": (("heldout_accuracy", ">=", 0.7),),
        },
        "sae_sparse": {
            "metric_ids": ("reconstruction_quality", "selectivity_margin"),
            "thresholds": (
                ("reconstruction_quality", ">=", 0.8),
                ("sae_stability", ">=", 0.5),
                ("selectivity_margin", ">=", 0.2),
            ),
        },
        "lens": {
            "metric_ids": ("readout_fidelity", "readout_margin"),
            "thresholds": (
                ("readout_fidelity", ">=", 0.99),
                ("readout_stability", ">=", 0.9),
                ("readout_margin", ">=", 0.1),
            ),
        },
        "geometry": {
            "metric_ids": ("fit_fidelity", "coverage_margin"),
            "thresholds": (
                ("fit_fidelity", ">=", 0.5),
                ("subspace_stability", ">=", 0.9),
                ("coverage_margin", ">=", 0.2),
            ),
        },
        "density": {
            "metric_ids": ("density_auroc", "flag_margin"),
            "thresholds": (
                ("density_auroc", ">=", 0.8),
                ("auroc_stability", "<=", 0.1),
                ("flag_margin", ">=", 0.3),
            ),
        },
        "clustering": {
            "metric_ids": ("label_agreement", "selectivity_margin"),
            "thresholds": (
                ("label_agreement", ">=", 0.8),
                ("cluster_stability", ">=", 0.8),
                ("selectivity_margin", ">=", 0.5),
            ),
        },
    }
    base: dict[str, object] = {
        "hypothesis_id": hypothesis_id,
        "symptom_id": "symptom-1",
        "family_id": "sparse_feature_instability",
        "target_id": "feature-3",
        "representation_id": "rep-80-17",
        "layer_id": "layer-2",
        "slice_id": "slice-A",
        "method": method,
        "expected_direction": "higher",
        "dataset_id": "dataset-80-17",
        "train_split_identity": "split-train-A",
        "eval_split_identity": "split-eval-A",
        "seeds": (7, 8),
        "control_ids": ("fidelity:control-fidelity", "stability:control-stability", "selectivity:control-selectivity"),
        "localization_bindings": (("layer", "layer-2"), ("slice", "slice-A")),
        **bases[method],
    }
    base.update(overrides)
    return ExplanationHypothesis(**base)  # type: ignore[arg-type]


_TRUE_ATOMS = np.array(
    [
        [1.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0],
        [0.0, 0.1, 1.0, 0.2, 0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.1, 1.0, 0.2, 0.0, 0.1],
        [0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 1.0, 0.2],
    ]
)
_TRUE_ATOMS = _TRUE_ATOMS / np.linalg.norm(_TRUE_ATOMS, axis=1, keepdims=True)


def _sparse_matrix(n: int, seed: int, *, with_atom0: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes = np.zeros((n, 4))
    for i in range(n):
        avail = [0, 1, 2, 3] if with_atom0 else [1, 2, 3]
        idx = rng.choice(avail, size=2, replace=False)
        codes[i, idx] = rng.normal(size=2) * 2.0
    return codes @ _TRUE_ATOMS + rng.normal(size=(n, 8)) * 0.1


def _sae_bundle() -> dict[str, object]:
    n = 300
    return {
        "symptom_matrix": _sparse_matrix(n, 1, with_atom0=True),
        "negative_matrix": _sparse_matrix(n, 3, with_atom0=False),
        "off_target_matrix": _sparse_matrix(n, 11, with_atom0=False),
        "sample_ids": [f"sample-{i}" for i in range(n)],
        "n_components": 4,
        "feature_label": "edge-detector-like feature 3",
        "symptom_relationship": "headline sparse feature activates selectively on the diagnosed symptom slice",
    }


def _lens_bundle(vocab: int = 10, target: int = 7, off: int = 2) -> dict[str, object]:
    from types import SimpleNamespace

    import torch

    from latent_anything._transformer_analysis import apply_logit_lens, softmax

    torch.manual_seed(0)
    dim = 8
    ln = torch.nn.LayerNorm(dim)
    head = torch.nn.Linear(dim, vocab, bias=False)
    torch.manual_seed(7)
    torch.nn.init.normal_(head.weight, std=1.0)
    model = SimpleNamespace(transformer=SimpleNamespace(ln_f=ln), lm_head=head)
    rng = np.random.default_rng(0)
    symptom = torch.as_tensor(rng.normal(size=(60, dim)) + np.array([2.0] + [0.0] * 7), dtype=torch.float32)
    benign = torch.as_tensor(rng.normal(size=(60, dim)), dtype=torch.float32)
    symptom_logits = apply_logit_lens(model, symptom)
    benign_logits = apply_logit_lens(model, benign)
    # Off-target: same lens, non-target token column; random: different-head lens.
    head2 = torch.nn.Linear(dim, vocab, bias=False)
    torch.manual_seed(8)
    torch.nn.init.normal_(head2.weight, std=1.0)
    model2 = SimpleNamespace(transformer=SimpleNamespace(ln_f=ln), lm_head=head2)
    random_logits = apply_logit_lens(model2, symptom)
    jitter = symptom_logits + rng.normal(size=symptom_logits.shape) * 0.001
    seed_variant = symptom_logits + rng.normal(size=symptom_logits.shape) * 0.01
    return {
        "symptom_logits": symptom_logits,
        "benign_logits": benign_logits,
        "off_target_logits": symptom_logits,
        "random_logits": random_logits,
        "reference_logits": symptom_logits + rng.normal(size=symptom_logits.shape) * 0.001,
        "seed_logits": {"seed-8": seed_variant},
        "target_index": target,
        "off_target_index": off,
        "token_ids": [f"tok-{i}" for i in range(60)],
        "sample_ids": [f"sample-{i}" for i in range(60)],
        "preprocessing_identity": "tokenizer-v1::lower=false",
        "promoted_projection": [float(v) for v in softmax(symptom_logits).mean(axis=0)],
        "symptom_relationship": "target readout mass separates the diagnosed symptom slice",
        "_jitter": jitter,
    }


def _geometry_bundle() -> dict[str, object]:
    rng = np.random.default_rng(0)
    basis_true = np.linalg.qr(rng.normal(size=(8, 2)))[0]
    symptom = rng.normal(size=(200, 2)) @ basis_true.T + rng.normal(size=(200, 8)) * 0.1
    benign = rng.normal(size=(200, 8))
    negative = rng.normal(size=(200, 8)) * 0.5
    return {
        "symptom_matrix": symptom,
        "benign_matrix": benign,
        "negative_matrix": negative,
        "subspace_rank": 2,
        "sample_ids": [f"sample-{i}" for i in range(200)],
        "seed_matrices": {"seed-8": rng.normal(size=(200, 2)) @ basis_true.T + rng.normal(size=(200, 8)) * 0.1},
        "symptom_relationship": "fitted subspace covers the diagnosed affected slice",
    }


def _density_bundle() -> dict[str, object]:
    rng = np.random.default_rng(0)
    return {
        "reference_matrix": rng.normal(size=(400, 4)),
        "calibration_matrix": rng.normal(size=(100, 4)),
        "symptom_matrix": rng.normal(size=(100, 4)) + 3.0,
        "negative_matrix": rng.normal(size=(100, 4)) + 0.2,
        "reference_identity": "rep-80-17",
        "test_identity": "rep-80-17::symptom-slice",
        "calibration_identity": "rep-80-17",
        "symptom_relationship": "density scores separate the diagnosed symptom slice from in-distribution data",
    }


def _clustering_bundle() -> dict[str, object]:
    rng = np.random.default_rng(0)
    symptom = np.concatenate([rng.normal(size=(60, 4)) + 3.0, rng.normal(size=(60, 4)) - 3.0])
    benign = rng.normal(size=(120, 4))
    labels = np.array([0] * 60 + [1] * 60)
    return {
        "symptom_matrix": symptom,
        "benign_matrix": benign,
        "symptom_labels": labels,
        "n_clusters": 2,
        "sample_ids": [f"sample-{i}" for i in range(120)],
        "cluster_label": "high-activation cluster",
        "symptom_relationship": "clusters align with the diagnosed symptom labels",
    }


def _inputs(**bundles: object) -> MethodInputs:
    return MethodInputs(**bundles)  # type: ignore[arg-type]


def test_each_method_executes_only_for_declared_hypothesis() -> None:
    calls = {"count": 0}
    hypothesis = _hypothesis("sae_sparse")
    (record,), _ = evaluate_feature_explanations((hypothesis,), _inputs())
    assert calls["count"] == 0
    assert record.outcome == "omitted"
    assert record.claim_allowed is False
    assert record.missing_evidence == ("omitted:undeclared-hypothesis-or-method",)


def test_missing_relationship_blocks_before_method_invocation() -> None:
    calls = {"count": 0}
    hypothesis = _hypothesis("geometry")
    bundle = dict(_geometry_bundle())
    del bundle["symptom_relationship"]
    before = dict(bundle)
    (record,), _ = evaluate_feature_explanations((hypothesis,), _inputs(geometry=bundle))
    assert record.outcome == "omitted"
    assert bundle == before
    assert calls["count"] == 0
    assert "relationship" in record.reason


def test_sae_positive_supports_and_negative_blocks() -> None:
    hypothesis = _hypothesis("sae_sparse")
    (record,), _ = evaluate_feature_explanations((hypothesis,), _inputs(sae_sparse=_sae_bundle()))
    assert record.outcome == "supported"
    assert record.claim_allowed is True
    assert record.provenance.get("promoted_label") == "edge-detector-like feature 3"
    assert record.stability["permutation_invariant"] is True
    # Noise negative: unstructured data cannot promote a feature label.
    bundle = dict(_sae_bundle())
    rng = np.random.default_rng(4)
    bundle["symptom_matrix"] = rng.normal(size=(300, 8))
    bundle["negative_matrix"] = rng.normal(size=(300, 8))
    bundle["off_target_matrix"] = rng.normal(size=(300, 8))
    (blocked,), _ = evaluate_feature_explanations((hypothesis,), _inputs(sae_sparse=bundle))
    assert blocked.outcome == "inconclusive"
    assert blocked.claim_allowed is False
    assert "promoted_label" not in blocked.provenance
    assert "redacted:semantic-feature-label" in blocked.missing_evidence


def test_raw_index_permutation_leaves_sae_stability_unchanged() -> None:
    from latent_anything._sae_lens_geometry_density_clustering import _matched_stability
    from latent_anything._sae_metrics import match_by_decoder_cosine

    rng = np.random.default_rng(0)
    atoms_a = rng.normal(size=(4, 8))
    atoms_b = atoms_a[np.array([2, 0, 3, 1])]  # raw-index permutation
    fraction, cosines, mean = _matched_stability(atoms_a, atoms_b)
    assert fraction == 1.0
    assert mean == pytest.approx(1.0)
    direct = match_by_decoder_cosine(atoms_a.T, atoms_b.T, 0.5)
    assert len(direct) == 4


def test_lens_positive_supports_and_unstable_blocks() -> None:
    hypothesis = _hypothesis("lens")
    (record,), _ = evaluate_feature_explanations((hypothesis,), _inputs(lens=_lens_bundle()))
    assert record.outcome == "supported"
    assert record.provenance.get("promoted_projection") is not None
    bundle = dict(_lens_bundle())
    bundle["benign_logits"] = bundle["symptom_logits"]  # no selectivity
    (blocked,), _ = evaluate_feature_explanations((hypothesis,), _inputs(lens=bundle))
    assert blocked.outcome == "inconclusive"
    assert "promoted_projection" not in blocked.provenance
    assert "redacted:semantic-projection" in blocked.missing_evidence


def test_geometry_positive_supports_and_noise_blocks() -> None:
    hypothesis = _hypothesis("geometry")
    (record,), _ = evaluate_feature_explanations((hypothesis,), _inputs(geometry=_geometry_bundle()))
    assert record.outcome == "supported"
    assert record.stability["principal_angle_metric"] is True
    bundle = dict(_geometry_bundle())
    rng = np.random.default_rng(5)
    bundle["symptom_matrix"] = rng.normal(size=(200, 8))
    (blocked,), _ = evaluate_feature_explanations((hypothesis,), _inputs(geometry=bundle))
    assert blocked.outcome == "inconclusive"
    assert blocked.claim_allowed is False


def test_subspace_permutation_invariance() -> None:
    from latent_anything.geometry import orthonormalize_directions, subspace_alignment

    rng = np.random.default_rng(0)
    basis = orthonormalize_directions(rng.normal(size=(8, 2)))
    permuted = basis[:, ::-1]
    assert subspace_alignment(basis, permuted) == pytest.approx(1.0)
    other = orthonormalize_directions(rng.normal(size=(8, 2)))
    assert subspace_alignment(basis, other) < 0.9


def test_density_positive_supports_with_one_fit_and_negative_blocks() -> None:
    import latent_anything.density as density_module

    hypothesis = _hypothesis("density")
    bundle = _density_bundle()
    fits = {"count": 0}
    real_fit = density_module.GaussianMixtureDensity.fit

    def _counting_fit(self: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        fits["count"] += 1
        return real_fit(self, *args, **kwargs)  # type: ignore[arg-type]

    density_module.GaussianMixtureDensity.fit = _counting_fit  # type: ignore[method-assign]
    try:
        (record,), _ = evaluate_feature_explanations((hypothesis,), _inputs(density=bundle))
    finally:
        density_module.GaussianMixtureDensity.fit = real_fit  # type: ignore[method-assign]
    assert record.outcome == "supported"
    assert fits["count"] == 1
    assert record.observed_effect["estimator_fits"] == 1.0
    assert record.leakage["one_fit"] is True
    negative = dict(bundle)
    negative["symptom_matrix"] = np.random.default_rng(6).normal(size=(100, 4)) + 0.2
    (blocked,), _ = evaluate_feature_explanations((hypothesis,), _inputs(density=negative))
    assert blocked.outcome == "inconclusive"
    assert "outlier" not in blocked.reason.lower() or "cause" not in blocked.reason.lower()


def test_clustering_positive_supports_and_shuffled_blocks() -> None:
    hypothesis = _hypothesis("clustering")
    (record,), _ = evaluate_feature_explanations((hypothesis,), _inputs(clustering=_clustering_bundle()))
    assert record.outcome == "supported"
    assert record.provenance.get("promoted_label") == "high-activation cluster"
    assert record.stability["permutation_invariant"] is True
    bundle = dict(_clustering_bundle())
    bundle["symptom_matrix"] = np.random.default_rng(6).normal(size=(120, 4))
    (blocked,), _ = evaluate_feature_explanations((hypothesis,), _inputs(clustering=bundle))
    assert blocked.outcome == "inconclusive"
    assert "promoted_label" not in blocked.provenance
    assert "redacted:semantic-cluster-label" in blocked.missing_evidence


def test_cluster_raw_id_permutation_leaves_ari_unchanged() -> None:
    from sklearn.metrics import adjusted_rand_score

    from latent_anything.clustering import KMeans, KMeansConfig

    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(size=(60, 4)) + 3.0, rng.normal(size=(60, 4)) - 3.0])
    first = KMeans(KMeansConfig(n_clusters=2, random_state=7, n_init=10)).fit_predict(data)
    second = KMeans(KMeansConfig(n_clusters=2, random_state=8, n_init=10)).fit_predict(data)
    base = float(adjusted_rand_score(first.assignments, second.assignments))
    permuted = np.array([1, 0])[np.asarray(second.assignments).ravel()]
    assert float(adjusted_rand_score(first.assignments, permuted)) == pytest.approx(base)


def test_each_method_missing_evidence_blocks() -> None:
    for method, bundle in (
        ("sae_sparse", {"symptom_matrix": [[1.0]]}),
        ("lens", {"symptom_logits": [[1.0]]}),
        ("geometry", {"symptom_matrix": [[1.0]]}),
        ("density", {"reference_matrix": [[1.0]]}),
        ("clustering", {"symptom_matrix": [[1.0]]}),
    ):
        full = dict(bundle)
        full["symptom_relationship"] = "declared relationship"
        (record,), _ = evaluate_feature_explanations((_hypothesis(method),), _inputs(**{method: full}))
        assert record.outcome in ("inconclusive", "unsupported", "omitted")
        assert record.claim_allowed is False


def test_all_five_dimensions_and_control_records_present() -> None:
    bundles = {
        "sae_sparse": _sae_bundle(),
        "lens": _lens_bundle(),
        "geometry": _geometry_bundle(),
        "density": _density_bundle(),
        "clustering": _clustering_bundle(),
    }
    hypotheses = tuple(_hypothesis(method, hypothesis_id=f"hyp-{method}") for method in bundles)
    (records, payload) = evaluate_feature_explanations(hypotheses, _inputs(**bundles))
    assert [r.method for r in records] == ["sae_sparse", "lens", "geometry", "density", "clustering"]
    for record in records:
        for dim in ("fidelity", "stability", "selectivity", "leakage", "uncertainty"):
            assert getattr(record, dim)["status"] in ("passed", "failed", "missing")
        assert record.central_outcomes
        assert sorted(record.control_outcomes) == ["control-fidelity", "control-selectivity", "control-stability"]
        assert record.to_dict()["kind"] == "explanation"
        assert record.to_dict()["causal"] is False
    assert payload["explainer_version"] == FEATURE_EXPLAINER_VERSION


def test_central_streams_deterministic_with_exact_reps() -> None:
    hypothesis = _hypothesis("geometry")
    (first,), _ = evaluate_feature_explanations((hypothesis,), _inputs(geometry=_geometry_bundle()))
    (second,), _ = evaluate_feature_explanations((hypothesis,), _inputs(geometry=_geometry_bundle()))
    assert first.to_dict() == second.to_dict()
    assert first.uncertainty["geometry_coverage"]["repetitions"] == 20
    assert first.uncertainty["geometry_coverage"]["seed"] != 0


def test_no_duplicate_expensive_work() -> None:
    import latent_anything.density as density_module

    hypothesis = _hypothesis("density")
    bundle = _density_bundle()
    counts = {"fit": 0, "calibrate": 0, "score": 0}
    real_fit = density_module.GaussianMixtureDensity.fit
    real_cal = density_module.GaussianMixtureDensity.calibrate
    real_score = density_module.GaussianMixtureDensity.score

    def _fit(self: object, *a: object, **k: object):  # type: ignore[no-untyped-def]
        counts["fit"] += 1
        return real_fit(self, *a, **k)  # type: ignore[arg-type]

    def _cal(self: object, *a: object, **k: object):  # type: ignore[no-untyped-def]
        counts["calibrate"] += 1
        return real_cal(self, *a, **k)  # type: ignore[arg-type]

    def _score(self: object, *a: object, **k: object):  # type: ignore[no-untyped-def]
        counts["score"] += 1
        return real_score(self, *a, **k)  # type: ignore[arg-type]

    density_module.GaussianMixtureDensity.fit = _fit  # type: ignore[method-assign]
    density_module.GaussianMixtureDensity.calibrate = _cal  # type: ignore[method-assign]
    density_module.GaussianMixtureDensity.score = _score  # type: ignore[method-assign]
    try:
        evaluate_feature_explanations((hypothesis,), _inputs(density=dict(bundle)))
    finally:
        density_module.GaussianMixtureDensity.fit = real_fit  # type: ignore[method-assign]
        density_module.GaussianMixtureDensity.calibrate = real_cal  # type: ignore[method-assign]
        density_module.GaussianMixtureDensity.score = real_score  # type: ignore[method-assign]
    # One fit + one calibration; scores: calibration + symptom + negative + shuffled null = 4.
    assert counts == {"fit": 1, "calibrate": 1, "score": 4}


def test_report_schema_canonical_workflow_integration() -> None:
    from latent_anything._diagnostic_report import validate_report_shape
    from latent_anything._portable_contract import canonical_json

    bundles = {
        "sae_sparse": _sae_bundle(),
        "lens": _lens_bundle(),
        "geometry": _geometry_bundle(),
        "density": _density_bundle(),
        "clustering": _clustering_bundle(),
    }
    hypotheses = tuple(_hypothesis(method, hypothesis_id=f"hyp-{method}") for method in bundles)
    (records, payload) = evaluate_feature_explanations(hypotheses, _inputs(**bundles))
    canonical_json([r.to_dict() for r in records])
    items = explanation_report_items(records)
    assert {i["kind"] for i in items} == {"explanation"}
    assert all(i["causal"] is False for i in items)
    report = {
        "schema_version": "diagnostic-report-schema-v1",
        "report_id": "report-80-17",
        "status": "declared",
        "capture_provenance": {
            "captures": [
                {
                    "capture_id": "c",
                    "model_revision": "m@r",
                    "dataset_split": "d@s",
                    "representation_identity": "rep",
                    "axes": ["sample", "feature"],
                    "artifact_refs": ["cap-1"],
                }
            ]
        },
        "symptoms": [
            {"id": "s-1", "description": "d", "metric_ids": ["m-1"], "evidence_refs": ["o-1"], "status": "observed"}
        ],
        "localization": [
            {
                "id": "l-1",
                "axis": "layer",
                "selection": "s",
                "evidence_refs": ["o-1"],
                "confidence": 0.8,
                "status": "observed",
            }
        ],
        "hypotheses": [
            {
                "id": "hyp-sae_sparse",
                "statement": "s",
                "evidence_refs": ["o-1"],
                "alternatives": ["a"],
                "status": "supported",
            }
        ],
        "statistical_evidence": [
            {
                "id": "e-1",
                "metric_id": "m-1",
                "estimate": 0.2,
                "uncertainty": {"kind": "interval", "lower": 0.1, "upper": 0.3},
                "control_refs": ["c-1"],
                "evidence_refs": ["a-1"],
                "status": "supported",
            }
        ],
        "interventions": [
            {
                "id": "i-1",
                "intervention": "i",
                "target": "t",
                "control_refs": ["c-1"],
                "outcome": {},
                "evidence_refs": ["a-1"],
                "status": "supported",
            }
        ],
        "comparisons": [
            {
                "id": "c-1",
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
                "id": "o-1",
                "kind": "observation",
                "status": "observed",
                "claim": "c",
                "evidence_refs": ["cap-1"],
                "control_refs": [],
                "causal": False,
                "claim_allowed": True,
                "missing_evidence": [],
            },
            *items,
        ],
    }
    validate_report_shape(report)

    import json
    from pathlib import Path

    manifest = dict(
        json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "artifacts/benchmark_manifest_sprint80_transformer_hidden_state_v1.json"
            ).read_text(encoding="utf-8")
        )
    )
    import dataclasses

    from latent_anything.diagnostics import (
        CaptureSelection,
        ControlSelection,
        DiagnosticRequest,
        DiagnosticSelection,
        OutputSelection,
    )

    manifest_rep = "openai-community-gpt2:hidden-states:layers0-11:dim768"
    manifest_id = str(manifest["manifest_id"])
    hypotheses = tuple(
        dataclasses.replace(
            h,
            representation_id=manifest_rep,
            dataset_id="wikitext-2-raw-v1-validation",
            manifest_id=manifest_id,
        )
        for h in hypotheses
    )
    bundles = dict(bundles)
    bundles["density"] = dict(bundles["density"])
    bundles["density"]["reference_identity"] = manifest_rep
    bundles["density"]["calibration_identity"] = manifest_rep
    bundles["density"]["test_identity"] = manifest_rep + "::symptom-slice"
    request = DiagnosticRequest(
        request_id="request-80-17",
        manifest_id=manifest_id,
        capture=CaptureSelection(capture_id="c", representation_identity=manifest_rep, axes=("sample", "feature")),
        diagnostics=DiagnosticSelection(family_ids=("separability_probe_leakage",)),
        controls=ControlSelection(
            control_ids=("control-capacity", "control-label-randomization", "control-nonseparable-negative"),
            metric_ids=("heldout-probe-accuracy", "probe-leakage-gap"),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-17"),
    )
    import numpy as np

    from latent_anything._redundancy_separability_detection import (
        LabeledBatch as _FeatBatch,
    )
    from latent_anything._redundancy_separability_detection import (
        detection_config_from_manifest as _feat_config,
    )
    from latent_anything._redundancy_separability_detection import (
        evaluate_detection as _feat_detect,
    )
    from latent_anything.latent_space import LatentSpace as _FeatSpace
    from latent_anything.latent_value import LatentValue as _FeatValue

    feat_request = DiagnosticRequest(
        request_id="request-80-17-detect",
        manifest_id=manifest_id,
        capture=CaptureSelection(capture_id="c", representation_identity=manifest_rep, axes=("sample", "feature")),
        diagnostics=DiagnosticSelection(family_ids=("separability_probe_leakage",)),
        controls=ControlSelection(
            control_ids=("control-capacity", "control-label-randomization", "control-nonseparable-negative"),
            metric_ids=("heldout-probe-accuracy", "probe-leakage-gap"),
        ),
        output=OutputSelection(output_location="artifacts/diagnostics/request-80-17"),
    )
    feat_config = _feat_config(feat_request, manifest)
    feat_rng = np.random.default_rng(0)
    feat_matrix = feat_rng.normal(size=(200, 6))
    feat_matrix[:100] += 1.5
    feat_matrix[100:] -= 1.5
    feat_labels = np.array([0] * 100 + [1] * 100)
    feat_space = _FeatSpace(dim=6, source_model="probe-80-17-real")
    feat_perm = np.random.default_rng(79).permutation(200)
    feat_train = tuple(int(item) for item in feat_perm[:140])
    feat_eval = tuple(int(item) for item in feat_perm[140:])
    feat_target = _FeatBatch(
        _FeatValue(feat_matrix, feat_space),
        tuple(int(item) for item in feat_labels),
        tuple(f"sample-{index}" for index in range(200)),
        feat_train,
        feat_eval,
        "split-train-A",
        "split-eval-A",
        "linear-logreg-C1.0-standardized",
    )
    feat_negative = _FeatBatch(
        _FeatValue(feat_rng.normal(size=(200, 6)), feat_space),
        tuple(int(item) for item in feat_labels),
        tuple(f"negative-{index}" for index in range(200)),
        feat_train,
        feat_eval,
        "split-train-A",
        "split-eval-A",
        "linear-logreg-C1.0-standardized",
    )
    _, feat_detect = _feat_detect(feat_target, feat_config, {"control-nonseparable-negative": feat_negative})
    assert "stage" not in str(dict(feat_detect))
    # Feature hypotheses bind the real detect family; their method metric IDs
    # are hypothesis-local (validated by method inputs), so the workflow test
    # declares the real detect metric vocabulary alongside them.
    hypotheses = tuple(
        dataclasses.replace(
            h,
            family_id="separability_probe_leakage",
            metric_ids=("heldout-probe-accuracy", "probe-leakage-gap"),
        )
        for h in hypotheses
    )
    executor = make_feature_explain_executor(hypotheses, _inputs(**bundles))
    assert executor.explainer_version == FEATURE_EXPLAINER_VERSION
    prior = (
        StageOutput(stage="capture", outcome="completed", payload={"capture_id": "c"}, artifact_refs=()),
        StageOutput(stage="detect", outcome="completed", payload=dict(feat_detect), artifact_refs=()),
        StageOutput(
            stage="localize",
            outcome="completed",
            payload={
                "family_id": "separability_probe_leakage",
                "metric_id": "heldout-probe-accuracy",
                "verdict": "localized",
                "layer_order": ["layer-2"],
                "affected_layers": ["layer-2"],
                "declared_slice_ids": ["slice-A"],
                "affected_slices": ["slice-A"],
            },
            artifact_refs=(),
        ),
    )
    produced = executor(
        StageInvocation(
            stage="explain",
            request=request,
            manifest=manifest,
            prior=prior,
            workflow_identity="w" * 64,
            request_digest="r" * 64,
            manifest_digest="m" * 64,
            config_digest="c" * 64,
        )
    )
    assert produced.outcome == "completed"
    # Declared controls drive execution: payload keys equal the declaration.
    (records2, _) = evaluate_feature_explanations(hypotheses, _inputs(**bundles))
    for record in records2:
        assert sorted(record.control_outcomes) == ["control-fidelity", "control-selectivity", "control-stability"]
    # Mismatched representation fails closed.
    from latent_anything._diagnostic_workflow import StageContractError

    wrong = dataclasses.replace(hypotheses[0], representation_id="SOME-OTHER-REP")
    wrong_executor = make_feature_explain_executor((wrong,), _inputs(**{wrong.method: bundles[wrong.method]}))
    with __import__("pytest").raises(StageContractError, match="representation"):
        wrong_executor(
            StageInvocation(
                stage="explain",
                request=request,
                manifest=manifest,
                prior=prior,
                workflow_identity="w" * 64,
                request_digest="r" * 64,
                manifest_digest="m" * 64,
                config_digest="c" * 64,
            )
        )

    def _run_feature(candidate: object) -> str:
        _executor = make_feature_explain_executor(
            (candidate,), _inputs(**{candidate.method: bundles[candidate.method]})
        )  # type: ignore[union-attr]
        _produced = _executor(
            StageInvocation(
                stage="explain",
                request=request,
                manifest=manifest,
                prior=prior,
                workflow_identity="w" * 64,
                request_digest="r" * 64,
                manifest_digest="m" * 64,
                config_digest="c" * 64,
            )
        )
        assert _produced.outcome == "completed"
        return "completed"

    assert _run_feature(hypotheses[0]) == "completed"
    # Status words are not detect families: the real payload carries
    # ``evidence_status`` values of ``"observed"``, which must not bind.
    observed_family = dataclasses.replace(hypotheses[0], hypothesis_id="hyp-observed", family_id="observed")
    with __import__("pytest").raises(StageContractError, match="family"):
        _run_feature(observed_family)
    # Control-kind words are not detect metrics: ``"null"`` names a control
    # kind in the real payload config, not a declared metric identity.
    null_metric = dataclasses.replace(
        hypotheses[0],
        hypothesis_id="hyp-null-metric",
        metric_ids=("null",),
    )
    with __import__("pytest").raises(StageContractError, match="metric"):
        _run_feature(null_metric)
    fabricated_family = dataclasses.replace(
        hypotheses[0], hypothesis_id="hyp-bad-family", family_id="family-FABRICATED"
    )
    with __import__("pytest").raises(StageContractError, match="family"):
        _run_feature(fabricated_family)
    # Axial bindings: matching checkpoint/token/time bindings complete;
    # mismatched/absent/not_applicable bindings fail before any callback.
    import tests.test_checkpoint_token_time_localization as _axial_tests
    from latent_anything._layer_slice_localization import evaluate_axial_localization

    _, _axial_payload = evaluate_axial_localization(_axial_tests._checkpoint_source())
    _, _token_payload = evaluate_axial_localization(_axial_tests._token_source())
    _, _time_payload = evaluate_axial_localization(_axial_tests._time_source())
    import dataclasses as _dc

    def _axial_run(bound_hypotheses: object, axial: object) -> str:
        from latent_anything._diagnostic_workflow import StageInvocation as _AxialInvocation
        from latent_anything._diagnostic_workflow import StageOutput as _AxialStage

        _executor = make_feature_explain_executor(
            bound_hypotheses,  # type: ignore[arg-type]
            _inputs(**{h.method: bundles[h.method] for h in bound_hypotheses}),  # type: ignore[union-attr]
        )
        _prior = (
            _AxialStage(stage="capture", outcome="completed", payload={"capture_id": "c"}, artifact_refs=()),
            _AxialStage(stage="detect", outcome="completed", payload=_axial_detect_payload(), artifact_refs=()),
            _AxialStage(stage="localize", outcome="completed", payload=dict(axial), artifact_refs=()),  # type: ignore[arg-type]
        )
        _produced = _executor(
            _AxialInvocation(
                stage="explain",
                request=request,
                manifest=manifest,
                prior=_prior,
                workflow_identity="w" * 64,
                request_digest="r" * 64,
                manifest_digest="m" * 64,
                config_digest="c" * 64,
            )
        )
        assert _produced.outcome == "completed"
        return "completed"

    def _axial_detect_payload() -> dict[str, object]:
        """Relabel a copy of the real detect payload to the axial family."""
        import copy as _copy

        relabeled = _copy.deepcopy(dict(feat_detect))
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

    def _axial_hypothesis(
        method: str,
        hypothesis_id: str,
        bindings: object,
        *,
        family_id: str = "sequence_trajectory_drift",
        metric_ids: object = ("axial-health",),
    ) -> ExplanationHypothesis:
        return _dc.replace(
            _hypothesis(method, hypothesis_id=hypothesis_id),
            representation_id=manifest_rep,
            dataset_id="wikitext-2-raw-v1-validation",
            manifest_id=manifest_id,
            family_id=family_id,
            metric_ids=metric_ids,  # type: ignore[arg-type]
            thresholds=tuple((metric, ">=", 0.1) for metric in metric_ids),  # type: ignore[union-attr]
            localization_bindings=bindings,  # type: ignore[arg-type]
        )

    assert (
        _axial_run(
            (_axial_hypothesis("sae_sparse", "hyp-axial-ckpt", (("checkpoint", "ckpt-1"),)),), dict(_axial_payload)
        )
        == "completed"
    )
    assert (
        _axial_run((_axial_hypothesis("lens", "hyp-axial-tok", (("token", "tok-2"),)),), dict(_token_payload))
        == "completed"
    )
    assert (
        _axial_run((_axial_hypothesis("geometry", "hyp-axial-time", (("time", "step-2"),)),), dict(_time_payload))
        == "completed"
    )
    with __import__("pytest").raises(StageContractError, match="detect family"):
        _axial_run(
            (
                _axial_hypothesis(
                    "sae_sparse",
                    "hyp-axial-wrong-fam",
                    (("checkpoint", "ckpt-1"),),
                    family_id="separability_probe_leakage",
                ),
            ),
            dict(_axial_payload),
        )
    _relabeled_axial = dict(_axial_payload)
    _relabeled_axial["family_id"] = "separability_probe_leakage"
    with __import__("pytest").raises(StageContractError, match="localize family"):
        _axial_run(
            (_axial_hypothesis("sae_sparse", "hyp-axial-loc-fam", (("checkpoint", "ckpt-1"),)),), _relabeled_axial
        )
    with __import__("pytest").raises(StageContractError, match="[Cc]heckpoint|[Ll]ocalization"):
        _axial_run(
            (_axial_hypothesis("sae_sparse", "hyp-bad-ckpt", (("checkpoint", "ckpt-FABRICATED"),)),),
            dict(_axial_payload),
        )
    with __import__("pytest").raises(StageContractError, match="[Tt]oken|[Ll]ocalization"):
        _axial_run((_axial_hypothesis("lens", "hyp-absent-tok", (("token", "tok-2"),)),), dict(_axial_payload))
    _not_applicable = dict(_axial_payload)
    _not_applicable["axes"] = [
        {"axis": "checkpoint", "status": "not_applicable", "affected": [], "earliest": None, "report_rows": []}
    ]
    with __import__("pytest").raises(StageContractError, match="[Ll]ocalization"):
        _axial_run((_axial_hypothesis("density", "hyp-na", (("checkpoint", "ckpt-1"),)),), _not_applicable)


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
        "evaluate_feature_explanations",
        "make_feature_explain_executor",
        "ExplanationHypothesis",
        "ControlPlan",
        "execute_plan",
    ):
        assert not hasattr(la, leaked)
    assert SUPPORTED_FEATURE_METHODS == ("sae_sparse", "lens", "geometry", "density", "clustering")
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    for name in (
        "benchmark_manifest_schema_v1.json",
        "diagnostic_report_schema_v1.json",
        "representation_problem_taxonomy_v1.json",
    ):
        assert (artifacts / name).exists()


def test_fail_closed_inputs_reject() -> None:
    with pytest.raises(ExplanationError, match="hypotheses must declare"):
        evaluate_feature_explanations((), _inputs(sae_sparse=_sae_bundle()))
    with pytest.raises(ExplanationError, match="inputs must be"):
        evaluate_feature_explanations((_hypothesis("sae_sparse"),), object())  # type: ignore[arg-type]
    with pytest.raises(ExplanationError, match="not a feature explanation method"):
        evaluate_feature_explanations((_hypothesis("probe"),), _inputs(sae_sparse=_sae_bundle()))
    with pytest.raises(ExplanationError, match="not a feature explanation method"):
        make_feature_explain_executor((_hypothesis("probe"),), _inputs())
    with pytest.raises(ExplanationError, match="context must be"):
        feature_explanation_payload(object())  # type: ignore[arg-type]
    with pytest.raises(ExplanationError, match="confidence_level"):
        evaluate_feature_explanations(
            (_hypothesis("sae_sparse"),),
            MethodInputs(),
            confidence_level=True,
        )
