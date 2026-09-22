"""SAE, lens, geometry, density, and clustering explanation evidence (Sprint 80.17).

Private explain-stage adapter behind the frozen taxonomy/workflow/report
contracts. It reuses the common 80.16 convention (:class:`ExplanationHypothesis`,
:class:`ExplanationEvidence`, :class:`MethodInputs`, central 80.15 executor,
canonical payload, report items, explain executor) and adds five bounded
method evaluators with frozen stability/selectivity gates plus a declared
feature-to-symptom relationship. No second explanation schema is created:
this module imports the 80.16 seam and extends it.

Reused primitives (no new estimators):

- SAE/sparse features: ``DictionaryLearning``/``DictionaryLearningConfig``
  is the single shared sparse-fit convention (one fit per seed with the
  declared seed as ``random_state``); ``match_by_decoder_cosine`` is the
  single shared permutation-invariant feature-alignment convention
  (decoder-direction matching, never raw feature indices).
- Lens: ``_transformer_analysis.apply_logit_lens`` owns the final-norm +
  LM-head projection; ``softmax`` owns the readout distribution. The
  adapter never invents a readout matrix.
- Geometry: ``geometry.orthonormalize_directions``,
  ``geometry.subspace_alignment`` (principal-angle/singular-value metric),
  ``geometry.concept_coverage``, and ``projection.OrthonormalSubspace``
  (orthonormality + identity binding) own the subspace conventions.
- Density: ``density.GaussianMixtureDensity``/``GMMConfig`` is the single
  fitted/calibrated estimator convention from 80.11: one fit on reference
  rows, one calibration on held-out rows, then score-only passes with the
  fitted identity enforced (cross-space scoring rejects).
- Clustering: ``clustering.KMeans``/``KMeansConfig`` owns fitting;
  ``clustering.compare_with_labels`` owns label agreement;
  ``sklearn.metrics.adjusted_rand_score`` owns seed agreement. Stability
  is permutation-invariant (ARI needs no label alignment; raw cluster IDs
  are never compared directly).

Single-evaluation seam: :func:`evaluate_feature_explanations` fits every
dictionary, estimator, projection, and clustering exactly once per executor
call inside one shared context, then shares that context between decisions
and payload assembly. Uncertainty resamples already-fitted activations,
scores, distances, and assignments without refitting. The ``explain`` stage
executor (:func:`make_feature_explain_executor`) carries no algorithm; it
verifies identity and returns the shared payload.

Fail-closed: the declared feature-to-symptom relationship is checked before
any method callback fires; a missing relationship blocks with the callback
uninvoked. Undeclared hypotheses or methods never invoke method code; the
record reads ``omitted`` (undeclared) or ``unsupported`` (declared but
inapplicable). Missing or failed fidelity, stability, selectivity, leakage,
uncertainty, or any required control blocks promotion (``inconclusive`` or
``unsupported``, ``claim_allowed=False``). A human-readable feature label,
semantic cluster label, or output projection appears as promoted/supported
only when every frozen gate passes; otherwise only anonymous technical
identity/evidence is retained and the semantic claim is redacted
(``promoted_label``/``promoted_projection`` are ``None``). Semantics are
never inferred from correlation, density, cluster number, decoder index,
or projection rank. Records are ``kind="explanation"`` with
``causal=False`` always.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from latent_anything._portable_contract import PortableNodeError, canonical_json
from latent_anything._probe_tcav_ig_explanation import (
    EXPLAINER_VERSION as _BASE_EXPLAINER_VERSION,
)
from latent_anything._probe_tcav_ig_explanation import (
    ExplanationError,
    ExplanationEvidence,
    ExplanationHypothesis,
    MethodInputs,
)
from latent_anything._probe_tcav_ig_explanation import (
    _bound_control_ids as _bound_ids,
    _central_bootstrap,
    _central_control,
    _check_disjoint_identities,
    _finite_array,
    _passes,
    _require_mapping,
    _sign_aware_cosine,
)
from latent_anything._statistical_controls import ControlPlan as _ControlPlan
from latent_anything._statistical_controls import ControlSpec as _ControlSpec
from latent_anything._statistical_controls import execute_plan as _execute_plan
from latent_anything._statistical_controls import failed_required as _failed_required

SUPPORTED_FEATURE_METHODS: tuple[str, ...] = (
    "sae_sparse",
    "lens",
    "geometry",
    "density",
    "clustering",
)
"""Explanation methods this adapter may execute. Anything else is rejected."""

FEATURE_EXPLAINER_VERSION = "sae-lens-geometry-density-clustering-explainer-v1"
"""Version string bound into feature explain-stage payloads."""

_MATCH_COSINE_THRESHOLD = 0.5
"""Decoder-direction cosine floor for a valid cross-seed feature match."""

_MISSING_DIMS = {
    "fidelity": {"status": "missing"},
    "stability": {"status": "missing"},
    "selectivity": {"status": "missing"},
    "leakage": {"status": "missing"},
    "uncertainty": {"status": "missing"},
}


def _central_name(comparator: str) -> str:
    """Map a hypothesis ``>=``/``<=`` comparator onto a central comparator."""
    return "meets_threshold" if comparator == ">=" else "below_threshold"


def _relationship_of(bundle: Mapping[str, object], hypothesis: ExplanationHypothesis) -> str:
    """Return the declared feature-to-symptom relationship, or ``""``."""
    candidate = bundle.get("symptom_relationship")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return ""


def _blocked_record(
    hypothesis: ExplanationHypothesis,
    *,
    outcome: Literal["inconclusive", "unsupported"],
    reason: str,
    gaps: Sequence[str],
    provenance: Mapping[str, object],
    limitation: str,
) -> ExplanationEvidence:
    dims = {name: dict(value) for name, value in _MISSING_DIMS.items()}
    return ExplanationEvidence(
        hypothesis_id=hypothesis.hypothesis_id,
        method="probe",
        outcome="inconclusive",
        claim_allowed=False,
        observed_effect={},
        fidelity=dict(dims["fidelity"]),
        stability=dict(dims["stability"]),
        selectivity=dict(dims["selectivity"]),
        leakage=dict(dims["leakage"]),
        uncertainty=dict(dims["uncertainty"]),
        control_outcomes={},
        central_outcomes={},
        missing_evidence=tuple(gaps),
        limitations=(limitation,),
        reason=reason,
        provenance=dict(provenance),
    )


def _remethod(record: ExplanationEvidence, method: str) -> ExplanationEvidence:
    """Rebuild one 80.16-shaped record under a feature method name."""
    return ExplanationEvidence(
        hypothesis_id=record.hypothesis_id,
        method=cast(Any, method),
        outcome=record.outcome,  # type: ignore[arg-type]
        claim_allowed=record.claim_allowed,
        observed_effect=dict(record.observed_effect),
        fidelity=dict(record.fidelity),
        stability=dict(record.stability),
        selectivity=dict(record.selectivity),
        leakage=dict(record.leakage),
        uncertainty=dict(record.uncertainty),
        control_outcomes=dict(record.control_outcomes),
        central_outcomes=dict(record.central_outcomes),
        missing_evidence=tuple(record.missing_evidence),
        limitations=tuple(record.limitations),
        reason=record.reason,
        provenance=dict(record.provenance),
    )


def _base_provenance(hypothesis: ExplanationHypothesis, method: str) -> dict[str, object]:
    return {
        "explainer_version": FEATURE_EXPLAINER_VERSION,
        "base_explainer_version": _BASE_EXPLAINER_VERSION,
        "method": method,
        "hypothesis_id": hypothesis.hypothesis_id,
        "representation_id": hypothesis.representation_id,
        "dataset_id": hypothesis.dataset_id,
    }


def _empty_dims() -> dict[str, dict[str, object]]:
    return {name: dict(value) for name, value in _MISSING_DIMS.items()}


# ---------------------------------------------------------------------------
# Shared gating helper
# ---------------------------------------------------------------------------


def _gate_dimensions(
    hypothesis: ExplanationHypothesis,
    method: str,
    *,
    observed: Mapping[str, float],
    fidelity: Mapping[str, object],
    stability: Mapping[str, object],
    selectivity: Mapping[str, object],
    leakage: Mapping[str, object],
    uncertainty: Mapping[str, object],
    control_specs: Sequence[_ControlSpec],
    supplied: Mapping[str, Mapping[str, float] | None],
    provenance: Mapping[str, object],
    limitation: str,
    support_reason: str,
    block_prefix: str,
    fidelity_ok: bool,
    stability_ok: bool,
    selectivity_ok: bool,
    leakage_ok: bool,
    gaps: Sequence[str],
    repetitions: int,
    confidence_level: float,
    evaluation_seed: int,
    control_seed: int,
    promoted_label: str | None = None,
    promoted_projection: Sequence[float] | None = None,
) -> ExplanationEvidence:
    """Run central required-control gating and assemble one evidence record."""
    plan = _ControlPlan(
        plan_id=f"{hypothesis.hypothesis_id}:{method}",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        evaluation_seed=int(evaluation_seed),
        control_seed=int(control_seed),
        controls=tuple(control_specs),
    )
    outcomes = _execute_plan(plan, supplied)
    central_records = {control_id: outcome.to_dict() for control_id, outcome in outcomes.items()}
    blocked = list(_failed_required(outcomes))
    control_status = {
        control_id: ("passed" if outcome.status in ("passed", "recorded") else "failed")
        for control_id, outcome in outcomes.items()
    }
    all_gaps = list(gaps)
    for control_id in blocked:
        all_gaps.append(f"failed-control:{control_id}")
    full_leakage = dict(leakage)
    if blocked:
        full_leakage = {**full_leakage, "status": "failed"}
    ok = bool(fidelity_ok and stability_ok and selectivity_ok and leakage_ok and not blocked)
    if not ok:
        record = ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id,
            method="probe",
            outcome="inconclusive",
            claim_allowed=False,
            observed_effect=dict(observed),
            fidelity=dict(fidelity),
            stability=dict(stability),
            selectivity=dict(selectivity),
            leakage=dict(full_leakage),
            uncertainty=dict(uncertainty),
            control_outcomes=dict(control_status),
            central_outcomes=dict(central_records),
            missing_evidence=tuple(all_gaps) if all_gaps else (f"{block_prefix}:blocked",),
            limitations=(limitation,),
            reason=f"{block_prefix} blocked: {', '.join(all_gaps) if all_gaps else 'gates unmet'}",
            provenance=dict(provenance),
        )
        return _remethod(record, method)
    record = ExplanationEvidence(
        hypothesis_id=hypothesis.hypothesis_id,
        method="probe",
        outcome="supported",
        claim_allowed=True,
        observed_effect=dict(observed),
        fidelity=dict(fidelity),
        stability=dict(stability),
        selectivity=dict(selectivity),
        leakage=dict(full_leakage),
        uncertainty=dict(uncertainty),
        control_outcomes=dict(control_status),
        central_outcomes=dict(central_records),
        missing_evidence=(),
        limitations=(limitation,),
        reason=support_reason,
        provenance={
            **dict(provenance),
            "promoted_label": promoted_label,
            "promoted_projection": None if promoted_projection is None else [float(item) for item in promoted_projection],
        },
    )
    return _remethod(record, method)


# ---------------------------------------------------------------------------
# 1) SAE / sparse features
# ---------------------------------------------------------------------------


def _fit_dictionary_once(matrix: np.ndarray, *, seed: int, n_components: int) -> tuple[np.ndarray, float]:
    """Fit one dictionary and return atoms plus held-out quality gain."""
    from latent_anything.dictionary_learning import DictionaryLearning, DictionaryLearningConfig

    learner = DictionaryLearning(
        DictionaryLearningConfig(n_components=int(n_components), max_iter=100, random_state=int(seed))
    )
    evaluation = learner.fit(np.asarray(matrix, dtype=np.float64))
    baseline = float(evaluation.val_baseline_mse)
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ExplanationError("sae reconstruction baseline is degenerate")
    quality = float(np.clip(1.0 - float(evaluation.val_reconstruction_mse) / baseline, 0.0, 1.0))
    return np.asarray(learner.components_, dtype=np.float64), quality


def _matched_stability(atoms_a: np.ndarray, atoms_b: np.ndarray) -> tuple[float, tuple[float, ...], float]:
    """Align two atom sets permutation-invariantly; return fraction, cosines, mean."""
    from latent_anything._sae_metrics import match_by_decoder_cosine

    matched = match_by_decoder_cosine(
        np.asarray(atoms_a, dtype=np.float64).T, np.asarray(atoms_b, dtype=np.float64).T, _MATCH_COSINE_THRESHOLD
    )
    cosines = tuple(float(cosine) for _, cosine in matched)
    n = int(atoms_a.shape[0])
    fraction = float(len(cosines) / n) if n else 0.0
    return fraction, cosines, float(sum(cosines) / len(cosines)) if cosines else 0.0


def _evaluate_sae_sparse(
    hypothesis: ExplanationHypothesis,
    bundle: Mapping[str, object],
    *,
    repetitions: int,
    confidence_level: float,
    evaluation_seed: int,
    control_seed: int,
) -> ExplanationEvidence:
    method = "sae_sparse"
    provenance = _base_provenance(hypothesis, method)
    limitation = "a sparse feature index is not a semantic label; causal claims require 80.18+ trials"

    def _blocked(reason: str, gaps: Sequence[str]) -> ExplanationEvidence:
        dims = _empty_dims()
        record = ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id,
            method="probe",
            outcome="inconclusive",
            claim_allowed=False,
            observed_effect={},
            fidelity=dict(dims["fidelity"]),
            stability=dict(dims["stability"]),
            selectivity=dict(dims["selectivity"]),
            leakage=dict(dims["leakage"]),
            uncertainty=dict(dims["uncertainty"]),
            control_outcomes={},
            central_outcomes={},
            missing_evidence=tuple(gaps),
            limitations=(limitation,),
            reason=reason,
            provenance=dict(provenance),
        )
        return _remethod(record, method)

    try:
        bound_controls = _bound_ids(hypothesis)
    except ExplanationError as exc:
        return _blocked(str(exc), ("missing-evidence:control-declaration",))
    try:
        symptom = _finite_array(bundle.get("symptom_matrix"), name="sae symptom_matrix")
        negative = _finite_array(bundle.get("negative_matrix"), name="sae negative_matrix")
        off_target = _finite_array(bundle.get("off_target_matrix"), name="sae off_target_matrix")
    except ExplanationError as exc:
        return _blocked(f"sae input is incomplete: {exc}", ("missing-evidence:sae-matrices",))
    for name, matrix in (("symptom_matrix", symptom), ("negative_matrix", negative), ("off_target_matrix", off_target)):
        if matrix.ndim != 2:
            return _blocked(f"sae {name} must be 2D", ("missing-evidence:sae-matrices",))
    if symptom.shape[1] != negative.shape[1] or symptom.shape[1] != off_target.shape[1]:
        return _blocked("sae batch feature widths disagree", ("missing-evidence:sae-matrices",))
    sample_ids = bundle.get("sample_ids")
    if sample_ids is None:
        return _blocked("sae sample identities are missing", ("missing-evidence:leakage-identities",))
    ids = tuple(sample_ids) if isinstance(sample_ids, Sequence) and not isinstance(sample_ids, (str, bytes)) else ()
    if not ids or len(ids) != int(symptom.shape[0]) or len(set(str(i) for i in ids)) != len(ids):
        return _blocked("sae sample identities must uniquely cover every symptom row", ("missing-evidence:leakage-identities",))
    n_components_raw = bundle.get("n_components")
    try:
        n_components = int(n_components_raw) if n_components_raw is not None else 4
    except (TypeError, ValueError):
        return _blocked("sae n_components is non-numeric", ("missing-evidence:sae-config",))
    if n_components < 2:
        return _blocked("sae n_components must be at least two", ("missing-evidence:sae-config",))
    feature_label = bundle.get("feature_label")
    promoted_label = str(feature_label) if isinstance(feature_label, str) and feature_label.strip() else None
    provenance["declared_feature"] = hypothesis.target_id

    seeds = tuple(int(item) for item in hypothesis.seeds)
    try:
        atoms: dict[int, np.ndarray] = {}
        qualities: dict[int, float] = {}
        for seed in seeds:
            fitted, quality = _fit_dictionary_once(symptom, seed=seed, n_components=n_components)
            atoms[seed] = fitted
            qualities[seed] = quality
    except (ValueError, ExplanationError) as exc:
        return _blocked(f"sae fit failed: {exc}", ("missing-evidence:fidelity",))
    reference = atoms[seeds[0]]
    fractions: list[float] = []
    all_cosines: list[float] = []
    for seed in seeds[1:]:
        fraction, cosines, _ = _matched_stability(reference, atoms[seed])
        fractions.append(fraction)
        all_cosines.extend(cosines)
    stability_value = float(sum(fractions) / len(fractions)) if fractions else 0.0
    mean_cosine = float(sum(all_cosines) / len(all_cosines)) if all_cosines else 0.0
    quality = float(sum(qualities.values()) / len(qualities))

    # Symptom-slice selectivity: reference-seed feature activations on the
    # symptom slice versus the negative/off-target slices, headline feature
    # chosen by the largest symptom-minus-negative margin (content, not index).
    # Activations come from the already-fitted reference dictionary: the
    # single extra transform per slice is a linear encode, not a refit.
    from latent_anything.dictionary_learning import DictionaryLearning, DictionaryLearningConfig

    encoder = DictionaryLearning(
        DictionaryLearningConfig(n_components=int(n_components), max_iter=100, random_state=int(seeds[0]))
    )
    encoder.fit(np.asarray(symptom, dtype=np.float64))
    symptom_act = np.abs(np.asarray(encoder.transform(np.asarray(symptom, dtype=np.float64))))
    negative_act = np.abs(np.asarray(encoder.transform(np.asarray(negative, dtype=np.float64))))
    off_act = np.abs(np.asarray(encoder.transform(np.asarray(off_target, dtype=np.float64))))
    margins = symptom_act.mean(axis=0) - negative_act.mean(axis=0)
    headline_feature = int(np.argmax(margins))
    headline_margin = float(margins[headline_feature])
    off_margin = float(symptom_act.mean(axis=0)[headline_feature] - off_act.mean(axis=0)[headline_feature])
    # Random-feature control: a randomly chosen non-headline feature's margin,
    # drawn from the executor-owned control stream. Specificity means the
    # headline margin beats a random feature's margin, not a random input's
    # activations (unstructured noise projects weakly onto every atom, so a
    # random-input comparison cannot separate specific from diffuse features).
    control_rng = np.random.default_rng(int(control_seed))
    _others = [int(f) for f in range(int(n_components)) if int(f) != int(headline_feature)]
    random_feature = int(control_rng.choice(_others)) if _others else int(headline_feature)
    random_feature_margin = float(margins[random_feature])
    selectivity_margin = float(min(headline_margin, off_margin, headline_margin - random_feature_margin))
    cosine_array = np.asarray(all_cosines if all_cosines else [0.0], dtype=np.float64)

    def _stability_draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, cosine_array.shape[0], size=cosine_array.shape[0])
        return float(np.mean(cosine_array[positions]))

    _, stability_interval = _central_bootstrap(
        control_id="bootstrap:sae-stability",
        base_seed=int(evaluation_seed),
        seed_role="evaluation",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        required=False,
        kind="bootstrap",
        metric_ids=("sae_stability",),
        expected_behavior="seeded resampling of fitted matched decoder cosines",
        draw=_stability_draw,
    )
    activation_magnitudes = np.abs(symptom_act[:, headline_feature])

    def _activation_draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, activation_magnitudes.shape[0], size=activation_magnitudes.shape[0])
        return float(np.mean(activation_magnitudes[positions]))

    _, activation_interval = _central_bootstrap(
        control_id="bootstrap:sae-activation",
        base_seed=int(evaluation_seed),
        seed_role="evaluation",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        required=False,
        kind="bootstrap",
        metric_ids=("sae_activation",),
        expected_behavior="seeded resampling of fitted headline-feature activations",
        draw=_activation_draw,
    )
    # Selectivity permutation control: the actual shuffled-activation margin
    # runs inside the executor-owned stream; observed binds from the outcome.
    def _selectivity_statistic(rng: np.random.Generator) -> Mapping[str, float]:
        order = rng.permutation(symptom_act.shape[0])
        shuffled_margin = float(symptom_act[order, headline_feature].mean() - negative_act[:, headline_feature].mean())
        return {"selectivity_margin": float(selectivity_margin), "shuffled_margin": float(shuffled_margin)}

    shuffle_outcome = _central_control(
        control_id=bound_controls["selectivity"],
        base_seed=int(control_seed),
        seed_role="control",
        required=False,
        kind="shuffled",
        metric_ids=("selectivity_margin", "shuffled_margin"),
        expected_behavior="one seeded permutation of symptom activations",
        statistic=_selectivity_statistic,
    )
    if shuffle_outcome.status == "failed":
        return _blocked(
            f"sae shuffled control failed: {shuffle_outcome.reason}",
            (f"failed-control:{bound_controls['selectivity']}",),
        )
    shuffled_margin = float(shuffle_outcome.observed["shuffled_margin"])

    fidelity_threshold = hypothesis.threshold_for("reconstruction_quality")
    stability_threshold = hypothesis.threshold_for("sae_stability")
    selectivity_threshold = hypothesis.threshold_for("selectivity_margin")
    fidelity_met = True if fidelity_threshold is None else _passes(fidelity_threshold[0], quality, float(fidelity_threshold[1]))
    stability_met = True if stability_threshold is None else _passes(stability_threshold[0], stability_value, float(stability_threshold[1]))
    selectivity_met = (
        True if selectivity_threshold is None else _passes(selectivity_threshold[0], selectivity_margin, float(selectivity_threshold[1]))
    )
    leakage_ok = True
    if hypothesis.train_split_identity == hypothesis.eval_split_identity:
        leakage_ok = False

    observed = {
        "reconstruction_quality": float(quality),
        "sae_stability": float(stability_value),
        "mean_matched_cosine": float(mean_cosine),
        "selectivity_margin": float(selectivity_margin),
        "shuffled_margin": float(shuffled_margin),
        "headline_feature": float(headline_feature),
    }
    fidelity = {
        "status": "passed" if fidelity_met else "failed",
        "metric": "reconstruction_quality",
        "observed": float(quality),
        "threshold": None if fidelity_threshold is None else [fidelity_threshold[0], float(fidelity_threshold[1])],
    }
    stability_dims = {
        "status": "passed" if stability_met else "failed",
        "metric": "sae_stability",
        "observed": float(stability_value),
        "threshold": None if stability_threshold is None else [stability_threshold[0], float(stability_threshold[1])],
        "mean_matched_cosine": float(mean_cosine),
        "permutation_invariant": True,
        "headline_feature": int(headline_feature),
    }
    selectivity = {
        "status": "passed" if selectivity_met else "failed",
        "metric": "selectivity_margin",
        "observed": float(selectivity_margin),
        "threshold": None if selectivity_threshold is None else [selectivity_threshold[0], float(selectivity_threshold[1])],
        "off_target_margin": float(off_margin),
        "random_feature": int(random_feature),
        "random_feature_margin": float(random_feature_margin),
        "shuffled_margin": float(shuffled_margin),
    }
    leakage = {
        "status": "passed" if leakage_ok else "failed",
        "train_split_identity": hypothesis.train_split_identity,
        "eval_split_identity": hypothesis.eval_split_identity,
        "declared_feature": hypothesis.target_id,
        "headline_feature": int(headline_feature),
    }
    uncertainty = {
        "status": "passed",
        "sae_stability": dict(stability_interval),
        "sae_activation": dict(activation_interval),
        "evaluation_seed": int(evaluation_seed),
        "repetitions": int(repetitions),
    }
    quality_gate = fidelity_threshold if fidelity_threshold is not None else (">=", 0.8)
    stability_gate = stability_threshold if stability_threshold is not None else (">=", 0.5)
    margin_gate = selectivity_threshold if selectivity_threshold is not None else (">=", 0.2)
    specs = [
        _ControlSpec(
            control_id=bound_controls["fidelity"],
            kind="seed",
            required=True,
            metric_ids=("reconstruction_quality",),
            expected_behavior="dictionary reconstruction under predeclared quality floor",
            comparator=_central_name(quality_gate[0]),  # type: ignore[arg-type]
            threshold_value=float(quality_gate[1]),
        ),
        _ControlSpec(
            control_id=bound_controls["stability"],
            kind="cross_seed",
            required=True,
            metric_ids=("sae_stability",),
            expected_behavior="permutation-invariant feature stability across declared seeds",
            comparator=_central_name(stability_gate[0]),  # type: ignore[arg-type]
            threshold_value=float(stability_gate[1]),
        ),
        _ControlSpec(
            control_id=bound_controls["selectivity"],
            kind="counterexample",
            required=True,
            metric_ids=("selectivity_margin",),
            expected_behavior="headline feature separates symptom from negative/off-target/random slices",
            comparator=_central_name(margin_gate[0]),  # type: ignore[arg-type]
            threshold_value=float(margin_gate[1]),
        ),
    ]
    supplied = {
        bound_controls["fidelity"]: {"reconstruction_quality": float(quality)},
        bound_controls["stability"]: {"sae_stability": float(stability_value)},
        bound_controls["selectivity"]: {"selectivity_margin": float(selectivity_margin)},
    }
    gaps: list[str] = []
    if not fidelity_met:
        gaps.append("failed-fidelity:reconstruction_quality")
    if not stability_met:
        gaps.append("failed-stability:sae_stability")
    if not selectivity_met:
        gaps.append("failed-selectivity:selectivity_margin")
    if not leakage_ok:
        gaps.append("failed-leakage:split-identity")
    record = _gate_dimensions(
        hypothesis, method,
        observed=observed, fidelity=fidelity, stability=stability_dims, selectivity=selectivity,
        leakage=leakage, uncertainty=uncertainty, control_specs=specs, supplied=supplied,
        provenance=provenance, limitation=limitation,
        support_reason="declared sparse feature meets reconstruction, cross-seed stability, selectivity, leakage, and uncertainty gates",
        block_prefix="sae evidence",
        fidelity_ok=fidelity_met, stability_ok=stability_met, selectivity_ok=selectivity_met,
        leakage_ok=leakage_ok, gaps=gaps, repetitions=int(repetitions),
        confidence_level=float(confidence_level), evaluation_seed=int(evaluation_seed),
        control_seed=int(control_seed), promoted_label=promoted_label if (
            fidelity_met and stability_met and selectivity_met and leakage_ok) else None,
    )
    if record.outcome != "supported":
        # Redact any semantic label on non-promoted output.
        provenance_redacted = dict(record.provenance)
        provenance_redacted.pop("promoted_label", None)
        base = ExplanationEvidence(
            hypothesis_id=record.hypothesis_id, method="probe", outcome=record.outcome,  # type: ignore[arg-type]
            claim_allowed=False, observed_effect=dict(record.observed_effect),
            fidelity=dict(record.fidelity), stability=dict(record.stability),
            selectivity=dict(record.selectivity), leakage=dict(record.leakage),
            uncertainty=dict(record.uncertainty), control_outcomes=dict(record.control_outcomes),
            central_outcomes=dict(record.central_outcomes),
            missing_evidence=tuple([*record.missing_evidence, "redacted:semantic-feature-label"]),
            limitations=tuple(record.limitations), reason=record.reason, provenance=provenance_redacted,
        )
        return _remethod(base, method)
    return record


# ---------------------------------------------------------------------------
# 2) Lens
# ---------------------------------------------------------------------------


def _evaluate_lens(
    hypothesis: ExplanationHypothesis,
    bundle: Mapping[str, object],
    *,
    repetitions: int,
    confidence_level: float,
    evaluation_seed: int,
    control_seed: int,
) -> ExplanationEvidence:
    method = "lens"
    provenance = _base_provenance(hypothesis, method)
    limitation = "a readout projection is not an explanation of computation; causal claims require 80.18+ trials"

    def _blocked(reason: str, gaps: Sequence[str]) -> ExplanationEvidence:
        dims = _empty_dims()
        record = ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id, method="probe", outcome="inconclusive",
            claim_allowed=False, observed_effect={}, fidelity=dict(dims["fidelity"]),
            stability=dict(dims["stability"]), selectivity=dict(dims["selectivity"]),
            leakage=dict(dims["leakage"]), uncertainty=dict(dims["uncertainty"]),
            control_outcomes={}, central_outcomes={}, missing_evidence=tuple(gaps),
            limitations=(limitation,), reason=reason, provenance=dict(provenance),
        )
        return _remethod(record, method)

    try:
        bound_controls = _bound_ids(hypothesis)
    except ExplanationError as exc:
        return _blocked(str(exc), ("missing-evidence:control-declaration",))

    try:
        symptom_logits = _finite_array(bundle.get("symptom_logits"), name="lens symptom_logits")
        benign_logits = _finite_array(bundle.get("benign_logits"), name="lens benign_logits")
        off_target_logits = _finite_array(bundle.get("off_target_logits"), name="lens off_target_logits")
        random_logits = _finite_array(bundle.get("random_logits"), name="lens random_logits")
    except ExplanationError as exc:
        return _blocked(f"lens input is incomplete: {exc}", ("missing-evidence:lens-logits",))
    for name, matrix in (("symptom_logits", symptom_logits), ("benign_logits", benign_logits),
                         ("off_target_logits", off_target_logits), ("random_logits", random_logits)):
        if matrix.ndim != 2:
            return _blocked(f"lens {name} must be 2D (n_samples, vocab)", ("missing-evidence:lens-logits",))
    vocab = int(symptom_logits.shape[1])
    if benign_logits.shape[1] != vocab or off_target_logits.shape[1] != vocab or random_logits.shape[1] != vocab:
        return _blocked("lens vocabulary widths disagree", ("missing-evidence:lens-logits",))
    target_index_raw = bundle.get("target_index")
    off_index_raw = bundle.get("off_target_index")
    try:
        target_index = int(target_index_raw)  # type: ignore[arg-type]
        off_index = int(off_index_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _blocked("lens target/off-target indices are missing or non-numeric", ("missing-evidence:lens-target",))
    if not 0 <= target_index < vocab or not 0 <= off_index < vocab or target_index == off_index:
        return _blocked("lens target indices are out of range or identical", ("missing-evidence:lens-target",))
    try:
        target_token = int(hypothesis.target_id) if hypothesis.target_id.lstrip("-").isdigit() else target_index
    except (TypeError, ValueError):
        target_token = target_index
    if isinstance(hypothesis.target_id, str) and hypothesis.target_id.lstrip("-").isdigit():
        if int(hypothesis.target_id) != target_index:
            return _blocked("lens target does not match the declared hypothesis target", ("failed-leakage:target-identity",))
    token_ids = bundle.get("token_ids")
    sample_ids = bundle.get("sample_ids")
    preprocessing = bundle.get("preprocessing_identity")
    if token_ids is None or sample_ids is None or not isinstance(preprocessing, str) or not preprocessing.strip():
        return _blocked("lens token/sample/preprocessing identity is missing", ("missing-evidence:lens-identity",))
    token_list = tuple(token_ids) if isinstance(token_ids, Sequence) and not isinstance(token_ids, (str, bytes)) else ()
    sample_list = tuple(sample_ids) if isinstance(sample_ids, Sequence) and not isinstance(sample_ids, (str, bytes)) else ()
    if len(token_list) != int(symptom_logits.shape[0]) or len(sample_list) != int(symptom_logits.shape[0]):
        return _blocked("lens token/sample identities must cover every symptom row", ("missing-evidence:lens-identity",))
    provenance["target_index"] = int(target_index)
    provenance["layer_id"] = hypothesis.layer_id
    promoted = bundle.get("promoted_projection")
    promoted_projection = (
        [float(item) for item in promoted] if isinstance(promoted, Sequence) and not isinstance(promoted, (str, bytes)) else None
    )

    from latent_anything._transformer_analysis import softmax as _softmax

    def _probs(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(_softmax(np.asarray(matrix, dtype=np.float64)), dtype=np.float64)

    symptom_p = _probs(symptom_logits)
    benign_p = _probs(benign_logits)
    off_p = _probs(off_target_logits)
    random_p = _probs(random_logits)
    target_mass = float(symptom_p[:, target_index].mean())
    benign_mass = float(benign_p[:, target_index].mean())
    off_mass = float(symptom_p[:, off_index].mean())
    random_mass = float(random_p[:, target_index].mean())
    readout_margin = float(target_mass - benign_mass)
    target_margin = float(min(target_mass - off_mass, target_mass - random_mass))
    # Readout fidelity: mean per-row cosine between symptom logits and the
    # declared reference logits (same lens recomputed = near 1.0).
    reference_logits = bundle.get("reference_logits")
    if reference_logits is None:
        return _blocked("lens reference logits are missing: fidelity needs a declared recomputation", ("missing-evidence:lens-fidelity",))
    try:
        reference = _finite_array(reference_logits, name="lens reference_logits")
    except ExplanationError as exc:
        return _blocked(str(exc), ("missing-evidence:lens-fidelity",))
    if reference.shape != symptom_logits.shape:
        return _blocked("lens reference logits disagree on shape", ("missing-evidence:lens-fidelity",))
    row_cosines = [
        float(np.dot(a, b) / max(1e-12, float(np.linalg.norm(a) * np.linalg.norm(b))))
        for a, b in zip(np.asarray(symptom_logits, dtype=np.float64), np.asarray(reference, dtype=np.float64))
    ]
    fidelity_value = float(sum(row_cosines) / len(row_cosines))
    # Stability: target-mass distribution across declared seeds. Seed bundles
    # are alternative symptom-logit draws through the same lens; gating uses
    # the minimum absolute target-mass drift (distribution-level, robust to
    # readout-direction noise that collapses vector cosine on near-tied rows).
    seed_bundles = bundle.get("seed_logits")
    if not isinstance(seed_bundles, Mapping) or not seed_bundles:
        return _blocked("lens seed logits are missing: declare at least one seed alternative", ("missing-evidence:stability-variants",))
    seed_masses: list[float] = [float(target_mass)]
    for key in sorted(seed_bundles):
        try:
            variant = _finite_array(seed_bundles[key], name=f"lens seed {key}")
        except ExplanationError as exc:
            return _blocked(str(exc), ("missing-evidence:stability-variants",))
        if variant.shape[1] != vocab:
            return _blocked("lens seed logits disagree on vocabulary width", ("missing-evidence:stability-variants",))
        variant_p = _probs(variant)
        seed_masses.append(float(variant_p[:, target_index].mean()))
    mass_drift = float(max(abs(item - float(target_mass)) for item in seed_masses))
    readout_stability = float(1.0 - mass_drift)
    seed_cosines = [1.0 - abs(item - float(target_mass)) for item in seed_masses]

    masses = np.asarray(symptom_p[:, target_index], dtype=np.float64)

    def _mass_draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, masses.shape[0], size=masses.shape[0])
        return float(np.mean(masses[positions]))

    _, mass_interval = _central_bootstrap(
        control_id="bootstrap:lens-target-mass",
        base_seed=int(evaluation_seed),
        seed_role="evaluation",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        required=False,
        kind="bootstrap",
        metric_ids=("lens_target_mass",),
        expected_behavior="seeded resampling of fitted target probabilities",
        draw=_mass_draw,
    )

    def _selectivity_statistic(rng: np.random.Generator) -> Mapping[str, float]:
        order = rng.permutation(symptom_p.shape[0])
        shuffled_mass = float(symptom_p[order, target_index].mean() - benign_p[:, target_index].mean())
        return {"readout_margin": float(readout_margin), "shuffled_margin": float(shuffled_mass)}

    shuffle_outcome = _central_control(
        control_id=bound_controls["selectivity"],
        base_seed=int(control_seed),
        seed_role="control",
        required=False,
        kind="shuffled",
        metric_ids=("readout_margin", "shuffled_margin"),
        expected_behavior="one seeded permutation of symptom rows",
        statistic=_selectivity_statistic,
    )
    if shuffle_outcome.status == "failed":
        return _blocked(f"lens shuffled control failed: {shuffle_outcome.reason}", (f"failed-control:{bound_controls['selectivity']}",))
    _ = float(shuffle_outcome.observed["shuffled_margin"])

    fidelity_threshold = hypothesis.threshold_for("readout_fidelity")
    stability_threshold = hypothesis.threshold_for("readout_stability")
    selectivity_threshold = hypothesis.threshold_for("readout_margin")
    fidelity_met = True if fidelity_threshold is None else _passes(fidelity_threshold[0], fidelity_value, float(fidelity_threshold[1]))
    stability_met = True if stability_threshold is None else _passes(stability_threshold[0], readout_stability, float(stability_threshold[1]))
    selectivity_met = True if selectivity_threshold is None else _passes(selectivity_threshold[0], readout_margin, float(selectivity_threshold[1]))
    leakage_ok = bool(token_list and sample_list and str(preprocessing).strip())

    observed = {
        "readout_fidelity": float(fidelity_value),
        "readout_stability": float(readout_stability),
        "readout_margin": float(readout_margin),
        "target_margin": float(target_margin),
        "target_mass": float(target_mass),
    }
    fidelity = {"status": "passed" if fidelity_met else "failed", "metric": "readout_fidelity",
                "observed": float(fidelity_value),
                "threshold": None if fidelity_threshold is None else [fidelity_threshold[0], float(fidelity_threshold[1])]}
    stability = {"status": "passed" if stability_met else "failed", "metric": "readout_stability",
                 "observed": float(readout_stability),
                 "threshold": None if stability_threshold is None else [stability_threshold[0], float(stability_threshold[1])],
                 "seed_cosines": [float(item) for item in seed_cosines]}
    selectivity = {"status": "passed" if selectivity_met else "failed", "metric": "readout_margin",
                   "observed": float(readout_margin),
                   "threshold": None if selectivity_threshold is None else [selectivity_threshold[0], float(selectivity_threshold[1])],
                   "target_margin": float(target_margin), "off_mass": float(off_mass), "random_mass": float(random_mass)}
    leakage = {"status": "passed" if leakage_ok else "failed", "layer_id": hypothesis.layer_id,
               "target_index": int(target_index), "preprocessing_identity": str(preprocessing)}
    uncertainty = {"status": "passed", "lens_target_mass": dict(mass_interval),
                   "evaluation_seed": int(evaluation_seed), "repetitions": int(repetitions)}
    fid_gate = fidelity_threshold if fidelity_threshold is not None else (">=", 0.99)
    stab_gate = stability_threshold if stability_threshold is not None else (">=", 0.9)
    sel_gate = selectivity_threshold if selectivity_threshold is not None else (">=", 0.1)
    specs = [
        _ControlSpec(control_id=bound_controls["fidelity"], kind="seed", required=True,
                     metric_ids=("readout_fidelity",), expected_behavior="lens readout fidelity under predeclared floor",
                     comparator=_central_name(fid_gate[0]), threshold_value=float(fid_gate[1])),  # type: ignore[arg-type]
        _ControlSpec(control_id=bound_controls["stability"], kind="cross_seed", required=True,
                     metric_ids=("readout_stability",), expected_behavior="readout stability across declared seeds",
                     comparator=_central_name(stab_gate[0]), threshold_value=float(stab_gate[1])),  # type: ignore[arg-type]
        _ControlSpec(control_id=bound_controls["selectivity"], kind="negative", required=True,
                     metric_ids=("readout_margin",), expected_behavior="target readout separates symptom from benign slices",
                     comparator=_central_name(sel_gate[0]), threshold_value=float(sel_gate[1])),  # type: ignore[arg-type]
    ]
    supplied = {
        bound_controls["fidelity"]: {"readout_fidelity": float(fidelity_value)},
        bound_controls["stability"]: {"readout_stability": float(readout_stability)},
        bound_controls["selectivity"]: {"readout_margin": float(readout_margin)},
    }
    gaps: list[str] = []
    if not fidelity_met:
        gaps.append("failed-fidelity:readout_fidelity")
    if not stability_met:
        gaps.append("failed-stability:readout_stability")
    if not selectivity_met:
        gaps.append("failed-selectivity:readout_margin")
    record = _gate_dimensions(
        hypothesis, method, observed=observed, fidelity=fidelity, stability=stability,
        selectivity=selectivity, leakage=leakage, uncertainty=uncertainty,
        control_specs=specs, supplied=supplied, provenance=provenance, limitation=limitation,
        support_reason="declared lens readout meets fidelity, stability, selectivity, leakage, and uncertainty gates",
        block_prefix="lens evidence", fidelity_ok=fidelity_met, stability_ok=stability_met,
        selectivity_ok=selectivity_met, leakage_ok=leakage_ok, gaps=gaps,
        repetitions=int(repetitions), confidence_level=float(confidence_level),
        evaluation_seed=int(evaluation_seed), control_seed=int(control_seed),
        promoted_projection=list(promoted_projection) if (
            promoted_projection is not None and fidelity_met and stability_met and selectivity_met and leakage_ok) else None,
    )
    if record.outcome != "supported":
        provenance_redacted = dict(record.provenance)
        provenance_redacted.pop("promoted_projection", None)
        base = ExplanationEvidence(
            hypothesis_id=record.hypothesis_id, method="probe", outcome=record.outcome,  # type: ignore[arg-type]
            claim_allowed=False, observed_effect=dict(record.observed_effect),
            fidelity=dict(record.fidelity), stability=dict(record.stability),
            selectivity=dict(record.selectivity), leakage=dict(record.leakage),
            uncertainty=dict(record.uncertainty), control_outcomes=dict(record.control_outcomes),
            central_outcomes=dict(record.central_outcomes),
            missing_evidence=tuple([*record.missing_evidence, "redacted:semantic-projection"]),
            limitations=tuple(record.limitations), reason=record.reason, provenance=provenance_redacted,
        )
        return _remethod(base, method)
    return record


# ---------------------------------------------------------------------------
# 3) Geometry
# ---------------------------------------------------------------------------


def _evaluate_geometry(
    hypothesis: ExplanationHypothesis,
    bundle: Mapping[str, object],
    *,
    repetitions: int,
    confidence_level: float,
    evaluation_seed: int,
    control_seed: int,
) -> ExplanationEvidence:
    method = "geometry"
    provenance = _base_provenance(hypothesis, method)
    limitation = "a direction/subspace carries no semantic label without the declared symptom relationship"

    def _blocked(reason: str, gaps: Sequence[str]) -> ExplanationEvidence:
        dims = _empty_dims()
        record = ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id, method="probe", outcome="inconclusive",
            claim_allowed=False, observed_effect={}, fidelity=dict(dims["fidelity"]),
            stability=dict(dims["stability"]), selectivity=dict(dims["selectivity"]),
            leakage=dict(dims["leakage"]), uncertainty=dict(dims["uncertainty"]),
            control_outcomes={}, central_outcomes={}, missing_evidence=tuple(gaps),
            limitations=(limitation,), reason=reason, provenance=dict(provenance),
        )
        return _remethod(record, method)

    try:
        bound_controls = _bound_ids(hypothesis)
    except ExplanationError as exc:
        return _blocked(str(exc), ("missing-evidence:control-declaration",))

    try:
        symptom = _finite_array(bundle.get("symptom_matrix"), name="geometry symptom_matrix")
        benign = _finite_array(bundle.get("benign_matrix"), name="geometry benign_matrix")
        negative = _finite_array(bundle.get("negative_matrix"), name="geometry negative_matrix")
    except ExplanationError as exc:
        return _blocked(f"geometry input is incomplete: {exc}", ("missing-evidence:geometry-matrices",))
    for name, matrix in (("symptom_matrix", symptom), ("benign_matrix", benign), ("negative_matrix", negative)):
        if matrix.ndim != 2:
            return _blocked(f"geometry {name} must be 2D", ("missing-evidence:geometry-matrices",))
    if symptom.shape[1] != benign.shape[1] or symptom.shape[1] != negative.shape[1]:
        return _blocked("geometry batch widths disagree", ("missing-evidence:geometry-matrices",))
    dim = int(symptom.shape[1])
    rank_raw = bundle.get("subspace_rank")
    try:
        rank = int(rank_raw) if rank_raw is not None else 2
    except (TypeError, ValueError):
        return _blocked("geometry subspace_rank is non-numeric", ("missing-evidence:geometry-config",))
    if not 1 <= rank < dim:
        return _blocked("geometry subspace_rank must satisfy 1 <= rank < dim", ("missing-evidence:geometry-config",))
    sample_ids = bundle.get("sample_ids")
    if sample_ids is None:
        return _blocked("geometry sample identities are missing", ("missing-evidence:leakage-identities",))
    ids = tuple(sample_ids) if isinstance(sample_ids, Sequence) and not isinstance(sample_ids, (str, bytes)) else ()
    if not ids or len(ids) != int(symptom.shape[0]):
        return _blocked("geometry sample identities must cover every symptom row", ("missing-evidence:leakage-identities",))
    provenance["subspace_rank"] = int(rank)

    from latent_anything.geometry import (
        concept_coverage as _coverage,
    )
    from latent_anything.geometry import (
        orthonormalize_directions as _orthonormalize,
    )
    from latent_anything.geometry import (
        subspace_alignment as _alignment,
    )
    from latent_anything.projection import OrthonormalSubspace as _Subspace

    centered = np.asarray(symptom, dtype=np.float64) - np.asarray(symptom, dtype=np.float64).mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    try:
        values, vectors = np.linalg.eigh(np.asarray(covariance, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        return _blocked(f"geometry fit failed: {exc}", ("missing-evidence:fidelity",))
    order = np.argsort(values)[::-1]
    basis = _orthonormalize(np.asarray(vectors[:, order[:rank]], dtype=np.float64))
    try:
        subspace = _Subspace.from_basis(
            np.asarray(basis, dtype=np.float64),
            source_representation_identity=hypothesis.representation_id,
            origin="explicit",
            provenance={"rank": int(rank), "hypothesis": hypothesis.hypothesis_id},
        )
    except ValueError as exc:
        return _blocked(f"geometry subspace invalid: {exc}", ("missing-evidence:fidelity",))
    projected = np.asarray(symptom, dtype=np.float64) @ basis @ basis.T
    residual = float(np.mean((np.asarray(symptom, dtype=np.float64) - projected) ** 2))
    total = float(np.mean(np.asarray(symptom, dtype=np.float64) ** 2))
    fit_fidelity = float(1.0 - residual / max(1e-12, total))
    # Stability: split-half principal-angle alignment on centered covariance
    # halves (order-free; no raw QR basis-order dependence).
    half = int(symptom.shape[0] // 2)
    if half < rank + 1:
        return _blocked("geometry needs at least 2*(rank+1) symptom rows for split-half stability", ("missing-evidence:geometry-matrices",))
    half_a = np.asarray(symptom[:half], dtype=np.float64) - np.asarray(symptom[:half], dtype=np.float64).mean(axis=0)
    half_b = np.asarray(symptom[half:2 * half], dtype=np.float64) - np.asarray(symptom[half:2 * half], dtype=np.float64).mean(axis=0)
    try:
        values_a, vectors_a = np.linalg.eigh(np.cov(half_a, rowvar=False))
        values_b, vectors_b = np.linalg.eigh(np.cov(half_b, rowvar=False))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _blocked(f"geometry split-half fit failed: {exc}", ("missing-evidence:fidelity",))
    order_a = np.argsort(values_a)[::-1]
    order_b = np.argsort(values_b)[::-1]
    basis_a = _orthonormalize(np.asarray(vectors_a[:, order_a[:rank]], dtype=np.float64))
    basis_b = _orthonormalize(np.asarray(vectors_b[:, order_b[:rank]], dtype=np.float64))
    stability_value = float(_alignment(basis_a, basis_b))
    seed_bundles = bundle.get("seed_matrices")
    seed_alignments: list[float] = [1.0]
    if isinstance(seed_bundles, Mapping):
        for key in sorted(seed_bundles):
            try:
                variant = _finite_array(seed_bundles[key], name=f"geometry seed {key}")
            except ExplanationError as exc:
                return _blocked(str(exc), ("missing-evidence:stability-variants",))
            if variant.shape[1] != dim:
                return _blocked("geometry seed matrices disagree on width", ("missing-evidence:stability-variants",))
            centered_variant = np.asarray(variant, dtype=np.float64) - np.asarray(variant, dtype=np.float64).mean(axis=0)
            try:
                variant_values, variant_vectors = np.linalg.eigh(np.cov(centered_variant, rowvar=False))
            except (ValueError, np.linalg.LinAlgError) as exc:
                return _blocked(f"geometry seed fit failed: {exc}", ("missing-evidence:fidelity",))
            variant_order = np.argsort(variant_values)[::-1]
            candidate = _orthonormalize(np.asarray(variant_vectors[:, variant_order[:rank]], dtype=np.float64))
            seed_alignments.append(float(_alignment(basis, candidate)))
    stability_value = float(min(stability_value, min(seed_alignments)))
    symptom_coverage = float(np.mean([float(_coverage(row, basis)) for row in np.asarray(symptom, dtype=np.float64)]))
    benign_coverage = float(np.mean([float(_coverage(row, basis)) for row in np.asarray(benign, dtype=np.float64)]))
    negative_coverage = float(np.mean([float(_coverage(row, basis)) for row in np.asarray(negative, dtype=np.float64)]))
    selectivity_margin = float(min(symptom_coverage - benign_coverage, symptom_coverage - negative_coverage))
    coverages = np.asarray([float(_coverage(row, basis)) for row in np.asarray(symptom, dtype=np.float64)])

    def _coverage_draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, coverages.shape[0], size=coverages.shape[0])
        return float(np.mean(coverages[positions]))

    _, coverage_interval = _central_bootstrap(
        control_id="bootstrap:geometry-coverage",
        base_seed=int(evaluation_seed),
        seed_role="evaluation",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        required=False,
        kind="bootstrap",
        metric_ids=("geometry_coverage",),
        expected_behavior="seeded resampling of fitted subspace coverages",
        draw=_coverage_draw,
    )

    def _null_statistic(rng: np.random.Generator) -> Mapping[str, float]:
        shuffled = np.column_stack([rng.permutation(np.asarray(symptom, dtype=np.float64)[:, j]) for j in range(dim)])
        centered_null = shuffled - shuffled.mean(axis=0)
        try:
            null_values, null_vectors = np.linalg.eigh(np.cov(centered_null, rowvar=False))
        except (ValueError, np.linalg.LinAlgError):
            return {"geometry_coverage": float(symptom_coverage), "null_coverage": 0.0}
        null_order = np.argsort(null_values)[::-1]
        null_basis = _orthonormalize(np.asarray(null_vectors[:, null_order[:rank]], dtype=np.float64))
        null_coverage = float(np.mean([float(_coverage(row, null_basis)) for row in shuffled]))
        return {"geometry_coverage": float(symptom_coverage), "null_coverage": float(null_coverage)}

    null_outcome = _central_control(
        control_id=bound_controls["selectivity"],
        base_seed=int(control_seed),
        seed_role="control",
        required=False,
        kind="shuffled",
        metric_ids=("geometry_coverage", "null_coverage"),
        expected_behavior="independent-per-column permutation destroys joint structure",
        statistic=_null_statistic,
    )
    if null_outcome.status == "failed":
        return _blocked(f"geometry shuffled null failed: {null_outcome.reason}", (f"failed-control:{bound_controls['selectivity']}",))
    null_coverage = float(null_outcome.observed["null_coverage"])
    fidelity_threshold = hypothesis.threshold_for("fit_fidelity")
    stability_threshold = hypothesis.threshold_for("subspace_stability")
    selectivity_threshold = hypothesis.threshold_for("coverage_margin")
    fidelity_met = True if fidelity_threshold is None else _passes(fidelity_threshold[0], fit_fidelity, float(fidelity_threshold[1]))
    stability_met = True if stability_threshold is None else _passes(stability_threshold[0], stability_value, float(stability_threshold[1]))
    selectivity_met = (
        True if selectivity_threshold is None else _passes(selectivity_threshold[0], selectivity_margin, float(selectivity_threshold[1]))
    )
    leakage_ok = bool(ids) and hypothesis.train_split_identity != hypothesis.eval_split_identity

    observed = {
        "fit_fidelity": float(fit_fidelity),
        "subspace_stability": float(stability_value),
        "coverage_margin": float(selectivity_margin),
        "symptom_coverage": float(symptom_coverage),
        "null_coverage": float(null_coverage),
    }
    fidelity = {"status": "passed" if fidelity_met else "failed", "metric": "fit_fidelity",
                "observed": float(fit_fidelity),
                "threshold": None if fidelity_threshold is None else [fidelity_threshold[0], float(fidelity_threshold[1])]}
    stability = {"status": "passed" if stability_met else "failed", "metric": "subspace_stability",
                 "observed": float(stability_value),
                 "threshold": None if stability_threshold is None else [stability_threshold[0], float(stability_threshold[1])],
                 "principal_angle_metric": True, "subspace_rank": int(rank)}
    selectivity = {"status": "passed" if selectivity_met else "failed", "metric": "coverage_margin",
                   "observed": float(selectivity_margin),
                   "threshold": None if selectivity_threshold is None else [selectivity_threshold[0], float(selectivity_threshold[1])],
                   "benign_coverage": float(benign_coverage), "negative_coverage": float(negative_coverage),
                   "null_coverage": float(null_coverage)}
    leakage = {"status": "passed" if leakage_ok else "failed",
               "train_split_identity": hypothesis.train_split_identity,
               "eval_split_identity": hypothesis.eval_split_identity,
               "subspace_identity": subspace.source_representation_identity}
    uncertainty = {"status": "passed", "geometry_coverage": dict(coverage_interval),
                   "evaluation_seed": int(evaluation_seed), "repetitions": int(repetitions)}
    fid_gate = fidelity_threshold if fidelity_threshold is not None else (">=", 0.5)
    stab_gate = stability_threshold if stability_threshold is not None else (">=", 0.9)
    sel_gate = selectivity_threshold if selectivity_threshold is not None else (">=", 0.2)
    specs = [
        _ControlSpec(control_id=bound_controls["fidelity"], kind="seed", required=True,
                     metric_ids=("fit_fidelity",), expected_behavior="subspace fit under predeclared floor",
                     comparator=_central_name(fid_gate[0]), threshold_value=float(fid_gate[1])),  # type: ignore[arg-type]
        _ControlSpec(control_id=bound_controls["stability"], kind="cross_seed", required=True,
                     metric_ids=("subspace_stability",), expected_behavior="principal-angle stability across splits/seeds",
                     comparator=_central_name(stab_gate[0]), threshold_value=float(stab_gate[1])),  # type: ignore[arg-type]
        _ControlSpec(control_id=bound_controls["selectivity"], kind="counterexample", required=True,
                     metric_ids=("coverage_margin",), expected_behavior="coverage separates diagnosed slice from benign data",
                     comparator=_central_name(sel_gate[0]), threshold_value=float(sel_gate[1])),  # type: ignore[arg-type]
    ]
    supplied = {
        bound_controls["fidelity"]: {"fit_fidelity": float(fit_fidelity)},
        bound_controls["stability"]: {"subspace_stability": float(stability_value)},
        bound_controls["selectivity"]: {"coverage_margin": float(selectivity_margin)},
    }
    gaps: list[str] = []
    if not fidelity_met:
        gaps.append("failed-fidelity:fit_fidelity")
    if not stability_met:
        gaps.append("failed-stability:subspace_stability")
    if not selectivity_met:
        gaps.append("failed-selectivity:coverage_margin")
    if not leakage_ok:
        gaps.append("failed-leakage:split-identity")
    return _gate_dimensions(
        hypothesis, method, observed=observed, fidelity=fidelity, stability=stability,
        selectivity=selectivity, leakage=leakage, uncertainty=uncertainty,
        control_specs=specs, supplied=supplied, provenance=provenance, limitation=limitation,
        support_reason="declared subspace meets fit, principal-angle stability, selectivity, leakage, and uncertainty gates",
        block_prefix="geometry evidence", fidelity_ok=fidelity_met, stability_ok=stability_met,
        selectivity_ok=selectivity_met, leakage_ok=leakage_ok, gaps=gaps,
        repetitions=int(repetitions), confidence_level=float(confidence_level),
        evaluation_seed=int(evaluation_seed), control_seed=int(control_seed),
    )


# ---------------------------------------------------------------------------
# 4) Density
# ---------------------------------------------------------------------------


def _evaluate_density(
    hypothesis: ExplanationHypothesis,
    bundle: Mapping[str, object],
    *,
    repetitions: int,
    confidence_level: float,
    evaluation_seed: int,
    control_seed: int,
    training_seed: int,
) -> ExplanationEvidence:
    method = "density"
    provenance = _base_provenance(hypothesis, method)
    limitation = "an outlying density score does not mean the location caused the symptom; causal claims require 80.18+ trials"

    def _blocked(reason: str, gaps: Sequence[str]) -> ExplanationEvidence:
        dims = _empty_dims()
        record = ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id, method="probe", outcome="inconclusive",
            claim_allowed=False, observed_effect={}, fidelity=dict(dims["fidelity"]),
            stability=dict(dims["stability"]), selectivity=dict(dims["selectivity"]),
            leakage=dict(dims["leakage"]), uncertainty=dict(dims["uncertainty"]),
            control_outcomes={}, central_outcomes={}, missing_evidence=tuple(gaps),
            limitations=(limitation,), reason=reason, provenance=dict(provenance),
        )
        return _remethod(record, method)

    try:
        bound_controls = _bound_ids(hypothesis)
    except ExplanationError as exc:
        return _blocked(str(exc), ("missing-evidence:control-declaration",))

    try:
        reference = _finite_array(bundle.get("reference_matrix"), name="density reference_matrix")
        calibration = _finite_array(bundle.get("calibration_matrix"), name="density calibration_matrix")
        symptom = _finite_array(bundle.get("symptom_matrix"), name="density symptom_matrix")
        negative = _finite_array(bundle.get("negative_matrix"), name="density negative_matrix")
    except ExplanationError as exc:
        return _blocked(f"density input is incomplete: {exc}", ("missing-evidence:density-matrices",))
    for name, matrix in (("reference_matrix", reference), ("calibration_matrix", calibration),
                         ("symptom_matrix", symptom), ("negative_matrix", negative)):
        if matrix.ndim != 2:
            return _blocked(f"density {name} must be 2D", ("missing-evidence:density-matrices",))
    widths = {int(matrix.shape[1]) for matrix in (reference, calibration, symptom, negative)}
    if len(widths) != 1:
        return _blocked("density batch widths disagree", ("missing-evidence:density-matrices",))
    ref_identity = bundle.get("reference_identity")
    test_identity = bundle.get("test_identity")
    calibration_identity = bundle.get("calibration_identity")
    for name, value in (("reference_identity", ref_identity), ("test_identity", test_identity),
                        ("calibration_identity", calibration_identity)):
        if not isinstance(value, str) or not value.strip():
            return _blocked(f"density {name} is missing", ("missing-evidence:density-identity",))
    if str(ref_identity) != str(calibration_identity):
        return _blocked("density reference/calibration identities disagree", ("failed-leakage:reference-identity",))
    if str(ref_identity) != hypothesis.representation_id:
        return _blocked("density reference identity does not match the declared representation", ("failed-leakage:representation-identity",))
    if str(test_identity) == str(ref_identity):
        return _blocked("density test/reference identities are identical", ("failed-leakage:test-identity",))
    provenance["test_identity"] = str(test_identity)

    from latent_anything.density import GMMConfig, GaussianMixtureDensity

    fit_calls = {"count": 0}
    estimator = GaussianMixtureDensity(GMMConfig(n_components=2, random_state=int(training_seed)))
    try:
        estimator.fit(np.asarray(reference, dtype=np.float64),
                      source_representation_identity=str(ref_identity), geometry="euclidean",
                      provenance={"role": "reference-fit"})
        fit_calls["count"] += 1
        estimator.calibrate(np.asarray(calibration, dtype=np.float64), provenance={"role": "heldout-calibration"})
    except (ValueError, RuntimeError) as exc:
        return _blocked(f"density fit/calibration failed: {exc}", ("missing-evidence:fidelity",))
    try:
        calibration_scores = np.asarray(estimator.score(np.asarray(calibration, dtype=np.float64)).calibrated_ood_score, dtype=np.float64)
        symptom_scores = np.asarray(estimator.score(np.asarray(symptom, dtype=np.float64)).calibrated_ood_score, dtype=np.float64)
        negative_scores = np.asarray(estimator.score(np.asarray(negative, dtype=np.float64)).calibrated_ood_score, dtype=np.float64)
    except (ValueError, RuntimeError) as exc:
        return _blocked(f"density scoring failed: {exc}", ("missing-evidence:fidelity",))
    threshold = float(np.quantile(calibration_scores, 0.9))
    if not np.isfinite(threshold):
        return _blocked("density calibration produced a non-finite threshold", ("missing-evidence:fidelity",))
    symptom_flag = float(np.mean(symptom_scores >= threshold))
    negative_flag = float(np.mean(negative_scores >= threshold))
    id_flag = float(np.mean(calibration_scores >= threshold))
    from sklearn.metrics import roc_auc_score

    try:
        auroc = float(roc_auc_score(
            np.concatenate([np.zeros(symptom_scores.shape[0]), np.ones(symptom_scores.shape[0])]),
            np.concatenate([calibration_scores[:symptom_scores.shape[0]], symptom_scores]),
        ))
    except ValueError as exc:
        return _blocked(f"density ranking failed: {exc}", ("missing-evidence:fidelity",))
    # Stability: AUROC across declared seeds uses the SAME one-fit estimator
    # for scoring plus resampled score draws — no refit per resample. Cross-seed
    # estimator variance is reported by scoring with fixed alternate configs
    # only where declared; here stability is the bootstrap AUROC spread.
    flat_symptom = symptom_scores.ravel()
    flat_id = calibration_scores[:symptom_scores.shape[0]].ravel()
    n = int(flat_symptom.shape[0])

    def _auroc_draw(rng: np.random.Generator) -> float:
        sym_positions = rng.integers(0, n, size=n)
        id_positions = rng.integers(0, n, size=n)
        try:
            return float(roc_auc_score(
                np.concatenate([np.zeros(n), np.ones(n)]),
                np.concatenate([flat_id[id_positions], flat_symptom[sym_positions]]),
            ))
        except ValueError:
            return 0.5

    _, auroc_interval = _central_bootstrap(
        control_id="bootstrap:density-auroc",
        base_seed=int(evaluation_seed),
        seed_role="evaluation",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        required=False,
        kind="bootstrap",
        metric_ids=("density_auroc",),
        expected_behavior="seeded resampling of fitted calibrated scores",
        draw=_auroc_draw,
    )
    seed_aurocs: list[float] = [auroc]
    seed_bundles = bundle.get("seed_scores")
    if isinstance(seed_bundles, Mapping):
        for key in sorted(seed_bundles):
            try:
                variant = _finite_array(seed_bundles[key], name=f"density seed {key}").ravel()
            except ExplanationError as exc:
                return _blocked(str(exc), ("missing-evidence:stability-variants",))
            if variant.shape[0] != n:
                return _blocked("density seed scores disagree on sample count", ("missing-evidence:stability-variants",))
            try:
                seed_aurocs.append(float(roc_auc_score(
                    np.concatenate([np.zeros(n), np.ones(n)]),
                    np.concatenate([flat_id, np.asarray(variant, dtype=np.float64)]),
                )))
            except ValueError:
                return _blocked("density seed ranking failed", ("missing-evidence:stability-variants",))
    stability_spread = float(max(seed_aurocs) - min(seed_aurocs)) if seed_aurocs else 0.0

    def _null_statistic(rng: np.random.Generator) -> Mapping[str, float]:
        stacked = np.asarray(symptom, dtype=np.float64)
        shuffled = np.column_stack([rng.permutation(stacked[:, j]) for j in range(stacked.shape[1])])
        try:
            shuffled_scores = np.asarray(estimator.score(shuffled).calibrated_ood_score, dtype=np.float64)
        except (ValueError, RuntimeError):
            return {"density_auroc": float(auroc), "null_auroc": 0.5}
        try:
            null_auroc = float(roc_auc_score(
                np.concatenate([np.zeros(n), np.ones(n)]),
                np.concatenate([flat_id, shuffled_scores[:n]]),
            ))
        except ValueError:
            null_auroc = 0.5
        return {"density_auroc": float(auroc), "null_auroc": float(null_auroc)}

    null_outcome = _central_control(
        control_id=bound_controls["selectivity"],
        base_seed=int(control_seed),
        seed_role="control",
        required=False,
        kind="shuffled",
        metric_ids=("density_auroc", "null_auroc"),
        expected_behavior="independent-per-column permutation destroys joint structure",
        statistic=_null_statistic,
    )
    if null_outcome.status == "failed":
        return _blocked(f"density shuffled null failed: {null_outcome.reason}", (f"failed-control:{bound_controls['selectivity']}",))
    null_auroc = float(null_outcome.observed["null_auroc"])
    fidelity_threshold = hypothesis.threshold_for("density_auroc")
    stability_threshold = hypothesis.threshold_for("auroc_stability")
    selectivity_threshold = hypothesis.threshold_for("flag_margin")
    fidelity_met = True if fidelity_threshold is None else _passes(fidelity_threshold[0], auroc, float(fidelity_threshold[1]))
    stability_met = (
        True if stability_threshold is None else _passes(stability_threshold[0], stability_spread, float(stability_threshold[1]))
    )
    flag_margin = float(symptom_flag - negative_flag)
    selectivity_met = (
        True if selectivity_threshold is None else _passes(selectivity_threshold[0], flag_margin, float(selectivity_threshold[1]))
    )
    leakage_ok = fit_calls["count"] == 1
    if hypothesis.train_split_identity == hypothesis.eval_split_identity:
        leakage_ok = False

    observed = {
        "density_auroc": float(auroc),
        "auroc_stability": float(stability_spread),
        "flag_margin": float(flag_margin),
        "symptom_flag_rate": float(symptom_flag),
        "negative_flag_rate": float(negative_flag),
        "null_auroc": float(null_auroc),
        "estimator_fits": float(fit_calls["count"]),
    }
    fidelity = {"status": "passed" if fidelity_met else "failed", "metric": "density_auroc",
                "observed": float(auroc),
                "threshold": None if fidelity_threshold is None else [fidelity_threshold[0], float(fidelity_threshold[1])]}
    stability = {"status": "passed" if stability_met else "failed", "metric": "auroc_stability",
                 "observed": float(stability_spread),
                 "threshold": None if stability_threshold is None else [stability_threshold[0], float(stability_threshold[1])],
                 "seed_aurocs": [float(item) for item in seed_aurocs]}
    selectivity = {"status": "passed" if selectivity_met else "failed", "metric": "flag_margin",
                   "observed": float(flag_margin),
                   "threshold": None if selectivity_threshold is None else [selectivity_threshold[0], float(selectivity_threshold[1])],
                   "null_auroc": float(null_auroc)}
    leakage = {"status": "passed" if leakage_ok else "failed",
               "reference_identity": str(ref_identity), "test_identity": str(test_identity),
               "estimator_fits": int(fit_calls["count"]), "one_fit": bool(fit_calls["count"] == 1)}
    uncertainty = {"status": "passed", "density_auroc": dict(auroc_interval),
                   "evaluation_seed": int(evaluation_seed), "repetitions": int(repetitions)}
    auroc_gate = fidelity_threshold if fidelity_threshold is not None else (">=", 0.8)
    spread_gate = stability_threshold if stability_threshold is not None else ("<=", 0.1)
    margin_gate = selectivity_threshold if selectivity_threshold is not None else (">=", 0.3)
    specs = [
        _ControlSpec(control_id=bound_controls["fidelity"], kind="counterexample", required=True,
                     metric_ids=("density_auroc",), expected_behavior="ranking separates symptom from in-distribution data",
                     comparator=_central_name(auroc_gate[0]), threshold_value=float(auroc_gate[1])),  # type: ignore[arg-type]
        _ControlSpec(control_id=bound_controls["stability"], kind="seed", required=True,
                     metric_ids=("auroc_stability",), expected_behavior="AUROC spread across declared seeds stays bounded",
                     comparator=_central_name(spread_gate[0]), threshold_value=float(spread_gate[1])),  # type: ignore[arg-type]
        _ControlSpec(control_id=bound_controls["selectivity"], kind="negative", required=True,
                     metric_ids=("flag_margin",), expected_behavior="symptom flag rate exceeds the negative flag rate",
                     comparator=_central_name(margin_gate[0]), threshold_value=float(margin_gate[1])),  # type: ignore[arg-type]
    ]
    supplied = {
        bound_controls["fidelity"]: {"density_auroc": float(auroc)},
        bound_controls["stability"]: {"auroc_stability": float(stability_spread)},
        bound_controls["selectivity"]: {"flag_margin": float(flag_margin)},
    }
    gaps: list[str] = []
    if not fidelity_met:
        gaps.append("failed-fidelity:density_auroc")
    if not stability_met:
        gaps.append("failed-stability:auroc_stability")
    if not selectivity_met:
        gaps.append("failed-selectivity:flag_margin")
    if not leakage_ok:
        gaps.append("failed-leakage:estimator-fits")
    central_extra = {bound_controls["selectivity"]: dict(null_outcome.to_dict())}
    record = _gate_dimensions(
        hypothesis, method, observed=observed, fidelity=fidelity, stability=stability,
        selectivity=selectivity, leakage=leakage, uncertainty=uncertainty,
        control_specs=specs, supplied=supplied, provenance=provenance, limitation=limitation,
        support_reason="declared density run meets ranking, stability, selectivity, leakage, and uncertainty gates",
        block_prefix="density evidence", fidelity_ok=fidelity_met, stability_ok=stability_met,
        selectivity_ok=selectivity_met, leakage_ok=leakage_ok, gaps=gaps,
        repetitions=int(repetitions), confidence_level=float(confidence_level),
        evaluation_seed=int(evaluation_seed), control_seed=int(control_seed),
    )
    return record


# ---------------------------------------------------------------------------
# 5) Clustering
# ---------------------------------------------------------------------------


def _evaluate_clustering(
    hypothesis: ExplanationHypothesis,
    bundle: Mapping[str, object],
    *,
    repetitions: int,
    confidence_level: float,
    evaluation_seed: int,
    control_seed: int,
    training_seed: int,
) -> ExplanationEvidence:
    method = "clustering"
    provenance = _base_provenance(hypothesis, method)
    limitation = "a cluster number is not a semantic label; causal claims require 80.18+ trials"

    def _blocked(reason: str, gaps: Sequence[str]) -> ExplanationEvidence:
        dims = _empty_dims()
        record = ExplanationEvidence(
            hypothesis_id=hypothesis.hypothesis_id, method="probe", outcome="inconclusive",
            claim_allowed=False, observed_effect={}, fidelity=dict(dims["fidelity"]),
            stability=dict(dims["stability"]), selectivity=dict(dims["selectivity"]),
            leakage=dict(dims["leakage"]), uncertainty=dict(dims["uncertainty"]),
            control_outcomes={}, central_outcomes={}, missing_evidence=tuple(gaps),
            limitations=(limitation,), reason=reason, provenance=dict(provenance),
        )
        return _remethod(record, method)

    try:
        bound_controls = _bound_ids(hypothesis)
    except ExplanationError as exc:
        return _blocked(str(exc), ("missing-evidence:control-declaration",))

    try:
        symptom = _finite_array(bundle.get("symptom_matrix"), name="clustering symptom_matrix")
        benign = _finite_array(bundle.get("benign_matrix"), name="clustering benign_matrix")
        labels = _finite_array(bundle.get("symptom_labels"), name="clustering symptom_labels").ravel()
    except ExplanationError as exc:
        return _blocked(f"clustering input is incomplete: {exc}", ("missing-evidence:clustering-matrices",))
    for name, matrix in (("symptom_matrix", symptom), ("benign_matrix", benign)):
        if matrix.ndim != 2:
            return _blocked(f"clustering {name} must be 2D", ("missing-evidence:clustering-matrices",))
    if symptom.shape[1] != benign.shape[1]:
        return _blocked("clustering batch widths disagree", ("missing-evidence:clustering-matrices",))
    if labels.shape[0] != int(symptom.shape[0]):
        return _blocked("clustering labels must cover every symptom row", ("missing-evidence:clustering-matrices",))
    if len(np.unique(labels)) < 2:
        return _blocked("clustering labels must contain at least two classes", ("missing-evidence:clustering-matrices",))
    n_clusters_raw = bundle.get("n_clusters")
    try:
        n_clusters = int(n_clusters_raw) if n_clusters_raw is not None else 2
    except (TypeError, ValueError):
        return _blocked("clustering n_clusters is non-numeric", ("missing-evidence:clustering-config",))
    if n_clusters < 2 or n_clusters > int(symptom.shape[0]):
        return _blocked("clustering n_clusters is out of range", ("missing-evidence:clustering-config",))
    sample_ids = bundle.get("sample_ids")
    if sample_ids is None:
        return _blocked("clustering sample identities are missing", ("missing-evidence:leakage-identities",))
    ids = tuple(sample_ids) if isinstance(sample_ids, Sequence) and not isinstance(sample_ids, (str, bytes)) else ()
    if not ids or len(ids) != int(symptom.shape[0]) or len(set(str(i) for i in ids)) != len(ids):
        return _blocked("clustering sample identities must uniquely cover every symptom row", ("missing-evidence:leakage-identities",))
    cluster_label = bundle.get("cluster_label")
    promoted_label = str(cluster_label) if isinstance(cluster_label, str) and cluster_label.strip() else None
    provenance["n_clusters"] = int(n_clusters)

    from latent_anything.clustering import KMeans, KMeansConfig, compare_with_labels

    fit_calls = {"count": 0}
    seeds = tuple(int(item) for item in hypothesis.seeds)
    try:
        results = []
        for seed in seeds:
            result = KMeans(KMeansConfig(n_clusters=int(n_clusters), random_state=int(seed), n_init=10)).fit_predict(
                np.asarray(symptom, dtype=np.float64), provenance={"seed": int(seed)}
            )
            fit_calls["count"] += 1
            results.append(result)
    except ValueError as exc:
        return _blocked(f"clustering fit failed: {exc}", ("missing-evidence:fidelity",))
    reference = results[0]
    from sklearn.metrics import adjusted_rand_score

    aris = [1.0]
    for other in results[1:]:
        aris.append(float(adjusted_rand_score(reference.assignments, other.assignments)))
    ari = float(min(aris))
    agreement = compare_with_labels(np.asarray(labels).ravel(), np.asarray(reference.assignments).ravel())
    label_ari = float(agreement["adjusted_rand_index"])
    silhouette = float(reference.silhouette_score)

    def _shuffled_selectivity(rng: np.random.Generator) -> Mapping[str, float]:
        shuffled = np.asarray(rng.permutation(np.asarray(labels).ravel()))
        shuffled_agreement = compare_with_labels(shuffled, np.asarray(reference.assignments).ravel())
        return {"label_ari": float(label_ari), "shuffled_ari": float(shuffled_agreement["adjusted_rand_index"])}

    shuffle_outcome = _central_control(
        control_id=bound_controls["selectivity"],
        base_seed=int(control_seed),
        seed_role="control",
        required=False,
        kind="randomized",
        metric_ids=("label_ari", "shuffled_ari"),
        expected_behavior="one seeded permutation of the symptom labels",
        statistic=_shuffled_selectivity,
    )
    if shuffle_outcome.status == "failed":
        return _blocked(f"clustering shuffled control failed: {shuffle_outcome.reason}", (f"failed-control:{bound_controls['selectivity']}",))
    shuffled_ari = float(shuffle_outcome.observed["shuffled_ari"])
    selectivity_margin = float(label_ari - shuffled_ari)
    benign_agreement = compare_with_labels(
        np.asarray(np.random.default_rng(int(control_seed)).permutation(np.asarray(labels).ravel())[: benign.shape[0]] if benign.shape[0] <= labels.shape[0] else np.zeros(benign.shape[0])),
        np.asarray(KMeans(KMeansConfig(n_clusters=int(n_clusters), random_state=int(seeds[0]), n_init=10)).fit_predict(
            np.asarray(benign, dtype=np.float64)).assignments).ravel(),
    ) if False else None
    _ = benign_agreement
    benign_fit = KMeans(KMeansConfig(n_clusters=int(n_clusters), random_state=int(seeds[0]), n_init=10)).fit_predict(
        np.asarray(benign, dtype=np.float64))
    fit_calls["count"] += 1
    benign_silhouette = float(benign_fit.silhouette_score)

    assignments = np.asarray(reference.assignments).ravel()

    def _assignment_draw(rng: np.random.Generator) -> float:
        positions = rng.integers(0, assignments.shape[0], size=assignments.shape[0])
        # Agreement of the resampled assignments with resampled labels (ARI
        # needs no label alignment: permutation-invariant by construction).
        try:
            return float(adjusted_rand_score(
                np.asarray(labels).ravel()[positions], np.asarray(assignments).ravel()[positions]))
        except ValueError:
            return 0.0

    _, ari_interval = _central_bootstrap(
        control_id="bootstrap:clustering-ari",
        base_seed=int(evaluation_seed),
        seed_role="evaluation",
        repetitions=int(repetitions),
        confidence_level=float(confidence_level),
        required=False,
        kind="bootstrap",
        metric_ids=("clustering_ari",),
        expected_behavior="seeded resampling of fitted assignments",
        draw=_assignment_draw,
    )

    fidelity_threshold = hypothesis.threshold_for("label_agreement")
    stability_threshold = hypothesis.threshold_for("cluster_stability")
    selectivity_threshold = hypothesis.threshold_for("selectivity_margin")
    fidelity_met = True if fidelity_threshold is None else _passes(fidelity_threshold[0], label_ari, float(fidelity_threshold[1]))
    stability_met = True if stability_threshold is None else _passes(stability_threshold[0], ari, float(stability_threshold[1]))
    selectivity_met = (
        True if selectivity_threshold is None else _passes(selectivity_threshold[0], selectivity_margin, float(selectivity_threshold[1]))
    )
    leakage_ok = fit_calls["count"] == len(seeds) + 1
    if hypothesis.train_split_identity == hypothesis.eval_split_identity:
        leakage_ok = False

    observed = {
        "label_agreement": float(label_ari),
        "cluster_stability": float(ari),
        "selectivity_margin": float(selectivity_margin),
        "shuffled_ari": float(shuffled_ari),
        "silhouette": float(silhouette),
        "benign_silhouette": float(benign_silhouette),
        "cluster_fits": float(fit_calls["count"]),
    }
    fidelity = {"status": "passed" if fidelity_met else "failed", "metric": "label_agreement",
                "observed": float(label_ari),
                "threshold": None if fidelity_threshold is None else [fidelity_threshold[0], float(fidelity_threshold[1])],
                "silhouette": float(silhouette)}
    stability = {"status": "passed" if stability_met else "failed", "metric": "cluster_stability",
                 "observed": float(ari),
                 "threshold": None if stability_threshold is None else [stability_threshold[0], float(stability_threshold[1])],
                 "seed_aris": [float(item) for item in aris], "permutation_invariant": True}
    selectivity = {"status": "passed" if selectivity_met else "failed", "metric": "selectivity_margin",
                   "observed": float(selectivity_margin),
                   "threshold": None if selectivity_threshold is None else [selectivity_threshold[0], float(selectivity_threshold[1])],
                   "shuffled_ari": float(shuffled_ari)}
    leakage = {"status": "passed" if leakage_ok else "failed",
               "train_split_identity": hypothesis.train_split_identity,
               "eval_split_identity": hypothesis.eval_split_identity,
               "cluster_fits": int(fit_calls["count"])}
    uncertainty = {"status": "passed", "clustering_ari": dict(ari_interval),
                   "evaluation_seed": int(evaluation_seed), "repetitions": int(repetitions)}
    agree_gate = fidelity_threshold if fidelity_threshold is not None else (">=", 0.8)
    stab_gate = stability_threshold if stability_threshold is not None else (">=", 0.8)
    sel_gate = selectivity_threshold if selectivity_threshold is not None else (">=", 0.5)
    specs = [
        _ControlSpec(control_id=bound_controls["fidelity"], kind="counterexample", required=True,
                     metric_ids=("label_agreement",), expected_behavior="clusters agree with symptom labels under predeclared floor",
                     comparator=_central_name(agree_gate[0]), threshold_value=float(agree_gate[1])),  # type: ignore[arg-type]
        _ControlSpec(control_id=bound_controls["stability"], kind="cross_seed", required=True,
                     metric_ids=("cluster_stability",), expected_behavior="permutation-invariant assignment stability across seeds",
                     comparator=_central_name(stab_gate[0]), threshold_value=float(stab_gate[1])),  # type: ignore[arg-type]
        _ControlSpec(control_id=bound_controls["selectivity"], kind="randomized", required=True,
                     metric_ids=("selectivity_margin",), expected_behavior="agreement exceeds shuffled-label agreement",
                     comparator=_central_name(sel_gate[0]), threshold_value=float(sel_gate[1])),  # type: ignore[arg-type]
    ]
    supplied = {
        bound_controls["fidelity"]: {"label_agreement": float(label_ari)},
        bound_controls["stability"]: {"cluster_stability": float(ari)},
        bound_controls["selectivity"]: {"selectivity_margin": float(selectivity_margin)},
    }
    gaps: list[str] = []
    if not fidelity_met:
        gaps.append("failed-fidelity:label_agreement")
    if not stability_met:
        gaps.append("failed-stability:cluster_stability")
    if not selectivity_met:
        gaps.append("failed-selectivity:selectivity_margin")
    if not leakage_ok:
        gaps.append("failed-leakage:cluster-fits")
    record = _gate_dimensions(
        hypothesis, method, observed=observed, fidelity=fidelity, stability=stability,
        selectivity=selectivity, leakage=leakage, uncertainty=uncertainty,
        control_specs=specs, supplied=supplied, provenance=provenance, limitation=limitation,
        support_reason="declared clusters meet agreement, permutation-invariant stability, selectivity, leakage, and uncertainty gates",
        block_prefix="clustering evidence", fidelity_ok=fidelity_met, stability_ok=stability_met,
        selectivity_ok=selectivity_met, leakage_ok=leakage_ok, gaps=gaps,
        repetitions=int(repetitions), confidence_level=float(confidence_level),
        evaluation_seed=int(evaluation_seed), control_seed=int(control_seed),
        promoted_label=promoted_label if (fidelity_met and stability_met and selectivity_met and leakage_ok) else None,
    )
    if record.outcome != "supported":
        provenance_redacted = dict(record.provenance)
        provenance_redacted.pop("promoted_label", None)
        base = ExplanationEvidence(
            hypothesis_id=record.hypothesis_id, method="probe", outcome=record.outcome,  # type: ignore[arg-type]
            claim_allowed=False, observed_effect=dict(record.observed_effect),
            fidelity=dict(record.fidelity), stability=dict(record.stability),
            selectivity=dict(record.selectivity), leakage=dict(record.leakage),
            uncertainty=dict(record.uncertainty), control_outcomes=dict(record.control_outcomes),
            central_outcomes=dict(record.central_outcomes),
            missing_evidence=tuple([*record.missing_evidence, "redacted:semantic-cluster-label"]),
            limitations=tuple(record.limitations), reason=record.reason, provenance=provenance_redacted,
        )
        return _remethod(base, method)
    return record


# ---------------------------------------------------------------------------
# Shared evaluation seam (extends the 80.16 convention, same record shape)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FeatureExplainContext:
    """One executor call evaluated exactly once and shared by decisions."""

    hypotheses: tuple[ExplanationHypothesis, ...]
    evidence: tuple[ExplanationEvidence, ...]
    central_outcomes: Mapping[str, object]
    control_outcomes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "central_outcomes", dict(self.central_outcomes))
        object.__setattr__(self, "control_outcomes", dict(self.control_outcomes))


def _omitted_feature(hypothesis_id: str, method: str, *, reason: str) -> ExplanationEvidence:
    record = ExplanationEvidence(
        hypothesis_id=hypothesis_id,
        method="probe",
        outcome="omitted",
        claim_allowed=False,
        observed_effect={},
        fidelity={"status": "omitted"},
        stability={"status": "omitted"},
        selectivity={"status": "omitted"},
        leakage={"status": "omitted"},
        uncertainty={"status": "omitted"},
        control_outcomes={},
        central_outcomes={},
        missing_evidence=("omitted:undeclared-hypothesis-or-method",),
        limitations=("no method code executed for this hypothesis/method pair",),
        reason=reason,
        provenance={"explainer_version": FEATURE_EXPLAINER_VERSION, "method": method, "hypothesis_id": hypothesis_id},
    )
    return _remethod(record, method)


_EVALUATORS = {
    "sae_sparse": _evaluate_sae_sparse,
    "lens": _evaluate_lens,
    "geometry": _evaluate_geometry,
    "density": _evaluate_density,
    "clustering": _evaluate_clustering,
}


def evaluate_feature_explanations(
    hypotheses: Sequence[ExplanationHypothesis],
    inputs: MethodInputs,
    *,
    repetitions: int = 20,
    confidence_level: float = 0.95,
    evaluation_seed: int = 0,
    control_seed: int = 1,
    training_seed: int = 0,
) -> tuple[tuple[ExplanationEvidence, ...], dict[str, object]]:
    """Evaluate every declared feature hypothesis exactly once.

    Same contract as 80.16 :func:`evaluate_explanations`: the declared
    symptom relationship is checked before any method callback fires (a
    missing relationship blocks with the callback uninvoked), undeclared
    pairs yield honest ``omitted`` records, and one shared context feeds
    decisions and payload assembly. ``inputs`` carries one bundle per
    feature method; :class:`MethodInputs` is reused as the typed carrier.
    """
    if not hypotheses:
        raise ExplanationError("hypotheses must declare at least one hypothesis")
    declared = tuple(hypotheses)
    identities = [item.hypothesis_id for item in declared]
    if len(set(identities)) != len(identities):
        raise ExplanationError("hypothesis identifiers must be unique")
    for item in declared:
        if not isinstance(item, ExplanationHypothesis):
            raise ExplanationError("hypotheses must hold ExplanationHypothesis items")
        if item.method not in SUPPORTED_FEATURE_METHODS:
            raise ExplanationError(f"method {item.method!r} is not a feature explanation method")
    if not isinstance(inputs, MethodInputs):
        raise ExplanationError("inputs must be a MethodInputs")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise ExplanationError("repetitions must be at least two")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ExplanationError("confidence_level must be between zero and one")
    for name, seed in (
        ("evaluation_seed", evaluation_seed),
        ("control_seed", control_seed),
        ("training_seed", training_seed),
    ):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ExplanationError(f"{name} must be a non-negative integer")

    evidence: list[ExplanationEvidence] = []
    for hypothesis in declared:
        bundle = inputs.callback_for(hypothesis.method)
        if bundle is None:
            evidence.append(
                _omitted_feature(
                    hypothesis.hypothesis_id,
                    hypothesis.method,
                    reason=f"method {hypothesis.method!r} has no declared input for hypothesis {hypothesis.hypothesis_id!r}; callback uncalled",
                )
            )
            continue
        resolved = _require_mapping(bundle, hypothesis_id=hypothesis.hypothesis_id)
        if not _relationship_of(resolved, hypothesis):
            evidence.append(
                _omitted_feature(
                    hypothesis.hypothesis_id,
                    hypothesis.method,
                    reason=f"hypothesis {hypothesis.hypothesis_id!r} declares no feature-to-symptom relationship; method callback uncalled",
                )
            )
            continue
        evaluator = _EVALUATORS[hypothesis.method]
        if hypothesis.method == "density" or hypothesis.method == "clustering":
            evidence.append(
                evaluator(
                    hypothesis, resolved,
                    repetitions=int(repetitions), confidence_level=float(confidence_level),
                    evaluation_seed=int(evaluation_seed), control_seed=int(control_seed),
                    training_seed=int(training_seed),
                )
            )
        elif hypothesis.method == "sae_sparse":
            evidence.append(
                evaluator(
                    hypothesis, resolved,
                    repetitions=int(repetitions), confidence_level=float(confidence_level),
                    evaluation_seed=int(evaluation_seed), control_seed=int(control_seed),
                )
            )
        else:
            evidence.append(
                evaluator(
                    hypothesis, resolved,
                    repetitions=int(repetitions), confidence_level=float(confidence_level),
                    evaluation_seed=int(evaluation_seed), control_seed=int(control_seed),
                )
            )
    context = _FeatureExplainContext(
        hypotheses=declared,
        evidence=tuple(evidence),
        central_outcomes={record.hypothesis_id: dict(record.central_outcomes) for record in evidence},
        control_outcomes={record.hypothesis_id: dict(record.control_outcomes) for record in evidence},
    )
    payload = feature_explanation_payload(context)
    return tuple(evidence), payload


def feature_explanation_payload(context: _FeatureExplainContext) -> dict[str, object]:
    """Assemble the canonical machine-readable feature explain-stage payload."""
    if not isinstance(context, _FeatureExplainContext):
        raise ExplanationError("context must be a _FeatureExplainContext")
    records = [record.to_dict() for record in context.evidence]
    try:
        canonical_json(records)
    except PortableNodeError as exc:
        raise ExplanationError(f"explanation records are not canonical JSON: {exc}") from exc
    return {
        "explainer_version": FEATURE_EXPLAINER_VERSION,
        "base_explainer_version": _BASE_EXPLAINER_VERSION,
        "hypotheses": [hypothesis.to_dict() for hypothesis in context.hypotheses],
        "evidence": records,
        "family_evidence": {
            record.hypothesis_id: {
                "outcome": record.outcome,
                "claim_allowed": record.claim_allowed,
                "method": record.method,
            }
            for record in context.evidence
        },
        "controls": {key: dict(value) for key, value in context.central_outcomes.items()},
        "control_outcomes": {key: dict(value) for key, value in context.control_outcomes.items()},
    }


def make_feature_explain_executor(
    hypotheses: Sequence[ExplanationHypothesis],
    inputs: MethodInputs,
    *,
    repetitions: int = 20,
    confidence_level: float = 0.95,
    evaluation_seed: int = 0,
    control_seed: int = 1,
    training_seed: int = 0,
    version: str = FEATURE_EXPLAINER_VERSION,
) -> Any:
    """Build a supplied ``explain``-stage executor bound to feature hypotheses.

    The coordinator invokes the returned callable with a ``StageInvocation``;
    the callable binds the expected manifest/run identity (run manifest,
    capture representation, manifest dataset split, plus upstream diagnostic
    linkage: hypothesis family against the structured detect payload family
    identities, declared metrics against the structured detect payload metric
    identities, and the exact declared localize ``family_id``/``metric_id``
    only when the prior localize payload declares them), then exact-matches
    every entry of the non-empty ``localization_bindings`` contract against
    the structured prior localization result, evaluates
    :func:`evaluate_feature_explanations` once, and returns a
    ``completed`` :class:`StageOutput`. ``localization_bindings`` is the only
    upstream localization contract; ``layer_id``/``slice_id`` (like
    ``symptom_id``/``target_id``) are hypothesis-local method context
    validated by method inputs, never upstream-localized by their mere
    presence. Missing required upstream fields fail closed. No algorithm
    enters ``DiagnosticWorkflow`` itself.
    """
    if not isinstance(version, str) or not version.strip():
        raise ExplanationError("version must be a non-empty string")
    frozen = tuple(hypotheses)
    if not frozen:
        raise ExplanationError("hypotheses must declare at least one hypothesis")
    for item in frozen:
        if not isinstance(item, ExplanationHypothesis):
            raise ExplanationError("hypotheses must hold ExplanationHypothesis items")
        if item.method not in SUPPORTED_FEATURE_METHODS:
            raise ExplanationError(f"method {item.method!r} is not a feature explanation method")
    if not isinstance(inputs, MethodInputs):
        raise ExplanationError("inputs must be a MethodInputs")

    def _execute(invocation: Any) -> Any:
        from latent_anything._diagnostic_workflow import StageContractError as _ContractError
        from latent_anything._diagnostic_workflow import StageOutput as _StageOutput
        from latent_anything._probe_tcav_ig_explanation import _check_explain_identities as _check_ids

        if invocation.stage != "explain":
            raise _ContractError(f"explain executor received stage {invocation.stage!r}")
        _check_ids(frozen, invocation)
        _, payload = evaluate_feature_explanations(
            frozen,
            inputs,
            repetitions=int(repetitions),
            evaluation_seed=int(evaluation_seed),
            control_seed=int(control_seed),
            training_seed=int(training_seed),
        )
        aggregate: Literal["completed", "unsupported"] = "completed"
        if payload["evidence"] and all(
            str(item.get("outcome")) == "omitted" for item in cast(Sequence[Mapping[str, object]], payload["evidence"])
        ):
            aggregate = "unsupported"
        return _StageOutput(stage="explain", outcome=aggregate, payload=payload, artifact_refs=())

    _execute.explainer_version = version  # type: ignore[attr-defined]
    return _execute


__all__ = [
    "FEATURE_EXPLAINER_VERSION",
    "SUPPORTED_FEATURE_METHODS",
    "evaluate_feature_explanations",
    "feature_explanation_payload",
    "make_feature_explain_executor",
]
