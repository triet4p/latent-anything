"""TCAV-style concept-sensitivity analysis for latent representations.

This module implements **T**esting with **C**oncept **A**ctivation **V**ectors
(TCAV, Kim et al. 2018) for the project's three integration types.  TCAV
measures how sensitive a model's scalar output is to a human-defined concept
at a chosen layer, using directional derivatives along the concept direction.

Key components
--------------
- :class:`ConceptDataset` — dataset (+ provenance) of concept vs reference examples.
- :func:`learn_mean_diff_direction` — concept direction as centroid difference.
- :func:`learn_linear_separator_direction` — concept direction as a regularised
  linear classifier's coefficient vector.
- :class:`TransformerLogitTarget` — a scalar target for transformer integrations:
  the logit of a specific token at a specific position.
- :class:`TCAVScore` — per-concept per-target TCAV result with significance.
- :class:`TCAVResult` — full analysis including random-concept baselines,
  multiple-comparison correction, and optional intervention cross-check.
- :func:`compute_tcav` — the main entry point for transformer integrations.
- :func:`intervention_agreement` — cross-check observational sensitivity with
  a bounded matched-norm intervention.

Design decisions
----------------
- **No ``Method`` protocol.**  Like ``LinearProbe`` and ``MLPProbe``, TCAV
  requires labelled concept sets during computation and does not follow the
  ``Method`` / ``AnalysisPipeline`` lifecycle.
- **Gradient computation is internal.**  PyTorch gradients are computed inside
  the module and never exposed on the public interface.
- **Integration-specific.**  The first implementation targets decoder-only
  transformers (GPT-2).  Diffusion support is deferred until a second concrete
  integration type is added for cross-check (Rule of Three).
- **No generic probe / concept extraction protocol.**  TCAV has different
  inputs, target semantics, and statistics from the two existing probes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from latent_anything._probe_split import stratified_split
from latent_anything._tcav_model import (
    compute_transformer_layer_gradient,
    extract_layer_activation,
)
from latent_anything._tcav_model import (
    intervention_agreement as _intervention_agreement,
)
from latent_anything._tcav_statistics import assemble_tcav_result

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false
# (torch has incomplete type stubs — these warnings are noise)

# ---------------------------------------------------------------------------
# Part 1 — Concept dataset with provenance  (Task 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptDataset:
    """Deterministic concept / reference dataset with full provenance.

    Parameters
    ----------
    concept_examples :
        Feature vectors for concept-positive examples,
        shape ``(n_concept, n_features)``.
    reference_examples :
        Feature vectors for concept-negative (reference) examples,
        shape ``(n_reference, n_features)``.
    concept_name :
        Human-readable label for this concept (e.g. ``"sentiment"``).
    source :
        Dataset source identifier (e.g. ``"imdb_reviews"``).
    representation_space :
        Which representation space the features come from
        (e.g. ``"gpt2_layer_8_hidden"``).
    model_version :
        Model provenance string (e.g. ``"gpt2@e7da7f221d5bf"``).
    provenance :
        Arbitrary additional provenance metadata.
    """

    concept_examples: np.ndarray
    reference_examples: np.ndarray
    concept_name: str
    source: str = ""
    representation_space: str = ""
    model_version: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.concept_examples.ndim != 2:
            raise ValueError(f"concept_examples must be 2D, got {self.concept_examples.ndim}D")
        if self.reference_examples.ndim != 2:
            raise ValueError(f"reference_examples must be 2D, got {self.reference_examples.ndim}D")
        if self.concept_examples.shape[1] != self.reference_examples.shape[1]:
            raise ValueError(
                f"feature dimension mismatch: concept {self.concept_examples.shape[1]} "
                f"vs reference {self.reference_examples.shape[1]}"
            )
        if not self.concept_name:
            raise ValueError("concept_name must not be empty")
        if self.concept_examples.shape[0] < 2:
            raise ValueError(f"need at least 2 concept examples, got {self.concept_examples.shape[0]}")
        if self.reference_examples.shape[0] < 2:
            raise ValueError(f"need at least 2 reference examples, got {self.reference_examples.shape[0]}")

    @property
    def n_features(self) -> int:
        """Feature dimensionality."""
        return self.concept_examples.shape[1]

    @property
    def n_concept(self) -> int:
        """Number of concept-positive examples."""
        return self.concept_examples.shape[0]

    @property
    def n_reference(self) -> int:
        """Number of concept-negative (reference) examples."""
        return self.reference_examples.shape[0]


# ---------------------------------------------------------------------------
# Part 2 — Concept direction learning  (Task 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptDirectionResult:
    """A learned concept direction with stability and separability metrics.

    Parameters
    ----------
    direction :
        Unit-norm direction vector of shape ``(n_features,)``.
    method :
        Method used to learn the direction (``"mean_diff"`` or
        ``"linear_separator"``).
    stability :
        Mean cosine similarity across bootstrap resamples of the
        concept/reference sets.  Higher → more stable direction.
    stability_ci95 :
        95 % confidence interval half-width for the stability estimate.
    separability_accuracy :
        Held-out classification accuracy of a linear classifier using
        this direction (mean-diff baseline) or the direction itself
        (linear-separator uses the classifier's own held-out accuracy).
    n_concept :
        Number of concept examples used to learn the direction.
    n_reference :
        Number of reference examples used to learn the direction.
    """

    direction: np.ndarray
    method: str
    stability: float
    stability_ci95: float
    separability_accuracy: float
    n_concept: int
    n_reference: int

    def __post_init__(self) -> None:
        if self.direction.ndim != 1:
            raise ValueError(f"direction must be 1D, got {self.direction.ndim}D")
        if self.method not in ("mean_diff", "linear_separator"):
            raise ValueError(f"unknown method: {self.method!r}")
        if not (0.0 <= self.stability <= 1.0 + 1e-9):
            raise ValueError(f"stability must be in [0, 1], got {self.stability}")
        if not (0.0 <= self.separability_accuracy <= 1.0 + 1e-9):
            raise ValueError(f"separability_accuracy must be in [0, 1], got {self.separability_accuracy}")


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return unit-norm copy of *v*."""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v.copy()
    return v.astype(np.float64) / norm


def learn_mean_diff_direction(
    dataset: ConceptDataset,
    *,
    n_bootstrap: int = 100,
    bootstrap_seed: int = 0,
    test_size: float = 0.3,
    split_seed: int = 0,
) -> ConceptDirectionResult:
    """Learn a concept direction as the normalised mean-difference vector.

    The direction is ``normalize(mean(concept) - mean(reference))``.
    Stability is the mean pairwise cosine similarity across bootstrap
    resamples of the concept and reference sets.

    Parameters
    ----------
    dataset :
        Concept/reference dataset.
    n_bootstrap :
        Number of bootstrap resamples for the stability estimate.
    bootstrap_seed :
        RNG seed for bootstrapping.
    test_size :
        Fraction of data held out for separability evaluation.
    split_seed :
        RNG seed for the separability train/test split.

    Returns
    -------
    ConceptDirectionResult
        Unit-norm direction, stability, and held-out separability.
    """
    concept = dataset.concept_examples
    reference = dataset.reference_examples

    # Main direction
    direction = _normalize(concept.mean(axis=0) - reference.mean(axis=0))

    # Bootstrap stability
    rng = np.random.default_rng(bootstrap_seed)
    cosines: list[float] = []
    for _ in range(n_bootstrap):
        c_boot = concept[rng.integers(0, concept.shape[0], size=concept.shape[0])]
        r_boot = reference[rng.integers(0, reference.shape[0], size=reference.shape[0])]
        d_boot = _normalize(c_boot.mean(axis=0) - r_boot.mean(axis=0))
        cosines.append(float(np.dot(direction, d_boot)))  # unit vectors → cosine

    cos_arr = np.asarray(cosines, dtype=np.float64)
    stability = float(cos_arr.mean())
    ci95 = float(1.96 * cos_arr.std(ddof=1) / np.sqrt(n_bootstrap)) if n_bootstrap > 1 else 0.0

    # Held-out separability
    all_x = np.concatenate([concept, reference], axis=0)
    all_y = np.concatenate([np.ones(concept.shape[0]), np.zeros(reference.shape[0])])
    train_mask, _, test_mask = stratified_split(
        all_y,
        test_size=test_size,
        val_size=0.0,
        random_state=split_seed,
    )
    train_x = all_x[train_mask]
    train_y = all_y[train_mask]
    test_x = all_x[test_mask]
    test_y = all_y[test_mask]

    # Project onto direction and threshold at the mean of projected training values
    proj_train = train_x @ direction
    threshold = float(np.mean(proj_train[train_y == 1])) * 0.5 + float(np.mean(proj_train[train_y == 0])) * 0.5
    proj_test = test_x @ direction
    preds = (proj_test >= threshold).astype(np.float64)
    separability = float(np.mean(preds == test_y))

    return ConceptDirectionResult(
        direction=direction,
        method="mean_diff",
        stability=stability,
        stability_ci95=ci95,
        separability_accuracy=separability,
        n_concept=concept.shape[0],
        n_reference=reference.shape[0],
    )


def learn_linear_separator_direction(
    dataset: ConceptDataset,
    *,
    n_bootstrap: int = 100,
    bootstrap_seed: int = 0,
    test_size: float = 0.3,
    split_seed: int = 0,
    c_value: float = 1.0,
) -> ConceptDirectionResult:
    """Learn a concept direction as a regularised linear classifier's coefs.

    Fits a logistic regression (L2) with balanced class weights on the
    concept vs reference sets and returns the unit-normalised coefficient
    vector as the concept direction.

    Stability is the mean cosine similarity of coefficient vectors across
    bootstrap resamples.  Separability is the held-out accuracy of the
    logistic regression classifier.

    Parameters
    ----------
    dataset :
        Concept/reference dataset.
    n_bootstrap :
        Number of bootstrap resamples.
    bootstrap_seed :
        RNG seed.
    test_size :
        Held-out fraction for separability evaluation.
    split_seed :
        RNG seed for the split.
    C :
        Inverse regularisation strength for ``LogisticRegression``
        (smaller = stronger regularisation).

    Returns
    -------
    ConceptDirectionResult
        Unit-norm direction, stability, and held-out separability.
    """
    from sklearn.linear_model import LogisticRegression

    concept = dataset.concept_examples
    reference = dataset.reference_examples
    all_x = np.concatenate([concept, reference], axis=0)
    all_y = np.concatenate([np.ones(concept.shape[0]), np.zeros(reference.shape[0])])

    # Main direction
    lr = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=split_seed,
    )
    lr.fit(all_x, all_y)
    direction = _normalize(np.asarray(lr.coef_[0]))

    # Bootstrap stability
    rng = np.random.default_rng(bootstrap_seed)
    cosines: list[float] = []
    n_total = all_x.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_total, size=n_total)
        x_boot = all_x[idx]
        y_boot = all_y[idx]
        # Ensure both classes are present
        if np.unique(y_boot).size < 2:
            cosines.append(1.0)
            continue
        lr_boot = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=bootstrap_seed,
        )
        lr_boot.fit(x_boot, y_boot)
        d_boot = _normalize(np.asarray(lr_boot.coef_[0]))
        cosines.append(float(np.dot(direction, d_boot)))

    cos_arr = np.asarray(cosines, dtype=np.float64)
    stability = float(cos_arr.mean())
    ci95 = float(1.96 * cos_arr.std(ddof=1) / np.sqrt(n_bootstrap)) if n_bootstrap > 1 else 0.0

    # Held-out separability (use the main classifier's own accuracy)
    train_mask, _, test_mask = stratified_split(
        all_y,
        test_size=test_size,
        val_size=0.0,
        random_state=split_seed,
    )
    lr_sep = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=split_seed,
    )
    lr_sep.fit(all_x[train_mask], all_y[train_mask])
    separability = float(lr_sep.score(all_x[test_mask], all_y[test_mask]))

    return ConceptDirectionResult(
        direction=direction,
        method="linear_separator",
        stability=stability,
        stability_ci95=ci95,
        separability_accuracy=separability,
        n_concept=concept.shape[0],
        n_reference=reference.shape[0],
    )


# ---------------------------------------------------------------------------
# Part 3 — Scalar model targets  (Task 3)
# ---------------------------------------------------------------------------


class TransformerLogitTarget(BaseModel):
    """Scalar target for a decoder-only transformer: the logit of one token.

    The target is the **logit** (pre-softmax) of ``token_id`` at sequence
    position ``position`` in batch ``batch_index``.  Using the logit rather
    than the probability avoids the non-linearity of softmax and keeps the
    gradient signal clean.

    Parameters
    ----------
    token_id :
        Vocabulary ID of the token whose logit is the target.
    position :
        Sequence position.  Use ``-1`` for the last token (the common
        choice for language-model scoring).
    batch_index :
        Which example in the batch to differentiate.
    """

    target_type: Literal["transformer_logit"] = "transformer_logit"
    token_id: int = Field(..., ge=0, description="Vocabulary ID of the target token")
    position: int = Field(default=-1, description="Sequence position (-1 = last)")
    batch_index: int = Field(default=0, ge=0, description="Batch index")


# ---------------------------------------------------------------------------
# Part 4 — TCAV scores and result types  (Task 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TCAVScore:
    """TCAV score for one concept at one layer for one scalar target.

    Parameters
    ----------
    concept_name :
        Human-readable concept label.
    layer :
        Layer index where the concept direction was learned and gradients
        were computed.
    target :
        The scalar target definition used for gradient computation.
    cav_direction :
        The learned concept direction with stability / separability.
    sensitivity :
        TCAV score: fraction of examples for which the directional
        derivative ``∇target · v_c`` is positive.  Ranges in ``[0, 1]``;
        > 0.5 means the concept positively influences the target.
    n_examples :
        Number of examples evaluated.
    n_positive :
        Number of examples with positive directional derivative.
    p_value :
        One-sided p-value from a binomial test against the null hypothesis
        that ``sensitivity = 0.5`` (the concept has no directional effect).
    per_example_sensitivities :
        Directional derivative ``∇target · v_c`` for each example,
        shape ``(n_examples,)``.
    """

    concept_name: str
    layer: int
    target: TransformerLogitTarget
    cav_direction: ConceptDirectionResult
    sensitivity: float
    n_examples: int
    n_positive: int
    p_value: float
    per_example_sensitivities: np.ndarray

    def __post_init__(self) -> None:
        if not (0.0 <= self.sensitivity <= 1.0 + 1e-9):
            raise ValueError(f"sensitivity must be in [0, 1], got {self.sensitivity}")
        if self.n_positive > self.n_examples:
            raise ValueError(f"n_positive ({self.n_positive}) > n_examples ({self.n_examples})")
        if self.per_example_sensitivities.shape != (self.n_examples,):
            raise ValueError(
                f"per_example_sensitivities shape {self.per_example_sensitivities.shape} != ({self.n_examples},)"
            )


@dataclass(frozen=True)
class TCAVResult:
    """Full TCAV analysis result for one concept-target pair.

    Parameters
    ----------
    scores :
        One :class:`TCAVScore` per seed (when ``n_seeds > 1``) or a single
        entry for the primary result.
    aggregate_score :
        Mean TCAV score across seeds.
    aggregate_ci95 :
        95 % confidence interval half-width across seeds.
    significance :
        ``"significant"`` if the aggregate score's corrected p-value is
        below the configured alpha, ``"not_significant"`` otherwise.
    corrected_p_value :
        P-value after Bonferroni correction for the number of concepts ×
        targets in the declared family of comparisons.
    n_random_concepts :
        Number of random (null) concepts evaluated.
    random_baseline_scores :
        TCAV scores for the random-concept baselines, shape
        ``(n_random_concepts,)``.
    random_baseline_mean :
        Mean TCAV score across random-concept baselines.
    random_baseline_std :
        Std of TCAV scores across random-concept baselines.
    empirical_p_value :
        Fraction of random-concept baselines with a TCAV score >= the
        real concept's aggregate TCAV score.
    intervention_agreement :
        Agreement rate with a matched-norm intervention cross-check
        (``None`` if not computed).  The fraction of interventions whose
        sign (output increase / decrease) matches the TCAV prediction.
    """

    scores: tuple[TCAVScore, ...]
    aggregate_score: float
    aggregate_ci95: float
    significance: str
    corrected_p_value: float
    n_random_concepts: int
    random_baseline_scores: np.ndarray
    random_baseline_mean: float
    random_baseline_std: float
    empirical_p_value: float
    intervention_agreement: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, TCAVScore):
                out[key] = asdict(value)
            elif isinstance(value, tuple):
                out[key] = [asdict(s) if isinstance(s, TCAVScore) else s for s in value]
            elif isinstance(value, (ConceptDirectionResult, TransformerLogitTarget)):
                out[key] = asdict(value)
            else:
                out[key] = value
        return out


# ---------------------------------------------------------------------------
# Part 5 — Optional model boundary (Task 3)
# ---------------------------------------------------------------------------
# Gradient and activation capture live in _tcav_model. These imports preserve
# the historical private test seams while keeping PyTorch out of this facade.


def _compute_transformer_layer_gradient(
    model: Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    layer: int,
    target: TransformerLogitTarget,
) -> np.ndarray:
    return compute_transformer_layer_gradient(model, input_ids, attention_mask, layer, target)


def _extract_layer_activation(
    model: Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    layer: int,
    batch_index: int = 0,
    position: int = -1,
) -> np.ndarray:
    return extract_layer_activation(model, input_ids, attention_mask, layer, batch_index, position)


# ---------------------------------------------------------------------------
# Part 6 — Main TCAV computation  (Tasks 3–5)
# ---------------------------------------------------------------------------


def compute_tcav(
    model: Any,
    target_layer: int,
    concept_dataset: ConceptDataset,
    input_ids_batch: np.ndarray,
    attention_mask_batch: np.ndarray,
    target: TransformerLogitTarget,
    *,
    direction_method: str = "mean_diff",
    n_bootstrap: int = 50,
    n_random_concepts: int = 50,
    n_seeds: int = 5,
    alpha: float = 0.05,
    n_concepts_family: int = 1,
    random_concept_seed: int = 42,
) -> TCAVResult:
    """Compute the TCAV score for one concept at one transformer layer.

    The steps are:

    1. Learn the concept direction from ``concept_dataset`` using the
       chosen method (``"mean_diff"`` or ``"linear_separator"``).
    2. For each example in the batch, compute the gradient of
       ``target`` w.r.t. activations at ``target_layer``.
    3. Compute the directional derivative ``∇target · v_c`` for each
       example and aggregate into the TCAV score.
    4. Repeat across seeds to estimate uncertainty.
    5. Evaluate random-concept baselines for null-distribution comparison.
    6. Compute empirical p-value and Bonferroni-corrected significance.

    Parameters
    ----------
    model :
        HuggingFace ``AutoModelForCausalLM`` on the correct device.
    target_layer :
        0-based layer index.
    concept_dataset :
        Concept vs reference examples (pre-computed activations at
        *target_layer* or input-space features — must match the
        dimensionality the direction is learned in).
    input_ids_batch :
        ``(batch_size, seq_len)`` int64 NumPy array — the examples to
        compute sensitivities on.
    attention_mask_batch :
        ``(batch_size, seq_len)`` int64 NumPy array.
    target :
        Scalar target: which token logit to differentiate.
    direction_method :
        ``"mean_diff"`` or ``"linear_separator"``.
    n_bootstrap :
        Bootstrap resamples for direction stability.
    n_random_concepts :
        Number of random (null) concept directions.
    n_seeds :
        Number of seeds for computing the aggregate TCAV score.
        Each seed uses a different bootstrap or classifier random state.
    alpha :
        Significance threshold after Bonferroni correction.
    n_concepts_family :
        Total number of concept–target comparisons in the declared
        family (used for Bonferroni correction).
    random_concept_seed :
        Base seed for random-concept generation.

    Returns
    -------
    TCAVResult
        Full analysis with scores, significance, and random baselines.
    """
    if direction_method == "mean_diff":
        cav_result = learn_mean_diff_direction(
            concept_dataset,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=0,
        )
    elif direction_method == "linear_separator":
        cav_result = learn_linear_separator_direction(
            concept_dataset,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=0,
        )
    else:
        raise ValueError(f"Unknown direction_method: {direction_method!r}")

    batch_size = input_ids_batch.shape[0]
    all_gradients: list[np.ndarray] = []
    for i in range(batch_size):
        grad = _compute_transformer_layer_gradient(
            model,
            input_ids=input_ids_batch[i : i + 1],
            attention_mask=attention_mask_batch[i : i + 1],
            layer=target_layer,
            target=target,
        )
        all_gradients.append(grad)

    grad_matrix = np.stack(all_gradients, axis=0)
    return assemble_tcav_result(
        concept_dataset=concept_dataset,
        target_layer=target_layer,
        target=target,
        grad_matrix=grad_matrix,
        initial_direction=cav_result,
        direction_method=direction_method,
        n_bootstrap=n_bootstrap,
        n_random_concepts=n_random_concepts,
        n_seeds=n_seeds,
        alpha=alpha,
        n_concepts_family=n_concepts_family,
        random_concept_seed=random_concept_seed,
    )


# ---------------------------------------------------------------------------
# Part 7 — Intervention cross-check  (Task 7)
# ---------------------------------------------------------------------------
def intervention_agreement(
    model: Any,
    target_layer: int,
    concept_direction: np.ndarray,
    input_ids_batch: np.ndarray,
    attention_mask_batch: np.ndarray,
    target: TransformerLogitTarget,
    *,
    strength: float = 1.0,
) -> float:
    """Cross-check TCAV sensitivity with a matched-norm intervention."""
    return _intervention_agreement(
        model=model,
        target_layer=target_layer,
        concept_direction=concept_direction,
        input_ids_batch=input_ids_batch,
        attention_mask_batch=attention_mask_batch,
        target=target,
        strength=strength,
    )


# ---------------------------------------------------------------------------
# Part 8 — Registry-constructable TCAV class  (Task 8)
# ---------------------------------------------------------------------------


class TCAVConfig(BaseModel):
    """Pydantic config for a :class:`TCAV` analysis run.

    Parameters
    ----------
    target_layer :
        Layer index where concept directions are learned and gradients are
        computed (0-based).
    direction_method :
        ``"mean_diff"`` or ``"linear_separator"``.
    n_bootstrap :
        Number of bootstrap resamples for direction stability estimates.
    n_random_concepts :
        Number of random (null) concept baselines.
    n_seeds :
        Number of seeds for computing the aggregate TCAV score.
    alpha :
        Significance threshold after Bonferroni correction.
    n_concepts_family :
        Total number of concept–target comparisons in the declared family.
    """

    target_layer: int = Field(default=8, ge=0, description="Layer index for concept direction and gradient")
    direction_method: str = Field(
        default="mean_diff",
        pattern="^(mean_diff|linear_separator)$",
        description="Direction learning method",
    )
    n_bootstrap: int = Field(default=50, ge=1, description="Bootstrap resamples for stability")
    n_random_concepts: int = Field(default=50, ge=1, description="Random concept baselines")
    n_seeds: int = Field(default=5, ge=1, description="Seeds for aggregate score")
    alpha: float = Field(default=0.05, gt=0, lt=1, description="Significance threshold")
    n_concepts_family: int = Field(default=1, ge=1, description="Comparisons in the declared family")


class TCAV:
    """Config-driven TCAV analysis entry point.

    Wraps :func:`compute_tcav` with a :class:`TCAVConfig` so that the
    analysis can be constructed via ``ObjectSpec`` / ``build_from_config``::

        from latent_anything.config import build_from_dict

        tcav = build_from_dict({"kind": "analysis", "name": "tcav",
                                "params": {"target_layer": 8, "direction_method": "mean_diff"}})
        result = tcav.compute(model=model, concept_dataset=ds, ...)

    Parameters
    ----------
    config : TCAVConfig, optional
        Analysis configuration.  Defaults to ``TCAVConfig()``.
    """

    def __init__(self, config: TCAVConfig | None = None, **kwargs: Any) -> None:
        # Support both direct config and ObjectSpec-style kwargs
        if kwargs:
            self._config = TCAVConfig(**kwargs)
        else:
            self._config = config if config is not None else TCAVConfig()

    @property
    def config(self) -> TCAVConfig:
        """Return the analysis configuration."""
        return self._config

    def compute(
        self,
        model: Any,
        concept_dataset: ConceptDataset,
        input_ids_batch: np.ndarray,
        attention_mask_batch: np.ndarray,
        target: TransformerLogitTarget,
        *,
        n_concepts_family: int | None = None,
    ) -> TCAVResult:
        """Run the full TCAV analysis.

        See :func:`compute_tcav` for parameter documentation.
        """
        cfg = self._config
        return compute_tcav(
            model=model,
            target_layer=cfg.target_layer,
            concept_dataset=concept_dataset,
            input_ids_batch=input_ids_batch,
            attention_mask_batch=attention_mask_batch,
            target=target,
            direction_method=cfg.direction_method,
            n_bootstrap=cfg.n_bootstrap,
            n_random_concepts=cfg.n_random_concepts,
            n_seeds=cfg.n_seeds,
            alpha=cfg.alpha,
            n_concepts_family=n_concepts_family if n_concepts_family is not None else cfg.n_concepts_family,
        )


# ---------------------------------------------------------------------------
# Part 9 — Convenience: build a concept dataset from raw text
# ---------------------------------------------------------------------------


def build_concept_dataset_from_text(
    integration: Any,
    concept_texts: list[str],
    reference_texts: list[str],
    concept_name: str,
    *,
    layer: int = 8,
    position: int = -1,
    max_length: int = 128,
    source: str = "",
) -> ConceptDataset:
    """Build a :class:`ConceptDataset` by extracting layer activations from text.

    Parameters
    ----------
    integration :
        A ``TransformerLMIntegration`` instance.
    concept_texts :
        Text prompts for the concept-positive set.
    reference_texts :
        Text prompts for the concept-negative (reference) set.
    concept_name :
        Label for this concept.
    layer :
        Transformer layer index to extract activations from.
    position :
        Token position to extract (-1 = last).
    max_length :
        Maximum tokenisation length.
    source :
        Optional dataset source identifier.

    Returns
    -------
    ConceptDataset
        Dataset with activations extracted from *integration*.
    """
    model, tokenizer, _ = integration._backend()  # type: ignore[reportPrivateUsage]

    def _extract(texts: list[str]) -> np.ndarray:
        acts: list[np.ndarray] = []
        for t in texts:
            encoded = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            ids_np = encoded["input_ids"].numpy()
            mask_np = encoded["attention_mask"].numpy()
            act = _extract_layer_activation(
                model,
                ids_np,
                mask_np,
                layer=layer,
                position=position,
            )
            acts.append(act)
        return np.stack(acts, axis=0)

    concept_acts = _extract(concept_texts)
    reference_acts = _extract(reference_texts)

    return ConceptDataset(
        concept_examples=concept_acts,
        reference_examples=reference_acts,
        concept_name=concept_name,
        source=source,
        representation_space=f"{integration.model_id}_layer_{layer}",
        model_version=integration.provenance,
    )
