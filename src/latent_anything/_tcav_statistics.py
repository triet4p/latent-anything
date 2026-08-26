"""Internal TCAV scoring, controls, and result assembly.

This module owns the statistical half of TCAV: directional score assembly,
bootstrap-seed aggregation, random-concept controls, p-values, and
significance correction.  Model execution stays in ``_tcav_model`` and the
public orchestration facade remains in ``tcav``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from latent_anything.tcav import (
        ConceptDataset,
        ConceptDirectionResult,
        TCAVResult,
        TransformerLogitTarget,
    )


def _normalize_direction(value: np.ndarray) -> np.ndarray:
    """Return the historical unit-normalization used by TCAV directions."""
    norm = np.linalg.norm(value)
    if norm < 1e-15:
        return value.copy()
    return value.astype(np.float64) / norm


def _binomial_p_value(n_positive: int, n_total: int, p_null: float = 0.5) -> float:
    """One-sided p-value: P(X >= n_positive) under Binomial(n_total, p_null)."""
    from scipy import stats  # type: ignore[reportMissingTypeStubs]

    return float(stats.binom.sf(n_positive - 1, n_total, p_null))


def assemble_tcav_result(
    concept_dataset: ConceptDataset,
    target_layer: int,
    target: TransformerLogitTarget,
    grad_matrix: np.ndarray,
    initial_direction: ConceptDirectionResult,
    *,
    direction_method: str,
    n_bootstrap: int,
    n_random_concepts: int,
    n_seeds: int,
    alpha: float,
    n_concepts_family: int,
    random_concept_seed: int,
) -> TCAVResult:
    """Assemble TCAV scores from gradients and a learned initial direction."""
    from latent_anything.tcav import (
        TCAVResult,
        TCAVScore,
        learn_linear_separator_direction,
        learn_mean_diff_direction,
    )

    batch_size = grad_matrix.shape[0]
    sensitivities = grad_matrix @ initial_direction.direction
    n_positive = int(np.sum(sensitivities > 0))
    sensitivity_fraction = n_positive / batch_size
    seed_results: list[TCAVScore] = [
        TCAVScore(
            concept_name=concept_dataset.concept_name,
            layer=target_layer,
            target=target,
            cav_direction=initial_direction,
            sensitivity=sensitivity_fraction,
            n_examples=batch_size,
            n_positive=n_positive,
            p_value=_binomial_p_value(n_positive, batch_size),
            per_example_sensitivities=sensitivities.copy(),
        )
    ]
    seed_scores: list[float] = [sensitivity_fraction]

    for seed_offset in range(1, n_seeds):
        if direction_method == "mean_diff":
            cav_s = learn_mean_diff_direction(
                concept_dataset,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=seed_offset * 1000,
            )
        else:
            cav_s = learn_linear_separator_direction(
                concept_dataset,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=seed_offset * 1000,
                split_seed=seed_offset * 1000,
            )
        sens_s = grad_matrix @ cav_s.direction
        n_pos_s = int(np.sum(sens_s > 0))
        frac_s = n_pos_s / batch_size
        seed_scores.append(frac_s)
        seed_results.append(
            TCAVScore(
                concept_name=concept_dataset.concept_name,
                layer=target_layer,
                target=target,
                cav_direction=cav_s,
                sensitivity=frac_s,
                n_examples=batch_size,
                n_positive=n_pos_s,
                p_value=_binomial_p_value(n_pos_s, batch_size),
                per_example_sensitivities=sens_s.copy(),
            )
        )

    aggregate_score = float(np.mean(seed_scores))
    aggregate_ci95 = float(1.96 * np.std(seed_scores, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0

    rng = np.random.default_rng(random_concept_seed)
    random_scores: list[float] = []
    n_total = concept_dataset.n_concept + concept_dataset.n_reference
    all_x = np.concatenate([concept_dataset.concept_examples, concept_dataset.reference_examples], axis=0)
    for random_index in range(n_random_concepts):
        perm = rng.permutation(n_total)
        random_concept_x = all_x[perm[: concept_dataset.n_concept]]
        random_reference_x = all_x[perm[concept_dataset.n_concept :]]
        if direction_method == "mean_diff":
            random_dir = _normalize_direction(random_concept_x.mean(axis=0) - random_reference_x.mean(axis=0))
        else:
            from sklearn.linear_model import LogisticRegression  # type: ignore[reportMissingTypeStubs]

            random_y = np.concatenate([np.ones(random_concept_x.shape[0]), np.zeros(random_reference_x.shape[0])])
            classifier = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=random_index,
            )
            classifier.fit(np.concatenate([random_concept_x, random_reference_x], axis=0), random_y)
            random_dir = _normalize_direction(np.asarray(classifier.coef_[0]))
        random_sens = grad_matrix @ random_dir
        random_scores.append(int(np.sum(random_sens > 0)) / batch_size)

    random_arr = np.asarray(random_scores, dtype=np.float64)
    random_mean = float(random_arr.mean())
    random_std = float(random_arr.std(ddof=1) if n_random_concepts > 1 else 0.0)
    empirical_p = float(np.mean(random_arr >= aggregate_score))
    corrected_p = min(1.0, empirical_p * n_concepts_family)
    significance = "significant" if corrected_p < alpha else "not_significant"

    return TCAVResult(
        scores=tuple(seed_results),
        aggregate_score=aggregate_score,
        aggregate_ci95=aggregate_ci95,
        significance=significance,
        corrected_p_value=corrected_p,
        n_random_concepts=n_random_concepts,
        random_baseline_scores=random_arr.copy(),
        random_baseline_mean=random_mean,
        random_baseline_std=random_std,
        empirical_p_value=empirical_p,
        intervention_agreement=None,
    )
