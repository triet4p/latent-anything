"""Train/holdout TCAV evaluation over cached transformer values."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from scripts._m14_l04_tcav_metrics import group_means, metric
from scripts._m14_l04_tcav_runtime import intervened_margin, seed_everything, task_margin

CONCEPT_FACTOR = "tone_positive"


def learn_direction(activations: np.ndarray, labels: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Reuse the project mean-difference primitive with a tiny-fixture fallback."""
    from latent_anything.tcav import ConceptDataset, learn_mean_diff_direction

    dataset = ConceptDataset(activations[labels == 1], activations[labels == 0], CONCEPT_FACTOR)
    try:
        result = learn_mean_diff_direction(dataset, n_bootstrap=1, bootstrap_seed=seed)
        return np.asarray(result.direction, dtype=np.float64)
    except ValueError as exc:
        if "stability must be in [0, 1]" not in str(exc):
            raise
        raw = np.mean(dataset.concept_examples, axis=0) - np.mean(dataset.reference_examples, axis=0)
        norm = float(np.linalg.norm(raw))
        if norm <= 1e-15:
            raise ValueError("TCAV concept direction is zero") from exc
        return (raw / norm).astype(np.float64)


def group_accuracy(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[bool], labels: Sequence[int]
) -> tuple[list[float], list[float]]:
    values = [
        {"group_id": row["group_id"], "correct": float(prediction == bool(label))}
        for row, prediction, label in zip(rows, predictions, labels, strict=True)
    ]
    return group_means(values, "correct"), [float(item["correct"]) for item in values]


def evaluate_cached(
    *,
    train_activations: np.ndarray,
    train_labels: np.ndarray,
    holdout_activations: np.ndarray,
    holdout_rows: Sequence[Mapping[str, Any]],
    holdout_labels: Sequence[int],
    real_rows: Sequence[Mapping[str, Any]],
    gradients: np.ndarray,
    seeds: Sequence[int],
    torch: Any,
    null_count: int,
    direction_fn: Callable[..., np.ndarray] = learn_direction,
) -> dict[str, Any]:
    """Fit directions on train values and evaluate independent heldout groups."""
    if tuple(seeds) != (17, 29, 41, 53, 67) or null_count != 99:
        raise ValueError("TCAV evaluation requires the frozen five seeds and 99 nulls")
    holdout_groups = sorted({str(item["group_id"]) for item in holdout_rows})
    train_groups = sorted({str(item["group_id"]) for item in real_rows if item["split"] == "train"})
    by_seed: list[dict[str, Any]] = []
    grouped_scores: dict[str, list[float]] = {group: [] for group in holdout_groups}
    grouped_accuracy: dict[str, list[float]] = {group: [] for group in holdout_groups}
    first_row_correct: list[float] = []
    null_scores: list[float] = []
    null_families: dict[str, list[float]] = {"shuffled": [], "random": [], "matched": []}
    null_family_counts = {"shuffled": 0, "random": 0, "matched": 0}
    null_seen = 0
    last_direction = np.zeros(train_activations.shape[1], dtype=np.float64)
    for seed in seeds:
        seed_everything(seed, torch)
        direction = direction_fn(train_activations, train_labels, seed=seed)
        last_direction = direction
        projections = holdout_activations @ direction
        centre = float(
            np.mean(
                np.concatenate(
                    (train_activations[train_labels == 1] @ direction, train_activations[train_labels == 0] @ direction)
                )
            )
        )
        predictions = (projections > centre).tolist()
        accuracy_groups, row_correct = group_accuracy(holdout_rows, predictions, holdout_labels)
        if seed == seeds[0]:
            first_row_correct = row_correct
        derivative_groups = group_means(
            [
                {"group_id": item["group_id"], "positive": float(np.dot(gradients[index], direction) > 0.0)}
                for index, item in enumerate(real_rows)
                if item["split"] == "holdout"
            ],
            "positive",
        )
        for group, value in zip(holdout_groups, accuracy_groups, strict=True):
            grouped_accuracy[group].append(value)
        for group, value in zip(holdout_groups, derivative_groups, strict=True):
            grouped_scores[group].append(value)
        seed_nulls: list[float] = []
        seed_null_norms: list[float] = []
        seed_null_families: list[str] = []
        rng = np.random.default_rng(seed)
        per_seed_nulls = 20 if seed != seeds[-1] else null_count - 80
        for _ in range(per_seed_nulls):
            family = ("shuffled", "random", "matched")[null_seen % 3]
            if family == "shuffled":
                null_direction = direction_fn(
                    train_activations, train_labels[rng.permutation(len(train_labels))], seed=seed
                )
            else:
                null_direction = rng.normal(size=direction.shape).astype(np.float64)
                if family == "matched":
                    random_norm = np.linalg.norm(null_direction)
                    if random_norm <= 0.0:
                        raise ValueError("random null direction has zero norm")
                    null_direction = (null_direction / random_norm) * float(np.linalg.norm(direction))
            null_value = float(
                np.mean(
                    group_means(
                        [
                            {
                                "group_id": item["group_id"],
                                "positive": float(np.dot(gradients[index], null_direction) > 0.0),
                            }
                            for index, item in enumerate(real_rows)
                            if item["split"] == "holdout"
                        ],
                        "positive",
                    )
                )
            )
            null_scores.append(null_value)
            null_families[family].append(null_value)
            null_family_counts[family] += 1
            seed_null_families.append(family)
            seed_nulls.append(null_value)
            seed_null_norms.append(float(np.linalg.norm(null_direction)))
            null_seen += 1
        by_seed.append(
            {
                "seed": seed,
                "train_groups": train_groups,
                "holdout_groups": holdout_groups,
                "heldout_group_accuracy": accuracy_groups,
                "heldout_row_correct": row_correct,
                "heldout_group_tcav": derivative_groups,
                "null_group_scores": seed_nulls,
                "null_direction_norms": seed_null_norms,
                "null_families": seed_null_families,
            }
        )
    return {
        "by_seed": by_seed,
        "grouped_accuracy": [float(np.mean(values)) for values in grouped_accuracy.values()],
        "grouped_scores": [float(np.mean(values)) for values in grouped_scores.values()],
        "first_row_correct": first_row_correct,
        "null_scores": null_scores,
        "null_families": null_families,
        "null_family_counts": null_family_counts,
        "direction": last_direction,
    }


def evaluate_interventions(
    *,
    model: Any,
    holdout_rows: Sequence[Mapping[str, Any]],
    real_rows: Sequence[Mapping[str, Any]],
    gradients: np.ndarray,
    off_target_gradients: np.ndarray,
    direction: np.ndarray,
    target_true: int,
    target_false: int,
    intervention_threshold: float,
) -> dict[str, Any]:
    """Run genuine +/- hidden-state, off-target-token, and identity controls."""
    agreement_values: list[float] = []
    for item in holdout_rows:
        source_index = next(i for i, candidate in enumerate(real_rows) if candidate["row_id"] == item["row_id"])
        plus = intervened_margin(
            model, item, layer=6, direction=direction, target_token=target_true, other_token=target_false, strength=1.0
        )
        minus = intervened_margin(
            model, item, layer=6, direction=direction, target_token=target_true, other_token=target_false, strength=-1.0
        )
        derivative = float(np.dot(gradients[source_index], direction))
        observed_sign = np.sign(plus - minus)
        agreement_values.append(float(observed_sign == np.sign(derivative) and observed_sign != 0.0))
    intervention = metric(
        group_means(
            [
                {"group_id": item["group_id"], "agreement": value}
                for item, value in zip(holdout_rows, agreement_values, strict=True)
            ],
            "agreement",
        ),
        seed=41,
        threshold=intervention_threshold,
        comparator=">",
    )
    off_target_groups = group_means(
        [
            {"group_id": item["group_id"], "value": float(np.dot(gradient, direction) > 0.0)}
            for item, gradient in zip(holdout_rows, off_target_gradients, strict=True)
        ],
        "value",
    )
    zero_differences = []
    for item in holdout_rows:
        baseline = intervened_margin(
            model, item, layer=6, direction=direction, target_token=target_true, other_token=target_false, strength=0.0
        )
        zero_differences.append(
            abs(baseline - task_margin(model, item, target_token=target_true, other_token=target_false))
        )
    zero_groups = group_means(
        [
            {"group_id": item["group_id"], "value": value}
            for item, value in zip(holdout_rows, zero_differences, strict=True)
        ],
        "value",
    )
    return {
        "intervention": intervention,
        "intervention_groups": group_means(
            [
                {"group_id": item["group_id"], "agreement": value}
                for item, value in zip(holdout_rows, agreement_values, strict=True)
            ],
            "agreement",
        ),
        "off_target_groups": off_target_groups,
        "zero_groups": zero_groups,
    }


__all__ = ["CONCEPT_FACTOR", "evaluate_cached", "evaluate_interventions", "group_accuracy", "learn_direction"]
