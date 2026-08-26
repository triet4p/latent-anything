"""Internal fitted-SAE metrics and cross-seed feature matching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from latent_anything.methods.sae import SAE

if TYPE_CHECKING:
    from latent_anything.sae_evaluation import (
        SAEConfig,
        SAEEvaluationResult,
        SAEStabilityResult,
    )


def _as_readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def evaluate_fitted(
    sae: SAE,
    train: np.ndarray,
    val: np.ndarray,
    config: SAEConfig,
    source_identity: str,
    provenance: dict[str, Any] | None,
) -> SAEEvaluationResult:
    """Compute typed reconstruction, activity, and geometry metrics."""
    from latent_anything.sae_evaluation import SAEEvaluationResult, SAEFeatureMetrics

    val_activations = np.asarray(sae.transform(val))
    train_reconstruction = np.asarray(sae.reconstruct(train))
    val_reconstruction = np.asarray(sae.reconstruct(val))
    train_mse = float(np.mean((train_reconstruction - train) ** 2))
    val_mse = float(np.mean((val_reconstruction - val) ** 2))
    frequencies = np.asarray((val_activations > 0).mean(axis=0), dtype=np.float64)
    mean_activations = np.asarray(val_activations.mean(axis=0), dtype=np.float64)
    positive_counts = (val_activations > 0).sum(axis=0).astype(np.float64)
    positive_sums = np.where(positive_counts > 0, val_activations.sum(axis=0), 0.0)
    mean_positive = np.divide(
        positive_sums,
        positive_counts,
        out=np.zeros_like(positive_sums),
        where=positive_counts > 0,
    )
    state = sae.state_dict()
    decoder_weights = np.asarray(state["decoder_weight"], dtype=np.float64)
    encoder_weights = np.asarray(state["encoder_weight"], dtype=np.float64)
    decoder_norms = np.asarray(np.linalg.norm(decoder_weights, axis=0), dtype=np.float64)
    encoder_norms = np.asarray(np.linalg.norm(encoder_weights, axis=1), dtype=np.float64)
    is_dead = np.asarray(frequencies < config.dead_frequency_threshold, dtype=bool)
    n_dead = int(is_dead.sum())
    features = tuple(
        SAEFeatureMetrics(
            feature_index=int(index),
            activation_frequency=float(frequencies[index]),
            mean_activation=float(mean_activations[index]),
            mean_positive_activation=float(mean_positive[index]),
            decoder_norm=float(decoder_norms[index]),
            encoder_norm=float(encoder_norms[index]),
            is_dead=bool(is_dead[index]),
        )
        for index in range(config.n_components)
    )
    merged_provenance = dict(provenance or {})
    merged_provenance.update(
        {
            "split": "train_validation",
            "val_fraction": config.val_fraction,
            "random_state": config.random_state,
            "n_epochs": config.n_epochs,
            "l1_coef": config.l1_coef,
            "train_samples": int(train.shape[0]),
            "val_samples": int(val.shape[0]),
            "input_features": int(train.shape[1]),
        }
    )
    return SAEEvaluationResult(
        config=config,
        n_train=int(train.shape[0]),
        n_val=int(val.shape[0]),
        reconstruction_mse=val_mse,
        train_reconstruction_mse=train_mse,
        mean_l0=float((val_activations > 0).sum(axis=1).mean()),
        mean_l1=float(val_activations.sum(axis=1).mean()),
        n_dead_features=n_dead,
        dead_fraction=float(n_dead / config.n_components),
        activation_frequencies=_as_readonly(frequencies),
        decoder_norms=_as_readonly(decoder_norms),
        features=features,
        val_activations=_as_readonly(val_activations),
        decoder_weights=_as_readonly(decoder_weights),
        source_representation_identity=source_identity,
        provenance=merged_provenance,
    )


def feature_direction(evaluation: SAEEvaluationResult, feature_index: int) -> np.ndarray:
    """Return the unit-norm input-space direction for one feature."""
    column = np.asarray(evaluation.decoder_weights[:, feature_index], dtype=np.float64)
    norm = float(np.linalg.norm(column))
    return column.copy() if norm < 1e-12 else column / norm


def match_by_decoder_cosine(reference: np.ndarray, other: np.ndarray, threshold: float) -> list[tuple[int, float]]:
    """Greedily match feature slots by decoder direction cosine."""
    reference_unit = reference / np.maximum(np.linalg.norm(reference, axis=0), 1e-12)[None, :]
    other_unit = other / np.maximum(np.linalg.norm(other, axis=0), 1e-12)[None, :]
    cosine_matrix = np.asarray(reference_unit.T @ other_unit, dtype=np.float64)
    used: set[int] = set()
    matched: list[tuple[int, float]] = []
    for i in range(cosine_matrix.shape[0]):
        best_j = -1
        best_cosine = -1.0
        for j in range(cosine_matrix.shape[1]):
            if j in used:
                continue
            cosine = float(cosine_matrix[i, j])
            if cosine > best_cosine:
                best_cosine = cosine
                best_j = j
        if best_j >= 0 and best_cosine >= threshold:
            used.add(best_j)
            matched.append((i, best_cosine))
    return matched


def assemble_stability(
    reports: tuple[SAEEvaluationResult, ...],
    config: SAEConfig,
    seed_values: tuple[int, ...],
) -> SAEStabilityResult:
    """Assemble cross-seed stability after decoder-direction matching."""
    from latent_anything.sae_evaluation import SAEStabilityResult

    reference = reports[0]
    matched_cosines: list[float] = []
    n_matched_total = 0
    n_pairs = 0
    for other in reports[1:]:
        matched = match_by_decoder_cosine(
            reference.decoder_weights,
            other.decoder_weights,
            config.matching_cosine_threshold,
        )
        n_pairs += 1
        n_matched_total += len(matched)
        matched_cosines.extend(cosine for _, cosine in matched)
    if n_pairs == 0:
        raise ValueError("stability analysis produced no seed comparisons")
    cosine_values = tuple(matched_cosines)
    return SAEStabilityResult(
        seeds=seed_values,
        reconstruction_mses=tuple(report.reconstruction_mse for report in reports),
        n_components=config.n_components,
        n_features_matched=n_matched_total,
        mean_matched_cosine=float(np.mean(cosine_values)) if cosine_values else 0.0,
        min_matched_cosine=float(np.min(cosine_values)) if cosine_values else 0.0,
        alignment_quality=float(n_matched_total / (n_pairs * config.n_components)),
        matched_cosines=cosine_values,
    )
