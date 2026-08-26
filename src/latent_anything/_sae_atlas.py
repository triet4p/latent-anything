"""Internal feature ranking and portable atlas persistence."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from latent_anything.sae_evaluation import (
        FeatureAtlas,
        FeatureAtlasEntry,
        FeatureRanking,
        SAEEvaluationResult,
    )


def _coerce_labels(example_labels: Sequence[str | None], n_examples: int) -> tuple[str | None, ...]:
    values = tuple(None if label is None else str(label) for label in example_labels)
    if len(values) != n_examples:
        raise ValueError(f"expected {n_examples} example labels, got {len(values)}")
    return values


def rank_feature_examples(
    evaluation: SAEEvaluationResult,
    feature_index: int,
    *,
    k: int = 5,
    example_labels: Sequence[str | None] | None = None,
) -> FeatureRanking:
    """Rank top-activating and bottom-activating examples for one feature."""
    from latent_anything.sae_evaluation import FeatureRanking

    if not 0 <= feature_index < evaluation.config.n_components:
        raise ValueError(f"feature_index {feature_index} is outside [0, {evaluation.config.n_components})")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n_val = evaluation.val_activations.shape[0]
    labels = _coerce_labels(example_labels, n_val) if example_labels is not None else None
    activations = evaluation.val_activations[:, feature_index]
    k_effective = min(k, n_val)
    top_indices = np.argsort(activations)[-k_effective:][::-1]
    bottom_indices = np.argsort(activations)[:k_effective]
    return FeatureRanking(
        feature_index=feature_index,
        top_example_indices=tuple(int(index) for index in top_indices),
        bottom_example_indices=tuple(int(index) for index in bottom_indices),
        top_activations=tuple(float(activations[index]) for index in top_indices),
        bottom_activations=tuple(float(activations[index]) for index in bottom_indices),
        top_labels=tuple(labels[index] for index in top_indices) if labels is not None else (None,) * k_effective,
        bottom_labels=tuple(labels[index] for index in bottom_indices) if labels is not None else (None,) * k_effective,
    )


def build_feature_atlas(
    evaluation: SAEEvaluationResult,
    *,
    feature_indices: Sequence[int] | None = None,
    k_examples: int = 5,
    k_decoder_dims: int = 10,
    example_labels: Sequence[str | None] | None = None,
) -> FeatureAtlas:
    """Build a JSON-serializable, frontend-independent feature atlas."""
    from latent_anything.sae_evaluation import FeatureAtlas, FeatureAtlasEntry

    n_components = evaluation.config.n_components
    selected = tuple(range(n_components)) if feature_indices is None else tuple(int(i) for i in feature_indices)
    for index in selected:
        if not 0 <= index < n_components:
            raise ValueError(f"feature index {index} is outside [0, {n_components})")
    labels = _coerce_labels(example_labels, evaluation.val_activations.shape[0]) if example_labels is not None else None
    entries: list[FeatureAtlasEntry] = []
    for index in selected:
        ranking = rank_feature_examples(evaluation, index, k=k_examples, example_labels=labels)
        decoder_column = np.asarray(evaluation.decoder_weights[:, index], dtype=np.float64)
        top_dim_indices = np.argsort(np.abs(decoder_column))[::-1][:k_decoder_dims]
        top_dims = tuple({"dim_index": int(dim), "weight": float(decoder_column[dim])} for dim in top_dim_indices)
        metrics = evaluation.features[index]
        top_examples = tuple(
            {
                "rank": rank + 1,
                "example_index": int(ranking.top_example_indices[rank]),
                "activation": ranking.top_activations[rank],
                "label": ranking.top_labels[rank],
            }
            for rank in range(len(ranking.top_example_indices))
        )
        bottom_examples = tuple(
            {
                "rank": rank + 1,
                "example_index": int(ranking.bottom_example_indices[rank]),
                "activation": ranking.bottom_activations[rank],
                "label": ranking.bottom_labels[rank],
            }
            for rank in range(len(ranking.bottom_example_indices))
        )
        entries.append(
            FeatureAtlasEntry(
                feature_index=index,
                is_dead=metrics.is_dead,
                activation_frequency=metrics.activation_frequency,
                mean_activation=metrics.mean_activation,
                decoder_norm=metrics.decoder_norm,
                encoder_norm=metrics.encoder_norm,
                top_examples=top_examples,
                bottom_examples=bottom_examples,
                top_decoder_dims=top_dims,
            )
        )
    return FeatureAtlas(
        entries=tuple(entries),
        n_components=n_components,
        n_examples=evaluation.val_activations.shape[0],
        source_representation_identity=evaluation.source_representation_identity,
        provenance=dict(evaluation.provenance),
    )


def save_feature_atlas(atlas: FeatureAtlas, path: str | os.PathLike[str]) -> None:
    """Write the feature atlas to a portable JSON artifact."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(atlas.to_dict(), handle, indent=2, sort_keys=True)


def load_feature_atlas(path: str | os.PathLike[str]) -> FeatureAtlas:
    """Load a feature atlas written by :func:`save_feature_atlas`."""
    from latent_anything.sae_evaluation import FeatureAtlas, FeatureAtlasEntry

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    entries = tuple(
        FeatureAtlasEntry(
            feature_index=int(entry["feature_index"]),
            is_dead=bool(entry["is_dead"]),
            activation_frequency=float(entry["activation_frequency"]),
            mean_activation=float(entry["mean_activation"]),
            decoder_norm=float(entry["decoder_norm"]),
            encoder_norm=float(entry["encoder_norm"]),
            top_examples=tuple(dict(item) for item in entry["top_examples"]),
            bottom_examples=tuple(dict(item) for item in entry["bottom_examples"]),
            top_decoder_dims=tuple(dict(item) for item in entry["top_decoder_dims"]),
        )
        for entry in data["entries"]
    )
    return FeatureAtlas(
        entries=entries,
        n_components=int(data["n_components"]),
        n_examples=int(data["n_examples"]),
        source_representation_identity=str(data["source_representation_identity"]),
        provenance=dict(data["provenance"]),
    )
