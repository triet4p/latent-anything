"""Sparse-autoencoder feature evaluation for interpretability claims.

This module evaluates the features of a fitted :class:`~latent_anything.methods.SAE`
on four axes:

1. **Reconstruction fidelity** — train/validation MSE so the fit does not
   look at the data it is scored on.
2. **Sparsity / activity** — L0/L1 activity, per-feature activation
   frequency, dead-feature detection, and decoder/encoder norms.
3. **Cross-seed stability** — features are *matched* across seeds by decoder
   direction cosine similarity instead of comparing arbitrary feature
   indices directly (feature indices permute freely between fits).
4. **Semantic usefulness** — a bounded cross-check against probes, concept
   directions, and causal steering/patching on a transformer seam.

The module also produces a **portable feature-atlas artifact**: a JSON
document with per-feature summaries, top/bottom example indices, and decoder
top-contributions, independent of any visualization frontend.

Design notes
------------
- PyTorch gradients are computed internally and never exposed.
- The analysis owns its lifecycle under the semantic ``analysis`` registry
  kind; it does not follow the ``Method`` fit/transform lifecycle.
- Attribution/causal signals here are observational evidence, not causal
  proof; strong claims must be paired with intervention agreement.
"""

# torch has incomplete type stubs — these warnings are noise.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from latent_anything.methods.sae import SAE
from latent_anything.probes import LinearProbe, LinearProbeConfig
from latent_anything.tcav import TransformerLogitTarget


class SAEConfig(BaseModel):
    """Validated, deterministic configuration for SAE feature evaluation."""

    n_components: int = Field(default=64, ge=1, description="Number of sparse latent features")
    l1_coef: float = Field(
        default=0.1,
        gt=0,
        description="L1 sparsity penalty on feature activations (normalized per element)",
    )
    learning_rate: float = Field(default=1e-2, gt=0, description="Adam learning rate")
    n_epochs: int = Field(default=500, ge=1, description="Training epochs on the train split")
    random_state: int = Field(default=0, ge=0, description="Seed for the train/validation split and fit")
    val_fraction: float = Field(default=0.2, ge=0, lt=1, description="Held-out validation fraction")
    min_val_samples: int = Field(default=16, ge=1, description="Minimum validation split size")
    dead_frequency_threshold: float = Field(
        default=1e-4, ge=0, le=1, description="Activation-frequency below this marks a feature dead"
    )
    matching_cosine_threshold: float = Field(
        default=0.5, ge=0, le=1, description="Minimum cosine for a valid cross-seed feature match"
    )


@dataclass(frozen=True)
class SAEFeatureMetrics:
    """Per-feature activity and geometry statistics."""

    feature_index: int
    activation_frequency: float
    mean_activation: float
    mean_positive_activation: float
    decoder_norm: float
    encoder_norm: float
    is_dead: bool


@dataclass(frozen=True)
class SAEEvaluationResult:
    """Typed result of fitting an SAE and evaluating it on held-out data."""

    config: SAEConfig
    n_train: int
    n_val: int
    reconstruction_mse: float
    train_reconstruction_mse: float
    mean_l0: float
    mean_l1: float
    n_dead_features: int
    dead_fraction: float
    activation_frequencies: np.ndarray
    decoder_norms: np.ndarray
    features: tuple[SAEFeatureMetrics, ...]
    val_activations: np.ndarray
    decoder_weights: np.ndarray
    source_representation_identity: str
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.activation_frequencies.ndim != 1:
            raise ValueError("activation_frequencies must be one-dimensional")
        if self.activation_frequencies.shape[0] != self.config.n_components:
            raise ValueError("activation_frequencies must have one entry per feature")
        if self.decoder_norms.shape != self.activation_frequencies.shape:
            raise ValueError("decoder_norms must match activation_frequencies")
        if self.val_activations.ndim != 2:
            raise ValueError("val_activations must be 2D")
        if self.val_activations.shape[1] != self.config.n_components:
            raise ValueError("val_activations must have one column per feature")
        if self.decoder_weights.ndim != 2:
            raise ValueError("decoder_weights must be 2D")
        if self.decoder_weights.shape[1] != self.config.n_components:
            raise ValueError("decoder_weights must have one column per feature")
        if len(self.features) != self.config.n_components:
            raise ValueError("features must have one entry per feature")
        for array in (self.activation_frequencies, self.decoder_norms, self.val_activations, self.decoder_weights):
            array.setflags(write=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary without the large activation matrices."""
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "reconstruction_mse": self.reconstruction_mse,
            "train_reconstruction_mse": self.train_reconstruction_mse,
            "mean_l0": self.mean_l0,
            "mean_l1": self.mean_l1,
            "n_dead_features": self.n_dead_features,
            "dead_fraction": self.dead_fraction,
            "activation_frequencies": self.activation_frequencies.tolist(),
            "decoder_norms": self.decoder_norms.tolist(),
            "feature_summaries": [
                {
                    "feature_index": feature.feature_index,
                    "activation_frequency": feature.activation_frequency,
                    "is_dead": feature.is_dead,
                }
                for feature in self.features
            ],
            "source_representation_identity": self.source_representation_identity,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class SAEStabilityResult:
    """Cross-seed feature stability after decoder-direction matching."""

    seeds: tuple[int, ...]
    reconstruction_mses: tuple[float, ...]
    n_components: int
    n_features_matched: int
    mean_matched_cosine: float
    min_matched_cosine: float
    alignment_quality: float
    matched_cosines: tuple[float, ...]
    method: Literal["decoder_matching"] = "decoder_matching"


@dataclass(frozen=True)
class FeatureRanking:
    """Ranked top-activating and bottom-activating examples for one feature."""

    feature_index: int
    top_example_indices: tuple[int, ...]
    bottom_example_indices: tuple[int, ...]
    top_activations: tuple[float, ...]
    bottom_activations: tuple[float, ...]
    top_labels: tuple[str | None, ...]
    bottom_labels: tuple[str | None, ...]


@dataclass(frozen=True)
class FeatureCrossCheck:
    """Probe, concept, and causal steering cross-checks for one feature."""

    feature_index: int
    probe_accuracy: float | None
    shuffled_label_accuracy: float | None
    concept_sensitivity: float | None
    intervention_effect: float | None
    intervention_agreement: float | None
    n_examples_checked: int
    has_causal_evidence: bool


@dataclass(frozen=True)
class FeatureAtlasEntry:
    """One queryable feature entry in the portable atlas."""

    feature_index: int
    is_dead: bool
    activation_frequency: float
    mean_activation: float
    decoder_norm: float
    encoder_norm: float
    top_examples: tuple[dict[str, Any], ...]
    bottom_examples: tuple[dict[str, Any], ...]
    top_decoder_dims: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FeatureAtlas:
    """Portable, queryable feature-atlas data artifact."""

    entries: tuple[FeatureAtlasEntry, ...]
    n_components: int
    n_examples: int
    source_representation_identity: str
    provenance: dict[str, Any]

    def entry(self, feature_index: int) -> FeatureAtlasEntry:
        """Return the atlas entry for *feature_index*."""
        for candidate in self.entries:
            if candidate.feature_index == feature_index:
                return candidate
        raise KeyError(f"atlas has no entry for feature index {feature_index}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the whole atlas."""
        return {
            "schema": "latent-anything/feature-atlas-v1",
            "n_components": self.n_components,
            "n_examples": self.n_examples,
            "source_representation_identity": self.source_representation_identity,
            "provenance": dict(self.provenance),
            "entries": [asdict(entry) for entry in self.entries],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_batch(data: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got {values.ndim}D")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and feature")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _as_readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _evaluate_fitted(
    sae: SAE,
    train: np.ndarray,
    val: np.ndarray,
    config: SAEConfig,
    source_identity: str,
    provenance: dict[str, Any] | None,
) -> SAEEvaluationResult:
    """Compute all evaluation metrics from one already-fitted SAE."""
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
    l0 = float((val_activations > 0).sum(axis=1).mean())
    l1 = float(val_activations.sum(axis=1).mean())

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
        mean_l0=l0,
        mean_l1=l1,
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


def _feature_direction(evaluation: SAEEvaluationResult, feature_index: int) -> np.ndarray:
    """Return the unit-norm input-space direction for *feature_index*."""
    column = np.asarray(evaluation.decoder_weights[:, feature_index], dtype=np.float64)
    norm = float(np.linalg.norm(column))
    if norm < 1e-12:
        return column.copy()
    return column / norm


def _match_by_decoder_cosine(
    reference: np.ndarray,
    other: np.ndarray,
    threshold: float,
) -> list[tuple[int, float]]:
    """Greedily match reference decoder columns to *other* by cosine similarity.

    Feature indices are not compared directly because an SAE permutes its
    feature slots between fits; only the decoder direction is meaningful.
    Each reference feature is matched to the best *unclaimed* other-seed
    feature whose cosine is at least *threshold*.
    """
    reference_unit = reference / np.maximum(np.linalg.norm(reference, axis=0), 1e-12)[None, :]
    other_unit = other / np.maximum(np.linalg.norm(other, axis=0), 1e-12)[None, :]
    cosine_matrix = np.asarray(reference_unit.T @ other_unit, dtype=np.float64)
    used: set[int] = set()
    matched: list[tuple[int, float]] = []
    n = cosine_matrix.shape[0]
    for i in range(n):
        best_j = -1
        best_cosine = -1.0
        for j in range(n):
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


def _gradient_at_layer(
    model: Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    layer: int,
    target: TransformerLogitTarget,
) -> np.ndarray:
    """Gradient of *target* logit w.r.t. the residual activation at *layer*."""
    import torch

    device = next(model.parameters()).device
    ids_t = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    mask_t = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
    activation: dict[str, Any] = {}

    def _hook(_module: Any, _inputs: Any, output: Any) -> None:
        out = output if hasattr(output, "shape") else output[0]
        out.retain_grad()
        activation["selected"] = out

    handle = None
    for name, module in model.named_modules():
        if name == f"transformer.h.{layer}":
            handle = module.register_forward_hook(_hook)
            break
    if handle is None:
        raise ValueError(f"Layer {layer} (transformer.h.{layer}) was not found in the model")

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = model(input_ids=ids_t, attention_mask=mask_t, output_hidden_states=False)
        batch = target.batch_index
        position = target.position if target.position >= 0 else int(outputs.logits.shape[1]) + target.position
        scalar = outputs.logits[batch, position, target.token_id]
        scalar.backward()
        if "selected" not in activation:
            raise RuntimeError(f"Hook at transformer.h.{layer} did not fire")
        selected = activation["selected"]
        if selected.grad is None:
            raise RuntimeError("gradient was not retained at the selected layer")
        grad = selected.grad[batch, position].detach().cpu().numpy()
        return np.asarray(grad, dtype=np.float64)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def _target_value(
    model: Any, input_ids: np.ndarray, attention_mask: np.ndarray, target: TransformerLogitTarget
) -> float:
    """Forward-only scalar target logit without any intervention."""
    import torch

    device = next(model.parameters()).device
    ids_t = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    mask_t = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids_t, attention_mask=mask_t, output_hidden_states=False).logits
    position = target.position if target.position >= 0 else int(logits.shape[1]) + target.position
    return float(logits[target.batch_index, position, target.token_id].item())


def _target_with_intervention(
    model: Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    target: TransformerLogitTarget,
    layer: int,
    direction: np.ndarray,
) -> float:
    """Scalar target logit after adding *direction* to the layer activation."""
    import torch

    device = next(model.parameters()).device
    ids_t = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    mask_t = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
    direction_t = torch.as_tensor(direction, dtype=torch.float32, device=device)

    def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
        out = output if hasattr(output, "shape") else output[0]
        return out + direction_t.to(dtype=out.dtype, device=out.device)

    handle = None
    for name, module in model.named_modules():
        if name == f"transformer.h.{layer}":
            handle = module.register_forward_hook(_hook)
            break
    if handle is None:
        raise ValueError(f"Layer {layer} (transformer.h.{layer}) was not found in the model")

    try:
        with torch.no_grad():
            logits = model(input_ids=ids_t, attention_mask=mask_t, output_hidden_states=False).logits
        position = target.position if target.position >= 0 else int(logits.shape[1]) + target.position
        return float(logits[target.batch_index, position, target.token_id].item())
    finally:
        handle.remove()


def _coerce_labels(example_labels: Sequence[str | None], n_examples: int) -> tuple[str | None, ...] | None:
    values = tuple(None if label is None else str(label) for label in example_labels)
    if len(values) != n_examples:
        raise ValueError(f"expected {n_examples} example labels, got {len(values)}")
    return values


# ---------------------------------------------------------------------------
# Primary evaluation class (registry-constructable)
# ---------------------------------------------------------------------------


class SAEFeatureEvaluation:
    """Config-driven SAE feature evaluation and feature-atlas builder.

    Wraps the concrete fitting/evaluation with a :class:`SAEConfig` so the
    analysis can be constructed via ``ObjectSpec`` / ``build_from_config``::

        from latent_anything.config import build_from_dict

        evaluator = build_from_dict(
            {"kind": "analysis", "name": "sae_evaluation", "params": {"n_components": 32}}
        )
        result = evaluator.fit(data, source_representation_identity="gpt2_layer_6")

    Parameters
    ----------
    config : SAEConfig, optional
        Evaluation configuration. Defaults to ``SAEConfig()``.
    """

    def __init__(self, config: SAEConfig | None = None, **kwargs: Any) -> None:
        if kwargs:
            self._config = SAEConfig(**kwargs)
        else:
            self._config = config if config is not None else SAEConfig()
        self._sae: SAE | None = None

    @property
    def config(self) -> SAEConfig:
        """Return the validated evaluation configuration."""
        return self._config

    @property
    def sae(self) -> SAE | None:
        """Return the most recently fitted SAE, if any."""
        return self._sae

    def fit(
        self,
        data: np.ndarray,
        *,
        val_data: np.ndarray | None = None,
        source_representation_identity: str = "",
        provenance: dict[str, Any] | None = None,
    ) -> SAEEvaluationResult:
        """Split train/validation, fit on train only, and evaluate on validation.

        Parameters
        ----------
        data :
            Activation vectors of shape ``(n_samples, n_features)``.
        val_data :
            Optional explicit validation split. When omitted, ``data`` is
            split deterministically using ``SAEConfig.random_state`` and
            ``SAEConfig.val_fraction``.
        source_representation_identity :
            Declared identity of the representation the activations come from.
        provenance :
            Additional caller provenance merged into the result.

        Returns
        -------
        SAEEvaluationResult
            Reconstruction, sparsity, dead-feature, and geometry statistics.
        """
        cfg = self._config
        values = _validate_batch(data, name="data")
        if val_data is not None:
            train = values
            val = _validate_batch(val_data, name="val_data")
        else:
            rng = np.random.default_rng(cfg.random_state)
            permuted = values[rng.permutation(values.shape[0])]
            split = max(1, int(permuted.shape[0] * (1.0 - cfg.val_fraction)))
            train, val = permuted[:split], permuted[split:]
        if val.shape[0] < cfg.min_val_samples:
            raise ValueError(f"validation split has {val.shape[0]} samples; need at least {cfg.min_val_samples}")
        sae = SAE(
            n_components=cfg.n_components,
            l1_coef=cfg.l1_coef,
            learning_rate=cfg.learning_rate,
            n_epochs=cfg.n_epochs,
            random_state=cfg.random_state,
        )
        sae.fit(train)
        self._sae = sae
        return _evaluate_fitted(sae, train, val, cfg, source_representation_identity, provenance)

    def stability(
        self,
        data: np.ndarray,
        *,
        seeds: Sequence[int] = (0, 1, 2),
        source_representation_identity: str = "",
    ) -> SAEStabilityResult:
        """Fit across seeds and measure decoder-direction feature stability."""
        values = _validate_batch(data, name="data")
        seed_values = tuple(int(seed) for seed in seeds)
        if not seed_values:
            raise ValueError("at least one seed is required")
        if len(seed_values) < 2:
            raise ValueError("stability analysis needs at least two seeds")
        reports: list[SAEEvaluationResult] = []
        for seed in seed_values:
            cfg = self._config.model_copy(update={"random_state": seed})
            evaluator = SAEFeatureEvaluation(cfg)
            reports.append(
                evaluator.fit(
                    values,
                    source_representation_identity=source_representation_identity,
                    provenance={"stability_seed": seed},
                )
            )
        reference = reports[0]
        matched_cosines: list[float] = []
        n_matched_total = 0
        n_pairs = 0
        for other in reports[1:]:
            matched = _match_by_decoder_cosine(
                reference.decoder_weights, other.decoder_weights, self._config.matching_cosine_threshold
            )
            n_pairs += 1
            n_matched_total += len(matched)
            matched_cosines.extend(cosine for _, cosine in matched)
        if n_pairs == 0:
            raise ValueError("stability analysis produced no seed comparisons")
        alignment_quality = n_matched_total / (n_pairs * self._config.n_components)
        cosine_values = tuple(matched_cosines)
        return SAEStabilityResult(
            seeds=seed_values,
            reconstruction_mses=tuple(report.reconstruction_mse for report in reports),
            n_components=self._config.n_components,
            n_features_matched=n_matched_total,
            mean_matched_cosine=float(np.mean(cosine_values)) if cosine_values else 0.0,
            min_matched_cosine=float(np.min(cosine_values)) if cosine_values else 0.0,
            alignment_quality=float(alignment_quality),
            matched_cosines=cosine_values,
        )

    def rank(
        self,
        feature_index: int,
        result: SAEEvaluationResult,
        *,
        k: int = 5,
        example_labels: Sequence[str | None] | None = None,
    ) -> FeatureRanking:
        """Rank top-activating and bottom-activating examples for one feature."""
        return rank_feature_examples(result, feature_index, k=k, example_labels=example_labels)

    def cross_check(
        self,
        feature_index: int,
        result: SAEEvaluationResult,
        **kwargs: Any,
    ) -> FeatureCrossCheck:
        """Run the probe/concept/steering cross-check for one feature."""
        return cross_check_feature(result, feature_index, **kwargs)

    def atlas(
        self,
        result: SAEEvaluationResult,
        *,
        feature_indices: Sequence[int] | None = None,
        k_examples: int = 5,
        k_decoder_dims: int = 10,
        example_labels: Sequence[str | None] | None = None,
    ) -> FeatureAtlas:
        """Build the portable feature-atlas artifact for a fitted result."""
        return build_feature_atlas(
            result,
            feature_indices=feature_indices,
            k_examples=k_examples,
            k_decoder_dims=k_decoder_dims,
            example_labels=example_labels,
        )

    def save_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Serialize the fitted SAE state to a portable checkpoint."""
        if self._sae is None:
            raise RuntimeError("no fitted SAE to checkpoint; call fit() first")
        self._sae.save_checkpoint(path)

    @staticmethod
    def load_checkpoint(path: str | os.PathLike[str]) -> SAEFeatureEvaluation:
        """Load a fitted SAE checkpoint into a fresh evaluation wrapper."""
        sae = SAE.load_checkpoint(path)
        evaluator = SAEFeatureEvaluation()
        evaluator._sae = sae  # type: ignore[reportPrivateUsage]
        return evaluator


# ---------------------------------------------------------------------------
# Functional wrappers
# ---------------------------------------------------------------------------


def evaluate_sae_features(
    data: np.ndarray,
    *,
    config: SAEConfig | None = None,
    val_data: np.ndarray | None = None,
    source_representation_identity: str = "",
    provenance: dict[str, Any] | None = None,
) -> SAEEvaluationResult:
    """Fit and evaluate an SAE on one dataset in a single call."""
    return SAEFeatureEvaluation(config).fit(
        data,
        val_data=val_data,
        source_representation_identity=source_representation_identity,
        provenance=provenance,
    )


def cross_seed_sae_stability(
    data: np.ndarray,
    *,
    config: SAEConfig | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    source_representation_identity: str = "",
) -> SAEStabilityResult:
    """Cross-seed feature stability for one dataset in a single call."""
    return SAEFeatureEvaluation(config).stability(
        data, seeds=seeds, source_representation_identity=source_representation_identity
    )


def rank_feature_examples(
    evaluation: SAEEvaluationResult,
    feature_index: int,
    *,
    k: int = 5,
    example_labels: Sequence[str | None] | None = None,
) -> FeatureRanking:
    """Rank the top-activating and bottom-activating examples for a feature."""
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


def cross_check_feature(
    evaluation: SAEEvaluationResult,
    feature_index: int,
    *,
    labels: np.ndarray | None = None,
    model: Any = None,
    input_ids: np.ndarray | None = None,
    attention_mask: np.ndarray | None = None,
    target: TransformerLogitTarget | None = None,
    layer: int | None = None,
    intervention_strength: float = 1.0,
    probe_random_state: int = 0,
) -> FeatureCrossCheck:
    """Cross-check one feature against probes, concepts, and causal steering.

    Parameters
    ----------
    evaluation :
        A fitted SAE evaluation whose decoder directions live in the residual
        space of *model*.
    feature_index :
        Which feature to cross-check.
    labels :
        Optional per-example labels aligned with the evaluation's validation
        activations, used for the linear-probe check with a shuffled-label
        control.
    model, input_ids, attention_mask, target, layer :
        When all are provided, compute the observational concept sensitivity
        (mean gradient · feature direction), a causal steering effect, and the
        sign-agreement between steering and gradient.
    intervention_strength :
        Steering magnitude for the intervention check.
    probe_random_state :
        Seed for the probe and its shuffled-label control.
    """
    if not 0 <= feature_index < evaluation.config.n_components:
        raise ValueError(f"feature_index {feature_index} is outside [0, {evaluation.config.n_components})")
    direction = _feature_direction(evaluation, feature_index)

    # ── Probe check with shuffled-label control ─────────────────────────
    probe_accuracy: float | None = None
    shuffled_accuracy: float | None = None
    if labels is not None:
        label_values = np.asarray(labels)
        if label_values.ndim != 1:
            raise ValueError("labels must be a 1D array")
        if label_values.shape[0] != evaluation.val_activations.shape[0]:
            raise ValueError("labels must align with the evaluation's validation examples")
        if np.unique(label_values).size >= 2:
            feature_column = evaluation.val_activations[:, [feature_index]]
            probe = LinearProbe(LinearProbeConfig(random_state=probe_random_state))
            probe_accuracy = float(probe.fit(feature_column, label_values).accuracy)
            rng = np.random.default_rng(probe_random_state)
            shuffled = rng.permutation(label_values)
            shuffled_accuracy = float(probe.fit(feature_column, shuffled).accuracy)

    # ── Concept sensitivity + causal steering on the transformer seam ───
    concept_sensitivity: float | None = None
    intervention_effect: float | None = None
    intervention_agreement: float | None = None
    n_examples_checked = 0
    has_causal_evidence = (
        model is not None
        and input_ids is not None
        and attention_mask is not None
        and target is not None
        and layer is not None
    )
    if has_causal_evidence:
        assert target is not None
        assert layer is not None
        ids = np.asarray(input_ids)
        mask = np.asarray(attention_mask)
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError("input_ids must be 2D and attention_mask must have the same shape")
        gradients: list[np.ndarray] = []
        baseline_values: list[float] = []
        intervened_values: list[float] = []
        for i in range(ids.shape[0]):
            gradients.append(_gradient_at_layer(model, ids[i : i + 1], mask[i : i + 1], layer, target))
            baseline_values.append(_target_value(model, ids[i : i + 1], mask[i : i + 1], target))
            intervened_values.append(
                _target_with_intervention(
                    model,
                    ids[i : i + 1],
                    mask[i : i + 1],
                    target,
                    layer,
                    intervention_strength * direction,
                )
            )
        gradient_matrix = np.stack(gradients)
        dots = np.asarray(gradient_matrix @ direction, dtype=np.float64)
        concept_sensitivity = float(dots.mean())
        effects = np.asarray(intervened_values, dtype=np.float64) - np.asarray(baseline_values, dtype=np.float64)
        intervention_effect = float(effects.mean())
        nonzero = dots != 0.0
        if np.any(nonzero):
            agreements = np.sign(effects[nonzero]) == np.sign(dots[nonzero])
            intervention_agreement = float(np.mean(agreements))
        else:
            intervention_agreement = 1.0
        n_examples_checked = ids.shape[0]

    return FeatureCrossCheck(
        feature_index=feature_index,
        probe_accuracy=probe_accuracy,
        shuffled_label_accuracy=shuffled_accuracy,
        concept_sensitivity=concept_sensitivity,
        intervention_effect=intervention_effect,
        intervention_agreement=intervention_agreement,
        n_examples_checked=n_examples_checked,
        has_causal_evidence=has_causal_evidence,
    )


def build_feature_atlas(
    evaluation: SAEEvaluationResult,
    *,
    feature_indices: Sequence[int] | None = None,
    k_examples: int = 5,
    k_decoder_dims: int = 10,
    example_labels: Sequence[str | None] | None = None,
) -> FeatureAtlas:
    """Build the portable feature-atlas artifact from a fitted evaluation.

    The atlas is pure data (JSON-serializable via :meth:`FeatureAtlas.to_dict`)
    and independent of any visualization frontend.
    """
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
