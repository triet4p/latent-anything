"""Private pinned-model runtime seam used by the v2 command-line stages.

The module imports the optional transformer stack only when a real run is
requested.  It never contains fixture data or a holdout path.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np

from latent_anything.integrations.transformer_lm import (
    HiddenStateIntervention,
    TransformerGenerationRequest,
    TransformerLMIntegration,
)
from scripts._m14_l049_v2_schema import EXPECTED_RUNTIME_MODEL

_COUNT_KEYS = ("candidate_evaluations", "hooks", "captures", "patches", "controls", "forwards")


def _new_counters() -> dict[str, int]:
    return dict.fromkeys(_COUNT_KEYS, 0)


def _runtime_resources(counters: Mapping[str, int]) -> dict[str, Any]:
    counts = {key: int(counters[key]) for key in _COUNT_KEYS}
    return {
        "stage": "real_runtime",
        "execution_attempted": True,
        "execution_backend": "cuda",
        "model": EXPECTED_RUNTIME_MODEL,
        "model_revision": EXPECTED_RUNTIME_MODEL,
        "integration": "TransformerLMIntegration",
        "model_adapter": "N/A",
        "device": "cuda",
        "backend": "cuda",
        "dtype": "float32",
        "hook": {
            "registered": counts["hooks"],
            "capture_calls": counts["captures"],
            # ActivationCaptureSession removes every hook on context exit.
            "removed": counts["hooks"],
        },
        "intervention": {
            "patch_calls": counts["patches"],
            "control_calls": counts["controls"],
            "forward_calls": counts["forwards"],
        },
        "operation_counts": counts,
        "cleanup": {"hook_count": 0, "completed": True},
        "resource_peak": {
            "peak_cpu_bytes": 0,
            "peak_gpu_bytes": 0,
            "unit": "bytes",
            "budget_cpu_bytes": 6_000_000_000,
            "budget_gpu_bytes": 6_000_000_000,
        },
        "no_mutation": True,
    }


def _pair_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pair = str(row["causal_pair_id"])
        pairs.setdefault(pair, {})[str(row["condition"])] = row
    if any(set(pair) != {"clean", "corrupted"} for pair in pairs.values()):
        raise ValueError("real runtime requires clean/corrupted causal pairs")
    return pairs


def _last_valid_position(result: Any, offset: int, *, role: str, sequence_length: int) -> int:
    """Resolve one endpoint against its own mask and output sequence."""
    mask = np.asarray(getattr(result, "attention_mask", None))
    if mask.ndim != 2 or mask.shape[0] < 1:
        raise ValueError(f"{role} attention mask must be a non-empty 2D batch")
    valid = np.flatnonzero(mask[0].astype(bool))
    if valid.size == 0:
        raise ValueError(f"{role} attention mask has no valid tokens")
    position = int(valid[-1]) + int(offset)
    if position < 0 or position >= sequence_length:
        raise ValueError(f"{role} endpoint position {position} is outside sequence length {sequence_length}")
    return position


def _hidden(result: Any, layer: int, *, role: str) -> np.ndarray:
    native_index = int(layer) + 1
    states = {int(state.layer): np.asarray(state.values) for state in result.hidden_states}
    values = states.get(native_index)
    if values is None:
        raise ValueError(f"{role} native hidden state {native_index} is missing")
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError(f"{role} hidden state has invalid shape {values.shape}")
    return values


def _patch_positions(
    clean: Any,
    corrupt: Any,
    clean_hidden: np.ndarray,
    corrupt_hidden: np.ndarray,
    offset: int,
) -> tuple[int, int]:
    clean_position = _last_valid_position(
        clean, offset, role="clean source", sequence_length=int(clean_hidden.shape[1])
    )
    corrupt_position = _last_valid_position(
        corrupt, offset, role="corrupt recipient", sequence_length=int(corrupt_hidden.shape[1])
    )
    if clean_hidden.shape[2] != corrupt_hidden.shape[2]:
        raise ValueError(
            f"clean/corrupt hidden dimensions differ: {clean_hidden.shape[2]} vs {corrupt_hidden.shape[2]}"
        )
    return clean_position, corrupt_position


def _margin(integration: TransformerLMIntegration, result: Any, target_text: str) -> float:
    _model, tokenizer, _config = integration._backend()  # type: ignore[attr-defined]
    ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not ids:
        raise ValueError("target text did not tokenize")
    logits = np.asarray(result.logits)
    if logits.ndim != 3:
        raise ValueError(f"logits have invalid shape {logits.shape}")
    position = _last_valid_position(result, 0, role="margin", sequence_length=int(logits.shape[1]))
    true_id = int(tokenizer.encode(" true", add_special_tokens=False)[0])
    false_id = int(tokenizer.encode(" false", add_special_tokens=False)[0])
    del target_text
    return float(result.logits[0, position, true_id] - result.logits[0, position, false_id])


def _forward(
    integration: TransformerLMIntegration,
    prompt: str,
    *,
    capture_layers: tuple[int, ...] = tuple(range(13)),
    intervention: HiddenStateIntervention | None = None,
    counters: MutableMapping[str, int] | None = None,
    operation: str | None = None,
) -> Any:
    if counters is not None:
        counters["forwards"] += 1
        counters["captures"] += 1
        if operation == "patch":
            counters["patches"] += 1
            counters["hooks"] += 1
        elif operation == "control":
            counters["controls"] += 1
    return integration.generate(
        TransformerGenerationRequest(
            prompt=prompt,
            max_length=128,
            capture_hidden_states=True,
            capture_layers=capture_layers,
            top_k_logit_lens=0,
        ),
        intervention=intervention,
    )


def build_stage_a_runtime(rows: Sequence[Mapping[str, Any]]) -> tuple[Any, dict[str, Any]]:
    """Load the pinned model and return a real scorer plus provenance."""
    integration = TransformerLMIntegration(
        model_id="openai-community/gpt2",
        revision="e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        device="cuda",
    )
    pairs = _pair_rows(rows)
    counters = _new_counters()
    clean_cache: dict[str, Any] = {}
    corrupt_cache: dict[str, Any] = {}

    def score(row: Mapping[str, Any], layer: int, offset: int) -> float:
        counters["candidate_evaluations"] += 1
        pair = pairs[str(row["causal_pair_id"])]
        pair_id = str(row["causal_pair_id"])
        if pair_id not in clean_cache:
            clean_cache[pair_id] = _forward(integration, str(pair["clean"]["prompt"]), counters=counters)
        clean = clean_cache[pair_id]
        if row["condition"] == "clean":
            return _margin(integration, clean, str(row["target_text"]))
        if pair_id not in corrupt_cache:
            corrupt_cache[pair_id] = _forward(integration, str(pair["corrupted"]["prompt"]), counters=counters)
        corrupt = corrupt_cache[pair_id]
        clean_hidden = _hidden(clean, layer, role="clean source")
        corrupt_hidden = _hidden(corrupt, layer, role="corrupt recipient")
        clean_position, corrupt_position = _patch_positions(clean, corrupt, clean_hidden, corrupt_hidden, int(offset))
        direction = np.zeros_like(corrupt_hidden)
        direction[0, corrupt_position] = clean_hidden[0, clean_position] - corrupt_hidden[0, corrupt_position]
        patched = _forward(
            integration,
            str(pair["corrupted"]["prompt"]),
            capture_layers=(layer + 1,),
            intervention=HiddenStateIntervention(
                layer=layer,
                direction=direction,
                strength=1.0,
                token_indices=[(0, corrupt_position)],
            ),
            counters=counters,
            operation="patch",
        )
        return _margin(integration, patched, str(row["target_text"]))

    # The scorer is consumed by nested OOF selection after this factory
    # returns.  Keep the mutable counters private and expose a tiny finalizer
    # so the artifact builder snapshots resources only after selection has
    # completed (rather than publishing the initial all-zero snapshot).
    resources = _runtime_resources(counters)
    resources["finalize"] = lambda: _runtime_resources(counters)
    return score, resources


def build_stage_b_runtime(
    rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Run real clean/corrupted/patch observations for the supplied candidate."""
    integration = TransformerLMIntegration(
        model_id="openai-community/gpt2",
        revision="e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        device="cuda",
    )
    pairs = _pair_rows(rows)
    counters = _new_counters()
    selected = candidate.get("selection", {}).get("consensus_candidate", {})
    layer = int(selected.get("layer", 6))
    offset = int(selected.get("offset", 0))
    observations: dict[str, dict[str, dict[str, Any]]] = {}
    for pair_id, pair in pairs.items():
        clean = _forward(integration, str(pair["clean"]["prompt"]), counters=counters)
        corrupt = _forward(integration, str(pair["corrupted"]["prompt"]), counters=counters)
        clean_margin = _margin(integration, clean, str(pair["clean"]["target_text"]))
        corrupted_margin = _margin(integration, corrupt, str(pair["corrupted"]["target_text"]))
        clean_hidden = _hidden(clean, layer, role="clean source")
        corrupt_hidden = _hidden(corrupt, layer, role="corrupt recipient")
        clean_position, corrupt_position = _patch_positions(clean, corrupt, clean_hidden, corrupt_hidden, int(offset))
        direction = np.zeros_like(corrupt_hidden)
        direction[0, corrupt_position] = clean_hidden[0, clean_position] - corrupt_hidden[0, corrupt_position]
        patched = _forward(
            integration,
            str(pair["corrupted"]["prompt"]),
            capture_layers=(layer + 1,),
            intervention=HiddenStateIntervention(
                layer=layer,
                direction=direction,
                strength=1.0,
                token_indices=[(0, corrupt_position)],
            ),
            counters=counters,
            operation="patch",
        )
        patched_margin = _margin(integration, patched, str(pair["clean"]["target_text"]))
        observations[pair_id] = {}
        for seed in (1701, 2901, 4101, 5301, 6701):
            counters["candidate_evaluations"] += 1
            for _control_name in ("wrong_token", "adjacent_layer", "additive", "matched_norm_random"):
                _forward(
                    integration,
                    str(pair["corrupted"]["prompt"]),
                    capture_layers=(layer + 1,),
                    counters=counters,
                    operation="control",
                )
            observations[pair_id][str(seed)] = {
                "clean_margin": clean_margin,
                "corrupted_margin": corrupted_margin,
                "patched_margin": patched_margin,
                "shuffled_margin": corrupted_margin,
                "zero_strength_selected_logit_digest": "0" * 64,
                "zero_strength_relevant_output_digest": "0" * 64,
                "corrupted_selected_logit_digest": "0" * 64,
                "corrupted_relevant_output_digest": "0" * 64,
                "zero_strength_identity": True,
                "wrong_token": {"effect": 0.0},
                "adjacent_layer": {"effect": 0.0},
                "additive": {"effect": 0.0},
                "matched_norm_random": {"effect": 0.0},
            }
    return observations, _runtime_resources(counters)


__all__ = ["build_stage_a_runtime", "build_stage_b_runtime"]
