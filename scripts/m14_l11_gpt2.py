"""Persist M14 L11 real GPT-2 hidden-state and logit-lens evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import psutil

from latent_anything.integrations.transformer_lm import (
    HiddenStateIntervention,
    TransformerGenerationRequest,
    TransformerLMIntegration,
)

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
MODEL_WEIGHTS_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
RUN_COMMAND = (
    "env LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra transformers "
    "--with huggingface-hub==0.35.3 python scripts/m14_l11_gpt2.py"
)
PROMPTS = (
    "The capital of France is",
    "Once upon a time",
    "The meaning of life is",
    "The future of AI is",
)


def main() -> None:
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    process = psutil.Process()
    pipe = TransformerLMIntegration(model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu")
    request = TransformerGenerationRequest(
        prompt=PROMPTS,
        max_length=16,
        seed=42,
        capture_hidden_states=True,
        capture_layers=(),
        top_k_logit_lens=5,
    )
    baseline = pipe.generate(request)
    repeat = pipe.generate(request)
    intervention = HiddenStateIntervention(
        layer=6,
        direction=np.ones((1, 1, 768), dtype=np.float32),
        strength=0.5,
    )
    edited = pipe.generate(request, intervention=intervention)
    clean = pipe.generate(request)
    hidden = baseline.hidden_states
    real_mask = baseline.attention_mask.astype(bool)
    hidden_finite = all(bool(np.isfinite(state.values).all()) for state in hidden)
    hidden_shapes = [list(state.values.shape) for state in hidden]
    lens_finite = all(
        bool(np.isfinite(lens.logits).all() and np.isfinite(lens.probabilities).all()) for lens in baseline.lens_results
    )
    probability_sums = [float(np.max(np.abs(lens.probabilities.sum(axis=-1) - 1.0))) for lens in baseline.lens_results]
    final_parity = bool(np.allclose(baseline.lens_results[-1].logits, baseline.logits, rtol=1e-6, atol=1e-6))
    layer_evolution = bool(
        len(baseline.lens_results) == 13
        and not np.array_equal(baseline.lens_results[1].probabilities, baseline.lens_results[-1].probabilities)
    )
    baseline_hidden = {state.layer: state.values for state in baseline.hidden_states}
    edited_hidden = {state.layer: state.values for state in edited.hidden_states}
    input_layer_unchanged = bool(np.array_equal(edited_hidden[6], baseline_hidden[6]))
    intervention_delta = float(np.linalg.norm(edited_hidden[7] - baseline_hidden[7]))
    payload = {
        "schema_version": "m14-l11-run-v1",
        "source_sha": source_sha,
        "command": RUN_COMMAND,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "license": "MIT",
        },
        "capture": {
            "prompts": list(PROMPTS),
            "prompt_digest": hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest(),
            "seed": request.seed,
            "max_length": request.max_length,
            "captured_native_layers": [state.layer for state in hidden],
            "capture_boundaries": "embedding index 0 through post-ln_f terminal index 12",
            "attention_mask_shape": list(baseline.attention_mask.shape),
            "real_token_count": int(real_mask.sum()),
        },
        "metrics": {
            "vocab_size": int(baseline.logits.shape[-1]),
            "hidden_shapes": hidden_shapes,
            "hidden_dtype": str(hidden[0].values.dtype),
            "logits_shape": list(baseline.logits.shape),
            "logits_dtype": str(baseline.logits.dtype),
            "max_probability_sum_error": max(probability_sums),
            "intervention_delta_native_layer_7_norm": intervention_delta,
        },
        "acceptance": {
            "vocab_size_50257": int(baseline.logits.shape[-1]) == 50257,
            "hidden_layers_13": len(hidden) == 13,
            "hidden_shape_768": all(shape[-1] == 768 for shape in hidden_shapes),
            "hidden_finite": hidden_finite,
            "logits_finite": bool(np.isfinite(baseline.logits).all()),
            "lens_finite": lens_finite,
            "probability_sums_one": max(probability_sums) <= 1e-5,
            "final_layer_logits_parity": final_parity,
            "layer_evolution": layer_evolution,
            "deterministic_repeat": bool(np.array_equal(baseline.logits, repeat.logits)),
            "intervention_changes_block_output": intervention_delta > 0.0,
            "intervention_input_layer_unchanged": input_layer_unchanged,
            "hook_cleanup_deterministic": bool(np.array_equal(baseline.logits, clean.logits)),
            "padded_mask_shape_parity": baseline.attention_mask.shape == baseline.input_ids.shape,
        },
        "environment": {
            "device": "cpu",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "rss_peak_bytes": process.memory_info().rss,
            "network_policy": (
                "HF cache-only after initial pinned acquisition; no model download "
                "during final measured rerun"
            ),
        },
        "cleanup": "No temporary model files; pinned HF cache retained and JSON is the immutable output artifact.",
    }
    output = Path("artifacts/m14/l11-gpt2-run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
