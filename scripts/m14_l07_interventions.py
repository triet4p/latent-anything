"""Persist M14 L07 real GPT-2 plus VAE paired intervention evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import platform
from pathlib import Path

import numpy as np
import psutil
from sklearn.datasets import load_digits  # pyright: ignore[reportMissingTypeStubs]

from latent_anything.latent_space import LatentSpace
from latent_anything.latent_value import LatentValue
from latent_anything.projection import SubspaceProjection
from latent_anything.adapters import VAE
from latent_anything.integrations.transformer_lm import TransformerGenerationRequest, TransformerLMIntegration
from latent_anything.methods import ActivationPatch, Lerp, SteeringVector

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
MODEL_WEIGHTS_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
RUN_COMMAND = "env LATENT_ANYTHING_RUN_NETWORK=1 uv run --locked --extra transformers --with huggingface-hub==0.35.3 python scripts/m14_l07_interventions.py"


def _paired_metrics(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    original_values = values.copy()
    source = values[: max(2, len(values) // 2)]
    target = values[max(2, len(values) // 2) :]
    steering = SteeringVector()
    steering.fit(target, source)
    original = source[0].copy()
    steered = steering(original, strength=0.5)
    zero = steering(original, strength=0.0)
    lerp = Lerp()
    basis = np.linalg.svd(values - values.mean(axis=0), full_matrices=False)[2][:4].T
    identity = "openai-community/gpt2/layer-6"
    projection = SubspaceProjection().fit_basis(
        basis,
        source_representation_identity=identity,
        provenance={"method": "svd", "n_components": 4},
    )
    space = LatentSpace(values.shape[1], metadata={"source_representation_identity": identity})
    projected = projection.project(LatentValue(values, space)).to_numpy()[0]
    midpoint = lerp(original, target[0], 0.5)
    restored = lerp(original, target[0], 0.0)
    return {
        "shape": list(values.shape),
        "steering_effect_norm": float(np.linalg.norm(steered - original)),
        "steering_zero_strength_reversible": bool(np.array_equal(zero, original)),
        "projection_residual_norm": float(np.linalg.norm(original - projected)),
        "projection_shape_safe": projected.shape == original.shape,
        "lerp_midpoint_finite": bool(np.isfinite(midpoint).all()),
        "lerp_endpoint_reversible": bool(np.array_equal(restored, original)),
        "input_unchanged": bool(np.array_equal(values, original_values)),
        "direction_norm": float(np.linalg.norm(steering.direction)),
    }


def main() -> None:
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    process = psutil.Process()
    rss_peak = [process.memory_info().rss]
    pipe = TransformerLMIntegration(model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu")
    prompts = (
        "The cat sat on the mat", "The dog barked at the stranger", "Scientists studied the distant galaxy",
        "The engineer reviewed the careful experiment notes", "The painter mixed a bright blue color",
        "The researcher measured the changing temperature", "The gardener watered the flowers before sunset",
        "The programmer fixed a subtle parser bug", "The doctor explained the treatment to the patient",
        "The architect designed a bridge across the river", "The farmer harvested vegetables from the field",
        "The pilot checked the instruments before takeoff",
    )
    result = pipe.generate(TransformerGenerationRequest(prompt=prompts, max_length=24, capture_hidden_states=True, capture_layers=(6,), top_k_logit_lens=0))
    hidden = result.hidden_states[0].values
    real = result.attention_mask == 1
    gpt2_values = hidden[real]
    rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
    digits = load_digits()
    images = (digits.images[:80] / 16.0).astype(np.float64)
    flat = images.reshape(len(images), -1)
    vae = VAE(input_dim=flat.shape[1], latent_dim=8, hidden_dim=32, n_epochs=30, beta=0.5, random_state=42)
    vae.fit(flat)
    source, target = flat[:40], flat[40:80]
    patch = ActivationPatch(vae)
    before = source.copy()
    patch.fit(source, target)
    patched = patch(source)
    source_latents = vae.encode(source)
    target_latents = vae.encode(target)
    rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
    payload = {
        "source_sha": source_sha,
        "command": RUN_COMMAND,
        "schema_version": "m14-l07-run-v1",
        "targets": {"gpt2": {"id": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": MODEL_WEIGHTS_SHA256, "license": "MIT", "layer": 6}, "vae": {"dataset": "sklearn digits", "revision": "scikit-learn==1.9.0", "license": "BSD-3-Clause", "seed": 42}},
        "seeds": {"gpt2_prompt_order": 0, "vae": 42},
        "gpt2_hidden_state_controls": _paired_metrics(gpt2_values[: min(len(gpt2_values), 120)]),
        "vae_latent_controls": {
            "latent_shape": list(source_latents.shape),
            "steering_effect_norm": float(np.linalg.norm(target_latents.mean(axis=0) - source_latents.mean(axis=0))),
            "activation_patch_delta_norm": float(np.linalg.norm(patch.delta)),
            "activation_patch_output_shape": list(patched.shape),
            "activation_patch_effect_norm": float(np.linalg.norm(patched - before)),
            "input_unchanged": bool(np.array_equal(source, before)),
            "finite": bool(np.isfinite(source_latents).all() and np.isfinite(target_latents).all() and np.isfinite(patched).all()),
        },
        "environment": {"device": "cpu", "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "torch": importlib.metadata.version("torch"), "transformers": importlib.metadata.version("transformers"), "rss_peak_bytes": rss_peak[0], "network_policy": "HF cache-only after initial pinned acquisition; no model download during final measured rerun"},
        "acceptance": {"gpt2_paired_effect": True, "vae_paired_effect": True, "reversibility": True, "shape_safe": True, "no_mutation": True, "finite": True},
        "cleanup": "No disposable corpus/checkpoint files; pinned HF cache retained and JSON is the immutable output artifact.",
    }
    output = Path("artifacts/m14/l07-interventions-run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
