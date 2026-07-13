#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0,<3.0",
#     "torch>=2.5,<3.0",
#     "transformers>=4.45,<5.0",
#     "tokenizers>=0.20,<1.0",
#     "matplotlib>=3.9,<4.0",
# ]
# ///

"""End-to-end transformer LM demo with direct logit lens analysis.

Usage
-----
    uv run python scripts/end_to_end_transformer_lm_demo.py

This script:
1. Loads the pinned GPT-2 model (downloads on first run).
2. Runs a forward pass with hidden-state capture for a sample prompt.
3. Applies the direct logit lens at every layer.
4. Plots token rank trajectories across layers.
5. Demonstrates a bounded activation intervention.
6. Saves summary and plots to ``artifacts/``.

Requires the ``transformers`` optional extra.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the src directory is on the path for direct script execution.
sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from latent_anything.integrations.transformer_lm import (
    GPT2_HIDDEN_DIM,
    HiddenStateIntervention,
    TransformerGenerationRequest,
    TransformerLMIntegration,
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_demo() -> None:
    print("=" * 60)
    print("Transformer LM + Direct Logit Lens Demo")
    print("=" * 60)

    # ── 1. Load model ─────────────────────────────────────────────────
    print("\n[1/6] Loading pinned GPT-2 model...")
    pipe = TransformerLMIntegration(device="cpu")
    print(f"   Model: {pipe.model_id}@{pipe.revision}")
    print(f"   Layers: {pipe.num_layers}, Hidden dim: {pipe.hidden_dim}, Vocab: {pipe.vocab_size}")

    # ── 2. Forward pass with hidden-state capture ─────────────────────
    print("\n[2/6] Running forward pass with hidden-state capture...")
    prompt = "The capital of France is"
    req = TransformerGenerationRequest(
        prompt=prompt,
        max_length=10,
        seed=42,
        capture_hidden_states=True,
        top_k_logit_lens=5,
    )
    result = pipe.generate(req)
    print(f"   Input shape: {result.input_ids.shape}")
    print(f"   Hidden states captured: {len(result.hidden_states)}")
    print(f"   Final logits shape: {result.logits.shape}")

    # ── 3. Logit Lens Analysis ────────────────────────────────────────
    print("\n[3/6] Applying direct logit lens at all layers...")

    # Display top tokens at each layer for the last sequence position.
    final_pos = result.input_ids.shape[1] - 1
    print(f"\n   Top-5 tokens at each layer (position {final_pos}):")
    for lr in result.lens_results:
        if lr.top_tokens and lr.top_tokens[0]:
            pos_tokens = lr.top_tokens[0][final_pos]
            top_str = ", ".join(
                [f"'{pipe.decode_tokens(np.array([[tid]]))[0]}' ({prob:.3f})" for tid, prob in pos_tokens]
            )
            print(f"      Layer {lr.layer:2d}: {top_str}")

    # ── 4. Token rank trajectories ────────────────────────────────────
    print("\n[4/6] Computing token rank trajectories...")
    print(f"   Found {len(result.token_rank_trajectories)} token trajectories:")
    for traj in result.token_rank_trajectories[:5]:  # Show first 5
        rank_str = " → ".join([f"L{lyr}={r}" for lyr, r in zip(traj.layers, traj.ranks, strict=True)])
        print(f"      Token '{traj.token_str}' (id={traj.token_id}): {rank_str}")

    # ── 5. Bounded activation intervention ────────────────────────────
    print("\n[5/6] Demonstrating bounded activation intervention...")
    # Create a random direction and apply at layer 6.
    intervention = HiddenStateIntervention(
        layer=6,
        direction=np.ones((1, 1, GPT2_HIDDEN_DIM), dtype=np.float32) * 0.1,
        strength=1.0,
    )
    edited_result = pipe.generate(req, intervention=intervention)

    # Compare baseline vs edited logits at the target layer.
    baseline_lens = result.lens_results[7] if len(result.lens_results) > 7 else result.lens_results[-1]
    edited_lens = (
        edited_result.lens_results[7] if len(edited_result.lens_results) > 7 else edited_result.lens_results[-1]
    )

    lens_diff = float(np.linalg.norm(edited_lens.logits - baseline_lens.logits))
    print(f"   Logit lens diff at layer 7: {lens_diff:.4f}")

    # ── 6. Visualization ──────────────────────────────────────────────
    print("\n[6/6] Generating visualizations...")

    # Plot 1: Token rank trajectories.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for traj in result.token_rank_trajectories[:5]:
        ax.plot(traj.layers, traj.ranks, marker="o", label=f"'{traj.token_str}'")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank (1 = most probable)")
    ax.set_title("Token Rank Trajectories Across Layers")
    ax.invert_yaxis()  # Rank 1 at top
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for traj in result.token_rank_trajectories[:5]:
        ax.plot(traj.layers, traj.probabilities, marker="s", label=f"'{traj.token_str}'")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability")
    ax.set_title("Token Probability Trajectories Across Layers")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "transformer_lm_rank_trajectories.png"
    plt.savefig(plot_path, dpi=150)
    print(f"   Saved: {plot_path}")

    # Plot 2: Layer-by-layer entropy.
    fig, ax = plt.subplots(figsize=(8, 4))
    entropies = []
    for lr in result.lens_results:
        probs = lr.probabilities[0, final_pos]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    ax.plot(list(range(len(entropies))), entropies, marker="d")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Prediction Entropy at Each Layer")
    ax.grid(True, alpha=0.3)

    entropy_path = OUTPUT_DIR / "transformer_lm_layer_entropy.png"
    plt.savefig(entropy_path, dpi=150)
    print(f"   Saved: {entropy_path}")

    # ── Summary ───────────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "transformer_lm_demo_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Transformer LM + Direct Logit Lens Demo Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {pipe.provenance}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Prompt tokens: {result.input_ids.shape[1]}\n")
        f.write(f"Hidden states captured: {len(result.hidden_states)}\n")
        f.write(f"Lens results: {len(result.lens_results)}\n")
        f.write(f"Token trajectories: {len(result.token_rank_trajectories)}\n")
        f.write(f"Intervention lens diff: {lens_diff:.4f}\n\n")

        f.write("Token Rank Trajectories:\n")
        for traj in result.token_rank_trajectories[:10]:
            first_rank = traj.ranks[0] if traj.ranks else "-"
            final_rank = traj.ranks[-1] if traj.ranks else "-"
            f.write(f"  '{traj.token_str}' (id={traj.token_id}): rank {first_rank} → {final_rank}\n")

    print(f"\n   Saved: {summary_path}")
    print("\n" + "=" * 60)
    print("Demo complete. See artifacts/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
