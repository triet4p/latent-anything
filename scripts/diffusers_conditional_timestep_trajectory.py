#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "latent-anything[diffusers-full]",
#     "matplotlib>=3.9,<4.0",
# ]
# ///

"""Sprint 37 artifact: timestep-trajectory analysis of a conditional diffusion run.

Produces
--------
artifacts/diffusers_conditional_timestep_trajectory.png
    A multi-panel figure showing:
    - Norm of scheduler latent states over timesteps
    - Cosine similarity between consecutive scheduler states
    - Decoded intermediate VAE latents (every N steps)
    - Denoiser activation norm at captured location

artifacts/diffusers_conditional_timestep_trajectory_summary.txt
    A textual summary of the run with metadata and metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Ensure the package is available when run as standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latent_anything.integrations.diffusers_conditional import (  # noqa: E402
    CONDITIONAL_MODEL_ID,
    CONDITIONAL_MODEL_REVISION,
    DiffusersConditionalPipeline,
    GenerationRequest,
)

PROMPT = "a photograph of an astronaut riding a horse"
NUM_STEPS = 30
SEED = 42
OUTPUT_DIR = Path("artifacts")
OUTPUT_PNG = OUTPUT_DIR / "diffusers_conditional_timestep_trajectory.png"
OUTPUT_SUMMARY = OUTPUT_DIR / "diffusers_conditional_timestep_trajectory_summary.txt"


def compute_consecutive_similarity(latents: list[np.ndarray]) -> list[float]:
    """Cosine similarity between consecutive flattened latent states."""
    sims: list[float] = []
    for i in range(1, len(latents)):
        a = latents[i - 1].ravel()
        b = latents[i].ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(np.dot(a, b) / denom) if denom > 1e-15 else 0.0
        sims.append(cos)
    return sims


def main() -> None:
    print(f"Running conditional diffusion with prompt: {PROMPT!r}")
    print(f"Using {NUM_STEPS} inference steps, seed={SEED}")

    pipe = DiffusersConditionalPipeline(device="cpu")
    req = GenerationRequest(
        prompt=PROMPT,
        num_inference_steps=NUM_STEPS,
        seed=SEED,
        capture_scheduler_states=True,
        capture_denoiser_location="mid_block",
    )
    result = pipe.generate(req)

    # Extract scheduler trajectory.
    scheduler_states = list(result.scheduler_states)
    timesteps = [s.timestep for s in scheduler_states]
    latents = [s.latent for s in scheduler_states]

    # Compute trajectory metrics.
    norms = [float(np.linalg.norm(lat)) for lat in latents]
    similarities = compute_consecutive_similarity(latents)

    # Denoiser activation norms (per step).
    act_steps: list[int] = []
    act_norms: list[float] = []
    for cap in result.denoiser_captures:
        act_steps.append(cap.step)
        act_norms.append(float(np.linalg.norm(cap.values)))

    # --- Build figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Timestep Trajectory — SD 1.5 @ {CONDITIONAL_MODEL_REVISION[:12]}…", fontsize=14, fontweight="bold")  # type: ignore[reportUnknownMemberType]

    # Panel 1: Norm trajectory
    ax = axes[0, 0]
    ax.plot(range(len(norms)), norms, marker=".", linestyle="-", color="steelblue")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Latent norm")
    ax.set_title("Scheduler latent norm over steps")
    ax.grid(True, alpha=0.3)

    # Panel 2: Cosine similarity (step-to-step)
    ax = axes[0, 1]
    ax.plot(range(1, len(similarities) + 1), similarities, marker=".", linestyle="-", color="darkorange")
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Consecutive step similarity")
    ax.grid(True, alpha=0.3)

    # Panel 3: Denoiser activation norm
    ax = axes[1, 0]
    if act_norms:
        ax.plot(act_steps, act_norms, marker=".", linestyle="-", color="crimson")
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("Activation norm")
        ax.set_title("Denoiser activation norm (mid_block)")
    else:
        ax.text(0.5, 0.5, "No denoiser captures", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Denoiser activations (not captured)")
    ax.grid(True, alpha=0.3)

    # Panel 4: Decoded intermediate states (handle unsupported case gracefully).
    ax = axes[1, 1]
    ax.text(
        0.5,
        0.5,
        "VAE decode of intermediate scheduler states\n"
        "requires component VAE adapter.\n"
        "Not supported in this sprint.\n\n"
        "See: DiffusersAutoencoderKLAdapter",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.set_title("Decoded intermediates")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_PNG), dpi=150)  # type: ignore[reportUnknownMemberType]
    print(f"Saved {OUTPUT_PNG}")

    # --- Summary text ---
    lines = [
        "=" * 60,
        "Sprint 37 — Conditional Diffusion Timestep Trajectory",
        "=" * 60,
        f"Model:      {CONDITIONAL_MODEL_ID}",
        f"Revision:   {CONDITIONAL_MODEL_REVISION}",
        f"Prompt:     {PROMPT}",
        f"Steps:      {NUM_STEPS}",
        f"Seed:       {SEED}",
        "Scheduler:  DDIM",
        "",
        f"Scheduler states captured: {len(scheduler_states)}",
        f"Denoiser captures:         {len(result.denoiser_captures)}",
        f"Final latent shape:        {result.final_vae_latent.shape}",
        f"Output image shape:        {result.images.shape}",
        "",
        "--- Norm trajectory ---",
    ]
    for _i, (step, t, n) in enumerate(zip(range(len(norms)), timesteps, norms)):
        lines.append(f"  step {step:3d}  timestep {t:4d}  norm {n:.4f}")

    if similarities:
        lines.append("")
        lines.append("--- Consecutive cosine similarity ---")
        for i, sim in enumerate(similarities):
            lines.append(f"  step {i}→{i + 1}:  {sim:.6f}")

    if act_norms:
        lines.append("")
        lines.append("--- Denoiser activation norms ---")
        for step, n in zip(act_steps, act_norms):
            lines.append(f"  step {step}:  {n:.4f}")

    lines.append("")
    lines.append("--- Failed / Unsupported cases ---")
    lines.append("  - VAE decode of intermediate scheduler latents: NOT SUPPORTED")
    lines.append("    (requires component DiffusersAutoencoderKLAdapter)")
    lines.append("  - Scheduler state at step 0 (initial noise): NOT CAPTURED")
    lines.append("    (callback_on_step_end fires after first scheduler.step())")
    lines.append("")

    summary = "\n".join(lines)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(OUTPUT_SUMMARY), "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {OUTPUT_SUMMARY}")
    print(summary)


if __name__ == "__main__":
    main()
