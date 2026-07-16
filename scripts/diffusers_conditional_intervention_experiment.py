#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "latent-anything[diffusers-full]",
#     "matplotlib>=3.9,<4.0",
# ]
# ///

"""Sprint 38 experiment: scheduler latent intervention controls, metrics, and sweep.

Compares four conditions
------------------------
1. **no-edit**       — Baseline (no intervention, original prompt)
2. **prompt-only**   — Change prompt (remove target concept), no intervention
3. **random-dir**    — Add random Gaussian direction to scheduler latents
4. **matched-norm**  — Add random direction scaled to match latent norm

Measures
--------
- Target latent change (cosine distance from baseline final latent)
- Content preservation (SSIM, MSE vs no-edit output image)
- Latent norm drift (relative norm change)
- Trajectory deviation (mean per-step cosine similarity)

Sweeps
------
- Intervention start timestep  (0, 5, 10, 15, 20, 25)
- Intervention strength        (0.1, 0.5, 1.0, 2.0, 5.0)

Artifacts
---------
artifacts/diffusers_conditional_intervention_metrics.txt
    Aggregate metric tables with uncertainty across seeds.

artifacts/diffusers_conditional_intervention_controls.png
    Paired-output comparison panels.

artifacts/diffusers_conditional_intervention_sweep.png
    Timestep and strength sweep heatmaps / line plots.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latent_anything.integrations.diffusers_conditional import (  # noqa: E402
    CONDITIONAL_MODEL_ID,
    CONDITIONAL_MODEL_REVISION,
    DiffusersConditionalPipeline,
    GenerationRequest,
    SchedulerIntervention,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROMPT = "a photograph of an astronaut riding a horse"
PROMPT_ABLATED = "a photograph of a horse"
NUM_STEPS = 30
SEEDS = (42, 123, 456)
DEFAULT_STRENGTH = 1.0
DEFAULT_STEP_RANGE = (15, 25)

OUTPUT_DIR = Path("artifacts")
METRICS_FILE = OUTPUT_DIR / "diffusers_conditional_intervention_metrics.txt"
CONTROLS_PNG = OUTPUT_DIR / "diffusers_conditional_intervention_controls.png"
SWEEP_PNG = OUTPUT_DIR / "diffusers_conditional_intervention_sweep.png"

# Sweep ranges
SWEEP_START_STEPS = [0, 5, 10, 15, 20, 25]
SWEEP_STRENGTHS = [0.1, 0.5, 1.0, 2.0, 5.0]

DTYPE = np.float32


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    """Aggregate metrics for one condition at one seed."""

    label: str
    seed: int
    final_cosine_dist: float  # 1 - cos(baseline_final, edited_final)
    ssim: float  # structural similarity index
    pixel_mse: float
    latent_norm_drift: float  # |norm(edited) - norm(baseline)| / norm(baseline)
    trajectory_cosine_mean: float  # mean(cos_sim(baseline_step, edited_step))
    trajectory_cosine_std: float
    counterexample: str | None  # description if a failure mode detected


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Simplified SSIM in [0, 1] — luminance-only, no Gaussian weighting.

    Used as a quick proxy; not a full perceptual metric.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    # Normalize each to [0, 1] if needed.
    a_norm = a.ravel().astype(np.float64)
    b_norm = b.ravel().astype(np.float64)
    if a_norm.min() < 0:
        a_norm = a_norm * 0.5 + 0.5
    if b_norm.min() < 0:
        b_norm = b_norm * 0.5 + 0.5
    mu_a = a_norm.mean()
    mu_b = b_norm.mean()
    sigma_a_sq = a_norm.var()
    sigma_b_sq = b_norm.var()
    sigma_ab = np.mean((a_norm - mu_a) * (b_norm - mu_b))  # type: ignore[reportUnknownVariableType]
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_val = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)  # type: ignore[reportUnknownVariableType]
    ssim_val /= (mu_a**2 + mu_b**2 + c1) * (sigma_a_sq + sigma_b_sq + c2)  # type: ignore[reportUnknownVariableType]
    return float(ssim_val)  # type: ignore[reportUnknownArgumentType]


def compute_trajectory_cosine(
    baseline_states: list[np.ndarray], edited_states: list[np.ndarray]
) -> tuple[float, float]:
    """Per-step cosine similarity between two trajectories."""
    min_len = min(len(baseline_states), len(edited_states))
    if min_len < 1:
        return 0.0, 0.0
    sims = np.array(
        [
            float(np.dot(b.ravel(), e.ravel()) / (np.linalg.norm(b.ravel()) * np.linalg.norm(e.ravel()) + 1e-15))
            for b, e in zip(baseline_states[:min_len], edited_states[:min_len])
        ]
    )
    return float(sims.mean()), float(sims.std(ddof=1) if len(sims) > 1 else 0.0)


def measure(baseline: Any, edited: Any, label: str, seed: int) -> Metrics:
    """Compute all metrics between a baseline and edited run."""
    # Final latent cosine distance.
    b_final = baseline.final_vae_latent.ravel()
    e_final = edited.final_vae_latent.ravel()
    cos_sim = float(np.dot(b_final, e_final) / (np.linalg.norm(b_final) * np.linalg.norm(e_final) + 1e-15))
    final_cosine_dist = 1.0 - cos_sim

    # Image-level metrics.
    pixel_mse = float(np.mean((baseline.images - edited.images) ** 2))  # type: ignore[reportUnknownArgumentType]
    ssim_val = compute_ssim(baseline.images, edited.images)

    # Latent norm drift (using final latent).
    b_norm = float(np.linalg.norm(baseline.final_vae_latent))
    e_norm = float(np.linalg.norm(edited.final_vae_latent))
    latent_norm_drift = abs(e_norm - b_norm) / (b_norm + 1e-15)

    # Trajectory cosine similarity.
    b_states = [s.latent for s in baseline.scheduler_states]
    e_states = [s.latent for s in edited.scheduler_states]
    traj_mean, traj_std = compute_trajectory_cosine(b_states, e_states)

    # Counterexample detection.
    counterexample = None
    if ssim_val < 0.3:
        counterexample = f"SSIM={ssim_val:.3f} — severe image degradation"
    elif latent_norm_drift > 0.5:
        counterexample = f"norm drift={latent_norm_drift:.2%} — trajectory destabilized"

    return Metrics(
        label=label,
        seed=seed,
        final_cosine_dist=final_cosine_dist,
        ssim=ssim_val,
        pixel_mse=pixel_mse,
        latent_norm_drift=latent_norm_drift,
        trajectory_cosine_mean=traj_mean,
        trajectory_cosine_std=traj_std,
        counterexample=counterexample,
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_condition(
    pipe: DiffusersConditionalPipeline,
    prompt: str,
    seed: int,
    intervention: SchedulerIntervention | None = None,
) -> Any:
    """Run one generation condition and return the result."""
    req = GenerationRequest(
        prompt=prompt,
        num_inference_steps=NUM_STEPS,
        seed=seed,
        capture_scheduler_states=True,
        capture_denoiser_location=None,
    )
    return pipe.generate(req, intervention=intervention)


def run_sweep_seed(
    pipe: DiffusersConditionalPipeline,
    seed: int,
    start_step: int,
    strength: float,
) -> Metrics:
    """Run a sweep point: random-dir intervention with given start and strength."""
    # Generate baseline (no intervention) for this seed.
    baseline = run_condition(pipe, PROMPT, seed, intervention=None)

    # Build the intervention.
    shape = baseline.final_vae_latent.shape
    intervention = DiffusersConditionalPipeline.random_direction(
        shape=shape, seed=seed, strength=strength, step_range=(start_step, NUM_STEPS)
    )
    edited = run_condition(pipe, PROMPT, seed, intervention=intervention)
    return measure(baseline, edited, f"sweep_s{start_step}_str{strength}", seed)


def format_table(rows: list[Metrics]) -> str:
    """Format a list of Metrics as a text table."""
    header = (
        f"{'Condition':<22} {'Seed':>5} {'CosDist':>8} {'SSIM':>6} "
        f"{'MSE':>10} {'NormDrift':>10} {'TrajCos':>8} {'TrajStd':>8}  Counterexample"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for m in rows:
        ce = m.counterexample if m.counterexample else "—"
        lines.append(
            f"{m.label:<22} {m.seed:>5} {m.final_cosine_dist:>8.4f} {m.ssim:>6.3f} "
            f"{m.pixel_mse:>10.2e} {m.latent_norm_drift:>10.2%} "
            f"{m.trajectory_cosine_mean:>8.4f} {m.trajectory_cosine_std:>8.4f}  {ce}"
        )
    return "\n".join(lines)


def aggregate_label(rows: list[Metrics]) -> str:
    """Aggregate (mean ± std) across seeds for one label."""
    if not rows:
        return ""
    label = rows[0].label
    final_cd = [r.final_cosine_dist for r in rows]
    ssims = [r.ssim for r in rows]
    mses = [r.pixel_mse for r in rows]
    drifts = [r.latent_norm_drift for r in rows]
    tmeans = [r.trajectory_cosine_mean for r in rows]
    m = lambda vals: (float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))  # type: ignore[reportUnknownLambdaType,reportUnknownArgumentType,reportUnknownVariableType,reportGeneralTypeIssues]  # noqa: E731
    return (
        f"{label:<22} {'agg':>5} {m(final_cd)[0]:>8.4f}±{m(final_cd)[1]:.4f} "
        f"{m(ssims)[0]:>6.3f}±{m(ssims)[1]:.3f} "
        f"{m(mses)[0]:>10.2e}±{m(mses)[1]:.1e} "
        f"{m(drifts)[0]:>10.2%}±{m(drifts)[1]:.2%} "
        f"{m(tmeans)[0]:>8.4f}±{m(tmeans)[1]:.4f} "
        f"{'—':>8}  —"
    )


def build_controls_figure(
    baseline: Any,
    controls: dict[str, Any],
    seed: int,
) -> Any:
    """Build a paired-output comparison figure for one seed."""
    n_controls = len(controls) + 1  # +1 for baseline
    fig, axes = plt.subplots(2, n_controls, figsize=(4 * n_controls, 8))
    fig.suptitle(f"Intervention controls comparison  (seed={seed})", fontsize=13, fontweight="bold")  # type: ignore[reportUnknownMemberType]

    names = ["baseline"] + list(controls.keys())
    results = [baseline] + list(controls.values())

    for col, (name, res) in enumerate(zip(names, results)):
        img = res.images[0]  # NHWC -> HWC
        axes[0, col].imshow(img)
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].axis("off")

        # Show latent norm profile.
        states = res.scheduler_states
        if states:
            norms = [float(np.linalg.norm(s.latent)) for s in states]
            axes[1, col].plot(range(len(norms)), norms, marker=".", markersize=3)
            axes[1, col].set_xlabel("step")
            axes[1, col].set_ylabel("latent norm")
            axes[1, col].grid(True, alpha=0.3)
        else:
            axes[1, col].text(0.5, 0.5, "no states", ha="center", va="center")

    plt.tight_layout()
    return fig


def build_sweep_figure(
    timestep_grid: dict[int, dict[float, Metrics]],
) -> Any:
    """Build a 2-panel sweep figure: timestep sweep + strength sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Fix strength, vary start step.
    ax = axes[0]
    for strength in SWEEP_STRENGTHS:
        steps = sorted(timestep_grid.keys())
        vals = []
        for s in steps:
            if strength in timestep_grid[s]:
                vals.append(timestep_grid[s][strength].final_cosine_dist)  # type: ignore[reportUnknownMemberType]
            else:
                vals.append(np.nan)  # type: ignore[reportUnknownMemberType]
        ax.plot(steps, vals, marker="o", label=f"strength={strength}")
    ax.set_xlabel("Intervention start step")
    ax.set_ylabel("Final cosine distance")
    ax.set_title("Timestep sweep (strength varies)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Fix start step, vary strength.
    ax = axes[1]
    for start_step in SWEEP_START_STEPS:
        strengths = sorted(timestep_grid[start_step].keys())
        vals = [timestep_grid[start_step][s].final_cosine_dist for s in strengths]
        ax.plot(strengths, vals, marker="o", label=f"start={start_step}")
    ax.set_xlabel("Intervention strength")
    ax.set_ylabel("Final cosine distance")
    ax.set_title("Strength sweep (start step varies)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Sprint 38 — Scheduler Latent Intervention Experiment")
    print("=" * 60)
    print(f"Model:     {CONDITIONAL_MODEL_ID} @ {CONDITIONAL_MODEL_REVISION[:12]}")
    print(f"Prompt:    {PROMPT!r}")
    print(f"Steps:     {NUM_STEPS}")
    print(f"Seeds:     {SEEDS}")
    print()

    pipe = DiffusersConditionalPipeline(device="cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[Metrics] = []
    counterexamples: list[str] = []

    # ------------------------------------------------------------------
    # Per-seed control comparison
    # ------------------------------------------------------------------
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # Baseline (no edit).
        baseline = run_condition(pipe, PROMPT, seed, intervention=None)

        # Prompt-only control.
        prompt_only = run_condition(pipe, PROMPT_ABLATED, seed, intervention=None)

        # Random-direction intervention.
        shape = baseline.final_vae_latent.shape
        random_int = DiffusersConditionalPipeline.random_direction(
            shape=shape, seed=seed, strength=DEFAULT_STRENGTH, step_range=DEFAULT_STEP_RANGE
        )
        random_edited = run_condition(pipe, PROMPT, seed, intervention=random_int)

        # Matched-norm intervention.
        # Use the latent at the first intervention step to match norms.
        ref_latent = (
            baseline.scheduler_states[DEFAULT_STEP_RANGE[0]].latent
            if len(baseline.scheduler_states) > DEFAULT_STEP_RANGE[0]
            else baseline.final_vae_latent
        )
        matched_int = DiffusersConditionalPipeline.matched_norm_direction(
            ref_latent, seed=seed, strength=DEFAULT_STRENGTH, step_range=DEFAULT_STEP_RANGE
        )
        matched_edited = run_condition(pipe, PROMPT, seed, intervention=matched_int)

        # Measure each condition against baseline.
        controls = {
            "prompt-only": prompt_only,
            "random-dir": random_edited,
            "matched-norm": matched_edited,
        }
        for label, edited in controls.items():
            m = measure(baseline, edited, label, seed)
            all_metrics.append(m)
            if m.counterexample:
                counterexamples.append(f"[seed={seed}, {label}] {m.counterexample}")
            print(
                f"  {label:<16} cos_dist={m.final_cosine_dist:.4f}  SSIM={m.ssim:.3f}  "
                f"norm_drift={m.latent_norm_drift:.2%}  traj_cos={m.trajectory_cosine_mean:.4f}"
            )

        # Build controls figure for the first seed only (to save time).
        if seed == SEEDS[0]:
            fig = build_controls_figure(baseline, controls, seed)
            fig.savefig(str(CONTROLS_PNG), dpi=150)  # type: ignore[reportUnknownMemberType]
            plt.close(fig)
            print(f"  Saved {CONTROLS_PNG}")

    # ------------------------------------------------------------------
    # Timestep and strength sweep (single seed to limit duration)
    # ------------------------------------------------------------------
    print("\n--- Sweep: timestep × strength ---")
    sweep_seed = SEEDS[0]
    baseline_sweep = run_condition(pipe, PROMPT, sweep_seed, intervention=None)
    timestep_grid: dict[int, dict[float, Metrics]] = {}

    for start_step in SWEEP_START_STEPS:
        timestep_grid[start_step] = {}
        for strength in SWEEP_STRENGTHS:
            shape = baseline_sweep.final_vae_latent.shape
            intervention = DiffusersConditionalPipeline.random_direction(
                shape=shape, seed=sweep_seed, strength=strength, step_range=(start_step, NUM_STEPS)
            )
            edited = run_condition(pipe, PROMPT, sweep_seed, intervention=intervention)
            m = measure(baseline_sweep, edited, "sweep", sweep_seed)
            timestep_grid[start_step][strength] = m
            all_metrics.append(m)
            print(
                f"  start={start_step:>2d}  strength={strength:>4.1f}  "
                f"cos_dist={m.final_cosine_dist:.4f}  SSIM={m.ssim:.3f}  "
                f"norm_drift={m.latent_norm_drift:.2%}"
            )
            if m.counterexample:
                counterexamples.append(f"[sweep, start={start_step}, strength={strength}] {m.counterexample}")

    # Build sweep figure.
    fig = build_sweep_figure(timestep_grid)
    fig.savefig(str(SWEEP_PNG), dpi=150)
    plt.close(fig)
    print(f"  Saved {SWEEP_PNG}")

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------
    lines = [
        "=" * 60,
        "Sprint 38 — Scheduler Latent Intervention Experiment - Results",
        "=" * 60,
        f"Model:     {CONDITIONAL_MODEL_ID}",
        f"Revision:  {CONDITIONAL_MODEL_REVISION}",
        f"Prompt:    {PROMPT!r}",
        f"Ablated:   {PROMPT_ABLATED!r}",
        f"Steps:     {NUM_STEPS}",
        f"Seeds:     {SEEDS}",
        f"Default strength: {DEFAULT_STRENGTH}",
        f"Default step range: {DEFAULT_STEP_RANGE}",
        "",
        "--- Per-seed metrics ---",
        format_table([m for m in all_metrics if "sweep" not in m.label]),
        "",
        "--- Aggregate (mean ± σ across seeds) ---",
        "",
    ]

    for label in ("prompt-only", "random-dir", "matched-norm"):
        seed_rows = [m for m in all_metrics if m.label == label and "sweep" not in m.label]
        if seed_rows:
            lines.append(aggregate_label(seed_rows))

    # Sweep summary (single seed, no uncertainty).
    lines.append("")
    lines.append("--- Sweep: intervention start step × strength ---")
    lines.append("(single seed, random-direction intervention)")
    sweep_rows = [m for m in all_metrics if "sweep" in m.label]
    lines.append(format_table(sweep_rows))

    # Counterexamples.
    lines.append("")
    lines.append("--- Counterexamples (failure modes detected) ---")
    if counterexamples:
        for ce in counterexamples:
            lines.append(f"  !! {ce}")
    else:
        lines.append("  None detected within declared thresholds.")

    # Promotion decisions (Task 8).
    lines.append("")
    lines.append("--- Evidence promotion check ---")
    lines.append("Thresholds per predeclaration:")
    lines.append("  D2 target:  cos_dist > 0.05 AND SSIM > 0.7 AND norm_drift < 20 % across >=2 seeds")
    lines.append("  D1:         otherwise")

    # Check per condition.
    for label in ("random-dir", "matched-norm"):
        seed_rows = [m for m in all_metrics if m.label == label and "sweep" not in m.label]
        passed = sum(1 for m in seed_rows if m.final_cosine_dist > 0.05 and m.ssim > 0.7 and m.latent_norm_drift < 0.20)
        if passed >= 2:
            lines.append(f"  {label}: D2 (thresholds met on {passed}/{len(seed_rows)} seeds)")
        else:
            lines.append(f"  {label}: D1 (thresholds met on {passed}/{len(seed_rows)} seeds)")

    lines.append("")
    lines.append("--- Note ---")
    lines.append("These results were generated with random directions (not learned concept")
    lines.append("directions).  A concept-specific direction (e.g., from activation patching")
    lines.append("or contrastive embedding pairs) would likely produce different targeting.")
    lines.append("The random-direction controls establish the lower bound of intervention effect.")

    summary = "\n".join(lines)
    with open(str(METRICS_FILE), "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSaved {METRICS_FILE}")

    # Print summary to stdout.
    intervention_target = [m for m in all_metrics if m.label == "random-dir" and "sweep" not in m.label]
    if intervention_target:
        avg_cos = float(np.mean([m.final_cosine_dist for m in intervention_target]))
        avg_ssim = float(np.mean([m.ssim for m in intervention_target]))
        avg_drift = float(np.mean([m.latent_norm_drift for m in intervention_target]))
        print(f"\n{'=' * 60}")
        print(f"Summary (random-dir, n={len(intervention_target)} seeds):")
        print(f"  Target change (cos dist):  {avg_cos:.4f}")
        print(f"  Content preservation (SSIM): {avg_ssim:.3f}")
        print(f"  Latent norm drift:          {avg_drift:.2%}")


if __name__ == "__main__":
    main()
