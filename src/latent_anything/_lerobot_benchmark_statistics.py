"""Private causal benchmark statistics, acceptance, and artifact assembly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from latent_anything.integrations.lerobot_smolvla import SmolVLAPolicyAdapter

if TYPE_CHECKING:
    from latent_anything.integrations.lerobot_benchmark import (
        BenchmarkCondition,
        BenchmarkEnvironmentBundle,
        CausalCorrelation,
        CausalCorrelationCell,
        ConditionSummary,
        EpisodeOutcome,
        SimulationBenchmarkConfig,
        SimulationBenchmarkResult,
    )


def wilson_ci(successes: Sequence[bool], z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a success proportion."""

    if not successes:
        raise ValueError("wilson_ci requires at least one episode")
    n = len(successes)
    proportion = sum(successes) / n
    denominator = 1.0 + z**2 / n
    centre = (proportion + z**2 / (2.0 * n)) / denominator
    margin = z * np.sqrt((proportion * (1.0 - proportion) + z**2 / (4.0 * n)) / n) / denominator
    return max(0.0, float(centre - margin)), min(1.0, float(centre + margin))


def normal_ci(values: Sequence[float], z: float = 1.96) -> tuple[float, float]:
    """Normal approximation interval for a continuous metric."""

    if not values:
        raise ValueError("normal_ci requires at least one value")
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean
    standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array)))
    return mean - z * standard_error, mean + z * standard_error


def summarize(
    outcomes: Sequence[EpisodeOutcome],
    condition: BenchmarkCondition,
    strength: float,
) -> ConditionSummary:
    """Aggregate one condition/strength cell with confidence intervals."""

    from latent_anything.integrations import lerobot_benchmark as benchmark

    if not outcomes:
        raise ValueError(f"no episodes for {condition} at strength {strength}")
    successes = [outcome.success for outcome in outcomes]
    returns = [outcome.sum_reward for outcome in outcomes]
    low, high = wilson_ci(successes)
    return_low, return_high = normal_ci(returns)
    return benchmark.ConditionSummary(
        condition=condition,
        strength=strength,
        n_episodes=len(outcomes),
        success_rate=sum(successes) / len(successes),
        success_ci_low=low,
        success_ci_high=high,
        mean_return=float(np.mean(returns)),
        return_ci_low=return_low,
        return_ci_high=return_high,
        mean_length=float(np.mean([outcome.length for outcome in outcomes])),
        mean_action_deviation=float(np.mean([outcome.mean_action_deviation for outcome in outcomes])),
        mean_query_latency_s=float(np.mean([outcome.mean_query_latency_s for outcome in outcomes])),
        first_query_latency_s=float(np.mean([outcome.first_query_latency_s for outcome in outcomes])),
    )


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank-based correlation over paired cells; None when underpowered."""

    if len(x) < 3 or len(x) != len(y):
        return None
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if float(np.std(x_array)) == 0.0 or float(np.std(y_array)) == 0.0:
        return None

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(values))
        return order.astype(float) + 1.0

    dx = ranks(x_array) - np.mean(ranks(x_array))
    dy = ranks(y_array) - np.mean(ranks(y_array))
    denominator = float(np.linalg.norm(dx) * np.linalg.norm(dy))
    if denominator == 0.0:
        return None
    return float(np.sum(dx * dy) / denominator)


def build_correlation(cells: Sequence[CausalCorrelationCell]) -> CausalCorrelation:
    """Compare offline explanation scores with environment effects."""

    from latent_anything.integrations import lerobot_benchmark as benchmark

    disagreements: list[str] = []
    notes: list[str] = []
    for cell in cells:
        if cell.condition != "targeted":
            continue
        if cell.on_target_fraction >= 0.8 and abs(cell.success_delta) < 0.2:
            disagreements.append(
                f"overstatement: targeted strength={cell.strength} has offline on-target "
                f"{cell.on_target_fraction:.2f} but the environment success delta is only "
                f"{cell.success_delta:+.2f}"
            )
        if cell.on_target_fraction < 0.5 and abs(cell.success_delta) >= 0.2:
            disagreements.append(
                f"understatement: targeted strength={cell.strength} has offline on-target "
                f"{cell.on_target_fraction:.2f} yet the environment success delta is "
                f"{cell.success_delta:+.2f}"
            )
        if cell.success_delta <= -0.2:
            disagreements.append(
                f"reversal: targeted strength={cell.strength} changed success by "
                f"{cell.success_delta:+.2f}, worse than the offline explanation suggested"
            )
    if len(cells) < 3:
        notes.append("fewer than three cells compared; Spearman correlation not computed")
    rho = spearman(
        [cell.on_target_fraction for cell in cells],
        [cell.mean_action_deviation for cell in cells],
    )
    if rho is None:
        notes.append("Spearman correlation not computed (insufficient cells or no variance)")
    else:
        notes.append("Spearman correlation computed between offline on-target fraction and mean action deviation")
    if not cells:
        raise ValueError("correlation requires at least one cell")
    return benchmark.CausalCorrelation(
        cells=tuple(cells),
        spearman_rho=rho,
        disagreements=tuple(disagreements),
        notes=tuple(notes),
    )


def run_simulation_benchmark(
    adapter: SmolVLAPolicyAdapter,
    environment: BenchmarkEnvironmentBundle,
    config: SimulationBenchmarkConfig,
    *,
    noise: np.ndarray | None = None,
) -> SimulationBenchmarkResult:
    """Run the deterministic four-condition causal benchmark."""

    from latent_anything._lerobot_benchmark_execution import run_episode
    from latent_anything.integrations import lerobot_benchmark as benchmark
    from latent_anything.integrations.lerobot_smolvla import (
        SmolVLAIntervention,
        measure_smolvla_intervention,
    )

    if not isinstance(noise, np.ndarray):
        noise = np.full(
            (1, adapter.metadata.chunk_size, adapter.metadata.max_action_dim),
            config.noise_value,
        )
    rng = np.random.default_rng(config.intervention_seed)
    random_direction = benchmark._random_expert_direction(adapter.expert_dim, rng)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
    targeted_direction = benchmark._targeted_expert_direction(adapter, action_axis=config.action_axis)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
    direction_for: dict[benchmark.BenchmarkCondition, np.ndarray] = {
        "random": random_direction,
        "targeted": targeted_direction,
    }

    outcomes: list[benchmark.EpisodeOutcome] = []
    probe_samples: list[dict[str, object]] = []
    reference_by_seed: dict[int, tuple[np.ndarray, ...]] = {}
    for seed in config.seeds:
        for condition in config.conditions:
            strengths = (0.0,) if condition in ("no_hook", "baseline") else config.strengths
            for strength in strengths:
                reference = reference_by_seed.get(seed)
                collect_probes = (
                    condition == "no_hook" and seed == config.seeds[0] and len(probe_samples) < config.probe_queries
                )
                outcome, samples = run_episode(
                    adapter,
                    environment,
                    seed=seed,
                    condition=condition,
                    strength=strength,
                    direction=direction_for.get(condition, np.zeros(adapter.expert_dim)),
                    noise=noise,
                    reference_actions=reference,
                    record_samples=collect_probes,
                )
                if condition == "no_hook":
                    reference_by_seed[seed] = outcome.actions
                outcomes.append(outcome)
                if samples:
                    probe_samples.extend(cast(list[dict[str, object]], samples))

    summaries = tuple(
        summarize(
            [outcome for outcome in outcomes if outcome.condition == condition and outcome.strength == strength],
            condition,
            strength,
        )
        for condition in config.conditions
        for strength in ((0.0,) if condition in ("no_hook", "baseline") else config.strengths)
    )
    summary_by_cell = {(summary.condition, summary.strength): summary for summary in summaries}

    offline_scores: list[benchmark.OfflineExplanationScore] = []
    if probe_samples:
        for condition in ("random", "targeted"):
            if condition not in config.conditions:
                continue
            direction = direction_for[condition]
            for strength in config.strengths:
                measurement = measure_smolvla_intervention(
                    adapter,
                    probe_samples,
                    noise=noise,
                    intervention=SmolVLAIntervention(direction=direction, strength=strength),
                )
                offline_scores.append(
                    benchmark.OfflineExplanationScore(
                        condition=cast(benchmark.BenchmarkCondition, condition),
                        strength=strength,
                        on_target_fraction=measurement.on_target_fraction,
                        action_change_norm=measurement.action_change_norm,
                        representation_drift=measurement.representation_drift,
                        probe_queries=len(probe_samples),
                    )
                )

    cells: list[benchmark.CausalCorrelationCell] = []
    no_hook_success_rate = summary_by_cell[("no_hook", 0.0)].success_rate
    for score in offline_scores:
        summary = summary_by_cell[(score.condition, score.strength)]
        cells.append(
            benchmark.CausalCorrelationCell(
                condition=score.condition,
                strength=score.strength,
                on_target_fraction=score.on_target_fraction,
                action_change_norm=score.action_change_norm,
                representation_drift=score.representation_drift,
                mean_action_deviation=summary.mean_action_deviation,
                success_delta=summary.success_rate - no_hook_success_rate,
            )
        )
    if not cells:
        correlation = benchmark.CausalCorrelation(
            cells=(),
            spearman_rho=None,
            disagreements=(),
            notes=(
                "no offline explanation scores were produced because the probe episode "
                "collected no executed-query samples; correlation is unavailable",
            ),
        )
    else:
        correlation = build_correlation(cells)

    checks: dict[str, bool] = {
        "baseline_actions_bit_exact": benchmark._baseline_is_bit_exact(outcomes),  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        "baseline_success_equals_no_hook": benchmark._baseline_success_matches(outcomes),  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        "intervention_changes_actions": benchmark._interventions_change_actions(outcomes),  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        "all_episodes_within_max_steps": benchmark._episodes_within_budget(outcomes, environment.max_episode_steps),  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
    }
    failures = [key for key, passed in checks.items() if not passed]
    acceptance = benchmark.BenchmarkAcceptance(passed=not failures, checks=checks, failures=tuple(failures))
    failure_analysis = benchmark.FailureAnalysis(
        per_condition={
            condition: tuple(outcome for outcome in outcomes if outcome.condition == condition and not outcome.success)
            for condition in config.conditions
        },
        notes=(
            "unsuccessful episodes are grouped per condition; each row records the seed, "
            "strength, length, and action deviation against the no_hook reference",
        ),
    )
    return benchmark.SimulationBenchmarkResult(
        config=config,
        environment_metadata=dict(environment.metadata),
        outcomes=tuple(outcomes),
        summaries=summaries,
        offline_scores=tuple(offline_scores),
        correlation=correlation,
        acceptance=acceptance,
        failure_analysis=failure_analysis,
        claim_scope=(
            "environment-level causal evidence for the SmolVLA action-expert intervention on "
            f"{config.env_type}/{config.task}: the official preprocess/select_action/postprocess path is "
            "executed per condition, episodes share seeds, noise, and initial states, and success/return/"
            "deviation/latency are compared against the no_hook control"
        ),
    )
