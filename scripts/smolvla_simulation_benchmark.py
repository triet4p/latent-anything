"""Deterministic SmolVLA causal simulation benchmark through LeRobot evaluation.

Runs the pinned SmolVLA checkpoint against the LIBERO ``libero_spatial``
simulation suite under the four predeclared conditions (``no_hook``,
``baseline``, ``random``, ``targeted``) and writes a reproducible artifact:

- ``smolvla_simulation_benchmark.json`` — full config, per-episode rows,
  per-condition summaries with Wilson intervals, offline explanation scores,
  offline-to-environment correlation, acceptance gate, and failure analysis;
- ``smolvla_simulation_benchmark_config.json`` — the standalone config for
  rerunning the identical experiment;
- ``smolvla_simulation_benchmark.png`` — success, action-deviation, and
  latency plots (video rendering is intentionally omitted: LIBERO videos are
  large and the quantitative failure analysis covers behavior).

The real lane requires CUDA and the ``lerobot-smolvla`` profile (which now
also carries LeRobot's Linux-only ``libero`` environment extra).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from latent_anything.integrations.lerobot_benchmark import (  # noqa: E402
    BenchmarkEnvironmentBundle,
    SimulationBenchmarkConfig,
    SimulationBenchmarkResult,
    build_libero_benchmark_environment,
    run_simulation_benchmark,
)
from latent_anything.integrations.lerobot_smolvla import (  # noqa: E402
    DEFAULT_SMOLVLA_CHECKPOINT,
    SmolVLAPolicyAdapter,
    load_smolvla_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="Episode seeds per condition.")
    parser.add_argument("--strengths", type=float, nargs="+", default=[1.0], help="Intervention strengths.")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None, help="LIBERO task ids to evaluate.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override the episode step budget.")
    parser.add_argument("--probe-queries", type=int, default=2, help="Executed queries collected for offline scores.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/smolvla_simulation_benchmark.json"))
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("artifacts/smolvla_simulation_benchmark_config.json"),
    )
    parser.add_argument("--plot-output", type=Path, default=Path("artifacts/smolvla_simulation_benchmark.png"))
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SimulationBenchmarkConfig:
    return SimulationBenchmarkConfig(
        seeds=tuple(args.seeds),
        strengths=tuple(args.strengths),
        task_ids=tuple(args.task_ids) if args.task_ids else None,
        max_episode_steps=args.max_steps,
        probe_queries=args.probe_queries,
    )


def load_runtime(args: argparse.Namespace) -> tuple[SmolVLAPolicyAdapter, BenchmarkEnvironmentBundle]:
    config = build_config(args)
    adapter = load_smolvla_policy(DEFAULT_SMOLVLA_CHECKPOINT, device=args.device)
    environment = build_libero_benchmark_environment(config)
    return adapter, environment


def write_plots(result: SimulationBenchmarkResult, output: Path) -> None:
    labels = [f"{summary.condition}@{summary.strength}" for summary in result.summaries]
    success = [summary.success_rate for summary in result.summaries]
    success_low = [summary.success_ci_low for summary in result.summaries]
    success_high = [summary.success_ci_high for summary in result.summaries]
    deviation = [summary.mean_action_deviation for summary in result.summaries]
    latency = [summary.mean_query_latency_s for summary in result.summaries]
    positions = np.arange(len(result.summaries))

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axis = cast(Axes, axes[0])
    yerr = [
        np.asarray(success) - np.asarray(success_low),
        np.asarray(success_high) - np.asarray(success),
    ]
    axis.bar(positions, success, yerr=yerr, capsize=4)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel("success rate")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Success rate per condition (Wilson 95% CI)")

    axis = cast(Axes, axes[1])
    axis.bar(positions, deviation)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel("mean action deviation vs no_hook")
    axis.set_title("Action deviation per condition")

    axis = cast(Axes, axes[2])
    axis.bar(positions, latency)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel("mean query latency (s)")
    axis.set_title("Policy query latency per condition")

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def print_failure_analysis(result: SimulationBenchmarkResult) -> None:
    failures = result.failure_analysis.per_condition
    print("Failure analysis (unsuccessful episodes per condition):")
    for condition, outcomes in failures.items():
        print(f"  {condition}: {len(outcomes)} failed episode(s)")
        for outcome in outcomes:
            print(
                f"    - {outcome.episode_key}: length={outcome.length}, deviation={outcome.mean_action_deviation:.4f}"
            )
    for note in result.failure_analysis.notes:
        print(f"  note: {note}")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    adapter, environment = load_runtime(args)
    try:
        result = run_simulation_benchmark(adapter, environment, config)
    finally:
        close = getattr(environment.env, "close", None)
        if callable(close):
            close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.config_output.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_plots(result, args.plot_output)

    print(f"Benchmark written to {args.output}")
    print(f"Config written to {args.config_output}")
    print(f"Plots written to {args.plot_output}")
    for summary in result.summaries:
        print(
            f"  {summary.condition}@{summary.strength}: success={summary.success_rate:.2f} "
            f"[{summary.success_ci_low:.2f}, {summary.success_ci_high:.2f}], "
            f"return={summary.mean_return:.2f}, deviation={summary.mean_action_deviation:.4f}, "
            f"latency={summary.mean_query_latency_s * 1000:.1f} ms"
        )
    for score in result.offline_scores:
        print(
            f"  offline {score.condition}@{score.strength}: on_target={score.on_target_fraction:.3f}, "
            f"action_change={score.action_change_norm:.4f}, drift={score.representation_drift:.4f}"
        )
    for disagreement in result.correlation.disagreements:
        print(f"  DISAGREEMENT: {disagreement}")
    for note in result.correlation.notes:
        print(f"  correlation note: {note}")
    print_failure_analysis(result)
    print("Acceptance checks:")
    for name, passed in result.acceptance.checks.items():
        print(f"  {name}: {passed}")
    if result.acceptance.passed:
        print("Acceptance: PASSED")
    else:
        print(f"Acceptance: FAILED ({', '.join(result.acceptance.failures)})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
