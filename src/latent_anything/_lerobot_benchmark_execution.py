"""Private causal benchmark episode execution through official LeRobot seams."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from latent_anything.integrations.lerobot_smolvla import SmolVLAPolicyAdapter

if TYPE_CHECKING:
    from latent_anything.integrations.lerobot_benchmark import (
        BenchmarkCondition,
        BenchmarkEnvironmentBundle,
        EpisodeOutcome,
    )


def run_episode(
    adapter: SmolVLAPolicyAdapter,
    environment: BenchmarkEnvironmentBundle,
    *,
    seed: int,
    condition: BenchmarkCondition,
    strength: float,
    direction: np.ndarray,
    noise: np.ndarray,
    reference_actions: Sequence[np.ndarray] | None = None,
    record_samples: bool = False,
) -> tuple[EpisodeOutcome, tuple[Mapping[str, object], ...]]:
    """Roll out one causal benchmark cell on a fresh, seeded environment."""

    from latent_anything.integrations import lerobot_benchmark as benchmark
    from latent_anything.integrations.lerobot_smolvla import SmolVLAIntervention

    if condition not in benchmark.VALID_CONDITIONS:
        raise ValueError(f"unknown benchmark condition {condition!r}")
    env = environment.env_factory()
    reset = cast(
        object,
        getattr(env, "reset", None),
    )
    step = cast(object, getattr(env, "step", None))
    if not callable(reset) or not callable(step):
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise TypeError("benchmark environment must expose reset() and step()")
    try:
        adapter.reset()
        observation, _info = cast(tuple[Mapping[str, object], Mapping[str, object]], reset(seed=seed))
        samples: list[Mapping[str, object]] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        query_latencies: list[float] = []
        query_steps: list[int] = []
        success = False
        terminated = False
        step_index = 0
        while step_index < environment.max_episode_steps:
            observed = cast(Mapping[str, object], observation)
            converted = environment.preprocess_observation(observed)
            with_task = dict(converted)
            with_task["task"] = [environment.task_description]
            sample = environment.env_preprocessor(with_task)
            started = time.perf_counter()
            if condition == "no_hook":
                selection = benchmark._official_select_action(adapter, sample, noise=noise)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
                raw_action = selection.action
                model_query_executed = selection.model_query_executed
            else:
                selection = adapter.select_action(
                    sample,
                    noise=noise,
                    intervention=SmolVLAIntervention(direction=direction, strength=strength),
                    episode_step=step_index,
                )
                raw_action = selection.action
                model_query_executed = selection.model_query_executed
            elapsed = time.perf_counter() - started
            if model_query_executed:
                if record_samples:
                    samples.append(sample)
                query_latencies.append(elapsed)
                query_steps.append(step_index)
            action = benchmark._action_to_numpy(raw_action)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
            actions.append(action)
            observation, reward, done, truncated, info = cast(
                tuple[object, object, object, object, Mapping[str, object]], step(action)
            )
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            terminated = benchmark._termination_flags(done) or benchmark._termination_flags(truncated)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
            success = benchmark._extract_success(info)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
            if terminated:
                break
            step_index += 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    del _info
    outcome = benchmark._build_outcome(  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        seed=seed,
        condition=condition,
        strength=strength,
        success=success,
        rewards=rewards,
        actions=actions,
        latencies=query_latencies,
        query_steps=query_steps,
        terminated=terminated,
        max_steps=environment.max_episode_steps,
        reference_actions=reference_actions,
    )
    return outcome, tuple(samples)
