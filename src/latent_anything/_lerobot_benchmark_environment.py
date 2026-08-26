"""Private official LIBERO/LeRobot environment construction."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from latent_anything.integrations.lerobot import LeRobotAPI, load_lerobot_api

if TYPE_CHECKING:
    from latent_anything.integrations.lerobot_benchmark import (
        BenchmarkEnvironmentBundle,
        SimulationBenchmarkConfig,
    )


def build_libero_benchmark_environment(
    config: SimulationBenchmarkConfig,
    *,
    api: LeRobotAPI | None = None,
) -> BenchmarkEnvironmentBundle:
    """Create the official LeRobot LIBERO vector environment and processors."""

    from latent_anything.integrations import lerobot_benchmark as benchmark

    upstream_api = api if api is not None else load_lerobot_api()
    benchmark._bootstrap_libero_config()  # pyright: ignore[reportPrivateUsage] # noqa: SLF001 - private benchmark seam
    envs_module = import_module("lerobot.envs")
    make_env_config = getattr(envs_module, "make_env_config")  # noqa: B009 - optional upstream symbol
    env_config = make_env_config(
        config.env_type,
        task=config.task,
        task_ids=list(config.task_ids) if config.task_ids is not None else None,
        episode_length=config.episode_length,
        observation_height=config.observation_height,
        observation_width=config.observation_width,
    )
    env_preprocessor, env_postprocessor = env_config.get_env_processors()
    del env_postprocessor
    task_id = config.task_ids[0] if config.task_ids else 0

    def create_env() -> object:
        envs = upstream_api.make_env(env_config, n_envs=1)
        suite_map = cast(dict[str, dict[int, object]], envs)
        suite_key = next((key for key in (config.task, config.env_type) if key in suite_map), None)
        if suite_key is None:
            raise ValueError(
                f"environment factory returned no {config.env_type!r}/{config.task!r} suite: {sorted(suite_map)}"
            )
        task_map = suite_map[suite_key]
        if task_id not in task_map:
            raise ValueError(f"environment factory returned no task {task_id} for {suite_key!r}: {sorted(task_map)}")
        return task_map[task_id]

    utils_module = import_module("lerobot.envs.utils")
    preprocess_observation = getattr(utils_module, "preprocess_observation")  # noqa: B009 - optional upstream symbol
    probe_env = create_env()
    try:
        task_description = benchmark._call_task_description(probe_env)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        max_steps = (
            int(config.max_episode_steps)
            if config.max_episode_steps is not None
            else benchmark._call_max_steps(probe_env)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        )
    finally:
        close = getattr(probe_env, "close", None)
        if callable(close):
            close()
    return benchmark.BenchmarkEnvironmentBundle(
        env_factory=create_env,
        env_preprocessor=env_preprocessor,
        preprocess_observation=preprocess_observation,
        task_description=task_description,
        max_episode_steps=max_steps,
        metadata={
            "env_type": config.env_type,
            "task": config.task,
            "task_id": task_id,
            "task_description": task_description,
            "max_episode_steps": max_steps,
            "n_envs": 1,
        },
    )
