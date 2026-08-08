"""Causal policy-explanation benchmark through LeRobot simulation evaluation.

Sprint 61 runs a controlled behavioral benchmark: the same fixed-noise policy
queries are executed inside the official LeRobot ``preprocess -> select_action
-> postprocess`` path and rolled out in a LeRobot simulation environment under
four conditions:

- ``no_hook`` — the official path with no capture hooks installed;
- ``baseline`` — the capture hooks installed with an intervention at strength
  zero (bit-exact identity, so behavior must match ``no_hook`` exactly);
- ``random`` — a seeded random action-expert direction added at every
  denoising step;
- ``targeted`` — the action-expert direction that induces the largest change
  along one declared action axis through the policy's own ``action_out_proj``.

Each condition replays the same episode seeds from the same initial state so
the only difference between conditions is the intervention. The benchmark
reports success rate, return, action deviation against the ``no_hook``
trajectory, per-query latency, and Wilson confidence intervals, then
correlates the offline explanation scores (on-target fraction, action change,
representation drift measured with :func:`measure_smolvla_intervention`) with
the environment-level effects and lists explicit disagreements.

Importing this module is safe in a base installation; LeRobot modules are
imported lazily by :func:`build_libero_benchmark_environment`.
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import import_module, machinery
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor, nn

from latent_anything.integrations.lerobot import LeRobotAPI, load_lerobot_api
from latent_anything.integrations.lerobot_smolvla import (
    SmolVLAIntervention,
    SmolVLAPolicyAdapter,
    measure_smolvla_intervention,
)

BenchmarkCondition = Literal["no_hook", "baseline", "random", "targeted"]

VALID_CONDITIONS: tuple[BenchmarkCondition, ...] = ("no_hook", "baseline", "random", "targeted")

SMOLVLA_BENCHMARK_ENV_TYPE = "libero"
SMOLVLA_BENCHMARK_TASK = "libero_spatial"
SMOLVLA_BENCHMARK_ACTION_AXIS = 0


class SimulationBenchmarkConfig(BaseModel):
    """Deterministic, tractable simulation benchmark for the SmolVLA policy.

    Every field except the episode-count/seeds controls is pinned; changing a
    field changes the evidence identity of the run.
    """

    model_config = ConfigDict(frozen=True)

    env_type: str = SMOLVLA_BENCHMARK_ENV_TYPE
    task: str = SMOLVLA_BENCHMARK_TASK
    task_ids: tuple[int, ...] | None = None
    episode_length: int | None = None
    observation_height: int = 256
    observation_width: int = 256
    seeds: tuple[int, ...] = (1, 2, 3)
    conditions: tuple[BenchmarkCondition, ...] = VALID_CONDITIONS
    strengths: tuple[float, ...] = (1.0,)
    intervention_seed: int = 0
    action_axis: int = SMOLVLA_BENCHMARK_ACTION_AXIS
    noise_value: float = 0.0
    probe_queries: int = 2
    max_episode_steps: int | None = None

    @field_validator("env_type", "task")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("env_type and task must not be empty")
        return value

    @field_validator("observation_height", "observation_width")
    @classmethod
    def _positive_observation(cls, value: int) -> int:
        if value < 1:
            raise ValueError("observation dimensions must be positive")
        return value

    @field_validator("intervention_seed", "action_axis")
    @classmethod
    def _non_negative_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("intervention_seed and action_axis must be non-negative")
        return value

    @field_validator("probe_queries")
    @classmethod
    def _positive_probes(cls, value: int) -> int:
        if value < 1:
            raise ValueError("probe_queries must be positive")
        return value

    @field_validator("conditions")
    @classmethod
    def _valid_conditions(cls, value: tuple[BenchmarkCondition, ...]) -> tuple[BenchmarkCondition, ...]:
        if not value:
            raise ValueError("conditions must not be empty")
        for condition in value:
            if condition not in VALID_CONDITIONS:
                raise ValueError(f"unknown benchmark condition {condition!r}")
        if "random" in value and "targeted" not in value:
            raise ValueError("a random condition requires the targeted control to be defined")
        return value

    @field_validator("strengths")
    @classmethod
    def _valid_strengths(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("strengths must not be empty")
        for strength in value:
            if not np.isfinite(strength) or strength == 0.0:
                raise ValueError("intervention strengths must be finite and non-zero")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> SimulationBenchmarkConfig:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        return self

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible config snapshot."""
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class BenchmarkEnvironmentBundle:
    """Raw LeRobot vector-environment factory plus the official processors.

    ``env_factory`` creates a fresh vector environment for every episode cell.
    This is deliberate: LIBERO advances its initial-state index on each
    ``reset``, so sharing one env instance between conditions would silently
    change the initial state and break the comparability the benchmark
    depends on. A fresh environment per (seed, condition) cell always starts
    from initial state 0 with the same seed.
    """

    env_factory: Callable[[], object]
    env_preprocessor: Callable[[Mapping[str, object]], Mapping[str, object]]
    preprocess_observation: Callable[[Mapping[str, object]], Mapping[str, object]]
    task_description: str
    max_episode_steps: int
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeOutcome:
    """One rolled-out episode under one condition."""

    seed: int
    condition: BenchmarkCondition
    strength: float
    success: bool
    sum_reward: float
    max_reward: float
    length: int
    terminated: bool
    mean_action_deviation: float
    mean_query_latency_s: float
    first_query_latency_s: float
    total_latency_s: float
    n_queries: int
    actions: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        frozen = tuple(np.array(action, copy=True) for action in self.actions)
        for action in frozen:
            action.setflags(write=False)
        object.__setattr__(self, "actions", frozen)
        if self.n_queries < 1:
            raise ValueError("n_queries must be positive")

    @property
    def episode_key(self) -> str:
        """Stable per-episode identifier used in summaries and failure lists."""
        return f"seed={self.seed}:condition={self.condition}:strength={self.strength}"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible episode row without action arrays."""
        return {
            "episode_key": self.episode_key,
            "seed": self.seed,
            "condition": self.condition,
            "strength": self.strength,
            "success": self.success,
            "sum_reward": self.sum_reward,
            "max_reward": self.max_reward,
            "length": self.length,
            "terminated": self.terminated,
            "mean_action_deviation": self.mean_action_deviation,
            "mean_query_latency_s": self.mean_query_latency_s,
            "first_query_latency_s": self.first_query_latency_s,
            "total_latency_s": self.total_latency_s,
            "n_queries": self.n_queries,
        }


@dataclass(frozen=True)
class ConditionSummary:
    """Aggregated metrics over the episodes of one (condition, strength) cell."""

    condition: BenchmarkCondition
    strength: float
    n_episodes: int
    success_rate: float
    success_ci_low: float
    success_ci_high: float
    mean_return: float
    return_ci_low: float
    return_ci_high: float
    mean_length: float
    mean_action_deviation: float
    mean_query_latency_s: float
    first_query_latency_s: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible cell summary."""
        return asdict(self)


def wilson_ci(successes: Sequence[bool], z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a success proportion over ``n`` episodes.

    The Wilson interval is honest for the small episode counts this benchmark
    can afford in simulation.
    """

    n = len(successes)
    if n == 0:
        raise ValueError("wilson_ci requires at least one episode")
    k = sum(successes)
    p_hat = k / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _normal_ci(values: Sequence[float], z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation 95% interval for a continuous metric."""

    if not values:
        raise ValueError("_normal_ci requires at least one value")
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, mean
    std = float(np.std(array, ddof=1))
    half = z * std / np.sqrt(len(array))
    return mean - half, mean + half


@dataclass(frozen=True)
class OfflineExplanationScore:
    """Offline, pre-environment measurement for one intervention direction."""

    condition: BenchmarkCondition
    strength: float
    on_target_fraction: float
    action_change_norm: float
    representation_drift: float
    probe_queries: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible offline score row."""
        return asdict(self)


@dataclass(frozen=True)
class CausalCorrelationCell:
    """One (condition, strength) cell pairing offline scores with effects."""

    condition: BenchmarkCondition
    strength: float
    on_target_fraction: float
    action_change_norm: float
    representation_drift: float
    mean_action_deviation: float
    success_delta: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible correlation cell."""
        return asdict(self)


@dataclass(frozen=True)
class CausalCorrelation:
    """Offline-to-environment comparison with declared disagreement rules."""

    cells: tuple[CausalCorrelationCell, ...]
    spearman_rho: float | None
    disagreements: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible correlation report."""
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "spearman_rho": self.spearman_rho,
            "disagreements": list(self.disagreements),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BenchmarkAcceptance:
    """Predeclared gate for promoting the causal-intervention claim."""

    passed: bool
    checks: Mapping[str, bool]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible acceptance report."""
        return {"passed": self.passed, "checks": dict(self.checks), "failures": list(self.failures)}


@dataclass(frozen=True)
class FailureAnalysis:
    """Per-condition failure bookkeeping over the benchmark episodes."""

    per_condition: Mapping[str, tuple[EpisodeOutcome, ...]]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible failure analysis."""
        return {
            "per_condition": {
                key: [outcome.to_dict() for outcome in outcomes] for key, outcomes in self.per_condition.items()
            },
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SimulationBenchmarkResult:
    """Complete typed result of one causal simulation benchmark run."""

    config: SimulationBenchmarkConfig
    environment_metadata: Mapping[str, object]
    outcomes: tuple[EpisodeOutcome, ...]
    summaries: tuple[ConditionSummary, ...]
    offline_scores: tuple[OfflineExplanationScore, ...]
    correlation: CausalCorrelation
    acceptance: BenchmarkAcceptance
    failure_analysis: FailureAnalysis
    claim_scope: str

    def to_dict(self) -> dict[str, object]:
        """Return a fully JSON-compatible benchmark artifact."""
        return {
            "claim_scope": self.claim_scope,
            "config": self.config.to_dict(),
            "environment_metadata": dict(self.environment_metadata),
            "summaries": [summary.to_dict() for summary in self.summaries],
            "episodes": [outcome.to_dict() for outcome in self.outcomes],
            "offline_scores": [score.to_dict() for score in self.offline_scores],
            "correlation": self.correlation.to_dict(),
            "acceptance": self.acceptance.to_dict(),
            "failure_analysis": self.failure_analysis.to_dict(),
        }


def _find_spec(name: str) -> machinery.ModuleSpec | None:
    """Resolve a module spec without executing the module (importlib indirection)."""

    return importlib.util.find_spec(name)


def _bootstrap_libero_config() -> None:
    """Pre-create LIBERO's user config so its import never prompts interactively.

    ``hf-libero`` 0.1.4 runs ``input()`` at import time when
    ``~/.libero/config.yaml`` is missing, which raises under captured stdin
    (pytest) and blocks headless lanes. The benchmark resolves the installed
    package paths and writes the same default config upstream would, so first
    use stays non-interactive. An existing config is preserved only while its
    recorded ``init_states`` path still exists; a stale config (for example
    from a previous temporary clone that has since been deleted) is refreshed.
    """

    config_path = Path.home() / ".libero" / "config.yaml"
    if config_path.exists() and _recorded_init_states_exist(config_path):
        return
    try:
        spec = _find_spec("libero.libero")
    except (ImportError, ModuleNotFoundError):
        spec = None
    if spec is None or spec.origin is None:
        return
    package_root = Path(spec.origin).resolve().parent
    entries = {
        "benchmark_root": package_root,
        "bddl_files": package_root / "bddl_files",
        "init_states": package_root / "init_files",
        "datasets": package_root.parent / "datasets",
        "assets": package_root / "assets",
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "".join(f"{key}: {value}\n" for key, value in entries.items()),
        encoding="utf-8",
    )


def _recorded_init_states_exist(config_path: Path) -> bool:
    """Return whether the recorded init_states path still points at files."""

    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("init_states:"):
            recorded = line.split(":", 1)[1].strip()
            return bool(recorded) and Path(recorded).is_dir()
    return False


def build_libero_benchmark_environment(
    config: SimulationBenchmarkConfig,
    *,
    api: LeRobotAPI | None = None,
) -> BenchmarkEnvironmentBundle:
    """Create the official LeRobot LIBERO vector environment and processors.

    The environment keeps upstream ownership: the ``libero`` env config is
    built through ``lerobot.envs.make_env_config``, the vector environment
    through ``LeRobotAPI.make_env`` with ``n_envs=1``, and the observation
    conversion through the env's own ``get_env_processors()`` plus
    ``lerobot.envs.utils.preprocess_observation``. LIBERO's user config is
    bootstrapped first so the upstream import never prompts interactively.
    """

    upstream_api = api if api is not None else load_lerobot_api()
    _bootstrap_libero_config()
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
        suite_map = cast(Mapping[str, Mapping[int, object]], envs)
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
        task_description = _call_task_description(probe_env)
        max_steps = (
            int(config.max_episode_steps) if config.max_episode_steps is not None else _call_max_steps(probe_env)
        )
    finally:
        close = getattr(probe_env, "close", None)
        if callable(close):
            close()
    return BenchmarkEnvironmentBundle(
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


def _call_task_description(env: object) -> str:
    """Resolve the natural-language task description from the vector env."""

    call = getattr(env, "call", None)
    if callable(call):
        for attribute in ("task_description", "task"):
            try:
                values = cast(Sequence[object], call(attribute))
                for value in values:
                    if isinstance(value, str) and value:
                        return value
            except (AttributeError, NotImplementedError, TypeError):
                continue
    raise ValueError("environment must expose a task_description attribute through call()")


def _call_max_steps(env: object) -> int:
    """Resolve the upstream max episode steps from the vector env."""

    call = getattr(env, "call", None)
    if callable(call):
        try:
            values = cast(Sequence[object], call("_max_episode_steps"))
        except (AttributeError, NotImplementedError, TypeError):
            pass
        else:
            for value in values:
                if isinstance(value, (int, np.integer)) and int(value) > 0:
                    return int(value)
    raise ValueError("environment must expose _max_episode_steps through call()")


def _policy_device(policy: nn.Module) -> torch.device:
    """Return the device of the policy's first parameter."""

    try:
        first = next(policy.parameters())
    except StopIteration:
        raise TypeError("policy must own at least one parameter") from None
    return first.device


def _noise_to_tensor(value: np.ndarray, *, device: torch.device) -> Tensor:
    """Convert the public NumPy noise boundary to the policy tensor type."""

    return Tensor(value).to(dtype=torch.float32, device=device)


def _official_select_action(
    adapter: SmolVLAPolicyAdapter,
    sample: Mapping[str, object],
    *,
    noise: np.ndarray,
) -> object:
    """Run the official preprocess -> select_action -> postprocess path without hooks."""

    policy = adapter.context.policy
    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a torch.nn.Module for the no-hook path")
    preprocessor = adapter.context.preprocessor
    postprocessor = adapter.context.postprocessor
    if not callable(preprocessor) or not callable(postprocessor):
        raise TypeError("context must expose callable preprocessor and postprocessor")
    prepared = cast(Mapping[str, object], preprocessor(sample))
    select = getattr(policy, "select_action", None)
    if not callable(select):
        raise TypeError("policy must expose LeRobot's select_action lifecycle")
    raw_action = select(prepared, noise=_noise_to_tensor(noise, device=_policy_device(policy)))
    return postprocessor(raw_action)


def _action_to_numpy(value: object) -> np.ndarray:
    """Extract the canonical batched action from a real or fixture post-processor result."""

    if isinstance(value, Mapping):
        for key in ("action", "actions"):
            if key in value:
                value = value[key]
                break
        else:
            raise TypeError("post-processor mapping must contain 'action' or 'actions'")
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
    elif isinstance(value, Tensor):
        array = value.detach().cpu().numpy().copy()
    else:
        array = np.asarray(value).copy()
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"expected a batched action, got shape {array.shape}")
    return array


def _extract_success(info: Mapping[str, object]) -> bool:
    """Resolve the per-episode success flag from a vector-env info mapping."""

    final_info = info.get("final_info")
    if isinstance(final_info, Mapping):
        entry = final_info.get(0)
        if isinstance(entry, Mapping) and entry.get("is_success") is not None:
            return bool(entry["is_success"])
    if isinstance(final_info, Sequence) and len(final_info) > 0:
        entry = final_info[0]
        if isinstance(entry, Mapping) and entry.get("is_success") is not None:
            return bool(entry["is_success"])
    is_success = info.get("is_success")
    if isinstance(is_success, (np.ndarray, list, tuple)) and len(is_success) > 0:
        return bool(is_success[0])
    if is_success is not None:
        return bool(is_success)
    return False


def _termination_flags(value: object, index: int = 0) -> bool:
    """Extract one batched termination flag from a vector-env step result."""

    if isinstance(value, (np.ndarray, list, tuple)):
        return bool(value[index])
    return bool(value)


def _random_expert_direction(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Return a seeded unit-norm direction in action-expert space."""

    vector = rng.normal(size=dim)
    return vector / float(np.linalg.norm(vector))


def _targeted_expert_direction(
    adapter: SmolVLAPolicyAdapter,
    *,
    action_axis: int,
) -> np.ndarray:
    """Return the expert direction inducing the declared action-axis change.

    The direction is the minimum-norm solution ``W[action_axis].T`` normalized,
    so the induced action change is pure along the declared axis — the most
    on-target intervention the offline explanation can name.
    """

    policy = adapter.context.policy
    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a torch.nn.Module to derive the targeted direction")
    action_out_proj = getattr(getattr(policy, "model", None), "action_out_proj", None)
    weight = getattr(action_out_proj, "weight", None)
    if not isinstance(weight, Tensor):
        raise TypeError("policy must expose model.action_out_proj.weight")
    matrix = weight.detach().cpu().numpy()
    if matrix.ndim != 2 or matrix.shape[1] != adapter.expert_dim:
        raise ValueError(f"action_out_proj weight shape {matrix.shape} does not match expert_dim={adapter.expert_dim}")
    if action_axis < 0 or action_axis >= adapter.action_dim:
        raise ValueError(f"action_axis {action_axis} outside the policy action dimension {adapter.action_dim}")
    unit = np.zeros(adapter.action_dim)
    unit[action_axis] = 1.0
    direction = matrix[: adapter.action_dim].T @ unit
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("targeted direction collapsed to zero; the action axis has no expert projection")
    return direction / norm


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
    """Roll out one episode under one condition on the official path.

    A fresh environment is created for the cell (so LIBERO starts from the
    same initial state as every other condition), reset with ``seed``, and
    closed after the episode. Every step goes through the official
    observation conversion, policy preprocessing, ``select_action``,
    postprocessing, and ``env.step``. ``record_samples`` collects the
    policy-ready samples of executed queries (used as offline probe inputs).
    """

    if condition not in VALID_CONDITIONS:
        raise ValueError(f"unknown benchmark condition {condition!r}")
    env = environment.env_factory()
    reset = cast(
        Callable[..., tuple[Mapping[str, object], Mapping[str, object]]],
        getattr(env, "reset", None),
    )
    step = cast(
        Callable[..., tuple[Mapping[str, object], object, object, object, Mapping[str, object]]],
        getattr(env, "step", None),
    )
    if not callable(reset) or not callable(step):
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise TypeError("benchmark environment must expose reset() and step()")
    try:
        adapter.reset()
        observation, _info = reset(seed=seed)
        samples: list[Mapping[str, object]] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        latencies: list[float] = []
        success = False
        terminated = False
        step_index = 0
        while step_index < environment.max_episode_steps:
            observed = cast(Mapping[str, object], observation)
            converted = environment.preprocess_observation(observed)
            with_task = dict(converted)
            with_task["task"] = [environment.task_description]
            sample = environment.env_preprocessor(with_task)
            is_query = step_index % adapter.chunk_size == 0
            if record_samples and is_query:
                samples.append(sample)
            started = time.perf_counter()
            if condition == "no_hook":
                raw_action = _official_select_action(adapter, sample, noise=noise)
            else:
                selection = adapter.select_action(
                    sample,
                    noise=noise,
                    intervention=SmolVLAIntervention(direction=direction, strength=strength),
                    episode_step=step_index,
                )
                raw_action = selection.action
            latencies.append(time.perf_counter() - started)
            action = _action_to_numpy(raw_action)
            actions.append(action)
            observation, reward, done, truncated, info = step(action)
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            terminated = _termination_flags(done) or _termination_flags(truncated)
            success = _extract_success(cast(Mapping[str, object], info))
            if terminated:
                break
            step_index += 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    del _info
    outcome = _build_outcome(
        seed=seed,
        condition=condition,
        strength=strength,
        success=success,
        rewards=rewards,
        actions=actions,
        latencies=latencies,
        terminated=terminated,
        max_steps=environment.max_episode_steps,
        reference_actions=reference_actions,
    )
    return outcome, tuple(samples)


def _build_outcome(
    *,
    seed: int,
    condition: BenchmarkCondition,
    strength: float,
    success: bool,
    rewards: Sequence[float],
    actions: Sequence[np.ndarray],
    latencies: Sequence[float],
    terminated: bool,
    max_steps: int,
    reference_actions: Sequence[np.ndarray] | None,
) -> EpisodeOutcome:
    """Assemble the episode metrics from the recorded steps."""

    if not rewards:
        raise ValueError("episode produced no steps")
    action_array = [np.asarray(action).reshape(-1) for action in actions]
    length = len(action_array)
    if length > max_steps:
        raise ValueError(f"episode exceeded the declared max steps {max_steps}")
    deviation = 0.0
    if reference_actions is not None:
        reference = [np.asarray(reference_action).reshape(-1) for reference_action in reference_actions]
        aligned = min(length, len(reference))
        if aligned > 0:
            deltas = [
                np.linalg.norm(actual - expected)
                for actual, expected in zip(action_array[:aligned], reference[:aligned], strict=True)
            ]
            deviation = float(np.mean(deltas))
    query_latencies = np.asarray(latencies, dtype=float)
    return EpisodeOutcome(
        seed=seed,
        condition=condition,
        strength=strength,
        success=success,
        sum_reward=float(np.sum(rewards)),
        max_reward=float(np.max(rewards)),
        length=length,
        terminated=terminated,
        mean_action_deviation=deviation,
        mean_query_latency_s=float(np.mean(query_latencies)),
        first_query_latency_s=float(query_latencies[0]),
        total_latency_s=float(np.sum(query_latencies)),
        n_queries=len(query_latencies),
        actions=tuple(action_array),
    )


def _summarize(
    outcomes: Sequence[EpisodeOutcome],
    condition: BenchmarkCondition,
    strength: float,
) -> ConditionSummary:
    """Aggregate one (condition, strength) cell into a summary with CIs."""

    if not outcomes:
        raise ValueError(f"no episodes for {condition} at strength {strength}")
    successes = [outcome.success for outcome in outcomes]
    returns = [outcome.sum_reward for outcome in outcomes]
    low, high = wilson_ci(successes)
    return_low, return_high = _normal_ci(returns)
    return ConditionSummary(
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


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
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
    denom = float(np.linalg.norm(dx) * np.linalg.norm(dy))
    if denom == 0.0:
        return None
    return float(np.sum(dx * dy) / denom)


def build_correlation(
    cells: Sequence[CausalCorrelationCell],
) -> CausalCorrelation:
    """Compare offline explanation scores with environment effects.

    Declared disagreement rules:

    - overstatement: the offline explanation reports an almost fully on-target
      direction but the environment success is essentially unchanged;
    - understatement: the offline explanation reports a mostly off-target
      direction but the environment effect on success is material;
    - reversal: a targeted intervention at the declared axis harmed success
      while the offline explanation predicted a controlled change.

    Spearman is reported only when at least three cells are compared and both
    axes have variance.
    """

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
    rho = _spearman(
        [cell.on_target_fraction for cell in cells],
        [cell.mean_action_deviation for cell in cells],
    )
    if rho is None:
        notes.append("Spearman correlation not computed (insufficient cells or no variance)")
    else:
        notes.append("Spearman correlation computed between offline on-target fraction and mean action deviation")
    if not cells:
        raise ValueError("correlation requires at least one cell")
    return CausalCorrelation(
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
    """Run the full causal benchmark for one policy against one environment.

    Episode order per seed is fixed: ``no_hook`` first (it produces the
    reference trajectory every other condition is compared against), then
    ``baseline``, then the non-zero intervention conditions at every declared
    strength. Every condition starts from the same initial state with the
    same seed and identical fixed noise — the environment factory creates a
    fresh vector environment per cell precisely so LIBERO's initial-state
    index cannot advance between conditions — and the only behavioral
    difference is the intervention.
    """

    if not isinstance(noise, np.ndarray):
        noise = np.full(
            (1, adapter.metadata.chunk_size, adapter.metadata.max_action_dim),
            config.noise_value,
        )
    rng = np.random.default_rng(config.intervention_seed)
    random_direction = _random_expert_direction(adapter.expert_dim, rng)
    targeted_direction = _targeted_expert_direction(adapter, action_axis=config.action_axis)
    direction_for: dict[BenchmarkCondition, np.ndarray] = {
        "random": random_direction,
        "targeted": targeted_direction,
    }

    outcomes: list[EpisodeOutcome] = []
    probe_samples: list[Mapping[str, object]] = []
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
                    direction=direction_for.get(condition, np.zeros(1)),
                    noise=noise,
                    reference_actions=reference,
                    record_samples=collect_probes,
                )
                if condition == "no_hook":
                    reference_by_seed[seed] = outcome.actions
                outcomes.append(outcome)
                if samples:
                    probe_samples.extend(samples)

    summaries = tuple(
        _summarize(
            [outcome for outcome in outcomes if outcome.condition == condition and outcome.strength == strength],
            condition,
            strength,
        )
        for condition in config.conditions
        for strength in ((0.0,) if condition in ("no_hook", "baseline") else config.strengths)
    )
    summary_by_cell = {(summary.condition, summary.strength): summary for summary in summaries}

    offline_scores: list[OfflineExplanationScore] = []
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
                    OfflineExplanationScore(
                        condition=cast(BenchmarkCondition, condition),
                        strength=strength,
                        on_target_fraction=measurement.on_target_fraction,
                        action_change_norm=measurement.action_change_norm,
                        representation_drift=measurement.representation_drift,
                        probe_queries=len(probe_samples),
                    )
                )

    cells: list[CausalCorrelationCell] = []
    no_hook_success_rate = summary_by_cell[("no_hook", 0.0)].success_rate
    for score in offline_scores:
        summary = summary_by_cell[(score.condition, score.strength)]
        cells.append(
            CausalCorrelationCell(
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
        correlation = CausalCorrelation(
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
        "baseline_actions_bit_exact": _baseline_is_bit_exact(outcomes),
        "baseline_success_equals_no_hook": _baseline_success_matches(outcomes),
        "intervention_changes_actions": _interventions_change_actions(outcomes),
        "all_episodes_within_max_steps": _episodes_within_budget(outcomes, environment.max_episode_steps),
    }
    failures = [key for key, passed in checks.items() if not passed]
    acceptance = BenchmarkAcceptance(passed=not failures, checks=checks, failures=tuple(failures))

    failure_analysis = FailureAnalysis(
        per_condition={
            condition: tuple(outcome for outcome in outcomes if outcome.condition == condition and not outcome.success)
            for condition in config.conditions
        },
        notes=(
            "unsuccessful episodes are grouped per condition; each row records the seed, "
            "strength, length, and action deviation against the no_hook reference",
        ),
    )
    return SimulationBenchmarkResult(
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


def _baseline_is_bit_exact(outcomes: Sequence[EpisodeOutcome]) -> bool:
    """Check that every baseline action matches the no_hook reference exactly."""

    no_hook = {outcome.seed: outcome for outcome in outcomes if outcome.condition == "no_hook"}
    baseline = [outcome for outcome in outcomes if outcome.condition == "baseline"]
    for outcome in baseline:
        reference = no_hook.get(outcome.seed)
        if reference is None:
            return False
        aligned = min(len(outcome.actions), len(reference.actions))
        if aligned == 0:
            return False
        for actual, expected in zip(outcome.actions[:aligned], reference.actions[:aligned], strict=True):
            if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                return False
    return True


def _baseline_success_matches(outcomes: Sequence[EpisodeOutcome]) -> bool:
    """Check that the baseline success flags equal the no_hook flags per seed."""

    no_hook = {outcome.seed: outcome for outcome in outcomes if outcome.condition == "no_hook"}
    for outcome in outcomes:
        if outcome.condition != "baseline":
            continue
        reference = no_hook.get(outcome.seed)
        if reference is None or reference.success != outcome.success:
            return False
    return True


def _interventions_change_actions(outcomes: Sequence[EpisodeOutcome]) -> bool:
    """Check that every non-zero intervention cell changed the actions."""

    threshold = 1e-9
    for outcome in outcomes:
        if outcome.condition in ("no_hook", "baseline"):
            continue
        if outcome.mean_action_deviation <= threshold:
            return False
    return True


def _episodes_within_budget(outcomes: Sequence[EpisodeOutcome], max_steps: int) -> bool:
    """Check that every episode stayed inside the declared step budget."""

    return all(outcome.length <= max_steps for outcome in outcomes)


__all__ = [
    "BenchmarkAcceptance",
    "BenchmarkCondition",
    "BenchmarkEnvironmentBundle",
    "CausalCorrelation",
    "CausalCorrelationCell",
    "ConditionSummary",
    "EpisodeOutcome",
    "FailureAnalysis",
    "OfflineExplanationScore",
    "SMOLVLA_BENCHMARK_ACTION_AXIS",
    "SMOLVLA_BENCHMARK_ENV_TYPE",
    "SMOLVLA_BENCHMARK_TASK",
    "SimulationBenchmarkConfig",
    "SimulationBenchmarkResult",
    "VALID_CONDITIONS",
    "build_correlation",
    "build_libero_benchmark_environment",
    "run_episode",
    "run_simulation_benchmark",
    "wilson_ci",
]
