from __future__ import annotations

from pathlib import Path

import numpy as np

from latent_anything import (
    FileSystemRunRecorder,
    MPPIPlanner,
    MPPIPlannerSpec,
    ObjectSpec,
    build_from_config,
    build_mppi_planner_from_config,
)
from latent_anything.mppi import MPPIConfig
from latent_anything.registry import KIND_RUNTIME


def _spec() -> MPPIPlannerSpec:
    return MPPIPlannerSpec(
        horizon=2,
        action_dim=1,
        lower_bounds=(-1.0,),
        upper_bounds=(1.0,),
        population_size=12,
        iterations=2,
        noise_std=(0.3,),
        temperature=0.4,
        seed=69,
    )


def test_mppi_config_builder_and_runtime_registry_construct_planner() -> None:
    spec = _spec()

    assert isinstance(build_mppi_planner_from_config(spec), MPPIPlanner)
    from_registry = build_from_config(
        ObjectSpec(
            kind=KIND_RUNTIME,
            name="mppi_planner",
            params={"config": MPPIConfig.model_validate(spec.model_dump())},
        )
    )
    assert isinstance(from_registry, MPPIPlanner)


def test_run_recorder_persists_mppi_plan_artifact_and_metrics(tmp_path: Path) -> None:
    planner = build_mppi_planner_from_config(_spec())
    result = planner.plan(lambda candidates: -np.sum(np.square(candidates), axis=(1, 2)))
    recorder = FileSystemRunRecorder(tmp_path)
    started = recorder.start("mppi", config=_spec().model_dump())
    completed = recorder.complete_mppi_plan(started.run_id, result)

    assert completed.status == "completed"
    assert completed.metrics["mppi_iterations"] == 2.0
    assert completed.metrics["mppi_samples"] == 24.0
    assert completed.artifacts[0].name == "mppi_plan.json"
    assert recorder.read_artifact(completed.artifacts[0])
