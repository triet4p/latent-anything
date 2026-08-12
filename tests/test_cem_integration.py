from __future__ import annotations

from pathlib import Path

import numpy as np

from latent_anything import (
    CEMPlanner,
    CEMPlannerSpec,
    FileSystemRunRecorder,
    ObjectSpec,
    build_cem_planner_from_config,
    build_from_config,
)
from latent_anything.cem import CEMConfig
from latent_anything.registry import KIND_RUNTIME


def _spec() -> CEMPlannerSpec:
    return CEMPlannerSpec(
        horizon=2,
        action_dim=1,
        lower_bounds=(-1.0,),
        upper_bounds=(1.0,),
        population_size=12,
        iterations=2,
        seed=68,
    )


def test_cem_config_builder_and_runtime_registry_construct_planner() -> None:
    spec = _spec()
    assert isinstance(build_cem_planner_from_config(spec), CEMPlanner)
    from_registry = build_from_config(
        ObjectSpec(
            kind=KIND_RUNTIME,
            name="cem_planner",
            params={"config": CEMConfig.model_validate(spec.model_dump())},
        )
    )
    assert isinstance(from_registry, CEMPlanner)


def test_run_recorder_persists_cem_plan_artifact_and_metrics(tmp_path: Path) -> None:
    planner = build_cem_planner_from_config(_spec())
    result = planner.plan(lambda candidates: -np.sum(np.square(candidates), axis=(1, 2)))
    recorder = FileSystemRunRecorder(tmp_path)
    started = recorder.start("cem", config=_spec().model_dump())
    completed = recorder.complete_cem_plan(started.run_id, result)

    assert completed.status == "completed"
    assert completed.metrics["cem_iterations"] == 2.0
    assert completed.artifacts[0].name == "cem_plan.json"
    assert recorder.read_artifact(completed.artifacts[0])
