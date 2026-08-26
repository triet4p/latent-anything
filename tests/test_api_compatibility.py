"""Runtime enforcement for the reviewed beta compatibility ledger."""

from __future__ import annotations

import json
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import latent_anything
from latent_anything import ObjectSpec, build_from_config
from latent_anything._transition_deterministic import DeterministicLatentTransition
from latent_anything._transition_stochastic import StochasticGaussianLatentTransition
from latent_anything._transition_types import (
    GaussianPrediction,
    StochasticOneStepMetrics,
    StochasticRolloutMetrics,
)
from latent_anything.cem import CEMIteration, CEMPlanResult
from latent_anything.cli import _parser, main  # pyright: ignore[reportPrivateUsage]
from latent_anything.latent_space import LatentSpace
from latent_anything.manipulation_pipeline import InterventionPipeline, ManipulationPipeline
from latent_anything.methods.b_protocols import BMethod, Intervention
from latent_anything.methods.protocols import AnalysisMethod, Method
from latent_anything.mppi import MPPIConfig, MPPIIteration, MPPIPlanResult
from latent_anything.pipeline_models import RolloutResult
from latent_anything.runtime.profiling import RuntimeProfile
from latent_anything.trajectory import Trajectory

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "API_COMPATIBILITY.md"


def test_canonical_symbols_are_exact_beta_aliases() -> None:
    assert AnalysisMethod is Method
    assert Intervention is BMethod
    assert InterventionPipeline is ManipulationPipeline
    assert latent_anything.AnalysisMethod is latent_anything.Method
    assert latent_anything.Intervention is Intervention


@pytest.mark.parametrize("legacy_kind, canonical_kind", [("method_a", "analysis"), ("method_b", "intervention")])
def test_legacy_registry_config_warns_once_and_preserves_factory(legacy_kind: str, canonical_kind: str) -> None:
    name = "pca" if legacy_kind == "method_a" else "lerp"
    params = {"n_components": 2} if name == "pca" else {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        legacy = build_from_config(ObjectSpec(kind=legacy_kind, name=name, params=params))
    canonical = build_from_config(ObjectSpec(kind=canonical_kind, name=name, params=params))
    assert len(captured) == 1
    assert captured[0].category is DeprecationWarning
    assert canonical_kind in str(captured[0].message)
    assert type(legacy) is type(canonical)


@pytest.mark.parametrize(
    "legacy, canonical", [("list-capture-points", "capture-points"), ("replay-run-config", "replay-run")]
)
def test_cli_aliases_share_parser_and_capture_output(legacy: str, canonical: str, tmp_path: Path) -> None:
    parser = _parser()
    subparsers = cast(Any, next(action for action in parser._actions if type(action).__name__ == "_SubParsersAction"))
    assert subparsers.choices[legacy] is subparsers.choices[canonical]
    args = ["--policy", "act"] if canonical == "capture-points" else ["missing", "--record-root", str(tmp_path)]
    if canonical == "capture-points":
        outputs = []
        for command in (canonical, legacy):
            stream = StringIO()
            with redirect_stdout(stream):
                assert main([command, *args]) == 0
            outputs.append(json.loads(stream.getvalue()))
        assert outputs[0] == outputs[1]
    else:
        failures = []
        for command in (canonical, legacy):
            with pytest.raises(FileNotFoundError) as caught:
                main([command, *args])
            failures.append((type(caught.value), str(caught.value)))
        assert failures[0] == failures[1]


def test_mppi_field_aliases_and_result_property_aliases_preserve_read_only_behavior() -> None:
    configs = [
        cast(Any, MPPIConfig)(horizon=1, action_dim=1, lower_bounds=(-1,), upper_bounds=(1,), **{key: 2.0})
        for key in ("lambda", "lambda_")
    ]
    assert [config.temperature for config in configs] == [2.0, 2.0]
    profile = RuntimeProfile(events=())
    cem_iteration = CEMIteration(0, 2, 1, 1.0, 0.1, 1.2, 1.1, np.array([0.2]), np.array([0.3]))
    cem = CEMPlanResult(np.zeros((1, 1)), 1.0, (cem_iteration,), (1.0,), profile, 1)
    mppi_iteration = MPPIIteration(0, 2, 1.0, 0.1, 1.2, 1.1, 1.0, 0.2, np.array([0.2]))
    mppi = MPPIPlanResult(np.zeros((1, 1)), 1.0, (mppi_iteration,), (1.0,), profile, 1, 2)
    assert cem.selected_actions is cem.actions
    assert mppi.selected_actions is mppi.actions
    with pytest.raises(ValueError):
        cem.selected_actions[0, 0] = 1.0

    trajectory = Trajectory(np.zeros((2, 1)))
    rollout = RolloutResult(np.zeros(1), np.zeros((1, 1)), trajectory, LatentSpace(1))
    assert rollout.states is rollout.trajectory


def test_transition_aliases_preserve_prediction_and_metric_semantics() -> None:
    states = np.arange(8.0).reshape(4, 2)
    actions = np.arange(4.0).reshape(4, 1)
    targets = states + 0.5
    space = LatentSpace(2)
    deterministic = DeterministicLatentTransition(space, 1).fit(states, actions, targets)
    np.testing.assert_array_equal(
        deterministic.predict(states[0], actions[0]), deterministic.step(states[0], actions[0])
    )
    stochastic = StochasticGaussianLatentTransition(space, 1).fit(states, actions, targets)
    prediction = stochastic.predict(states[0], actions[0])
    np.testing.assert_array_equal(prediction.mean, stochastic.step(states[0], actions[0]))
    np.testing.assert_array_equal(prediction.std, prediction.scale)
    one_step = StochasticOneStepMetrics(1.5, 0.9, 0.2, 0.1, 0.3, 4, 0.01)
    assert one_step.nll == one_step.negative_log_likelihood
    rollout_metrics = StochasticRolloutMetrics((1.0,), (0.8,), (0.2,), (0.3,), 1.0, 0.8, 0.2, 0.3, 0.01, True)
    assert rollout_metrics.nll_by_horizon == rollout_metrics.negative_log_likelihood_by_horizon
    assert rollout_metrics.errors_by_horizon == rollout_metrics.mean_error_by_horizon
    prediction_copy = GaussianPrediction(np.array([1.0]), np.array([0.2]))
    with pytest.raises(ValueError):
        prediction_copy.std[0] = 0.0


def test_ledger_covers_every_snapshot_beta_alias_family() -> None:
    snapshot = json.loads((ROOT / "artifacts" / "api_freeze_snapshot_0.1.0b1.json").read_text(encoding="utf-8"))
    ledger_text = LEDGER.read_text(encoding="utf-8")
    aliases = snapshot["sections"]["B_beta_compatibility"]
    for group in (
        "symbol_aliases",
        "registry_kind_aliases",
        "cli_aliases",
        "config_aliases",
        "result_property_aliases",
        "transition_aliases",
    ):
        for row in aliases[group]:
            assert str(row["legacy"]) in ledger_text
            assert str(row["canonical"]) in ledger_text
    assert "result-envelope-v0" in ledger_text
    assert "Windows artifact paths" in ledger_text
    assert "`0.2.0` (never released)" in ledger_text
    assert "implemented Unreleased / Sprint78.29" in ledger_text
    assert "deprecated current Unreleased / Sprint31" in ledger_text
    assert "since 0.2.0" not in ledger_text.lower()
