"""Offline compatibility and import-isolation tests for the LeRobot boundary."""

from __future__ import annotations

import subprocess
import sys

import pytest

from latent_anything.integrations.lerobot import (
    SUPPORTED_LEROBOT_SPEC,
    LeRobotEvaluationResult,
    LeRobotPolicyContext,
    check_lerobot_compatibility,
)


def test_supported_lerobot_runtime_report_is_explicit() -> None:
    report = check_lerobot_compatibility(
        lerobot_version="0.6.0",
        torch_version="2.7.1",
        numpy_version="2.2.6",
        python_version=(3, 12),
    )

    assert report.supported
    assert report.diagnostic == "LeRobot compatibility check passed."
    assert report.to_dict()["lerobot_version"] == "0.6.0"
    assert SUPPORTED_LEROBOT_SPEC == ">=0.6.0,<0.7.0"


@pytest.mark.parametrize(
    ("lerobot_version", "torch_version", "numpy_version", "expected"),
    [
        ("0.5.1", "2.7.1", "2.2.6", "LeRobot"),
        ("0.7.0", "2.7.1", "2.2.6", "LeRobot"),
        ("0.6.0", "2.6.0", "2.2.6", "Torch"),
        ("0.6.0", "2.7.1", "2.3.0", "NumPy"),
    ],
)
def test_unsupported_lerobot_runtime_reports_conflict(
    lerobot_version: str,
    torch_version: str,
    numpy_version: str,
    expected: str,
) -> None:
    report = check_lerobot_compatibility(
        lerobot_version=lerobot_version,
        torch_version=torch_version,
        numpy_version=numpy_version,
        python_version=(3, 12),
    )

    assert not report.supported
    assert expected in report.diagnostic
    assert "uv sync --extra lerobot" in report.diagnostic


def test_base_import_does_not_load_lerobot_modules() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import latent_anything; import latent_anything.integrations.lerobot; "
            "assert not any(name == 'lerobot' or name.startswith('lerobot.') for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_extra_install_loads_supported_upstream_seams_on_cpu() -> None:
    pytest.importorskip("lerobot")
    from latent_anything.integrations.lerobot import load_lerobot_api

    api = load_lerobot_api()

    assert api.dataset_type.__name__ == "LeRobotDataset"
    assert api.policy_processor_pipeline_type.__name__ in {"PolicyProcessorPipeline", "DataProcessorPipeline"}
    assert callable(api.make_policy)
    assert callable(api.make_pre_post_processors)
    assert callable(api.make_env)
    assert callable(api.evaluation_main)
    assert callable(api.register_third_party_plugins)


def test_policy_context_keeps_raw_upstream_objects() -> None:
    policy = object()
    preprocessor = object()
    postprocessor = object()
    dataset = object()
    environment = object()
    context = LeRobotPolicyContext(
        policy_name="act",
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        environment=environment,
    )

    assert context.policy is policy
    assert context.preprocessor is preprocessor
    assert context.postprocessor is postprocessor
    assert context.dataset is dataset
    assert context.environment is environment


def test_evaluation_result_validates_and_serializes_bridge_owned_fields() -> None:
    result = LeRobotEvaluationResult(
        episodes=4,
        success_rate=0.75,
        metrics={"mean_return": 1.5},
        metadata={"revision": "fixture"},
    )

    assert result.to_dict() == {
        "episodes": 4,
        "success_rate": 0.75,
        "metrics": {"mean_return": 1.5},
        "metadata": {"revision": "fixture"},
    }
    with pytest.raises(ValueError, match="success_rate"):
        LeRobotEvaluationResult(episodes=1, success_rate=1.1, metrics={})
