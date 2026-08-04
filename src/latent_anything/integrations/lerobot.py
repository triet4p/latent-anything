"""Lazy LeRobot boundary and bridge-owned integration result types.

Importing this module is safe in a base installation. LeRobot modules are
loaded only by :func:`load_lerobot` or :func:`load_lerobot_api` after the
supported-version and runtime compatibility checks pass.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from importlib import import_module, metadata
from types import ModuleType
from typing import Any

from latent_anything.integrations import require_optional

SUPPORTED_LEROBOT_SPEC = ">=0.6.0,<0.7.0"
SUPPORTED_LEROBOT_EXTRA = "lerobot[dataset,evaluation]"
SUPPORTED_LEROBOT_VERSION = "0.6.x"


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    """Parse the release portion of a conventional three-part version."""

    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _installed_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing it."""

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class LeRobotCompatibilityReport:
    """Diagnostic snapshot for the supported LeRobot runtime boundary."""

    supported: bool
    lerobot_version: str | None
    torch_version: str | None
    numpy_version: str | None
    python_version: str
    issues: tuple[str, ...] = ()

    @property
    def diagnostic(self) -> str:
        """Return an actionable error message for an unsupported runtime."""

        if self.supported:
            return "LeRobot compatibility check passed."
        details = "; ".join(self.issues) if self.issues else "unknown compatibility failure"
        return (
            f"LeRobot compatibility check failed for supported window {SUPPORTED_LEROBOT_SPEC}: "
            f"{details}. Recreate the environment with `uv sync --extra lerobot`."
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic snapshot."""

        return asdict(self)


def check_lerobot_compatibility(
    *,
    lerobot_version: str | None = None,
    torch_version: str | None = None,
    numpy_version: str | None = None,
    python_version: tuple[int, int] | None = None,
) -> LeRobotCompatibilityReport:
    """Check the version and runtime constraints required by this bridge.

    The function uses distribution metadata rather than importing LeRobot,
    Torch, or NumPy. Explicit versions are accepted so resolver and
    unsupported-version tests can run without installing alternate runtimes.
    """

    resolved_lerobot = lerobot_version if lerobot_version is not None else _installed_version("lerobot")
    resolved_torch = torch_version if torch_version is not None else _installed_version("torch")
    resolved_numpy = numpy_version if numpy_version is not None else _installed_version("numpy")
    resolved_python = python_version if python_version is not None else (sys.version_info.major, sys.version_info.minor)
    issues: list[str] = []

    if resolved_lerobot is None:
        issues.append("LeRobot is not installed")
    else:
        version = _version_tuple(resolved_lerobot)
        if version is None or version < (0, 6, 0) or version >= (0, 7, 0):
            issues.append(f"LeRobot {resolved_lerobot!r} is outside the supported {SUPPORTED_LEROBOT_VERSION} window")

    if resolved_python < (3, 12):
        issues.append(f"Python {resolved_python[0]}.{resolved_python[1]} is below LeRobot's 3.12 floor")
    if resolved_torch is None:
        issues.append("Torch is not installed")
    else:
        version = _version_tuple(resolved_torch)
        if version is None or version < (2, 7, 0) or version >= (2, 12, 0):
            issues.append(f"Torch {resolved_torch!r} must satisfy >=2.7,<2.12 for LeRobot 0.6.x")
    if resolved_numpy is None:
        issues.append("NumPy is not installed")
    else:
        version = _version_tuple(resolved_numpy)
        if version is None or version < (2, 0, 0) or version >= (2, 3, 0):
            issues.append(f"NumPy {resolved_numpy!r} must satisfy >=2.0,<2.3 for LeRobot 0.6.x")

    return LeRobotCompatibilityReport(
        supported=not issues,
        lerobot_version=resolved_lerobot,
        torch_version=resolved_torch,
        numpy_version=resolved_numpy,
        python_version=f"{resolved_python[0]}.{resolved_python[1]}",
        issues=tuple(issues),
    )


def load_lerobot() -> ModuleType:
    """Load the LeRobot package after checking its supported runtime window."""

    report = check_lerobot_compatibility()
    if report.lerobot_version is None:
        return require_optional("lerobot", extra="lerobot")
    if not report.supported:
        raise ImportError(report.diagnostic)
    return require_optional("lerobot", extra="lerobot")


@dataclass(frozen=True)
class LeRobotAPI:
    """Raw upstream factories used by the bridge, without wrapper classes."""

    make_policy: Callable[..., object]
    make_pre_post_processors: Callable[..., object]
    dataset_type: type[object]
    policy_processor_pipeline_type: type[object]
    make_env: Callable[..., object]
    evaluation_main: Callable[..., object]
    register_third_party_plugins: Callable[[], object]


def _required_symbol(module: ModuleType, name: str) -> Any:
    """Get one supported upstream symbol with a focused upgrade diagnostic."""

    try:
        return getattr(module, name)
    except AttributeError as error:
        raise ImportError(
            f"LeRobot {SUPPORTED_LEROBOT_VERSION} no longer exposes {module.__name__}.{name}. "
            "Review docs/LEROBOT_INTEGRATION.md before upgrading the supported window."
        ) from error


def load_lerobot_api() -> LeRobotAPI:
    """Load the policy, processor, dataset, environment, and eval seams."""

    load_lerobot()
    policies = import_module("lerobot.policies")
    processor = import_module("lerobot.processor")
    dataset = import_module("lerobot.datasets.lerobot_dataset")
    environments = import_module("lerobot.envs")
    evaluation = import_module("lerobot.scripts.lerobot_eval")
    import_utils = import_module("lerobot.utils.import_utils")
    return LeRobotAPI(
        make_policy=_required_symbol(policies, "make_policy"),
        make_pre_post_processors=_required_symbol(policies, "make_pre_post_processors"),
        dataset_type=_required_symbol(dataset, "LeRobotDataset"),
        policy_processor_pipeline_type=_required_symbol(processor, "PolicyProcessorPipeline"),
        make_env=_required_symbol(environments, "make_env"),
        evaluation_main=_required_symbol(evaluation, "main"),
        register_third_party_plugins=_required_symbol(import_utils, "register_third_party_plugins"),
    )


@dataclass(frozen=True)
class LeRobotPolicyContext:
    """Bridge metadata around raw LeRobot policy and runtime objects.

    The policy, processors, dataset, and environment fields deliberately keep
    their upstream objects unchanged. Future capture and dataset tasks can
    attach framework metadata without cloning LeRobot's policy or data model.
    """

    policy_name: str
    policy: object
    preprocessor: object
    postprocessor: object
    dataset: object | None = None
    environment: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LeRobotEvaluationResult:
    """Bridge-owned summary of metrics returned by a LeRobot evaluation run."""

    episodes: int
    success_rate: float | None
    metrics: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.episodes < 0:
            raise ValueError("episodes must be non-negative")
        if self.success_rate is not None and not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        """Return a serializable result snapshot without changing raw upstream objects."""

        return asdict(self)


__all__ = [
    "LeRobotAPI",
    "LeRobotCompatibilityReport",
    "LeRobotEvaluationResult",
    "LeRobotPolicyContext",
    "SUPPORTED_LEROBOT_SPEC",
    "SUPPORTED_LEROBOT_VERSION",
    "SUPPORTED_LEROBOT_EXTRA",
    "check_lerobot_compatibility",
    "load_lerobot",
    "load_lerobot_api",
]
