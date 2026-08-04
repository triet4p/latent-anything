# Task Summary: Sprint 56 Task 01 — LeRobot compatibility audit

**Sprint:** Sprint 56
**Task:** Audit the current stable LeRobot release and supported seams

## Summary of Work

Audited the upstream LeRobot release available at sprint activation on 2026-08-04. The current stable PyPI release is `0.6.1` (uploaded 2026-08-03; the lock records the wheel hash `sha256:1894516040c65f80a45bd9741f8174aae90ed5d93da0627ab4f1a85fd8d75e90`). The supported bridge window is therefore the `0.6.x` line, with Python `>=3.12`, Torch `>=2.7,<2.12`, NumPy `>=2.0,<2.3`, and the upstream package's own `torchvision` and OpenCV constraints left to its resolver metadata.

The bridge will consume these upstream seams without reimplementing them:

* policy construction and pre/post processors from `lerobot.policies`;
* `PolicyProcessorPipeline` and processor steps from `lerobot.processor`;
* `LeRobotDataset` and its v3 metadata/data/video layout from `lerobot.datasets.lerobot_dataset`;
* environment construction and evaluation through `lerobot.envs` and the `lerobot-eval` entry point;
* third-party policy registration through LeRobot's `PreTrainedConfig` registry and convention-based policy/processor factories.

## Evidence Sources

* [LeRobot PyPI release history](https://pypi.org/project/lerobot/) — current release and Python classifiers.
* [LeRobot current pyproject](https://github.com/huggingface/lerobot/blob/main/pyproject.toml) — dependency constraints, extras, and CLI entry points.
* [Policy factory](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/factory.py) — policy and processor factory seams.
* [LeRobotDataset](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py) — dataset construction and v3 data contract.
* [Processor package](https://github.com/huggingface/lerobot/blob/main/src/lerobot/processor/__init__.py) — processor pipeline surface.
* [Evaluation CLI](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py) — evaluation boundary.
* [Adding a Policy](https://huggingface.co/docs/lerobot/bring_your_own_policies) — supported third-party policy/plugin path.

## Testing

* **Test File:** `tests/test_lerobot_integration.py` (added in Task 04)
* **Status:** Pending until the compatibility boundary is implemented

## Additional Notes

LeRobot is still alpha software and its `main` branch has moved beyond the stable `0.6.0` tag. The bridge must not silently follow `main`; an upstream upgrade requires the checklist and compatibility lane defined later in this sprint.
