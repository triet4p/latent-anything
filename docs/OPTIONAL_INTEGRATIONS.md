# Optional Integrations

The base package never imports Diffusers, Transformers, gsplat, or LeRobot.
Install one boundary deliberately: `uv sync --extra diffusers`,
`uv sync --extra transformers`, `uv sync --extra 3d`, or `uv sync --extra lerobot`.

Unit tests are CPU-only and use tiny local fixtures. Tests that require a GPU,
network acquisition, or large checkpoints must be explicitly marked `gpu`,
`network`, or `large_download`; cached model paths are supplied by CI and tests
must never download a model implicitly.

Integration upgrades require a bounded version range, a lower-bound import
smoke test, and a pinned model/backend revision in the consuming sprint.
