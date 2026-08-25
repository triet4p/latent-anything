# External Plugin Author Guide

Sprint 73 supports external Python distributions through standard
`importlib.metadata` entry points. Discovery is deliberately explicit and
small: listing declarations reads metadata only; loading a plugin imports its
callable and registers it in a caller-owned `Registry`.

## Supported groups

Declare one of these canonical groups in your distribution's `pyproject.toml`:

| Group | Capability | Registry kind |
| --- | --- | --- |
| `latent_anything.adapter` | Model/representation adapter | `adapter` |
| `latent_anything.analysis` | Representation analysis | `analysis` |
| `latent_anything.intervention` | Representation intervention/transformation | `intervention` |
| `latent_anything.transition` | Latent-state transition | existing `runtime` |
| `latent_anything.planner` | Action-sequence planner | existing `runtime` |

`intervention` is canonical. Do not create a second `transformation` group.
Transition and planner groups are capability names; their current built-ins
remain registered under `runtime` until concrete evidence justifies a new
registry kind.

## Minimal declaration

```toml
[project.entry-points."latent_anything.adapter"]
acme-normalizer = "acme_latent:AcmeNormalizer"
```

The target must be a callable class or factory. It must declare the supported
plugin contract version on that callable:

```python
class AcmeNormalizer:
    __latent_anything_plugin_api_version__ = "1"

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
```

The target is not imported while users list plugins. A missing or unsupported
version is reported as an isolated load failure; it does not prevent healthy
plugins from loading.

## Build from config

```python
from latent_anything.config import ObjectSpec, build_from_config
from latent_anything.plugin_discovery import load_entry_points
from latent_anything.registry import Registry

registry = Registry("external")
report = load_entry_points(registry=registry)
if report.failures:
    for issue in report.failures:
        print(issue.reason)

normalizer = build_from_config(
    ObjectSpec(
        kind="adapter",
        name="acme-normalizer",
        params={"scale": 0.5},
    ),
    registry=registry,
)
```

`Registry` rejects duplicate names. Sprint 73 discovery applies an explicit
non-overriding policy before registration: an existing name wins; among
external declarations, the first declaration in canonical group/name order,
then distribution name/version order, then the metadata-only declaration key
derived from entry-point group/name/value/module/attribute fields wins.
Skipped duplicates are reported and their targets are never loaded.

## Provenance and testing

Successful entries carry `source`, `entry_point_group`,
`entry_point_value`, `distribution`, `version`, and
`plugin_api_version` metadata. Persist those fields with your own run/config
artifact when reproducibility matters. Test the installed distribution, not a
checkout import: install a wheel or local package into a temporary environment,
verify metadata listing does not import your module, then explicitly load and
construct through `ObjectSpec`.

The repository's separately installed hello-world proof is
`tests/test_plugin_installation.py`; copy its temporary-target/subprocess
pattern. `tests/plugin_harness.py` contains a small assertion helper for
provenance checks.

## Security and scope

Entry-point targets are untrusted third-party Python code. Listing is safe from
plugin imports, but explicit loading executes arbitrary import-time and factory
code. Load only distributions you trust, use a caller-owned registry when
isolating plugins, and inspect `DiscoveryReport.issues`. The discovery layer is
not a dependency-injection container, workflow language, sandbox, or remote
execution protocol. Plugin APIs remain the concrete callable/config seams
already proven by built-ins.
