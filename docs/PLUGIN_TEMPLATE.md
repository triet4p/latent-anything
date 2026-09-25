# Plugin Template

Copy this minimal layout and replace the provider/name with your own
lowercase, provider-qualified entry name:

The dependency range in this template is intended for use after framework
version `1.0.0` is published. The repository's current `1.0.0` metadata is an
unpublished candidate; it cannot be installed from a package index.

```text
my-latent-plugin/
├── pyproject.toml
├── src/
│   └── my_latent_plugin/
│       └── __init__.py
└── tests/
    └── test_plugin.py
```

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "my-latent-plugin"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["latent-anything>=1.0.0,<2.0.0"]

[project.entry-points."latent_anything.analysis"]
my-provider-feature = "my_latent_plugin:MyFeature"

[tool.setuptools.packages.find]
where = ["src"]
```

## `src/my_latent_plugin/__init__.py`

```python
from __future__ import annotations


class MyFeature:
    __latent_anything_plugin_api_version__ = "1"

    def __init__(self, width: int = 2) -> None:
        self.width = width
```

## Contract test

Install the package into a temporary target and test its declared entry point
through the framework. The test should assert listing, no import during
listing, explicit loading, config construction, execution, and provenance.
Use `tests/test_plugin_installation.py` and `tests/plugin_harness.py` in this
repository as the reference harness. Do not modify latent-anything source at
plugin installation time.
