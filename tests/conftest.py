"""Shared fixtures for latent-anything tests."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep optional model-download tests opt-in, as documented by their marker."""
    del config
    if os.environ.get("LATENT_ANYTHING_RUN_NETWORK") == "1":
        return
    skip_network = pytest.mark.skip(reason="network integration tests require LATENT_ANYTHING_RUN_NETWORK=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
