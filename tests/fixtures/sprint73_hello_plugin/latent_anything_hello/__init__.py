"""Minimal separately installed Sprint 73 adapter fixture."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HelloAdapter:
    """A tiny callable adapter used only for external-install proof."""

    __latent_anything_plugin_api_version__ = "1"

    prefix: str = "hello"

    def __call__(self, value: str = "world") -> str:
        """Return a deterministic greeting for one input value."""

        return f"{self.prefix}:{value}"


__all__ = ["HelloAdapter"]
