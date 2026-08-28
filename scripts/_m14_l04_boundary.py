"""Lazy concrete integration boundary for future L04 real handlers."""

from __future__ import annotations

from typing import Any

INTEGRATION_FACTORY = "latent_anything.integrations.transformer_lm.TransformerLMIntegration"


def transformer_integration_type() -> type[Any]:
    """Resolve the concrete integration class without constructing a model."""
    from latent_anything.integrations.transformer_lm import TransformerLMIntegration

    return TransformerLMIntegration
