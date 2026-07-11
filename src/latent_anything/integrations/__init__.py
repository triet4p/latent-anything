"""Optional integration boundaries; base imports never load their backends."""

from latent_anything.integrations._optional import require_optional as require_optional

__all__ = ["require_optional"]
