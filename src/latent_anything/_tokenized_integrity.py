"""Private tokenizer checkpoint/schema binding helpers."""

from __future__ import annotations


def validate_tokenizer_binding(actual_version: str, bound_version: str) -> None:
    """Reject a VQVAE checkpoint mutation after model construction."""

    if actual_version != bound_version:
        raise ValueError("tokenizer checkpoint changed after TokenizedWorldModel construction")


def validate_codebook_version(requested: str | None, tokenizer_version: str) -> None:
    """Validate an optional sequence/checkpoint codebook identity."""

    if requested is not None and requested != tokenizer_version:
        raise ValueError(f"codebook_version {requested!r} does not match tokenizer {tokenizer_version!r}")


def validate_sequence_codebook_version(requested: str | None, bound_version: str) -> None:
    if requested is not None and requested != bound_version:
        raise ValueError("token sequence codebook_version does not match the frozen tokenizer")
