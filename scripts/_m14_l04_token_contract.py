"""Injected tokenizer seam for the future M14 L04 real-run preflight."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from scripts._m14_l04_contract_common import ContractValidationError


def validate_target_tokens(
    tokenize: Callable[[str], object], target_texts: Sequence[str], *, expected_token_count: int = 1
) -> None:
    """Validate target token cardinality through an injected tokenizer seam.

    The callback is intentionally the only tokenizer dependency. This helper
    records no token IDs and performs no model/tokenizer resolution; it is for
    the future real-run preflight and deterministic fake-only unit tests.
    """
    for target_text in target_texts:
        try:
            encoded = tokenize(target_text)
        except Exception as exc:  # noqa: BLE001 - convert arbitrary tokenizer failures to contract errors
            raise ContractValidationError(f"tokenizer failed for target {target_text!r}") from exc
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
        if isinstance(encoded, (str, bytes)) or not isinstance(encoded, Sequence):
            raise ContractValidationError(f"tokenizer result for {target_text!r} has no token sequence")
        if len(encoded) != expected_token_count:
            raise ContractValidationError(
                f"target {target_text!r} resolves to {len(encoded)} tokens; expected {expected_token_count}"
            )
