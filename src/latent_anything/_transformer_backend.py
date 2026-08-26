"""Private lazy backend loading for the transformer integration."""

from __future__ import annotations

from typing import Any

from latent_anything.integrations import require_optional


def load_backend(
    model_id: str,
    revision: str,
    *,
    device: str,
    torch_dtype: Any,
) -> tuple[Any, Any, Any]:
    transformers = require_optional("transformers", extra="transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
    )
    model = model.to(device)
    model.eval()
    config = model.config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, config


def tokenize(
    tokenizer: Any,
    prompt: str | tuple[str, ...],
    *,
    max_length: int,
    return_tensors: str,
) -> dict[str, Any]:
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors,
    )
    return dict(encoded)
