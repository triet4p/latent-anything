"""Opt-in structural smoke for the public Hugging Face I-JEPA checkpoint."""

from __future__ import annotations

import pytest

CHECKPOINT_ID = "facebook/ijepa_vith14_1k"
CHECKPOINT_REVISION = "be440b1cac639542ae553e71a9c7afd925ab5fac"


@pytest.mark.network
@pytest.mark.large_download
def test_public_ijepa_checkpoint_is_decoder_free() -> None:
    """Load the pinned public checkpoint only in the explicit network lane."""

    transformers = pytest.importorskip("transformers")
    model = transformers.IJepaModel.from_pretrained(CHECKPOINT_ID, revision=CHECKPOINT_REVISION)

    assert model.config.model_type == "ijepa"
    assert not hasattr(model, "decoder")
    assert model.config.hidden_size > 0
