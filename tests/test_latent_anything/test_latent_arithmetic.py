"""Tests for latent arithmetic with coordinate-system compatibility checks.

Covers ``LatentValue`` arithmetic (add/subtract/add_scaled/scale and the
``+``/``-`` operators), the canonical coordinate identity, rejection of
arithmetic across unrelated coordinate systems (geometry, shape, identity,
revision), and operation/provenance metadata on the immutable outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything import LatentSpace, LatentValue
from latent_anything.latent_value import assert_arithmetic_compatible, coordinate_identity


def _space(dim: int = 4, model: str = "modelA", revision: str = "v1") -> LatentSpace:
    return LatentSpace(dim=dim, source_model=model, metadata={"revision": revision})


def _value(data: np.ndarray, space: LatentSpace) -> LatentValue:
    return LatentValue(np.asarray(data, dtype=np.float64), space)


class TestCoordinateIdentity:
    def test_builds_from_model_and_revision(self) -> None:
        value = _value(np.array([1.0, 2.0, 3.0, 4.0]), _space())
        assert value.identity == "modelA::v1"

    def test_explicit_identity_takes_priority(self) -> None:
        space = LatentSpace(dim=2, source_model="modelA", metadata={"source_representation_identity": "run-7"})
        value = _value(np.array([1.0, 2.0]), space)
        assert value.identity == "run-7::modelA"

    def test_empty_when_undeclared(self) -> None:
        value = _value(np.array([1.0, 2.0]), LatentSpace(dim=2))
        assert value.identity == ""

    def test_metadata_identity_and_revision_feed_the_token(self) -> None:
        space = LatentSpace(dim=2, source_model="gpt2")
        value = _value(np.array([1.0, 2.0]), space)
        assert coordinate_identity(space, {"model_version": "e7da7f2"}) == "gpt2::e7da7f2"
        assert value.identity == "gpt2"


class TestArithmeticCompatibility:
    def test_matching_values_are_compatible(self) -> None:
        assert_arithmetic_compatible(_value(np.ones(4), _space()), _value(np.ones(4), _space()))

    def test_geometry_mismatch_rejected(self) -> None:
        a = _value(np.array([1.0, 0.0, 0.0, 0.0]), _space())
        unit = LatentSpace(dim=4, geometry="unit_norm", source_model="modelA", metadata={"revision": "v1"})
        b = _value(np.array([1.0, 0.0, 0.0, 0.0]), unit)
        with pytest.raises(ValueError, match="geometry"):
            assert_arithmetic_compatible(a, b)

    def test_point_shape_mismatch_rejected(self) -> None:
        a = _value(np.ones(4), _space())
        b = _value(np.ones(2), _space(dim=2))
        with pytest.raises(ValueError, match="point shape"):
            assert_arithmetic_compatible(a, b)

    def test_stored_shape_mismatch_rejected(self) -> None:
        a = _value(np.ones(4), _space())
        b = _value(np.ones((2, 4)), _space())
        with pytest.raises(ValueError, match="stored shape"):
            assert_arithmetic_compatible(a, b)

    def test_unclaimed_identity_rejected(self) -> None:
        a = _value(np.ones(4), LatentSpace(dim=4))
        b = _value(np.ones(4), LatentSpace(dim=4))
        with pytest.raises(ValueError, match="declared coordinate-system identity"):
            assert_arithmetic_compatible(a, b)

    def test_model_mismatch_rejected(self) -> None:
        a = _value(np.ones(4), _space())
        b = _value(np.ones(4), _space(model="modelB"))
        with pytest.raises(ValueError, match="unrelated coordinate systems"):
            assert_arithmetic_compatible(a, b)

    def test_revision_mismatch_rejected(self) -> None:
        a = _value(np.ones(4), _space())
        b = _value(np.ones(4), _space(revision="v2"))
        with pytest.raises(ValueError, match="unrelated coordinate systems"):
            assert_arithmetic_compatible(a, b)


class TestArithmeticOperations:
    def test_add_and_subtract(self) -> None:
        a = _value(np.array([1.0, 2.0, 3.0, 4.0]), _space())
        b = _value(np.array([2.0, 3.0, 4.0, 5.0]), _space())
        assert np.allclose((a + b).to_numpy(), [3.0, 5.0, 7.0, 9.0])
        assert np.allclose((a - b).to_numpy(), [-1.0, -1.0, -1.0, -1.0])

    def test_add_scaled_and_scale(self) -> None:
        a = _value(np.array([1.0, 2.0, 3.0, 4.0]), _space())
        b = _value(np.array([2.0, 3.0, 4.0, 5.0]), _space())
        assert np.allclose(a.add_scaled(b, 2.0).to_numpy(), [5.0, 8.0, 11.0, 14.0])
        assert np.allclose(a.scale(3.0).to_numpy(), [3.0, 6.0, 9.0, 12.0])

    def test_batch_arithmetic(self) -> None:
        a = _value(np.array([[1.0, 2.0], [3.0, 4.0]]), _space(dim=2))
        b = _value(np.array([[10.0, 10.0], [20.0, 20.0]]), _space(dim=2))
        assert np.allclose((a + b).to_numpy(), [[11.0, 12.0], [23.0, 24.0]])

    def test_operators_work_via_dunders(self) -> None:
        a = _value(np.array([1.0, 2.0]), _space(dim=2))
        b = _value(np.array([3.0, 4.0]), _space(dim=2))
        assert isinstance(a + b, LatentValue)
        assert isinstance(a - b, LatentValue)

    def test_non_latent_value_operand_not_implemented(self) -> None:
        a = _value(np.array([1.0, 2.0]), _space(dim=2))
        assert a.__add__(5) is NotImplemented
        assert a.__sub__(5) is NotImplemented

    def test_scale_rejects_non_finite(self) -> None:
        a = _value(np.array([1.0, 2.0]), _space(dim=2))
        with pytest.raises(ValueError, match="finite"):
            a.scale(np.nan)  # type: ignore[arg-type]

    def test_invalid_arithmetic_raises_not_plausible_array(self) -> None:
        a = _value(np.ones(4), _space())
        b = _value(np.ones(4), _space(model="other"))
        with pytest.raises(ValueError, match="unrelated coordinate systems"):
            _ = a + b


class TestArithmeticMetadata:
    def test_output_metadata_records_operation(self) -> None:
        a = _value(np.ones(4), _space())
        b = _value(np.ones(4), _space())
        total = a + b
        operation = total.metadata["operation"]
        assert operation["kind"] == "latent_arithmetic"
        assert operation["op"] == "add"
        assert operation["coefficients"] == (1.0, 1.0)

    def test_provenance_chain_grows_across_ops(self) -> None:
        a = _value(np.ones(4), _space())
        b = _value(np.ones(4), _space())
        intermediate = a + b
        final = intermediate.scale(2.0)
        assert len(final.metadata["provenance"]) == 2
        assert final.metadata["provenance"][0]["op"] == "add"
        assert final.metadata["provenance"][1]["op"] == "scale"

    def test_output_is_immutable(self) -> None:
        a = _value(np.ones(4), _space())
        b = _value(np.ones(4), _space())
        total = a + b
        with pytest.raises(TypeError):
            total.metadata["operation"]["op"] = "mutated"  # type: ignore[index]
        output = total.to_numpy()
        output[:] = 0.0
        assert not np.allclose(total.to_numpy(), output)

    def test_inputs_remain_unchanged(self) -> None:
        a = _value(np.array([1.0, 2.0]), _space(dim=2))
        b = _value(np.array([3.0, 4.0]), _space(dim=2))
        _ = a + b
        assert np.allclose(a.to_numpy(), [1.0, 2.0])
        assert np.allclose(b.to_numpy(), [3.0, 4.0])
