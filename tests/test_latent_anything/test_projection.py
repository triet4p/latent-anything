"""Tests for orthonormal subspace projection and removal.

Covers the pure algorithms in ``geometry.py`` (basis validation,
orthonormalization, ``P z`` / ``(I - P) z``, coverage, alignment) and the
stateful contract in ``projection.py`` (``OrthonormalSubspace`` validation,
serialization, basis families, ``SubspaceProjection`` config construction and
the project/remove/coverage/transfer operations with identity binding and
operation metadata).
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_anything import LatentSpace, LatentValue, SubspaceProjection, build_from_dict
from latent_anything.geometry import (
    concept_coverage,
    orthonormalize_directions,
    project_point,
    remove_point,
    subspace_alignment,
    validate_orthonormal_basis,
)
from latent_anything.projection import (
    OrthonormalSubspace,
    SubspaceProjectionConfig,
)

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false


def _identity(space: LatentSpace) -> str:
    return f"{space.source_model}::{space.metadata['revision']}"


def _space(dim: int = 4, model: str = "modelA", revision: str = "v1") -> LatentSpace:
    return LatentSpace(dim=dim, source_model=model, metadata={"revision": revision})


def _value(data: np.ndarray, space: LatentSpace) -> LatentValue:
    return LatentValue(np.asarray(data, dtype=np.float64), space)


# ── Pure geometry functions ─────────────────────────────────────────


class TestValidateOrthonormalBasis:
    def test_accepts_identity_block(self) -> None:
        basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        validate_orthonormal_basis(basis, dim=3)

    def test_wrong_row_count_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            validate_orthonormal_basis(np.eye(2), dim=3)

    def test_full_rank_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_basis"):
            validate_orthonormal_basis(np.eye(3), dim=3)

    def test_empty_basis_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_basis"):
            validate_orthonormal_basis(np.empty((3, 0)), dim=3)

    def test_non_orthonormal_columns_rejected(self) -> None:
        basis = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="U.T U"):
            validate_orthonormal_basis(basis, dim=3)

    def test_non_finite_rejected(self) -> None:
        basis = np.array([[np.nan, 0.0], [0.0, 1.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="finite"):
            validate_orthonormal_basis(basis, dim=3)


class TestOrthonormalizeDirections:
    def test_returns_orthonormal_basis(self) -> None:
        directions = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        basis = orthonormalize_directions(directions.T)
        validate_orthonormal_basis(basis, dim=3)
        assert basis.shape == (3, 2)

    def test_collinear_directions_rank_reduced(self) -> None:
        directions = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        basis = orthonormalize_directions(directions.T)
        assert basis.shape == (3, 1)

    def test_single_direction(self) -> None:
        basis = orthonormalize_directions(np.array([[3.0, 4.0]]).T)
        assert basis.shape == (2, 1)
        assert np.isclose(basis[0, 0], 0.6) and np.isclose(basis[1, 0], 0.8)

    def test_zero_directions_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero-dimensional"):
            orthonormalize_directions(np.zeros((3, 2)))


class TestProjectionAlgorithms:
    def test_projection_onto_axis_subspace(self) -> None:
        basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        point = np.array([3.0, 4.0, 5.0])
        assert np.allclose(project_point(point, basis), [3.0, 4.0, 0.0])
        assert np.allclose(remove_point(point, basis), [0.0, 0.0, 5.0])

    def test_idempotence(self) -> None:
        basis = orthonormalize_directions(np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]]).T)
        point = np.array([2.0, -3.0, 7.0])
        once = project_point(point, basis)
        twice = project_point(once, basis)
        assert np.allclose(once, twice)

    def test_orthogonality_of_components(self) -> None:
        basis = orthonormalize_directions(np.array([[1.0, 2.0, 3.0]]).T)
        point = np.array([5.0, -1.0, 2.0])
        projected = project_point(point, basis)
        residual = remove_point(point, basis)
        assert np.isclose(np.dot(projected, residual), 0.0, atol=1e-10)

    def test_reconstruction(self) -> None:
        basis = orthonormalize_directions(np.array([[1.0, 1.0, 1.0]]).T)
        point = np.array([1.0, 2.0, 3.0])
        assert np.allclose(project_point(point, basis) + remove_point(point, basis), point)

    def test_batched_projection(self) -> None:
        basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        points = np.array([[1.0, 0.0, 5.0], [0.0, 2.0, 6.0]])
        projected = project_point(points, basis)
        assert projected.shape == (2, 3)
        assert np.allclose(projected, [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    def test_coverage_inside_and_outside(self) -> None:
        basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        inside = np.array([1.0, 0.0, 0.0])
        outside = np.array([0.0, 0.0, 1.0])
        assert np.isclose(concept_coverage(inside, basis), 1.0)
        assert np.isclose(concept_coverage(outside, basis), 0.0)

    def test_subspace_alignment_identical_and_orthogonal(self) -> None:
        basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        rotated = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        orthogonal = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        assert np.isclose(subspace_alignment(basis, rotated), 1.0)
        assert np.isclose(subspace_alignment(basis, orthogonal), 0.0, atol=1e-10)


# ── OrthonormalSubspace value ───────────────────────────────────────


class TestOrthonormalSubspace:
    def test_from_basis_validates_and_owns(self) -> None:
        basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        subspace = OrthonormalSubspace.from_basis(basis, source_representation_identity="modelA::v1", origin="explicit")
        assert subspace.n_basis == 2
        assert subspace.dim == 3
        basis[0, 0] = 99.0
        with pytest.raises(ValueError):
            subspace.basis[0, 0] = 0.0  # type: ignore[index]

    def test_requires_non_empty_identity(self) -> None:
        with pytest.raises(ValueError, match="identity"):
            OrthonormalSubspace.from_basis(
                np.array([[1.0], [0.0], [0.0]]), source_representation_identity="  ", origin="explicit"
            )

    def test_rejects_unknown_origin(self) -> None:
        with pytest.raises(ValueError, match="origin"):
            OrthonormalSubspace.from_basis(
                np.array([[1.0], [0.0], [0.0]]), source_representation_identity="id", origin="svm"
            )

    def test_non_orthonormal_rejected(self) -> None:
        with pytest.raises(ValueError, match="orthonormal"):
            OrthonormalSubspace.from_basis(
                np.array([[1.0], [1.0], [0.0]]), source_representation_identity="id", origin="explicit"
            )

    def test_to_dict_round_trip(self) -> None:
        subspace = OrthonormalSubspace.from_basis(
            np.array([[1.0], [0.0], [0.0]]),
            source_representation_identity="id",
            origin="concept",
            provenance={"seed": 0},
        )
        restored = OrthonormalSubspace.from_dict(subspace.to_dict())
        assert np.allclose(restored.basis, subspace.basis)
        assert restored.source_representation_identity == "id"
        assert restored.origin == "concept"
        assert restored.provenance["seed"] == 0

    def test_save_load_round_trip(self, tmp_path: object) -> None:
        subspace = OrthonormalSubspace.from_basis(
            np.array([[1.0], [0.0], [0.0]]),
            source_representation_identity="id",
            origin="pca",
        )
        path = tmp_path / "subspace.npz"  # type: ignore[union-attr]
        subspace.save(path)  # type: ignore[arg-type]
        restored = OrthonormalSubspace.load(path)  # type: ignore[arg-type]
        assert np.allclose(restored.basis, subspace.basis)
        assert restored.origin == "pca"

    def test_from_concept_direction_normalizes(self) -> None:
        subspace = OrthonormalSubspace.from_concept_direction(np.array([3.0, 4.0]), source_representation_identity="id")
        assert subspace.n_basis == 1
        assert np.allclose(subspace.basis[:, 0], [0.6, 0.8])

    def test_from_probe_coefficients_multiclass_orthonormalized(self) -> None:
        coefficients = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        subspace = OrthonormalSubspace.from_probe_coefficients(coefficients, source_representation_identity="id")
        assert subspace.origin == "probe"
        assert subspace.n_basis == 2

    def test_fit_pca_keeps_top_components(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(size=(200, 5))
        subspace = OrthonormalSubspace.fit_pca(data, n_components=2, source_representation_identity="id")
        assert subspace.origin == "pca"
        assert subspace.basis.shape == (5, 2)

    def test_from_pca_reads_components(self) -> None:
        from sklearn.decomposition import PCA  # pyright: ignore[reportMissingTypeStubs]

        rng = np.random.default_rng(1)
        pca = PCA(n_components=2)
        pca.fit(rng.normal(size=(200, 5)))  # pyright: ignore[reportUnknownMemberType]
        subspace = OrthonormalSubspace.from_pca(pca, source_representation_identity="id")  # pyright: ignore[reportUnknownArgumentType]
        assert subspace.origin == "pca"
        assert subspace.basis.shape == (5, 2)


# ── SubspaceProjection operations ───────────────────────────────────


class TestSubspaceProjection:
    def _fitted(self, dim: int = 4) -> SubspaceProjection:
        space = _space(dim=dim)
        basis = np.eye(dim, dim // 2)
        subspace = OrthonormalSubspace.from_basis(
            basis, source_representation_identity=_identity(space), origin="explicit"
        )
        return SubspaceProjection.from_subspace(subspace)

    def test_requires_fitted_subspace(self) -> None:
        projection = SubspaceProjection()
        with pytest.raises(RuntimeError, match="no fitted subspace"):
            projection.project(_value(np.array([1.0, 2.0, 3.0, 4.0]), _space()))

    def test_project_remove_reconstruct(self) -> None:
        projection = self._fitted()
        space = _space()
        point = np.array([1.0, 2.0, 3.0, 4.0])
        projected = projection.project(_value(point, space))
        removed = projection.remove(_value(point, space))
        assert np.allclose(projected.to_numpy() + removed.to_numpy(), point)
        assert np.isclose(np.dot(projected.to_numpy(), removed.to_numpy()), 0.0)

    def test_remove_is_idempotent(self) -> None:
        projection = self._fitted()
        point = _value(np.array([1.0, 2.0, 3.0, 4.0]), _space())
        removed = projection.remove(point)
        assert np.allclose(projection.remove(removed).to_numpy(), removed.to_numpy())

    def test_batch_operations(self) -> None:
        projection = self._fitted()
        batch = _value(np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]), _space())
        projected = projection.project(batch)
        removed = projection.remove(batch)
        assert np.allclose(projected.to_numpy() + removed.to_numpy(), batch.to_numpy())
        assert projected.is_batch is True

    def test_outputs_are_immutable(self) -> None:
        projection = self._fitted()
        removed = projection.remove(_value(np.array([1.0, 2.0, 3.0, 4.0]), _space()))
        with pytest.raises(TypeError):
            removed.metadata["operation"]["op"] = "mutated"  # type: ignore[index]
        output = removed.to_numpy()
        output[:] = 0.0
        assert not np.allclose(removed.to_numpy(), output)

    def test_metadata_records_operation_and_provenance(self) -> None:
        projection = self._fitted()
        space = _space()
        removed = projection.remove(_value(np.array([1.0, 2.0, 3.0, 4.0]), space))
        operation = removed.metadata["operation"]
        assert operation["kind"] == "subspace_projection"
        assert operation["op"] == "remove"
        assert operation["basis_origin"] == "explicit"
        provenance = removed.metadata["provenance"]
        assert provenance[0]["op"] == "remove"

    def test_provenance_chain_grows(self) -> None:
        projection = self._fitted()
        space = _space()
        removed = projection.remove(_value(np.array([1.0, 2.0, 3.0, 4.0]), space))
        projected = projection.project(removed)
        assert len(projected.metadata["provenance"]) == 2

    def test_cross_identity_rejected(self) -> None:
        projection = self._fitted()
        other_space = _space(model="modelB")
        with pytest.raises(ValueError, match="unrelated coordinate systems"):
            projection.remove(_value(np.array([1.0, 2.0, 3.0, 4.0]), other_space))

    def test_wrong_shape_rejected(self) -> None:
        projection = self._fitted(dim=4)
        with pytest.raises(ValueError, match="shape"):
            projection.remove(_value(np.array([1.0, 2.0, 3.0]), _space(dim=3)))

    def test_non_euclidean_geometry_rejected(self) -> None:
        projection = self._fitted(dim=4)
        unit_space = LatentSpace(dim=4, geometry="unit_norm", source_model="modelA", metadata={"revision": "v1"})
        point = np.array([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="euclidean"):
            projection.remove(_value(point, unit_space))

    def test_coverage_scalar_and_batch(self) -> None:
        projection = self._fitted()
        inside = _value(np.array([1.0, 0.0, 0.0, 0.0]), _space())
        assert np.isclose(projection.coverage(inside), 1.0)
        outside = _value(np.array([0.0, 0.0, 1.0, 1.0]), _space())
        assert np.isclose(projection.coverage(outside), 0.0)
        batch = _value(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]), _space())
        coverages = projection.coverage(batch)
        assert isinstance(coverages, np.ndarray)
        assert np.allclose(coverages, [1.0, 0.0])

    def test_transfer_keeps_target_content_and_source_concept(self) -> None:
        projection = self._fitted()
        space = _space()
        source = _value(np.array([10.0, 10.0, 0.0, 0.0]), space)
        target = _value(np.array([0.0, 0.0, 7.0, 7.0]), space)
        transferred = projection.transfer(source, target)
        assert np.allclose(transferred.to_numpy(), [10.0, 10.0, 7.0, 7.0])

    def test_transfer_shape_mismatch_rejected(self) -> None:
        projection = self._fitted()
        space = _space()
        source = _value(np.array([[10.0, 10.0, 0.0, 0.0]]), space)
        target = _value(np.array([0.0, 0.0, 7.0, 7.0]), space)
        with pytest.raises(ValueError, match="equal value shapes"):
            projection.transfer(source, target)

    def test_fit_basis_uses_config_n_basis(self) -> None:
        projection = SubspaceProjection(SubspaceProjectionConfig(n_basis=1))
        projection.fit_basis(np.eye(4, 2), source_representation_identity="modelA::v1")
        assert projection.subspace.n_basis == 1

    def test_config_construction_from_registry(self) -> None:
        projection = build_from_dict({"kind": "intervention", "name": "subspace_projection"})
        assert isinstance(projection, SubspaceProjection)
        assert projection.config.n_basis is None
