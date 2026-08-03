"""Deterministic constrained-manipulation and multi-view evidence tests."""

import numpy as np
import pytest

from latent_anything.gaussian_3d import (
    edit_color,
    edit_opacity,
    merge_gaussians,
    naive_parameter_arithmetic,
    remove_gaussians,
    rigid_transform,
)
from latent_anything.gaussian_3d_evaluation import evaluate_multiview
from latent_anything.pose import SE3, SO3


def scene() -> np.ndarray:
    value = np.zeros((3, 14), dtype=np.float64)
    value[:, 2] = [2.0, 2.4, 2.8]
    value[:, 6] = 1.0
    value[:, 7:10] = 0.1
    value[:, 10] = [0.8, 0.5, 0.3]
    value[:, 11:14] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return value


def test_rigid_transform_preserves_scales_and_applies_se3_to_positions() -> None:
    original = scene()
    transform = SE3(SO3.exp(np.array([0.0, 0.0, np.pi / 2])), np.array([1.0, 0.0, 0.0]))
    edited = rigid_transform(original, transform)
    np.testing.assert_allclose(edited[0, :3], [1.0, 0.0, 2.0])
    np.testing.assert_allclose(edited[:, 7:10], original[:, 7:10])
    np.testing.assert_allclose(original[0, :3], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(np.linalg.norm(edited[:, 3:7], axis=1), 1.0)


def test_opacity_color_removal_and_merge_are_bounded() -> None:
    original = scene()
    edited = edit_opacity(original, [0], value=0.2)
    edited = edit_color(edited, [1], [0.1, 0.0, 0.3], mode="add")
    reduced = remove_gaussians(edited, [2])
    merged = merge_gaussians(reduced, [[0, 1]])
    assert merged.shape == (1, 14)
    assert 0.0 <= merged[0, 10] <= 1.0
    assert np.all((merged[0, 11:14] >= 0.0) & (merged[0, 11:14] <= 1.0))


def test_invalid_naive_parameter_arithmetic_is_rejected_by_renderer_contract() -> None:
    delta = np.zeros((3, 14), dtype=np.float64)
    delta[:, 10] = 0.4
    invalid = naive_parameter_arithmetic(scene(), delta)
    assert invalid[0, 10] > 1.0
    with pytest.raises(ValueError, match="opacity"):
        edit_opacity(scene(), [0], value=1.4)


def test_multiview_metrics_report_target_change_and_zero_off_target_drift() -> None:
    baseline = scene()
    edited = edit_opacity(baseline, [0], value=0.2)

    def render(value: np.ndarray, camera: object) -> np.ndarray:
        del camera
        return value[:, 10:11].sum() * np.ones((4, 4, 3))

    result = evaluate_multiview(
        baseline,
        edited,
        target_indices=[0],
        cameras=[object(), object()],
        render=render,
    )
    assert result.target_position_change == 0.0
    assert result.off_target_drift == 0.0
    assert result.multi_view_image_consistency == 1.0
    assert result.render_quality_degradation > 0.0
