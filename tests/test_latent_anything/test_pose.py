"""Properties and integration tests for SO(3)/SE(3) pose geometry."""

import numpy as np
import pytest

from latent_anything import SE3, SO3, LatentSpace, LatentValue, PoseConfig, PoseMetadata, PoseTrajectory


def _rotation() -> SO3:
    return SO3.exp(np.array([0.2, -0.1, 0.3]))


def test_so3_group_operations_and_log_exp_round_trip() -> None:
    rotation = _rotation()
    identity = SO3.identity()
    assert np.allclose(rotation.compose(rotation.inverse()).matrix, identity.matrix)
    assert np.allclose(SO3.exp(rotation.log()).matrix, rotation.matrix)
    assert np.isclose(identity.distance(rotation), np.linalg.norm(rotation.log()))
    assert np.allclose(rotation.interpolate(identity, 0.0).matrix, rotation.matrix)
    assert np.allclose(rotation.interpolate(identity, 1.0).matrix, identity.matrix)


def test_se3_group_operations_and_valid_interpolation() -> None:
    metadata = PoseMetadata(parent_frame="world", child_frame="tool")
    pose = SE3(_rotation(), np.array([0.3, -0.2, 0.5]), metadata=metadata)
    recovered = SE3.exp(pose.log(), metadata=metadata)
    assert np.allclose(recovered.matrix, pose.matrix, atol=1e-8)
    assert np.allclose(pose.compose(pose.inverse()).matrix, np.eye(4))
    middle = pose.interpolate(SE3.identity(metadata=metadata), 0.5)
    assert np.allclose(middle.matrix[3], [0, 0, 0, 1])
    assert np.isclose(np.linalg.det(middle.matrix[:3, :3]), 1.0)


def test_pose_frame_mismatch_is_explicit() -> None:
    world_tool = SE3.identity(metadata=PoseMetadata("world", "tool"))
    camera_gripper = SE3.identity(metadata=PoseMetadata("camera", "gripper"))
    with pytest.raises(ValueError, match="frame mismatch"):
        world_tool.compose(camera_gripper)
    with pytest.raises(ValueError, match="matching parent and child"):
        world_tool.distance(camera_gripper)


def test_pose_trajectory_slicing_and_lerobot_metadata() -> None:
    metadata = PoseMetadata("world", "tool")
    trajectory = PoseTrajectory([SE3.identity(metadata=metadata), SE3(np.eye(3), np.ones(3), metadata=metadata)])
    assert len(trajectory[1:]) == 1
    assert trajectory.to_numpy().shape == (2, 4, 4)
    assert trajectory.lerobot_metadata()["features"]["observation.state"]["shape"] == [4, 4]
    assert trajectory.group_distance()[1] > 0


def test_pose_serialization_and_latent_space_boundary() -> None:
    metadata = PoseMetadata("world", "tool")
    pose = SE3(_rotation(), np.array([1.0, 2.0, 3.0]), metadata=metadata)
    rebuilt = SE3.from_dict(pose.to_dict())
    assert np.allclose(rebuilt.matrix, pose.matrix)
    assert PoseConfig(parent_frame="world", child_frame="tool").metadata() == metadata
    space = LatentSpace(dim=16, geometry="se3")
    value = LatentValue(pose.matrix, space)
    assert np.allclose(value.to_numpy(), pose.matrix)
    with pytest.raises(ValueError, match="Expected shape"):
        space.validate_point(np.eye(3))
