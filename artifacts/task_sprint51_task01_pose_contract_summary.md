# Task Summary: Sprint 51 Task 01 — Pose Contract

Added matrix-backed SO(3)/SE(3) representations with explicit parent/child frames and metre/radian units. Euler angles are not an implicit boundary; quaternion conversion requires an explicit `xyzw` or `wxyz` order.

Evidence: `tests/test_latent_anything/test_pose.py::test_pose_serialization_and_latent_space_boundary`.
