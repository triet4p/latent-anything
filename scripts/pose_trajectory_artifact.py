"""Generate Sprint 51 pose interpolation evidence and a motion plot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from latent_anything import SE3, SO3, PoseMetadata


def main() -> None:
    metadata = PoseMetadata("world", "tool")
    start = SE3.identity(metadata=metadata)
    end = SE3(SO3.exp(np.array([0.0, 0.0, np.pi / 2])), np.array([1.0, 0.5, 0.25]), metadata=metadata)
    ts = np.linspace(0.0, 1.0, 21)
    group_path = [start.interpolate(end, float(t)) for t in ts]
    elementwise = [(1.0 - t) * start.matrix + t * end.matrix for t in ts]
    elementwise_rotation_error = [
        float(np.linalg.norm(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3))) for matrix in elementwise
    ]
    group_distances = [start.distance(pose) for pose in group_path]
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    report = {
        "representation": "SE3 homogeneous matrix with SO3 matrix rotation",
        "frames": metadata.to_dict(),
        "n_steps": len(ts),
        "group_distance_endpoint": group_distances[-1],
        "group_distance_monotonic": bool(np.all(np.diff(group_distances) >= -1e-10)),
        "max_elementwise_rotation_orthogonality_error": max(elementwise_rotation_error),
        "group_interpolation_max_rotation_orthogonality_error": max(
            float(np.linalg.norm(pose.rotation.matrix.T @ pose.rotation.matrix - np.eye(3))) for pose in group_path
        ),
    }
    (output / "pose_trajectory_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(6, 4))
    positions = np.stack([pose.translation for pose in group_path])
    axis.plot(positions[:, 0], positions[:, 1], "o-", label="SE(3) interpolation")  # type: ignore[reportUnknownMemberType]
    axis.scatter(  # type: ignore[reportUnknownMemberType]
        [start.translation[0], end.translation[0]], [start.translation[1], end.translation[1]], color="black"
    )
    axis.set(xlabel="x (m)", ylabel="y (m)", title="Pose trajectory in world frame")
    axis.legend()  # type: ignore[reportUnknownMemberType]
    axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(output / "pose_trajectory.png", dpi=140)  # type: ignore[reportUnknownMemberType]
    plt.close(figure)


if __name__ == "__main__":
    main()
