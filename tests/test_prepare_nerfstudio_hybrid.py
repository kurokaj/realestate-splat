from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_nerfstudio_from_colmap import (
    Camera,
    ImagePose,
    Point3D,
    build_hybrid_transforms,
    colmap_camera_to_world_to_nerfstudio_transform,
    colmap_pose_to_nerfstudio_transform,
    hybrid_frames,
    reference_arkit_camera_to_world_to_nerfstudio_transform,
)


def identity_pose(x: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def hybrid_artifact() -> dict:
    frames = [
        {
            "image_name": "frame_000000.jpg",
            "width": 1920,
            "height": 1440,
            "capture_id": "capture-1",
            "frame_index": 0,
            "pose_source": "colmap_registered",
            "replacement_reasons": [],
            "camera_intrinsics": [1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 960.0, 720.0, 1.0],
            "camera_to_world_colmap_opencv_row_major": identity_pose(),
            "camera_to_world_reference_arkit_row_major": identity_pose(),
        },
        {
            "image_name": "frame_000001.jpg",
            "width": 1920,
            "height": 1440,
            "capture_id": "capture-1",
            "frame_index": 1,
            "pose_source": "arkit_propagated",
            "replacement_reasons": ["missing_colmap_pose"],
            "camera_intrinsics": [1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 960.0, 720.0, 1.0],
            "camera_to_world_colmap_opencv_row_major": identity_pose(0.1),
            "camera_to_world_reference_arkit_row_major": identity_pose(0.1),
        },
    ]
    return {
        "schema_version": 1,
        "kind": "buildvision3d_arkit_hybrid_camera_set",
        "status": "completed",
        "source_run_id": "run-a",
        "selection_sha256": "a" * 64,
        "artifact_uri": "r2://bucket/hybrid.json",
        "frame_count": 2,
        "colmap_registered_count": 1,
        "arkit_propagated_count": 1,
        "transition_validation": {"review_required_count": 0},
        "captures": [{"capture_id": "capture-1", "frames": frames}],
    }


class HybridNerfstudioPreparationTests(unittest.TestCase):
    def test_camera_to_world_conversion_matches_colmap_axis_policy(self) -> None:
        transform = colmap_camera_to_world_to_nerfstudio_transform(identity_pose(2.0))

        self.assertEqual(
            transform,
            [
                [1.0, -0.0, -0.0, 2.0],
                [0.0, -0.0, -1.0, 0.0],
                [-0.0, 1.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        self.assertEqual(
            transform,
            colmap_pose_to_nerfstudio_transform(
                ImagePose(1, (1.0, 0.0, 0.0, 0.0), (-2.0, 0.0, 0.0), 1, "frame.jpg")
            ),
        )

    def test_rejects_hybrid_artifact_from_another_colmap_run(self) -> None:
        with self.assertRaisesRegex(SystemExit, "source run does not match"):
            hybrid_frames(hybrid_artifact(), expected_colmap_run_id="run-b")

    def test_reference_arkit_conversion_preserves_gravity_as_nerfstudio_z_up(self) -> None:
        pose = identity_pose()
        pose[1][3] = 2.0
        pose[2][3] = -3.0
        transform = reference_arkit_camera_to_world_to_nerfstudio_transform(pose)
        self.assertEqual([transform[0][3], transform[1][3], transform[2][3]], [0.0, 3.0, 2.0])

    def test_builds_all_hybrid_frames_with_base_colmap_intrinsics_and_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            frames_dir = run_dir / "frames_selected"
            reports_dir = run_dir / "reports"
            frames_dir.mkdir()
            reports_dir.mkdir()
            for index in range(2):
                (frames_dir / f"frame_{index:06d}.jpg").write_bytes(b"test")
            manifest = {
                "images": [
                    {
                        "image_name": f"frame_{index:06d}.jpg",
                        "camera_group": "skanea-capture",
                        "camera_group_id": "skanea-capture_1920x1440",
                        "width": 1920,
                        "height": 1440,
                        "location": "room",
                        "source_id": f"source-{index}",
                    }
                    for index in range(2)
                ]
            }
            (reports_dir / "image_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            cameras = {1: Camera(1, "SIMPLE_RADIAL", 1920, 1440, (1200.0, 960.0, 720.0, 0.01))}
            images = [ImagePose(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "frame_000000.jpg")]
            points = [Point3D(1, (0.0, 0.0, 1.0), (255, 0, 0), 0.2)]

            with patch("scripts.prepare_nerfstudio_from_colmap.image_size", return_value=(1920, 1440)):
                transforms = build_hybrid_transforms(
                    run_dir=run_dir,
                    frames_dir=frames_dir,
                    hybrid_artifact=hybrid_artifact(),
                    images=images,
                    cameras=cameras,
                    points=points,
                    point_cloud_path=run_dir / "colmap_points3D.ply",
                    expected_colmap_run_id="run-a",
                )

        self.assertEqual(len(transforms["frames"]), 2)
        self.assertEqual(transforms["frames"][1]["hybrid_pose_source"], "arkit_propagated")
        self.assertEqual(transforms["frames"][1]["colmap_camera_id"], 1)
        self.assertEqual(transforms["frames"][1]["fl_x"], 1200.0)
        self.assertEqual(transforms["buildvision3d"]["colmap_registered_count"], 1)
        self.assertEqual(transforms["buildvision3d"]["arkit_propagated_count"], 1)
        self.assertEqual(transforms["buildvision3d"]["initialization_source"], "base_colmap_sparse_txt")

    def test_builds_gravity_aligned_hybrid_frames_with_lidar_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            frames_dir = run_dir / "frames_selected"
            reports_dir = run_dir / "reports"
            frames_dir.mkdir()
            reports_dir.mkdir()
            for index in range(2):
                (frames_dir / f"frame_{index:06d}.jpg").write_bytes(b"test")
            manifest = {
                "images": [
                    {
                        "image_name": f"frame_{index:06d}.jpg",
                        "camera_group": "skanea-capture",
                        "camera_group_id": "skanea-capture_1920x1440",
                        "width": 1920,
                        "height": 1440,
                        "source_id": f"source-{index}",
                    }
                    for index in range(2)
                ]
            }
            (reports_dir / "image_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            cameras = {1: Camera(1, "SIMPLE_RADIAL", 1920, 1440, (1200.0, 960.0, 720.0, 0.01))}
            images = [ImagePose(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "frame_000000.jpg")]
            points = [Point3D(1, (0.0, 0.0, 1.0), (255, 0, 0), 0.2)]
            lidar = {
                "selection_sha256": "b" * 64,
                "artifact_uri": "r2://bucket/lidar.json",
                "point_cloud_stats": {"count": 4, "xyz_min": [-1, -1, -1], "xyz_max": [1, 1, 1]},
            }
            with patch("scripts.prepare_nerfstudio_from_colmap.image_size", return_value=(1920, 1440)):
                transforms = build_hybrid_transforms(
                    run_dir=run_dir,
                    frames_dir=frames_dir,
                    hybrid_artifact=hybrid_artifact(),
                    images=images,
                    cameras=cameras,
                    points=points,
                    point_cloud_path=run_dir / "lidar_initialization.ply",
                    expected_colmap_run_id="run-a",
                    lidar_initialization=lidar,
                )

        self.assertEqual(transforms["buildvision3d"]["initialization_source"], "high_confidence_lidar")
        self.assertEqual(
            transforms["buildvision3d"]["coordinate_frame"],
            "reference_arkit_gravity_aligned_nerfstudio_z_up",
        )
        self.assertEqual(transforms["buildvision3d"]["point_cloud_stats"]["count"], 4)


if __name__ == "__main__":
    unittest.main()
