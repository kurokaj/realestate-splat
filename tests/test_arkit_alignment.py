from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for candidate in (ROOT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.prepare_nerfstudio_from_colmap import ImagePose


def rot_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def rotmat_to_qvec(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        values = (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            0.25 * scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        values = (
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            0.25 * scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        values = (
            (rotation[1, 0] - rotation[0, 1]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            0.25 * scale,
        )
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(float(value / norm) for value in values)


@unittest.skipIf(np is None, "NumPy alignment dependency unavailable")
class ArkitAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global ARKIT_TO_OPENCV_CAMERA_AXES, analyze_arkit_alignment
        global build_aligned_arkit_trajectories, build_hybrid_camera_set
        global read_colmap_images_streaming, select_hybrid_camera_poses
        global build_alignment_viewer_overlay, camera_row, normalize_sparse_viewer_payload
        from controller_common.colmap_viewer import (
            build_alignment_viewer_overlay,
            camera_row,
            normalize_sparse_viewer_payload,
        )
        from realestate_splat.arkit_alignment import (
            ARKIT_TO_OPENCV_CAMERA_AXES,
            analyze_arkit_alignment,
            build_aligned_arkit_trajectories,
            build_hybrid_camera_set,
            read_colmap_images_streaming,
            select_hybrid_camera_poses,
        )

    def test_recovers_similarity_and_rejects_one_position_outlier(self) -> None:
        source_centers = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [1.5, 0.4, 0.2], [0.7, 1.1, 0.3], [-0.2, 0.8, 0.1], [0.3, 0.3, 0.5]],
            dtype=np.float64,
        )
        scale = 2.5
        world_rotation = rot_z(0.35)
        translation = np.array([4.0, -2.0, 1.5])
        target_centers = (scale * (world_rotation @ source_centers.T)).T + translation
        target_centers[-1] += np.array([4.0, -3.0, 2.0])

        manifest_images = []
        colmap_images = []
        for index, (source_center, target_center) in enumerate(zip(source_centers, target_centers)):
            image_name = f"frame_{index:06d}.jpg"
            arkit_transform = np.eye(4)
            arkit_transform[:3, 3] = source_center
            colmap_camera_to_world = world_rotation @ ARKIT_TO_OPENCV_CAMERA_AXES
            world_to_camera = colmap_camera_to_world.T
            tvec = -world_to_camera @ target_center
            manifest_images.append(
                {
                    "image_name": image_name,
                    "source_kind": "skanea_rgbd_frame",
                    "rgbd": {
                        "capture_id": "capture-1",
                        "frame_index": index,
                        "camera_transform": arkit_transform.reshape(16, order="F").tolist(),
                        "camera_transform_layout": "4x4 column-major camera-to-world",
                    },
                }
            )
            colmap_images.append(
                ImagePose(index + 1, rotmat_to_qvec(world_to_camera), tuple(tvec), 1, image_name)
            )

        report = analyze_arkit_alignment(
            {"images": manifest_images},
            colmap_images,
            source_run_id="run-a",
            threshold_fraction=0.08,
            ransac_iterations=500,
        )

        self.assertEqual(report["status"], "completed")
        capture = report["captures"][0]
        self.assertEqual(capture["inlier_count"], 5)
        self.assertAlmostEqual(capture["scale_colmap_units_per_meter"], scale, places=6)
        self.assertLess(capture["position_residual_colmap_units"]["max"], 1e-5)
        self.assertLess(capture["position_residual_meters"]["max"], 1e-5)
        self.assertLess(capture["angular_residual_degrees"]["max"], 1e-5)
        self.assertFalse(capture["correspondences"][-1]["inlier"])

        hybrid = build_hybrid_camera_set(
            report,
            {"images": manifest_images},
            colmap_images,
            source_run_id="run-a",
        )
        self.assertEqual(hybrid["status"], "completed")
        self.assertEqual(hybrid["frame_count"], 6)
        self.assertEqual(hybrid["colmap_registered_count"], 5)
        self.assertEqual(hybrid["arkit_propagated_count"], 1)
        self.assertEqual(len(hybrid["selection_sha256"]), 64)
        self.assertEqual(
            hybrid,
            build_hybrid_camera_set(
                report,
                {"images": manifest_images},
                colmap_images,
                source_run_id="run-a",
            ),
        )
        self.assertEqual(hybrid["transition_validation"]["source_transition_count"], 1)
        hybrid_frames = hybrid["captures"][0]["frames"]
        self.assertEqual(hybrid_frames[-1]["pose_source"], "arkit_propagated")
        self.assertTrue(
            np.allclose(
                np.asarray(hybrid_frames[-1]["camera_to_world_reference_arkit_row_major"])[:3, 3],
                source_centers[-1],
                atol=1e-5,
            )
        )

        unregistered_transform = np.eye(4)
        unregistered_transform[:3, 3] = [0.25, 0.5, 0.75]
        manifest_images.append(
            {
                "image_name": "frame_000006.jpg",
                "source_kind": "skanea_rgbd_frame",
                "rgbd": {
                    "capture_id": "capture-1",
                    "frame_index": 6,
                    "camera_transform": unregistered_transform.reshape(16, order="F").tolist(),
                    "camera_transform_layout": "4x4 column-major camera-to-world",
                },
            }
        )
        trajectory = build_aligned_arkit_trajectories(report, {"images": manifest_images})
        self.assertEqual(trajectory["frame_count"], 7)
        self.assertEqual(trajectory["registered_count"], 6)
        self.assertEqual(trajectory["inlier_count"], 5)
        self.assertEqual(trajectory["outlier_count"], 1)
        self.assertEqual(trajectory["missing_count"], 1)
        self.assertEqual(trajectory["captures"][0]["frames"][-1]["classification"], "missing")
        self.assertEqual(trajectory["hybrid_arkit_count"], 7)

        viewer_overlay = build_alignment_viewer_overlay(report, {"images": manifest_images})
        viewer_position = np.asarray(viewer_overlay["captures"][0]["frames"][0]["arkit_position"] + [1.0])
        display_transform = np.asarray(viewer_overlay["display_transform_row_major"])
        recovered_arkit_position = display_transform @ viewer_position
        self.assertTrue(np.allclose(recovered_arkit_position[:3], source_centers[0], atol=1e-5))
        self.assertEqual(viewer_overlay["display_coordinate_convention"], "reference ARKit world: meters, +Y up (gravity aligned)")

    def test_hybrid_selection_expands_consecutive_unstable_segment(self) -> None:
        frames = [
            {
                "frame_index": index,
                "registered": True,
                "inlier": index not in {7, 8},
                "position_residual_meters": 0.01,
                "angular_residual_degrees": 0.5,
                "aligned_arkit_center_colmap": [float(index), 0.0, 0.0],
                "colmap_center": [float(index), 0.1, 0.0],
            }
            for index in range(15)
        ]

        selected = select_hybrid_camera_poses(frames)

        self.assertEqual([row["hybrid_pose_source"] for row in selected[:2]], ["colmap", "colmap"])
        self.assertTrue(all(row["hybrid_pose_source"] == "arkit" for row in selected[2:14]))
        self.assertEqual(selected[14]["hybrid_pose_source"], "colmap")
        self.assertIn("alignment_outlier", selected[7]["hybrid_replacement_reasons"])
        self.assertEqual(selected[2]["hybrid_replacement_reasons"], ["unstable_segment_guard"])

    def test_hybrid_selection_does_not_expand_isolated_missing_frame(self) -> None:
        frames = [
            {
                "frame_index": index,
                "registered": index != 3,
                "inlier": None if index == 3 else True,
                "position_residual_meters": None if index == 3 else 0.01,
                "angular_residual_degrees": None if index == 3 else 0.5,
                "aligned_arkit_center_colmap": [float(index), 0.0, 0.0],
                "colmap_center": None if index == 3 else [float(index), 0.1, 0.0],
            }
            for index in range(7)
        ]

        selected = select_hybrid_camera_poses(frames)

        self.assertEqual(
            [row["hybrid_pose_source"] for row in selected],
            ["colmap", "colmap", "colmap", "arkit", "colmap", "colmap", "colmap"],
        )
        self.assertEqual(selected[3]["hybrid_replacement_reasons"], ["missing_colmap"])

    def test_colmap_forward_points_toward_positive_camera_z(self) -> None:
        image = ImagePose(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "frame.jpg")
        row = camera_row(image)
        self.assertEqual(row["forward"], [0.0, 1.0, -0.0])

    def test_legacy_sparse_viewer_forward_is_upgraded_in_memory(self) -> None:
        upgraded = normalize_sparse_viewer_payload(
            {"schema_version": 1, "cameras": [{"name": "frame.jpg", "forward": [1.0, -2.0, 3.0]}]}
        )
        self.assertEqual(upgraded["schema_version"], 2)
        self.assertEqual(upgraded["upgraded_from_schema_version"], 1)
        self.assertEqual(upgraded["cameras"][0]["forward"], [-1.0, 2.0, -3.0])

    def test_generic_video_manifest_is_not_applicable(self) -> None:
        image = ImagePose(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "video.jpg")
        report = analyze_arkit_alignment(
            {"images": [{"image_name": "video.jpg", "source_kind": "video_frame"}]},
            [image],
        )
        self.assertEqual(report["status"], "not_applicable")
        self.assertEqual(report["capture_count"], 0)

    def test_streaming_colmap_reader_handles_empty_observation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "images.txt"
            path.write_text(
                "# Image list\n"
                "1 1 0 0 0 0 0 0 1 first.jpg\n"
                "\n"
                "2 1 0 0 0 1 2 3 1 second.jpg\n"
                "10 20 -1\n",
                encoding="utf-8",
            )
            images = read_colmap_images_streaming(path)
        self.assertEqual([image.name for image in images], ["first.jpg", "second.jpg"])


if __name__ == "__main__":
    unittest.main()
