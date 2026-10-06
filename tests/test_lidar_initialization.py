from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for candidate in (ROOT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


@unittest.skipIf(np is None or cv2 is None, "NumPy/OpenCV LiDAR dependencies unavailable")
class LidarInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global build_lidar_initialization, write_lidar_initialization_ply
        from realestate_splat.lidar_initialization import (
            build_lidar_initialization,
            write_lidar_initialization_ply,
        )

    def test_builds_supported_rgb_colored_gravity_aligned_points(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:, :] = [10, 20, 30]
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.assertTrue(ok)
        files = {
            "memory://rgb-0": encoded.tobytes(),
            "memory://rgb-1": encoded.tobytes(),
            "memory://depth-0": struct.pack("<4f", 1.0, 1.0, 1.0, 1.0),
            "memory://depth-1": struct.pack("<4f", 1.0, 1.0, 1.0, 1.0),
            "memory://confidence-0": bytes([2, 2, 2, 2]),
            "memory://confidence-1": bytes([2, 2, 2, 2]),
        }
        artifact, points, colors = build_lidar_initialization(
            self._hybrid(),
            files.__getitem__,
            pixel_stride=1,
            voxel_meters=0.10,
            min_frame_support=2,
            max_preview_points=100,
        )

        self.assertEqual(artifact["status"], "completed")
        self.assertEqual(artifact["processed_frame_count"], 2)
        self.assertGreater(len(points), 0)
        self.assertEqual(len(points), artifact["filtered_voxel_count"])
        self.assertTrue(np.allclose(points[:, 1], 1.0))  # ARKit z=-1 becomes Nerfstudio y=+1.
        self.assertLess(float(np.min(points[:, 2])), 0.0)
        self.assertGreater(float(np.max(points[:, 2])), 0.0)
        self.assertEqual(colors.shape, points.shape)
        self.assertEqual(len(artifact["selection_sha256"]), 64)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "init.ply"
            write_lidar_initialization_ply(path, points, colors)
            self.assertIn(f"element vertex {len(points)}", path.read_text(encoding="utf-8"))

    def test_rejects_multiple_independent_captures(self) -> None:
        hybrid = self._hybrid()
        hybrid["captures"].append({"capture_id": "capture-2", "frames": []})
        with self.assertRaisesRegex(ValueError, "exactly one continuous ARKit capture"):
            build_lidar_initialization(hybrid, lambda _uri: b"")

    @staticmethod
    def _hybrid() -> dict:
        frames = []
        for index, source in enumerate(("colmap_registered", "arkit_propagated")):
            frames.append(
                {
                    "capture_id": "capture-1",
                    "frame_index": index,
                    "image_name": f"frame-{index}.jpg",
                    "source_id": f"source-{index}",
                    "width": 4,
                    "height": 4,
                    "pose_source": source,
                    "camera_intrinsics": [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0],
                    "camera_to_world_reference_arkit_row_major": np.eye(4).tolist(),
                    "rgb_uri": f"memory://rgb-{index}",
                    "depth": {"uri": f"memory://depth-{index}", "width": 2, "height": 2},
                    "confidence": {"uri": f"memory://confidence-{index}", "width": 2, "height": 2},
                }
            )
        return {
            "status": "completed",
            "source_run_id": "run-a",
            "selection_sha256": "a" * 64,
            "captures": [{"capture_id": "capture-1", "frames": frames}],
        }


if __name__ == "__main__":
    unittest.main()
