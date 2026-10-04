from __future__ import annotations

import struct
import sys
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


@unittest.skipIf(np is None, "NumPy depth dependency unavailable")
class DepthDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global build_depth_diagnostic, compact_depth_diagnostic_summary
        from realestate_splat.depth_diagnostic import (
            build_depth_diagnostic,
            compact_depth_diagnostic_summary,
        )

    def test_back_projects_only_valid_high_confidence_samples(self) -> None:
        files = {
            "memory://depth-0": struct.pack("<4f", 1.0, 1.0, float("nan"), 9.0),
            "memory://confidence-0": bytes([2, 1, 2, 2]),
            "memory://depth-1": struct.pack("<4f", 1.0, 1.0, 1.0, 1.0),
            "memory://confidence-1": bytes([2, 2, 2, 2]),
        }
        hybrid = self._hybrid()
        artifact = build_depth_diagnostic(
            hybrid,
            files.__getitem__,
            pixel_stride=1,
            output_voxel_meters=0.05,
            agreement_voxel_meters=0.25,
            max_viewer_points=100,
        )

        self.assertEqual(artifact["status"], "completed")
        self.assertEqual(artifact["processed_frame_count"], 2)
        self.assertEqual(artifact["high_confidence_sample_count"], 7)
        self.assertEqual(artifact["valid_high_confidence_sample_count"], 5)
        self.assertEqual(artifact["sampled_point_count"], 5)
        self.assertEqual(artifact["source_statistics"]["colmap_registered"]["frame_count"], 1)
        self.assertEqual(artifact["source_statistics"]["arkit_propagated"]["frame_count"], 1)
        self.assertTrue(all(point["position"][2] == -1.0 for point in artifact["points"]))
        self.assertGreater(artifact["overlap_statistics"]["multi_frame_cell_count"], 0)
        self.assertEqual(len(artifact["selection_sha256"]), 64)
        self.assertNotIn("points", compact_depth_diagnostic_summary(artifact))

    def test_skips_malformed_frame_but_keeps_valid_frames(self) -> None:
        files = {
            "memory://depth-1": struct.pack("<4f", 1.0, 1.0, 1.0, 1.0),
            "memory://confidence-1": bytes([2, 2, 2, 2]),
        }
        hybrid = self._hybrid()
        hybrid["captures"][0]["frames"][0]["depth"] = None
        artifact = build_depth_diagnostic(hybrid, files.__getitem__, pixel_stride=1)
        self.assertEqual(artifact["status"], "completed_with_skips")
        self.assertEqual(artifact["processed_frame_count"], 1)
        self.assertEqual(artifact["skipped_frame_count"], 1)
        self.assertIn("missing depth", artifact["skipped_frames"][0]["reason"])

    @staticmethod
    def _hybrid() -> dict:
        frames = []
        for index, source in enumerate(("colmap_registered", "arkit_propagated")):
            transform = np.eye(4)
            transform[0, 3] = index * 0.01
            frames.append(
                {
                    "capture_id": "capture-1",
                    "frame_index": index,
                    "image_name": f"frame-{index}.jpg",
                    "width": 4,
                    "height": 4,
                    "pose_source": source,
                    "camera_intrinsics": [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0],
                    "camera_to_world_reference_arkit_row_major": transform.tolist(),
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
