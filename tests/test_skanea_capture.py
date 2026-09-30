from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from controller_common.raw_upload import merge_with_existing_manifest
from scripts import preprocess_video
from scripts.run_preprocess_stage import attach_raw_source_metadata
from realestate_splat.skanea_capture import (
    SkaneaCaptureError,
    build_skanea_manifest_fragment,
    load_skanea_capture,
    registered_artifact_paths,
)


class SkaneaCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.capture_dir = Path(self.temp.name) / "session-test"
        (self.capture_dir / "rgb").mkdir(parents=True)
        (self.capture_dir / "depth").mkdir()
        (self.capture_dir / "confidence").mkdir()
        (self.capture_dir / "rgb" / "000000.jpg").write_bytes(b"jpeg")
        (self.capture_dir / "depth" / "000000.f32").write_bytes(struct.pack("<4f", 1, 2, 3, 4))
        (self.capture_dir / "confidence" / "000000.u8").write_bytes(bytes([2, 2, 1, 0]))
        self.manifest = {
            "schemaVersion": 2,
            "sessionID": "ABC-123",
            "appVersion": "1.0",
            "deviceModel": "iPhone 17 Pro",
            "cameraLens": "wide-1x",
            "captureMode": "indoorRoom",
            "startedAt": "2026-09-25T10:00:00Z",
            "endedAt": "2026-09-25T10:01:00Z",
            "dataDescription": {
                "depthEncoding": "Float32 little-endian, tightly packed rows, meters",
                "confidenceEncoding": "UInt8 ARConfidenceLevel raw values: 0 low, 1 medium, 2 high",
                "cameraIntrinsicsLayout": "3x3 column-major",
                "cameraTransformLayout": "4x4 column-major camera-to-world",
                "worldCoordinateSystem": "ARKit right-handed world coordinates, gravity aligned",
            },
            "frames": [{
                "index": 0,
                "timestamp": 123.5,
                "rgbFile": "rgb/000000.jpg",
                "rgbWidth": 1920,
                "rgbHeight": 1440,
                "depthFile": "depth/000000.f32",
                "depthWidth": 2,
                "depthHeight": 2,
                "confidenceFile": "confidence/000000.u8",
                "confidenceWidth": 2,
                "confidenceHeight": 2,
                "cameraIntrinsics": [1, 0, 0, 0, 1, 0, 0.5, 0.5, 1],
                "cameraTransform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            }],
            "droppedFrames": [],
        }
        self.write_manifest()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_manifest(self) -> None:
        (self.capture_dir / "capture.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def fragment(self) -> dict:
        fragment, _ = build_skanea_manifest_fragment(
            project_id="project-1",
            capture_dir=self.capture_dir,
            destination_uri="r2://bucket/projects/project-1/raw",
            location="Living Room",
            created_at="2026-09-25T10:02:00Z",
        )
        return fragment

    def test_builds_visual_source_with_rgbd_associations(self) -> None:
        fragment = self.fragment()
        source = fragment["sources"][0]
        self.assertEqual(source["location"], "living_room")
        self.assertEqual(source["source_kind"], "skanea_rgbd_frame")
        self.assertEqual(source["rgbd"]["frame_index"], 0)
        self.assertEqual(source["rgbd"]["depth"]["width"], 2)
        paths = registered_artifact_paths(fragment)
        self.assertIn("skanea/abc_123/rgb/000000.jpg", paths)
        self.assertIn("skanea/abc_123/depth/000000.f32", paths)
        self.assertIn("skanea/abc_123/confidence/000000.u8", paths)
        self.assertIn("skanea/abc_123/capture.json", paths)

    def test_registers_spatial_and_derived_artifacts(self) -> None:
        spatial = self.capture_dir / "spatial" / "roomplan"
        quality = self.capture_dir / "derived" / "quality"
        spatial.mkdir(parents=True)
        quality.mkdir(parents=True)
        (spatial / "room.json").write_text("{}", encoding="utf-8")
        (quality / "blur-summary.json").write_text("{}", encoding="utf-8")

        fragment = self.fragment()
        capture = fragment["captures"][0]
        self.assertIn("skanea/abc_123/spatial/roomplan/room.json", capture["derived_artifacts"])
        self.assertIn("skanea/abc_123/derived/quality/blur-summary.json", capture["derived_artifacts"])
        paths = registered_artifact_paths(fragment)
        self.assertIn("skanea/abc_123/derived/quality/blur-summary.json", paths)

    def test_rejects_unfinalized_capture(self) -> None:
        self.manifest["endedAt"] = None
        self.write_manifest()
        with self.assertRaisesRegex(SkaneaCaptureError, "not finalized"):
            load_skanea_capture(self.capture_dir)

    def test_rejects_wrong_depth_size(self) -> None:
        (self.capture_dir / "depth" / "000000.f32").write_bytes(b"short")
        with self.assertRaisesRegex(SkaneaCaptureError, "expected 16"):
            load_skanea_capture(self.capture_dir)

    def test_rejects_path_traversal(self) -> None:
        self.manifest["frames"][0]["rgbFile"] = "../secret.jpg"
        self.write_manifest()
        with self.assertRaisesRegex(SkaneaCaptureError, "unsafe"):
            load_skanea_capture(self.capture_dir)

    def test_rejects_unregistered_file(self) -> None:
        (self.capture_dir / "unexpected.txt").write_text("not part of the capture contract", encoding="utf-8")
        with self.assertRaisesRegex(SkaneaCaptureError, "unregistered file"):
            load_skanea_capture(self.capture_dir)

    def test_manifest_merge_preserves_existing_sources_and_captures(self) -> None:
        incoming = self.fragment()
        existing = {
            "schema_version": 3,
            "project_id": "project-1",
            "created_at": "earlier",
            "sources": [{"relative_path": "coverage/kitchen/video.mov", "source_id": "video"}],
            "captures": [{"capture_id": "other", "manifest_sha256": "hash"}],
        }
        merged = merge_with_existing_manifest(
            incoming,
            "r2://bucket/projects/project-1/raw",
            None,
            existing=existing,
        )
        self.assertEqual(len(merged["sources"]), 2)
        self.assertEqual({item["capture_id"] for item in merged["captures"]}, {"other", "ABC-123"})
        self.assertEqual(merged["created_at"], "earlier")

    def test_preprocess_manifest_retains_rgbd_frame_identity(self) -> None:
        source = self.fragment()["sources"][0]
        staged_name = source["preprocess_name"]
        image_manifest = {
            "images": [{
                "image_name": "selected.jpg",
                "source_path": f"/tmp/group/{staged_name}",
                "source_id": "temporary",
                "camera_group": "coverage_image",
            }]
        }
        report = {"frames": [{"source_image": f"/tmp/group/{staged_name}"}]}
        attach_raw_source_metadata(report, image_manifest, {staged_name: source})
        selected = image_manifest["images"][0]
        self.assertEqual(selected["source_id"], source["source_id"])
        self.assertEqual(selected["rgbd"]["capture_id"], "ABC-123")
        self.assertEqual(selected["rgbd"]["frame_index"], 0)
        self.assertEqual(report["frames"][0]["capture_frame_index"], 0)
        self.assertEqual(report["frames"][0]["source_kind"], "skanea_rgbd_frame")
        self.assertEqual(report["frames"][0]["role"], "coverage_image")

    @unittest.skipIf(preprocess_video.IMPORT_ERROR is not None, "OpenCV preprocessing dependencies unavailable")
    def test_skanea_passthrough_keeps_blurry_images_above_target_max(self) -> None:
        image_dir = Path(self.temp.name) / "preprocess-input"
        output_dir = Path(self.temp.name) / "preprocess-output"
        image_dir.mkdir()
        output_dir.mkdir()
        image_paths = []
        for index in range(2):
            path = image_dir / f"skanea_{index:06d}.jpg"
            image = preprocess_video.np.zeros((24, 32, 3), dtype="uint8")
            self.assertTrue(preprocess_video.cv2.imwrite(str(path), image))
            image_paths.append(path)
        settings = dict(preprocess_video.PROFILE_DEFAULTS["indoor_room"])
        settings["target_max"] = 1
        settings["min_blur"] = 9999.0

        result = preprocess_video.process_coverage_images(
            image_paths,
            output_dir,
            settings,
            passthrough_image_names={path.name for path in image_paths},
        )

        self.assertEqual(result.saved_count, 2)
        self.assertTrue(all(record.selected_final for record in result.records))
        self.assertTrue(all(record.selected_by == "source_passthrough" for record in result.records))


if __name__ == "__main__":
    unittest.main()
