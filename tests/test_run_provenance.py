from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from controller_common.run_provenance import build_colmap_run_provenance, compact_colmap_provenance


class RunProvenanceTests(unittest.TestCase):
    def test_colmap_provenance_fingerprints_effective_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "image_manifest.json"
            manifest_bytes = b'{"schema_version":1,"images":[{"image_name":"000001.jpg"}]}\n'
            manifest.write_bytes(manifest_bytes)

            provenance = build_colmap_run_provenance(
                project_id="first_lidar_test",
                stage_run_id="run-123",
                status="completed",
                started_at="2026-09-30T10:00:00+00:00",
                finished_at="2026-09-30T10:05:00+00:00",
                input_uri="r2://bucket/preprocess/current",
                preprocess_group_outputs=[
                    {
                        "group_key": "room",
                        "output_uri": "r2://bucket/groups/room/current",
                        "stage_run_id": "preprocess-run-456",
                    }
                ],
                image_manifest_path=manifest,
                image_count=1,
                output_uri="r2://bucket/colmap",
                container_image="example/colmap:latest",
                provider_api_version="v2",
                repository_commit="abc123",
                colmap_runtime={"summary": "COLMAP 4.0.4", "output": "COLMAP 4.0.4"},
                stage_command=["python3", "scripts/run_colmap.py"],
                reconstruction_report={
                    "settings": {
                        "mode": "global",
                        "matcher": "sequential",
                        "feature_extractor": "SIFT",
                        "matching_type": "SIFT_BRUTEFORCE",
                        "camera_model": "SIMPLE_RADIAL",
                    },
                    "commands": [{"name": "feature_extractor"}],
                    "selected_sparse_model": "colmap/sparse/0",
                },
            )

        self.assertEqual(
            provenance["input"]["image_manifest"]["sha256"],
            hashlib.sha256(manifest_bytes).hexdigest(),
        )
        self.assertEqual(provenance["runtime"]["provider_api_version"], "v2")
        self.assertEqual(
            provenance["input"]["preprocess_group_outputs"][0]["stage_run_id"],
            "preprocess-run-456",
        )
        self.assertEqual(provenance["configuration"]["intrinsics_source"], "colmap_image_reader_estimated")
        self.assertEqual(provenance["configuration"]["matcher"], "sequential")
        compact = compact_colmap_provenance(provenance)
        self.assertEqual(compact["colmap_version"], "COLMAP 4.0.4")
        self.assertEqual(compact["input_image_count"], 1)
        self.assertEqual(compact["repository_commit"], "abc123")


if __name__ == "__main__":
    unittest.main()
