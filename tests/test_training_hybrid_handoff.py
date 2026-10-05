from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_training_stage import (
    build_training_commands,
    training_input_uris,
    validate_nerfstudio_colmap_initialization,
)


class HybridTrainingHandoffTests(unittest.TestCase):
    def test_builds_hybrid_prepare_command_with_immutable_source_identity(self) -> None:
        args = SimpleNamespace(
            pixi_bin="pixi",
            nerfstudio_dir="/opt/nerfstudio",
            python_bin="python3",
            num_downscales=1,
            prepare_with_pixi=False,
            camera_source="arkit_hybrid",
            colmap_source_run_id="colmap-run-a",
            project_id="project",
            experiment_name=None,
            method="splatfacto",
            max_steps=100,
            save_every=50,
            eval_every=50,
            use_scale_regularization=True,
            train_option=[],
        )

        prepare_command, _train_command = build_training_commands(args, Path("/tmp/training-run"))

        self.assertIn("--hybrid-camera-set", prepare_command)
        self.assertIn("/tmp/training-run/reports/hybrid_camera_set.json", prepare_command)
        self.assertIn("--expected-colmap-run-id", prepare_command)
        self.assertIn("colmap-run-a", prepare_command)

    def test_stage_inputs_include_hybrid_artifact(self) -> None:
        self.assertEqual(
            training_input_uris(
                "r2://bucket/preprocess/current/",
                "r2://bucket/colmap/runs/run-a/",
                "r2://bucket/colmap/analyses/hybrid.json",
            ),
            [
                "r2://bucket/preprocess/current",
                "r2://bucket/colmap/runs/run-a",
                "r2://bucket/colmap/analyses/hybrid.json",
            ],
        )

    def test_preflight_accepts_consistent_hybrid_counts_and_sparse_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            data_dir = run_dir / "nerfstudio"
            data_dir.mkdir()
            (data_dir / "colmap_points3D.ply").write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n",
                encoding="utf-8",
            )
            transforms = {
                "ply_file_path": "colmap_points3D.ply",
                "frames": [{"file_path": "images/a.jpg"}, {"file_path": "images/b.jpg"}],
                "buildvision3d": {
                    "camera_source": "arkit_hybrid",
                    "frame_count": 2,
                    "colmap_registered_count": 1,
                    "arkit_propagated_count": 1,
                    "hybrid_selection_sha256": "a" * 64,
                },
            }
            (data_dir / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")

            validate_nerfstudio_colmap_initialization(run_dir)


if __name__ == "__main__":
    unittest.main()
