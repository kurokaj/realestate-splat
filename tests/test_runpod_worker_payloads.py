from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from controller_worker.main import (
    build_runpod_colmap_pod_payload,
    build_training_stage_shell_command,
    guarded_runpod_stage_shell_command,
)


class WorkerRunpodPayloadTests(unittest.TestCase):
    def test_colmap_worker_maps_ui_inputs_to_v2_payload(self) -> None:
        environment = {
            "AWS_ACCESS_KEY_ID": "access",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "R2_BUCKET": "bucket",
            "R2_ENDPOINT": "https://example.invalid",
        }
        with patch.dict(os.environ, environment, clear=False):
            payload = build_runpod_colmap_pod_payload(
                {"project_id": "project", "id": "run-123"},
                {
                    "gpu_type_ids": ["RTX A4500"],
                    "gpu_count": 1,
                    "cloud_type": "COMMUNITY",
                    "container_disk_gb": 30,
                    "min_vcpu_per_gpu": 6,
                    "min_ram_per_gpu": 24,
                },
                image="example/colmap:immutable-tag",
                remote_command="python3 scripts/run_colmap_stage.py",
            )

        self.assertEqual(payload["gpu"]["id"], "NVIDIA RTX A4500")
        self.assertEqual(payload["gpu"]["count"], 1)
        self.assertEqual(payload["gpu"]["minVcpuCountPerGpu"], 6)
        self.assertEqual(payload["gpu"]["minRamPerGpu"], 24)
        self.assertEqual(payload["image"], "example/colmap:immutable-tag")
        self.assertEqual(payload["disk"], 30)
        self.assertEqual(payload["cloud"], "COMMUNITY")
        self.assertEqual(payload["ports"], [])
        self.assertFalse(payload["startSsh"])
        self.assertFalse(payload["startJupyter"])
        self.assertNotIn("gpuTypeIds", payload)

    def test_remote_guard_does_not_turn_a_failed_restart_into_success_wait(self) -> None:
        shell_command = guarded_runpod_stage_shell_command(
            stage_run={"id": "colmap-run-123"},
            repo_url="https://example.invalid/repository.git",
            git_ref="main",
            command=["python3", "scripts/run_colmap_stage.py"],
            stage_label="COLMAP",
        )

        self.assertIn("colmap-run-123.completed", shell_command)
        self.assertIn("colmap-run-123.failed", shell_command)
        self.assertIn("already failed; exiting again for controller cleanup", shell_command)
        self.assertIn("stage failed with exit code $stage_exit", shell_command)
        failed_restart_line = next(
            line for line in shell_command.splitlines() if "already failed" in line
        )
        self.assertNotIn("sleep infinity", failed_restart_line)

    def test_training_command_passes_hybrid_camera_artifact(self) -> None:
        with patch.dict(
            os.environ,
            {"CONTROLLER_REPO_URL": "https://example.invalid/repository.git", "CONTROLLER_GIT_REF": "main"},
            clear=False,
        ):
            command = build_training_stage_shell_command(
                {"project_id": "project", "id": "training-123"},
                {
                    "preprocess_uri": "r2://bucket/preprocess/current",
                    "colmap_uri": "r2://bucket/colmap/runs/run-a",
                    "output_uri": "r2://bucket/training/a-hybrid",
                    "camera_source": "arkit_hybrid",
                    "colmap_source_run_id": "run-a",
                    "hybrid_camera_set_uri": "r2://bucket/analyses/hybrid.json",
                },
            )

        self.assertIn("--camera-source arkit_hybrid", command)
        self.assertIn("--colmap-source-run-id run-a", command)
        self.assertIn("--hybrid-camera-set-uri r2://bucket/analyses/hybrid.json", command)

    def test_training_command_passes_lidar_initialization_artifact(self) -> None:
        with patch.dict(
            os.environ,
            {"CONTROLLER_REPO_URL": "https://example.invalid/repository.git", "CONTROLLER_GIT_REF": "main"},
            clear=False,
        ):
            command = build_training_stage_shell_command(
                {"project_id": "project", "id": "training-123"},
                {
                    "preprocess_uri": "r2://bucket/preprocess/current",
                    "colmap_uri": "r2://bucket/colmap/runs/run-a",
                    "output_uri": "r2://bucket/training/a-hybrid-lidar",
                    "camera_source": "arkit_hybrid",
                    "colmap_source_run_id": "run-a",
                    "hybrid_camera_set_uri": "r2://bucket/analyses/hybrid.json",
                    "lidar_initialization_uri": "r2://bucket/analyses/lidar.json",
                    "merge_colmap_initialization": True,
                },
            )

        self.assertIn("--lidar-initialization-uri r2://bucket/analyses/lidar.json", command)
        self.assertIn("--merge-colmap-initialization", command)


if __name__ == "__main__":
    unittest.main()
