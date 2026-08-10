"""Configuration helpers for the local controller stack."""

from __future__ import annotations

import os
import sys
from typing import Optional

from controller_common.runpod_gpus import DEFAULT_COLMAP_GPU, DEFAULT_TRAINING_GPU, normalize_gpu_types

DEFAULT_DATABASE_URL = "postgresql://buildvision3d:buildvision3d@localhost:5432/buildvision3d"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def controller_id() -> str:
    return os.environ.get("CONTROLLER_ID", "local-controller")


def poll_interval_seconds() -> float:
    raw_value = os.environ.get("CONTROLLER_POLL_INTERVAL_SECONDS", "2")
    try:
        return max(0.1, float(raw_value))
    except ValueError:
        return 2.0


def fake_stage_duration_seconds() -> float:
    raw_value = os.environ.get("FAKE_STAGE_DURATION_SECONDS", "1")
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return 1.0


def stage_python_bin() -> str:
    return os.environ.get("CONTROLLER_STAGE_PYTHON_BIN", sys.executable)


def default_r2_bucket() -> str:
    return os.environ.get("R2_BUCKET", "buildvision3d-pipeline")


def r2_endpoint() -> Optional[str]:
    return os.environ.get("R2_ENDPOINT")


def default_colmap_provider() -> str:
    return os.environ.get("CONTROLLER_DEFAULT_COLMAP_PROVIDER", "local_fake")


def default_training_provider() -> str:
    return os.environ.get("CONTROLLER_DEFAULT_TRAINING_PROVIDER", "runpod_training")


def runpod_api_key() -> Optional[str]:
    return os.environ.get("RUNPOD_API_KEY")


def runpod_colmap_image() -> str:
    return os.environ.get(
        "RUNPOD_COLMAP_IMAGE",
        "docker.io/blackjokuro/buildvision3d-colmap-gpu:cuda12.4-colmap-r2-runtime-sm75-sm86-sm89-r2",
    )


def controller_repo_url() -> Optional[str]:
    return os.environ.get("CONTROLLER_REPO_URL")


def controller_git_ref() -> str:
    return os.environ.get("CONTROLLER_GIT_REF", "main")


def runpod_colmap_gpu_types() -> list[str]:
    raw_value = os.environ.get("RUNPOD_COLMAP_GPU_TYPES", DEFAULT_COLMAP_GPU)
    return normalize_gpu_types(raw_value.split(","))


def runpod_colmap_cloud_type() -> str:
    return os.environ.get("RUNPOD_COLMAP_CLOUD_TYPE", "COMMUNITY")


def runpod_colmap_container_disk_gb() -> int:
    raw_value = os.environ.get("RUNPOD_COLMAP_CONTAINER_DISK_GB", "80")
    try:
        return max(20, int(raw_value))
    except ValueError:
        return 80


def runpod_colmap_poll_seconds() -> float:
    raw_value = os.environ.get("RUNPOD_COLMAP_POLL_SECONDS", "30")
    try:
        return max(5.0, float(raw_value))
    except ValueError:
        return 30.0


def runpod_colmap_timeout_seconds() -> float:
    raw_value = os.environ.get("RUNPOD_COLMAP_TIMEOUT_SECONDS", "7200")
    try:
        return max(300.0, float(raw_value))
    except ValueError:
        return 7200.0


def runpod_training_image() -> str:
    return os.environ.get(
        "RUNPOD_TRAINING_IMAGE",
        "docker.io/blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu:cuda11.8-pixi-splatfacto-r2-optbuildvision-sm75-sm86-sm89-r3",
    )


def runpod_training_gpu_types() -> list[str]:
    raw_value = os.environ.get("RUNPOD_TRAINING_GPU_TYPES", DEFAULT_TRAINING_GPU)
    return normalize_gpu_types(raw_value.split(","))


def runpod_training_cloud_type() -> str:
    return os.environ.get("RUNPOD_TRAINING_CLOUD_TYPE", "SECURE")


def runpod_training_container_disk_gb() -> int:
    raw_value = os.environ.get("RUNPOD_TRAINING_CONTAINER_DISK_GB", "160")
    try:
        return max(40, int(raw_value))
    except ValueError:
        return 160


def runpod_training_poll_seconds() -> float:
    raw_value = os.environ.get("RUNPOD_TRAINING_POLL_SECONDS", "30")
    try:
        return max(5.0, float(raw_value))
    except ValueError:
        return 30.0


def runpod_training_timeout_seconds() -> float:
    raw_value = os.environ.get("RUNPOD_TRAINING_TIMEOUT_SECONDS", "14400")
    try:
        return max(600.0, float(raw_value))
    except ValueError:
        return 14400.0
