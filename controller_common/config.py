"""Configuration helpers for the local controller stack."""

from __future__ import annotations

import os
import sys
from typing import Optional


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
