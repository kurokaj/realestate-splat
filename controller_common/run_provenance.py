"""Stable provenance helpers for remote reconstruction runs."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_version(command: Sequence[str], *, timeout_seconds: float = 10) -> dict[str, Any]:
    """Capture a bounded, exact runtime version header without failing the stage."""
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return {
            "command": list(command),
            "summary": None,
            "output": None,
            "returncode": None,
            "error": str(exc),
        }
    output = (completed.stdout or "").strip()
    nonempty_lines = [line.strip() for line in output.splitlines() if line.strip()]
    return {
        "command": list(command),
        "summary": nonempty_lines[0] if nonempty_lines else None,
        "output": output[:4096] or None,
        "returncode": completed.returncode,
        "error": None,
    }


def repository_revision(root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return None
    revision = (completed.stdout or "").strip()
    return revision if completed.returncode == 0 and revision else None


def build_colmap_run_provenance(
    *,
    project_id: str,
    stage_run_id: str,
    status: str,
    started_at: str,
    finished_at: str,
    input_uri: str,
    preprocess_group_outputs: Sequence[Mapping[str, Any]],
    image_manifest_path: Path,
    image_count: int,
    output_uri: str,
    container_image: Optional[str],
    provider_api_version: Optional[str],
    repository_commit: Optional[str],
    colmap_runtime: Mapping[str, Any],
    stage_command: Optional[Sequence[str]],
    reconstruction_report: Mapping[str, Any],
) -> dict[str, Any]:
    settings = reconstruction_report.get("settings")
    if not isinstance(settings, Mapping):
        settings = {}
    commands = reconstruction_report.get("commands")
    if not isinstance(commands, list):
        commands = []
    output_base = output_uri.rstrip("/")
    manifest_sha256 = sha256_file(image_manifest_path)
    return {
        "schema_version": 1,
        "stage": "colmap",
        "project_id": project_id,
        "stage_run_id": stage_run_id,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "input": {
            "preprocess_uri": input_uri.rstrip("/"),
            "preprocess_group_outputs": [dict(item) for item in preprocess_group_outputs],
            "image_count": image_count,
            "image_manifest": {
                "name": image_manifest_path.name,
                "sha256": manifest_sha256,
                "size_bytes": image_manifest_path.stat().st_size if image_manifest_path.is_file() else None,
            },
        },
        "runtime": {
            "provider": "runpod",
            "provider_api_version": provider_api_version,
            "container_image": container_image,
            "repository_commit": repository_commit,
            "colmap": dict(colmap_runtime),
        },
        "configuration": {
            "mapper_mode": settings.get("mode"),
            "feature_extractor": settings.get("feature_extractor"),
            "matcher": settings.get("matcher"),
            "matching_type": settings.get("matching_type"),
            "camera_model": settings.get("camera_model"),
            "intrinsics_source": settings.get("intrinsics_source") or "colmap_image_reader_estimated",
            "effective_settings": dict(settings),
            "stage_command": list(stage_command or []),
            "colmap_commands": commands,
        },
        "outputs": {
            "current_uri": f"{output_base}/current",
            "history_uri": f"{output_base}/runs/{stage_run_id}",
            "selected_sparse_model": reconstruction_report.get("selected_sparse_model"),
            "reconstruction_report_uri": f"{output_base}/current/reconstruction_report.json",
            "run_provenance_uri": f"{output_base}/current/run_provenance.json",
        },
    }


def compact_colmap_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    input_record = provenance.get("input") if isinstance(provenance.get("input"), Mapping) else {}
    manifest = input_record.get("image_manifest") if isinstance(input_record.get("image_manifest"), Mapping) else {}
    runtime = provenance.get("runtime") if isinstance(provenance.get("runtime"), Mapping) else {}
    colmap = runtime.get("colmap") if isinstance(runtime.get("colmap"), Mapping) else {}
    configuration = provenance.get("configuration") if isinstance(provenance.get("configuration"), Mapping) else {}
    outputs = provenance.get("outputs") if isinstance(provenance.get("outputs"), Mapping) else {}
    return {
        "runpod_api_version": runtime.get("provider_api_version"),
        "colmap_version": colmap.get("summary"),
        "repository_commit": runtime.get("repository_commit"),
        "container_image": runtime.get("container_image"),
        "input_manifest_sha256": manifest.get("sha256"),
        "input_image_count": input_record.get("image_count"),
        "camera_model": configuration.get("camera_model"),
        "intrinsics_source": configuration.get("intrinsics_source"),
        "run_provenance_uri": outputs.get("run_provenance_uri"),
    }
