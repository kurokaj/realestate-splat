"""Assemble approved per-location preprocess outputs for COLMAP."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from controller_common.config import default_r2_bucket
from src.realestate_splat.storage import copy_file, sync_directory


def preprocess_output_base_uri(value: str, project_id: str) -> str:
    base = str(value or f"r2://{default_r2_bucket()}/projects/{project_id}/preprocess").rstrip("/")
    while base.endswith("/current"):
        base = base.rsplit("/current", 1)[0].rstrip("/")
    if "/groups/" in base:
        base = base.split("/groups/", 1)[0].rstrip("/")
    return base


def assembled_project_preprocess_uri(project: dict[str, Any]) -> str:
    project_id = str(project.get("id") or "")
    base_uri = preprocess_output_base_uri(
        project.get("preprocess_current_uri") or f"r2://{default_r2_bucket()}/projects/{project_id}/preprocess",
        project_id,
    )
    return f"{base_uri.rstrip('/')}/current"


def parse_group_output_specs(values: list[str]) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for raw in values:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid preprocess group output JSON: {raw}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Preprocess group output must be a JSON object")
        group_key = str(payload.get("group_key") or "")
        output_uri = str(payload.get("output_uri") or "").rstrip("/")
        if not group_key or not output_uri:
            raise ValueError("Preprocess group output requires group_key and output_uri")
        outputs.append({"group_key": group_key, "output_uri": output_uri})
    return outputs


def assemble_preprocess_groups_local(
    *,
    group_outputs: list[dict[str, str]],
    destination_dir: Path,
    endpoint_url: Optional[str],
    project_id: str = "",
    raw_uri: str = "",
) -> str:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if not group_outputs:
        return str(destination_dir)

    with tempfile.TemporaryDirectory(prefix=f"buildvision3d-assemble-preprocess-{project_id or 'project'}-") as temp_dir:
        root = Path(temp_dir)
        frames_dir = destination_dir / "frames_selected"
        frames_dir.mkdir(parents=True, exist_ok=True)
        manifests: list[dict[str, Any]] = []
        capture_reports: list[dict[str, Any]] = []
        source_uris: list[str] = []
        source_groups: list[str] = []

        for index, group in enumerate(group_outputs):
            group_key = str(group.get("group_key") or "")
            output_uri = str(group.get("output_uri") or "").rstrip("/")
            if not group_key or not output_uri:
                raise ValueError("Preprocess assembly group outputs require group_key and output_uri")
            group_dir = root / f"group_{index:03d}"
            sync_directory(output_uri, group_dir, endpoint_url=endpoint_url)
            source_uris.append(output_uri)
            source_groups.append(group_key)
            copy_group_frames(group_dir / "frames_selected", frames_dir, group_key)
            manifest = load_optional_local_json(group_dir / "image_manifest.json")
            if manifest:
                manifests.append(manifest)
            capture_report = load_optional_local_json(group_dir / "capture_report.json")
            if capture_report:
                capture_reports.append(capture_report)

        image_manifest = merge_image_manifests(manifests)
        frame_count = sum(1 for path in frames_dir.iterdir() if path.is_file())
        image_count = len(image_manifest.get("images", []))
        if frame_count == 0:
            raise ValueError("Assembled preprocess/current has no frames_selected files")
        if image_count == 0:
            raise ValueError("Assembled preprocess/current has no image_manifest images")
        write_local_json(destination_dir / "image_manifest.json", image_manifest)
        if capture_reports:
            write_local_json(destination_dir / "capture_report.json", merge_capture_reports(capture_reports, source_uris))
        write_local_json(
            destination_dir / "preprocess_summary.json",
            {
                "schema_version": 1,
                "project_id": project_id,
                "assembled_locally": True,
                "source_group_count": len(source_groups),
                "source_groups": source_groups,
                "source_uris": source_uris,
                "frame_count": frame_count,
                "image_manifest_count": image_count,
            },
        )
        if raw_uri:
            try:
                copy_file(f"{raw_uri.rstrip('/')}/sources_manifest.json", destination_dir / "sources_manifest.json", endpoint_url=endpoint_url)
            except Exception:
                pass
    return str(destination_dir)


def copy_group_frames(source_dir: Path, destination_dir: Path, group_key: str) -> None:
    if not source_dir.exists():
        raise ValueError(f"Approved preprocess output is missing frames_selected for {group_key}")
    for source in sorted(source_dir.iterdir()):
        if not source.is_file():
            continue
        destination = destination_dir / source.name
        if destination.exists():
            raise ValueError(f"Duplicate preprocessed frame name while assembling COLMAP input: {source.name}")
        shutil.copy2(source, destination)


def load_optional_local_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse assembled preprocess artifact: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Assembled preprocess artifact must be a JSON object: {path.name}")
    return payload


def write_local_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def merge_image_manifests(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    camera_groups_by_id: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for image in manifest.get("images", []) if isinstance(manifest.get("images"), list) else []:
            if isinstance(image, dict):
                images.append(dict(image))
        for group in manifest.get("camera_groups", []) if isinstance(manifest.get("camera_groups"), list) else []:
            if not isinstance(group, dict):
                continue
            group_id = str(group.get("id") or group.get("camera_group") or "")
            if group_id:
                camera_groups_by_id[group_id] = dict(group)
    return {
        "schema_version": 1,
        "images": images,
        "camera_groups": sorted(camera_groups_by_id.values(), key=lambda item: str(item.get("id") or "")),
    }


def merge_capture_reports(reports: list[dict[str, Any]], source_uris: list[str]) -> dict[str, Any]:
    videos = [video for report in reports for video in report.get("videos", []) if isinstance(video, dict)]
    frames = [frame for report in reports for frame in report.get("frames", []) if isinstance(frame, dict)]
    hero_images = [hero for report in reports for hero in report.get("hero_images", []) if isinstance(hero, dict)]
    return {
        "schema_version": 1,
        "assembled": True,
        "source_uris": source_uris,
        "videos": videos,
        "frames": frames,
        "hero_images": hero_images,
        "summary": {
            "selected_frame_count": len([frame for frame in frames if frame.get("output_file")]),
            "video_count": len(videos),
            "hero_image_count": len(hero_images),
        },
    }
