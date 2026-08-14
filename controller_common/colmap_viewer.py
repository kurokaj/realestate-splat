"""Helpers for browser-friendly sparse COLMAP viewer artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.prepare_nerfstudio_from_colmap import (
    colmap_pose_to_nerfstudio_transform,
    qvec_to_rotmat,
    read_cameras,
    read_images,
)

ROLE_CAMERA_COLORS: dict[str, dict[str, list[int]]] = {
    "coverage": {
        "stroke": [255, 204, 96],
        "fill": [255, 221, 133],
    },
    "hero": {
        "stroke": [90, 214, 130],
        "fill": [128, 234, 160],
    },
    "unknown": {
        "stroke": [163, 178, 194],
        "fill": [192, 204, 217],
    },
}


def useful_lines(path: Path) -> Iterable[str]:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            yield line


def read_points3d(path: Path) -> list[dict[str, Any]]:
    points = []
    for line in useful_lines(path):
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            points.append(
                {
                    "id": int(parts[0]),
                    "xyz": [float(parts[1]), float(parts[2]), float(parts[3])],
                    "rgb": [int(parts[4]), int(parts[5]), int(parts[6])],
                    "error": float(parts[7]) if len(parts) > 7 else None,
                }
            )
        except ValueError:
            continue
    return points


def convert_xyz_to_nerfstudio_axes(xyz: Sequence[float]) -> list[float]:
    x, y, z = xyz
    return [x, z, -y]


def build_sparse_viewer_payload(sparse_txt_dir: Path, *, max_points: int = 25000) -> dict[str, Any]:
    cameras = read_cameras(sparse_txt_dir / "cameras.txt")
    images = read_images(sparse_txt_dir / "images.txt")
    points = read_points3d(sparse_txt_dir / "points3D.txt")
    manifest_by_name = load_manifest_by_name(sparse_txt_dir)

    sampled_points = sample_points(points, max_points=max_points)
    converted_points = [
        {
            "position": round_vector(convert_xyz_to_nerfstudio_axes(point["xyz"])),
            "color": point["rgb"],
        }
        for point in sampled_points
    ]
    camera_rows = [camera_row(image, manifest_by_name.get(image.name)) for image in images]
    bounds = scene_bounds([point["position"] for point in converted_points], [camera["position"] for camera in camera_rows])
    return {
        "schema_version": 1,
        "point_count": len(points),
        "point_sample_count": len(converted_points),
        "camera_count": len(camera_rows),
        "bounds": bounds,
        "points": converted_points,
        "cameras": camera_rows,
        "camera_models": [
            {"camera_id": camera.camera_id, "model": camera.model, "width": camera.width, "height": camera.height}
            for camera in cameras.values()
        ],
    }


def load_manifest_by_name(sparse_txt_dir: Path) -> dict[str, dict[str, Any]]:
    reports_dir = sparse_txt_dir.parents[2] / "reports"
    manifest_path = reports_dir / "image_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    images = manifest.get("images") or []
    if not isinstance(images, list):
        return {}
    by_name: dict[str, dict[str, Any]] = {}
    for entry in images:
        if not isinstance(entry, dict):
            continue
        image_name = entry.get("image_name")
        if image_name:
            by_name[str(image_name)] = dict(entry)
    return by_name


def camera_colors(role: str) -> dict[str, list[int]]:
    return ROLE_CAMERA_COLORS.get(role, ROLE_CAMERA_COLORS["unknown"])


def camera_row(image: Any, manifest_entry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    transform = colmap_pose_to_nerfstudio_transform(image)
    rotation = qvec_to_rotmat(image.qvec)
    forward_cv = [-rotation[2][0], -rotation[2][1], -rotation[2][2]]
    up_cv = [-rotation[1][0], -rotation[1][1], -rotation[1][2]]
    role = str((manifest_entry or {}).get("role") or "unknown")
    colors = camera_colors(role)
    return {
        "image_id": image.image_id,
        "name": image.name,
        "role": role,
        "camera_group": (manifest_entry or {}).get("camera_group"),
        "position": round_vector([transform[0][3], transform[1][3], transform[2][3]]),
        "forward": round_vector(convert_xyz_to_nerfstudio_axes(forward_cv)),
        "up": round_vector(convert_xyz_to_nerfstudio_axes(up_cv)),
        "stroke_color": colors["stroke"],
        "fill_color": colors["fill"],
    }


def sample_points(points: list[dict[str, Any]], *, max_points: int) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    return points[::step][:max_points]


def round_vector(values: Sequence[float], *, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def scene_bounds(point_positions: list[Sequence[float]], camera_positions: list[Sequence[float]]) -> dict[str, Any]:
    values = point_positions + camera_positions
    if not values:
        return {"center": [0.0, 0.0, 0.0], "radius": 1.0}
    mins = [min(vector[index] for vector in values) for index in range(3)]
    maxs = [max(vector[index] for vector in values) for index in range(3)]
    center = [(mins[index] + maxs[index]) / 2.0 for index in range(3)]
    radius = max(
        1.0,
        max(math.dist(vector, center) for vector in values),
    )
    return {"center": round_vector(center), "radius": round(radius, 6)}


def write_sparse_viewer_payload(sparse_txt_dir: Path, output_path: Path, *, max_points: int = 25000) -> Path:
    payload = build_sparse_viewer_payload(sparse_txt_dir, max_points=max_points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
