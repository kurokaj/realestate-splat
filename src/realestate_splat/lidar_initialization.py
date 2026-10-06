"""Build deterministic, RGB-colored LiDAR seed points for Splatfacto."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .depth_diagnostic import (
    DEFAULT_MAX_DEPTH_METERS,
    DEFAULT_MIN_DEPTH_METERS,
    DEFAULT_OUTPUT_VOXEL_METERS,
    DEFAULT_PIXEL_STRIDE,
    HIGH_CONFIDENCE_VALUE,
)


DEFAULT_MIN_FRAME_SUPPORT = 2
DEFAULT_MAX_PREVIEW_POINTS = 75_000

AttachmentLoader = Callable[[str], bytes]


def build_lidar_initialization(
    hybrid_camera_set: Mapping[str, Any],
    attachment_loader: AttachmentLoader,
    *,
    min_depth_meters: float = DEFAULT_MIN_DEPTH_METERS,
    max_depth_meters: float = DEFAULT_MAX_DEPTH_METERS,
    pixel_stride: int = DEFAULT_PIXEL_STRIDE,
    voxel_meters: float = DEFAULT_OUTPUT_VOXEL_METERS,
    min_frame_support: int = DEFAULT_MIN_FRAME_SUPPORT,
    max_preview_points: int = DEFAULT_MAX_PREVIEW_POINTS,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return metadata plus filtered points/colors in Nerfstudio's +Z-up world.

    The first production slice intentionally accepts one continuous ARKit
    capture. A single capture may span many rooms because every frame shares
    one gravity-aligned ARKit world. Independently started captures require a
    scene-assembly transform before their metric clouds can be merged.
    """
    if hybrid_camera_set.get("status") != "completed":
        raise ValueError("LiDAR initialization requires a completed A_hybrid camera set")
    captures = [item for item in hybrid_camera_set.get("captures") or [] if isinstance(item, Mapping)]
    if len(captures) != 1:
        raise ValueError(
            "LiDAR initialization currently requires exactly one continuous ARKit capture; "
            "independent captures must first be joined by the future scene-assembly workflow"
        )
    if min_depth_meters <= 0 or max_depth_meters <= min_depth_meters:
        raise ValueError("Depth range must be positive and increasing")
    if pixel_stride < 1 or voxel_meters <= 0 or min_frame_support < 1 or max_preview_points < 1:
        raise ValueError("Stride, voxel size, frame support, and preview budget must be positive")

    settings = {
        "confidence_value": HIGH_CONFIDENCE_VALUE,
        "min_depth_meters": min_depth_meters,
        "max_depth_meters": max_depth_meters,
        "pixel_stride": pixel_stride,
        "voxel_meters": voxel_meters,
        "min_frame_support": min_frame_support,
        "max_preview_points": max_preview_points,
    }
    capture = captures[0]
    capture_id = str(capture.get("capture_id") or "")
    voxels: dict[tuple[int, int, int], dict[str, Any]] = {}
    frame_rows: list[dict[str, Any]] = []
    skipped_frames: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []

    for raw_frame in capture.get("frames") or []:
        if not isinstance(raw_frame, Mapping):
            continue
        frame = dict(raw_frame)
        fingerprint = {
            "capture_id": capture_id,
            "frame_index": frame.get("frame_index"),
            "image_name": frame.get("image_name"),
            "pose_source": frame.get("pose_source"),
            "depth_uri": _attachment_uri(frame, "depth"),
            "confidence_uri": _attachment_uri(frame, "confidence"),
            "rgb_uri": frame.get("rgb_uri"),
        }
        fingerprints.append(fingerprint)
        try:
            processed = _process_frame(
                frame,
                attachment_loader,
                min_depth_meters=min_depth_meters,
                max_depth_meters=max_depth_meters,
                pixel_stride=pixel_stride,
            )
        except (ImportError, OSError, ValueError, KeyError) as exc:
            skipped_frames.append(
                {
                    "capture_id": capture_id,
                    "frame_index": frame.get("frame_index"),
                    "image_name": frame.get("image_name"),
                    "reason": str(exc),
                }
            )
            continue
        points = processed.pop("points")
        colors = processed.pop("colors")
        _merge_frame_voxels(voxels, points, colors, voxel_size=voxel_meters)
        frame_rows.append(processed)

    if not frame_rows:
        reasons = "; ".join(item["reason"] for item in skipped_frames[:3])
        raise ValueError(f"No RGB-D frames could be processed{': ' + reasons if reasons else ''}")

    retained_keys = sorted(key for key, row in voxels.items() if int(row["frame_support"]) >= min_frame_support)
    if not retained_keys:
        raise ValueError("No LiDAR voxels survived the minimum cross-frame support filter")
    points_arkit = np.asarray(
        [np.asarray(voxels[key]["position_sum"], dtype=np.float64) / int(voxels[key]["sample_count"]) for key in retained_keys],
        dtype=np.float64,
    )
    colors = np.asarray(
        [
            np.clip(
                np.rint(np.asarray(voxels[key]["color_sum"], dtype=np.float64) / int(voxels[key]["sample_count"])),
                0,
                255,
            )
            for key in retained_keys
        ],
        dtype=np.uint8,
    )
    # ARKit world is right-handed with +Y up. Nerfstudio is right-handed with
    # +Z up. This fixed rotation maps (x, y, z) -> (x, -z, y).
    points_nerfstudio = np.column_stack(
        (points_arkit[:, 0], -points_arkit[:, 2], points_arkit[:, 1])
    )
    if not np.all(np.isfinite(points_nerfstudio)):
        raise ValueError("Filtered LiDAR initialization contains non-finite points")

    support_counts = Counter(int(voxels[key]["frame_support"]) for key in retained_keys)
    preview_indexes = _deterministic_preview_indexes(retained_keys, max_preview_points)
    preview_points = [
        {
            "position": [round(float(value), 6) for value in points_nerfstudio[index]],
            "color": [int(value) for value in colors[index]],
            "frame_support": int(voxels[retained_keys[index]]["frame_support"]),
            "sample_count": int(voxels[retained_keys[index]]["sample_count"]),
        }
        for index in preview_indexes
    ]
    fingerprint_payload = {
        "builder_schema_version": 1,
        "source_run_id": hybrid_camera_set.get("source_run_id"),
        "source_hybrid_selection_sha256": hybrid_camera_set.get("selection_sha256"),
        "capture_id": capture_id,
        "settings": settings,
        "frames": fingerprints,
    }
    selection_sha256 = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact = {
        "schema_version": 1,
        "kind": "buildvision3d_lidar_training_initialization",
        "status": "completed_with_skips" if skipped_frames else "completed",
        "source_run_id": hybrid_camera_set.get("source_run_id"),
        "source_hybrid_selection_sha256": hybrid_camera_set.get("selection_sha256"),
        "selection_sha256": selection_sha256,
        "capture_id": capture_id,
        "settings": settings,
        "coordinate_convention": "Nerfstudio/OpenGL world in meters; +Z up; camera looks -Z",
        "initialization_policy": "LiDAR-only surface seeds; no unsupported COLMAP sparse points",
        "frame_count": len(capture.get("frames") or []),
        "processed_frame_count": len(frame_rows),
        "skipped_frame_count": len(skipped_frames),
        "skipped_frames": skipped_frames,
        "sampled_point_count": sum(int(row["sampled_point_count"]) for row in frame_rows),
        "unfiltered_voxel_count": len(voxels),
        "filtered_voxel_count": len(retained_keys),
        "removed_low_support_voxel_count": len(voxels) - len(retained_keys),
        "frame_support_distribution": {str(key): value for key, value in sorted(support_counts.items())},
        "point_cloud_stats": _point_cloud_stats(points_nerfstudio),
        "preview_point_count": len(preview_points),
        "preview_points": preview_points,
    }
    return artifact, points_nerfstudio, colors


def write_lidar_initialization_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape != (len(colors), 3) or colors.shape != (len(points), 3):
        raise ValueError("LiDAR point and color arrays must both be Nx3")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def compact_lidar_initialization_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": artifact.get("schema_version"),
        "kind": artifact.get("kind"),
        "status": artifact.get("status"),
        "source_run_id": artifact.get("source_run_id"),
        "source_hybrid_selection_sha256": artifact.get("source_hybrid_selection_sha256"),
        "selection_sha256": artifact.get("selection_sha256"),
        "capture_id": artifact.get("capture_id"),
        "settings": artifact.get("settings"),
        "frame_count": artifact.get("frame_count"),
        "processed_frame_count": artifact.get("processed_frame_count"),
        "skipped_frame_count": artifact.get("skipped_frame_count"),
        "sampled_point_count": artifact.get("sampled_point_count"),
        "unfiltered_voxel_count": artifact.get("unfiltered_voxel_count"),
        "filtered_voxel_count": artifact.get("filtered_voxel_count"),
        "removed_low_support_voxel_count": artifact.get("removed_low_support_voxel_count"),
        "frame_support_distribution": artifact.get("frame_support_distribution"),
        "point_cloud_stats": artifact.get("point_cloud_stats"),
        "preview_point_count": artifact.get("preview_point_count"),
        "artifact_uri": artifact.get("artifact_uri"),
        "ply_uri": artifact.get("ply_uri"),
        "preview_uri": artifact.get("preview_uri"),
    }


def _process_frame(
    frame: Mapping[str, Any],
    attachment_loader: AttachmentLoader,
    *,
    min_depth_meters: float,
    max_depth_meters: float,
    pixel_stride: int,
) -> dict[str, Any]:
    depth_meta = frame.get("depth") if isinstance(frame.get("depth"), Mapping) else {}
    confidence_meta = frame.get("confidence") if isinstance(frame.get("confidence"), Mapping) else {}
    depth_uri = str(depth_meta.get("uri") or "")
    confidence_uri = str(confidence_meta.get("uri") or "")
    rgb_uri = str(frame.get("rgb_uri") or "")
    if not depth_uri or not confidence_uri or not rgb_uri:
        raise ValueError("missing RGB, depth, or confidence URI")
    depth_width = _positive_int(depth_meta.get("width"), "depth width")
    depth_height = _positive_int(depth_meta.get("height"), "depth height")
    if (depth_width, depth_height) != (
        _positive_int(confidence_meta.get("width"), "confidence width"),
        _positive_int(confidence_meta.get("height"), "confidence height"),
    ):
        raise ValueError("depth and confidence dimensions do not match")
    rgb_width = _positive_int(frame.get("width"), "RGB width")
    rgb_height = _positive_int(frame.get("height"), "RGB height")
    intrinsics = np.asarray(frame.get("camera_intrinsics") or [], dtype=np.float64)
    transform = np.asarray(frame.get("camera_to_world_reference_arkit_row_major") or [], dtype=np.float64)
    if intrinsics.size != 9 or not np.all(np.isfinite(intrinsics)):
        raise ValueError("camera intrinsics must contain nine finite values")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("reference camera transform must be a finite 4x4 matrix")

    expected_pixels = depth_width * depth_height
    depth_bytes = attachment_loader(depth_uri)
    confidence_bytes = attachment_loader(confidence_uri)
    if len(depth_bytes) != expected_pixels * 4 or len(confidence_bytes) != expected_pixels:
        raise ValueError("depth or confidence payload has an unexpected byte size")
    depth = np.frombuffer(depth_bytes, dtype="<f4").reshape((depth_height, depth_width))
    confidence = np.frombuffer(confidence_bytes, dtype=np.uint8).reshape((depth_height, depth_width))
    valid = (confidence == HIGH_CONFIDENCE_VALUE) & np.isfinite(depth)
    valid &= (depth >= min_depth_meters) & (depth <= max_depth_meters)
    sampled = np.zeros_like(valid)
    sampled[::pixel_stride, ::pixel_stride] = valid[::pixel_stride, ::pixel_stride]
    rows, columns = np.nonzero(sampled)
    selected_depth = depth[rows, columns].astype(np.float64)

    scale_x = depth_width / rgb_width
    scale_y = depth_height / rgb_height
    fx = float(intrinsics[0] * scale_x)
    fy = float(intrinsics[4] * scale_y)
    cx = float(intrinsics[6] * scale_x)
    cy = float(intrinsics[7] * scale_y)
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError("scaled depth intrinsics are invalid")
    camera_points = np.column_stack(
        (
            (columns.astype(np.float64) - cx) * selected_depth / fx,
            (cy - rows.astype(np.float64)) * selected_depth / fy,
            -selected_depth,
        )
    )
    points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    rgb = _decode_rgb(attachment_loader(rgb_uri))
    if rgb.shape[:2] != (rgb_height, rgb_width):
        raise ValueError(
            f"RGB dimensions do not match metadata: decoded={rgb.shape[1]}x{rgb.shape[0]}, "
            f"declared={rgb_width}x{rgb_height}"
        )
    rgb_rows = np.clip(np.rint((rows + 0.5) / scale_y - 0.5).astype(np.int64), 0, rgb_height - 1)
    rgb_columns = np.clip(np.rint((columns + 0.5) / scale_x - 0.5).astype(np.int64), 0, rgb_width - 1)
    colors = rgb[rgb_rows, rgb_columns]
    return {
        "capture_id": frame.get("capture_id"),
        "frame_index": int(frame.get("frame_index") or 0),
        "image_name": frame.get("image_name"),
        "pose_source": frame.get("pose_source"),
        "sampled_point_count": int(len(points)),
        "points": points,
        "colors": colors,
    }


def _decode_rgb(data: bytes) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("OpenCV is required to color LiDAR initialization points") from exc
    encoded = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("RGB image could not be decoded")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _merge_frame_voxels(
    destination: dict[tuple[int, int, int], dict[str, Any]],
    points: np.ndarray,
    colors: np.ndarray,
    *,
    voxel_size: float,
) -> None:
    if not len(points):
        return
    keys = np.floor(points / voxel_size).astype(np.int64)
    unique_keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    position_sums = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis], minlength=len(unique_keys)) for axis in range(3)]
    )
    color_sums = np.column_stack(
        [np.bincount(inverse, weights=colors[:, axis], minlength=len(unique_keys)) for axis in range(3)]
    )
    for key_values, position_sum, color_sum, count in zip(unique_keys, position_sums, color_sums, counts):
        key = tuple(int(value) for value in key_values)
        row = destination.setdefault(
            key,
            {
                "position_sum": np.zeros(3, dtype=np.float64),
                "color_sum": np.zeros(3, dtype=np.float64),
                "sample_count": 0,
                "frame_support": 0,
            },
        )
        row["position_sum"] += position_sum
        row["color_sum"] += color_sum
        row["sample_count"] += int(count)
        row["frame_support"] += 1


def _deterministic_preview_indexes(keys: Sequence[tuple[int, int, int]], max_points: int) -> list[int]:
    if len(keys) <= max_points:
        return list(range(len(keys)))
    ranked = sorted(
        range(len(keys)),
        key=lambda index: hashlib.sha256(f"{keys[index][0]},{keys[index][1]},{keys[index][2]}".encode()).digest(),
    )
    return sorted(ranked[:max_points])


def _point_cloud_stats(points: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(len(points)),
        "xyz_min": [round(float(value), 9) for value in np.min(points, axis=0)],
        "xyz_max": [round(float(value), 9) for value in np.max(points, axis=0)],
        "reprojection_error_median": None,
    }


def _attachment_uri(frame: Mapping[str, Any], key: str) -> str | None:
    value = frame.get(key)
    return str(value.get("uri")) if isinstance(value, Mapping) and value.get("uri") else None


def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number
