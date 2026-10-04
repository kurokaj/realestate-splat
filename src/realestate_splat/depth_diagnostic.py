"""High-confidence Skanea depth diagnostics for an immutable hybrid camera set."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np


DEFAULT_MIN_DEPTH_METERS = 0.15
DEFAULT_MAX_DEPTH_METERS = 8.0
DEFAULT_PIXEL_STRIDE = 2
DEFAULT_OUTPUT_VOXEL_METERS = 0.03
DEFAULT_AGREEMENT_VOXEL_METERS = 0.10
DEFAULT_MAX_VIEWER_POINTS = 150_000
HIGH_CONFIDENCE_VALUE = 2

AttachmentLoader = Callable[[str], bytes]


def build_depth_diagnostic(
    hybrid_camera_set: Mapping[str, Any],
    attachment_loader: AttachmentLoader,
    *,
    min_depth_meters: float = DEFAULT_MIN_DEPTH_METERS,
    max_depth_meters: float = DEFAULT_MAX_DEPTH_METERS,
    pixel_stride: int = DEFAULT_PIXEL_STRIDE,
    output_voxel_meters: float = DEFAULT_OUTPUT_VOXEL_METERS,
    agreement_voxel_meters: float = DEFAULT_AGREEMENT_VOXEL_METERS,
    max_viewer_points: int = DEFAULT_MAX_VIEWER_POINTS,
) -> dict[str, Any]:
    """Back-project high-confidence depth through the selected hybrid poses.

    The returned cloud is a bounded diagnostic derivative. Raw depth and
    confidence objects are read but never modified.
    """
    if hybrid_camera_set.get("status") not in {"completed", "completed_review_required"}:
        raise ValueError("Depth diagnostic requires a completed hybrid camera set")
    if min_depth_meters <= 0 or max_depth_meters <= min_depth_meters:
        raise ValueError("Depth range must be positive and increasing")
    if pixel_stride < 1:
        raise ValueError("Pixel stride must be at least one")
    if output_voxel_meters <= 0 or agreement_voxel_meters <= 0:
        raise ValueError("Voxel sizes must be positive")
    if max_viewer_points < 1:
        raise ValueError("Viewer point budget must be positive")

    settings = {
        "confidence_value": HIGH_CONFIDENCE_VALUE,
        "min_depth_meters": min_depth_meters,
        "max_depth_meters": max_depth_meters,
        "pixel_stride": pixel_stride,
        "output_voxel_meters": output_voxel_meters,
        "agreement_voxel_meters": agreement_voxel_meters,
        "max_viewer_points": max_viewer_points,
    }
    output_voxels: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    agreement_cells: dict[tuple[int, int, int], dict[str, Any]] = {}
    capture_rows: list[dict[str, Any]] = []
    skipped_frames: list[dict[str, Any]] = []
    source_stats: dict[str, dict[str, int]] = defaultdict(_empty_source_stats)
    all_frame_fingerprints: list[dict[str, Any]] = []

    for capture in hybrid_camera_set.get("captures") or []:
        if not isinstance(capture, Mapping):
            continue
        capture_id = str(capture.get("capture_id") or "")
        frame_rows: list[dict[str, Any]] = []
        for frame in capture.get("frames") or []:
            if not isinstance(frame, Mapping):
                continue
            frame_index = int(frame.get("frame_index") or 0)
            image_name = str(frame.get("image_name") or "")
            pose_source = str(frame.get("pose_source") or "unknown")
            depth_meta = frame.get("depth") if isinstance(frame.get("depth"), Mapping) else {}
            confidence_meta = frame.get("confidence") if isinstance(frame.get("confidence"), Mapping) else {}
            fingerprint = {
                "capture_id": capture_id,
                "frame_index": frame_index,
                "image_name": image_name,
                "pose_source": pose_source,
                "depth_uri": depth_meta.get("uri"),
                "confidence_uri": confidence_meta.get("uri"),
            }
            all_frame_fingerprints.append(fingerprint)
            try:
                processed = _process_frame(
                    frame,
                    attachment_loader,
                    min_depth_meters=min_depth_meters,
                    max_depth_meters=max_depth_meters,
                    pixel_stride=pixel_stride,
                )
            except (OSError, ValueError, KeyError) as exc:
                skipped_frames.append(
                    {
                        "capture_id": capture_id,
                        "frame_index": frame_index,
                        "image_name": image_name,
                        "reason": str(exc),
                    }
                )
                continue

            points = processed.pop("points")
            frame_rows.append(processed)
            stats = source_stats[pose_source]
            stats["frame_count"] += 1
            for key in (
                "depth_sample_count",
                "high_confidence_sample_count",
                "valid_high_confidence_sample_count",
                "sampled_point_count",
            ):
                stats[key] += int(processed[key])
            _merge_output_voxels(
                output_voxels,
                points,
                pose_source=pose_source,
                voxel_size=output_voxel_meters,
            )
            _merge_agreement_cells(
                agreement_cells,
                points,
                pose_source=pose_source,
                voxel_size=agreement_voxel_meters,
            )

        capture_rows.append(
            {
                "capture_id": capture_id,
                "frame_count": len(capture.get("frames") or []),
                "processed_frame_count": len(frame_rows),
                "skipped_frame_count": sum(1 for item in skipped_frames if item["capture_id"] == capture_id),
                "frames": frame_rows,
            }
        )

    processed_frame_count = sum(row["processed_frame_count"] for row in capture_rows)
    if processed_frame_count == 0:
        reasons = "; ".join(item["reason"] for item in skipped_frames[:3])
        raise ValueError(f"No depth frames could be processed{': ' + reasons if reasons else ''}")

    point_rows = _viewer_points(output_voxels, max_points=max_viewer_points)
    overlap = _agreement_statistics(agreement_cells)
    source_summary = {key: dict(value) for key, value in sorted(source_stats.items())}
    fingerprint_payload = {
        "builder_schema_version": 1,
        "source_run_id": hybrid_camera_set.get("source_run_id"),
        "source_hybrid_selection_sha256": hybrid_camera_set.get("selection_sha256"),
        "settings": settings,
        "frames": all_frame_fingerprints,
    }
    selection_sha256 = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": "buildvision3d_high_confidence_depth_diagnostic",
        "status": "completed_with_skips" if skipped_frames else "completed",
        "source_run_id": hybrid_camera_set.get("source_run_id"),
        "source_hybrid_selection_sha256": hybrid_camera_set.get("selection_sha256"),
        "selection_sha256": selection_sha256,
        "settings": settings,
        "coordinate_conventions": {
            "points": "meters in each capture's original gravity-aligned ARKit reference world; +Y up",
            "depth_camera": "+X right, +Y up, camera looks along -Z",
            "intrinsics": "captured-image intrinsics scaled independently into the depth-map dimensions",
        },
        "capture_count": len(capture_rows),
        "frame_count": sum(row["frame_count"] for row in capture_rows),
        "processed_frame_count": processed_frame_count,
        "skipped_frame_count": len(skipped_frames),
        "depth_sample_count": sum(row["depth_sample_count"] for row in source_summary.values()),
        "high_confidence_sample_count": sum(
            row["high_confidence_sample_count"] for row in source_summary.values()
        ),
        "valid_high_confidence_sample_count": sum(
            row["valid_high_confidence_sample_count"] for row in source_summary.values()
        ),
        "sampled_point_count": sum(row["sampled_point_count"] for row in source_summary.values()),
        "voxel_count": len(output_voxels),
        "viewer_point_count": len(point_rows),
        "source_statistics": source_summary,
        "overlap_statistics": overlap,
        "captures": capture_rows,
        "skipped_frames": skipped_frames,
        "points": point_rows,
    }


def compact_depth_diagnostic_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": artifact.get("status"),
        "source_run_id": artifact.get("source_run_id"),
        "source_hybrid_selection_sha256": artifact.get("source_hybrid_selection_sha256"),
        "selection_sha256": artifact.get("selection_sha256"),
        "capture_count": artifact.get("capture_count"),
        "frame_count": artifact.get("frame_count"),
        "processed_frame_count": artifact.get("processed_frame_count"),
        "skipped_frame_count": artifact.get("skipped_frame_count"),
        "depth_sample_count": artifact.get("depth_sample_count"),
        "high_confidence_sample_count": artifact.get("high_confidence_sample_count"),
        "valid_high_confidence_sample_count": artifact.get("valid_high_confidence_sample_count"),
        "sampled_point_count": artifact.get("sampled_point_count"),
        "voxel_count": artifact.get("voxel_count"),
        "viewer_point_count": artifact.get("viewer_point_count"),
        "source_statistics": artifact.get("source_statistics"),
        "overlap_statistics": artifact.get("overlap_statistics"),
        "settings": artifact.get("settings"),
    }


def _empty_source_stats() -> dict[str, int]:
    return {
        "frame_count": 0,
        "depth_sample_count": 0,
        "high_confidence_sample_count": 0,
        "valid_high_confidence_sample_count": 0,
        "sampled_point_count": 0,
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
    if not depth_uri or not confidence_uri:
        raise ValueError("missing depth or confidence URI")
    depth_width = _positive_int(depth_meta.get("width"), "depth width")
    depth_height = _positive_int(depth_meta.get("height"), "depth height")
    confidence_width = _positive_int(confidence_meta.get("width"), "confidence width")
    confidence_height = _positive_int(confidence_meta.get("height"), "confidence height")
    if (depth_width, depth_height) != (confidence_width, confidence_height):
        raise ValueError("depth and confidence dimensions do not match")
    rgb_width = _positive_int(frame.get("width"), "RGB width")
    rgb_height = _positive_int(frame.get("height"), "RGB height")
    intrinsics = np.asarray(frame.get("camera_intrinsics") or [], dtype=np.float64)
    if intrinsics.size != 9 or not np.all(np.isfinite(intrinsics)):
        raise ValueError("camera intrinsics must contain nine finite values")
    camera_to_world = np.asarray(
        frame.get("camera_to_world_reference_arkit_row_major") or [], dtype=np.float64
    )
    if camera_to_world.shape != (4, 4) or not np.all(np.isfinite(camera_to_world)):
        raise ValueError("reference camera transform must be a finite 4x4 matrix")

    expected_pixels = depth_width * depth_height
    depth_bytes = attachment_loader(depth_uri)
    confidence_bytes = attachment_loader(confidence_uri)
    if len(depth_bytes) != expected_pixels * 4:
        raise ValueError(f"depth payload has {len(depth_bytes)} bytes; expected {expected_pixels * 4}")
    if len(confidence_bytes) != expected_pixels:
        raise ValueError(
            f"confidence payload has {len(confidence_bytes)} bytes; expected {expected_pixels}"
        )
    depth = np.frombuffer(depth_bytes, dtype="<f4").reshape((depth_height, depth_width))
    confidence = np.frombuffer(confidence_bytes, dtype=np.uint8).reshape((depth_height, depth_width))
    high_confidence = confidence == HIGH_CONFIDENCE_VALUE
    physically_valid = high_confidence & np.isfinite(depth)
    physically_valid &= depth >= min_depth_meters
    physically_valid &= depth <= max_depth_meters
    sampled_mask = np.zeros_like(physically_valid)
    sampled_mask[::pixel_stride, ::pixel_stride] = physically_valid[::pixel_stride, ::pixel_stride]
    rows, columns = np.nonzero(sampled_mask)
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
    points = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    if points.size and not np.all(np.isfinite(points)):
        raise ValueError("back-projected points contain non-finite coordinates")
    valid_depths = depth[physically_valid].astype(np.float64)
    return {
        "capture_id": frame.get("capture_id"),
        "frame_index": int(frame.get("frame_index") or 0),
        "image_name": frame.get("image_name"),
        "pose_source": frame.get("pose_source"),
        "depth_sample_count": expected_pixels,
        "high_confidence_sample_count": int(np.count_nonzero(high_confidence)),
        "valid_high_confidence_sample_count": int(np.count_nonzero(physically_valid)),
        "sampled_point_count": int(points.shape[0]),
        "valid_depth_meters": _distribution(valid_depths),
        "points": points,
    }


def _merge_output_voxels(
    destination: dict[tuple[int, int, int, str], dict[str, Any]],
    points: np.ndarray,
    *,
    pose_source: str,
    voxel_size: float,
) -> None:
    if not points.size:
        return
    keys, sums, counts, _ = _group_points(points, voxel_size)
    for key_values, point_sum, count in zip(keys, sums, counts):
        key = (*tuple(int(value) for value in key_values), pose_source)
        row = destination.setdefault(
            key,
            {"sum": np.zeros(3), "count": 0, "frame_support": 0, "sources": defaultdict(int)},
        )
        row["sum"] += point_sum
        row["count"] += int(count)
        row["frame_support"] += 1
        row["sources"][pose_source] += 1


def _merge_agreement_cells(
    destination: dict[tuple[int, int, int], dict[str, Any]],
    points: np.ndarray,
    *,
    pose_source: str,
    voxel_size: float,
) -> None:
    if not points.size:
        return
    keys, sums, counts, outer_sums = _group_points(points, voxel_size)
    centroids = sums / counts[:, None]
    for key_values, point_sum, count, outer_sum, centroid in zip(
        keys, sums, counts, outer_sums, centroids
    ):
        key = tuple(int(value) for value in key_values)
        row = destination.setdefault(
            key,
            {
                "sum": np.zeros(3),
                "outer_sum": np.zeros((3, 3)),
                "count": 0,
                "centroid_sum": np.zeros(3),
                "centroid_outer_sum": np.zeros((3, 3)),
                "frame_support": 0,
                "source_sum": defaultdict(lambda: np.zeros(3)),
                "source_count": defaultdict(int),
                "source_frame_support": defaultdict(int),
            },
        )
        row["sum"] += point_sum
        row["outer_sum"] += outer_sum
        row["count"] += int(count)
        row["centroid_sum"] += centroid
        row["centroid_outer_sum"] += np.outer(centroid, centroid)
        row["frame_support"] += 1
        row["source_sum"][pose_source] += point_sum
        row["source_count"][pose_source] += int(count)
        row["source_frame_support"][pose_source] += 1


def _group_points(
    points: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    keys, inverse, counts = np.unique(voxel_keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis], minlength=len(keys)) for axis in range(3)]
    )
    outer_sums = np.empty((len(keys), 3, 3), dtype=np.float64)
    for left in range(3):
        for right in range(3):
            outer_sums[:, left, right] = np.bincount(
                inverse,
                weights=points[:, left] * points[:, right],
                minlength=len(keys),
            )
    return keys, sums, counts.astype(np.int64), outer_sums


def _viewer_points(
    voxels: Mapping[tuple[int, int, int, str], Mapping[str, Any]], *, max_points: int
) -> list[dict[str, Any]]:
    keys = list(voxels)
    if len(keys) > max_points:
        keys = sorted(keys, key=_voxel_hash)[:max_points]
    points = []
    for key in keys:
        row = voxels[key]
        position = np.asarray(row["sum"], dtype=np.float64) / int(row["count"])
        sources = row["sources"]
        has_arkit = int(sources.get("arkit_propagated", 0)) > 0
        if has_arkit:
            source = "arkit_propagated"
            color = [255, 158, 64]
        else:
            source = "colmap_registered"
            color = [66, 206, 255]
        points.append(
            {
                "position": [round(float(value), 6) for value in position],
                "color": color,
                "pose_source": source,
                "frame_support": int(row["frame_support"]),
                "sample_count": int(row["count"]),
            }
        )
    return points


def _agreement_statistics(cells: Mapping[tuple[int, int, int], Mapping[str, Any]]) -> dict[str, Any]:
    overlapping = []
    cross_source_offsets = []
    centroid_spreads = []
    normal_spreads = []
    overlapping_sample_count = 0
    total_sample_count = sum(int(row["count"]) for row in cells.values())
    for row in cells.values():
        support = int(row["frame_support"])
        if support < 2:
            continue
        overlapping.append(row)
        overlapping_sample_count += int(row["count"])
        centroid_covariance = (
            np.asarray(row["centroid_outer_sum"]) / support
            - np.outer(row["centroid_sum"] / support, row["centroid_sum"] / support)
        )
        centroid_spreads.append(math.sqrt(max(0.0, float(np.trace(centroid_covariance)))))
        count = int(row["count"])
        if count >= 6 and support >= 3:
            covariance = (
                np.asarray(row["outer_sum"]) / count
                - np.outer(row["sum"] / count, row["sum"] / count)
            )
            eigenvalues = np.linalg.eigvalsh(covariance)
            normal_spreads.append(math.sqrt(max(0.0, float(eigenvalues[0]))))
        source_count = row["source_count"]
        colmap_count = int(source_count.get("colmap_registered", 0))
        arkit_count = int(source_count.get("arkit_propagated", 0))
        if colmap_count and arkit_count:
            colmap_center = row["source_sum"]["colmap_registered"] / colmap_count
            arkit_center = row["source_sum"]["arkit_propagated"] / arkit_count
            cross_source_offsets.append(float(np.linalg.norm(colmap_center - arkit_center)))
    return {
        "cell_count": len(cells),
        "multi_frame_cell_count": len(overlapping),
        "multi_frame_sample_fraction": (
            round(overlapping_sample_count / total_sample_count, 6) if total_sample_count else 0.0
        ),
        "frame_centroid_spread_meters": _distribution(np.asarray(centroid_spreads)),
        "local_surface_normal_spread_meters": _distribution(np.asarray(normal_spreads)),
        "cross_pose_source_centroid_offset_meters": _distribution(np.asarray(cross_source_offsets)),
        "notes": {
            "frame_centroid_spread_meters": "RMS spread of per-frame centroids inside multi-frame agreement cells.",
            "local_surface_normal_spread_meters": "Smallest-axis standard deviation inside locally overlapping cells; a surface-thickness proxy, not a calibrated accuracy bound.",
            "cross_pose_source_centroid_offset_meters": "Distance between COLMAP-pose and ARKit-pose point centroids where both sources observe the same agreement cell.",
        },
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0, "min": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "min": round(float(np.min(finite)), 6),
        "median": round(float(np.median(finite)), 6),
        "p90": round(float(np.percentile(finite, 90)), 6),
        "p95": round(float(np.percentile(finite, 95)), 6),
        "max": round(float(np.max(finite)), 6),
    }


def _positive_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is missing or invalid") from exc
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _voxel_hash(key: Sequence[Any]) -> int:
    x, y, z = (int(value) & 0xFFFFFFFFFFFFFFFF for value in key[:3])
    value = (x * 0x9E3779B185EBCA87) ^ (y * 0xC2B2AE3D27D4EB4F) ^ (z * 0x165667B19E3779F9)
    if len(key) > 3 and key[3] == "arkit_propagated":
        value ^= 0xA24BAED4963EE407
    value ^= value >> 33
    value *= 0xFF51AFD7ED558CCD
    value ^= value >> 33
    return value & 0xFFFFFFFFFFFFFFFF
