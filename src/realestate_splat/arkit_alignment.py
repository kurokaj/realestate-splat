"""Robust ARKit-to-COLMAP camera trajectory alignment diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.prepare_nerfstudio_from_colmap import ImagePose, qvec_to_rotmat


ARKIT_TO_OPENCV_CAMERA_AXES = np.diag([1.0, -1.0, -1.0])
DEFAULT_HYBRID_MAX_POSITION_RESIDUAL_METERS = 0.05
DEFAULT_HYBRID_MAX_ANGULAR_RESIDUAL_DEGREES = 3.0
DEFAULT_HYBRID_GUARD_FRAMES = 5
DEFAULT_HYBRID_MIN_UNSTABLE_RUN = 2


@dataclass(frozen=True)
class CameraCorrespondence:
    image: ImagePose
    manifest_entry: Mapping[str, Any]
    capture_id: str
    frame_index: int
    arkit_center: np.ndarray
    arkit_camera_to_world: np.ndarray
    colmap_center: np.ndarray
    colmap_camera_to_world: np.ndarray


def analyze_arkit_alignment(
    image_manifest: Mapping[str, Any],
    images: Sequence[ImagePose],
    *,
    source_run_id: str | None = None,
    threshold_fraction: float = 0.05,
    ransac_iterations: int = 2000,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Align each Skanea capture trajectory to one COLMAP reconstruction.

    The estimated similarity maps ARKit world coordinates in meters into the
    arbitrary COLMAP reconstruction frame. The report is diagnostic: it does
    not mutate either source trajectory or claim that propagated poses were
    visually registered by COLMAP.
    """
    manifest_by_name, manifest_skanea_count = _skanea_manifest_entries(image_manifest)
    grouped: dict[str, list[CameraCorrespondence]] = defaultdict(list)
    missing_manifest: list[str] = []

    for image in images:
        entry = manifest_by_name.get(image.name)
        if entry is None:
            missing_manifest.append(image.name)
            continue
        correspondence = _camera_correspondence(image, entry)
        grouped[correspondence.capture_id].append(correspondence)

    capture_reports = []
    for capture_id in sorted(grouped):
        rows = sorted(grouped[capture_id], key=lambda row: (row.frame_index, row.image.name))
        if len(rows) < 3:
            capture_reports.append(
                {
                    "capture_id": capture_id,
                    "status": "insufficient_data",
                    "correspondence_count": len(rows),
                    "reason": "At least three registered camera correspondences are required.",
                }
            )
            continue
        capture_reports.append(
            _analyze_capture(
                capture_id,
                rows,
                threshold_fraction=threshold_fraction,
                ransac_iterations=ransac_iterations,
                random_seed=random_seed,
            )
        )

    completed = [report for report in capture_reports if report.get("status") == "completed"]
    if completed:
        status = "completed"
    elif manifest_skanea_count == 0:
        status = "not_applicable"
    else:
        status = "insufficient_data"

    return {
        "schema_version": 1,
        "status": status,
        "source_run_id": source_run_id,
        "coordinate_conventions": {
            "arkit_transform_layout": "4x4 column-major camera-to-world",
            "arkit_world": "right-handed, gravity-aligned, meters",
            "arkit_camera": "+X right, +Y up, camera looks along -Z",
            "colmap_pose": "world-to-camera quaternion and translation",
            "colmap_camera": "+X right, +Y down, camera looks along +Z",
            "camera_basis_conversion": "diag(1, -1, -1) applied on the right of ARKit camera-to-world rotation",
            "similarity_direction": "COLMAP_point = scale * rotation * ARKit_point + translation",
        },
        "settings": {
            "ransac_iterations": ransac_iterations,
            "random_seed": random_seed,
            "inlier_threshold_fraction_of_colmap_trajectory_extent": threshold_fraction,
        },
        "manifest_skanea_image_count": manifest_skanea_count,
        "registered_colmap_image_count": len(images),
        "registered_images_without_exact_skanea_manifest_identity": sorted(missing_manifest),
        "capture_count": len(capture_reports),
        "completed_capture_count": len(completed),
        "correspondence_count": sum(int(report.get("correspondence_count") or 0) for report in capture_reports),
        "inlier_count": sum(int(report.get("inlier_count") or 0) for report in completed),
        "captures": capture_reports,
    }


def analyze_arkit_alignment_files(
    image_manifest_path: Path,
    images_txt_path: Path,
    *,
    source_run_id: str | None = None,
    threshold_fraction: float = 0.05,
    ransac_iterations: int = 2000,
    random_seed: int = 0,
) -> dict[str, Any]:
    import json

    manifest = json.loads(image_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Image manifest root must be a JSON object")
    images = read_colmap_images_streaming(images_txt_path)
    return analyze_arkit_alignment(
        manifest,
        images,
        source_run_id=source_run_id,
        threshold_fraction=threshold_fraction,
        ransac_iterations=ransac_iterations,
        random_seed=random_seed,
    )


def read_colmap_images_streaming(path: Path) -> list[ImagePose]:
    """Read only pose rows without retaining large POINTS2D rows in memory."""
    images: list[ImagePose] = []
    expect_pose = True
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if expect_pose:
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    raise ValueError(f"Malformed COLMAP image pose line in {path}: {line[:200]}")
                images.append(
                    ImagePose(
                        int(parts[0]),
                        tuple(float(value) for value in parts[1:5]),
                        tuple(float(value) for value in parts[5:8]),
                        int(parts[8]),
                        " ".join(parts[9:]),
                    )
                )
                expect_pose = False
            else:
                # Every pose row is followed by exactly one POINTS2D row. It
                # may be empty, so do not filter blank lines in this state.
                expect_pose = True
    if not images:
        raise ValueError(f"No registered COLMAP images found in {path}")
    return images


def compact_alignment_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    captures = report.get("captures") if isinstance(report.get("captures"), list) else []
    completed = [item for item in captures if isinstance(item, dict) and item.get("status") == "completed"]
    return {
        "status": report.get("status"),
        "source_run_id": report.get("source_run_id"),
        "capture_count": report.get("capture_count"),
        "completed_capture_count": report.get("completed_capture_count"),
        "correspondence_count": report.get("correspondence_count"),
        "inlier_count": report.get("inlier_count"),
        "captures": [
            {
                "capture_id": item.get("capture_id"),
                "correspondence_count": item.get("correspondence_count"),
                "inlier_count": item.get("inlier_count"),
                "scale_colmap_units_per_meter": item.get("scale_colmap_units_per_meter"),
                "position_residual_colmap_units": item.get("position_residual_colmap_units"),
                "position_residual_meters": item.get("position_residual_meters"),
                "angular_residual_degrees": item.get("angular_residual_degrees"),
            }
            for item in completed
        ],
    }


def build_aligned_arkit_trajectories(
    report: Mapping[str, Any],
    image_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a completed diagnostic transform to every ARKit camera pose.

    Unlike the diagnostic correspondences, this includes manifest frames that
    COLMAP did not register. Those ARKit-only poses are the candidates that a
    future hybrid camera model may propagate after local continuity review.
    Coordinates remain in the COLMAP world convention; presentation layers may
    apply their own axis conversion.
    """
    manifest_by_name, _ = _skanea_manifest_entries(image_manifest)
    capture_reports = report.get("captures") if isinstance(report.get("captures"), list) else []
    completed_by_capture = {
        str(item.get("capture_id")): item
        for item in capture_reports
        if isinstance(item, dict) and item.get("status") == "completed" and item.get("capture_id")
    }
    grouped_entries: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in manifest_by_name.values():
        rgbd = entry.get("rgbd")
        if isinstance(rgbd, dict):
            grouped_entries[str(rgbd.get("capture_id") or "")].append(entry)

    trajectories = []
    for capture_id in sorted(completed_by_capture):
        capture_report = completed_by_capture[capture_id]
        matrix_values = capture_report.get("similarity_matrix_row_major")
        scale = float(capture_report.get("scale_colmap_units_per_meter") or 0.0)
        if not isinstance(matrix_values, list) or len(matrix_values) != 4 or scale <= 0.0:
            continue
        similarity = np.asarray(matrix_values, dtype=np.float64)
        if similarity.shape != (4, 4) or not np.all(np.isfinite(similarity)):
            continue
        world_rotation = similarity[:3, :3] / scale
        correspondence_by_name = {
            str(item.get("image_name")): item
            for item in capture_report.get("correspondences") or []
            if isinstance(item, dict) and item.get("image_name")
        }
        frames = []
        for entry in sorted(
            grouped_entries.get(capture_id, []),
            key=lambda item: (
                int((item.get("rgbd") or {}).get("frame_index") or 0),
                str(item.get("image_name") or ""),
            ),
        ):
            rgbd = entry.get("rgbd")
            values = rgbd.get("camera_transform") if isinstance(rgbd, dict) else None
            if not isinstance(values, list) or len(values) != 16:
                continue
            arkit_transform = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
            if not np.all(np.isfinite(arkit_transform)):
                continue
            center = arkit_transform[:3, 3]
            aligned_center = similarity[:3, :3] @ center + similarity[:3, 3]
            aligned_camera_to_world = world_rotation @ arkit_transform[:3, :3] @ ARKIT_TO_OPENCV_CAMERA_AXES
            correspondence = correspondence_by_name.get(str(entry.get("image_name") or ""))
            registered = correspondence is not None
            inlier = bool(correspondence.get("inlier")) if correspondence else None
            frames.append(
                {
                    "image_name": entry.get("image_name"),
                    "capture_id": capture_id,
                    "frame_index": int(rgbd.get("frame_index") or 0),
                    "registered": registered,
                    "inlier": inlier,
                    "classification": "missing" if not registered else ("inlier" if inlier else "outlier"),
                    "aligned_arkit_center_colmap": _vector(aligned_center),
                    "aligned_arkit_forward_colmap": _vector(aligned_camera_to_world[:, 2]),
                    "aligned_arkit_up_colmap": _vector(-aligned_camera_to_world[:, 1]),
                    "colmap_center": correspondence.get("colmap_center") if correspondence else None,
                    "position_residual_meters": correspondence.get("position_residual_meters") if correspondence else None,
                    "angular_residual_degrees": correspondence.get("angular_residual_degrees") if correspondence else None,
                }
            )
        frames = select_hybrid_camera_poses(frames)
        trajectories.append(
            {
                "capture_id": capture_id,
                "scale_colmap_units_per_meter": scale,
                "inlier_threshold_meters": capture_report.get("inlier_threshold_meters"),
                "colmap_to_arkit_matrix_row_major": [
                    [_round(value, 12) for value in row]
                    for row in np.linalg.inv(similarity)
                ],
                "hybrid_settings": {
                    "max_position_residual_meters": DEFAULT_HYBRID_MAX_POSITION_RESIDUAL_METERS,
                    "max_angular_residual_degrees": DEFAULT_HYBRID_MAX_ANGULAR_RESIDUAL_DEGREES,
                    "guard_frames": DEFAULT_HYBRID_GUARD_FRAMES,
                    "minimum_unstable_run_for_guard": DEFAULT_HYBRID_MIN_UNSTABLE_RUN,
                },
                "frame_count": len(frames),
                "registered_count": sum(1 for frame in frames if frame["registered"]),
                "inlier_count": sum(1 for frame in frames if frame["inlier"] is True),
                "outlier_count": sum(1 for frame in frames if frame["inlier"] is False),
                "missing_count": sum(1 for frame in frames if not frame["registered"]),
                "hybrid_colmap_count": sum(1 for frame in frames if frame["hybrid_pose_source"] == "colmap"),
                "hybrid_arkit_count": sum(1 for frame in frames if frame["hybrid_pose_source"] == "arkit"),
                "frames": frames,
            }
        )
    return {
        "schema_version": 1,
        "source_run_id": report.get("source_run_id"),
        "capture_count": len(trajectories),
        "frame_count": sum(int(item["frame_count"]) for item in trajectories),
        "registered_count": sum(int(item["registered_count"]) for item in trajectories),
        "inlier_count": sum(int(item["inlier_count"]) for item in trajectories),
        "outlier_count": sum(int(item["outlier_count"]) for item in trajectories),
        "missing_count": sum(int(item["missing_count"]) for item in trajectories),
        "hybrid_colmap_count": sum(int(item["hybrid_colmap_count"]) for item in trajectories),
        "hybrid_arkit_count": sum(int(item["hybrid_arkit_count"]) for item in trajectories),
        "captures": trajectories,
    }


def select_hybrid_camera_poses(
    frames: Sequence[Mapping[str, Any]],
    *,
    max_position_residual_meters: float = DEFAULT_HYBRID_MAX_POSITION_RESIDUAL_METERS,
    max_angular_residual_degrees: float = DEFAULT_HYBRID_MAX_ANGULAR_RESIDUAL_DEGREES,
    guard_frames: int = DEFAULT_HYBRID_GUARD_FRAMES,
    minimum_unstable_run_for_guard: int = DEFAULT_HYBRID_MIN_UNSTABLE_RUN,
) -> list[dict[str, Any]]:
    """Choose COLMAP or aligned ARKit poses without switching blindly per frame.

    Missing cameras, RANSAC-rejected correspondences, and stricter residual
    failures use ARKit. Consecutive unstable sequences are expanded by a small
    guard band because visual drift can begin before a correspondence crosses
    the hard threshold. Isolated failures are replaced without expanding into
    otherwise stable visual poses.
    """
    selected = [dict(frame) for frame in frames]
    base_reasons: dict[int, list[str]] = {}
    for index, frame in enumerate(selected):
        reasons: list[str] = []
        if not frame.get("registered"):
            reasons.append("missing_colmap")
        elif frame.get("inlier") is False:
            reasons.append("alignment_outlier")
        position_residual = frame.get("position_residual_meters")
        if position_residual is not None and float(position_residual) > max_position_residual_meters:
            reasons.append("position_residual")
        angular_residual = frame.get("angular_residual_degrees")
        if angular_residual is not None and float(angular_residual) > max_angular_residual_degrees:
            reasons.append("angular_residual")
        if reasons:
            base_reasons[index] = reasons

    guarded: set[int] = set()
    unstable_indexes = sorted(base_reasons)
    run_start = 0
    while run_start < len(unstable_indexes):
        run_end = run_start + 1
        while run_end < len(unstable_indexes):
            previous_index = unstable_indexes[run_end - 1]
            current_index = unstable_indexes[run_end]
            previous_frame = int(selected[previous_index].get("frame_index") or previous_index)
            current_frame = int(selected[current_index].get("frame_index") or current_index)
            if current_frame != previous_frame + 1:
                break
            run_end += 1
        run = unstable_indexes[run_start:run_end]
        if len(run) >= max(1, minimum_unstable_run_for_guard):
            first = max(0, run[0] - max(0, guard_frames))
            last = min(len(selected), run[-1] + max(0, guard_frames) + 1)
            guarded.update(range(first, last))
        run_start = run_end

    for index, frame in enumerate(selected):
        reasons = list(base_reasons.get(index, []))
        if index in guarded and not reasons:
            reasons.append("unstable_segment_guard")
        use_arkit = bool(reasons)
        frame["hybrid_pose_source"] = "arkit" if use_arkit else "colmap"
        frame["hybrid_replacement_reasons"] = reasons
        frame["hybrid_position_colmap"] = (
            frame.get("aligned_arkit_center_colmap") if use_arkit else frame.get("colmap_center")
        )
        frame["hybrid_forward_colmap"] = (
            frame.get("aligned_arkit_forward_colmap") if use_arkit else None
        )
    return selected


def build_hybrid_camera_set(
    report: Mapping[str, Any],
    image_manifest: Mapping[str, Any],
    images: Sequence[ImagePose],
    *,
    source_run_id: str | None = None,
) -> dict[str, Any]:
    """Build a versioned camera set without mutating the source COLMAP model."""
    trajectories = build_aligned_arkit_trajectories(report, image_manifest)
    manifest_by_name, _ = _skanea_manifest_entries(image_manifest)
    images_by_name = {image.name: image for image in images}
    capture_reports = {
        str(item.get("capture_id")): item
        for item in report.get("captures") or []
        if isinstance(item, dict) and item.get("status") == "completed" and item.get("capture_id")
    }
    artifact_captures = []
    all_transitions = []

    for trajectory in trajectories.get("captures") or []:
        capture_id = str(trajectory.get("capture_id") or "")
        capture_report = capture_reports.get(capture_id)
        if not capture_report:
            continue
        similarity = np.asarray(capture_report.get("similarity_matrix_row_major"), dtype=np.float64)
        scale = float(capture_report.get("scale_colmap_units_per_meter") or 0.0)
        if similarity.shape != (4, 4) or scale <= 0.0 or not np.all(np.isfinite(similarity)):
            continue
        world_rotation = similarity[:3, :3] / scale
        colmap_to_reference = np.linalg.inv(similarity)
        artifact_frames = []

        for selected in trajectory.get("frames") or []:
            image_name = str(selected.get("image_name") or "")
            entry = manifest_by_name.get(image_name)
            if not entry:
                continue
            rgbd = entry.get("rgbd") if isinstance(entry.get("rgbd"), dict) else {}
            raw_values = rgbd.get("camera_transform")
            if not isinstance(raw_values, list) or len(raw_values) != 16:
                continue
            raw_arkit = np.asarray(raw_values, dtype=np.float64).reshape((4, 4), order="F")
            predicted_colmap = np.eye(4, dtype=np.float64)
            predicted_colmap[:3, :3] = world_rotation @ raw_arkit[:3, :3] @ ARKIT_TO_OPENCV_CAMERA_AXES
            predicted_colmap[:3, 3] = similarity[:3, :3] @ raw_arkit[:3, 3] + similarity[:3, 3]

            colmap_image = images_by_name.get(image_name)
            registered_colmap = _colmap_camera_to_world(colmap_image) if colmap_image else None
            use_arkit = selected.get("hybrid_pose_source") == "arkit" or registered_colmap is None
            selected_colmap = predicted_colmap if use_arkit else registered_colmap

            selected_reference = np.eye(4, dtype=np.float64)
            selected_reference[:3, 3] = (colmap_to_reference @ np.append(selected_colmap[:3, 3], 1.0))[:3]
            selected_reference[:3, :3] = (
                world_rotation.T @ selected_colmap[:3, :3] @ ARKIT_TO_OPENCV_CAMERA_AXES
            )
            pose_source = "arkit_propagated" if use_arkit else "colmap_registered"
            frame = {
                "image_name": image_name,
                "image_path": entry.get("path"),
                "width": entry.get("width"),
                "height": entry.get("height"),
                "location": entry.get("location"),
                "source_id": entry.get("source_id"),
                "camera_group": entry.get("camera_group"),
                "capture_id": capture_id,
                "frame_index": int(rgbd.get("frame_index") or 0),
                "timestamp_seconds": rgbd.get("timestamp_seconds"),
                "pose_source": pose_source,
                "replacement_reasons": list(selected.get("hybrid_replacement_reasons") or []),
                "colmap_registration_available": registered_colmap is not None,
                "alignment_classification": selected.get("classification"),
                "position_residual_meters": selected.get("position_residual_meters"),
                "angular_residual_degrees": selected.get("angular_residual_degrees"),
                "camera_to_world_colmap_opencv_row_major": _matrix(selected_colmap),
                "camera_to_world_reference_arkit_row_major": _matrix(selected_reference),
                "camera_intrinsics": list(rgbd.get("camera_intrinsics") or []),
                "camera_intrinsics_layout": rgbd.get("camera_intrinsics_layout"),
                "depth": rgbd.get("depth"),
                "confidence": rgbd.get("confidence"),
                "capture_manifest_uri": rgbd.get("capture_manifest_uri"),
                "_raw_arkit_camera_to_world": raw_arkit,
            }
            artifact_frames.append(frame)

        transitions = _hybrid_source_transitions(artifact_frames)
        all_transitions.extend({"capture_id": capture_id, **item} for item in transitions)
        for frame in artifact_frames:
            frame.pop("_raw_arkit_camera_to_world", None)
        artifact_captures.append(
            {
                "capture_id": capture_id,
                "reference_world": "this capture's original gravity-aligned ARKit world",
                "scale_colmap_units_per_meter": scale,
                "frame_count": len(artifact_frames),
                "colmap_registered_count": sum(
                    1 for frame in artifact_frames if frame["pose_source"] == "colmap_registered"
                ),
                "arkit_propagated_count": sum(
                    1 for frame in artifact_frames if frame["pose_source"] == "arkit_propagated"
                ),
                "source_transition_count": len(transitions),
                "review_required_transition_count": sum(
                    1 for transition in transitions if transition["status"] == "review_required"
                ),
                "source_transitions": transitions,
                "frames": artifact_frames,
            }
        )

    settings = {
        "max_position_residual_meters": DEFAULT_HYBRID_MAX_POSITION_RESIDUAL_METERS,
        "max_angular_residual_degrees": DEFAULT_HYBRID_MAX_ANGULAR_RESIDUAL_DEGREES,
        "guard_frames": DEFAULT_HYBRID_GUARD_FRAMES,
        "minimum_unstable_run_for_guard": DEFAULT_HYBRID_MIN_UNSTABLE_RUN,
        "transition_review_translation_delta_meters": 0.10,
        "transition_review_angular_delta_degrees": 5.0,
    }
    selection_fingerprint = {
        "builder_schema_version": 1,
        "source_run_id": source_run_id or report.get("source_run_id"),
        "settings": settings,
        "frames": [
            {
                "capture_id": frame["capture_id"],
                "frame_index": frame["frame_index"],
                "image_name": frame["image_name"],
                "pose_source": frame["pose_source"],
                "replacement_reasons": frame["replacement_reasons"],
                "camera_to_world_colmap_opencv_row_major": frame[
                    "camera_to_world_colmap_opencv_row_major"
                ],
                "camera_to_world_reference_arkit_row_major": frame[
                    "camera_to_world_reference_arkit_row_major"
                ],
            }
            for capture in artifact_captures
            for frame in capture["frames"]
        ],
    }
    selection_sha256 = hashlib.sha256(
        json.dumps(selection_fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    review_count = sum(1 for transition in all_transitions if transition["status"] == "review_required")
    return {
        "schema_version": 1,
        "kind": "buildvision3d_arkit_hybrid_camera_set",
        "status": "completed_review_required" if review_count else "completed",
        "source_run_id": source_run_id or report.get("source_run_id"),
        "source_alignment_schema_version": report.get("schema_version"),
        "selection_sha256": selection_sha256,
        "settings": settings,
        "coordinate_conventions": {
            "camera_to_world_colmap_opencv_row_major": "COLMAP world units; +X right, +Y down, camera looks +Z",
            "camera_to_world_reference_arkit_row_major": "meters in the capture's gravity-aligned ARKit world; +X right, +Y up, camera looks -Z",
        },
        "capture_count": len(artifact_captures),
        "frame_count": sum(capture["frame_count"] for capture in artifact_captures),
        "colmap_registered_count": sum(capture["colmap_registered_count"] for capture in artifact_captures),
        "arkit_propagated_count": sum(capture["arkit_propagated_count"] for capture in artifact_captures),
        "transition_validation": {
            "source_transition_count": len(all_transitions),
            "review_required_count": review_count,
            "max_translation_step_delta_meters": _maximum(
                transition["translation_step_delta_meters"] for transition in all_transitions
            ),
            "max_angular_step_delta_degrees": _maximum(
                transition["angular_step_delta_degrees"] for transition in all_transitions
            ),
        },
        "captures": artifact_captures,
    }


def compact_hybrid_camera_set_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": artifact.get("status"),
        "source_run_id": artifact.get("source_run_id"),
        "selection_sha256": artifact.get("selection_sha256"),
        "capture_count": artifact.get("capture_count"),
        "frame_count": artifact.get("frame_count"),
        "colmap_registered_count": artifact.get("colmap_registered_count"),
        "arkit_propagated_count": artifact.get("arkit_propagated_count"),
        "transition_validation": artifact.get("transition_validation"),
    }


def _colmap_camera_to_world(image: ImagePose) -> np.ndarray:
    world_to_camera = np.asarray(qvec_to_rotmat(image.qvec), dtype=np.float64)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = world_to_camera.T
    camera_to_world[:3, 3] = -world_to_camera.T @ np.asarray(image.tvec, dtype=np.float64)
    return camera_to_world


def _hybrid_source_transitions(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    transitions = []
    for previous, current in zip(frames, frames[1:]):
        if previous.get("pose_source") == current.get("pose_source"):
            continue
        previous_selected = np.asarray(previous["camera_to_world_reference_arkit_row_major"], dtype=np.float64)
        current_selected = np.asarray(current["camera_to_world_reference_arkit_row_major"], dtype=np.float64)
        previous_raw = np.asarray(previous["_raw_arkit_camera_to_world"], dtype=np.float64)
        current_raw = np.asarray(current["_raw_arkit_camera_to_world"], dtype=np.float64)
        selected_step = current_selected[:3, 3] - previous_selected[:3, 3]
        arkit_step = current_raw[:3, 3] - previous_raw[:3, 3]
        translation_delta = float(np.linalg.norm(selected_step - arkit_step))
        selected_angle = rotation_angle_degrees(previous_selected[:3, :3].T @ current_selected[:3, :3])
        arkit_angle = rotation_angle_degrees(previous_raw[:3, :3].T @ current_raw[:3, :3])
        angular_delta = abs(selected_angle - arkit_angle)
        status = "review_required" if translation_delta > 0.10 or angular_delta > 5.0 else "pass"
        transitions.append(
            {
                "from_frame_index": previous.get("frame_index"),
                "to_frame_index": current.get("frame_index"),
                "from_pose_source": previous.get("pose_source"),
                "to_pose_source": current.get("pose_source"),
                "selected_translation_step_meters": _round(float(np.linalg.norm(selected_step))),
                "arkit_translation_step_meters": _round(float(np.linalg.norm(arkit_step))),
                "translation_step_delta_meters": _round(translation_delta),
                "selected_angular_step_degrees": _round(selected_angle),
                "arkit_angular_step_degrees": _round(arkit_angle),
                "angular_step_delta_degrees": _round(angular_delta),
                "status": status,
            }
        )
    return transitions


def _matrix(values: np.ndarray, *, digits: int = 12) -> list[list[float]]:
    return [[_round(value, digits) for value in row] for row in values]


def _maximum(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return _round(max(finite)) if finite else None


def _skanea_manifest_entries(
    image_manifest: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], int]:
    by_name: dict[str, Mapping[str, Any]] = {}
    count = 0
    images = image_manifest.get("images")
    if not isinstance(images, list):
        return by_name, count
    identities: set[tuple[str, int]] = set()
    for raw_entry in images:
        if not isinstance(raw_entry, dict):
            continue
        rgbd = raw_entry.get("rgbd")
        if raw_entry.get("source_kind") != "skanea_rgbd_frame" or not isinstance(rgbd, dict):
            continue
        image_name = str(raw_entry.get("image_name") or "")
        capture_id = str(rgbd.get("capture_id") or "")
        frame_index = rgbd.get("frame_index")
        if not image_name or not capture_id or not isinstance(frame_index, int):
            raise ValueError("Skanea image manifest entry is missing image_name, capture_id, or integer frame_index")
        identity = (capture_id, frame_index)
        if image_name in by_name:
            raise ValueError(f"Duplicate Skanea image_name in image manifest: {image_name}")
        if identity in identities:
            raise ValueError(f"Duplicate Skanea capture/frame identity in image manifest: {capture_id}/{frame_index}")
        by_name[image_name] = raw_entry
        identities.add(identity)
        count += 1
    return by_name, count


def _camera_correspondence(image: ImagePose, entry: Mapping[str, Any]) -> CameraCorrespondence:
    rgbd = entry.get("rgbd")
    if not isinstance(rgbd, dict):
        raise ValueError(f"Skanea manifest entry for {image.name} has no rgbd object")
    values = rgbd.get("camera_transform")
    if not isinstance(values, list) or len(values) != 16:
        raise ValueError(f"Skanea manifest entry for {image.name} has no 16-value camera_transform")
    layout = str(rgbd.get("camera_transform_layout") or "").lower()
    if "column-major" not in layout or "camera-to-world" not in layout:
        raise ValueError(f"Unsupported ARKit camera transform convention for {image.name}: {layout!r}")

    arkit_transform = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
    if not np.all(np.isfinite(arkit_transform)):
        raise ValueError(f"ARKit camera transform contains non-finite values for {image.name}")
    if not np.allclose(arkit_transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        raise ValueError(f"ARKit camera transform has an invalid homogeneous row for {image.name}")

    colmap_world_to_camera = np.asarray(qvec_to_rotmat(image.qvec), dtype=np.float64)
    colmap_camera_to_world = colmap_world_to_camera.T
    colmap_center = -colmap_camera_to_world @ np.asarray(image.tvec, dtype=np.float64)
    return CameraCorrespondence(
        image=image,
        manifest_entry=entry,
        capture_id=str(rgbd["capture_id"]),
        frame_index=int(rgbd["frame_index"]),
        arkit_center=arkit_transform[:3, 3],
        arkit_camera_to_world=arkit_transform[:3, :3],
        colmap_center=colmap_center,
        colmap_camera_to_world=colmap_camera_to_world,
    )


def _analyze_capture(
    capture_id: str,
    rows: Sequence[CameraCorrespondence],
    *,
    threshold_fraction: float,
    ransac_iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    source = np.stack([row.arkit_center for row in rows])
    target = np.stack([row.colmap_center for row in rows])
    extent = float(np.linalg.norm(np.max(target, axis=0) - np.min(target, axis=0)))
    if not math.isfinite(extent) or extent <= 1e-9:
        return {
            "capture_id": capture_id,
            "status": "insufficient_data",
            "correspondence_count": len(rows),
            "reason": "COLMAP camera trajectory has no measurable extent.",
        }
    threshold = max(extent * threshold_fraction, 1e-9)
    scale, rotation, translation, inliers = robust_similarity_transform(
        source,
        target,
        threshold=threshold,
        iterations=ransac_iterations,
        random_seed=random_seed,
    )
    aligned = apply_similarity(source, scale, rotation, translation)
    position_residuals = np.linalg.norm(aligned - target, axis=1)
    angular_residuals = []
    correspondences = []
    for index, row in enumerate(rows):
        predicted_camera_to_world = rotation @ row.arkit_camera_to_world @ ARKIT_TO_OPENCV_CAMERA_AXES
        angular_residual = rotation_angle_degrees(row.colmap_camera_to_world.T @ predicted_camera_to_world)
        angular_residuals.append(angular_residual)
        correspondences.append(
            {
                "image_name": row.image.name,
                "capture_id": capture_id,
                "frame_index": row.frame_index,
                "inlier": bool(inliers[index]),
                "position_residual_colmap_units": _round(position_residuals[index]),
                "position_residual_meters": _round(position_residuals[index] / scale),
                "angular_residual_degrees": _round(angular_residual),
                "arkit_center_meters": _vector(row.arkit_center),
                "aligned_arkit_center_colmap": _vector(aligned[index]),
                "colmap_center": _vector(row.colmap_center),
            }
        )

    angular_array = np.asarray(angular_residuals, dtype=np.float64)
    inlier_positions = position_residuals[inliers]
    inlier_angles = angular_array[inliers]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return {
        "capture_id": capture_id,
        "status": "completed",
        "correspondence_count": len(rows),
        "inlier_count": int(np.count_nonzero(inliers)),
        "rejected_count": int(len(rows) - np.count_nonzero(inliers)),
        "trajectory_extent_colmap_units": _round(extent),
        "trajectory_extent_meters": _round(extent / scale),
        "inlier_threshold_colmap_units": _round(threshold),
        "inlier_threshold_meters": _round(threshold / scale),
        "scale_colmap_units_per_meter": _round(scale, 9),
        "scale_meters_per_colmap_unit": _round(1.0 / scale, 9),
        "rotation_row_major": [[_round(value, 12) for value in row] for row in rotation],
        "translation_colmap_units": _vector(translation, 12),
        "similarity_matrix_row_major": [[_round(value, 12) for value in row] for row in transform],
        "position_residual_colmap_units": distribution(inlier_positions),
        "position_residual_meters": distribution(inlier_positions / scale),
        "position_residual_all_correspondences_colmap_units": distribution(position_residuals),
        "position_residual_all_correspondences_meters": distribution(position_residuals / scale),
        "angular_residual_degrees": distribution(inlier_angles),
        "angular_residual_all_correspondences_degrees": distribution(angular_array),
        "correspondences": correspondences,
    }


def robust_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    iterations: int,
    random_seed: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity inputs must be equally sized Nx3 arrays")
    count = source.shape[0]
    if count < 3:
        raise ValueError("At least three correspondences are required")
    rng = np.random.default_rng(random_seed)
    best_inliers: np.ndarray | None = None
    best_median = math.inf
    for _ in range(max(1, iterations)):
        sample = rng.choice(count, size=3, replace=False)
        if np.linalg.matrix_rank(source[sample] - np.mean(source[sample], axis=0)) < 2:
            continue
        try:
            scale, rotation, translation = estimate_similarity_transform(source[sample], target[sample])
        except ValueError:
            continue
        residuals = np.linalg.norm(apply_similarity(source, scale, rotation, translation) - target, axis=1)
        inliers = residuals <= threshold
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 3:
            continue
        median = float(np.median(residuals[inliers]))
        if best_inliers is None or inlier_count > int(np.count_nonzero(best_inliers)) or (
            inlier_count == int(np.count_nonzero(best_inliers)) and median < best_median
        ):
            best_inliers = inliers
            best_median = median
    if best_inliers is None:
        raise ValueError("Could not estimate a non-degenerate robust similarity transform")

    for _ in range(3):
        scale, rotation, translation = estimate_similarity_transform(source[best_inliers], target[best_inliers])
        residuals = np.linalg.norm(apply_similarity(source, scale, rotation, translation) - target, axis=1)
        refined = residuals <= threshold
        if np.count_nonzero(refined) < 3 or np.array_equal(refined, best_inliers):
            break
        best_inliers = refined
    scale, rotation, translation = estimate_similarity_transform(source[best_inliers], target[best_inliers])
    return scale, rotation, translation, best_inliers


def estimate_similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(source) < 3:
        raise ValueError("At least three points are required for a 3D similarity transform")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.sum(source_centered * source_centered) / len(source))
    if source_variance <= 1e-15:
        raise ValueError("Source trajectory is degenerate")
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(left) * np.linalg.det(right_t) < 0.0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_t
    scale = float(np.sum(singular_values * sign) / source_variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Estimated similarity scale is not positive and finite")
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (rotation @ points.T)).T + translation


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = float((np.trace(rotation) - 1.0) / 2.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "p90": None, "p95": None, "max": None, "rmse": None}
    return {
        "count": int(finite.size),
        "min": _round(np.min(finite)),
        "median": _round(np.median(finite)),
        "mean": _round(np.mean(finite)),
        "p90": _round(np.percentile(finite, 90)),
        "p95": _round(np.percentile(finite, 95)),
        "max": _round(np.max(finite)),
        "rmse": _round(math.sqrt(float(np.mean(finite * finite)))),
    }


def _round(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def _vector(values: Sequence[float] | np.ndarray, digits: int = 6) -> list[float]:
    return [_round(value, digits) for value in values]
