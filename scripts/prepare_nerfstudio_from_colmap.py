#!/usr/bin/env python3
"""Prepare a Nerfstudio transforms dataset from an existing COLMAP text model.

This is the multi-camera handoff path for Buildvision3D runs. The normal
Nerfstudio ``ns-process-data images --skip-colmap`` path is still preferred for
single-camera reconstructions; this script exists because that importer can
mis-pair image dimensions and camera intrinsics on newer multi-camera COLMAP
models.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: Tuple[float, ...]


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    qvec: Tuple[float, float, float, float]
    tvec: Tuple[float, float, float]
    camera_id: int
    name: str


@dataclass(frozen=True)
class Point3D:
    point_id: int
    xyz: Tuple[float, float, float]
    rgb: Tuple[int, int, int]
    error: float
    track_length: int = 0


@dataclass(frozen=True)
class ColoredPoint:
    xyz: Tuple[float, float, float]
    rgb: Tuple[int, int, int]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Nerfstudio transforms.json dataset from COLMAP TXT output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", required=True, type=Path, help="Run directory.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Source image directory.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Output Nerfstudio data directory.")
    parser.add_argument(
        "--colmap-model-dir",
        required=True,
        type=Path,
        help="COLMAP sparse model directory. The sibling colmap/sparse_txt export is used when needed.",
    )
    parser.add_argument(
        "--hybrid-camera-set",
        type=Path,
        help="Optional immutable A_hybrid camera-set JSON. When supplied, all accepted hybrid frames are exported.",
    )
    parser.add_argument(
        "--expected-colmap-run-id",
        help="Expected source COLMAP stage run id for the hybrid artifact.",
    )
    parser.add_argument(
        "--lidar-initialization",
        type=Path,
        help="Optional immutable LiDAR-initialization metadata JSON. Requires --hybrid-camera-set.",
    )
    parser.add_argument(
        "--lidar-initialization-ply",
        type=Path,
        help="RGB-colored gravity-aligned PLY referenced by --lidar-initialization.",
    )
    parser.add_argument(
        "--merge-colmap-initialization",
        action="store_true",
        help="Merge quality-filtered, metric-aligned COLMAP points into the LiDAR initialization.",
    )
    parser.add_argument("--num-downscales", type=int, default=2, help="Number of images_2/images_4/... folders to create.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output data directory.")
    return parser.parse_args(argv)


def resolve_under_run(run_dir: Path, value: Path) -> Path:
    expanded = value.expanduser()
    if expanded.is_absolute():
        return expanded
    return run_dir / expanded


def colmap_text_dir_from_model_dir(colmap_model_dir: Path) -> Path:
    if (colmap_model_dir / "cameras.txt").exists() and (colmap_model_dir / "images.txt").exists():
        return colmap_model_dir
    colmap_dir = colmap_model_dir
    for parent in [colmap_model_dir, *colmap_model_dir.parents]:
        if parent.name == "colmap":
            colmap_dir = parent
            break
    return colmap_dir / "sparse_txt"


def useful_lines(path: Path) -> Iterable[str]:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            yield line


def read_cameras(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    for line in useful_lines(path):
        parts = line.split()
        if len(parts) < 5:
            raise SystemExit(f"Malformed COLMAP camera line in {path}: {line}")
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = tuple(float(value) for value in parts[4:])
        cameras[camera_id] = Camera(camera_id, model, width, height, params)
    if not cameras:
        raise SystemExit(f"No cameras found in {path}")
    return cameras


def read_images(path: Path) -> List[ImagePose]:
    lines = list(useful_lines(path))
    images: List[ImagePose] = []
    for index in range(0, len(lines), 2):
        parts = lines[index].split()
        if len(parts) < 10:
            raise SystemExit(f"Malformed COLMAP image line in {path}: {lines[index]}")
        image_id = int(parts[0])
        qvec = tuple(float(value) for value in parts[1:5])
        tvec = tuple(float(value) for value in parts[5:8])
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        images.append(ImagePose(image_id, qvec, tvec, camera_id, name))
    if not images:
        raise SystemExit(f"No registered images found in {path}")
    return images


def read_points3d(path: Path) -> List[Point3D]:
    points: List[Point3D] = []
    for line in useful_lines(path):
        parts = line.split()
        if len(parts) < 8:
            raise SystemExit(f"Malformed COLMAP points3D line in {path}: {line}")
        points.append(
            Point3D(
                point_id=int(parts[0]),
                xyz=(float(parts[1]), float(parts[2]), float(parts[3])),
                rgb=(int(parts[4]), int(parts[5]), int(parts[6])),
                error=float(parts[7]),
                track_length=max(0, (len(parts) - 8) // 2),
            )
        )
    if not points:
        raise SystemExit(f"No COLMAP sparse points found in {path}; refusing random Splatfacto initialization.")
    return points


def qvec_to_rotmat(qvec: Sequence[float]) -> List[List[float]]:
    qw, qx, qy, qz = qvec
    return [
        [
            1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
            2.0 * qx * qy - 2.0 * qz * qw,
            2.0 * qz * qx + 2.0 * qy * qw,
        ],
        [
            2.0 * qx * qy + 2.0 * qz * qw,
            1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
            2.0 * qy * qz - 2.0 * qx * qw,
        ],
        [
            2.0 * qz * qx - 2.0 * qy * qw,
            2.0 * qy * qz + 2.0 * qx * qw,
            1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
        ],
    ]


def colmap_pose_to_nerfstudio_transform(image: ImagePose) -> List[List[float]]:
    world_to_camera = qvec_to_rotmat(image.qvec)
    tvec = image.tvec

    # Match Nerfstudio's COLMAP conversion:
    # 1. invert COLMAP's world-to-camera OpenCV pose to camera-to-world,
    # 2. flip camera Y/Z axes from OpenCV to OpenGL,
    # 3. remap COLMAP world axes into Nerfstudio's world convention.
    rotation_t = [[world_to_camera[row][col] for row in range(3)] for col in range(3)]
    center = [-sum(rotation_t[row][col] * tvec[col] for col in range(3)) for row in range(3)]

    opengl_camera_to_world = [
        [rotation_t[0][0], -rotation_t[0][1], -rotation_t[0][2], center[0]],
        [rotation_t[1][0], -rotation_t[1][1], -rotation_t[1][2], center[1]],
        [rotation_t[2][0], -rotation_t[2][1], -rotation_t[2][2], center[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    transform = [
        opengl_camera_to_world[0],
        opengl_camera_to_world[2],
        [-value for value in opengl_camera_to_world[1]],
        opengl_camera_to_world[3],
    ]
    return transform


def colmap_camera_to_world_to_nerfstudio_transform(
    camera_to_world: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Convert an OpenCV camera-to-world matrix in COLMAP world axes to Nerfstudio."""
    matrix = validate_camera_to_world_matrix(camera_to_world)
    opengl_camera_to_world = [
        [matrix[row][0], -matrix[row][1], -matrix[row][2], matrix[row][3]]
        for row in range(3)
    ]
    opengl_camera_to_world.append([0.0, 0.0, 0.0, 1.0])
    return [
        opengl_camera_to_world[0],
        opengl_camera_to_world[2],
        [-value for value in opengl_camera_to_world[1]],
        opengl_camera_to_world[3],
    ]


def reference_arkit_camera_to_world_to_nerfstudio_transform(
    camera_to_world: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Rotate ARKit's +Y-up OpenGL world into Nerfstudio's +Z-up world."""
    matrix = validate_camera_to_world_matrix(camera_to_world)
    world_rotation = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    return [
        [sum(world_rotation[row][index] * matrix[index][column] for index in range(4)) for column in range(4)]
        for row in range(4)
    ]


def validate_camera_to_world_matrix(
    value: Sequence[Sequence[float]],
) -> List[List[float]]:
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in value):
        raise SystemExit("Hybrid camera transform must be a 4x4 row-major matrix.")
    try:
        matrix = [[float(item) for item in row] for row in value]
    except (TypeError, ValueError) as exc:
        raise SystemExit("Hybrid camera transform contains a non-numeric value.") from exc
    if not all(math.isfinite(item) for row in matrix for item in row):
        raise SystemExit("Hybrid camera transform contains a non-finite value.")
    if any(abs(matrix[3][index] - expected) > 1e-5 for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise SystemExit("Hybrid camera transform has an invalid homogeneous row.")

    rotation = [row[:3] for row in matrix[:3]]
    for row in rotation:
        if abs(sum(item * item for item in row) - 1.0) > 2e-3:
            raise SystemExit("Hybrid camera transform rotation is not normalized.")
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if abs(sum(rotation[first][index] * rotation[second][index] for index in range(3))) > 2e-3:
            raise SystemExit("Hybrid camera transform rotation is not orthogonal.")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 2e-3:
        raise SystemExit("Hybrid camera transform rotation must be right-handed.")
    return matrix


def colmap_point_to_nerfstudio(point: Point3D) -> Tuple[float, float, float]:
    x, y, z = point.xyz
    return (x, z, -y)


def camera_intrinsics(camera: Camera) -> Dict[str, Any]:
    params = camera.params
    model = camera.model.upper()
    intrinsics: Dict[str, Any] = {
        "w": camera.width,
        "h": camera.height,
        "camera_model": "OPENCV",
    }

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        intrinsics.update({"fl_x": f, "fl_y": f, "cx": cx, "cy": cy})
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        intrinsics.update({"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy})
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        intrinsics.update({"fl_x": f, "fl_y": f, "cx": cx, "cy": cy, "k1": k1})
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params
        intrinsics.update({"fl_x": f, "fl_y": f, "cx": cx, "cy": cy, "k1": k1, "k2": k2})
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        intrinsics.update({"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "p1": p1, "p2": p2})
    elif model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        intrinsics.update(
            {
                "camera_model": "OPENCV_FISHEYE",
                "fl_x": fx,
                "fl_y": fy,
                "cx": cx,
                "cy": cy,
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "k4": k4,
            }
        )
    else:
        raise SystemExit(
            f"Unsupported COLMAP camera model for custom Nerfstudio export: {camera.model}. "
            "Use SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, or OPENCV_FISHEYE."
        )

    intrinsics.setdefault("k1", 0.0)
    intrinsics.setdefault("k2", 0.0)
    intrinsics.setdefault("p1", 0.0)
    intrinsics.setdefault("p2", 0.0)
    return intrinsics


def image_size(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise SystemExit("Pillow or OpenCV is required for multi-camera Nerfstudio data preparation.") from exc

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SystemExit(f"Could not read image dimensions: {path}")
        height, width = image.shape[:2]
        return width, height

    with Image.open(path) as image:
        return image.size


def copy_registered_images(images: Sequence[ImagePose], frames_dir: Path, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        source = frames_dir / image.name
        if not source.exists():
            raise SystemExit(f"Registered COLMAP image is missing from frames directory: {source}")
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Unsupported registered image extension: {source}")
        shutil.copy2(source, images_dir / image.name)


def copy_named_images(image_names: Sequence[str], frames_dir: Path, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for image_name in image_names:
        relative = Path(image_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"Unsafe hybrid image name: {image_name}")
        source = frames_dir / relative
        if not source.exists():
            raise SystemExit(f"Hybrid camera image is missing from frames directory: {source}")
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Unsupported hybrid image extension: {source}")
        destination = images_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_downscales(images_dir: Path, num_downscales: int) -> None:
    if num_downscales < 0:
        raise SystemExit("--num-downscales must be zero or greater.")
    if num_downscales == 0:
        return

    try:
        from PIL import Image
    except ImportError:
        build_downscales_with_opencv(images_dir, num_downscales)
        return

    resample = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
    source_images = [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    for level in range(1, num_downscales + 1):
        factor = 2**level
        target_dir = images_dir.parent / f"images_{factor}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in source_images:
            with Image.open(source) as image:
                width, height = image.size
                target_size = (max(1, math.floor(width / factor)), max(1, math.floor(height / factor)))
                image.resize(target_size, resample=resample).save(target_dir / source.name)


def build_downscales_with_opencv(images_dir: Path, num_downscales: int) -> None:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("Pillow or OpenCV is required for --num-downscales > 0.") from exc

    source_images = [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    for level in range(1, num_downscales + 1):
        factor = 2**level
        target_dir = images_dir.parent / f"images_{factor}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in source_images:
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise SystemExit(f"Could not read image for downscale: {source}")
            height, width = image.shape[:2]
            target_size = (max(1, math.floor(width / factor)), max(1, math.floor(height / factor)))
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(target_dir / source.name), resized):
                raise SystemExit(f"Could not write downscaled image: {target_dir / source.name}")


def manifest_by_name(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    manifest_path = run_dir / "reports" / "image_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(entry["image_name"]): dict(entry)
        for entry in manifest.get("images", [])
        if isinstance(entry, dict) and entry.get("image_name")
    }


def manifest_camera_group_id(entry: Mapping[str, Any]) -> str:
    explicit = entry.get("camera_group_id")
    if explicit:
        return str(explicit)
    return "{}_{}x{}".format(entry.get("camera_group"), entry.get("width"), entry.get("height"))


def validate_manifest_camera_count(cameras: Mapping[int, Camera], manifest: Mapping[str, Mapping[str, Any]]) -> None:
    if not manifest:
        return
    manifest_group_ids = {manifest_camera_group_id(entry) for entry in manifest.values()}
    if len(manifest_group_ids) <= 1:
        return
    if len(cameras) != len(manifest_group_ids):
        raise SystemExit(
            "COLMAP camera count does not match manifest camera groups. "
            f"COLMAP cameras={len(cameras)}, manifest groups={len(manifest_group_ids)}. "
            "Rerun COLMAP with manifest camera grouping enabled before training."
        )


def build_transforms(
    *,
    run_dir: Path,
    frames_dir: Path,
    images: Sequence[ImagePose],
    cameras: Mapping[int, Camera],
    points: Sequence[Point3D],
    point_cloud_path: Path,
) -> Dict[str, Any]:
    manifest = manifest_by_name(run_dir)
    validate_manifest_camera_count(cameras, manifest)
    frames: List[Dict[str, Any]] = []
    for image in images:
        camera = cameras.get(image.camera_id)
        if camera is None:
            raise SystemExit(f"Image {image.name} references missing camera id {image.camera_id}")
        source_path = frames_dir / image.name
        actual_width, actual_height = image_size(source_path)
        if (actual_width, actual_height) != (camera.width, camera.height):
            raise SystemExit(
                "COLMAP camera dimensions do not match image pixels for "
                f"{image.name}: camera={camera.width}x{camera.height}, image={actual_width}x{actual_height}"
            )

        manifest_entry = manifest.get(image.name)
        group_id = manifest_camera_group_id(manifest_entry) if manifest_entry else None

        frame = {
            "file_path": f"images/{image.name}",
            "transform_matrix": colmap_pose_to_nerfstudio_transform(image),
            "colmap_image_id": image.image_id,
            "colmap_camera_id": image.camera_id,
            "intrinsics_source": "colmap_camera",
            **camera_intrinsics(camera),
        }
        if manifest_entry:
            frame["role"] = manifest_entry.get("role")
            frame["camera_group"] = manifest_entry.get("camera_group")
            frame["camera_group_id"] = group_id
            frame["location"] = manifest_entry.get("location")
            frame["source_id"] = manifest_entry.get("source_id")
        frames.append(frame)

    return {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "ply_file_path": point_cloud_path.name,
        "frames": frames,
        "buildvision3d": {
            "source": "colmap_sparse_txt",
            "registered_images": len(frames),
            "camera_count": len(cameras),
            "colmap_sparse_point_count": len(points),
            "point_cloud_path": point_cloud_path.name,
            "point_cloud_stats": point_cloud_stats(points),
            "multi_camera": len(cameras) > 1,
            "intrinsics_mode": "colmap_camera",
        },
    }


def hybrid_frames(
    artifact: Mapping[str, Any],
    *,
    expected_colmap_run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if artifact.get("kind") != "buildvision3d_arkit_hybrid_camera_set":
        raise SystemExit("Hybrid camera artifact has an unsupported kind.")
    if artifact.get("status") != "completed":
        raise SystemExit("Hybrid camera artifact must be completed with no transitions requiring review.")
    source_run_id = str(artifact.get("source_run_id") or "")
    if expected_colmap_run_id and source_run_id != expected_colmap_run_id:
        raise SystemExit(
            "Hybrid camera artifact source run does not match selected COLMAP run: "
            f"expected {expected_colmap_run_id}, found {source_run_id or '<missing>'}."
        )
    transition_validation = artifact.get("transition_validation")
    if not isinstance(transition_validation, Mapping) or int(transition_validation.get("review_required_count") or 0) != 0:
        raise SystemExit("Hybrid camera artifact has unresolved source transitions.")

    frames: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for capture in artifact.get("captures") or []:
        if not isinstance(capture, Mapping):
            continue
        for raw_frame in capture.get("frames") or []:
            if not isinstance(raw_frame, Mapping):
                continue
            frame = dict(raw_frame)
            image_name = str(frame.get("image_name") or "")
            if not image_name:
                raise SystemExit("Hybrid camera artifact contains a frame without image_name.")
            if image_name in seen_names:
                raise SystemExit(f"Hybrid camera artifact contains duplicate image: {image_name}")
            pose_source = str(frame.get("pose_source") or "")
            if pose_source not in {"colmap_registered", "arkit_propagated"}:
                raise SystemExit(f"Hybrid camera frame {image_name} has unsupported pose source: {pose_source}")
            intrinsics = frame.get("camera_intrinsics")
            if not isinstance(intrinsics, list) or len(intrinsics) != 9:
                raise SystemExit(f"Hybrid camera frame {image_name} is missing captured intrinsics.")
            try:
                intrinsics_values = [float(value) for value in intrinsics]
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"Hybrid camera frame {image_name} has invalid captured intrinsics.") from exc
            if not all(math.isfinite(value) for value in intrinsics_values) or intrinsics_values[0] <= 0 or intrinsics_values[4] <= 0:
                raise SystemExit(f"Hybrid camera frame {image_name} has invalid captured intrinsics.")
            validate_camera_to_world_matrix(frame.get("camera_to_world_colmap_opencv_row_major"))
            seen_names.add(image_name)
            frames.append(frame)

    expected_count = int(artifact.get("frame_count") or 0)
    if not frames or len(frames) != expected_count:
        raise SystemExit(
            f"Hybrid camera frame count mismatch: artifact declares {expected_count}, contains {len(frames)}."
        )
    source_counts = {
        "colmap_registered": sum(1 for frame in frames if frame["pose_source"] == "colmap_registered"),
        "arkit_propagated": sum(1 for frame in frames if frame["pose_source"] == "arkit_propagated"),
    }
    if source_counts["colmap_registered"] != int(artifact.get("colmap_registered_count") or 0):
        raise SystemExit("Hybrid COLMAP pose count does not match artifact summary.")
    if source_counts["arkit_propagated"] != int(artifact.get("arkit_propagated_count") or 0):
        raise SystemExit("Hybrid ARKit pose count does not match artifact summary.")
    return frames


def validate_lidar_initialization(
    artifact: Mapping[str, Any],
    hybrid_artifact: Mapping[str, Any],
    ply_path: Path,
) -> None:
    if artifact.get("kind") != "buildvision3d_lidar_training_initialization":
        raise SystemExit("LiDAR initialization artifact has an unsupported kind.")
    if artifact.get("status") != "completed":
        raise SystemExit("LiDAR initialization must be completed without skipped frames.")
    if str(artifact.get("source_run_id") or "") != str(hybrid_artifact.get("source_run_id") or ""):
        raise SystemExit("LiDAR initialization source run does not match A_hybrid.")
    if str(artifact.get("source_hybrid_selection_sha256") or "") != str(
        hybrid_artifact.get("selection_sha256") or ""
    ):
        raise SystemExit("LiDAR initialization selection does not match A_hybrid.")
    captures = [item for item in hybrid_artifact.get("captures") or [] if isinstance(item, Mapping)]
    if len(captures) != 1 or str(captures[0].get("capture_id") or "") != str(artifact.get("capture_id") or ""):
        raise SystemExit("LiDAR initialization capture identity does not match A_hybrid.")
    declared_count = int(artifact.get("filtered_voxel_count") or 0)
    actual_count = read_ply_vertex_count(ply_path)
    if declared_count <= 0 or actual_count != declared_count:
        raise SystemExit(
            f"LiDAR initialization PLY count mismatch: artifact={declared_count}, ply={actual_count}."
        )


def read_ply_vertex_count(path: Path) -> int:
    if not path.exists():
        raise SystemExit(f"Initialization PLY does not exist: {path}")
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[-1])
            if line == "end_header":
                break
    raise SystemExit(f"Initialization PLY has no vertex count: {path}")


def read_colored_ascii_ply(path: Path) -> List[ColoredPoint]:
    """Read the simple XYZ/RGB ASCII PLY written by the LiDAR initializer."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise SystemExit(f"Initialization PLY has an invalid header: {path}")
    if len(lines) < 2 or lines[1].strip() != "format ascii 1.0":
        raise SystemExit(f"Merged initialization requires an ASCII PLY: {path}")
    vertex_count = None
    properties: List[str] = []
    data_index = None
    in_vertex = False
    for index, raw_line in enumerate(lines[2:], start=2):
        line = raw_line.strip()
        if line.startswith("element "):
            parts = line.split()
            in_vertex = len(parts) == 3 and parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if line.startswith("property ") and in_vertex:
            properties.append(line.split()[-1])
            continue
        if line == "end_header":
            data_index = index + 1
            break
    if vertex_count is None or data_index is None:
        raise SystemExit(f"Initialization PLY is missing vertex metadata: {path}")
    required = ("x", "y", "z", "red", "green", "blue")
    if any(name not in properties for name in required):
        raise SystemExit(f"Initialization PLY must contain XYZ and RGB properties: {path}")
    property_indexes = {name: properties.index(name) for name in required}
    rows = lines[data_index : data_index + vertex_count]
    if len(rows) != vertex_count:
        raise SystemExit(f"Initialization PLY declares {vertex_count} vertices but contains {len(rows)}")
    points: List[ColoredPoint] = []
    for row in rows:
        values = row.split()
        try:
            xyz = tuple(float(values[property_indexes[name]]) for name in ("x", "y", "z"))
            rgb = tuple(int(values[property_indexes[name]]) for name in ("red", "green", "blue"))
        except (IndexError, TypeError, ValueError) as exc:
            raise SystemExit(f"Initialization PLY contains an invalid vertex row: {row}") from exc
        if not all(math.isfinite(value) for value in xyz):
            raise SystemExit("Initialization PLY contains a non-finite vertex")
        points.append(ColoredPoint(xyz=xyz, rgb=rgb))
    return points


def _transpose3(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(matrix[column][row]) for column in range(3)] for row in range(3)]


def _multiply3(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [sum(float(left[row][index]) * float(right[index][column]) for index in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _transform3(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(sum(float(matrix[row][index]) * float(vector[index]) for index in range(3)) for row in range(3))


def colmap_to_reference_similarity(
    hybrid_artifact: Mapping[str, Any],
) -> Tuple[float, List[List[float]], Tuple[float, float, float]]:
    """Recover the COLMAP-world to reference-ARKit similarity from A_hybrid."""
    captures = [item for item in hybrid_artifact.get("captures") or [] if isinstance(item, Mapping)]
    if len(captures) != 1:
        raise SystemExit("Merged initialization currently requires one continuous A_hybrid capture.")
    capture = captures[0]
    scale_colmap_per_meter = float(capture.get("scale_colmap_units_per_meter") or 0.0)
    if not math.isfinite(scale_colmap_per_meter) or scale_colmap_per_meter <= 0:
        raise SystemExit("A_hybrid capture is missing a valid COLMAP-to-meter scale.")
    frames = [item for item in capture.get("frames") or [] if isinstance(item, Mapping)]
    if not frames:
        raise SystemExit("A_hybrid capture contains no frames.")
    first_colmap = validate_camera_to_world_matrix(frames[0].get("camera_to_world_colmap_opencv_row_major"))
    first_reference = validate_camera_to_world_matrix(frames[0].get("camera_to_world_reference_arkit_row_major"))
    camera_axes = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    rotation = _multiply3(
        _multiply3([row[:3] for row in first_reference[:3]], camera_axes),
        _transpose3([row[:3] for row in first_colmap[:3]]),
    )
    scale = 1.0 / scale_colmap_per_meter
    rotated_center = _transform3(rotation, [row[3] for row in first_colmap[:3]])
    translation = tuple(first_reference[row][3] - scale * rotated_center[row] for row in range(3))

    max_center_error = 0.0
    for frame in frames:
        colmap_pose = validate_camera_to_world_matrix(frame.get("camera_to_world_colmap_opencv_row_major"))
        reference_pose = validate_camera_to_world_matrix(frame.get("camera_to_world_reference_arkit_row_major"))
        predicted_rotation = _transform3(rotation, [row[3] for row in colmap_pose[:3]])
        predicted = tuple(scale * predicted_rotation[row] + translation[row] for row in range(3))
        expected = tuple(reference_pose[row][3] for row in range(3))
        max_center_error = max(
            max_center_error,
            math.sqrt(sum((predicted[index] - expected[index]) ** 2 for index in range(3))),
        )
    if max_center_error > 1e-3:
        raise SystemExit(
            f"Recovered COLMAP-to-ARKit similarity is inconsistent with A_hybrid ({max_center_error:.6f} m)."
        )
    return scale, rotation, translation


def transform_colmap_point_to_reference_nerfstudio(
    point: Point3D,
    *,
    scale: float,
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
) -> ColoredPoint:
    rotated = _transform3(rotation, point.xyz)
    reference = tuple(scale * rotated[index] + float(translation[index]) for index in range(3))
    # Reference ARKit is +Y-up. The training frame is right-handed +Z-up.
    return ColoredPoint(
        xyz=(reference[0], -reference[2], reference[1]),
        rgb=point.rgb,
    )


def _grid_key(position: Sequence[float], cell_size: float) -> Tuple[int, int, int]:
    return tuple(math.floor(float(value) / cell_size) for value in position)


def nearest_point_distance(
    position: Sequence[float],
    grid: Mapping[Tuple[int, int, int], Sequence[ColoredPoint]],
    *,
    cell_size: float,
) -> Optional[float]:
    center = _grid_key(position, cell_size)
    best_squared: Optional[float] = None
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            for offset_z in (-1, 0, 1):
                for candidate in grid.get(
                    (center[0] + offset_x, center[1] + offset_y, center[2] + offset_z), ()
                ):
                    squared = sum(
                        (float(position[index]) - candidate.xyz[index]) ** 2 for index in range(3)
                    )
                    if best_squared is None or squared < best_squared:
                        best_squared = squared
    return math.sqrt(best_squared) if best_squared is not None else None


def build_lidar_colmap_initialization(
    lidar_points: Sequence[ColoredPoint],
    colmap_points: Sequence[Point3D],
    hybrid_artifact: Mapping[str, Any],
    *,
    min_track_length: int = 3,
    max_reprojection_error_px: float = 1.0,
    duplicate_radius_meters: float = 0.03,
    max_surface_distance_meters: float = 0.15,
) -> Tuple[List[ColoredPoint], Dict[str, Any]]:
    """Merge LiDAR seeds with conservative, surface-supported COLMAP points."""
    if not lidar_points:
        raise SystemExit("Cannot merge COLMAP points into an empty LiDAR initialization.")
    scale, rotation, translation = colmap_to_reference_similarity(hybrid_artifact)
    grid: Dict[Tuple[int, int, int], List[ColoredPoint]] = {}
    for point in lidar_points:
        grid.setdefault(_grid_key(point.xyz, max_surface_distance_meters), []).append(point)

    quality_retained = 0
    surface_supported = 0
    duplicate_count = 0
    retained_colmap: List[ColoredPoint] = []
    retained_errors: List[float] = []
    retained_tracks: List[int] = []
    for point in colmap_points:
        if point.track_length < min_track_length or point.error > max_reprojection_error_px:
            continue
        quality_retained += 1
        transformed = transform_colmap_point_to_reference_nerfstudio(
            point,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        distance = nearest_point_distance(
            transformed.xyz,
            grid,
            cell_size=max_surface_distance_meters,
        )
        if distance is None or distance > max_surface_distance_meters:
            continue
        surface_supported += 1
        if distance <= duplicate_radius_meters:
            duplicate_count += 1
            continue
        retained_colmap.append(transformed)
        retained_errors.append(point.error)
        retained_tracks.append(point.track_length)

    merged = [*lidar_points, *retained_colmap]
    xs = [point.xyz[0] for point in merged]
    ys = [point.xyz[1] for point in merged]
    zs = [point.xyz[2] for point in merged]
    settings = {
        "min_colmap_track_length": min_track_length,
        "max_colmap_reprojection_error_px": max_reprojection_error_px,
        "duplicate_radius_meters": duplicate_radius_meters,
        "max_lidar_surface_distance_meters": max_surface_distance_meters,
    }
    details = {
        "policy": "LiDAR seeds plus filtered, LiDAR-surface-supported, non-duplicate COLMAP points",
        "settings": settings,
        "lidar_point_count": len(lidar_points),
        "colmap_input_point_count": len(colmap_points),
        "colmap_quality_retained_count": quality_retained,
        "colmap_surface_supported_count": surface_supported,
        "colmap_duplicate_removed_count": duplicate_count,
        "colmap_added_point_count": len(retained_colmap),
        "merged_point_count": len(merged),
        "colmap_added_reprojection_error_median": _median(retained_errors),
        "colmap_added_track_length_median": _median([float(value) for value in retained_tracks]),
        "colmap_units_per_meter": 1.0 / scale,
        "point_cloud_stats": {
            "count": len(merged),
            "xyz_min": [min(xs), min(ys), min(zs)],
            "xyz_max": [max(xs), max(ys), max(zs)],
        },
    }
    return merged, details


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    return ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def write_colored_point_cloud_ply(path: Path, points: Sequence[ColoredPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point in points:
            file.write(
                f"{point.xyz[0]:.9g} {point.xyz[1]:.9g} {point.xyz[2]:.9g} "
                f"{point.rgb[0]} {point.rgb[1]} {point.rgb[2]}\n"
            )


def build_hybrid_transforms(
    *,
    run_dir: Path,
    frames_dir: Path,
    hybrid_artifact: Mapping[str, Any],
    images: Sequence[ImagePose],
    cameras: Mapping[int, Camera],
    points: Sequence[Point3D],
    point_cloud_path: Path,
    expected_colmap_run_id: Optional[str] = None,
    lidar_initialization: Optional[Mapping[str, Any]] = None,
    merged_initialization: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = manifest_by_name(run_dir)
    if not manifest:
        raise SystemExit("Hybrid training requires reports/image_manifest.json.")
    validate_manifest_camera_count(cameras, manifest)
    selected_frames = hybrid_frames(hybrid_artifact, expected_colmap_run_id=expected_colmap_run_id)
    registered_by_name = {image.name: image for image in images}
    camera_ids_by_group: Dict[str, set[int]] = {}
    for image in images:
        entry = manifest.get(image.name)
        if entry:
            camera_ids_by_group.setdefault(manifest_camera_group_id(entry), set()).add(image.camera_id)

    output_frames: List[Dict[str, Any]] = []
    for selected in selected_frames:
        image_name = str(selected["image_name"])
        manifest_entry = manifest.get(image_name)
        if manifest_entry is None:
            raise SystemExit(f"Hybrid image is missing from image manifest: {image_name}")
        source_path = frames_dir / image_name
        if not source_path.exists():
            raise SystemExit(f"Hybrid image is missing from frames directory: {source_path}")
        actual_width, actual_height = image_size(source_path)
        declared_size = (int(selected.get("width") or 0), int(selected.get("height") or 0))
        if declared_size != (actual_width, actual_height):
            raise SystemExit(
                f"Hybrid image dimensions do not match pixels for {image_name}: "
                f"artifact={declared_size[0]}x{declared_size[1]}, image={actual_width}x{actual_height}"
            )

        registered_image = registered_by_name.get(image_name)
        group_id = manifest_camera_group_id(manifest_entry)
        if registered_image is not None:
            camera_id = registered_image.camera_id
        else:
            group_camera_ids = camera_ids_by_group.get(group_id, set())
            if len(group_camera_ids) == 1:
                camera_id = next(iter(group_camera_ids))
            elif len(cameras) == 1:
                camera_id = next(iter(cameras))
            else:
                raise SystemExit(
                    f"Could not resolve one COLMAP camera for propagated hybrid image {image_name} "
                    f"in camera group {group_id}."
                )
        camera = cameras.get(camera_id)
        if camera is None:
            raise SystemExit(f"Hybrid image {image_name} references missing camera id {camera_id}")
        if (camera.width, camera.height) != (actual_width, actual_height):
            raise SystemExit(
                f"COLMAP camera dimensions do not match hybrid image {image_name}: "
                f"camera={camera.width}x{camera.height}, image={actual_width}x{actual_height}"
            )

        if lidar_initialization is not None:
            transform_matrix = reference_arkit_camera_to_world_to_nerfstudio_transform(
                selected["camera_to_world_reference_arkit_row_major"]
            )
        else:
            transform_matrix = colmap_camera_to_world_to_nerfstudio_transform(
                selected["camera_to_world_colmap_opencv_row_major"]
            )
        output_frames.append(
            {
                "file_path": f"images/{image_name}",
                "transform_matrix": transform_matrix,
                "colmap_image_id": registered_image.image_id if registered_image else None,
                "colmap_camera_id": camera_id,
                "intrinsics_source": "base_colmap_camera_group",
                "hybrid_pose_source": selected["pose_source"],
                "hybrid_replacement_reasons": list(selected.get("replacement_reasons") or []),
                "capture_id": selected.get("capture_id"),
                "frame_index": selected.get("frame_index"),
                "alignment_classification": selected.get("alignment_classification"),
                "position_residual_meters": selected.get("position_residual_meters"),
                "angular_residual_degrees": selected.get("angular_residual_degrees"),
                "role": manifest_entry.get("role"),
                "camera_group": manifest_entry.get("camera_group"),
                "camera_group_id": group_id,
                "location": manifest_entry.get("location"),
                "source_id": manifest_entry.get("source_id"),
                **camera_intrinsics(camera),
            }
        )

    if lidar_initialization is not None:
        initialization_source = (
            "high_confidence_lidar_plus_filtered_colmap"
            if merged_initialization is not None
            else "high_confidence_lidar"
        )
        source = (
            "arkit_hybrid_with_lidar_colmap_initialization"
            if merged_initialization is not None
            else "arkit_hybrid_with_lidar_initialization"
        )
        coordinate_frame = "reference_arkit_gravity_aligned_nerfstudio_z_up"
        if merged_initialization is not None:
            initialization_stats = (
                merged_initialization.get("point_cloud_stats")
                if isinstance(merged_initialization.get("point_cloud_stats"), Mapping)
                else {}
            )
        else:
            initialization_stats = (
                lidar_initialization.get("point_cloud_stats")
                if isinstance(lidar_initialization.get("point_cloud_stats"), Mapping)
                else {}
            )
    else:
        initialization_source = "base_colmap_sparse_txt"
        source = "arkit_hybrid_camera_set"
        coordinate_frame = "colmap_derived_nerfstudio_axes"
        initialization_stats = point_cloud_stats(points)
    return {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "ply_file_path": point_cloud_path.name,
        "frames": output_frames,
        "buildvision3d": {
            "source": source,
            "camera_source": "arkit_hybrid",
            "hybrid_source_run_id": hybrid_artifact.get("source_run_id"),
            "hybrid_selection_sha256": hybrid_artifact.get("selection_sha256"),
            "hybrid_artifact_uri": hybrid_artifact.get("artifact_uri"),
            "frame_count": len(output_frames),
            "colmap_registered_count": sum(
                1 for frame in output_frames if frame["hybrid_pose_source"] == "colmap_registered"
            ),
            "arkit_propagated_count": sum(
                1 for frame in output_frames if frame["hybrid_pose_source"] == "arkit_propagated"
            ),
            "camera_count": len(cameras),
            "colmap_sparse_point_count": len(points),
            "initialization_source": initialization_source,
            "coordinate_frame": coordinate_frame,
            "lidar_initialization_selection_sha256": (
                lidar_initialization.get("selection_sha256") if lidar_initialization is not None else None
            ),
            "lidar_initialization_artifact_uri": (
                lidar_initialization.get("artifact_uri") if lidar_initialization is not None else None
            ),
            "initialization_merge": dict(merged_initialization) if merged_initialization is not None else None,
            "point_cloud_path": point_cloud_path.name,
            "point_cloud_stats": initialization_stats,
            "multi_camera": len(cameras) > 1,
            "intrinsics_mode": "base_colmap_camera_group",
        },
    }


def point_cloud_stats(points: Sequence[Point3D]) -> Dict[str, Any]:
    converted = [colmap_point_to_nerfstudio(point) for point in points]
    xs = [point[0] for point in converted]
    ys = [point[1] for point in converted]
    zs = [point[2] for point in converted]
    errors = sorted(point.error for point in points)
    midpoint = len(errors) // 2
    median_error = errors[midpoint] if len(errors) % 2 else (errors[midpoint - 1] + errors[midpoint]) / 2.0
    return {
        "count": len(points),
        "xyz_min": [min(xs), min(ys), min(zs)],
        "xyz_max": [max(xs), max(ys), max(zs)],
        "reprojection_error_min": errors[0],
        "reprojection_error_median": median_error,
        "reprojection_error_max": errors[-1],
    }


def write_point_cloud_ply(path: Path, points: Sequence[Point3D]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point in points:
            x, y, z = colmap_point_to_nerfstudio(point)
            r, g, b = point.rgb
            file.write(f"{x:.9g} {y:.9g} {z:.9g} {r} {g} {b}\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = args.run.expanduser()
    frames_dir = resolve_under_run(run_dir, args.frames_dir)
    data_dir = resolve_under_run(run_dir, args.data_dir)
    colmap_model_dir = resolve_under_run(run_dir, args.colmap_model_dir)
    colmap_text_dir = colmap_text_dir_from_model_dir(colmap_model_dir)
    hybrid_camera_set_path = (
        resolve_under_run(run_dir, args.hybrid_camera_set) if args.hybrid_camera_set is not None else None
    )
    lidar_initialization_path = (
        resolve_under_run(run_dir, args.lidar_initialization) if args.lidar_initialization is not None else None
    )
    lidar_initialization_ply_path = (
        resolve_under_run(run_dir, args.lidar_initialization_ply)
        if args.lidar_initialization_ply is not None
        else None
    )
    if (lidar_initialization_path is None) != (lidar_initialization_ply_path is None):
        raise SystemExit("--lidar-initialization and --lidar-initialization-ply must be supplied together.")
    if lidar_initialization_path is not None and hybrid_camera_set_path is None:
        raise SystemExit("LiDAR initialization requires --hybrid-camera-set.")
    if args.merge_colmap_initialization and lidar_initialization_path is None:
        raise SystemExit("--merge-colmap-initialization requires LiDAR initialization inputs.")

    cameras_path = colmap_text_dir / "cameras.txt"
    images_path = colmap_text_dir / "images.txt"
    points_path = colmap_text_dir / "points3D.txt"
    if not cameras_path.exists() or not images_path.exists() or not points_path.exists():
        raise SystemExit(
            "COLMAP text export is required for multi-camera Nerfstudio preparation. "
            f"Expected {cameras_path}, {images_path}, and {points_path}. Rerun COLMAP with --export-text."
        )
    if not frames_dir.exists():
        raise SystemExit(f"Frames directory does not exist: {frames_dir}")
    if data_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output data directory already exists: {data_dir}. Use --overwrite to replace it.")
        shutil.rmtree(data_dir)

    cameras = read_cameras(cameras_path)
    images = read_images(images_path)
    points = read_points3d(points_path)
    hybrid_artifact = None
    selected_hybrid_frames: List[Dict[str, Any]] = []
    if hybrid_camera_set_path is not None:
        if not hybrid_camera_set_path.exists():
            raise SystemExit(f"Hybrid camera-set file does not exist: {hybrid_camera_set_path}")
        hybrid_artifact = json.loads(hybrid_camera_set_path.read_text(encoding="utf-8"))
        if not isinstance(hybrid_artifact, dict):
            raise SystemExit("Hybrid camera-set JSON root must be an object.")
        selected_hybrid_frames = hybrid_frames(
            hybrid_artifact,
            expected_colmap_run_id=args.expected_colmap_run_id,
        )
    images_dir = data_dir / "images"
    point_cloud_name = (
        "lidar_colmap_initialization.ply"
        if args.merge_colmap_initialization
        else "lidar_initialization.ply"
        if lidar_initialization_path is not None
        else "colmap_points3D.ply"
    )
    point_cloud_path = data_dir / point_cloud_name
    print(f"Preparing multi-camera Nerfstudio dataset from {colmap_text_dir}")
    print(f"  registered images: {len(images)}")
    if hybrid_artifact is not None:
        print(f"  selected hybrid images: {len(selected_hybrid_frames)}")
    print(f"  cameras: {len(cameras)}")
    print(f"  sparse points: {len(points)}")
    print(f"  output: {data_dir}")

    data_dir.mkdir(parents=True, exist_ok=True)
    if hybrid_artifact is not None:
        copy_named_images([str(frame["image_name"]) for frame in selected_hybrid_frames], frames_dir, images_dir)
    else:
        copy_registered_images(images, frames_dir, images_dir)
    build_downscales(images_dir, int(args.num_downscales))
    lidar_initialization = None
    merged_initialization = None
    if lidar_initialization_path is not None and lidar_initialization_ply_path is not None:
        if not lidar_initialization_path.exists():
            raise SystemExit(f"LiDAR initialization metadata does not exist: {lidar_initialization_path}")
        lidar_initialization = json.loads(lidar_initialization_path.read_text(encoding="utf-8"))
        if not isinstance(lidar_initialization, dict) or hybrid_artifact is None:
            raise SystemExit("LiDAR initialization metadata root must be an object.")
        validate_lidar_initialization(lidar_initialization, hybrid_artifact, lidar_initialization_ply_path)
        if args.merge_colmap_initialization:
            lidar_points = read_colored_ascii_ply(lidar_initialization_ply_path)
            merged_points, merged_initialization = build_lidar_colmap_initialization(
                lidar_points,
                points,
                hybrid_artifact,
            )
            write_colored_point_cloud_ply(point_cloud_path, merged_points)
            print(
                "  merged initialization: "
                f"{merged_initialization['lidar_point_count']} LiDAR + "
                f"{merged_initialization['colmap_added_point_count']} filtered COLMAP = "
                f"{merged_initialization['merged_point_count']} points"
            )
        else:
            shutil.copy2(lidar_initialization_ply_path, point_cloud_path)
    else:
        write_point_cloud_ply(point_cloud_path, points)
    print(f"Wrote {point_cloud_path}")
    if hybrid_artifact is not None:
        transforms = build_hybrid_transforms(
            run_dir=run_dir,
            frames_dir=frames_dir,
            hybrid_artifact=hybrid_artifact,
            images=images,
            cameras=cameras,
            points=points,
            point_cloud_path=point_cloud_path,
            expected_colmap_run_id=args.expected_colmap_run_id,
            lidar_initialization=lidar_initialization,
            merged_initialization=merged_initialization,
        )
    else:
        transforms = build_transforms(
            run_dir=run_dir,
            frames_dir=frames_dir,
            images=images,
            cameras=cameras,
            points=points,
            point_cloud_path=point_cloud_path,
        )
    write_json(data_dir / "transforms.json", transforms)
    print(f"Wrote {data_dir / 'transforms.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
