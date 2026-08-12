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
    images_dir = data_dir / "images"
    point_cloud_path = data_dir / "colmap_points3D.ply"
    print(f"Preparing multi-camera Nerfstudio dataset from {colmap_text_dir}")
    print(f"  registered images: {len(images)}")
    print(f"  cameras: {len(cameras)}")
    print(f"  sparse points: {len(points)}")
    print(f"  output: {data_dir}")

    data_dir.mkdir(parents=True, exist_ok=True)
    copy_registered_images(images, frames_dir, images_dir)
    build_downscales(images_dir, int(args.num_downscales))
    write_point_cloud_ply(point_cloud_path, points)
    print(f"Wrote {point_cloud_path}")
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
