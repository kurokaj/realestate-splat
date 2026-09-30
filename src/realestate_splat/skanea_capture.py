"""Validation and manifest registration for finalized Skanea RGB-D captures."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class SkaneaCaptureError(ValueError):
    """Raised when a Skanea capture cannot be safely registered."""


@dataclass(frozen=True)
class ValidatedSkaneaCapture:
    directory: Path
    manifest: dict[str, Any]
    capture_id: str
    capture_slug: str
    manifest_sha256: str
    frames: list[dict[str, Any]]
    derived_artifacts: list[str]


def load_skanea_capture(capture_dir: Path) -> ValidatedSkaneaCapture:
    """Validate one finalized schema-v2 Skanea session without changing it."""
    directory = capture_dir.expanduser().resolve()
    if not directory.is_dir():
        raise SkaneaCaptureError(f"Skanea capture directory does not exist: {directory}")
    if (directory / ".active-capture").exists():
        raise SkaneaCaptureError("Skanea capture is still marked active and cannot be imported")

    manifest_path = directory / "capture.json"
    manifest_bytes = _read_file(manifest_path, "capture manifest")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkaneaCaptureError(f"Skanea capture.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SkaneaCaptureError("Skanea capture.json must contain a JSON object")
    if manifest.get("schemaVersion") != 2:
        raise SkaneaCaptureError(
            f"Unsupported Skanea schemaVersion {manifest.get('schemaVersion')!r}; only finalized schema 2 is accepted"
        )
    capture_id = str(manifest.get("sessionID") or "").strip()
    if not capture_id:
        raise SkaneaCaptureError("Skanea capture.json is missing sessionID")
    if not manifest.get("endedAt"):
        raise SkaneaCaptureError("Skanea capture has no endedAt value and is not finalized")
    frames_value = manifest.get("frames")
    if not isinstance(frames_value, list) or not frames_value:
        raise SkaneaCaptureError("Skanea capture.json must contain at least one frame")

    seen_indices: set[int] = set()
    frames: list[dict[str, Any]] = []
    for position, value in enumerate(frames_value):
        if not isinstance(value, dict):
            raise SkaneaCaptureError(f"Skanea frame at position {position} is not an object")
        frame = _validate_frame(directory, value, position)
        index = frame["index"]
        if index in seen_indices:
            raise SkaneaCaptureError(f"Skanea capture contains duplicate frame index {index}")
        seen_indices.add(index)
        frames.append(frame)

    ignored_artifact_names = {".DS_Store", ".active-capture", ".capture-frames.ndjson"}
    artifact_roots = (directory / "spatial", directory / "derived")
    derived_artifacts = [
        path.relative_to(directory).as_posix()
        for artifact_root in artifact_roots
        if artifact_root.is_dir()
        for path in sorted(artifact_root.rglob("*"))
        if path.is_file() and path.name not in ignored_artifact_names
    ]

    registered_files = {"capture.json", *derived_artifacts}
    for frame in frames:
        registered_files.add(frame["rgbFile"])
        if frame.get("depthFile"):
            registered_files.add(frame["depthFile"])
        if frame.get("confidenceFile"):
            registered_files.add(frame["confidenceFile"])
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise SkaneaCaptureError(f"Skanea capture must not contain symbolic links: {path}")
        if not path.is_file() or path.name in ignored_artifact_names:
            continue
        relative_path = path.relative_to(directory).as_posix()
        if relative_path not in registered_files:
            raise SkaneaCaptureError(f"Skanea capture contains an unregistered file: {relative_path}")

    return ValidatedSkaneaCapture(
        directory=directory,
        manifest=manifest,
        capture_id=capture_id,
        capture_slug=_safe_id(capture_id),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        frames=sorted(frames, key=lambda item: item["index"]),
        derived_artifacts=derived_artifacts,
    )


def build_skanea_manifest_fragment(
    *,
    project_id: str,
    capture_dir: Path,
    destination_uri: str,
    location: str,
    created_at: str,
) -> tuple[dict[str, Any], ValidatedSkaneaCapture]:
    """Build the project-manifest fragment for one validated capture."""
    capture = load_skanea_capture(capture_dir)
    if not location.strip():
        raise SkaneaCaptureError("Skanea import location must not be empty")
    location_id = _safe_id(location)
    raw_root = f"skanea/{capture.capture_slug}"
    destination_base = destination_uri.rstrip("/")
    manifest_relative_path = f"{raw_root}/capture.json"
    manifest_uri = f"{destination_base}/{manifest_relative_path}"
    data_description = capture.manifest.get("dataDescription")
    if not isinstance(data_description, dict):
        data_description = {}

    sources: list[dict[str, Any]] = []
    for frame in capture.frames:
        index = frame["index"]
        source_id = f"skanea_{capture.capture_slug}_{index:06d}"
        rgb_relative_path = f"{raw_root}/{frame['rgbFile']}"
        rgbd: dict[str, Any] = {
            "schema_version": 1,
            "capture_id": capture.capture_id,
            "frame_index": index,
            "timestamp_seconds": frame["timestamp"],
            "capture_manifest_relative_path": manifest_relative_path,
            "capture_manifest_uri": manifest_uri,
            "camera_intrinsics": frame["cameraIntrinsics"],
            "camera_intrinsics_layout": data_description.get("cameraIntrinsicsLayout"),
            "camera_transform": frame["cameraTransform"],
            "camera_transform_layout": data_description.get("cameraTransformLayout"),
            "world_coordinate_system": data_description.get("worldCoordinateSystem"),
        }
        if frame.get("depthFile"):
            relative_path = f"{raw_root}/{frame['depthFile']}"
            rgbd["depth"] = {
                "relative_path": relative_path,
                "uri": f"{destination_base}/{relative_path}",
                "width": frame["depthWidth"],
                "height": frame["depthHeight"],
                "encoding": data_description.get("depthEncoding") or "Float32 little-endian, meters",
                "invalid_values": data_description.get("depthInvalidValues"),
            }
        if frame.get("confidenceFile"):
            relative_path = f"{raw_root}/{frame['confidenceFile']}"
            rgbd["confidence"] = {
                "relative_path": relative_path,
                "uri": f"{destination_base}/{relative_path}",
                "width": frame["confidenceWidth"],
                "height": frame["confidenceHeight"],
                "encoding": data_description.get("confidenceEncoding") or "UInt8 ARConfidenceLevel",
            }
        sources.append(
            {
                "source_id": source_id,
                "role": "coverage_image",
                "relative_path": rgb_relative_path,
                "uri": f"{destination_base}/{rgb_relative_path}",
                "location": location_id,
                "related_sources": None,
                "camera_group": f"skanea_{capture.capture_slug}",
                "colmap_policy": "include",
                "width": frame["rgbWidth"],
                "height": frame["rgbHeight"],
                "duration_seconds": None,
                "source_kind": "skanea_rgbd_frame",
                "preprocess_name": f"{source_id}.jpg",
                "rgbd": rgbd,
            }
        )

    capture_record = {
        "capture_id": capture.capture_id,
        "kind": "skanea_rgbd",
        "schema_version": capture.manifest.get("schemaVersion"),
        "manifest_sha256": capture.manifest_sha256,
        "manifest_relative_path": manifest_relative_path,
        "manifest_uri": manifest_uri,
        "raw_root_relative_path": raw_root,
        "location": location_id,
        "capture_mode": capture.manifest.get("captureMode"),
        "app_version": capture.manifest.get("appVersion"),
        "device_model": capture.manifest.get("deviceModel"),
        "camera_lens": capture.manifest.get("cameraLens"),
        "started_at": capture.manifest.get("startedAt"),
        "ended_at": capture.manifest.get("endedAt"),
        "frame_count": len(capture.frames),
        "derived_artifacts": [f"{raw_root}/{path}" for path in capture.derived_artifacts],
    }
    return (
        {
            "schema_version": 4,
            "project_id": project_id,
            "created_at": created_at,
            "input_dir": str(capture.directory),
            "base_uri": destination_base,
            "sources": sources,
            "captures": [capture_record],
        },
        capture,
    )


def registered_artifact_paths(manifest: dict[str, Any]) -> set[str]:
    """Return every raw object explicitly registered by a project manifest."""
    paths = {"sources_manifest.json"}
    for source in manifest.get("sources", []):
        if not isinstance(source, dict) or not source.get("relative_path"):
            raise SkaneaCaptureError("project manifest contains an invalid source entry")
        _add_registered_path(paths, source.get("relative_path"), "source relative_path")
        rgbd = source.get("rgbd")
        if not isinstance(rgbd, dict):
            continue
        _add_registered_path(paths, rgbd.get("capture_manifest_relative_path"), "capture manifest path")
        for key in ("depth", "confidence"):
            attachment = rgbd.get(key)
            if isinstance(attachment, dict):
                _add_registered_path(paths, attachment.get("relative_path"), f"{key} relative_path")
    for capture in manifest.get("captures", []):
        if not isinstance(capture, dict):
            continue
        _add_registered_path(paths, capture.get("manifest_relative_path"), "capture manifest path")
        artifacts = capture.get("derived_artifacts")
        if artifacts is None:
            continue
        if not isinstance(artifacts, list):
            raise SkaneaCaptureError("capture derived_artifacts must be a list")
        for artifact in artifacts:
            _add_registered_path(paths, artifact, "derived artifact path")
    return paths


def _validate_frame(directory: Path, value: dict[str, Any], position: int) -> dict[str, Any]:
    index = _integer(value.get("index"), f"frame {position} index", minimum=0)
    timestamp = _number(value.get("timestamp"), f"frame {index} timestamp")
    rgb_path = _relative_path(value.get("rgbFile"), f"frame {index} rgbFile")
    rgb_width = _integer(value.get("rgbWidth"), f"frame {index} rgbWidth", minimum=1)
    rgb_height = _integer(value.get("rgbHeight"), f"frame {index} rgbHeight", minimum=1)
    _read_file(directory / rgb_path, f"frame {index} RGB file")

    depth_path = _optional_relative_path(value.get("depthFile"), f"frame {index} depthFile")
    depth_width = _optional_dimension(value.get("depthWidth"), depth_path, f"frame {index} depthWidth")
    depth_height = _optional_dimension(value.get("depthHeight"), depth_path, f"frame {index} depthHeight")
    if depth_path is not None:
        _require_exact_size(directory / depth_path, depth_width * depth_height * 4, f"frame {index} depth file")

    confidence_path = _optional_relative_path(value.get("confidenceFile"), f"frame {index} confidenceFile")
    confidence_width = _optional_dimension(value.get("confidenceWidth"), confidence_path, f"frame {index} confidenceWidth")
    confidence_height = _optional_dimension(value.get("confidenceHeight"), confidence_path, f"frame {index} confidenceHeight")
    if confidence_path is not None:
        if depth_path is None:
            raise SkaneaCaptureError(f"frame {index} has confidence without depth")
        _require_exact_size(
            directory / confidence_path,
            confidence_width * confidence_height,
            f"frame {index} confidence file",
        )

    return {
        "index": index,
        "timestamp": timestamp,
        "rgbFile": rgb_path.as_posix(),
        "rgbWidth": rgb_width,
        "rgbHeight": rgb_height,
        "depthFile": depth_path.as_posix() if depth_path else None,
        "depthWidth": depth_width,
        "depthHeight": depth_height,
        "confidenceFile": confidence_path.as_posix() if confidence_path else None,
        "confidenceWidth": confidence_width,
        "confidenceHeight": confidence_height,
        "cameraIntrinsics": _number_vector(value.get("cameraIntrinsics"), 9, f"frame {index} cameraIntrinsics"),
        "cameraTransform": _number_vector(value.get("cameraTransform"), 16, f"frame {index} cameraTransform"),
    }


def _safe_id(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_") or "capture"


def _read_file(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise SkaneaCaptureError(f"Missing {label}: {path}")
    return path.read_bytes()


def _relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SkaneaCaptureError(f"{label} must be a non-empty relative path")
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SkaneaCaptureError(f"{label} is unsafe: {value}")
    return path


def _optional_relative_path(value: Any, label: str) -> Optional[Path]:
    if value is None:
        return None
    return _relative_path(value, label)


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SkaneaCaptureError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SkaneaCaptureError(f"{label} must be a finite number")
    return float(value)


def _number_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SkaneaCaptureError(f"{label} must contain exactly {length} values")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _optional_dimension(value: Any, path: Optional[Path], label: str) -> Optional[int]:
    if path is None:
        if value is not None:
            raise SkaneaCaptureError(f"{label} is present without its file")
        return None
    return _integer(value, label, minimum=1)


def _require_exact_size(path: Path, expected: int, label: str) -> None:
    if not path.is_file():
        raise SkaneaCaptureError(f"Missing {label}: {path}")
    actual = path.stat().st_size
    if actual != expected:
        raise SkaneaCaptureError(f"{label} has {actual} bytes; expected {expected}")


def _add_registered_path(paths: set[str], value: Any, label: str) -> None:
    if value is None:
        return
    paths.add(_relative_path(value, label).as_posix())
