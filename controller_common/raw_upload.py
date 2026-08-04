"""Raw media upload helpers for the controller API."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.cli import write_json  # noqa: E402
from realestate_splat.media_manifest import build_sources_manifest  # noqa: E402
from realestate_splat.storage import copy_file, sync_directory  # noqa: E402


DEFAULT_EXCLUDES = [
    ".DS_Store",
    ".gitkeep",
    "__pycache__/*",
    "*.pyc",
]


def safe_relative_upload_path(filename: str) -> Path:
    cleaned = filename.replace("\\", "/").lstrip("/")
    parts = []
    for part in Path(cleaned).parts:
        if part in {"", ".", ".."}:
            continue
        parts.append(part)
    if not parts:
        raise ValueError("Uploaded file must have a filename")
    return Path(*parts)


def write_upload_file(root: Path, filename: str, stream: BinaryIO) -> Path:
    relative_path = safe_relative_upload_path(filename)
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(stream, handle)
    return destination


def upload_raw_directory(
    *,
    project_id: str,
    input_dir: Path,
    destination_uri: str,
    endpoint_url: Optional[str],
    metadata_overrides: Optional[dict[str, dict[str, Any]]] = None,
    delete: bool = False,
    dry_run: bool = False,
) -> dict:
    manifest = build_sources_manifest(project_id, input_dir, destination_uri)
    apply_manifest_overrides(manifest, metadata_overrides or {})
    manifest_path = input_dir / "sources_manifest.json"
    write_json(manifest_path, manifest)
    sync_directory(
        input_dir,
        destination_uri,
        endpoint_url=endpoint_url,
        delete=delete,
        dry_run=dry_run,
        exclude=DEFAULT_EXCLUDES,
    )
    copy_file(
        manifest_path,
        f"{destination_uri.rstrip('/')}/sources_manifest.json",
        endpoint_url=endpoint_url,
        dry_run=dry_run,
    )
    return manifest


def metadata_overrides_by_filename(metadata: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not metadata:
        return {}
    files = metadata.get("files")
    if not isinstance(files, list):
        raise ValueError("metadata_json.files must be a list")
    overrides = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("metadata_json.files entries must be objects")
        filename = entry.get("filename") or entry.get("relative_path")
        if not filename:
            raise ValueError("metadata_json.files entries require filename or relative_path")
        relative_path = safe_relative_upload_path(str(filename)).as_posix()
        overrides[relative_path] = dict(entry)
    return overrides


def apply_manifest_overrides(manifest: dict, overrides: dict[str, dict[str, Any]]) -> None:
    if not overrides:
        return
    sources = manifest.get("sources") or []
    sources_by_path = {source.get("relative_path"): source for source in sources}
    allowed_fields = {
        "source_id",
        "role",
        "location",
        "related_sources",
        "camera_group",
        "colmap_policy",
    }
    for relative_path, override in overrides.items():
        source = sources_by_path.get(relative_path)
        if source is None:
            raise ValueError(f"metadata_json references an uploaded file that was not discovered: {relative_path}")
        original_role = source.get("role")
        for field in allowed_fields:
            if field in override:
                source[field] = override[field]
        normalize_overridden_source(source, override, original_role)


def normalize_overridden_source(source: dict[str, Any], override: dict[str, Any], original_role: Optional[str]) -> None:
    role = source.get("role")
    location = source.get("location")
    role_changed = original_role != role
    if role == "hero_image":
        if "source_id" not in override and (role_changed or not source.get("source_id")):
            stem = Path(source.get("relative_path", "hero")).stem
            source["source_id"] = f"hero_{location}_{stem}" if location else f"hero_{stem}"
        if "camera_group" not in override and (role_changed or not source.get("camera_group")):
            source["camera_group"] = f"hero_{location}" if location else "hero_images"
        if "colmap_policy" not in override:
            source["colmap_policy"] = "optional"
    elif role == "coverage_image":
        if "camera_group" not in override and (role_changed or not source.get("camera_group")):
            source["camera_group"] = "coverage_images"
        if "colmap_policy" not in override:
            source["colmap_policy"] = "include"
    elif role == "coverage_video":
        if "camera_group" not in override and (role_changed or not source.get("camera_group")):
            source["camera_group"] = f"video_{source.get('source_id')}"
        if "colmap_policy" not in override:
            source["colmap_policy"] = "include"


def manifest_summary(manifest: dict) -> dict:
    sources = manifest.get("sources") or []
    counts: dict[str, int] = {}
    for source in sources:
        role = source.get("role", "unknown")
        counts[role] = counts.get(role, 0) + 1
    return {
        "project_id": manifest.get("project_id"),
        "base_uri": manifest.get("base_uri"),
        "source_count": len(sources),
        "role_counts": counts,
    }


def uploaded_file_names(paths: Iterable[Path], root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in paths]
