"""Raw media upload helpers for the controller API."""

from __future__ import annotations

import shutil
import sys
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.cli import write_json  # noqa: E402
from realestate_splat.media_manifest import build_sources_manifest  # noqa: E402
from realestate_splat.storage import copy_file, delete_file, sync_directory  # noqa: E402


DEFAULT_EXCLUDES = [
    ".DS_Store",
    ".gitkeep",
    "__pycache__/*",
    "*.pyc",
    "sources_manifest.json",
]


class DuplicateCoverageVideoError(ValueError):
    def __init__(self, location: str, existing: list[str]) -> None:
        self.location = location
        self.existing = existing
        super().__init__(
            f"A coverage video already exists for location '{location}': {', '.join(existing)}"
        )


class CoverageSourceConflictError(ValueError):
    def __init__(self, location: str, roles: list[str]) -> None:
        self.location = location
        self.roles = roles
        super().__init__(
            f"Location '{location}' cannot mix coverage video and coverage images: {', '.join(roles)}"
        )


def safe_relative_upload_path(filename: str | Path) -> Path:
    cleaned = str(filename).replace("\\", "/").lstrip("/")
    parts = []
    for part in Path(cleaned).parts:
        if part in {"", ".", ".."}:
            continue
        parts.append(part)
    if not parts:
        raise ValueError("Uploaded file must have a filename")
    return Path(*parts)


def write_upload_file(root: Path, filename: str | Path, stream: BinaryIO) -> Path:
    relative_path = safe_relative_upload_path(filename)
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(stream, handle)
    return destination


def grouped_upload_path(filename: str | Path, metadata: Optional[dict[str, Any]] = None) -> Path:
    original = safe_relative_upload_path(filename)
    metadata = metadata or {}
    role = str(metadata.get("role") or "").strip()
    if role == "auto":
        role = ""
    if not role:
        role = "coverage_video" if original.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi", ".mkv"} else "coverage_image"
    location = str(metadata.get("location") or "unassigned").strip()
    location = safe_relative_upload_path(location).as_posix().replace("/", "_")
    root = "hero" if role == "hero_image" else "coverage"
    return Path(root) / location / original.name


def upload_raw_directory(
    *,
    project_id: str,
    input_dir: Path,
    destination_uri: str,
    endpoint_url: Optional[str],
    metadata_overrides: Optional[dict[str, dict[str, Any]]] = None,
    delete: bool = False,
    override_video: bool = False,
    dry_run: bool = False,
) -> dict:
    manifest = build_sources_manifest(project_id, input_dir, destination_uri)
    apply_manifest_overrides(manifest, metadata_overrides or {})
    loaded_at = utc_now()
    for source in manifest.get("sources", []):
        source["loaded_at"] = loaded_at
    existing = load_manifest(destination_uri, endpoint_url) if not dry_run else {}
    existing_sources = existing.get("sources") if isinstance(existing.get("sources"), list) else []
    incoming_sources = manifest.get("sources") if isinstance(manifest.get("sources"), list) else []
    duplicate_videos = duplicate_coverage_videos(existing_sources + incoming_sources)
    incoming_duplicate_videos = duplicate_coverage_videos(incoming_sources)
    if incoming_duplicate_videos:
        location, paths = next(iter(incoming_duplicate_videos.items()))
        raise DuplicateCoverageVideoError(location, paths)
    coverage_conflicts = mixed_coverage_sources(existing_sources + incoming_sources)
    incoming_coverage_conflicts = mixed_coverage_sources(incoming_sources)
    if incoming_coverage_conflicts:
        location, roles = next(iter(incoming_coverage_conflicts.items()))
        raise CoverageSourceConflictError(location, roles)
    if coverage_conflicts and not override_video:
        location, roles = next(iter(coverage_conflicts.items()))
        raise CoverageSourceConflictError(location, roles)
    if duplicate_videos and not override_video:
        location, paths = next(iter(duplicate_videos.items()))
        raise DuplicateCoverageVideoError(location, paths)
    deleted_group_keys: set[str] = set()
    if not dry_run:
        if (duplicate_videos or coverage_conflicts) and override_video:
            replacement_locations = set(duplicate_videos) | set(coverage_conflicts)
            deleted_group_keys = {
                source_group_key(source)
                for source in existing_sources
                if isinstance(source, dict)
                and source.get("role") in {"coverage_video", "coverage_image"}
                and str(source.get("location") or "unassigned") in replacement_locations
            }
            old_paths = {
                source.get("relative_path")
                for source in existing_sources
                if isinstance(source, dict)
                and source.get("role") in {"coverage_video", "coverage_image"}
                and str(source.get("location") or "unassigned") in replacement_locations
            }
            for old_path in old_paths - {source.get("relative_path") for source in incoming_sources}:
                if old_path:
                    delete_file(f"{destination_uri.rstrip('/')}/{old_path}", endpoint_url=endpoint_url)
            existing = {
                **existing,
                "sources": [source for source in existing_sources if source.get("relative_path") not in old_paths],
            }
        manifest = merge_with_existing_manifest(manifest, destination_uri, endpoint_url, existing=existing)
    stale_reasons = dict(existing.get("preprocess_stale_reasons") or {})
    for group_key in deleted_group_keys:
        stale_reasons[group_key] = "deleted"
    for source in incoming_sources:
        if isinstance(source, dict):
            stale_reasons[source_group_key(source)] = "uploaded"
    manifest["preprocess_stale_reasons"] = stale_reasons
    manifest["preprocess_stale_groups"] = sorted(stale_reasons)
    sync_directory(
        input_dir,
        destination_uri,
        endpoint_url=endpoint_url,
        delete=delete,
        dry_run=dry_run,
        exclude=DEFAULT_EXCLUDES,
    )
    with tempfile.TemporaryDirectory(prefix="buildvision3d-raw-manifest-") as temp_dir:
        manifest_path = Path(temp_dir) / "sources_manifest.json"
        write_json(manifest_path, manifest)
        copy_file(
            manifest_path,
            f"{destination_uri.rstrip('/')}/sources_manifest.json",
            endpoint_url=endpoint_url,
            dry_run=dry_run,
        )
    return manifest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(destination_uri: str, endpoint_url: Optional[str]) -> dict[str, Any]:
    manifest_uri = f"{destination_uri.rstrip('/')}/sources_manifest.json"
    with tempfile.TemporaryDirectory(prefix="buildvision3d-load-manifest-") as temp_dir:
        manifest_path = Path(temp_dir) / "sources_manifest.json"
        try:
            copy_file(manifest_uri, manifest_path, endpoint_url=endpoint_url)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                return {}
            raise RuntimeError(f"Existing raw sources manifest could not be loaded: {manifest_uri}") from exc
        except FileNotFoundError:
            return {}
        except Exception as exc:
            raise RuntimeError(f"Existing raw sources manifest could not be loaded: {manifest_uri}") from exc
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Existing raw sources manifest is not valid JSON: {manifest_uri}") from exc
    return payload if isinstance(payload, dict) else {}


def merge_with_existing_manifest(
    incoming: dict[str, Any],
    destination_uri: str,
    endpoint_url: Optional[str],
    *,
    existing: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    existing = existing if existing is not None else load_manifest(destination_uri, endpoint_url)
    existing_sources = existing.get("sources") if isinstance(existing.get("sources"), list) else []
    incoming_sources = incoming.get("sources") if isinstance(incoming.get("sources"), list) else []
    by_path = {
        source.get("relative_path"): source
        for source in existing_sources
        if isinstance(source, dict) and source.get("relative_path")
    }
    for source in incoming_sources:
        if isinstance(source, dict) and source.get("relative_path"):
            by_path[source["relative_path"]] = source
    return {
        **incoming,
        "schema_version": max(int(existing.get("schema_version") or 0), int(incoming.get("schema_version") or 0), 3),
        "created_at": existing.get("created_at") or incoming.get("created_at"),
        "updated_at": utc_now(),
        "sources": [by_path[path] for path in sorted(by_path)],
    }


def source_group_key(source: dict[str, Any]) -> str:
    return f"location:{source.get('location') or 'unassigned'}"


def duplicate_coverage_videos(sources: Iterable[Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for source in sources:
        if not isinstance(source, dict) or source.get("role") != "coverage_video":
            continue
        location = str(source.get("location") or "unassigned")
        grouped.setdefault(location, []).append(str(source.get("relative_path") or source.get("source_id") or "video"))
    return {location: paths for location, paths in grouped.items() if len(paths) > 1}


def mixed_coverage_sources(sources: Iterable[Any]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}
    for source in sources:
        if not isinstance(source, dict) or source.get("role") not in {"coverage_video", "coverage_image"}:
            continue
        location = str(source.get("location") or "unassigned")
        grouped.setdefault(location, set()).add(str(source.get("role")))
    return {
        location: sorted(roles)
        for location, roles in grouped.items()
        if {"coverage_video", "coverage_image"}.issubset(roles)
    }


def remove_raw_source(
    *,
    destination_uri: str,
    relative_path: str,
    endpoint_url: Optional[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove one manifest source and its corresponding raw object."""
    safe_path = safe_relative_upload_path(relative_path).as_posix()
    manifest = load_manifest(destination_uri, endpoint_url)
    sources = manifest.get("sources") if isinstance(manifest.get("sources"), list) else []
    remaining = [source for source in sources if isinstance(source, dict) and source.get("relative_path") != safe_path]
    if len(remaining) == len(sources):
        raise ValueError(f"Raw source is not present in the manifest: {safe_path}")
    if not dry_run:
        deleted_groups = {
            source_group_key(source)
            for source in sources
            if isinstance(source, dict) and source.get("relative_path") == safe_path
        }
        stale_reasons = dict(manifest.get("preprocess_stale_reasons") or {})
        for group_key in deleted_groups:
            stale_reasons[group_key] = "deleted"
        delete_file(f"{destination_uri.rstrip('/')}/{safe_path}", endpoint_url=endpoint_url)
        manifest = {
            **manifest,
            "updated_at": utc_now(),
            "sources": remaining,
            "preprocess_stale_reasons": stale_reasons,
            "preprocess_stale_groups": sorted(stale_reasons),
        }
        with tempfile.TemporaryDirectory(prefix="buildvision3d-raw-remove-") as temp_dir:
            manifest_path = Path(temp_dir) / "sources_manifest.json"
            write_json(manifest_path, manifest)
            copy_file(
                manifest_path,
                f"{destination_uri.rstrip('/')}/sources_manifest.json",
                endpoint_url=endpoint_url,
            )
    return {"relative_path": safe_path, "remaining_source_count": len(remaining), "dry_run": dry_run}


def clear_preprocess_stale_groups(
    *,
    destination_uri: str,
    group_keys: Iterable[str],
    endpoint_url: Optional[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Clear stale markers for groups that were successfully preprocessed."""
    manifest = load_manifest(destination_uri, endpoint_url)
    stale_reasons = dict(manifest.get("preprocess_stale_reasons") or {})
    cleared = []
    for group_key in sorted({str(key) for key in group_keys if key}):
        if group_key in stale_reasons:
            stale_reasons.pop(group_key, None)
            cleared.append(group_key)
    manifest = {
        **manifest,
        "updated_at": utc_now(),
        "preprocess_stale_reasons": stale_reasons,
        "preprocess_stale_groups": sorted(stale_reasons),
    }
    if not dry_run and cleared:
        with tempfile.TemporaryDirectory(prefix="buildvision3d-raw-clear-stale-") as temp_dir:
            manifest_path = Path(temp_dir) / "sources_manifest.json"
            write_json(manifest_path, manifest)
            copy_file(
                manifest_path,
                f"{destination_uri.rstrip('/')}/sources_manifest.json",
                endpoint_url=endpoint_url,
            )
    return {"cleared_groups": cleared, "remaining_stale_groups": sorted(stale_reasons), "dry_run": dry_run}


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
