"""Raw media discovery and manifest helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from realestate_splat.cli import utc_now, write_json


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"}
IGNORED_NAMES = {".DS_Store", ".gitkeep"}


@dataclass
class SourceMedia:
    source_id: str
    role: str
    relative_path: str
    uri: str
    location: Optional[str] = None
    related_sources: Optional[List[str]] = None
    camera_group: Optional[str] = None
    colmap_policy: str = "include"


def discover_raw_media(input_dir: Path, destination_uri: str) -> List[SourceMedia]:
    input_dir = input_dir.expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    destination_base = destination_uri.rstrip("/")
    sources: List[SourceMedia] = []

    for path in sorted(candidate for candidate in input_dir.iterdir() if candidate.is_file()):
        if path.name in IGNORED_NAMES:
            continue
        suffix = path.suffix.lower()
        relative_path = path.relative_to(input_dir).as_posix()
        if suffix in VIDEO_SUFFIXES:
            source_id = safe_source_id(path.stem)
            sources.append(
                SourceMedia(
                    source_id=source_id,
                    role="coverage_video",
                    relative_path=relative_path,
                    uri=f"{destination_base}/{relative_path}",
                    location=source_id,
                    camera_group=f"video_{source_id}",
                )
            )
        elif suffix in IMAGE_SUFFIXES:
            source_id = safe_source_id(path.stem)
            sources.append(
                SourceMedia(
                    source_id=f"coverage_image_{source_id}",
                    role="coverage_image",
                    relative_path=relative_path,
                    uri=f"{destination_base}/{relative_path}",
                    location=None,
                    camera_group="coverage_images",
                )
            )

    hero_dir = input_dir / "hero"
    if hero_dir.exists():
        if not hero_dir.is_dir():
            raise NotADirectoryError(f"Hero path exists but is not a directory: {hero_dir}")
        for path in sorted(candidate for candidate in hero_dir.rglob("*") if candidate.is_file()):
            if path.name in IGNORED_NAMES or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative_path = path.relative_to(input_dir).as_posix()
            relative_to_hero = path.relative_to(hero_dir)
            location = safe_source_id(relative_to_hero.parts[0]) if len(relative_to_hero.parts) > 1 else "hero"
            source_id = f"hero_{location}_{safe_source_id(path.stem)}"
            related = related_coverage_sources(sources, location)
            sources.append(
                SourceMedia(
                    source_id=source_id,
                    role="hero_image",
                    relative_path=relative_path,
                    uri=f"{destination_base}/{relative_path}",
                    location=location,
                    related_sources=related,
                    camera_group=f"hero_{location}",
                    colmap_policy="optional",
                )
            )

    return sources


def related_coverage_sources(sources: Iterable[SourceMedia], location: str) -> List[str]:
    related = [
        source.source_id
        for source in sources
        if source.role == "coverage_video" and source.location == location
    ]
    return related


def safe_source_id(value: str) -> str:
    chars = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_", " "}:
            chars.append("_")
    normalized = "".join(chars).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "source"


def build_sources_manifest(project_id: str, input_dir: Path, destination_uri: str) -> Dict[str, Any]:
    sources = discover_raw_media(input_dir, destination_uri)
    return {
        "schema_version": 1,
        "project_id": project_id,
        "created_at": utc_now(),
        "input_dir": str(input_dir),
        "base_uri": destination_uri.rstrip("/"),
        "sources": [asdict(source) for source in sources],
    }


def write_sources_manifest(project_id: str, input_dir: Path, destination_uri: str, output_path: Path) -> Dict[str, Any]:
    manifest = build_sources_manifest(project_id, input_dir, destination_uri)
    write_json(output_path, manifest)
    return manifest
