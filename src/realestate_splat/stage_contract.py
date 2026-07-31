"""Artifact and stage result contracts for the production runtime roadmap."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from realestate_splat.cli import utc_now, write_json


DEFAULT_ARTIFACT_EXCLUDES = {
    ".DS_Store",
}


@dataclass
class ArtifactEntry:
    path: str
    size_bytes: int
    sha256: Optional[str] = None


@dataclass
class ArtifactManifest:
    schema_version: int
    project_id: str
    created_at: str
    base_uri: str
    files: List[ArtifactEntry]

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "base_uri": self.base_uri,
            "files": [asdict(entry) for entry in self.files],
        }


@dataclass
class StageResult:
    schema_version: int
    project_id: str
    pipeline_run_id: Optional[str]
    stage_run_id: Optional[str]
    stage: str
    status: str
    started_at: str
    finished_at: str
    input_uris: List[str] = field(default_factory=list)
    output_uris: List[str] = field(default_factory=list)
    artifact_manifest_uri: Optional[str] = None
    logs_uri: Optional[str] = None
    metrics_uri: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def build_artifact_manifest(
    root: Path,
    *,
    project_id: str,
    base_uri: str,
    include_sha256: bool = False,
    exclude_names: Iterable[str] = DEFAULT_ARTIFACT_EXCLUDES,
) -> ArtifactManifest:
    root = root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Artifact root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Artifact root is not a directory: {root}")

    excluded = set(exclude_names)
    files: List[ArtifactEntry] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if any(part in excluded for part in path.relative_to(root).parts):
            continue
        relative_path = path.relative_to(root).as_posix()
        files.append(
            ArtifactEntry(
                path=relative_path,
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path) if include_sha256 else None,
            )
        )
    return ArtifactManifest(
        schema_version=1,
        project_id=project_id,
        created_at=utc_now(),
        base_uri=base_uri.rstrip("/"),
        files=files,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(
    root: Path,
    output_path: Path,
    *,
    project_id: str,
    base_uri: str,
    include_sha256: bool = False,
) -> ArtifactManifest:
    manifest = build_artifact_manifest(
        root,
        project_id=project_id,
        base_uri=base_uri,
        include_sha256=include_sha256,
    )
    write_json(output_path, manifest.to_json())
    return manifest


def write_stage_result(path: Path, result: StageResult | Mapping[str, Any]) -> None:
    payload = result.to_json() if isinstance(result, StageResult) else dict(result)
    write_json(path, payload)
