"""Storage URI helpers for local paths and S3-compatible object storage."""

from __future__ import annotations

import os
import fnmatch
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
from urllib.parse import urlparse


OBJECT_STORAGE_SCHEMES = {"s3", "r2"}


@dataclass(frozen=True)
class StorageUri:
    raw: str
    scheme: str
    bucket: Optional[str]
    key: str
    path: Optional[Path]

    @property
    def is_object_storage(self) -> bool:
        return self.scheme in OBJECT_STORAGE_SCHEMES

    @property
    def is_local(self) -> bool:
        return self.scheme in {"", "file", "local"}

    def as_local_path(self) -> Path:
        if self.path is None:
            raise ValueError(f"Storage URI is not local: {self.raw}")
        return self.path

    def as_aws_uri(self) -> str:
        if not self.is_object_storage or not self.bucket:
            raise ValueError(f"Storage URI is not S3-compatible object storage: {self.raw}")
        if self.key:
            return f"s3://{self.bucket}/{self.key}"
        return f"s3://{self.bucket}"


def parse_storage_uri(value: str | Path) -> StorageUri:
    raw = str(value)
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()

    if scheme in OBJECT_STORAGE_SCHEMES:
        bucket = parsed.netloc
        if not bucket:
            raise ValueError(f"Object storage URI must include a bucket: {raw}")
        return StorageUri(
            raw=raw,
            scheme=scheme,
            bucket=bucket,
            key=parsed.path.lstrip("/"),
            path=None,
        )

    if scheme == "file":
        return StorageUri(raw=raw, scheme=scheme, bucket=None, key="", path=Path(parsed.path).expanduser())
    if scheme == "local":
        local_path = parsed.path or parsed.netloc
        return StorageUri(raw=raw, scheme=scheme, bucket=None, key="", path=Path(local_path).expanduser())
    if scheme:
        raise ValueError(f"Unsupported storage URI scheme '{scheme}' in {raw}")

    return StorageUri(raw=raw, scheme="", bucket=None, key="", path=Path(raw).expanduser())


def object_storage_endpoint(uri: StorageUri, endpoint_url: Optional[str]) -> Optional[str]:
    if endpoint_url:
        return endpoint_url
    if uri.scheme == "r2":
        return os.environ.get("R2_ENDPOINT")
    return os.environ.get("AWS_ENDPOINT_URL")


def aws_base_command(uri: StorageUri, endpoint_url: Optional[str]) -> List[str]:
    command = ["aws"]
    endpoint = object_storage_endpoint(uri, endpoint_url)
    if endpoint:
        command.extend(["--endpoint-url", endpoint])
    return command


def run_command(command: Sequence[str], dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(list(command), check=True)


def sync_directory(
    source: str | Path,
    destination: str | Path,
    *,
    endpoint_url: Optional[str] = None,
    delete: bool = False,
    dry_run: bool = False,
    exclude: Sequence[str] = (),
) -> None:
    source_uri = parse_storage_uri(source)
    destination_uri = parse_storage_uri(destination)

    if source_uri.is_local and destination_uri.is_local:
        sync_local_directory(
            source_uri.as_local_path(),
            destination_uri.as_local_path(),
            delete=delete,
            dry_run=dry_run,
            exclude=exclude,
        )
        return

    object_uri = source_uri if source_uri.is_object_storage else destination_uri
    command = aws_base_command(object_uri, endpoint_url)
    command.extend(["s3", "sync", aws_sync_arg(source_uri), aws_sync_arg(destination_uri)])
    for pattern in exclude:
        command.extend(["--exclude", pattern])
    if delete:
        command.append("--delete")
    run_command(command, dry_run)


def copy_file(
    source: str | Path,
    destination: str | Path,
    *,
    endpoint_url: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    source_uri = parse_storage_uri(source)
    destination_uri = parse_storage_uri(destination)

    if source_uri.is_local and destination_uri.is_local:
        copy_local_file(source_uri.as_local_path(), destination_uri.as_local_path(), dry_run=dry_run)
        return

    object_uri = source_uri if source_uri.is_object_storage else destination_uri
    command = aws_base_command(object_uri, endpoint_url)
    command.extend(["s3", "cp", aws_sync_arg(source_uri), aws_sync_arg(destination_uri)])
    run_command(command, dry_run)


def delete_file(
    uri: str | Path,
    *,
    endpoint_url: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Delete one explicitly identified local or object-storage file."""
    storage_uri = parse_storage_uri(uri)
    if storage_uri.is_local:
        path = storage_uri.as_local_path()
        print(f"$ delete-local {path}", flush=True)
        if not dry_run and path.exists():
            if not path.is_file():
                raise IsADirectoryError(f"Cannot delete non-file path: {path}")
            path.unlink()
        return

    command = aws_base_command(storage_uri, endpoint_url)
    command.extend(["s3", "rm", aws_sync_arg(storage_uri)])
    run_command(command, dry_run)


def abort_multipart_uploads(
    prefix: str | Path,
    *,
    endpoint_url: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    prefix_uri = parse_storage_uri(prefix)
    if not prefix_uri.is_object_storage or not prefix_uri.bucket:
        raise ValueError(f"Multipart upload cleanup requires object storage URI: {prefix}")

    list_command = aws_base_command(prefix_uri, endpoint_url)
    list_command.extend(
        [
            "s3api",
            "list-multipart-uploads",
            "--bucket",
            prefix_uri.bucket,
            "--prefix",
            prefix_uri.key,
        ]
    )
    print("$ " + " ".join(list_command), flush=True)
    if dry_run:
        return 0

    result = subprocess.run(list_command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout or "{}")
    uploads = payload.get("Uploads") or []
    count = 0
    for upload in uploads:
        key = upload.get("Key")
        upload_id = upload.get("UploadId")
        if not key or not upload_id:
            continue
        abort_command = aws_base_command(prefix_uri, endpoint_url)
        abort_command.extend(
            [
                "s3api",
                "abort-multipart-upload",
                "--bucket",
                prefix_uri.bucket,
                "--key",
                str(key),
                "--upload-id",
                str(upload_id),
            ]
        )
        run_command(abort_command, dry_run=False)
        count += 1
    return count


def aws_sync_arg(uri: StorageUri) -> str:
    if uri.is_object_storage:
        return uri.as_aws_uri()
    return str(uri.as_local_path())


def sync_local_directory(
    source: Path,
    destination: Path,
    *,
    delete: bool,
    dry_run: bool,
    exclude: Sequence[str] = (),
) -> None:
    source = source.expanduser()
    destination = destination.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {source}")
    print(f"$ sync-local {source} {destination}", flush=True)
    if dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    if delete:
        for child in sorted(destination.rglob("*"), key=lambda path: len(path.parts), reverse=True):
            relative = child.relative_to(destination)
            if should_exclude(relative, exclude):
                continue
            target = source / relative
            if not target.exists():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    for child in source.rglob("*"):
        relative = child.relative_to(source)
        if should_exclude(relative, exclude):
            continue
        target = destination / relative
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def copy_local_file(source: Path, destination: Path, *, dry_run: bool) -> None:
    source = source.expanduser()
    destination = destination.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    if not source.is_file():
        raise IsADirectoryError(f"Source is not a file: {source}")
    print(f"$ copy-local {source} {destination}", flush=True)
    if dry_run:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def should_exclude(relative_path: Path, patterns: Sequence[str]) -> bool:
    normalized = relative_path.as_posix()
    return any(fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(relative_path.name, pattern) for pattern in patterns)
