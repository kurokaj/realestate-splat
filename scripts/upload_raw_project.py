#!/usr/bin/env python3
"""Upload raw capture media and write a production sources_manifest.json."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.media_manifest import build_sources_manifest  # noqa: E402
from realestate_splat.storage import copy_file, sync_directory  # noqa: E402
from realestate_splat.cli import write_json  # noqa: E402


DEFAULT_EXCLUDES = [
    ".DS_Store",
    ".gitkeep",
    "__pycache__/*",
    "*.pyc",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload raw project media to local/S3-compatible storage and create sources_manifest.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-id", required=True, help="Stable project id, e.g. house_001.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Raw capture folder containing videos/images/hero/.")
    parser.add_argument(
        "--destination-uri",
        required=True,
        help="Destination raw URI, e.g. r2://bucket/projects/house_001/raw.",
    )
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint URL. For r2://, R2_ENDPOINT is used by default.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Optional local path for the generated sources_manifest.json.",
    )
    parser.add_argument("--delete", action="store_true", help="Delete destination files absent from the source.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands and manifest summary without uploading.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    input_dir = args.input_dir.expanduser()
    manifest = build_sources_manifest(args.project_id, input_dir, args.destination_uri)

    manifest_path = args.manifest_path
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    if manifest_path is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="buildvision3d-raw-upload-")
        manifest_path = Path(temp_dir.name) / "sources_manifest.json"

    if args.dry_run:
        print_manifest_summary(manifest)
        print(f"$ write-json {manifest_path}")
    else:
        write_json(manifest_path, manifest)
        print(f"Wrote sources manifest: {manifest_path}")

    sync_directory(
        input_dir,
        args.destination_uri,
        endpoint_url=args.endpoint_url,
        delete=args.delete,
        dry_run=args.dry_run,
        exclude=DEFAULT_EXCLUDES,
    )

    copy_file(
        manifest_path,
        f"{args.destination_uri.rstrip('/')}/sources_manifest.json",
        endpoint_url=args.endpoint_url,
        dry_run=args.dry_run,
    )

    if temp_dir is not None:
        temp_dir.cleanup()


def print_manifest_summary(manifest: dict) -> None:
    sources = manifest.get("sources") or []
    counts = {}
    for source in sources:
        role = source.get("role", "unknown")
        counts[role] = counts.get(role, 0) + 1
    print(f"Project: {manifest.get('project_id')}")
    print(f"Destination: {manifest.get('base_uri')}")
    print(f"Sources: {len(sources)}")
    for role, count in sorted(counts.items()):
        print(f"  {role}: {count}")


if __name__ == "__main__":
    main()
