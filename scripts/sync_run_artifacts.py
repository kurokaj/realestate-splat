#!/usr/bin/env python3
"""Mirror Buildvision3D run artifacts to or from local/S3-compatible storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.stage_contract import write_artifact_manifest  # noqa: E402
from realestate_splat.storage import sync_directory  # noqa: E402


DEFAULT_EXCLUDES = [
    ".DS_Store",
    "__pycache__/*",
    "*.pyc",
    "*.zip",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload or download a run directory using the Milestone 4 storage contract.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload = subparsers.add_parser("upload", help="Mirror a local run directory to local/S3-compatible storage.")
    add_common_args(upload)
    upload.add_argument("--run", required=True, type=Path, help="Local run directory to upload.")
    upload.add_argument("--destination-uri", required=True, help="Destination URI, e.g. r2://bucket/projects/id/preprocess.")
    upload.add_argument(
        "--manifest-path",
        type=Path,
        help="Where to write the generated local artifact manifest before upload.",
    )
    upload.add_argument(
        "--sha256",
        action="store_true",
        help="Include SHA-256 hashes in the artifact manifest. This is slower for large runs.",
    )

    download = subparsers.add_parser("download", help="Mirror a stored run directory to a local run directory.")
    add_common_args(download)
    download.add_argument("--source-uri", required=True, help="Source URI, e.g. r2://bucket/projects/id/preprocess.")
    download.add_argument("--run", required=True, type=Path, help="Local destination run directory.")

    return parser.parse_args(argv)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-id", help="Project identifier. Defaults to the run directory name when available.")
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint URL. For r2://, R2_ENDPOINT is used by default.")
    parser.add_argument("--delete", action="store_true", help="Delete destination files that are absent from the source.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned sync commands without copying.")


def project_id_from_args(args: argparse.Namespace) -> str:
    if args.project_id:
        return args.project_id
    if getattr(args, "run", None) is not None:
        return Path(args.run).name
    raise SystemExit("--project-id is required when no run directory is provided.")


def upload_run(args: argparse.Namespace) -> None:
    run_dir = args.run.expanduser()
    if not args.dry_run and not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    manifest_path = args.manifest_path or (run_dir / "artifact_manifest.json")
    if args.dry_run:
        print(f"$ write-artifact-manifest {run_dir} {manifest_path}")
    else:
        write_artifact_manifest(
            run_dir,
            manifest_path,
            project_id=project_id_from_args(args),
            base_uri=args.destination_uri,
            include_sha256=args.sha256,
        )
        print(f"Wrote artifact manifest: {manifest_path}")
    sync_directory(
        run_dir,
        args.destination_uri,
        endpoint_url=args.endpoint_url,
        delete=args.delete,
        dry_run=args.dry_run,
        exclude=DEFAULT_EXCLUDES,
    )


def download_run(args: argparse.Namespace) -> None:
    sync_directory(
        args.source_uri,
        args.run,
        endpoint_url=args.endpoint_url,
        delete=args.delete,
        dry_run=args.dry_run,
        exclude=DEFAULT_EXCLUDES,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "upload":
        upload_run(args)
    elif args.command == "download":
        download_run(args)
    else:
        raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
