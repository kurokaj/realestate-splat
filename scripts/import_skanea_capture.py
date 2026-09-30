#!/usr/bin/env python3
"""Validate and register one finalized Skanea RGB-D capture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from controller_common.raw_upload import upload_skanea_capture  # noqa: E402
from realestate_splat.cli import write_json  # noqa: E402
from realestate_splat.skanea_capture import load_skanea_capture  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Skanea schema-v2 capture and register its RGB-D frames in sources_manifest.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-id", required=True, help="Existing Buildvision3D project id.")
    parser.add_argument("--capture-dir", required=True, type=Path, help="Finalized Skanea session folder containing capture.json.")
    parser.add_argument("--location", required=True, help="Stable room/location label used by preprocessing and hybrid matching.")
    parser.add_argument("--destination-uri", required=True, help="Project raw URI or local raw directory.")
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint URL. For r2://, R2_ENDPOINT is used by default.")
    parser.add_argument("--manifest-path", type=Path, help="Optional local copy of the resulting manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the proposed registration without uploading.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    capture_id = load_skanea_capture(args.capture_dir).capture_id
    manifest = upload_skanea_capture(
        project_id=args.project_id,
        capture_dir=args.capture_dir,
        location=args.location,
        destination_uri=args.destination_uri,
        endpoint_url=args.endpoint_url,
        dry_run=args.dry_run,
    )
    if args.manifest_path:
        write_json(args.manifest_path, manifest)
        print(f"Wrote manifest: {args.manifest_path}")
    capture = next(
        item
        for item in manifest["captures"]
        if item.get("capture_id") == capture_id
    )
    print(f"Validated Skanea capture: {capture['capture_id']}")
    print(f"Location: {capture['location']}")
    print(f"Registered RGB-D frames: {capture['frame_count']}")
    print(f"Raw root: {capture['raw_root_relative_path']}")
    print("Mode: dry run; no files uploaded" if args.dry_run else "Capture registered and uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
