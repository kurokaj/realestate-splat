#!/usr/bin/env python3
"""Generate and upload sparse viewer artifacts for an existing COLMAP current output."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from controller_common.colmap_viewer import write_sparse_viewer_payload
from src.realestate_splat.storage import copy_file, sync_directory


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill viewer/sparse_scene.json for an existing COLMAP output prefix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-uri", required=True, help="COLMAP current URI, e.g. r2://bucket/projects/id/colmap/current")
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint URL.")
    parser.add_argument("--max-points", type=int, default=25000, help="Maximum point samples stored in viewer JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned storage commands without writing.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="buildvision3d-colmap-viewer-") as temp_dir:
        root = Path(temp_dir)
        local_current = root / "current"
        sync_directory(
            args.input_uri.rstrip("/"),
            local_current,
            endpoint_url=args.endpoint_url,
            dry_run=args.dry_run,
        )
        sparse_txt_dir = local_current / "sparse_txt"
        if not sparse_txt_dir.exists():
            raise SystemExit(f"COLMAP current output is missing sparse_txt/: {sparse_txt_dir}")
        viewer_path = write_sparse_viewer_payload(
            sparse_txt_dir,
            local_current / "viewer" / "sparse_scene.json",
            max_points=args.max_points,
        )
        copy_file(
            viewer_path,
            f"{args.input_uri.rstrip('/')}/viewer/sparse_scene.json",
            endpoint_url=args.endpoint_url,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
