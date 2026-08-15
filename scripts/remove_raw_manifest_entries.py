#!/usr/bin/env python3
"""Remove explicitly named entries from a raw project's sources manifest.

This edits only sources_manifest.json. It does not delete media objects.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from controller_common.raw_upload import utc_now  # noqa: E402
from realestate_splat.cli import write_json  # noqa: E402
from realestate_splat.storage import copy_file  # noqa: E402


REMOVE_RELATIVE_PATHS = {
    "IMG20260814145818.jpg",
    "IMG20260814145951.jpg",
    "IMG20260814150103.jpg",
    "IMG20260814151534.jpg",
    "IMG20260814151518.jpg",
    "IMG20260814151529.jpg",
    "IMG20260814151540.jpg",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-uri", required=True, help="Raw project prefix, e.g. r2://bucket/projects/id/raw")
    parser.add_argument("--endpoint-url", help="Optional S3-compatible endpoint override")
    parser.add_argument("--apply", action="store_true", help="Upload the edited manifest")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest_uri = f"{args.raw_uri.rstrip('/')}/sources_manifest.json"
    with tempfile.TemporaryDirectory(prefix="buildvision3d-manifest-cleanup-") as temp_dir:
        local_manifest = Path(temp_dir) / "sources_manifest.json"
        copy_file(manifest_uri, local_manifest, endpoint_url=args.endpoint_url)
        manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        sources = manifest.get("sources") if isinstance(manifest, dict) else None
        if not isinstance(sources, list):
            raise SystemExit("Manifest does not contain a sources list.")

        removed = [
            source.get("relative_path")
            for source in sources
            if isinstance(source, dict) and source.get("relative_path") in REMOVE_RELATIVE_PATHS
        ]
        missing = sorted(REMOVE_RELATIVE_PATHS - set(removed))
        print(f"Manifest: {manifest_uri}")
        print(f"Entries to remove: {len(removed)}")
        for relative_path in sorted(removed):
            print(f"  remove {relative_path}")
        for relative_path in missing:
            print(f"  not present {relative_path}")

        if not args.apply:
            print("Dry run only. Re-run with --apply to upload the edited manifest.")
            return 0

        manifest["sources"] = [
            source
            for source in sources
            if not (isinstance(source, dict) and source.get("relative_path") in REMOVE_RELATIVE_PATHS)
        ]
        manifest["updated_at"] = utc_now()
        write_json(local_manifest, manifest)
        copy_file(local_manifest, manifest_uri, endpoint_url=args.endpoint_url)
        print(f"Uploaded updated manifest with {len(manifest['sources'])} remaining sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
