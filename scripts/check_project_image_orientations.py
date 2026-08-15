#!/usr/bin/env python3
"""Report landscape/portrait still-image entries in raw and preprocess manifests."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.storage import copy_file  # noqa: E402


IMAGE_ROLES = {"coverage_image", "hero_image", "coverage", "hero"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-uri", required=True, help="Raw project prefix containing sources_manifest.json")
    parser.add_argument(
        "--preprocess-uri",
        help="Optional preprocess/current prefix containing reports/image_manifest.json or image_manifest.json",
    )
    parser.add_argument("--endpoint-url", help="Optional S3-compatible endpoint override")
    return parser.parse_args(argv)


def load_json_uri(uri: str, endpoint_url: Optional[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="buildvision3d-orientation-check-") as temp_dir:
        local_path = Path(temp_dir) / "manifest.json"
        copy_file(uri, local_path, endpoint_url=endpoint_url)
        payload = json.loads(local_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object at {uri}")
    return payload


def classify(width: Any, height: Any) -> str:
    try:
        width_value = int(width)
        height_value = int(height)
    except (TypeError, ValueError):
        return "unknown"
    if width_value > height_value:
        return "landscape"
    if height_value > width_value:
        return "portrait"
    return "square"


def report_entries(label: str, entries: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") not in IMAGE_ROLES:
            continue
        orientation = classify(entry.get("width"), entry.get("height"))
        rows.append(
            {
                "source": label,
                "path": entry.get("relative_path") or entry.get("source_path") or entry.get("image_name"),
                "role": entry.get("role"),
                "width": entry.get("width"),
                "height": entry.get("height"),
                "orientation": orientation,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    raw = load_json_uri(f"{args.raw_uri.rstrip('/')}/sources_manifest.json", args.endpoint_url)
    rows = report_entries("raw", raw.get("sources") if isinstance(raw.get("sources"), list) else [])

    if args.preprocess_uri:
        base = args.preprocess_uri.rstrip("/")
        try:
            preprocess = load_json_uri(f"{base}/reports/image_manifest.json", args.endpoint_url)
        except Exception:
            preprocess = load_json_uri(f"{base}/image_manifest.json", args.endpoint_url)
        rows.extend(report_entries("preprocess", preprocess.get("images") or []))

    print(f"Still-image entries checked: {len(rows)}")
    for row in rows:
        print(
            f"{row['source']}: {row['orientation']:9s} "
            f"{row['width']}x{row['height']} {row['role']} {row['path']}"
        )
    landscape = [row for row in rows if row["orientation"] == "landscape"]
    print(f"Landscape still images: {len(landscape)}")
    return 1 if landscape else 0


if __name__ == "__main__":
    raise SystemExit(main())
