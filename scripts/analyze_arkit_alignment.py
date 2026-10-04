#!/usr/bin/env python3
"""Create an ARKit-to-COLMAP alignment diagnostic from preserved artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for candidate in (ROOT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from realestate_splat.arkit_alignment import analyze_arkit_alignment_files


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", required=True, type=Path)
    parser.add_argument("--images-txt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-run-id")
    parser.add_argument("--threshold-fraction", type=float, default=0.05)
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = analyze_arkit_alignment_files(
        args.image_manifest,
        args.images_txt,
        source_run_id=args.source_run_id,
        threshold_fraction=args.threshold_fraction,
        ransac_iterations=args.ransac_iterations,
        random_seed=args.random_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("status", "source_run_id", "capture_count", "correspondence_count", "inlier_count")}, indent=2))
    print(f"Wrote ARKit alignment diagnostic: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
