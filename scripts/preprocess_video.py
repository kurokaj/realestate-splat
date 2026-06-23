#!/usr/bin/env python3
"""Local video preprocessing for the real estate splat pipeline.

This script intentionally avoids writing an extracted raw-frame cache. It scores
candidate frames in a first pass, trims the selected frame list if needed, then
writes only the final frames to ``frames_selected/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError as import_error:  # pragma: no cover - exercised before deps install
    cv2 = None
    np = None
    IMPORT_ERROR = import_error
else:
    IMPORT_ERROR = None


PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "indoor_room": {
        "candidate_fps": 2.0,
        "target_min": 100,
        "target_max": 250,
        "min_blur": 70.0,
        "min_brightness": 35.0,
        "max_brightness": 225.0,
        "min_contrast": 16.0,
        "min_entropy": 3.2,
        "duplicate_hash_threshold": 4,
        "duplicate_pixel_threshold": 0.018,
        "force_keep_interval": 3.0,
        "coverage_window_seconds": 2.0,
        "min_frames_per_window": 1,
        "coverage_hard_min_blur": 20.0,
        "coverage_hard_min_brightness": 20.0,
        "coverage_hard_max_brightness": 245.0,
        "coverage_hard_min_contrast": 8.0,
        "coverage_hard_min_entropy": 2.0,
    },
    "small_apartment": {
        "candidate_fps": 2.5,
        "target_min": 300,
        "target_max": 700,
        "min_blur": 70.0,
        "min_brightness": 35.0,
        "max_brightness": 225.0,
        "min_contrast": 16.0,
        "min_entropy": 3.2,
        "duplicate_hash_threshold": 4,
        "duplicate_pixel_threshold": 0.018,
        "force_keep_interval": 3.0,
        "coverage_window_seconds": 2.0,
        "min_frames_per_window": 1,
        "coverage_hard_min_blur": 20.0,
        "coverage_hard_min_brightness": 20.0,
        "coverage_hard_max_brightness": 245.0,
        "coverage_hard_min_contrast": 8.0,
        "coverage_hard_min_entropy": 2.0,
    },
    "indoor_house": {
        "candidate_fps": 3.0,
        "target_min": 800,
        "target_max": 1800,
        "min_blur": 70.0,
        "min_brightness": 35.0,
        "max_brightness": 225.0,
        "min_contrast": 16.0,
        "min_entropy": 3.2,
        "duplicate_hash_threshold": 4,
        "duplicate_pixel_threshold": 0.018,
        "force_keep_interval": 3.0,
        "coverage_window_seconds": 2.0,
        "min_frames_per_window": 1,
        "coverage_hard_min_blur": 20.0,
        "coverage_hard_min_brightness": 20.0,
        "coverage_hard_max_brightness": 245.0,
        "coverage_hard_min_contrast": 8.0,
        "coverage_hard_min_entropy": 2.0,
    },
    "outdoor_orbit": {
        "candidate_fps": 2.0,
        "target_min": 300,
        "target_max": 900,
        "min_blur": 80.0,
        "min_brightness": 30.0,
        "max_brightness": 235.0,
        "min_contrast": 14.0,
        "min_entropy": 3.0,
        "duplicate_hash_threshold": 4,
        "duplicate_pixel_threshold": 0.016,
        "force_keep_interval": 3.0,
        "coverage_window_seconds": 2.0,
        "min_frames_per_window": 1,
        "coverage_hard_min_blur": 20.0,
        "coverage_hard_min_brightness": 20.0,
        "coverage_hard_max_brightness": 245.0,
        "coverage_hard_min_contrast": 8.0,
        "coverage_hard_min_entropy": 2.0,
    },
}

REPORT_FILENAMES = {
    "capture_json": "capture_report.json",
    "capture_html": "capture_report.html",
    "contact_sheet": "frame_contact_sheet.jpg",
    "gpu_json": "gpu_recommendation.json",
}


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_seconds: float
    blur_score: float
    brightness: float
    contrast: float
    entropy: float
    ahash_hex: str
    selected_initial: bool
    selected_final: bool = False
    reject_reason: Optional[str] = None
    selected_by: Optional[str] = None
    quality_score: float = 0.0
    hash_distance_to_previous_selected: Optional[int] = None
    pixel_difference_to_previous_selected: Optional[float] = None
    output_file: Optional[str] = None


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: Optional[int]
    duration_seconds: Optional[float]
    width: int
    height: int


def require_dependencies() -> None:
    if IMPORT_ERROR is None:
        return

    message = (
        "Missing Python dependency: {dep}\n\n"
        "Install milestone 1 dependencies with:\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  python -m pip install -r requirements.txt"
    ).format(dep=IMPORT_ERROR)
    raise SystemExit(message)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select reconstruction-friendly frames from a real estate capture video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, type=Path, help="Source video file.")
    parser.add_argument("--out", required=True, type=Path, help="Output run directory.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS),
        default="indoor_room",
        help="Preset tuned for an expected scene type.",
    )
    parser.add_argument("--candidate-fps", type=float, help="Candidate extraction rate.")
    parser.add_argument("--target-min", type=int, help="Expected lower bound for selected frames.")
    parser.add_argument("--target-max", type=int, help="Trim selected frames to this count.")
    parser.add_argument("--min-blur", type=float, help="Minimum Laplacian variance blur score.")
    parser.add_argument("--min-brightness", type=float, help="Minimum mean grayscale brightness.")
    parser.add_argument("--max-brightness", type=float, help="Maximum mean grayscale brightness.")
    parser.add_argument("--min-contrast", type=float, help="Minimum grayscale standard deviation.")
    parser.add_argument("--min-entropy", type=float, help="Minimum grayscale histogram entropy.")
    parser.add_argument(
        "--duplicate-hash-threshold",
        type=int,
        help="Average-hash hamming distance threshold for near duplicates.",
    )
    parser.add_argument(
        "--duplicate-pixel-threshold",
        type=float,
        help="Mean absolute 32x32 grayscale difference threshold for near duplicates.",
    )
    parser.add_argument(
        "--force-keep-interval",
        type=float,
        help="Keep a frame at least this often even if it resembles the previous selected frame.",
    )
    parser.add_argument(
        "--coverage-window-seconds",
        type=float,
        help="Gap-aware selection window size; set 0 to disable coverage fallback.",
    )
    parser.add_argument(
        "--min-frames-per-window",
        type=int,
        help="Minimum final frames to keep in each coverage window when possible.",
    )
    parser.add_argument(
        "--coverage-hard-min-blur",
        type=float,
        help="Fallback frames below this blur score remain rejected.",
    )
    parser.add_argument(
        "--coverage-hard-min-brightness",
        type=float,
        help="Fallback frames below this brightness remain rejected.",
    )
    parser.add_argument(
        "--coverage-hard-max-brightness",
        type=float,
        help="Fallback frames above this brightness remain rejected.",
    )
    parser.add_argument(
        "--coverage-hard-min-contrast",
        type=float,
        help="Fallback frames below this contrast remain rejected.",
    )
    parser.add_argument(
        "--coverage-hard-min-entropy",
        type=float,
        help="Fallback frames below this entropy remain rejected.",
    )
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start time within video.")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="Optional processing duration from --start-seconds.",
    )
    parser.add_argument(
        "--contact-sheet-frames",
        type=int,
        default=48,
        help="Maximum selected frames to include in the contact sheet.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality for selected frames.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated frame_*.jpg files in frames_selected/.",
    )
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> Dict[str, Any]:
    settings = dict(PROFILE_DEFAULTS[args.profile])
    override_names = [
        "candidate_fps",
        "target_min",
        "target_max",
        "min_blur",
        "min_brightness",
        "max_brightness",
        "min_contrast",
        "min_entropy",
        "duplicate_hash_threshold",
        "duplicate_pixel_threshold",
        "force_keep_interval",
        "coverage_window_seconds",
        "min_frames_per_window",
        "coverage_hard_min_blur",
        "coverage_hard_min_brightness",
        "coverage_hard_max_brightness",
        "coverage_hard_min_contrast",
        "coverage_hard_min_entropy",
    ]
    for name in override_names:
        value = getattr(args, name)
        if value is not None:
            settings[name] = value

    settings["profile"] = args.profile
    settings["start_seconds"] = args.start_seconds
    settings["duration_seconds"] = args.duration_seconds
    settings["contact_sheet_frames"] = args.contact_sheet_frames
    settings["jpeg_quality"] = args.jpeg_quality
    return settings


def validate_args(args: argparse.Namespace, settings: Dict[str, Any]) -> None:
    if not args.video.exists():
        raise SystemExit(f"Video file does not exist: {args.video}")
    if not args.video.is_file():
        raise SystemExit(f"Video path is not a file: {args.video}")
    if settings["candidate_fps"] <= 0:
        raise SystemExit("--candidate-fps must be greater than zero.")
    if settings["target_min"] < 0:
        raise SystemExit("--target-min must be zero or greater.")
    if settings["target_max"] <= 0:
        raise SystemExit("--target-max must be greater than zero.")
    if settings["target_min"] > settings["target_max"]:
        raise SystemExit("--target-min cannot be greater than --target-max.")
    if args.start_seconds < 0:
        raise SystemExit("--start-seconds cannot be negative.")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise SystemExit("--duration-seconds must be greater than zero.")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100.")
    if settings["coverage_window_seconds"] < 0:
        raise SystemExit("--coverage-window-seconds cannot be negative.")
    if settings["min_frames_per_window"] < 0:
        raise SystemExit("--min-frames-per-window cannot be negative.")
    if settings["min_frames_per_window"] > 0 and settings["coverage_window_seconds"] <= 0:
        raise SystemExit("--coverage-window-seconds must be greater than zero when coverage fallback is enabled.")
    if settings["coverage_hard_min_brightness"] > settings["coverage_hard_max_brightness"]:
        raise SystemExit("--coverage-hard-min-brightness cannot exceed --coverage-hard-max-brightness.")


def prepare_output_dirs(out_dir: Path, overwrite: bool) -> Tuple[Path, Path]:
    frames_dir = out_dir / "frames_selected"
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(frames_dir.glob("frame_*.jpg"))
    if existing_frames and not overwrite:
        raise SystemExit(
            f"{frames_dir} already contains generated frames. "
            "Use --overwrite to replace them."
        )
    if overwrite:
        for frame_path in existing_frames:
            frame_path.unlink()

    return frames_dir, reports_dir


def open_video(video_path: Path) -> Any:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    return cap


def read_video_info(video_path: Path, cap: Any) -> VideoInfo:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.001 or math.isnan(fps):
        fps = 30.0

    raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_count = raw_frame_count if raw_frame_count > 0 else None
    duration = frame_count / fps if frame_count is not None else None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return VideoInfo(
        path=str(video_path),
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        width=width,
        height=height,
    )


def average_hash(gray: Any, hash_size: int = 8) -> int:
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = small > mean
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def frame_signature(gray: Any) -> Any:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return small.astype("float32") / 255.0


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def pixel_difference(left: Any, right: Any) -> float:
    return float(np.mean(np.abs(left - right)))


def image_entropy(gray: Any) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    probabilities = hist / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def quality_score(metrics: Dict[str, float], settings: Dict[str, Any]) -> float:
    blur = min(metrics["blur_score"] / max(float(settings["min_blur"]), 1.0), 2.5)
    contrast = min(metrics["contrast"] / max(float(settings["min_contrast"]), 1.0), 2.5)
    entropy = min(metrics["entropy"] / max(float(settings["min_entropy"]), 1.0), 2.5)
    brightness_balance = 1.0 - min(abs(metrics["brightness"] - 128.0) / 128.0, 1.0)
    score = (50.0 * blur) + (20.0 * contrast) + (20.0 * entropy) + (10.0 * brightness_balance)
    return round(score, 3)


def score_frame(frame: Any) -> Tuple[Dict[str, float], int, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics = {
        "blur_score": blur_score,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "entropy": image_entropy(gray),
    }
    return metrics, average_hash(gray), frame_signature(gray)


def first_quality_rejection(metrics: Dict[str, float], settings: Dict[str, Any]) -> Optional[str]:
    if metrics["blur_score"] < settings["min_blur"]:
        return "blur"
    if metrics["brightness"] < settings["min_brightness"]:
        return "too_dark"
    if metrics["brightness"] > settings["max_brightness"]:
        return "too_bright"
    if metrics["contrast"] < settings["min_contrast"]:
        return "low_contrast"
    if metrics["entropy"] < settings["min_entropy"]:
        return "low_texture"
    return None


def analyze_video(video_path: Path, settings: Dict[str, Any]) -> Tuple[VideoInfo, List[FrameRecord], List[FrameRecord]]:
    cap = open_video(video_path)
    try:
        video_info = read_video_info(video_path, cap)
        source_fps = video_info.fps
        sample_stride = max(1, int(round(source_fps / float(settings["candidate_fps"]))))
        start_frame = max(0, int(round(float(settings["start_seconds"]) * source_fps)))
        end_time = None
        if settings["duration_seconds"] is not None:
            end_time = float(settings["start_seconds"]) + float(settings["duration_seconds"])

        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        records: List[FrameRecord] = []
        selected_initial: List[FrameRecord] = []
        last_selected_hash: Optional[int] = None
        last_selected_signature: Optional[Any] = None
        last_selected_timestamp: Optional[float] = None

        frame_index = start_frame
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_index / source_fps
            if end_time is not None and timestamp > end_time:
                break

            should_score = ((frame_index - start_frame) % sample_stride) == 0
            if should_score:
                metrics, ahash_value, signature = score_frame(frame)
                reject_reason = first_quality_rejection(metrics, settings)
                hash_distance_value: Optional[int] = None
                pixel_difference_value: Optional[float] = None

                if reject_reason is None and last_selected_hash is not None and last_selected_signature is not None:
                    hash_distance_value = hamming_distance(ahash_value, last_selected_hash)
                    pixel_difference_value = pixel_difference(signature, last_selected_signature)
                    is_near_duplicate = (
                        hash_distance_value <= int(settings["duplicate_hash_threshold"])
                        and pixel_difference_value <= float(settings["duplicate_pixel_threshold"])
                    )
                    if is_near_duplicate and last_selected_timestamp is not None:
                        elapsed_since_selected = timestamp - last_selected_timestamp
                        if elapsed_since_selected < float(settings["force_keep_interval"]):
                            reject_reason = "duplicate"

                record = FrameRecord(
                    frame_index=frame_index,
                    timestamp_seconds=round(timestamp, 3),
                    blur_score=round(metrics["blur_score"], 3),
                    brightness=round(metrics["brightness"], 3),
                    contrast=round(metrics["contrast"], 3),
                    entropy=round(metrics["entropy"], 3),
                    ahash_hex=f"{ahash_value:016x}",
                    selected_initial=reject_reason is None,
                    reject_reason=reject_reason,
                    quality_score=quality_score(metrics, settings),
                    hash_distance_to_previous_selected=hash_distance_value,
                    pixel_difference_to_previous_selected=(
                        round(pixel_difference_value, 5) if pixel_difference_value is not None else None
                    ),
                )
                records.append(record)

                if reject_reason is None:
                    selected_initial.append(record)
                    last_selected_hash = ahash_value
                    last_selected_signature = signature
                    last_selected_timestamp = timestamp

            frame_index += 1
    finally:
        cap.release()

    return video_info, records, selected_initial


def pick_evenly(items: Sequence[FrameRecord], limit: int) -> List[FrameRecord]:
    if len(items) <= limit:
        return list(items)
    if limit <= 1:
        return [items[0]]

    positions = []
    seen = set()
    for idx in range(limit):
        position = int(round(idx * (len(items) - 1) / (limit - 1)))
        if position not in seen:
            positions.append(position)
            seen.add(position)

    cursor = 0
    while len(positions) < limit and cursor < len(items):
        if cursor not in seen:
            positions.append(cursor)
            seen.add(cursor)
        cursor += 1

    return [items[position] for position in sorted(positions[:limit])]


def add_unique_frame(target: List[FrameRecord], seen_ids: set, record: FrameRecord) -> None:
    record_id = id(record)
    if record_id in seen_ids:
        return
    target.append(record)
    seen_ids.add(record_id)


def best_quality_frames(records: Sequence[FrameRecord], limit: int) -> List[FrameRecord]:
    if limit <= 0:
        return []
    ranked = sorted(
        records,
        key=lambda record: (
            record.quality_score,
            record.blur_score,
            record.contrast,
            record.entropy,
            -abs(record.brightness - 128.0),
        ),
        reverse=True,
    )
    return ranked[:limit]


def coverage_fallback_allowed(record: FrameRecord, settings: Dict[str, Any]) -> bool:
    if record.blur_score < float(settings["coverage_hard_min_blur"]):
        return False
    if record.brightness < float(settings["coverage_hard_min_brightness"]):
        return False
    if record.brightness > float(settings["coverage_hard_max_brightness"]):
        return False
    if record.contrast < float(settings["coverage_hard_min_contrast"]):
        return False
    if record.entropy < float(settings["coverage_hard_min_entropy"]):
        return False
    return True


def records_by_coverage_window(records: Sequence[FrameRecord], window_seconds: float) -> Dict[int, List[FrameRecord]]:
    windows: Dict[int, List[FrameRecord]] = {}
    for record in records:
        window_index = int(record.timestamp_seconds // window_seconds)
        windows.setdefault(window_index, []).append(record)
    return windows


def finalize_selection(
    records: Sequence[FrameRecord],
    selected_initial: Sequence[FrameRecord],
    settings: Dict[str, Any],
) -> List[FrameRecord]:
    for record in records:
        record.selected_final = False
        record.selected_by = None

    target_max = int(settings["target_max"])
    window_seconds = float(settings["coverage_window_seconds"])
    min_frames_per_window = int(settings["min_frames_per_window"])
    normal_selected = sorted(selected_initial, key=lambda record: (record.timestamp_seconds, record.frame_index))
    normal_ids = {id(record) for record in normal_selected}

    mandatory: List[FrameRecord] = []
    mandatory_ids: set = set()
    coverage_fallback_ids: set = set()

    if records and min_frames_per_window > 0 and window_seconds > 0:
        windows = records_by_coverage_window(records, window_seconds)
        for _window_index, window_records in sorted(windows.items()):
            normal_in_window = [record for record in window_records if id(record) in normal_ids]
            normal_required = best_quality_frames(normal_in_window, min_frames_per_window)
            for record in normal_required:
                add_unique_frame(mandatory, mandatory_ids, record)

            needed = min_frames_per_window - len(normal_required)
            if needed <= 0:
                continue

            fallback_candidates = [
                record
                for record in window_records
                if id(record) not in normal_ids and coverage_fallback_allowed(record, settings)
            ]
            for record in best_quality_frames(fallback_candidates, needed):
                add_unique_frame(mandatory, mandatory_ids, record)
                coverage_fallback_ids.add(id(record))

    pool: List[FrameRecord] = []
    pool_ids: set = set()
    for record in mandatory:
        add_unique_frame(pool, pool_ids, record)
    for record in normal_selected:
        add_unique_frame(pool, pool_ids, record)

    if len(mandatory) >= target_max:
        selected_final = pick_evenly(
            sorted(mandatory, key=lambda record: (record.timestamp_seconds, record.frame_index)),
            target_max,
        )
    else:
        extra_budget = target_max - len(mandatory)
        extra_candidates = [
            record for record in pool if id(record) not in mandatory_ids
        ]
        selected_final = list(mandatory)
        for record in pick_evenly(
            sorted(extra_candidates, key=lambda record: (record.timestamp_seconds, record.frame_index)),
            extra_budget,
        ):
            add_unique_frame(selected_final, {id(item) for item in selected_final}, record)

    selected_final = sorted(selected_final, key=lambda record: (record.timestamp_seconds, record.frame_index))
    selected_ids = {id(record) for record in selected_final}
    for record in records:
        if id(record) in selected_ids:
            record.selected_final = True
            record.selected_by = "coverage_fallback" if id(record) in coverage_fallback_ids else "quality"
        elif record.selected_initial:
            record.reject_reason = "trimmed_after_target_max"

    return selected_final


def assign_output_paths(selected_final: Sequence[FrameRecord]) -> None:
    for sequence, record in enumerate(selected_final, start=1):
        record.output_file = f"frames_selected/frame_{sequence:06d}.jpg"


def save_selected_frames(video_path: Path, selected_final: Sequence[FrameRecord], out_dir: Path, settings: Dict[str, Any]) -> int:
    if not selected_final:
        return 0

    selected_by_index = {record.frame_index: record for record in selected_final}
    max_index = max(selected_by_index)

    cap = open_video(video_path)
    saved_count = 0
    try:
        current_index = 0
        while current_index <= max_index:
            ok, frame = cap.read()
            if not ok:
                break

            record = selected_by_index.get(current_index)
            if record is not None and record.output_file is not None:
                output_path = out_dir / record.output_file
                ok = cv2.imwrite(
                    str(output_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(settings["jpeg_quality"])],
                )
                if not ok:
                    raise SystemExit(f"Could not write selected frame: {output_path}")
                saved_count += 1

            current_index += 1
    finally:
        cap.release()

    expected_count = len(selected_final)
    if saved_count != expected_count:
        raise SystemExit(f"Saved {saved_count} selected frames, expected {expected_count}.")
    return saved_count


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return round(float(np.percentile(np.array(values, dtype="float32"), q)), 3)


def metric_summary(records: Sequence[FrameRecord], attr_name: str) -> Dict[str, Optional[float]]:
    values = [float(getattr(record, attr_name)) for record in records]
    if not values:
        return {"min": None, "p10": None, "p50": None, "mean": None, "p90": None, "max": None}
    return {
        "min": round(min(values), 3),
        "p10": percentile(values, 10),
        "p50": percentile(values, 50),
        "mean": round(sum(values) / len(values), 3),
        "p90": percentile(values, 90),
        "max": round(max(values), 3),
    }


def count_rejections(records: Sequence[FrameRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        if record.selected_final:
            continue
        reason = record.reject_reason or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def selected_by_counts(selected_final: Sequence[FrameRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in selected_final:
        key = record.selected_by or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def coverage_summary(
    records: Sequence[FrameRecord],
    selected_final: Sequence[FrameRecord],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    window_seconds = float(settings["coverage_window_seconds"])
    min_frames_per_window = int(settings["min_frames_per_window"])
    if not records or min_frames_per_window <= 0 or window_seconds <= 0:
        return {
            "enabled": False,
            "window_seconds": window_seconds,
            "min_frames_per_window": min_frames_per_window,
        }

    windows = records_by_coverage_window(records, window_seconds)
    selected_counts: Dict[int, int] = {}
    for record in selected_final:
        window_index = int(record.timestamp_seconds // window_seconds)
        selected_counts[window_index] = selected_counts.get(window_index, 0) + 1

    windows_below_minimum = []
    for window_index, window_records in sorted(windows.items()):
        selected_count = selected_counts.get(window_index, 0)
        if selected_count < min_frames_per_window:
            windows_below_minimum.append(
                {
                    "window_index": window_index,
                    "start_seconds": round(window_index * window_seconds, 3),
                    "end_seconds": round((window_index + 1) * window_seconds, 3),
                    "candidate_frames": len(window_records),
                    "selected_frames": selected_count,
                }
            )

    timestamps = [record.timestamp_seconds for record in selected_final]
    largest_gap = None
    if len(timestamps) >= 2:
        largest_gap = round(max(right - left for left, right in zip(timestamps, timestamps[1:])), 3)

    fallback_reasons: Dict[str, int] = {}
    for record in selected_final:
        if record.selected_by != "coverage_fallback":
            continue
        reason = record.reject_reason or "unknown"
        fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

    return {
        "enabled": True,
        "window_seconds": window_seconds,
        "min_frames_per_window": min_frames_per_window,
        "candidate_window_count": len(windows),
        "windows_below_minimum_count": len(windows_below_minimum),
        "windows_below_minimum": windows_below_minimum,
        "coverage_fallback_frame_count": selected_by_counts(selected_final).get("coverage_fallback", 0),
        "coverage_fallback_reasons": dict(sorted(fallback_reasons.items())),
        "largest_selected_gap_seconds": largest_gap,
    }


def warning_flags(
    records: Sequence[FrameRecord],
    selected_final: Sequence[FrameRecord],
    settings: Dict[str, Any],
) -> List[str]:
    warnings: List[str] = []
    candidate_count = len(records)
    selected_count = len(selected_final)
    rejection_counts = count_rejections(records)

    if selected_count == 0:
        warnings.append("no_selected_frames")
    elif selected_count < int(settings["target_min"]):
        warnings.append("too_few_selected_frames")

    coverage = coverage_summary(records, selected_final, settings)
    if coverage.get("enabled"):
        required_for_windows = coverage["candidate_window_count"] * int(settings["min_frames_per_window"])
        if int(settings["target_max"]) < required_for_windows:
            warnings.append("target_max_below_coverage_minimum")
        if coverage["windows_below_minimum_count"] > 0:
            warnings.append("coverage_gaps_remain")

    if rejection_counts.get("trimmed_after_target_max", 0) > 0:
        warnings.append("selected_frames_trimmed_to_target_max")

    if candidate_count > 0:
        blur_rate = rejection_counts.get("blur", 0) / candidate_count
        duplicate_rate = rejection_counts.get("duplicate", 0) / candidate_count
        if blur_rate >= 0.35:
            warnings.append("high_blur_rate")
        if duplicate_rate >= 0.50:
            warnings.append("high_duplicate_rate")

        entropy_mean = metric_summary(records, "entropy")["mean"]
        if entropy_mean is not None and entropy_mean < float(settings["min_entropy"]) + 0.3:
            warnings.append("low_texture_scene")

        brightness_values = [record.brightness for record in records]
        brightness_std = float(np.std(np.array(brightness_values, dtype="float32")))
        if brightness_std >= 45.0:
            warnings.append("lighting_unstable")

    return warnings


def resolution_label(width: int, height: int) -> str:
    pixels = width * height
    max_dim = max(width, height)
    if pixels >= 7_000_000 or max_dim >= 3840:
        return "4K+"
    if pixels >= 3_000_000 or max_dim >= 2560:
        return "2K+"
    if max_dim >= 1920:
        return "1080p-2K"
    return "sub-1080p"


def gpu_recommendation(selected_count: int, video_info: VideoInfo, warnings: Sequence[str]) -> Dict[str, Any]:
    width = video_info.width
    height = video_info.height
    label = resolution_label(width, height)
    high_resolution = label in {"2K+", "4K+"}

    if selected_count <= 0:
        tier = {
            "suggested_gpu": "none",
            "vram_gb_min": None,
            "vram_gb_preferred": None,
            "reason": "No selected frames were produced.",
        }
    elif selected_count < 300:
        tier = {
            "suggested_gpu": "RTX A6000 48GB or similar",
            "vram_gb_min": 24,
            "vram_gb_preferred": 48 if high_resolution else 40,
            "reason": "<300 selected images; 24-40GB may work, 48GB is the safer Verda development default.",
        }
    elif selected_count <= 800:
        tier = {
            "suggested_gpu": "RTX A6000 48GB or similar",
            "vram_gb_min": 40,
            "vram_gb_preferred": 48,
            "reason": "300-800 selected images; use a 40-48GB GPU.",
        }
    elif selected_count <= 1500:
        tier = {
            "suggested_gpu": "A100 80GB, RTX PRO 6000 96GB, or similar",
            "vram_gb_min": 48,
            "vram_gb_preferred": 80,
            "reason": "800-1,500 selected images; prefer a 48-80GB GPU.",
        }
    elif selected_count <= 2500:
        tier = {
            "suggested_gpu": "RTX PRO 6000 96GB or A100 80GB",
            "vram_gb_min": 80,
            "vram_gb_preferred": 96,
            "reason": "1,500-2,500 selected images; prefer 80-96GB VRAM.",
        }
    else:
        tier = {
            "suggested_gpu": "96GB+ GPU or split the scene",
            "vram_gb_min": 96,
            "vram_gb_preferred": 96,
            "reason": "2,500+ selected images; split the scene or use 96GB+ VRAM.",
        }

    notes = [
        "Recommendation is intentionally conservative to reduce out-of-memory risk.",
        "Use development settings before paying for a full reconstruction run.",
    ]
    if high_resolution and selected_count >= 300:
        notes.append("2K+ source resolution increases memory pressure; downscale if early tests fail.")

    return {
        "schema_version": 1,
        "selected_images": selected_count,
        "resolution": {
            "width": width,
            "height": height,
            "label": label,
        },
        "recommendation": tier,
        "warnings": list(warnings),
        "notes": notes,
    }


def relative_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def build_capture_report(
    video_info: VideoInfo,
    records: Sequence[FrameRecord],
    selected_final: Sequence[FrameRecord],
    settings: Dict[str, Any],
    out_dir: Path,
    reports_dir: Path,
    warnings: Sequence[str],
) -> Dict[str, Any]:
    rejection_counts = count_rejections(records)
    outputs = {
        "frames_selected": "frames_selected",
        "capture_report_json": relative_to(reports_dir / REPORT_FILENAMES["capture_json"], out_dir),
        "capture_report_html": relative_to(reports_dir / REPORT_FILENAMES["capture_html"], out_dir),
        "frame_contact_sheet": relative_to(reports_dir / REPORT_FILENAMES["contact_sheet"], out_dir),
        "gpu_recommendation_json": relative_to(reports_dir / REPORT_FILENAMES["gpu_json"], out_dir),
    }

    selected_timestamps = [record.timestamp_seconds for record in selected_final]
    selection_counts = selected_by_counts(selected_final)
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "video": asdict(video_info),
        "settings": settings,
        "summary": {
            "candidate_frame_count": len(records),
            "selected_frame_count": len(selected_final),
            "selected_by": selection_counts,
            "coverage_fallback_frame_count": selection_counts.get("coverage_fallback", 0),
            "rejected_frame_count": len(records) - len(selected_final),
            "rejections": rejection_counts,
            "blur_score_distribution": metric_summary(records, "blur_score"),
            "brightness_distribution": metric_summary(records, "brightness"),
            "contrast_distribution": metric_summary(records, "contrast"),
            "entropy_distribution": metric_summary(records, "entropy"),
            "selected_frame_timeline_seconds": selected_timestamps,
            "coverage": coverage_summary(records, selected_final, settings),
        },
        "warnings": list(warnings),
        "outputs": outputs,
        "frames": [frame_record_to_dict(record) for record in records],
    }


def frame_record_to_dict(record: FrameRecord) -> Dict[str, Any]:
    data = asdict(record)
    if record.selected_final and record.selected_by == "coverage_fallback":
        data["decision"] = "coverage_fallback"
    elif record.selected_final:
        data["decision"] = "selected"
    elif record.reject_reason == "trimmed_after_target_max":
        data["decision"] = "trimmed"
    else:
        data["decision"] = "rejected"
    return data


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fit_image_to_cell(image: Any, width: int, height: int) -> Any:
    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    cell = np.full((height, width, 3), 245, dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    cell[y : y + resized_height, x : x + resized_width] = resized
    return cell


def create_contact_sheet(
    out_dir: Path,
    reports_dir: Path,
    selected_final: Sequence[FrameRecord],
    max_frames: int,
) -> Path:
    output_path = reports_dir / REPORT_FILENAMES["contact_sheet"]
    if not selected_final:
        sheet = np.full((320, 640, 3), 245, dtype=np.uint8)
        cv2.putText(
            sheet,
            "No frames selected",
            (170, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return output_path

    contact_records = pick_evenly(selected_final, max(1, max_frames))
    columns = min(6, max(1, int(math.ceil(math.sqrt(len(contact_records))))))
    rows = int(math.ceil(len(contact_records) / columns))
    cell_width = 220
    cell_height = 160
    image_height = 132
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 230, dtype=np.uint8)

    for index, record in enumerate(contact_records):
        if record.output_file is None:
            continue
        image = cv2.imread(str(out_dir / record.output_file))
        if image is None:
            continue
        thumb = fit_image_to_cell(image, cell_width, image_height)
        row = index // columns
        column = index % columns
        y = row * cell_height
        x = column * cell_width
        sheet[y : y + image_height, x : x + cell_width] = thumb
        label = f"{record.timestamp_seconds:.1f}s  #{record.frame_index}"
        cv2.putText(
            sheet,
            label,
            (x + 8, y + image_height + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return output_path


def table_rows(mapping: Dict[str, Any]) -> str:
    rows = []
    for key, value in mapping.items():
        rows.append(
            "<tr><th>{key}</th><td>{value}</td></tr>".format(
                key=html.escape(str(key).replace("_", " ")),
                value=html.escape(str(value)),
            )
        )
    return "\n".join(rows)


def metric_table(report: Dict[str, Any]) -> str:
    metric_names = [
        ("Blur", "blur_score_distribution"),
        ("Brightness", "brightness_distribution"),
        ("Contrast", "contrast_distribution"),
        ("Entropy", "entropy_distribution"),
    ]
    rows = []
    for label, key in metric_names:
        values = report["summary"][key]
        rows.append(
            "<tr><th>{}</th><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(label),
                html.escape(str(values["min"])),
                html.escape(str(values["p10"])),
                html.escape(str(values["p50"])),
                html.escape(str(values["p90"])),
                html.escape(str(values["max"])),
            )
        )
    return "\n".join(rows)


def timeline_html(report: Dict[str, Any]) -> str:
    duration = report["video"].get("duration_seconds")
    timestamps = report["summary"].get("selected_frame_timeline_seconds", [])
    if not duration or duration <= 0:
        return '<div class="timeline empty"></div>'
    ticks = []
    for timestamp in timestamps:
        left = max(0.0, min(100.0, float(timestamp) / float(duration) * 100.0))
        ticks.append(f'<span style="left:{left:.3f}%"></span>')
    return '<div class="timeline">{}</div>'.format("".join(ticks))


def selected_frame_rows(report: Dict[str, Any], limit: int = 100) -> str:
    selected = [frame for frame in report["frames"] if frame["decision"] in {"selected", "coverage_fallback"}]
    rows = []
    for frame in selected[:limit]:
        rows.append(
            "<tr><td>{file}</td><td>{time}</td><td>{selected_by}</td><td>{fallback_reason}</td><td>{blur}</td><td>{brightness}</td><td>{contrast}</td><td>{entropy}</td><td>{score}</td></tr>".format(
                file=html.escape(str(frame.get("output_file"))),
                time=html.escape(str(frame.get("timestamp_seconds"))),
                selected_by=html.escape(str(frame.get("selected_by"))),
                fallback_reason=html.escape(str(frame.get("reject_reason") or "")),
                blur=html.escape(str(frame.get("blur_score"))),
                brightness=html.escape(str(frame.get("brightness"))),
                contrast=html.escape(str(frame.get("contrast"))),
                entropy=html.escape(str(frame.get("entropy"))),
                score=html.escape(str(frame.get("quality_score"))),
            )
        )
    if len(selected) > limit:
        rows.append(
            '<tr><td colspan="9">Showing first {} of {} selected frames.</td></tr>'.format(
                limit,
                len(selected),
            )
        )
    return "\n".join(rows)


def write_html_report(report: Dict[str, Any], gpu_report: Dict[str, Any], reports_dir: Path) -> Path:
    output_path = reports_dir / REPORT_FILENAMES["capture_html"]
    warnings = report["warnings"]
    warning_html = (
        "<ul>{}</ul>".format("".join(f"<li>{html.escape(str(item))}</li>" for item in warnings))
        if warnings
        else "<p>No warning flags.</p>"
    )
    summary_rows = table_rows(
        {
            "video": report["video"]["path"],
            "duration_seconds": report["video"]["duration_seconds"],
            "resolution": f'{report["video"]["width"]}x{report["video"]["height"]}',
            "candidate_frames": report["summary"]["candidate_frame_count"],
            "selected_frames": report["summary"]["selected_frame_count"],
            "coverage_fallback_frames": report["summary"]["coverage_fallback_frame_count"],
            "largest_selected_gap_seconds": report["summary"]["coverage"].get("largest_selected_gap_seconds"),
            "rejected_frames": report["summary"]["rejected_frame_count"],
            "profile": report["settings"]["profile"],
            "candidate_fps": report["settings"]["candidate_fps"],
            "coverage_window_seconds": report["settings"]["coverage_window_seconds"],
            "min_frames_per_window": report["settings"]["min_frames_per_window"],
            "recommended_gpu": gpu_report["recommendation"]["suggested_gpu"],
        }
    )
    rejection_rows = table_rows(report["summary"]["rejections"] or {"none": 0})

    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Capture Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1c2430;
      --muted: #657080;
      --line: #d8dee7;
      --panel: #f6f8fb;
      --accent: #2267c7;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
      line-height: 1.45;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1, h2 {{
      line-height: 1.15;
      margin: 0 0 14px;
    }}
    h1 {{
      font-size: 30px;
    }}
    h2 {{
      font-size: 20px;
      margin-top: 32px;
    }}
    .meta {{
      color: var(--muted);
      margin-bottom: 24px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 10px 0 18px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: var(--panel);
      font-weight: 650;
    }}
    img {{
      display: block;
      max-width: 100%;
      border: 1px solid var(--line);
    }}
    .timeline {{
      position: relative;
      height: 38px;
      border: 1px solid var(--line);
      background: linear-gradient(90deg, #f9fafc, #eef3f8);
      overflow: hidden;
    }}
    .timeline span {{
      position: absolute;
      top: 5px;
      bottom: 5px;
      width: 2px;
      background: var(--accent);
    }}
    .empty {{
      background: var(--panel);
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      background: var(--panel);
      padding: 1px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
<main>
  <h1>Capture Report</h1>
  <div class="meta">Generated {created_at}</div>

  <h2>Summary</h2>
  <table>{summary_rows}</table>

  <h2>Warning Flags</h2>
  {warning_html}

  <h2>Selected Frame Timeline</h2>
  {timeline}
  <p class="note">Each blue tick is a final selected frame across the source video duration.</p>

  <h2>Contact Sheet</h2>
  <img src="{contact_sheet}" alt="Selected frame contact sheet">

  <h2>Rejections</h2>
  <table>{rejection_rows}</table>

  <h2>Quality Distributions</h2>
  <table>
    <tr><th>Metric</th><th>Min</th><th>P10</th><th>P50</th><th>P90</th><th>Max</th></tr>
    {metric_rows}
  </table>

  <h2>Selected Frames</h2>
  <table>
    <tr><th>File</th><th>Time (s)</th><th>Selected By</th><th>Fallback Reason</th><th>Blur</th><th>Brightness</th><th>Contrast</th><th>Entropy</th><th>Score</th></tr>
    {selected_rows}
  </table>
</main>
</body>
</html>
""".format(
        created_at=html.escape(str(report["created_at"])),
        summary_rows=summary_rows,
        warning_html=warning_html,
        timeline=timeline_html(report),
        contact_sheet=html.escape(REPORT_FILENAMES["contact_sheet"]),
        rejection_rows=rejection_rows,
        metric_rows=metric_table(report),
        selected_rows=selected_frame_rows(report),
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path


def write_run_config(out_dir: Path, args: argparse.Namespace, settings: Dict[str, Any]) -> Path:
    output_path = out_dir / "run_config.json"
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "video": str(args.video),
        "out": str(args.out),
        "settings": settings,
    }
    write_json(output_path, payload)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    require_dependencies()
    settings = build_settings(args)
    validate_args(args, settings)

    out_dir = args.out
    frames_dir, reports_dir = prepare_output_dirs(out_dir, args.overwrite)
    del frames_dir

    video_info, records, selected_initial = analyze_video(args.video, settings)
    selected_final = finalize_selection(records, selected_initial, settings)
    assign_output_paths(selected_final)
    saved_count = save_selected_frames(args.video, selected_final, out_dir, settings)

    warnings = warning_flags(records, selected_final, settings)
    gpu_report = gpu_recommendation(saved_count, video_info, warnings)
    create_contact_sheet(
        out_dir=out_dir,
        reports_dir=reports_dir,
        selected_final=selected_final,
        max_frames=int(settings["contact_sheet_frames"]),
    )

    capture_report = build_capture_report(
        video_info=video_info,
        records=records,
        selected_final=selected_final,
        settings=settings,
        out_dir=out_dir,
        reports_dir=reports_dir,
        warnings=warnings,
    )

    write_json(reports_dir / REPORT_FILENAMES["capture_json"], capture_report)
    write_json(reports_dir / REPORT_FILENAMES["gpu_json"], gpu_report)
    write_html_report(capture_report, gpu_report, reports_dir)
    write_run_config(out_dir, args, settings)

    print(f"Processed: {args.video}")
    print(f"Candidate frames scored: {len(records)}")
    print(f"Selected frames written: {saved_count}")
    print(f"Frames: {out_dir / 'frames_selected'}")
    print(f"Report: {reports_dir / REPORT_FILENAMES['capture_html']}")
    print(f"GPU recommendation: {gpu_report['recommendation']['suggested_gpu']}")
    if warnings:
        print("Warnings: " + ", ".join(warnings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
