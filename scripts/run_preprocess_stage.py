#!/usr/bin/env python3
"""Run the CPU preprocessing stage from raw storage into app-friendly artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.cli import CommandResult, run_logged_command, utc_now, write_json  # noqa: E402
from realestate_splat.stage_contract import StageResult, write_stage_result  # noqa: E402
from realestate_splat.storage import sync_directory  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw project media, run preprocessing, and upload current/history stage artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-id", required=True, help="Stable project id.")
    parser.add_argument("--raw-uri", required=True, help="Raw media URI, e.g. r2://bucket/projects/id/raw.")
    parser.add_argument("--output-uri", required=True, help="Preprocess output URI, e.g. r2://bucket/projects/id/preprocess.")
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint URL. For r2://, R2_ENDPOINT is used by default.")
    parser.add_argument("--stage-run-id", help="Stable preprocessing run id. Defaults to a UTC timestamp.")
    parser.add_argument("--pipeline-run-id", help="Optional parent pipeline run id for stage_result.json.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run scripts/preprocess_video.py.",
    )
    parser.add_argument("--profile", default="indoor_room", help="Preprocess profile passed to scripts/preprocess_video.py.")
    parser.add_argument(
        "--preprocess-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument passed to scripts/preprocess_video.py. Repeat for multiple args, e.g. --preprocess-arg=--target-max=700.",
    )
    parser.add_argument("--work-dir", type=Path, help="Scratch directory. Defaults to a temporary directory.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep temporary scratch files after completion.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned storage and preprocessing commands without running them.")
    return parser.parse_args(argv)


def stage_run_id() -> str:
    timestamp = utc_now().split(".", 1)[0]
    return timestamp.replace("+00:00", "Z").replace(":", "").replace("-", "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_id = args.stage_run_id or f"preprocess_{stage_run_id()}"

    if args.work_dir is not None:
        work_dir = args.work_dir.expanduser()
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix=f"buildvision3d-{args.project_id}-{run_id}-")
        work_dir = Path(temp_dir.name)

    raw_dir = work_dir / "raw"
    local_run_dir = work_dir / "preprocess_run"
    logs_dir = work_dir / "logs"
    current_dir = work_dir / "upload_current"
    history_dir = work_dir / "upload_history"
    started_at = utc_now()

    try:
        if args.dry_run:
            print_plan(args, run_id, raw_dir, local_run_dir, current_dir, history_dir)
            return 0

        raw_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        sync_directory(args.raw_uri, raw_dir, endpoint_url=args.endpoint_url)
        enforce_manifest_sources(raw_dir)

        preprocess_result = run_preprocess(args, raw_dir, local_run_dir, logs_dir)
        prepare_upload_payloads(
            project_id=args.project_id,
            stage_run_id=run_id,
            raw_uri=args.raw_uri,
            output_uri=args.output_uri,
            raw_dir=raw_dir,
            local_run_dir=local_run_dir,
            logs_dir=logs_dir,
            current_dir=current_dir,
            history_dir=history_dir,
            started_at=started_at,
            preprocess_result=preprocess_result,
            pipeline_run_id=args.pipeline_run_id,
        )

        upload_payloads(args, run_id, current_dir, history_dir)
        print(f"Preprocess stage complete: {run_id}")
        print(f"Current output: {args.output_uri.rstrip('/')}/current/")
        print(f"History output: {args.output_uri.rstrip('/')}/runs/{run_id}/")
        return 0
    except Exception as exc:
        if not args.dry_run:
            try:
                prepare_failed_history_payload(
                    project_id=args.project_id,
                    pipeline_run_id=args.pipeline_run_id,
                    stage_run_id=run_id,
                    raw_uri=args.raw_uri,
                    output_uri=args.output_uri,
                    current_dir=current_dir,
                    history_dir=history_dir,
                    logs_dir=logs_dir,
                    started_at=started_at,
                    error=exc,
                )
                sync_directory(
                    current_dir,
                    f"{args.output_uri.rstrip('/')}/current",
                    endpoint_url=args.endpoint_url,
                    delete=True,
                    exclude=["capture_report.html"],
                )
                sync_directory(
                    history_dir,
                    f"{args.output_uri.rstrip('/')}/runs/{run_id}",
                    endpoint_url=args.endpoint_url,
                    delete=True,
                    exclude=["capture_report.html"],
                )
            except Exception as upload_error:
                print(f"Could not upload failed stage metadata: {upload_error}", file=sys.stderr)
        print(f"Preprocess stage failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_dir is not None and not args.keep_work_dir:
            temp_dir.cleanup()
        elif args.keep_work_dir:
            print(f"Kept work directory: {work_dir}")


def print_plan(
    args: argparse.Namespace,
    run_id: str,
    raw_dir: Path,
    local_run_dir: Path,
    current_dir: Path,
    history_dir: Path,
) -> None:
    preprocess_command = build_preprocess_command(args, raw_dir, local_run_dir)
    print(f"Stage run id: {run_id}")
    print(f"$ sync {args.raw_uri} -> {raw_dir}")
    print("$ " + " ".join(preprocess_command))
    print(f"$ prepare current payload -> {current_dir}")
    print(f"$ prepare history payload -> {history_dir}")
    print(f"$ sync {current_dir} -> {args.output_uri.rstrip('/')}/current")
    print(f"$ sync {history_dir} -> {args.output_uri.rstrip('/')}/runs/{run_id}")


def build_preprocess_command(args: argparse.Namespace, raw_dir: Path, local_run_dir: Path) -> List[str]:
    command = [
        args.python_bin,
        "scripts/preprocess_video.py",
        "--input-dir",
        str(raw_dir),
        "--out",
        str(local_run_dir),
        "--profile",
        args.profile,
        "--overwrite",
    ]
    command.extend(args.preprocess_arg)
    return command


def run_preprocess(args: argparse.Namespace, raw_dir: Path, local_run_dir: Path, logs_dir: Path) -> CommandResult:
    command = build_preprocess_command(args, raw_dir, local_run_dir)
    return run_logged_command("preprocess", command, logs_dir, Path.cwd())


def prepare_upload_payloads(
    *,
    project_id: str,
    stage_run_id: str,
    raw_uri: str,
    output_uri: str,
    raw_dir: Path,
    local_run_dir: Path,
    logs_dir: Path,
    current_dir: Path,
    history_dir: Path,
    started_at: str,
    preprocess_result: CommandResult,
    pipeline_run_id: Optional[str],
) -> None:
    current_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    copy_tree(local_run_dir / "frames_selected", current_dir / "frames_selected")
    copy_if_exists(local_run_dir / "reports" / "capture_report.json", current_dir / "capture_report.json")
    copy_if_exists(local_run_dir / "reports" / "image_manifest.json", current_dir / "image_manifest.json")
    copy_if_exists(raw_dir / "sources_manifest.json", current_dir / "sources_manifest.json")

    summary = preprocess_summary(local_run_dir)
    write_json(current_dir / "preprocess_summary.json", summary)

    finished_at = utc_now()
    result = StageResult(
        schema_version=1,
        project_id=project_id,
        pipeline_run_id=pipeline_run_id,
        stage_run_id=stage_run_id,
        stage="preprocess",
        status="completed",
        started_at=started_at,
        finished_at=finished_at,
        input_uris=[raw_uri.rstrip("/")],
        output_uris=[
            f"{output_uri.rstrip('/')}/current",
            f"{output_uri.rstrip('/')}/runs/{stage_run_id}",
        ],
        artifact_manifest_uri=None,
        logs_uri=None,
        metrics_uri=f"{output_uri.rstrip('/')}/current/preprocess_summary.json",
        metadata={
            "preprocess_command": preprocess_result.command,
            "preprocess_duration_seconds": preprocess_result.duration_seconds,
            "summary": summary,
        },
    )
    write_stage_result(current_dir / "stage_result.json", result)
    write_stage_result(history_dir / "stage_result.json", result)
    copy_if_exists(local_run_dir / "reports" / "capture_report.json", history_dir / "capture_report.json")
    copy_if_exists(current_dir / "preprocess_summary.json", history_dir / "preprocess_summary.json")


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    shutil.copytree(source, destination, dirs_exist_ok=True)


def enforce_manifest_sources(raw_dir: Path) -> None:
    """Make the raw manifest authoritative over objects found in the prefix.

    The raw prefix can contain objects removed from the manifest. They must not
    leak into preprocessing simply because the storage sync still sees them.
    """
    manifest_path = raw_dir / "sources_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Raw sources manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    sources = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(sources, list):
        raise RuntimeError("Raw sources manifest must contain a sources list.")

    allowed = {"sources_manifest.json"}
    for source in sources:
        if not isinstance(source, dict) or not source.get("relative_path"):
            raise RuntimeError("Raw sources manifest contains an invalid source entry.")
        relative_path = Path(str(source["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Raw sources manifest contains an unsafe relative path: {relative_path}")
        allowed.add(relative_path.as_posix())

    for path in sorted(raw_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        relative_path = path.relative_to(raw_dir).as_posix()
        if path.is_file() and relative_path not in allowed:
            path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def preprocess_summary(local_run_dir: Path) -> Dict[str, Any]:
    capture_report = read_json(local_run_dir / "reports" / "capture_report.json")
    image_manifest = read_json(local_run_dir / "reports" / "image_manifest.json")

    summary = capture_report.get("summary") if isinstance(capture_report, dict) else {}
    images = image_manifest.get("images") if isinstance(image_manifest, dict) else []
    videos = capture_report.get("videos", []) if isinstance(capture_report, dict) else []
    coverage_images = capture_report.get("coverage_images", {}) if isinstance(capture_report, dict) else {}
    hero = capture_report.get("hero", {}) if isinstance(capture_report, dict) else {}
    return {
        "schema_version": 1,
        "created_at": capture_report.get("created_at") if isinstance(capture_report, dict) else None,
        "command": capture_report.get("command") if isinstance(capture_report, dict) else None,
        "settings": compact_settings(capture_report.get("settings", {}) if isinstance(capture_report, dict) else {}),
        "selected_frame_count": summary.get("selected_frame_count"),
        "coverage_image_count": summary.get("selected_coverage_image_count"),
        "hero_image_count": summary.get("hero_image_count"),
        "total_image_count": summary.get("total_image_count"),
        "candidate_frame_count": summary.get("candidate_frame_count"),
        "coverage_image_candidate_count": summary.get("coverage_image_candidate_count"),
        "rejected_frame_count": summary.get("rejected_frame_count"),
        "selected_by": summary.get("selected_by", {}),
        "rejections": summary.get("rejections", {}),
        "coverage_fallback_frame_count": summary.get("coverage_fallback_frame_count"),
        "metric_distributions": {
            "blur_score": summary.get("blur_score_distribution"),
            "brightness": summary.get("brightness_distribution"),
            "contrast": summary.get("contrast_distribution"),
            "entropy": summary.get("entropy_distribution"),
        },
        "warnings": capture_report.get("warnings", []) if isinstance(capture_report, dict) else [],
        "image_manifest_count": len(images) if isinstance(images, list) else None,
        "videos": [compact_video_summary(video) for video in videos if isinstance(video, dict)],
        "coverage_images": compact_coverage_image_summary(coverage_images if isinstance(coverage_images, dict) else {}),
        "hero": {
            "image_count": hero.get("image_count", 0) if isinstance(hero, dict) else 0,
            "locations": hero.get("locations", {}) if isinstance(hero, dict) else {},
        },
        "selected_timeline": selected_timeline(capture_report),
    }


def compact_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "profile",
        "candidate_fps",
        "target_min",
        "target_max",
        "coverage_window_seconds",
        "min_frames_per_window",
        "start_seconds",
        "duration_seconds",
        "jpeg_quality",
    ]
    return {key: settings.get(key) for key in keep if key in settings}


def compact_video_summary(video: Dict[str, Any]) -> Dict[str, Any]:
    coverage = video.get("coverage") or {}
    return {
        "source_id": video.get("source_id"),
        "candidate_frame_count": video.get("candidate_frame_count"),
        "selected_frame_count": video.get("selected_frame_count"),
        "selected_by": video.get("selected_by", {}),
        "rejected_frame_count": video.get("rejected_frame_count"),
        "rejections": video.get("rejections", {}),
        "warnings": video.get("warnings", []),
        "duration_seconds": ((video.get("video") or {}).get("duration_seconds") if isinstance(video.get("video"), dict) else None),
        "coverage": {
            "enabled": coverage.get("enabled"),
            "largest_selected_gap_seconds": coverage.get("largest_selected_gap_seconds"),
            "windows_below_minimum_count": coverage.get("windows_below_minimum_count"),
            "coverage_fallback_frame_count": coverage.get("coverage_fallback_frame_count"),
        },
    }


def compact_coverage_image_summary(coverage_images: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_count": coverage_images.get("image_count", 0),
        "selected_image_count": coverage_images.get("selected_image_count", 0),
        "rejected_image_count": coverage_images.get("rejected_image_count", 0),
        "rejections": coverage_images.get("rejections", {}),
        "warnings": coverage_images.get("warnings", []),
    }


def selected_timeline(capture_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    frames = capture_report.get("frames", []) if isinstance(capture_report, dict) else []
    timeline = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        if frame.get("decision") not in {"selected", "coverage_fallback"}:
            continue
        timeline.append(
            {
                "source_id": frame.get("source_id"),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "selected_by": frame.get("selected_by"),
                "decision": frame.get("decision"),
                "output_file": frame.get("output_file"),
                "quality_score": frame.get("quality_score"),
            }
        )
    return timeline


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def upload_payloads(args: argparse.Namespace, stage_run_id: str, current_dir: Path, history_dir: Path) -> None:
    output = args.output_uri.rstrip("/")
    sync_directory(
        current_dir,
        f"{output}/current",
        endpoint_url=args.endpoint_url,
        delete=True,
        exclude=["capture_report.html"],
    )
    sync_directory(
        history_dir,
        f"{output}/runs/{stage_run_id}",
        endpoint_url=args.endpoint_url,
        delete=True,
        exclude=["capture_report.html"],
    )


def prepare_failed_history_payload(
    *,
    project_id: str,
    pipeline_run_id: Optional[str],
    stage_run_id: str,
    raw_uri: str,
    output_uri: str,
    current_dir: Path,
    history_dir: Path,
    logs_dir: Path,
    started_at: str,
    error: Exception,
) -> None:
    current_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    copy_tree(logs_dir, current_dir / "logs")
    finished_at = utc_now()
    result = StageResult(
        schema_version=1,
        project_id=project_id,
        pipeline_run_id=pipeline_run_id,
        stage_run_id=stage_run_id,
        stage="preprocess",
        status="failed",
        started_at=started_at,
        finished_at=finished_at,
        input_uris=[raw_uri.rstrip("/")],
        output_uris=[f"{output_uri.rstrip('/')}/current"],
        logs_uri=f"{output_uri.rstrip('/')}/current/logs/preprocess.log",
        metrics_uri=None,
        error_message=str(error),
        metadata={},
    )
    write_stage_result(current_dir / "stage_result.json", result)
    write_stage_result(history_dir / "stage_result.json", result)


if __name__ == "__main__":
    raise SystemExit(main())
