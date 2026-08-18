#!/usr/bin/env python3
"""Run the CPU preprocessing stage from raw storage into app-friendly artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
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
        "--group-config-json",
        help="JSON list of per-group settings. Groups are keyed by role and location.",
    )
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
    if args.group_config_json:
        return run_group_preprocess(args, raw_dir, local_run_dir, logs_dir)
    command = build_preprocess_command(args, raw_dir, local_run_dir)
    return run_logged_command("preprocess", command, logs_dir, Path.cwd())


def run_group_preprocess(
    args: argparse.Namespace,
    raw_dir: Path,
    local_run_dir: Path,
    logs_dir: Path,
) -> CommandResult:
    """Run the existing preprocessor per manifest group and merge its artifacts.

    The individual invocations deliberately use the same public CLI as the
    legacy single-directory path. This keeps the image scoring behavior in one
    place while allowing independent settings for each source group.
    """
    try:
        requested = json.loads(args.group_config_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--group-config-json is not valid JSON: {exc}") from exc
    if not isinstance(requested, list):
        raise ValueError("--group-config-json must be a JSON list")

    manifest = read_json(raw_dir / "sources_manifest.json")
    sources = manifest.get("sources") if isinstance(manifest, dict) else []
    if not isinstance(sources, list):
        raise ValueError("Raw manifest must contain a sources list")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        role = str(source.get("role") or "coverage_image")
        location = str(source.get("location") or "unassigned")
        grouped.setdefault(f"{role}:{location}", []).append(source)

    configs = {str(item.get("group_key")): item for item in requested if isinstance(item, dict) and item.get("group_key")}
    if not configs:
        configs = {
            key: {"group_key": key, "profile": args.profile, "preprocess_args": list(args.preprocess_arg)}
            for key in grouped
        }
    run_started = time.monotonic()
    merged_reports: List[Dict[str, Any]] = []
    merged_images: List[Dict[str, Any]] = []
    local_run_dir.mkdir(parents=True, exist_ok=True)
    (local_run_dir / "frames_selected").mkdir(parents=True, exist_ok=True)
    group_summaries: List[Dict[str, Any]] = []

    for index, (group_key, config) in enumerate(sorted(configs.items())):
        group_sources = grouped.get(group_key, [])
        if not group_sources:
            continue
        group_root = logs_dir.parent / "groups" / f"{index:03d}_{slug(group_key)}"
        group_input = group_root / "input"
        group_output = group_root / "output"
        group_input.mkdir(parents=True, exist_ok=True)
        for source in group_sources:
            relative = Path(str(source["relative_path"]))
            source_path = raw_dir / relative
            if not source_path.exists():
                raise FileNotFoundError(f"Manifest source is missing from raw download: {relative}")
            role = str(source.get("role") or "coverage_image")
            location = str(source.get("location") or "unassigned")
            if role == "hero_image":
                destination = group_input / "hero" / slug(location) / source_path.name
            else:
                destination = group_input / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

        profile = str(config.get("profile") or args.profile)
        command = [args.python_bin, "scripts/preprocess_video.py", "--input-dir", str(group_input), "--out", str(group_output), "--profile", profile, "--overwrite"]
        command.extend(str(value) for value in config.get("preprocess_args", args.preprocess_arg))
        result = run_logged_command(f"preprocess_group_{slug(group_key)}", command, logs_dir, Path.cwd())
        report = read_json(group_output / "reports" / "capture_report.json")
        image_manifest = read_json(group_output / "reports" / "image_manifest.json")
        prefix = slug(group_key)
        patch_group_artifacts(report, image_manifest, prefix, group_key, local_run_dir)
        merged_reports.append(report)
        merged_images.extend(image_manifest.get("images", []) if isinstance(image_manifest, dict) else [])
        group_summaries.append({"group_key": group_key, "profile": profile, "command": command, "duration_seconds": result.duration_seconds})

    aggregate_report = merge_group_reports(merged_reports, group_summaries, raw_dir, local_run_dir)
    reports_dir = local_run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "capture_report.json", aggregate_report)
    write_json(reports_dir / "image_manifest.json", {"schema_version": 1, "images": merged_images, "camera_groups": camera_groups_from_images(merged_images)})
    write_json(reports_dir / "gpu_recommendation.json", {"schema_version": 1, "recommendation": {"suggested_gpu": "RTX 2000 Ada"}, "warnings": []})
    return CommandResult(
        name="preprocess_groups",
        command=["grouped preprocess", *[item["group_key"] for item in group_summaries]],
        log_path=str(logs_dir),
        returncode=0,
        started_at=utc_now(),
        finished_at=utc_now(),
        duration_seconds=time.monotonic() - run_started,
        cwd=str(Path.cwd()),
    )


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_") or "group"


def patch_group_artifacts(report: Dict[str, Any], image_manifest: Dict[str, Any], prefix: str, group_key: str, aggregate_dir: Path) -> None:
    source_root = Path(str(report.get("out") or ""))
    for path in sorted(source_root.glob("frames_selected/*")) if source_root.exists() else []:
        destination_name = f"{prefix}_{path.name}"
        shutil.copy2(path, aggregate_dir / "frames_selected" / destination_name)
        old = f"frames_selected/{path.name}"
        new = f"frames_selected/{destination_name}"
        replace_artifact_path(report, old, new)
        replace_artifact_path(image_manifest, old, new)
    for item in image_manifest.get("images", []) if isinstance(image_manifest, dict) else []:
        if isinstance(item, dict):
            role = str(item.get("role") or "coverage")
            location = group_key.split(":", 1)[1] if ":" in group_key else "unassigned"
            item["group_key"] = group_key
            item["location"] = location
            item["camera_group"] = f"{role}_{slug(location)}"
    for video in report.get("videos", []) if isinstance(report, dict) else []:
        if isinstance(video, dict):
            video["group_key"] = group_key
            video["location"] = group_key.split(":", 1)[1] if ":" in group_key else "unassigned"
    for frame in report.get("frames", []) if isinstance(report, dict) else []:
        if isinstance(frame, dict):
            frame["group_key"] = group_key
            frame["location"] = group_key.split(":", 1)[1] if ":" in group_key else "unassigned"
    for hero in report.get("hero_images", []) if isinstance(report, dict) else []:
        if isinstance(hero, dict):
            hero["group_key"] = group_key


def replace_artifact_path(value: Any, old: str, new: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and child == old:
                value[key] = new
            else:
                replace_artifact_path(child, old, new)
    elif isinstance(value, list):
        for child in value:
            replace_artifact_path(child, old, new)


def merge_group_reports(reports: List[Dict[str, Any]], groups: List[Dict[str, Any]], raw_dir: Path, out_dir: Path) -> Dict[str, Any]:
    videos = [video for report in reports for video in report.get("videos", []) if isinstance(video, dict)]
    coverage_images = [report.get("coverage_images") for report in reports if isinstance(report.get("coverage_images"), dict)]
    heroes = [report.get("hero") for report in reports if isinstance(report.get("hero"), dict)]
    hero_images = [hero for report in reports for hero in report.get("hero_images", []) if isinstance(hero, dict)]
    frames = [frame for report in reports for frame in report.get("frames", []) if isinstance(frame, dict)]
    selected = sum(int((report.get("summary") or {}).get("selected_frame_count") or 0) for report in reports)
    selected_images = sum(int((report.get("summary") or {}).get("selected_coverage_image_count") or 0) for report in reports)
    hero_count = sum(int(hero.get("image_count") or 0) for hero in heroes)
    summary = {
        "selected_frame_count": selected,
        "selected_coverage_image_count": selected_images,
        "hero_image_count": hero_count,
        "total_image_count": selected + selected_images + hero_count,
        "candidate_frame_count": sum(int((r.get("summary") or {}).get("candidate_frame_count") or 0) for r in reports),
        "selected_by": merge_counts([(r.get("summary") or {}).get("selected_by", {}) for r in reports]),
        "rejections": merge_counts([(r.get("summary") or {}).get("rejections", {}) for r in reports]),
        "coverage_fallback_frame_count": 0,
        "rejected_frame_count": sum(int((r.get("summary") or {}).get("rejected_frame_count") or 0) for r in reports),
    }
    for metric in ("blur_score", "brightness", "contrast", "entropy"):
        values = [frame.get(metric) for frame in frames if isinstance(frame.get(metric), (int, float))]
        summary[f"{metric}_distribution"] = metric_distribution(values)
    return {
        "schema_version": 1, "created_at": utc_now(), "command": "grouped preprocess",
        "input": {"mode": "grouped_project_media", "input_dir": str(raw_dir), "group_count": len(groups)},
        "settings": {"groups": groups}, "videos": videos, "coverage_images": merge_dict_counts(coverage_images),
        "hero": merge_hero_summaries(heroes), "hero_images": hero_images, "frames": frames,
        "summary": summary,
        "warnings": sorted({warning for report in reports for warning in report.get("warnings", [])}),
        "out": str(out_dir),
    }


def merge_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            result[key] = result.get(key, 0) + int(value or 0)
    return dict(sorted(result.items()))


def metric_distribution(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {key: None for key in ("min", "p10", "p25", "p50", "p75", "p90", "max", "mean")}
    ordered = sorted(float(value) for value in values)
    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"min": ordered[0], "p10": percentile(.10), "p25": percentile(.25), "p50": percentile(.50), "p75": percentile(.75), "p90": percentile(.90), "max": ordered[-1], "mean": sum(ordered) / len(ordered)}


def merge_dict_counts(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"source_id": "coverage_images", "image_count": 0, "selected_image_count": 0, "selected_initial_count": 0, "rejected_image_count": 0, "rejections": {}, "warnings": []}
    for item in items:
        for key in ("image_count", "selected_image_count", "selected_initial_count", "rejected_image_count"):
            result[key] += int(item.get(key) or 0)
        result["warnings"].extend(item.get("warnings") or [])
    result["warnings"] = sorted(set(result["warnings"]))
    return result


def merge_hero_summaries(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    locations: Dict[str, int] = {}
    for item in items:
        for location, count in (item.get("locations") or {}).items():
            locations[location] = locations.get(location, 0) + int(count)
    return {"image_count": sum(locations.values()), "locations": locations}


def camera_groups_from_images(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for image in images:
        key = f"{image.get('camera_group')}_{image.get('width')}x{image.get('height')}"
        group = groups.setdefault(key, {"id": key, "camera_group": image.get("camera_group"), "role": image.get("role"), "width": image.get("width"), "height": image.get("height"), "image_count": 0, "locations": []})
        group["image_count"] += 1
        if image.get("location") and image["location"] not in group["locations"]:
            group["locations"].append(image["location"])
    return sorted(groups.values(), key=lambda item: item["id"])


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
        "min_blur",
        "min_brightness",
        "max_brightness",
        "min_contrast",
        "min_entropy",
        "force_keep_interval",
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
