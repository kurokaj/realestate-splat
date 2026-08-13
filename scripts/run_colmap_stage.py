#!/usr/bin/env python3
"""Run the COLMAP GPU stage from preprocess artifacts in object storage."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.cli import CommandResult, run_logged_command, utc_now, write_json  # noqa: E402
from realestate_splat.stage_contract import StageResult, write_stage_result  # noqa: E402
from realestate_splat.storage import copy_file, sync_directory  # noqa: E402
from controller_common.colmap_viewer import write_sparse_viewer_payload  # noqa: E402


DEFAULT_COLMAP_BIN = Path("/opt/colmap-cuda/bin/colmap")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download preprocess artifacts, run COLMAP, and upload current/history stage artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-id", required=True, help="Stable project id.")
    parser.add_argument(
        "--input-uri",
        required=True,
        help="Preprocess current URI, e.g. r2://bucket/projects/id/preprocess/current.",
    )
    parser.add_argument("--output-uri", required=True, help="COLMAP output URI, e.g. r2://bucket/projects/id/colmap.")
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint URL. For r2://, R2_ENDPOINT is used by default.")
    parser.add_argument("--stage-run-id", help="Stable COLMAP run id. Defaults to a UTC timestamp.")
    parser.add_argument("--pipeline-run-id", help="Optional parent pipeline run id for stage_result.json.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used to run scripts/run_colmap.py.")
    parser.add_argument(
        "--colmap-bin",
        default=str(DEFAULT_COLMAP_BIN),
        help="Absolute COLMAP binary path inside the GPU runtime.",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON/YAML config passed to scripts/run_colmap.py.")
    parser.add_argument("--mode", choices=["incremental", "global"], default="global", help="COLMAP mapper mode.")
    parser.add_argument(
        "--feature-extractor",
        choices=["SIFT", "sift", "ALIKED_N16ROT", "ALIKED_N32"],
        default="SIFT",
        help="COLMAP FeatureExtraction.type.",
    )
    parser.add_argument("--matcher", choices=["exhaustive", "sequential", "vocab_tree"], default="exhaustive")
    parser.add_argument(
        "--matching-type",
        choices=["SIFT_BRUTEFORCE", "SIFT_LIGHTGLUE", "ALIKED_BRUTEFORCE", "ALIKED_LIGHTGLUE"],
        default="SIFT_BRUTEFORCE",
        help="COLMAP FeatureMatching.type.",
    )
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL", help="COLMAP ImageReader camera model.")
    parser.add_argument("--single-camera", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-image-size", type=int, default=3200)
    parser.add_argument("--option-namespace", choices=["auto", "feature", "sift"], default="auto")
    parser.add_argument("--view-graph-calibrator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifest-camera-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--feature-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional COLMAP feature_extractor option. Repeat for multiple options.",
    )
    parser.add_argument(
        "--matcher-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional COLMAP matcher option. Repeat for multiple options.",
    )
    parser.add_argument(
        "--view-graph-calibrator-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional COLMAP view_graph_calibrator option. Repeat for multiple options.",
    )
    parser.add_argument(
        "--mapper-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional COLMAP mapper/global_mapper option. Repeat for multiple options.",
    )
    parser.add_argument(
        "--colmap-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra raw argument passed to scripts/run_colmap.py. Repeat for multiple args.",
    )
    parser.add_argument("--work-dir", type=Path, help="Scratch directory. Defaults to a temporary directory.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep temporary scratch files after completion.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned storage and COLMAP commands without running them.")
    return parser.parse_args(argv)


def stage_run_id() -> str:
    timestamp = utc_now().split(".", 1)[0]
    return timestamp.replace("+00:00", "Z").replace(":", "").replace("-", "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_id = args.stage_run_id or f"colmap_{stage_run_id()}"

    if args.work_dir is not None:
        work_dir = args.work_dir.expanduser()
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix=f"buildvision3d-{args.project_id}-{run_id}-")
        work_dir = Path(temp_dir.name)

    input_dir = work_dir / "preprocess_current"
    local_run_dir = work_dir / "colmap_run"
    logs_dir = work_dir / "logs"
    current_dir = work_dir / "upload_current"
    history_dir = work_dir / "upload_history"
    started_at = utc_now()

    try:
        if args.dry_run:
            print_plan(args, run_id, input_dir, local_run_dir, current_dir, history_dir)
            return 0

        input_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        sync_directory(args.input_uri, input_dir, endpoint_url=args.endpoint_url)
        prepare_local_run(input_dir, local_run_dir)

        colmap_result = run_colmap(args, local_run_dir, logs_dir)
        prepare_upload_payloads(
            project_id=args.project_id,
            pipeline_run_id=args.pipeline_run_id,
            stage_run_id=run_id,
            input_uri=args.input_uri,
            output_uri=args.output_uri,
            local_run_dir=local_run_dir,
            logs_dir=logs_dir,
            current_dir=current_dir,
            history_dir=history_dir,
            started_at=started_at,
            colmap_result=colmap_result,
        )

        upload_payloads(args, run_id, current_dir, history_dir)
        print(f"COLMAP stage complete: {run_id}")
        print(f"Current output: {args.output_uri.rstrip('/')}/current/")
        print(f"History output: {args.output_uri.rstrip('/')}/runs/{run_id}/")
        return 0
    except Exception as exc:
        if not args.dry_run:
            try:
                prepare_failed_payloads(
                    project_id=args.project_id,
                    pipeline_run_id=args.pipeline_run_id,
                    stage_run_id=run_id,
                    input_uri=args.input_uri,
                    output_uri=args.output_uri,
                    local_run_dir=local_run_dir,
                    logs_dir=logs_dir,
                    current_dir=current_dir,
                    history_dir=history_dir,
                    started_at=started_at,
                    error=exc,
                )
                upload_payloads(args, run_id, current_dir, history_dir)
            except Exception as upload_error:
                print(f"Could not upload failed stage metadata: {upload_error}", file=sys.stderr)
        print(f"COLMAP stage failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_dir is not None and not args.keep_work_dir:
            temp_dir.cleanup()
        elif args.keep_work_dir:
            print(f"Kept work directory: {work_dir}")


def print_plan(
    args: argparse.Namespace,
    run_id: str,
    input_dir: Path,
    local_run_dir: Path,
    current_dir: Path,
    history_dir: Path,
) -> None:
    print(f"Stage run id: {run_id}")
    print(f"$ sync {args.input_uri} -> {input_dir}")
    print(f"$ prepare local COLMAP run -> {local_run_dir}")
    print("$ " + " ".join(build_colmap_command(args, local_run_dir)))
    print(f"$ prepare current payload -> {current_dir}")
    print(f"$ prepare history payload -> {history_dir}")
    print(f"$ sync {current_dir} -> {args.output_uri.rstrip('/')}/current")
    print(f"$ sync {history_dir} -> {args.output_uri.rstrip('/')}/runs/{run_id}")


def prepare_local_run(input_dir: Path, local_run_dir: Path) -> None:
    frames_dir = input_dir / "frames_selected"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Preprocess input is missing frames_selected/: {frames_dir}")
    if not any(path.is_file() for path in frames_dir.iterdir()):
        raise RuntimeError(f"Preprocess input has no selected frame files: {frames_dir}")

    copy_tree(frames_dir, local_run_dir / "frames_selected")
    reports_dir = local_run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(input_dir / "image_manifest.json", reports_dir / "image_manifest.json")
    copy_if_exists(input_dir / "capture_report.json", reports_dir / "capture_report.json")
    copy_if_exists(input_dir / "preprocess_summary.json", reports_dir / "preprocess_summary.json")


def build_colmap_command(args: argparse.Namespace, local_run_dir: Path) -> List[str]:
    command = [
        args.python_bin,
        "scripts/run_colmap.py",
        "--run",
        str(local_run_dir),
        "--colmap-bin",
        str(args.colmap_bin),
        "--mode",
        args.mode,
        "--feature-extractor",
        args.feature_extractor,
        "--matcher",
        args.matcher,
        "--matching-type",
        args.matching_type,
        "--camera-model",
        args.camera_model,
        "--use-gpu" if args.use_gpu else "--no-use-gpu",
        "--max-image-size",
        str(args.max_image_size),
        "--option-namespace",
        args.option_namespace,
        "--view-graph-calibrator" if args.view_graph_calibrator else "--no-view-graph-calibrator",
        "--manifest-camera-groups" if args.manifest_camera_groups else "--no-manifest-camera-groups",
        "--overwrite",
    ]
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    if args.single_camera is not None:
        command.append("--single-camera" if args.single_camera else "--no-single-camera")
    for option in args.feature_option:
        command.extend(["--feature-option", option])
    for option in args.matcher_option:
        command.extend(["--matcher-option", option])
    for option in args.view_graph_calibrator_option:
        command.extend(["--view-graph-calibrator-option", option])
    for option in args.mapper_option:
        command.extend(["--mapper-option", option])
    command.extend(args.colmap_arg)
    return command


def run_colmap(args: argparse.Namespace, local_run_dir: Path, logs_dir: Path) -> CommandResult:
    command = build_colmap_command(args, local_run_dir)
    return run_logged_command("run_colmap_stage", command, logs_dir, Path.cwd())


def prepare_upload_payloads(
    *,
    project_id: str,
    pipeline_run_id: Optional[str],
    stage_run_id: str,
    input_uri: str,
    output_uri: str,
    local_run_dir: Path,
    logs_dir: Path,
    current_dir: Path,
    history_dir: Path,
    started_at: str,
    colmap_result: CommandResult,
) -> None:
    current_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    copy_colmap_outputs(local_run_dir, current_dir)
    report_path = local_run_dir / "reports" / "reconstruction_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"COLMAP finished without reconstruction_report.json: {report_path}")
    report = read_json(report_path)
    write_json(current_dir / "reconstruction_report.json", report)
    write_json(history_dir / "reconstruction_report.json", report)
    generate_viewer_payloads(current_dir=current_dir, history_dir=history_dir)

    finished_at = utc_now()
    result = StageResult(
        schema_version=1,
        project_id=project_id,
        pipeline_run_id=pipeline_run_id,
        stage_run_id=stage_run_id,
        stage="colmap",
        status="completed",
        started_at=started_at,
        finished_at=finished_at,
        input_uris=[input_uri.rstrip("/")],
        output_uris=[
            f"{output_uri.rstrip('/')}/current",
            f"{output_uri.rstrip('/')}/runs/{stage_run_id}",
        ],
        artifact_manifest_uri=None,
        logs_uri=None,
        metrics_uri=f"{output_uri.rstrip('/')}/current/reconstruction_report.json",
        metadata={
            "colmap_command": colmap_result.command,
            "colmap_duration_seconds": colmap_result.duration_seconds,
            "summary": colmap_stage_summary(report),
        },
    )
    write_stage_result(current_dir / "stage_result.json", result)
    write_stage_result(history_dir / "stage_result.json", result)


def copy_colmap_outputs(local_run_dir: Path, current_dir: Path) -> None:
    colmap_dir = local_run_dir / "colmap"
    if not colmap_dir.exists():
        raise FileNotFoundError(f"COLMAP run did not create output directory: {colmap_dir}")

    copy_if_exists(colmap_dir / "database.db", current_dir / "database.db")
    copy_if_exists(colmap_dir / "database_global.db", current_dir / "database_global.db")
    copy_tree(colmap_dir / "sparse", current_dir / "sparse")
    copy_tree(colmap_dir / "sparse_txt", current_dir / "sparse_txt")


def generate_viewer_payloads(*, current_dir: Path, history_dir: Path) -> None:
    sparse_txt_dir = current_dir / "sparse_txt"
    if not sparse_txt_dir.exists():
        return
    viewer_path = write_sparse_viewer_payload(sparse_txt_dir, current_dir / "viewer" / "sparse_scene.json")
    copy_if_exists(viewer_path, history_dir / "viewer" / "sparse_scene.json")


def prepare_failed_payloads(
    *,
    project_id: str,
    pipeline_run_id: Optional[str],
    stage_run_id: str,
    input_uri: str,
    output_uri: str,
    local_run_dir: Path,
    logs_dir: Path,
    current_dir: Path,
    history_dir: Path,
    started_at: str,
    error: Exception,
) -> None:
    current_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    copy_tree(logs_dir, current_dir / "logs")
    copy_tree(local_run_dir / "colmap" / "logs", current_dir / "logs" / "colmap")
    copy_if_exists(local_run_dir / "reports" / "reconstruction_report.json", current_dir / "reconstruction_report.json")

    finished_at = utc_now()
    result = StageResult(
        schema_version=1,
        project_id=project_id,
        pipeline_run_id=pipeline_run_id,
        stage_run_id=stage_run_id,
        stage="colmap",
        status="failed",
        started_at=started_at,
        finished_at=finished_at,
        input_uris=[input_uri.rstrip("/")],
        output_uris=[f"{output_uri.rstrip('/')}/current"],
        logs_uri=f"{output_uri.rstrip('/')}/current/logs",
        metrics_uri=f"{output_uri.rstrip('/')}/current/reconstruction_report.json",
        error_message=str(error),
        metadata={},
    )
    write_stage_result(current_dir / "stage_result.json", result)
    write_stage_result(history_dir / "stage_result.json", result)


def colmap_stage_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    if not report:
        return {}
    return {
        "status": report.get("status"),
        "mode": (report.get("settings") or {}).get("mode") if isinstance(report.get("settings"), dict) else None,
        "feature_extractor": (report.get("settings") or {}).get("feature_extractor") if isinstance(report.get("settings"), dict) else None,
        "matcher": (report.get("settings") or {}).get("matcher") if isinstance(report.get("settings"), dict) else None,
        "matching_type": (report.get("settings") or {}).get("matching_type") if isinstance(report.get("settings"), dict) else None,
        "camera_model": (report.get("settings") or {}).get("camera_model") if isinstance(report.get("settings"), dict) else None,
        "image_count": (report.get("input") or {}).get("image_count") if isinstance(report.get("input"), dict) else None,
        "selected_sparse_model": report.get("selected_sparse_model"),
        "reconstruction_metrics": report.get("reconstruction_metrics", {}),
        "manifest_reconstruction": report.get("manifest_reconstruction", {}),
        "camera_groups": report.get("camera_groups", []),
    }


def upload_payloads(args: argparse.Namespace, stage_run_id: str, current_dir: Path, history_dir: Path) -> None:
    output = args.output_uri.rstrip("/")
    sync_directory(
        current_dir,
        f"{output}/current",
        endpoint_url=args.endpoint_url,
        delete=True,
        exclude=["reconstruction_report.html"],
    )
    sync_directory(
        history_dir,
        f"{output}/runs/{stage_run_id}",
        endpoint_url=args.endpoint_url,
        delete=True,
        exclude=["reconstruction_report.html"],
    )
    write_upload_complete_markers(args, stage_run_id, current_dir, history_dir)


def write_upload_complete_markers(args: argparse.Namespace, stage_run_id: str, current_dir: Path, history_dir: Path) -> None:
    marker_payload = {
        "stage": "colmap",
        "stage_run_id": stage_run_id,
        "uploaded_at": utc_now(),
        "required_objects": required_upload_objects(current_dir),
    }
    current_marker = current_dir / "upload_complete.json"
    history_marker = history_dir / "upload_complete.json"
    write_json(current_marker, marker_payload)
    write_json(history_marker, marker_payload)
    output = args.output_uri.rstrip("/")
    copy_file(current_marker, f"{output}/current/upload_complete.json", endpoint_url=args.endpoint_url)
    copy_file(history_marker, f"{output}/runs/{stage_run_id}/upload_complete.json", endpoint_url=args.endpoint_url)


def required_upload_objects(current_dir: Path) -> list[str]:
    required = [
        "stage_result.json",
        "reconstruction_report.json",
        "viewer/sparse_scene.json",
        "sparse_txt/cameras.txt",
        "sparse_txt/images.txt",
        "sparse_txt/points3D.txt",
    ]
    return list(required)


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    shutil.copytree(source, destination, dirs_exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
