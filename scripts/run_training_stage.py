#!/usr/bin/env python3
"""Run the Nerfstudio/Splatfacto stage from R2-backed COLMAP artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import struct
from shutil import which
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.cli import CommandResult, run_logged_command, utc_now, write_json  # noqa: E402
from realestate_splat.stage_contract import StageResult, write_stage_result  # noqa: E402
from realestate_splat.storage import copy_file, sync_directory  # noqa: E402
from controller_common.preprocess_assembly import assemble_preprocess_groups_local, parse_group_output_specs  # noqa: E402


DEFAULT_PIXI_BIN = Path("/opt/buildvision/pixi/bin/pixi")
LEGACY_PIXI_BIN = Path("/workspace/pixi/bin/pixi")
DEFAULT_NERFSTUDIO_DIR = Path("/opt/buildvision/nerfstudio")
LEGACY_NERFSTUDIO_DIR = Path("/workspace/opt/nerfstudio")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download preprocess/COLMAP artifacts, run Splatfacto, and upload current/history stage artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-id", required=True, help="Stable project id.")
    parser.add_argument(
        "--preprocess-uri",
        required=True,
        help="Preprocess current URI, e.g. r2://bucket/projects/id/preprocess/current.",
    )
    parser.add_argument("--colmap-uri", required=True, help="COLMAP current URI, e.g. r2://bucket/projects/id/colmap/current.")
    parser.add_argument(
        "--preprocess-group-output",
        action="append",
        default=[],
        metavar="JSON",
        help="Approved grouped preprocess output as JSON; groups are assembled locally for training.",
    )
    parser.add_argument("--output-uri", required=True, help="Training output URI, e.g. r2://bucket/projects/id/training.")
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint URL. For r2://, R2_ENDPOINT is used by default.")
    parser.add_argument("--stage-run-id", help="Stable training run id. Defaults to a UTC timestamp.")
    parser.add_argument("--pipeline-run-id", help="Optional parent pipeline run id for stage_result.json.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used for local prep scripts when --no-prepare-with-pixi is set.",
    )
    parser.add_argument(
        "--prepare-with-pixi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run prepare_nerfstudio_from_colmap.py inside the Pixi/Nerfstudio environment.",
    )
    parser.add_argument("--pixi-bin", default=str(DEFAULT_PIXI_BIN), help="Path to Pixi in the Nerfstudio image.")
    parser.add_argument("--nerfstudio-dir", default=str(DEFAULT_NERFSTUDIO_DIR), help="Nerfstudio source directory with pixi.toml.")
    parser.add_argument("--method", default="splatfacto", help="Nerfstudio method, e.g. splatfacto.")
    parser.add_argument("--experiment-name", help="Nerfstudio experiment name. Defaults to project id.")
    parser.add_argument("--max-steps", type=int, default=100, help="Training iterations for smoke/full runs.")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint interval.")
    parser.add_argument("--eval-every", type=int, default=50, help="Eval interval.")
    parser.add_argument("--num-downscales", type=int, default=1, help="Downscale levels for prepared Nerfstudio data.")
    parser.add_argument(
        "--export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export the latest trained gaussian splat to training/current/exports/splat.ply.",
    )
    parser.add_argument(
        "--use-scale-regularization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicit Splatfacto scale regularization setting. Omit to use the Nerfstudio method default.",
    )
    parser.add_argument("--export-name", default="splat.ply", help="Canonical exported PLY filename under exports/.")
    parser.add_argument(
        "--train-option",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra raw argument passed to ns-train. Repeat for multiple args.",
    )
    parser.add_argument("--work-dir", type=Path, help="Scratch directory. Defaults to a temporary directory.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep temporary scratch files after completion.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned storage and training commands without running them.")
    return parser.parse_args(argv)


def stage_run_id() -> str:
    timestamp = utc_now().split(".", 1)[0]
    return timestamp.replace("+00:00", "Z").replace(":", "").replace("-", "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        args.pixi_bin = str(resolve_pixi_bin(args.pixi_bin))
        args.nerfstudio_dir = str(resolve_nerfstudio_dir(args.nerfstudio_dir))
    run_id = args.stage_run_id or f"training_{stage_run_id()}"

    if args.work_dir is not None:
        work_dir = args.work_dir.expanduser()
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix=f"buildvision3d-{args.project_id}-{run_id}-")
        work_dir = Path(temp_dir.name)

    preprocess_dir = work_dir / "preprocess_current"
    colmap_dir = work_dir / "colmap_current"
    local_run_dir = work_dir / "training_run"
    logs_dir = work_dir / "logs"
    current_dir = work_dir / "upload_current"
    history_dir = work_dir / "upload_history"
    started_at = utc_now()

    try:
        if args.dry_run:
            print_plan(args, run_id, preprocess_dir, colmap_dir, local_run_dir, current_dir, history_dir)
            return 0

        preprocess_dir.mkdir(parents=True, exist_ok=True)
        colmap_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        group_outputs = parse_group_output_specs(args.preprocess_group_output)
        if group_outputs:
            assemble_preprocess_groups_local(
                group_outputs=group_outputs,
                destination_dir=preprocess_dir,
                endpoint_url=args.endpoint_url,
                project_id=args.project_id,
            )
        else:
            sync_directory(args.preprocess_uri, preprocess_dir, endpoint_url=args.endpoint_url)
        sync_directory(args.colmap_uri, colmap_dir, endpoint_url=args.endpoint_url)
        prepare_local_run(preprocess_dir, colmap_dir, local_run_dir)

        command_results = run_training(args, local_run_dir, logs_dir)
        prepare_upload_payloads(
            project_id=args.project_id,
            pipeline_run_id=args.pipeline_run_id,
            stage_run_id=run_id,
            preprocess_uri=args.preprocess_uri,
            colmap_uri=args.colmap_uri,
            output_uri=args.output_uri,
            local_run_dir=local_run_dir,
            logs_dir=logs_dir,
            current_dir=current_dir,
            history_dir=history_dir,
            started_at=started_at,
            method=args.method,
            command_results=command_results,
        )

        upload_payloads(args, run_id, current_dir, history_dir)
        print(f"Training stage complete: {run_id}")
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
                    preprocess_uri=args.preprocess_uri,
                    colmap_uri=args.colmap_uri,
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
        print(f"Training stage failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_dir is not None and not args.keep_work_dir:
            temp_dir.cleanup()
        elif args.keep_work_dir:
            print(f"Kept work directory: {work_dir}")


def resolve_pixi_bin(configured: str) -> Path:
    path_hit = which("pixi")
    if path_hit:
        return Path(path_hit)

    candidates = [
        Path(configured),
        DEFAULT_PIXI_BIN,
        LEGACY_PIXI_BIN,
        Path("/usr/local/bin/pixi"),
        Path("/usr/bin/pixi"),
        Path.home() / ".pixi" / "bin" / "pixi",
    ]
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists() and expanded.is_file():
            return expanded
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find pixi. Checked: {joined}")


def resolve_nerfstudio_dir(configured: str) -> Path:
    candidates = [
        Path(configured),
        DEFAULT_NERFSTUDIO_DIR,
        LEGACY_NERFSTUDIO_DIR,
        Path("/opt/nerfstudio"),
        Path("/workspace/nerfstudio"),
    ]
    env_value = os.environ.get("NERFSTUDIO_DIR")
    if env_value:
        candidates.append(Path(env_value))
    for candidate in candidates:
        expanded = candidate.expanduser()
        if (expanded / "pixi.toml").exists():
            return expanded
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find Nerfstudio pixi.toml. Checked: {joined}")


def print_plan(
    args: argparse.Namespace,
    run_id: str,
    preprocess_dir: Path,
    colmap_dir: Path,
    local_run_dir: Path,
    current_dir: Path,
    history_dir: Path,
) -> None:
    print(f"Stage run id: {run_id}")
    print(f"$ sync {args.preprocess_uri} -> {preprocess_dir}")
    print(f"$ sync {args.colmap_uri} -> {colmap_dir}")
    print(f"$ prepare local training run -> {local_run_dir}")
    for command in build_training_commands(args, local_run_dir):
        print("$ " + " ".join(command))
    if args.export:
        print(
            "$ "
            + " ".join(
                build_export_command(
                    args,
                    local_run_dir,
                    local_run_dir / "gsplat" / "outputs" / "<experiment>" / "splatfacto" / "<timestamp>" / "config.yml",
                )
            )
        )
        print(f"$ copy exported .ply -> {local_run_dir / 'exports' / args.export_name}")
        print(f"$ prepare current payload -> {current_dir}")
    print(f"$ prepare history payload -> {history_dir}")
    print(f"$ sync {current_dir} -> {args.output_uri.rstrip('/')}/current")
    print(f"$ sync {history_dir} -> {args.output_uri.rstrip('/')}/runs/{run_id}")


def prepare_local_run(preprocess_dir: Path, colmap_dir: Path, local_run_dir: Path) -> None:
    frames_dir = preprocess_dir / "frames_selected"
    sparse_txt_dir = colmap_dir / "sparse_txt"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Preprocess input is missing frames_selected/: {frames_dir}")
    if not sparse_txt_dir.exists():
        raise FileNotFoundError(f"COLMAP input is missing sparse_txt/: {sparse_txt_dir}")

    copy_tree(frames_dir, local_run_dir / "frames_selected")
    copy_tree(sparse_txt_dir, local_run_dir / "colmap" / "sparse_txt")
    copy_tree(colmap_dir / "sparse", local_run_dir / "colmap" / "sparse")

    reports_dir = local_run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(preprocess_dir / "image_manifest.json", reports_dir / "image_manifest.json")
    copy_if_exists(preprocess_dir / "capture_report.json", reports_dir / "capture_report.json")
    copy_if_exists(preprocess_dir / "preprocess_summary.json", reports_dir / "preprocess_summary.json")
    copy_if_exists(colmap_dir / "reconstruction_report.json", reports_dir / "reconstruction_report.json")
    copy_if_exists(colmap_dir / "stage_result.json", reports_dir / "colmap_stage_result.json")


def build_training_commands(args: argparse.Namespace, local_run_dir: Path) -> List[List[str]]:
    data_dir = local_run_dir / "nerfstudio"
    output_dir = local_run_dir / "gsplat" / "outputs"
    experiment_name = args.experiment_name or args.project_id
    prepare_args = [
        "scripts/prepare_nerfstudio_from_colmap.py",
        "--run",
        str(local_run_dir),
        "--frames-dir",
        str(local_run_dir / "frames_selected"),
        "--data-dir",
        str(data_dir),
        "--colmap-model-dir",
        str(local_run_dir / "colmap" / "sparse_txt"),
        "--num-downscales",
        str(args.num_downscales),
        "--overwrite",
    ]
    if args.prepare_with_pixi:
        prepare_command = [
            args.pixi_bin,
            "run",
            "--manifest-path",
            str(Path(args.nerfstudio_dir) / "pixi.toml"),
            "python",
            *prepare_args,
        ]
    else:
        prepare_command = [args.python_bin, *prepare_args]
    train_command = [
        args.pixi_bin,
        "run",
        "--manifest-path",
        str(Path(args.nerfstudio_dir) / "pixi.toml"),
        "ns-train",
        args.method,
        f"--data={data_dir}",
        f"--output-dir={output_dir}",
        f"--experiment-name={experiment_name}",
        f"--max-num-iterations={args.max_steps}",
        f"--steps-per-save={args.save_every}",
        f"--steps-per-eval-batch={args.eval_every}",
        f"--steps-per-eval-image={args.eval_every}",
        "--viewer.quit-on-train-completion=True",
    ]
    if args.use_scale_regularization is not None:
        value = "True" if args.use_scale_regularization else "False"
        train_command.append(f"--pipeline.model.use-scale-regularization={value}")
    train_command.extend(args.train_option)
    return [prepare_command, train_command]


def build_export_command(args: argparse.Namespace, local_run_dir: Path, load_config: Path) -> List[str]:
    export_dir = local_run_dir / "gsplat" / "exports" / "gaussian_splat"
    return [
        args.pixi_bin,
        "run",
        "--manifest-path",
        str(Path(args.nerfstudio_dir) / "pixi.toml"),
        "ns-export",
        "gaussian-splat",
        "--load-config",
        str(load_config),
        "--output-dir",
        str(export_dir),
    ]


def run_training(args: argparse.Namespace, local_run_dir: Path, logs_dir: Path) -> List[CommandResult]:
    gsplat_dir = local_run_dir / "gsplat"
    gsplat_dir.mkdir(parents=True, exist_ok=True)
    results = []
    prepare_command, train_command = build_training_commands(args, local_run_dir)
    results.append(run_logged_command("prepare_nerfstudio_data", prepare_command, logs_dir, Path.cwd()))
    validate_nerfstudio_colmap_initialization(local_run_dir)
    results.append(run_logged_command("train_splatfacto", train_command, logs_dir, gsplat_dir))
    if args.export:
        config_path = latest_file(local_run_dir / "gsplat" / "outputs", "config.yml")
        if config_path is None:
            raise RuntimeError("Training finished but no config.yml was found for export.")
        export_command = build_export_command(args, local_run_dir, config_path)
        results.append(run_logged_command("export_gaussian_splat", export_command, logs_dir, gsplat_dir))
        source_ply = find_exported_ply(local_run_dir / "gsplat" / "exports" / "gaussian_splat")
        if source_ply is None:
            raise RuntimeError("ns-export finished but no .ply file was found.")
        export_output = local_run_dir / "exports" / args.export_name
        export_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_ply, export_output)
    return results


def validate_nerfstudio_colmap_initialization(local_run_dir: Path) -> None:
    transforms_path = local_run_dir / "nerfstudio" / "transforms.json"
    if not transforms_path.exists():
        raise RuntimeError(f"Nerfstudio transforms.json is missing: {transforms_path}")
    transforms = read_json(transforms_path)
    ply_file_path = transforms.get("ply_file_path")
    if not ply_file_path:
        raise RuntimeError("Nerfstudio transforms.json is missing ply_file_path; refusing possible random initialization.")
    point_cloud_path = (transforms_path.parent / str(ply_file_path)).resolve()
    if not point_cloud_path.exists():
        raise RuntimeError(f"Nerfstudio COLMAP initialization PLY is missing: {point_cloud_path}")
    point_count = parse_ply_vertex_count(point_cloud_path)
    if not point_count:
        raise RuntimeError(f"Nerfstudio COLMAP initialization PLY has no vertices: {point_cloud_path}")
    print(f"Verified COLMAP sparse initialization: {point_count} points from {point_cloud_path}", flush=True)


def prepare_upload_payloads(
    *,
    project_id: str,
    pipeline_run_id: Optional[str],
    stage_run_id: str,
    preprocess_uri: str,
    colmap_uri: str,
    output_uri: str,
    local_run_dir: Path,
    logs_dir: Path,
    current_dir: Path,
    history_dir: Path,
    started_at: str,
    method: str,
    command_results: Sequence[CommandResult],
) -> None:
    current_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    summary = training_summary(local_run_dir, command_results, method=method)
    write_json(current_dir / "training_summary.json", summary)
    write_json(history_dir / "training_summary.json", summary)
    copy_training_outputs(local_run_dir, current_dir)
    copy_tree(local_run_dir / "exports", current_dir / "exports")
    copy_if_exists(local_run_dir / "nerfstudio" / "transforms.json", current_dir / "nerfstudio" / "transforms.json")
    copy_if_exists(local_run_dir / "nerfstudio" / "colmap_points3D.ply", current_dir / "nerfstudio" / "colmap_points3D.ply")

    finished_at = utc_now()
    result = StageResult(
        schema_version=1,
        project_id=project_id,
        pipeline_run_id=pipeline_run_id,
        stage_run_id=stage_run_id,
        stage="training",
        status="completed",
        started_at=started_at,
        finished_at=finished_at,
        input_uris=[preprocess_uri.rstrip("/"), colmap_uri.rstrip("/")],
        output_uris=[
            f"{output_uri.rstrip('/')}/current",
            f"{output_uri.rstrip('/')}/runs/{stage_run_id}",
        ],
        artifact_manifest_uri=None,
        logs_uri=None,
        metrics_uri=f"{output_uri.rstrip('/')}/current/training_summary.json",
        metadata={"summary": summary},
    )
    write_stage_result(current_dir / "stage_result.json", result)
    write_stage_result(history_dir / "stage_result.json", result)


def copy_training_outputs(local_run_dir: Path, current_dir: Path) -> None:
    output_dir = local_run_dir / "gsplat" / "outputs"
    if not output_dir.exists():
        raise FileNotFoundError(f"Training did not create output directory: {output_dir}")
    copy_tree(output_dir, current_dir / "outputs")


def prepare_failed_payloads(
    *,
    project_id: str,
    pipeline_run_id: Optional[str],
    stage_run_id: str,
    preprocess_uri: str,
    colmap_uri: str,
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
    if (local_run_dir / "gsplat" / "outputs").exists():
        copy_tree(local_run_dir / "gsplat" / "outputs", current_dir / "outputs_partial")

    finished_at = utc_now()
    result = StageResult(
        schema_version=1,
        project_id=project_id,
        pipeline_run_id=pipeline_run_id,
        stage_run_id=stage_run_id,
        stage="training",
        status="failed",
        started_at=started_at,
        finished_at=finished_at,
        input_uris=[preprocess_uri.rstrip("/"), colmap_uri.rstrip("/")],
        output_uris=[f"{output_uri.rstrip('/')}/current"],
        logs_uri=f"{output_uri.rstrip('/')}/current/logs",
        metrics_uri=None,
        error_message=str(error),
        metadata={},
    )
    write_stage_result(current_dir / "stage_result.json", result)
    write_stage_result(history_dir / "stage_result.json", result)


def training_summary(local_run_dir: Path, command_results: Sequence[CommandResult], *, method: str) -> Dict[str, Any]:
    output_dir = local_run_dir / "gsplat" / "outputs"
    exports_dir = local_run_dir / "exports"
    config_path = latest_file(output_dir, "config.yml")
    dataparser_path = latest_file(output_dir, "dataparser_transforms.json")
    checkpoint_files = sorted(output_dir.glob("**/*.ckpt"))
    ply_files = sorted(exports_dir.glob("*.ply")) if exports_dir.exists() else []
    transforms = read_json(local_run_dir / "nerfstudio" / "transforms.json")
    buildvision3d = transforms.get("buildvision3d") if isinstance(transforms.get("buildvision3d"), dict) else {}
    init_ply = local_run_dir / "nerfstudio" / str(transforms.get("ply_file_path") or "colmap_points3D.ply")
    exported_ply = ply_files[-1] if ply_files else None
    gaussian_stats = gaussian_ply_diagnostics(exported_ply, buildvision3d.get("point_cloud_stats", {}))
    diagnostics = build_training_diagnostics(
        colmap_init_point_count=parse_ply_vertex_count(init_ply),
        colmap_init_stats=buildvision3d.get("point_cloud_stats", {}),
        checkpoint_count=len(checkpoint_files),
        exported_ply_vertices=parse_ply_vertex_count(exported_ply),
        gaussian_stats=gaussian_stats,
    )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "method": method,
        "output_dir": relative_or_string(output_dir, local_run_dir),
        "exports_dir": relative_or_string(exports_dir, local_run_dir) if exports_dir.exists() else None,
        "exported_ply": relative_or_string(exported_ply, local_run_dir),
        "exported_ply_vertices": parse_ply_vertex_count(exported_ply),
        "gaussian_diagnostics": gaussian_stats,
        "training_diagnostics": diagnostics,
        "selected_config": relative_or_string(config_path, local_run_dir),
        "dataparser_transforms": relative_or_string(dataparser_path, local_run_dir),
        "colmap_init_ply": relative_or_string(init_ply if init_ply.exists() else None, local_run_dir),
        "colmap_init_point_count": parse_ply_vertex_count(init_ply),
        "colmap_init_stats": buildvision3d.get("point_cloud_stats", {}),
        "checkpoint_count": len(checkpoint_files),
        "latest_checkpoint": relative_or_string(checkpoint_files[-1] if checkpoint_files else None, local_run_dir),
        "commands": [
            {
                "name": result.name,
                "command": result.command,
                "duration_seconds": result.duration_seconds,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
            }
            for result in command_results
        ],
    }


def build_training_diagnostics(
    *,
    colmap_init_point_count: Optional[int],
    colmap_init_stats: dict[str, Any],
    checkpoint_count: int,
    exported_ply_vertices: Optional[int],
    gaussian_stats: dict[str, Any],
) -> Dict[str, Any]:
    return {
        "colmap_init_point_count": colmap_init_point_count,
        "colmap_init_xyz_min": colmap_init_stats.get("xyz_min"),
        "colmap_init_xyz_max": colmap_init_stats.get("xyz_max"),
        "colmap_init_error_median": colmap_init_stats.get("reprojection_error_median"),
        "checkpoint_count": checkpoint_count,
        "exported_ply_vertices": exported_ply_vertices,
        "oversized_gaussian_detected": gaussian_stats.get("oversized_gaussian_detected"),
        "oversized_gaussian_count": gaussian_stats.get("oversized_gaussian_count"),
        "oversized_gaussian_ratio_max": gaussian_stats.get("oversized_gaussian_ratio_max"),
        "gaussian_scale_p95": gaussian_stats.get("scale_exp_max_axis_p95"),
        "gaussian_scale_p99": gaussian_stats.get("scale_exp_max_axis_p99"),
        "gaussian_scale_max": gaussian_stats.get("scale_exp_max_axis_max"),
        "gaussian_anisotropy_p99": gaussian_stats.get("anisotropy_p99"),
        "gaussian_anisotropy_max": gaussian_stats.get("anisotropy_max"),
    }


def latest_file(root: Path, name: str) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = sorted(root.glob(f"**/{name}"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def find_exported_ply(export_dir: Path) -> Optional[Path]:
    if not export_dir.exists():
        return None
    candidates = [path for path in export_dir.glob("**/*.ply") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_size, path.stat().st_mtime))


def parse_ply_vertex_count(path: Optional[Path]) -> Optional[int]:
    if path is None or not path.exists():
        return None
    try:
        with path.open("rb") as file:
            for _ in range(200):
                raw_line = file.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                match = re.match(r"element\s+vertex\s+(\d+)", line)
                if match:
                    return int(match.group(1))
                if line == "end_header":
                    break
    except OSError:
        return None
    return None


def gaussian_ply_diagnostics(path: Optional[Path], colmap_init_stats: dict[str, Any]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    header = read_ply_header(path)
    if not header:
        return {"available": False, "reason": "could_not_read_ply_header"}
    scale_names = [name for name in ("scale_0", "scale_1", "scale_2") if name in header["property_names"]]
    if len(scale_names) != 3:
        return {
            "available": False,
            "reason": "scale_properties_missing",
            "vertex_count": header["vertex_count"],
            "properties": header["property_names"],
        }

    scale_rows = read_ply_vertex_properties(path, header, scale_names)
    if not scale_rows:
        return {"available": False, "reason": "no_scale_rows", "vertex_count": header["vertex_count"]}

    exp_axes: list[tuple[float, float, float]] = []
    max_axes: list[float] = []
    anisotropies: list[float] = []
    for row in scale_rows:
        axes = tuple(safe_exp(value) for value in row)
        exp_axes.append(axes)
        max_axis = max(axes)
        min_axis = max(min(axes), 1e-12)
        max_axes.append(max_axis)
        anisotropies.append(max_axis / min_axis)

    scene_diagonal = point_cloud_diagonal(colmap_init_stats)
    threshold_ratio = 0.05
    threshold = scene_diagonal * threshold_ratio if scene_diagonal else None
    oversized_count = sum(1 for value in max_axes if threshold is not None and value > threshold)
    max_axis = max(max_axes) if max_axes else None
    return {
        "available": True,
        "vertex_count": len(scale_rows),
        "scale_property_names": scale_names,
        "scene_bbox_diagonal": round_float(scene_diagonal),
        "oversized_threshold_scene_ratio": threshold_ratio,
        "oversized_threshold_value": round_float(threshold),
        "oversized_gaussian_detected": bool(oversized_count),
        "oversized_gaussian_count": oversized_count,
        "oversized_gaussian_fraction": round_float(oversized_count / len(max_axes)) if max_axes else None,
        "oversized_gaussian_ratio_max": round_float(max_axis / scene_diagonal) if max_axis is not None and scene_diagonal else None,
        "scale_exp_max_axis_min": round_float(percentile(max_axes, 0)),
        "scale_exp_max_axis_median": round_float(percentile(max_axes, 50)),
        "scale_exp_max_axis_p95": round_float(percentile(max_axes, 95)),
        "scale_exp_max_axis_p99": round_float(percentile(max_axes, 99)),
        "scale_exp_max_axis_max": round_float(percentile(max_axes, 100)),
        "anisotropy_median": round_float(percentile(anisotropies, 50)),
        "anisotropy_p95": round_float(percentile(anisotropies, 95)),
        "anisotropy_p99": round_float(percentile(anisotropies, 99)),
        "anisotropy_max": round_float(percentile(anisotropies, 100)),
    }


def read_ply_header(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open("rb") as file:
            first = file.readline().decode("utf-8", errors="replace").strip()
            if first != "ply":
                return None
            fmt = "ascii"
            vertex_count = 0
            vertex_properties: list[tuple[str, str]] = []
            in_vertex = False
            while True:
                raw_line = file.readline()
                if not raw_line:
                    return None
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line.startswith("format "):
                    fmt = line.split()[1]
                elif line.startswith("element "):
                    parts = line.split()
                    in_vertex = parts[1] == "vertex"
                    if in_vertex:
                        vertex_count = int(parts[2])
                elif in_vertex and line.startswith("property "):
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] != "list":
                        vertex_properties.append((parts[1], parts[2]))
                elif line == "end_header":
                    return {
                        "format": fmt,
                        "vertex_count": vertex_count,
                        "properties": vertex_properties,
                        "property_names": [name for _, name in vertex_properties],
                        "data_offset": file.tell(),
                    }
    except (OSError, ValueError):
        return None


def read_ply_vertex_properties(path: Path, header: dict[str, Any], names: Sequence[str]) -> list[tuple[float, ...]]:
    if header["format"] == "ascii":
        return read_ascii_ply_vertex_properties(path, header, names)
    if header["format"] == "binary_little_endian":
        return read_binary_ply_vertex_properties(path, header, names, endian="<")
    if header["format"] == "binary_big_endian":
        return read_binary_ply_vertex_properties(path, header, names, endian=">")
    return []


def read_ascii_ply_vertex_properties(path: Path, header: dict[str, Any], names: Sequence[str]) -> list[tuple[float, ...]]:
    property_names = header["property_names"]
    indexes = [property_names.index(name) for name in names]
    rows: list[tuple[float, ...]] = []
    with path.open("rb") as file:
        file.seek(header["data_offset"])
        for _ in range(header["vertex_count"]):
            raw_line = file.readline()
            if not raw_line:
                break
            parts = raw_line.decode("utf-8", errors="replace").strip().split()
            if len(parts) < len(property_names):
                continue
            rows.append(tuple(float(parts[index]) for index in indexes))
    return rows


def read_binary_ply_vertex_properties(path: Path, header: dict[str, Any], names: Sequence[str], *, endian: str) -> list[tuple[float, ...]]:
    type_formats = {
        "char": "b",
        "int8": "b",
        "uchar": "B",
        "uint8": "B",
        "short": "h",
        "int16": "h",
        "ushort": "H",
        "uint16": "H",
        "int": "i",
        "int32": "i",
        "uint": "I",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    properties = header["properties"]
    property_names = header["property_names"]
    fmt = endian + "".join(type_formats.get(type_name, "f") for type_name, _ in properties)
    row_struct = struct.Struct(fmt)
    indexes = [property_names.index(name) for name in names]
    rows: list[tuple[float, ...]] = []
    with path.open("rb") as file:
        file.seek(header["data_offset"])
        for _ in range(header["vertex_count"]):
            payload = file.read(row_struct.size)
            if len(payload) != row_struct.size:
                break
            values = row_struct.unpack(payload)
            rows.append(tuple(float(values[index]) for index in indexes))
    return rows


def safe_exp(value: float) -> float:
    if value > 50:
        return math.exp(50)
    if value < -50:
        return math.exp(-50)
    return math.exp(value)


def point_cloud_diagonal(stats: dict[str, Any]) -> Optional[float]:
    xyz_min = stats.get("xyz_min")
    xyz_max = stats.get("xyz_max")
    if not isinstance(xyz_min, list) or not isinstance(xyz_max, list) or len(xyz_min) != 3 or len(xyz_max) != 3:
        return None
    try:
        return math.sqrt(sum((float(high) - float(low)) ** 2 for low, high in zip(xyz_min, xyz_max)))
    except (TypeError, ValueError):
        return None


def percentile(values: Sequence[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if percent <= 0:
        return ordered[0]
    if percent >= 100:
        return ordered[-1]
    position = (len(ordered) - 1) * (percent / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def round_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 6)


def upload_payloads(args: argparse.Namespace, stage_run_id: str, current_dir: Path, history_dir: Path) -> None:
    output = args.output_uri.rstrip("/")
    validate_complete_payload(current_dir)
    sync_directory(current_dir, f"{output}/current", endpoint_url=args.endpoint_url, delete=True)
    sync_directory(history_dir, f"{output}/runs/{stage_run_id}", endpoint_url=args.endpoint_url, delete=True)
    write_upload_complete_markers(args, stage_run_id, current_dir, history_dir)


def write_upload_complete_markers(args: argparse.Namespace, stage_run_id: str, current_dir: Path, history_dir: Path) -> None:
    marker_payload = {
        "stage": "training",
        "stage_run_id": stage_run_id,
        "uploaded_at": utc_now(),
        "uploaded_objects": uploaded_objects(current_dir),
    }
    current_marker = current_dir / "upload_complete.json"
    history_marker = history_dir / "upload_complete.json"
    write_json(current_marker, marker_payload)
    write_json(history_marker, marker_payload)
    output = args.output_uri.rstrip("/")
    copy_file(current_marker, f"{output}/current/upload_complete.json", endpoint_url=args.endpoint_url)
    copy_file(history_marker, f"{output}/runs/{stage_run_id}/upload_complete.json", endpoint_url=args.endpoint_url)


def validate_complete_payload(current_dir: Path) -> None:
    stage_result_path = current_dir / "stage_result.json"
    if not stage_result_path.is_file():
        raise FileNotFoundError("Training stage payload is incomplete; missing: stage_result.json")
    stage_result = json.loads(stage_result_path.read_text(encoding="utf-8"))
    if stage_result.get("status") != "completed":
        return
    required = [
        "stage_result.json",
        "training_summary.json",
        "nerfstudio/transforms.json",
        "nerfstudio/colmap_points3D.ply",
    ]
    missing = [relative_path for relative_path in required if not (current_dir / relative_path).is_file()]
    if missing:
        raise FileNotFoundError(f"Training stage payload is incomplete; missing: {', '.join(missing)}")


def uploaded_objects(current_dir: Path) -> list[str]:
    expected = [
        "stage_result.json",
        "training_summary.json",
        "nerfstudio/transforms.json",
        "nerfstudio/colmap_points3D.ply",
    ]
    optional_patterns = [
        "outputs/**/config.yml",
        "outputs/**/*.ckpt",
        "exports/*.ply",
    ]
    existing: list[str] = [relative_path for relative_path in expected if (current_dir / relative_path).is_file()]
    for pattern in optional_patterns:
        candidates = sorted(path for path in current_dir.glob(pattern) if path.is_file())
        if candidates:
            existing.append(candidates[-1].relative_to(current_dir).as_posix())
    return existing


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    shutil.copytree(source, destination, dirs_exist_ok=True)


def relative_or_string(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
