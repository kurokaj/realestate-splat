#!/usr/bin/env python3
"""Run the Nerfstudio/Splatfacto stage from R2-backed COLMAP artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from shutil import which
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.cli import CommandResult, run_logged_command, utc_now, write_json  # noqa: E402
from realestate_splat.stage_contract import StageResult, write_stage_result  # noqa: E402
from realestate_splat.storage import sync_directory  # noqa: E402


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
    command_results: Sequence[CommandResult],
) -> None:
    current_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    summary = training_summary(local_run_dir, command_results)
    write_json(current_dir / "training_summary.json", summary)
    write_json(history_dir / "training_summary.json", summary)
    copy_training_outputs(local_run_dir, current_dir)
    copy_tree(local_run_dir / "exports", current_dir / "exports")
    copy_if_exists(local_run_dir / "nerfstudio" / "transforms.json", current_dir / "nerfstudio" / "transforms.json")

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


def training_summary(local_run_dir: Path, command_results: Sequence[CommandResult]) -> Dict[str, Any]:
    output_dir = local_run_dir / "gsplat" / "outputs"
    exports_dir = local_run_dir / "exports"
    config_path = latest_file(output_dir, "config.yml")
    dataparser_path = latest_file(output_dir, "dataparser_transforms.json")
    checkpoint_files = sorted(output_dir.glob("**/*.ckpt"))
    ply_files = sorted(exports_dir.glob("*.ply")) if exports_dir.exists() else []
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "output_dir": relative_or_string(output_dir, local_run_dir),
        "exports_dir": relative_or_string(exports_dir, local_run_dir) if exports_dir.exists() else None,
        "exported_ply": relative_or_string(ply_files[-1] if ply_files else None, local_run_dir),
        "selected_config": relative_or_string(config_path, local_run_dir),
        "dataparser_transforms": relative_or_string(dataparser_path, local_run_dir),
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


def upload_payloads(args: argparse.Namespace, stage_run_id: str, current_dir: Path, history_dir: Path) -> None:
    output = args.output_uri.rstrip("/")
    sync_directory(current_dir, f"{output}/current", endpoint_url=args.endpoint_url, delete=True)
    sync_directory(history_dir, f"{output}/runs/{stage_run_id}", endpoint_url=args.endpoint_url, delete=True)


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
