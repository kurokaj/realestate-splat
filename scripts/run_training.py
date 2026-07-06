#!/usr/bin/env python3
"""Run training for a preprocessed real estate splat run.

The first supported backend is Nerfstudio / splatfacto, executed through the
Pixi-managed Nerfstudio checkout. Reconstruction remains owned by
``scripts/run_colmap.py`` and the custom COLMAP binary.
"""

from __future__ import annotations

import argparse
import shutil
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.cli import (  # noqa: E402
    CommandResult,
    command_results_to_json,
    environment_report,
    latest_existing_file,
    parse_key_value_pairs,
    relative_to,
    resolve_executable,
    resolve_under_run,
    run_logged_command,
    utc_now,
    write_json,
)
from realestate_splat.config import load_config  # noqa: E402
from realestate_splat.training.backends import build_backend_commands, validate_backend_name  # noqa: E402


DEFAULT_NERFSTUDIO_DIR = Path("/workspace/opt/nerfstudio")
DEFAULT_COLMAP_BINARY = Path("/workspace/opt/colmap-install/bin/colmap")
REPORT_NAME = "training_report.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backend-dispatched training for a Buildvision3D run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", required=True, type=Path, help="Run directory.")
    parser.add_argument("--config", type=Path, help="Optional JSON/YAML pipeline config.")
    parser.add_argument("--backend", choices=["splatfacto", "raw_gsplat"], help="Training backend.")
    parser.add_argument("--nerfstudio-dir", type=Path, help="Pixi-managed Nerfstudio source directory.")
    parser.add_argument("--pixi-bin", type=Path, help="Path to pixi.")
    parser.add_argument("--method", help="Nerfstudio method name, e.g. splatfacto or splatfacto-big.")
    parser.add_argument("--frames-dir", type=Path, help="Input frames directory, absolute or relative to --run.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Nerfstudio processed dataset directory, absolute or relative to --run.",
    )
    parser.add_argument("--colmap-model-dir", type=Path, help="COLMAP model directory, absolute or relative to --run.")
    parser.add_argument("--output-dir", type=Path, help="Training output directory, absolute or relative to --run.")
    parser.add_argument("--experiment-name", help="Nerfstudio experiment name.")
    parser.add_argument("--max-steps", type=int, help="Training iterations.")
    parser.add_argument("--save-every", type=int, help="Checkpoint interval.")
    parser.add_argument("--eval-every", type=int, help="Eval interval.")
    parser.add_argument("--num-downscales", type=int, help="Nerfstudio downscale levels to generate.")
    parser.add_argument(
        "--max-dataset-size",
        type=int,
        help="Reserved frame-limit setting for backends that support it; splatfacto images data prep keeps all selected frames.",
    )
    parser.add_argument(
        "--skip-data-prepare",
        action="store_true",
        help="Require an existing Nerfstudio data dir with transforms.json.",
    )
    parser.add_argument("--overwrite-data", action="store_true", help="Replace generated Nerfstudio data first.")
    parser.add_argument(
        "--process-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional backend data-preparation option. Can be repeated.",
    )
    parser.add_argument(
        "--train-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional backend training option. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing outputs.")
    return parser.parse_args(argv)


def deep_merge(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def default_config(run_dir: Path) -> Dict[str, Any]:
    return {
        "training": {
            "backend": "splatfacto",
            "nerfstudio_dir": str(DEFAULT_NERFSTUDIO_DIR),
            "use_existing_colmap": True,
            "method": "splatfacto",
            "max_steps": 5000,
            "save_every": 1000,
            "eval_every": 1000,
            "num_downscales": 2,
            "max_dataset_size": -1,
            "frames_dir": "frames_selected",
            "data_dir": "nerfstudio",
            "colmap_model_dir": "colmap/sparse/0",
            "output_dir": "gsplat/outputs",
            "experiment_name": run_dir.name,
            "process_options": {},
            "train_options": {},
        },
        "colmap": {
            "binary": str(DEFAULT_COLMAP_BINARY),
            "mapper": "global_mapper",
            "use_nerfstudio_colmap": False,
        },
    }


def build_config(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    config = default_config(run_dir)
    config = deep_merge(config, load_config(args.config))
    training = config["training"]

    cli_overrides = {
        "backend": args.backend,
        "nerfstudio_dir": str(args.nerfstudio_dir) if args.nerfstudio_dir is not None else None,
        "method": args.method,
        "frames_dir": str(args.frames_dir) if args.frames_dir is not None else None,
        "data_dir": str(args.data_dir) if args.data_dir is not None else None,
        "colmap_model_dir": str(args.colmap_model_dir) if args.colmap_model_dir is not None else None,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "experiment_name": args.experiment_name,
        "max_steps": args.max_steps,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "num_downscales": args.num_downscales,
        "max_dataset_size": args.max_dataset_size,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            training[key] = value

    process_options = dict(training.get("process_options") or {})
    process_options.update(parse_key_value_pairs(args.process_option, "--process-option"))
    training["process_options"] = process_options

    train_options = dict(training.get("train_options") or {})
    train_options.update(parse_key_value_pairs(args.train_option, "--train-option"))
    training["train_options"] = train_options

    if args.pixi_bin is not None:
        training["pixi_bin"] = str(args.pixi_bin)
    if args.skip_data_prepare:
        training["skip_data_prepare"] = True
    return config


def validate_config(config: Mapping[str, Any], run_dir: Path, dry_run: bool) -> Dict[str, Any]:
    training = dict(config["training"])
    colmap = dict(config["colmap"])
    backend = str(training["backend"])
    validate_backend_name(backend)

    if not bool(training.get("use_existing_colmap", True)):
        raise SystemExit("training.use_existing_colmap must be true; Nerfstudio must not own reconstruction.")
    if bool(colmap.get("use_nerfstudio_colmap", False)):
        raise SystemExit("colmap.use_nerfstudio_colmap must be false; ignore Nerfstudio's COLMAP dependency.")

    colmap_binary = Path(str(colmap.get("binary", DEFAULT_COLMAP_BINARY))).expanduser()
    if not colmap_binary.is_absolute():
        raise SystemExit("colmap.binary must be an absolute path.")
    if not dry_run and not colmap_binary.exists():
        raise SystemExit(f"Authoritative COLMAP binary does not exist: {colmap_binary}")

    nerfstudio_dir = Path(str(training.get("nerfstudio_dir", DEFAULT_NERFSTUDIO_DIR))).expanduser()
    if not nerfstudio_dir.is_absolute():
        raise SystemExit("training.nerfstudio_dir must be an absolute path.")
    if not dry_run and not (nerfstudio_dir / "pixi.toml").exists():
        raise SystemExit(f"Nerfstudio Pixi manifest does not exist: {nerfstudio_dir / 'pixi.toml'}")

    pixi_bin = resolve_executable(
        Path(str(training["pixi_bin"])) if training.get("pixi_bin") else None,
        "pixi",
        dry_run,
    )

    frames_dir = resolve_under_run(run_dir, training["frames_dir"])
    data_dir = resolve_under_run(run_dir, training["data_dir"])
    colmap_model_dir = resolve_under_run(run_dir, training["colmap_model_dir"])
    output_dir = resolve_under_run(run_dir, training["output_dir"])

    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if not frames_dir.exists():
        raise SystemExit(f"Frames directory does not exist: {frames_dir}")
    if not dry_run and not colmap_model_dir.exists() and not training.get("skip_data_prepare"):
        raise SystemExit(
            f"COLMAP model directory does not exist: {colmap_model_dir}\n"
            "Run scripts/run_colmap.py first."
        )

    for key in ["max_steps", "save_every", "eval_every"]:
        if training.get(key) is not None and int(training[key]) <= 0:
            raise SystemExit(f"training.{key} must be greater than zero.")
    if int(training["num_downscales"]) < 0:
        raise SystemExit("training.num_downscales must be zero or greater.")
    max_dataset_size = int(training["max_dataset_size"])
    if max_dataset_size == 0 or max_dataset_size < -1:
        raise SystemExit("training.max_dataset_size must be -1 or greater than zero.")

    return {
        "backend": backend,
        "run_dir": run_dir,
        "python_bin": sys.executable,
        "method": training["method"],
        "pixi_bin": pixi_bin,
        "nerfstudio_dir": nerfstudio_dir,
        "frames_dir": frames_dir,
        "data_dir": data_dir,
        "colmap_model_dir": colmap_model_dir,
        "output_dir": output_dir,
        "experiment_name": training["experiment_name"],
        "max_steps": training["max_steps"],
        "save_every": training["save_every"],
        "eval_every": training["eval_every"],
        "num_downscales": training["num_downscales"],
        "max_dataset_size": training["max_dataset_size"],
        "process_options": training["process_options"],
        "train_options": training["train_options"],
        "colmap_binary": colmap_binary,
        "colmap_mapper": colmap.get("mapper"),
        "use_existing_colmap": True,
        "use_nerfstudio_colmap": False,
    }


def find_latest_training_config(output_dir: Path) -> Optional[Path]:
    return latest_existing_file(output_dir.glob("**/config.yml"))


def data_prepare_command_name(command: Sequence[str]) -> str:
    if any(Path(part).name == "prepare_nerfstudio_from_colmap.py" for part in command):
        return "prepare_nerfstudio_from_colmap.py"
    if "ns-process-data" in command:
        return "ns-process-data"
    return Path(command[0]).name if command else "unknown"


def print_dry_run(commands: Mapping[str, List[str]], settings: Mapping[str, Any], should_prepare_data: bool) -> None:
    print("Dry run. No files will be created or modified.\n")
    print(f"# backend: {settings['backend']}")
    print(f"# authoritative COLMAP: {settings['colmap_binary']}")
    if should_prepare_data:
        prepare_command = commands["prepare_nerfstudio_data"]
        if data_prepare_command_name(prepare_command) == "prepare_nerfstudio_from_colmap.py":
            print("# data prepare: custom COLMAP TXT -> Nerfstudio transforms exporter (multi-camera)")
        else:
            print("# data prepare: Nerfstudio ns-process-data images (single-camera/default)")
        print(f"$ {shlex.join(commands['prepare_nerfstudio_data'])}")
    else:
        print("# Nerfstudio data preparation skipped; transforms.json already exists.")
    print(f"$ {shlex.join(commands['train_splatfacto'])}")


def build_report(
    *,
    run_dir: Path,
    settings: Mapping[str, Any],
    status: str,
    dry_run: bool,
    started_at: str,
    finished_at: str,
    commands: Sequence[CommandResult],
    selected_config: Optional[Path],
    error: Optional[str],
) -> Dict[str, Any]:
    gsplat_dir = run_dir / "gsplat"
    return {
        "schema_version": 1,
        "created_at": finished_at,
        "status": status,
        "dry_run": dry_run,
        "backend": settings["backend"],
        "started_at": started_at,
        "finished_at": finished_at,
        "run": str(run_dir),
        "input": {
            "frames_dir": relative_to(settings["frames_dir"], run_dir),
            "colmap_model_dir": relative_to(settings["colmap_model_dir"], run_dir),
            "nerfstudio_data_dir": relative_to(settings["data_dir"], run_dir),
        },
        "settings": {
            "method": settings["method"],
            "nerfstudio_dir": str(settings["nerfstudio_dir"]),
            "use_existing_colmap": settings["use_existing_colmap"],
            "colmap_binary": str(settings["colmap_binary"]),
            "colmap_mapper": settings["colmap_mapper"],
            "use_nerfstudio_colmap": settings["use_nerfstudio_colmap"],
            "max_steps": settings["max_steps"],
            "save_every": settings["save_every"],
            "eval_every": settings["eval_every"],
            "num_downscales": settings["num_downscales"],
            "max_dataset_size": settings["max_dataset_size"],
            "process_options": settings["process_options"],
            "train_options": settings["train_options"],
            "data_prepare_command": settings.get("data_prepare_command"),
        },
        "outputs": {
            "gsplat": relative_to(gsplat_dir, run_dir),
            "outputs": relative_to(settings["output_dir"], run_dir),
            "logs": relative_to(gsplat_dir / "logs", run_dir),
            "training_report_json": relative_to(run_dir / "reports" / REPORT_NAME, run_dir),
        },
        "selected_config": relative_to(selected_config, run_dir),
        "environment": environment_report(
            {
                "pixi": str(settings["pixi_bin"]),
                "nerfstudio_dir": str(settings["nerfstudio_dir"]),
                "colmap": str(settings["colmap_binary"]),
            }
        ),
        "commands": command_results_to_json(commands),
        "error": error,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = args.run.expanduser()
    config = build_config(args, run_dir)
    settings = validate_config(config, run_dir, args.dry_run)

    transforms_path = settings["data_dir"] / "transforms.json"
    should_prepare_data = not args.skip_data_prepare and not transforms_path.exists()
    if args.overwrite_data:
        should_prepare_data = True

    commands = build_backend_commands(settings["backend"], settings)
    settings["data_prepare_command"] = data_prepare_command_name(commands["prepare_nerfstudio_data"])

    if args.dry_run:
        print_dry_run(commands, settings, should_prepare_data)
        return 0

    if args.skip_data_prepare and not transforms_path.exists():
        raise SystemExit(f"--skip-data-prepare requires an existing transforms.json at {transforms_path}")
    if args.overwrite_data and settings["data_dir"].exists():
        shutil.rmtree(settings["data_dir"])

    gsplat_dir = run_dir / "gsplat"
    logs_dir = gsplat_dir / "logs"
    reports_dir = run_dir / "reports"
    gsplat_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    settings["output_dir"].mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    command_results: List[CommandResult] = []
    status = "success"
    error: Optional[str] = None
    selected_config: Optional[Path] = None

    try:
        if should_prepare_data:
            command_results.append(
                run_logged_command("prepare_nerfstudio_data", commands["prepare_nerfstudio_data"], logs_dir, gsplat_dir)
            )
        else:
            print(f"Nerfstudio data already exists: {transforms_path}")
        command_results.append(run_logged_command("train_splatfacto", commands["train_splatfacto"], logs_dir, gsplat_dir))
        selected_config = find_latest_training_config(settings["output_dir"])
        if selected_config is None:
            raise RuntimeError(f"Training finished but no config.yml was found under {settings['output_dir']}.")
    except Exception as exc:
        status = "failed"
        error = str(exc)
        finished_at = utc_now()
        report = build_report(
            run_dir=run_dir,
            settings=settings,
            status=status,
            dry_run=False,
            started_at=started_at,
            finished_at=finished_at,
            commands=command_results,
            selected_config=selected_config,
            error=error,
        )
        report_path = reports_dir / REPORT_NAME
        write_json(report_path, report)
        raise SystemExit(f"{error}\nWrote failure report: {report_path}") from exc

    finished_at = utc_now()
    report = build_report(
        run_dir=run_dir,
        settings=settings,
        status=status,
        dry_run=False,
        started_at=started_at,
        finished_at=finished_at,
        commands=command_results,
        selected_config=selected_config,
        error=error,
    )
    report_path = reports_dir / REPORT_NAME
    write_json(report_path, report)
    print(f"\nTraining complete. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
