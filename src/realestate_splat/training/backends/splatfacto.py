"""Nerfstudio / splatfacto backend command builder."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from realestate_splat.cli import option_map_to_cli


REPO_ROOT = Path(__file__).resolve().parents[4]


def colmap_model_path_for_process_data(data_dir: Path, colmap_model_dir: Path) -> str:
    try:
        return os.path.relpath(colmap_model_dir, start=data_dir)
    except ValueError:
        return str(colmap_model_dir)


def pixi_command(pixi_bin: str, nerfstudio_dir: Path, args: List[str]) -> List[str]:
    return [
        pixi_bin,
        "run",
        "--manifest-path",
        str(nerfstudio_dir / "pixi.toml"),
        *args,
    ]


def count_cameras_in_text_model(colmap_model_dir: Path) -> Optional[int]:
    candidate_dirs = [colmap_model_dir]
    for parent in [colmap_model_dir, *colmap_model_dir.parents]:
        if parent.name == "colmap":
            candidate_dirs.append(parent / "sparse_txt")
            break

    for candidate_dir in candidate_dirs:
        cameras_path = candidate_dir / "cameras.txt"
        if not cameras_path.exists():
            continue
        count = 0
        for raw_line in cameras_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                count += 1
        return count
    return None


def count_cameras_in_report(run_dir: Path) -> Optional[int]:
    report_path = run_dir / "reports" / "reconstruction_report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    metrics = report.get("reconstruction_metrics") or {}
    cameras = metrics.get("cameras")
    if cameras is None:
        return None
    try:
        return int(cameras)
    except (TypeError, ValueError):
        return None


def should_use_custom_multicamera_prepare(settings: Mapping[str, Any]) -> bool:
    data_dir = Path(str(settings["data_dir"]))
    colmap_model_dir = Path(str(settings["colmap_model_dir"]))
    run_dir = Path(str(settings.get("run_dir", data_dir.parent)))

    camera_count = count_cameras_in_text_model(colmap_model_dir)
    if camera_count is None:
        camera_count = count_cameras_in_report(run_dir)
    return camera_count is not None and camera_count > 1


def build_custom_multicamera_process_data_command(settings: Mapping[str, Any]) -> List[str]:
    script_path = REPO_ROOT / "scripts" / "prepare_nerfstudio_from_colmap.py"
    args = [
        str(settings.get("python_bin") or sys.executable),
        str(script_path),
        "--run",
        str(settings["run_dir"]),
        "--frames-dir",
        str(settings["frames_dir"]),
        "--data-dir",
        str(settings["data_dir"]),
        "--colmap-model-dir",
        str(settings["colmap_model_dir"]),
        "--num-downscales",
        str(int(settings["num_downscales"])),
        "--overwrite",
    ]
    return args


def build_process_data_command(settings: Mapping[str, Any]) -> List[str]:
    if should_use_custom_multicamera_prepare(settings):
        return build_custom_multicamera_process_data_command(settings)

    data_dir = Path(str(settings["data_dir"]))
    colmap_model_dir = Path(str(settings["colmap_model_dir"]))
    num_downscales = int(settings["num_downscales"])
    if num_downscales < 0:
        raise SystemExit("--num-downscales must be zero or greater.")

    args = [
        "ns-process-data",
        "images",
        "--data",
        str(settings["frames_dir"]),
        "--output-dir",
        str(data_dir),
        "--skip-colmap",
        "--colmap-model-path",
        colmap_model_path_for_process_data(data_dir, colmap_model_dir),
        "--num-downscales",
        str(num_downscales),
        "--colmap-cmd",
        str(settings["colmap_binary"]),
    ]
    args.extend(option_map_to_cli(settings.get("process_options", {})))
    return pixi_command(str(settings["pixi_bin"]), Path(str(settings["nerfstudio_dir"])), args)


def build_train_command(settings: Mapping[str, Any]) -> List[str]:
    args = [
        "ns-train",
        str(settings["method"]),
        f"--data={settings['data_dir']}",
        f"--output-dir={settings['output_dir']}",
        f"--experiment-name={settings['experiment_name']}",
        "--viewer.quit-on-train-completion=True",
    ]
    if settings.get("max_steps") is not None:
        args.append(f"--max-num-iterations={settings['max_steps']}")
    if settings.get("save_every") is not None:
        args.append(f"--steps-per-save={settings['save_every']}")
    if settings.get("eval_every") is not None:
        args.append(f"--steps-per-eval-batch={settings['eval_every']}")
        args.append(f"--steps-per-eval-image={settings['eval_every']}")
    args.extend(option_map_to_cli(settings.get("train_options", {})))
    return pixi_command(str(settings["pixi_bin"]), Path(str(settings["nerfstudio_dir"])), args)


def build_commands(settings: Mapping[str, Any]) -> Dict[str, List[str]]:
    return {
        "prepare_nerfstudio_data": build_process_data_command(settings),
        "train_splatfacto": build_train_command(settings),
    }
