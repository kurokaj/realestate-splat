"""Nerfstudio / splatfacto backend command builder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

from realestate_splat.cli import option_map_to_cli


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


def build_process_data_command(settings: Mapping[str, Any]) -> List[str]:
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
        "--max-dataset-size",
        str(settings["max_dataset_size"]),
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
