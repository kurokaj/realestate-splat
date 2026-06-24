#!/usr/bin/env python3
"""Run COLMAP reconstruction for a preprocessed real estate splat run.

This script is intended to run on the Verda GPU instance after the selected
frames have been uploaded to the run directory. It follows the run directory
contract documented in ``docs/realestate_splat_project_plan.md``:

    runs/<scene>/
      frames_selected/
      colmap/
        database.db
        sparse/
        logs/
      reports/
        reconstruction_report.json

The implementation uses COLMAP's CLI directly instead of Python bindings so the
same script can be called manually over SSH now and by orchestration later.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_VERDA_COLMAP = Path("/workspace/opt/colmap-install/bin/colmap")
REPORT_NAME = "reconstruction_report.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "binary": str(DEFAULT_VERDA_COLMAP),
    "mode": "incremental",
    "matcher": "exhaustive",
    "image_dir": "frames_selected",
    "database_name": "database.db",
    "camera_model": "SIMPLE_RADIAL",
    "single_camera": True,
    "use_gpu": True,
    "max_image_size": 3200,
    "option_namespace": "auto",
    "view_graph_calibrator": True,
    "sequential_overlap": 10,
    "export_text": True,
    "undistort": False,
    "feature_options": {},
    "matcher_options": {},
    "view_graph_calibrator_options": {},
    "mapper_options": {},
}


@dataclass
class CommandResult:
    name: str
    command: List[str]
    log_path: str
    returncode: int
    started_at: str
    finished_at: str
    duration_seconds: float


@dataclass(frozen=True)
class ColmapOptionNames:
    feature_use_gpu: str
    feature_max_image_size: str
    matching_use_gpu: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run COLMAP for a preprocessed Buildvision3D run directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Run directory containing frames_selected/ and reports/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON/YAML config. If it has a top-level 'colmap' key, that section is used.",
    )
    parser.add_argument(
        "--mode",
        choices=["incremental", "global"],
        help="COLMAP reconstruction mode.",
    )
    parser.add_argument(
        "--matcher",
        choices=["exhaustive", "sequential", "vocab_tree"],
        help="COLMAP matcher to run before mapping.",
    )
    parser.add_argument(
        "--colmap-bin",
        type=Path,
        help=f"Absolute path to the authoritative COLMAP executable. Defaults to {DEFAULT_VERDA_COLMAP}.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Image directory, absolute or relative to --run.",
    )
    parser.add_argument("--database-name", help="Database filename under run/colmap/.")
    parser.add_argument("--camera-model", help="COLMAP camera model for ImageReader.")
    parser.add_argument(
        "--single-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Treat all frames as coming from one camera.",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable COLMAP GPU feature extraction and matching.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        help="Maximum image dimension used by SIFT extraction.",
    )
    parser.add_argument(
        "--option-namespace",
        choices=["auto", "feature", "sift"],
        help=(
            "COLMAP option namespace for extraction/matching GPU flags. "
            "'feature' uses FeatureExtraction/FeatureMatching; 'sift' uses older SiftExtraction/SiftMatching."
        ),
    )
    parser.add_argument(
        "--view-graph-calibrator",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For global_mapper, copy the database and run view_graph_calibrator before mapping.",
    )
    parser.add_argument(
        "--sequential-overlap",
        type=int,
        help="Sequential matcher overlap when --matcher sequential is used.",
    )
    parser.add_argument(
        "--vocab-tree",
        type=Path,
        help="Vocabulary tree path when --matcher vocab_tree is used.",
    )
    parser.add_argument(
        "--export-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export the selected sparse model as TXT for easier inspection.",
    )
    parser.add_argument(
        "--undistort",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also run image_undistorter into run/colmap/dense/.",
    )
    parser.add_argument(
        "--feature-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional COLMAP feature_extractor option. Can be repeated.",
    )
    parser.add_argument(
        "--matcher-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional COLMAP matcher option. Can be repeated.",
    )
    parser.add_argument(
        "--view-graph-calibrator-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional COLMAP view_graph_calibrator option. Can be repeated.",
    )
    parser.add_argument(
        "--mapper-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional COLMAP mapper/global_mapper option. Can be repeated.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated COLMAP outputs in the run directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without creating or modifying COLMAP outputs.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Config file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        loaded = load_yaml_like(text, path)
    else:
        raise SystemExit(f"Unsupported config extension for {path}; use .json, .yaml, or .yml.")

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config root must be a mapping: {path}")

    colmap_section = loaded.get("colmap", loaded)
    if not isinstance(colmap_section, dict):
        raise SystemExit(f"Config 'colmap' section must be a mapping: {path}")
    return normalize_keys(colmap_section)


def load_yaml_like(text: str, path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return parse_simple_yaml(text, path)

    loaded = yaml.safe_load(text)
    return loaded or {}


def parse_simple_yaml(text: str, path: Path) -> Dict[str, Any]:
    """Parse the small mapping-only YAML shape used by the planned configs.

    This intentionally supports only nested dictionaries and scalar values. If
    future configs need lists or advanced YAML features, installing PyYAML on the
    Verda environment is the right move.
    """

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if "\t" in raw_line:
            raise SystemExit(f"{path}:{line_number}: tabs are not supported by the fallback YAML parser.")
        if ":" not in line:
            raise SystemExit(f"{path}:{line_number}: expected KEY: VALUE.")

        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        key = normalize_key(key.strip())
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise SystemExit(f"{path}:{line_number}: invalid indentation.")

        parent = stack[-1][1]
        if not value:
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)

    return root


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def normalize_key(key: str) -> str:
    return key.replace("-", "_")


def normalize_keys(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in mapping.items():
        new_key = normalize_key(str(key))
        if isinstance(value, dict):
            normalized[new_key] = normalize_keys(value)
        else:
            normalized[new_key] = value
    return normalized


def parse_option_pairs(raw_pairs: Sequence[str], option_name: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for raw_pair in raw_pairs:
        if "=" not in raw_pair:
            raise SystemExit(f"{option_name} expects KEY=VALUE, got: {raw_pair}")
        key, value = raw_pair.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"{option_name} has an empty key: {raw_pair}")
        parsed[key] = parse_scalar(value.strip())
    return parsed


def build_settings(args: argparse.Namespace) -> Dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    settings.update(load_config(args.config))
    normalize_mapper_setting(settings)

    cli_overrides = {
        "binary": str(args.colmap_bin) if args.colmap_bin is not None else None,
        "mode": args.mode,
        "matcher": args.matcher,
        "image_dir": str(args.image_dir) if args.image_dir is not None else None,
        "database_name": args.database_name,
        "camera_model": args.camera_model,
        "single_camera": args.single_camera,
        "use_gpu": args.use_gpu,
        "max_image_size": args.max_image_size,
        "option_namespace": args.option_namespace,
        "view_graph_calibrator": args.view_graph_calibrator,
        "sequential_overlap": args.sequential_overlap,
        "vocab_tree": str(args.vocab_tree) if args.vocab_tree is not None else None,
        "export_text": args.export_text,
        "undistort": args.undistort,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            settings[key] = value

    merge_option_map(settings, "feature_options", parse_option_pairs(args.feature_option, "--feature-option"))
    merge_option_map(settings, "matcher_options", parse_option_pairs(args.matcher_option, "--matcher-option"))
    merge_option_map(
        settings,
        "view_graph_calibrator_options",
        parse_option_pairs(args.view_graph_calibrator_option, "--view-graph-calibrator-option"),
    )
    merge_option_map(settings, "mapper_options", parse_option_pairs(args.mapper_option, "--mapper-option"))

    validate_settings(settings)
    return settings


def normalize_mapper_setting(settings: Dict[str, Any]) -> None:
    mapper = settings.get("mapper")
    if mapper is None:
        return

    mapper_name = str(mapper)
    if mapper_name == "global_mapper":
        settings["mode"] = "global"
    elif mapper_name in {"mapper", "incremental_mapper", "incremental"}:
        settings["mode"] = "incremental"
    else:
        raise SystemExit("colmap.mapper must be global_mapper, mapper, incremental_mapper, or incremental.")


def merge_option_map(settings: Dict[str, Any], key: str, additions: Dict[str, Any]) -> None:
    current = settings.get(key) or {}
    if not isinstance(current, dict):
        raise SystemExit(f"{key} must be a mapping.")
    merged = dict(current)
    merged.update(additions)
    settings[key] = merged


def validate_settings(settings: Mapping[str, Any]) -> None:
    binary = Path(str(settings["binary"])).expanduser()
    if not binary.is_absolute():
        raise SystemExit("colmap.binary / --colmap-bin must be an absolute path; do not rely on PATH.")
    if settings.get("use_nerfstudio_colmap"):
        raise SystemExit("colmap.use_nerfstudio_colmap must be false; run_colmap.py owns reconstruction.")
    if settings["mode"] not in {"incremental", "global"}:
        raise SystemExit("--mode must be incremental or global.")
    if settings["matcher"] not in {"exhaustive", "sequential", "vocab_tree"}:
        raise SystemExit("--matcher must be exhaustive, sequential, or vocab_tree.")
    if settings["option_namespace"] not in {"auto", "feature", "sift"}:
        raise SystemExit("--option-namespace must be auto, feature, or sift.")
    if int(settings["max_image_size"]) <= 0:
        raise SystemExit("--max-image-size must be greater than zero.")
    if int(settings["sequential_overlap"]) <= 0:
        raise SystemExit("--sequential-overlap must be greater than zero.")
    if settings["matcher"] == "vocab_tree" and not settings.get("vocab_tree"):
        raise SystemExit("--vocab-tree is required when --matcher vocab_tree is used.")


def should_run_view_graph_calibrator(settings: Mapping[str, Any]) -> bool:
    return settings["mode"] == "global" and bool(settings.get("view_graph_calibrator"))


def resolve_colmap_bin(settings: Mapping[str, Any], dry_run: bool) -> str:
    binary = Path(str(settings["binary"])).expanduser()
    if dry_run:
        return str(binary)
    if binary.exists():
        return str(binary)

    raise SystemExit(
        f"Could not find the authoritative COLMAP binary at {binary}. "
        f"On Verda it should be {DEFAULT_VERDA_COLMAP}. Do not rely on PATH or Nerfstudio's COLMAP."
    )


def resolve_under_run(run_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return run_dir / path


def relative_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def bool_as_colmap(value: Any) -> str:
    return "1" if bool(value) else "0"


def append_options(command: List[str], options: Mapping[str, Any]) -> None:
    for key, value in sorted(options.items()):
        if value is None:
            continue
        option_name = key if str(key).startswith("--") else f"--{key}"
        command.extend([option_name, str(value)])


def forced_option_names(namespace: str) -> ColmapOptionNames:
    if namespace == "feature":
        return ColmapOptionNames(
            feature_use_gpu="--FeatureExtraction.use_gpu",
            feature_max_image_size="--FeatureExtraction.max_image_size",
            matching_use_gpu="--FeatureMatching.use_gpu",
        )
    if namespace == "sift":
        return ColmapOptionNames(
            feature_use_gpu="--SiftExtraction.use_gpu",
            feature_max_image_size="--SiftExtraction.max_image_size",
            matching_use_gpu="--SiftMatching.use_gpu",
        )
    raise ValueError(f"Unsupported forced COLMAP option namespace: {namespace}")


def colmap_command_help(colmap_bin: str, command: str) -> str:
    completed = subprocess.run(
        [colmap_bin, command, "-h"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    output = completed.stdout or ""
    if completed.returncode != 0:
        raise SystemExit(
            f"Could not query COLMAP help for '{command}' with {colmap_bin}.\n"
            f"Exit code: {completed.returncode}\n"
            f"Output:\n{output}"
        )
    return output


def option_supported(help_text: str, option_name: str) -> bool:
    return option_name in help_text


def resolve_colmap_option_names(colmap_bin: str, settings: Mapping[str, Any], dry_run: bool) -> ColmapOptionNames:
    namespace = str(settings["option_namespace"])
    if namespace != "auto":
        return forced_option_names(namespace)
    if dry_run:
        return forced_option_names("feature")

    feature_help = colmap_command_help(colmap_bin, "feature_extractor")
    if option_supported(feature_help, "--FeatureExtraction.use_gpu"):
        feature_use_gpu = "--FeatureExtraction.use_gpu"
        feature_max_image_size = "--FeatureExtraction.max_image_size"
    elif option_supported(feature_help, "--SiftExtraction.use_gpu"):
        feature_use_gpu = "--SiftExtraction.use_gpu"
        feature_max_image_size = "--SiftExtraction.max_image_size"
    else:
        raise SystemExit(
            "Could not find a supported feature extraction GPU option in COLMAP help. "
            "Expected --FeatureExtraction.use_gpu or --SiftExtraction.use_gpu."
        )

    if not option_supported(feature_help, feature_max_image_size):
        raise SystemExit(f"COLMAP help did not contain expected max image size option: {feature_max_image_size}")

    matcher_help = colmap_command_help(colmap_bin, f"{settings['matcher']}_matcher")
    if option_supported(matcher_help, "--FeatureMatching.use_gpu"):
        matching_use_gpu = "--FeatureMatching.use_gpu"
    elif option_supported(matcher_help, "--SiftMatching.use_gpu"):
        matching_use_gpu = "--SiftMatching.use_gpu"
    else:
        raise SystemExit(
            "Could not find a supported feature matching GPU option in COLMAP help. "
            "Expected --FeatureMatching.use_gpu or --SiftMatching.use_gpu."
        )

    return ColmapOptionNames(
        feature_use_gpu=feature_use_gpu,
        feature_max_image_size=feature_max_image_size,
        matching_use_gpu=matching_use_gpu,
    )


def prepare_output_paths(run_dir: Path, settings: Mapping[str, Any], overwrite: bool, dry_run: bool) -> Dict[str, Path]:
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise SystemExit(f"Run path is not a directory: {run_dir}")

    image_dir = resolve_under_run(run_dir, settings["image_dir"])
    if not image_dir.exists():
        raise SystemExit(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise SystemExit(f"Image path is not a directory: {image_dir}")

    colmap_dir = run_dir / "colmap"
    database_path = colmap_dir / str(settings["database_name"])
    database_global_path = colmap_dir / "database_global.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_text_dir = colmap_dir / "sparse_txt"
    dense_dir = colmap_dir / "dense"
    logs_dir = colmap_dir / "logs"
    reports_dir = run_dir / "reports"

    if not dry_run:
        existing_outputs = [
            path
            for path in [database_path, database_global_path, sparse_dir, sparse_text_dir, dense_dir]
            if path.exists()
        ]
        if existing_outputs and not overwrite:
            formatted = "\n  ".join(str(path) for path in existing_outputs)
            raise SystemExit(
                "COLMAP outputs already exist. Use --overwrite to replace generated outputs:\n"
                f"  {formatted}"
            )
        if overwrite:
            for path in existing_outputs:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

        colmap_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "image_dir": image_dir,
        "colmap_dir": colmap_dir,
        "database_path": database_path,
        "database_global_path": database_global_path,
        "sparse_dir": sparse_dir,
        "sparse_text_dir": sparse_text_dir,
        "dense_dir": dense_dir,
        "logs_dir": logs_dir,
        "reports_dir": reports_dir,
    }


def count_images(image_dir: Path) -> int:
    return sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def build_core_commands(
    colmap_bin: str,
    settings: Mapping[str, Any],
    paths: Mapping[str, Path],
    option_names: ColmapOptionNames,
) -> List[Tuple[str, List[str]]]:
    database_path = str(paths["database_path"])
    mapper_database_path = str(paths["database_global_path"] if should_run_view_graph_calibrator(settings) else paths["database_path"])
    image_dir = str(paths["image_dir"])
    sparse_dir = str(paths["sparse_dir"])

    feature_command = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        database_path,
        "--image_path",
        image_dir,
        "--ImageReader.camera_model",
        str(settings["camera_model"]),
        "--ImageReader.single_camera",
        bool_as_colmap(settings["single_camera"]),
        option_names.feature_use_gpu,
        bool_as_colmap(settings["use_gpu"]),
        option_names.feature_max_image_size,
        str(int(settings["max_image_size"])),
    ]
    append_options(feature_command, settings.get("feature_options", {}))

    matcher_command = build_matcher_command(colmap_bin, settings, database_path, option_names)

    commands: List[Tuple[str, List[str]]] = [
        ("feature_extractor", feature_command),
        (f"{settings['matcher']}_matcher", matcher_command),
    ]

    if should_run_view_graph_calibrator(settings):
        commands.append(
            (
                "copy_database_for_global_mapper",
                [
                    "cp",
                    database_path,
                    mapper_database_path,
                ],
            )
        )
        view_graph_calibrator_command = [
            colmap_bin,
            "view_graph_calibrator",
            "--database_path",
            mapper_database_path,
        ]
        append_options(view_graph_calibrator_command, settings.get("view_graph_calibrator_options", {}))
        commands.append(("view_graph_calibrator", view_graph_calibrator_command))

    mapper_name = "global_mapper" if settings["mode"] == "global" else "mapper"
    mapper_command = [
        colmap_bin,
        mapper_name,
        "--database_path",
        mapper_database_path,
        "--image_path",
        image_dir,
        "--output_path",
        sparse_dir,
    ]
    append_options(mapper_command, settings.get("mapper_options", {}))

    commands.append((mapper_name, mapper_command))
    return commands


def build_matcher_command(
    colmap_bin: str,
    settings: Mapping[str, Any],
    database_path: str,
    option_names: ColmapOptionNames,
) -> List[str]:
    matcher = str(settings["matcher"])
    command = [
        colmap_bin,
        f"{matcher}_matcher",
        "--database_path",
        database_path,
        option_names.matching_use_gpu,
        bool_as_colmap(settings["use_gpu"]),
    ]
    if matcher == "sequential":
        command.extend(["--SequentialMatching.overlap", str(int(settings["sequential_overlap"]))])
    if matcher == "vocab_tree":
        command.extend(["--VocabTreeMatching.vocab_tree_path", str(settings["vocab_tree"])])
    append_options(command, settings.get("matcher_options", {}))
    return command


def run_command(name: str, command: Sequence[str], logs_dir: Path) -> CommandResult:
    log_path = logs_dir / f"{name}.log"
    started_at = utc_now()
    start_time = time.monotonic()

    print(f"\n$ {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        returncode = process.wait()

    finished_at = utc_now()
    duration = round(time.monotonic() - start_time, 3)
    result = CommandResult(
        name=name,
        command=list(command),
        log_path=str(log_path),
        returncode=returncode,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
    )
    if returncode != 0:
        raise RuntimeError(f"COLMAP command failed ({name}) with exit code {returncode}. See {log_path}")
    return result


def find_sparse_model(sparse_dir: Path) -> Optional[Path]:
    if not sparse_dir.exists():
        return None
    model_dirs = sorted(path for path in sparse_dir.iterdir() if path.is_dir())
    if not model_dirs:
        return None
    preferred = sparse_dir / "0"
    if preferred in model_dirs:
        return preferred
    return model_dirs[0]


def build_followup_commands(
    colmap_bin: str,
    settings: Mapping[str, Any],
    paths: Mapping[str, Path],
    model_dir: Path,
) -> List[Tuple[str, List[str]]]:
    commands: List[Tuple[str, List[str]]] = []
    if settings.get("export_text"):
        commands.append(
            (
                "model_converter",
                [
                    colmap_bin,
                    "model_converter",
                    "--input_path",
                    str(model_dir),
                    "--output_path",
                    str(paths["sparse_text_dir"]),
                    "--output_type",
                    "TXT",
                ],
            )
        )
    commands.append(
        (
            "model_analyzer",
            [
                colmap_bin,
                "model_analyzer",
                "--path",
                str(model_dir),
            ],
        )
    )
    if settings.get("undistort"):
        commands.append(
            (
                "image_undistorter",
                [
                    colmap_bin,
                    "image_undistorter",
                    "--image_path",
                    str(paths["image_dir"]),
                    "--input_path",
                    str(model_dir),
                    "--output_path",
                    str(paths["dense_dir"]),
                    "--output_type",
                    "COLMAP",
                ],
            )
        )
    return commands


def colmap_version(colmap_bin: str, dry_run: bool) -> Optional[str]:
    if dry_run:
        return None
    try:
        completed = subprocess.run(
            [colmap_bin, "-h"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except Exception:
        return None
    first_line = (completed.stdout or "").splitlines()
    return first_line[0] if first_line else None


def build_report(
    *,
    run_dir: Path,
    settings: Mapping[str, Any],
    paths: Mapping[str, Path],
    image_count: int,
    colmap_bin: str,
    option_names: Optional[ColmapOptionNames],
    commands: Sequence[CommandResult],
    status: str,
    started_at: str,
    finished_at: str,
    selected_model: Optional[Path],
    error: Optional[str],
    dry_run: bool,
) -> Dict[str, Any]:
    outputs = {
        "database": relative_to(paths["database_path"], run_dir),
        "database_global": relative_to(paths["database_global_path"], run_dir),
        "sparse": relative_to(paths["sparse_dir"], run_dir),
        "sparse_text": relative_to(paths["sparse_text_dir"], run_dir),
        "dense": relative_to(paths["dense_dir"], run_dir),
        "logs": relative_to(paths["logs_dir"], run_dir),
        "reconstruction_report_json": relative_to(paths["reports_dir"] / REPORT_NAME, run_dir),
    }
    return {
        "schema_version": 1,
        "created_at": finished_at,
        "status": status,
        "dry_run": dry_run,
        "started_at": started_at,
        "finished_at": finished_at,
        "run": str(run_dir),
        "input": {
            "image_dir": relative_to(paths["image_dir"], run_dir),
            "image_count": image_count,
        },
        "settings": dict(settings),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "colmap_bin": colmap_bin,
            "colmap_help_header": colmap_version(colmap_bin, dry_run),
            "colmap_option_names": asdict(option_names) if option_names is not None else None,
        },
        "outputs": outputs,
        "selected_sparse_model": relative_to(selected_model, run_dir) if selected_model else None,
        "commands": [asdict(command) for command in commands],
        "error": error,
    }


def write_report(reports_dir: Path, report: Mapping[str, Any]) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / REPORT_NAME
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def print_dry_run(commands: Sequence[Tuple[str, Sequence[str]]], paths: Mapping[str, Path], settings: Mapping[str, Any]) -> None:
    print("Dry run. No files will be created or modified.\n")
    for _name, command in commands:
        print(f"$ {shlex.join(command)}")

    assumed_model = paths["sparse_dir"] / "0"
    for _name, command in build_followup_commands("colmap", settings, paths, assumed_model):
        shown = list(command)
        shown[0] = commands[0][1][0] if commands else "colmap"
        print(f"$ {shlex.join(shown)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    settings = build_settings(args)
    run_dir = args.run.expanduser()
    paths = prepare_output_paths(run_dir, settings, args.overwrite, args.dry_run)
    image_count = count_images(paths["image_dir"])
    if image_count == 0:
        raise SystemExit(f"No supported images found in {paths['image_dir']}")

    colmap_bin = resolve_colmap_bin(settings, args.dry_run)
    option_names = resolve_colmap_option_names(colmap_bin, settings, args.dry_run)
    core_commands = build_core_commands(colmap_bin, settings, paths, option_names)

    if args.dry_run:
        print_dry_run(core_commands, paths, settings)
        return 0

    started_at = utc_now()
    command_results: List[CommandResult] = []
    selected_model: Optional[Path] = None
    status = "success"
    error: Optional[str] = None

    try:
        for name, command in core_commands:
            command_results.append(run_command(name, command, paths["logs_dir"]))

        selected_model = find_sparse_model(paths["sparse_dir"])
        if selected_model is None:
            raise RuntimeError(f"COLMAP did not produce a sparse model under {paths['sparse_dir']}.")

        if settings.get("export_text"):
            paths["sparse_text_dir"].mkdir(parents=True, exist_ok=True)

        for name, command in build_followup_commands(colmap_bin, settings, paths, selected_model):
            command_results.append(run_command(name, command, paths["logs_dir"]))
    except Exception as exc:
        status = "failed"
        error = str(exc)
        finished_at = utc_now()
        report = build_report(
            run_dir=run_dir,
            settings=settings,
            paths=paths,
            image_count=image_count,
            colmap_bin=colmap_bin,
            option_names=option_names,
            commands=command_results,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            selected_model=selected_model,
            error=error,
            dry_run=False,
        )
        report_path = write_report(paths["reports_dir"], report)
        raise SystemExit(f"{error}\nWrote failure report: {report_path}") from exc

    finished_at = utc_now()
    report = build_report(
        run_dir=run_dir,
        settings=settings,
        paths=paths,
        image_count=image_count,
        colmap_bin=colmap_bin,
        option_names=option_names,
        commands=command_results,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        selected_model=selected_model,
        error=error,
        dry_run=False,
    )
    report_path = write_report(paths["reports_dir"], report)
    print(f"\nCOLMAP reconstruction complete. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
