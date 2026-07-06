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
import html
import json
import platform
import re
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
REPORT_HTML_NAME = "reconstruction_report.html"
IMAGE_MANIFEST_NAME = "image_manifest.json"

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
    settings["_single_camera_explicit"] = args.single_camera is not None

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


def load_image_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "reports" / IMAGE_MANIFEST_NAME
    if not manifest_path.exists():
        return {"schema_version": 1, "images": [], "camera_groups": []}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse image manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit(f"Image manifest root must be an object: {manifest_path}")
    images = manifest.get("images") or []
    if not isinstance(images, list):
        raise SystemExit(f"Image manifest 'images' must be a list: {manifest_path}")
    return manifest


def manifest_camera_group_id(entry: Mapping[str, Any]) -> str:
    explicit = entry.get("camera_group_id")
    if explicit:
        return str(explicit)
    return "{}_{}x{}".format(entry.get("camera_group"), entry.get("width"), entry.get("height"))


def should_use_manifest_camera_groups(manifest: Mapping[str, Any]) -> bool:
    images = manifest.get("images") or []
    group_ids = {manifest_camera_group_id(entry) for entry in images if isinstance(entry, dict)}
    return len(group_ids) > 1


def apply_manifest_camera_policy(settings: Dict[str, Any], manifest: Mapping[str, Any]) -> None:
    if should_use_manifest_camera_groups(manifest) and not settings.get("_single_camera_explicit"):
        settings["single_camera"] = False
        settings["camera_group_source"] = "image_manifest"


def manifest_images_by_name(manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in manifest.get("images") or []:
        if not isinstance(entry, dict):
            continue
        image_name = entry.get("image_name")
        if image_name:
            by_name[str(image_name)] = dict(entry)
    return by_name


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


def parse_model_analyzer_metrics(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {}

    patterns = {
        "rigs": (r"\bRigs:\s+(\d+)", int),
        "cameras": (r"\bCameras:\s+(\d+)", int),
        "frames": (r"\bFrames:\s+(\d+)", int),
        "registered_frames": (r"\bRegistered frames:\s+(\d+)", int),
        "images": (r"\bImages:\s+(\d+)", int),
        "registered_images": (r"\bRegistered images:\s+(\d+)", int),
        "points": (r"\bPoints:\s+(\d+)", int),
        "observations": (r"\bObservations:\s+(\d+)", int),
        "mean_track_length": (r"\bMean track length:\s+([0-9.eE+-]+)", float),
        "mean_observations_per_image": (r"\bMean observations per image:\s+([0-9.eE+-]+)", float),
        "mean_reprojection_error_px": (r"\bMean reprojection error:\s+([0-9.eE+-]+)px", float),
    }

    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics: Dict[str, Any] = {}
    for key, (pattern, caster) in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = caster(match.group(1))
    return metrics


def parse_registered_image_names(sparse_text_dir: Path) -> List[str]:
    images_txt = sparse_text_dir / "images.txt"
    if not images_txt.exists():
        return []
    names: List[str] = []
    data_line_index = 0
    for raw_line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if data_line_index % 2 == 0:
            parts = line.split()
            if len(parts) >= 10:
                names.append(parts[9])
        data_line_index += 1
    return names


def build_manifest_reconstruction_summary(manifest: Mapping[str, Any], sparse_text_dir: Path) -> Dict[str, Any]:
    images = [entry for entry in manifest.get("images") or [] if isinstance(entry, dict)]
    registered_names = set(parse_registered_image_names(sparse_text_dir))
    hero_images = [entry for entry in images if entry.get("role") == "hero"]
    hero_registered = [entry for entry in hero_images if entry.get("image_name") in registered_names]
    hero_dropped = [entry for entry in hero_images if entry.get("image_name") not in registered_names]
    return {
        "hero": {
            "total": len(hero_images),
            "registered": len(hero_registered),
            "dropped": len(hero_dropped),
            "registered_images": [entry.get("image_name") for entry in hero_registered],
            "dropped_images": [entry.get("image_name") for entry in hero_dropped],
        },
        "registered_image_count_from_text": len(registered_names),
    }


def build_camera_group_summary(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for entry in manifest.get("images") or []:
        if not isinstance(entry, dict):
            continue
        group_id = manifest_camera_group_id(entry)
        group = groups.setdefault(
            group_id,
            {
                "id": group_id,
                "camera_group": entry.get("camera_group"),
                "role": entry.get("role"),
                "width": entry.get("width"),
                "height": entry.get("height"),
                "image_count": 0,
                "hero": entry.get("role") == "hero",
                "locations": set(),
            },
        )
        group["image_count"] += 1
        if entry.get("role") == "hero":
            group["hero"] = True
        if entry.get("location"):
            group["locations"].add(entry["location"])

    output = []
    for group in groups.values():
        output.append(
            {
                "id": group["id"],
                "camera_group": group["camera_group"],
                "role": group["role"],
                "width": group["width"],
                "height": group["height"],
                "image_count": group["image_count"],
                "hero": group["hero"],
                "locations": sorted(group["locations"]),
            }
        )
    return sorted(output, key=lambda item: str(item["id"]))


def build_report(
    *,
    run_dir: Path,
    settings: Mapping[str, Any],
    paths: Mapping[str, Path],
    image_count: int,
    image_manifest: Mapping[str, Any],
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
    reconstruction_metrics = parse_model_analyzer_metrics(paths["logs_dir"] / "model_analyzer.log")
    manifest_summary = build_manifest_reconstruction_summary(image_manifest, paths["sparse_text_dir"])
    camera_groups = build_camera_group_summary(image_manifest)
    outputs = {
        "database": relative_to(paths["database_path"], run_dir),
        "database_global": relative_to(paths["database_global_path"], run_dir),
        "sparse": relative_to(paths["sparse_dir"], run_dir),
        "sparse_text": relative_to(paths["sparse_text_dir"], run_dir),
        "dense": relative_to(paths["dense_dir"], run_dir),
        "logs": relative_to(paths["logs_dir"], run_dir),
        "reconstruction_report_json": relative_to(paths["reports_dir"] / REPORT_NAME, run_dir),
        "reconstruction_report_html": relative_to(paths["reports_dir"] / REPORT_HTML_NAME, run_dir),
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
        "settings": {key: value for key, value in settings.items() if not str(key).startswith("_")},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "colmap_bin": colmap_bin,
            "colmap_help_header": colmap_version(colmap_bin, dry_run),
            "colmap_option_names": asdict(option_names) if option_names is not None else None,
        },
        "outputs": outputs,
        "reconstruction_metrics": reconstruction_metrics,
        "manifest_reconstruction": manifest_summary,
        "camera_groups": camera_groups,
        "selected_sparse_model": relative_to(selected_model, run_dir) if selected_model else None,
        "commands": [asdict(command) for command in commands],
        "error": error,
    }


def write_report(reports_dir: Path, report: Mapping[str, Any]) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / REPORT_NAME
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def table_rows(mapping: Mapping[str, Any]) -> str:
    rows = []
    for key, value in mapping.items():
        rows.append(
            "<tr><th>{key}</th><td>{value}</td></tr>".format(
                key=html.escape(str(key).replace("_", " ")),
                value=html.escape(str(value)),
            )
        )
    return "\n".join(rows)


def command_rows(commands: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for command in commands:
        rows.append(
            "<tr><td>{name}</td><td>{returncode}</td><td>{duration}</td><td><code>{log}</code></td></tr>".format(
                name=html.escape(str(command.get("name"))),
                returncode=html.escape(str(command.get("returncode"))),
                duration=html.escape(str(command.get("duration_seconds"))),
                log=html.escape(str(command.get("log_path"))),
            )
        )
    return "\n".join(rows)


def metric_rows(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return '<tr><td colspan="2">No model analyzer metrics found.</td></tr>'
    preferred_order = [
        "rigs",
        "cameras",
        "frames",
        "registered_frames",
        "images",
        "registered_images",
        "points",
        "observations",
        "mean_track_length",
        "mean_observations_per_image",
        "mean_reprojection_error_px",
    ]
    rows = []
    for key in preferred_order:
        if key not in metrics:
            continue
        rows.append(
            "<tr><th>{key}</th><td>{value}</td></tr>".format(
                key=html.escape(key.replace("_", " ")),
                value=html.escape(str(metrics[key])),
            )
        )
    return "\n".join(rows)


def hero_registration_rows(summary: Mapping[str, Any]) -> str:
    hero = summary.get("hero") or {}
    dropped = hero.get("dropped_images") or []
    return table_rows(
        {
            "hero_total": hero.get("total", 0),
            "hero_registered": hero.get("registered", 0),
            "hero_dropped": hero.get("dropped", 0),
            "dropped_images": ", ".join(str(item) for item in dropped) if dropped else "none",
        }
    )


def camera_group_rows(groups: Sequence[Mapping[str, Any]]) -> str:
    if not groups:
        return '<tr><td colspan="7">No image manifest camera groups found.</td></tr>'
    rows = []
    for group in groups:
        rows.append(
            "<tr><td>{id}</td><td>{camera_group}</td><td>{role}</td><td>{hero}</td><td>{resolution}</td><td>{count}</td><td>{locations}</td></tr>".format(
                id=html.escape(str(group.get("id"))),
                camera_group=html.escape(str(group.get("camera_group"))),
                role=html.escape(str(group.get("role"))),
                hero=html.escape(str(group.get("hero"))),
                resolution=html.escape("{}x{}".format(group.get("width"), group.get("height"))),
                count=html.escape(str(group.get("image_count"))),
                locations=html.escape(", ".join(str(item) for item in group.get("locations") or [])),
            )
        )
    return "\n".join(rows)


def write_html_report(reports_dir: Path, report: Mapping[str, Any]) -> Path:
    output_path = reports_dir / REPORT_HTML_NAME
    settings = report.get("settings", {})
    outputs = report.get("outputs", {})
    error = report.get("error")
    error_html = (
        "<p><strong>Error:</strong> {}</p>".format(html.escape(str(error)))
        if error
        else "<p>No reconstruction error recorded.</p>"
    )
    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reconstruction Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1c2430;
      --muted: #657080;
      --line: #d8dee7;
      --panel: #f6f8fb;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
      line-height: 1.45;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1, h2 {{
      line-height: 1.15;
      margin: 0 0 14px;
    }}
    h1 {{ font-size: 30px; }}
    h2 {{
      font-size: 20px;
      margin-top: 30px;
    }}
    .meta {{ color: var(--muted); }}
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
    code {{
      background: var(--panel);
      padding: 1px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
<main>
  <h1>Reconstruction Report</h1>
  <div class="meta">Generated {created_at}</div>

  <h2>Summary</h2>
  <table>{summary_rows}</table>

  <h2>Settings</h2>
  <table>{settings_rows}</table>

  <h2>Outputs</h2>
  <table>{outputs_rows}</table>

  <h2>Model Analyzer</h2>
  <table>{metrics_rows}</table>

  <h2>Hero Registration</h2>
  <table>{hero_rows}</table>

  <h2>Camera Groups</h2>
  <table>
    <tr><th>ID</th><th>Camera Group</th><th>Role</th><th>Hero</th><th>Resolution</th><th>Images</th><th>Locations</th></tr>
    {camera_group_rows}
  </table>

  <h2>Commands</h2>
  <table>
    <tr><th>Name</th><th>Exit Code</th><th>Duration (s)</th><th>Log</th></tr>
    {command_rows}
  </table>

  <h2>Status Detail</h2>
  {error_html}
</main>
</body>
</html>
""".format(
        created_at=html.escape(str(report.get("created_at"))),
        summary_rows=table_rows(
            {
                "status": report.get("status"),
                "run": report.get("run"),
                "image_dir": (report.get("input") or {}).get("image_dir"),
                "image_count": (report.get("input") or {}).get("image_count"),
                "selected_sparse_model": report.get("selected_sparse_model"),
                "started_at": report.get("started_at"),
                "finished_at": report.get("finished_at"),
            }
        ),
        settings_rows=table_rows(
            {
                "mode": settings.get("mode"),
                "matcher": settings.get("matcher"),
                "camera_model": settings.get("camera_model"),
                "single_camera": settings.get("single_camera"),
                "use_gpu": settings.get("use_gpu"),
                "max_image_size": settings.get("max_image_size"),
                "view_graph_calibrator": settings.get("view_graph_calibrator"),
            }
        ),
        outputs_rows=table_rows(outputs),
        metrics_rows=metric_rows(report.get("reconstruction_metrics") or {}),
        hero_rows=hero_registration_rows(report.get("manifest_reconstruction") or {}),
        camera_group_rows=camera_group_rows(report.get("camera_groups") or []),
        command_rows=command_rows(report.get("commands") or []),
        error_html=error_html,
    )
    output_path.write_text(document, encoding="utf-8")
    return output_path


def print_dry_run(
    commands: Sequence[Tuple[str, Sequence[str]]],
    paths: Mapping[str, Path],
    settings: Mapping[str, Any],
    image_manifest: Mapping[str, Any],
) -> None:
    print("Dry run. No files will be created or modified.\n")
    for name, command in commands:
        print(f"$ {shlex.join(command)}")
        if name == "feature_extractor" and should_use_manifest_camera_groups(image_manifest):
            groups = build_camera_group_summary(image_manifest)
            if settings.get("single_camera"):
                print("# manifest camera groups detected, but explicit --single-camera keeps one shared COLMAP camera.")
            else:
                print("# manifest camera groups detected; COLMAP runs with --ImageReader.single_camera 0")
                print("# COLMAP will keep rig/frame metadata valid and estimate separate camera intrinsics.")
            for group in groups:
                print(
                    "#   manifest group {id}: {count} images, role={role}, resolution={width}x{height}".format(
                        id=group.get("id"),
                        count=group.get("image_count"),
                        role=group.get("role"),
                        width=group.get("width"),
                        height=group.get("height"),
                    )
                )

    assumed_model = paths["sparse_dir"] / "0"
    for _name, command in build_followup_commands("colmap", settings, paths, assumed_model):
        shown = list(command)
        shown[0] = commands[0][1][0] if commands else "colmap"
        print(f"$ {shlex.join(shown)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    settings = build_settings(args)
    run_dir = args.run.expanduser()
    image_manifest = load_image_manifest(run_dir)
    apply_manifest_camera_policy(settings, image_manifest)
    paths = prepare_output_paths(run_dir, settings, args.overwrite, args.dry_run)
    image_count = count_images(paths["image_dir"])
    if image_count == 0:
        raise SystemExit(f"No supported images found in {paths['image_dir']}")

    colmap_bin = resolve_colmap_bin(settings, args.dry_run)
    option_names = resolve_colmap_option_names(colmap_bin, settings, args.dry_run)
    core_commands = build_core_commands(colmap_bin, settings, paths, option_names)

    if args.dry_run:
        print_dry_run(core_commands, paths, settings, image_manifest)
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
            image_manifest=image_manifest,
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
        html_report_path = write_html_report(paths["reports_dir"], report)
        raise SystemExit(f"{error}\nWrote failure report: {report_path}\nHTML report: {html_report_path}") from exc

    finished_at = utc_now()
    report = build_report(
        run_dir=run_dir,
        settings=settings,
        paths=paths,
        image_count=image_count,
        image_manifest=image_manifest,
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
    html_report_path = write_html_report(paths["reports_dir"], report)
    print(f"\nCOLMAP reconstruction complete. Report: {report_path}")
    print(f"HTML report: {html_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
