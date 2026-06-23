#!/usr/bin/env python3
"""Export final artifacts from a trained splat run.

The first canonical artifact is ``final/scene.ply``. Viewer-specific exports
such as .splat can be added once the browser viewer target is fixed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.cli import (  # noqa: E402
    CommandResult,
    command_results_to_json,
    environment_report,
    latest_existing_file,
    relative_to,
    resolve_executable,
    resolve_under_run,
    run_logged_command,
    utc_now,
    write_json,
)
from realestate_splat.config import load_config  # noqa: E402


DEFAULT_NERFSTUDIO_DIR = Path("/workspace/opt/nerfstudio")


REPORT_NAME = "export_report.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export final scene artifacts from a trained Buildvision3D run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", required=True, type=Path, help="Run directory.")
    parser.add_argument("--config", type=Path, help="Optional JSON/YAML pipeline config.")
    parser.add_argument(
        "--backend",
        choices=["splatfacto", "raw_gsplat"],
        default="splatfacto",
        help="Training/export backend. raw_gsplat is reserved for future work.",
    )
    parser.add_argument(
        "--load-config",
        type=Path,
        help="Nerfstudio config.yml to export. Defaults to training_report.json, then latest gsplat output.",
    )
    parser.add_argument("--nerfstudio-dir", type=Path, help="Pixi-managed Nerfstudio source directory.")
    parser.add_argument("--pixi-bin", type=Path, help="Path to pixi.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Intermediate export directory, absolute or relative to --run.",
    )
    parser.add_argument(
        "--output-ply",
        type=Path,
        help="Canonical output PLY path, absolute or relative to --run.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing final/scene.ply.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing outputs.")
    return parser.parse_args(argv)


def validate_backend(backend: str) -> None:
    if backend == "raw_gsplat":
        raise SystemExit(
            "backend=raw_gsplat is reserved for a later custom exporter. "
            "Use backend=splatfacto for milestone 2."
        )


def resolve_nerfstudio_dir(run_config: Dict[str, Any], cli_value: Optional[Path], dry_run: bool) -> Path:
    if cli_value is not None:
        nerfstudio_dir = cli_value.expanduser()
    else:
        training_config = run_config.get("training") or {}
        nerfstudio_dir = Path(str(training_config.get("nerfstudio_dir", DEFAULT_NERFSTUDIO_DIR))).expanduser()
    if not nerfstudio_dir.is_absolute():
        raise SystemExit("training.nerfstudio_dir / --nerfstudio-dir must be an absolute path.")
    if not dry_run and not (nerfstudio_dir / "pixi.toml").exists():
        raise SystemExit(f"Nerfstudio Pixi manifest does not exist: {nerfstudio_dir / 'pixi.toml'}")
    return nerfstudio_dir


def read_training_report_config(run_dir: Path) -> Optional[Path]:
    report_path = run_dir / "reports" / "training_report.json"
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    selected = payload.get("selected_config")
    if not selected:
        return None
    return resolve_under_run(run_dir, selected)


def find_latest_training_config(run_dir: Path) -> Optional[Path]:
    from_report = read_training_report_config(run_dir)
    if from_report is not None and from_report.exists():
        return from_report
    return latest_existing_file((run_dir / "gsplat").glob("outputs/**/config.yml"))


def resolve_load_config(run_dir: Path, cli_value: Optional[Path], dry_run: bool) -> Path:
    if cli_value is not None:
        path = resolve_under_run(run_dir, cli_value)
    else:
        found = find_latest_training_config(run_dir)
        if found is None:
            if dry_run:
                return run_dir / "gsplat" / "outputs" / "<experiment>" / "splatfacto" / "<timestamp>" / "config.yml"
            raise SystemExit(
                "Could not find a Nerfstudio config.yml. Run scripts/run_training.py first or pass --load-config."
            )
        path = found
    if not path.exists() and not dry_run:
        raise SystemExit(f"Nerfstudio config.yml does not exist: {path}")
    return path


def build_export_command(pixi_bin: str, nerfstudio_dir: Path, load_config: Path, export_dir: Path) -> List[str]:
    return [
        pixi_bin,
        "--manifest-path",
        str(nerfstudio_dir / "pixi.toml"),
        "run",
        "ns-export",
        "gaussian-splat",
        "--load-config",
        str(load_config),
        "--output-dir",
        str(export_dir),
    ]


def find_exported_ply(export_dir: Path) -> Optional[Path]:
    candidates = [path for path in export_dir.glob("**/*.ply") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_size, path.stat().st_mtime))


def write_viewer_config(run_dir: Path, final_dir: Path, output_ply: Path) -> Path:
    payload = {
        "schema_version": 1,
        "scene": {
            "format": "gaussian_splat_ply",
            "path": relative_to(output_ply, final_dir),
            "canonical": True,
        },
        "notes": [
            "PLY is the first canonical milestone-2 export.",
            "Viewer-specific .splat or SuperSplat/PlayCanvas output should be added after the browser viewer is chosen.",
        ],
    }
    viewer_config_path = final_dir / "viewer_config.json"
    write_json(viewer_config_path, payload)
    return viewer_config_path


def print_dry_run(command: Sequence[str], output_ply: Path) -> None:
    print("Dry run. No files will be created or modified.\n")
    print(f"$ {shlex.join(command)}")
    print(f"# copy exported .ply to {output_ply}")


def build_report(
    *,
    run_dir: Path,
    backend: str,
    status: str,
    dry_run: bool,
    started_at: str,
    finished_at: str,
    pixi_bin: str,
    nerfstudio_dir: Path,
    load_config: Path,
    export_dir: Path,
    output_ply: Path,
    source_ply: Optional[Path],
    viewer_config: Optional[Path],
    commands: Sequence[CommandResult],
    error: Optional[str],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": finished_at,
        "status": status,
        "dry_run": dry_run,
        "backend": backend,
        "started_at": started_at,
        "finished_at": finished_at,
        "run": str(run_dir),
        "input": {
            "load_config": relative_to(load_config, run_dir),
        },
        "outputs": {
            "export_dir": relative_to(export_dir, run_dir),
            "source_ply": relative_to(source_ply, run_dir),
            "scene_ply": relative_to(output_ply, run_dir),
            "viewer_config_json": relative_to(viewer_config, run_dir),
            "export_report_json": relative_to(run_dir / "reports" / REPORT_NAME, run_dir),
        },
        "environment": environment_report({"pixi": pixi_bin, "nerfstudio_dir": str(nerfstudio_dir)}),
        "commands": command_results_to_json(commands),
        "error": error,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_backend(args.backend)
    run_config = load_config(args.config)

    run_dir = args.run.expanduser()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise SystemExit(f"Run path is not a directory: {run_dir}")

    nerfstudio_dir = resolve_nerfstudio_dir(run_config, args.nerfstudio_dir, args.dry_run)
    pixi_bin = resolve_executable(args.pixi_bin, "pixi", args.dry_run)
    load_config_path = resolve_load_config(run_dir, args.load_config, args.dry_run)
    export_dir = resolve_under_run(run_dir, args.export_dir or "gsplat/exports/gaussian_splat")
    output_ply = resolve_under_run(run_dir, args.output_ply or "final/scene.ply")
    final_dir = output_ply.parent
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "gsplat" / "logs"
    command = build_export_command(pixi_bin, nerfstudio_dir, load_config_path, export_dir)

    if args.dry_run:
        print_dry_run(command, output_ply)
        return 0

    if output_ply.exists() and not args.overwrite:
        raise SystemExit(f"{output_ply} already exists. Use --overwrite to replace it.")

    export_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    command_results: List[CommandResult] = []
    status = "success"
    error: Optional[str] = None
    source_ply: Optional[Path] = None
    viewer_config: Optional[Path] = None

    try:
        command_results.append(run_logged_command("export_gaussian_splat", command, logs_dir))
        source_ply = find_exported_ply(export_dir)
        if source_ply is None:
            raise RuntimeError(f"ns-export finished but no .ply file was found under {export_dir}.")
        shutil.copy2(source_ply, output_ply)
        viewer_config = write_viewer_config(run_dir, final_dir, output_ply)
    except Exception as exc:
        status = "failed"
        error = str(exc)
        finished_at = utc_now()
        report = build_report(
            run_dir=run_dir,
            backend=args.backend,
            status=status,
            dry_run=False,
            started_at=started_at,
            finished_at=finished_at,
            pixi_bin=pixi_bin,
            nerfstudio_dir=nerfstudio_dir,
            load_config=load_config_path,
            export_dir=export_dir,
            output_ply=output_ply,
            source_ply=source_ply,
            viewer_config=viewer_config,
            commands=command_results,
            error=error,
        )
        report_path = reports_dir / REPORT_NAME
        write_json(report_path, report)
        raise SystemExit(f"{error}\nWrote failure report: {report_path}") from exc

    finished_at = utc_now()
    report = build_report(
        run_dir=run_dir,
        backend=args.backend,
        status=status,
        dry_run=False,
        started_at=started_at,
        finished_at=finished_at,
        pixi_bin=pixi_bin,
        nerfstudio_dir=nerfstudio_dir,
        load_config=load_config_path,
        export_dir=export_dir,
        output_ply=output_ply,
        source_ply=source_ply,
        viewer_config=viewer_config,
        commands=command_results,
        error=error,
    )
    report_path = reports_dir / REPORT_NAME
    write_json(report_path, report)
    print(f"\nExport complete. Canonical scene: {output_ply}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
