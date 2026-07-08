#!/usr/bin/env python3
"""Export final artifacts from a trained splat run.

The first canonical artifact is ``final/scene.ply``. Viewer-specific exports
such as .splat can be added once the browser viewer target is fixed.
"""

from __future__ import annotations

import argparse
import html
import json
import re
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
SPLAT_REPORT_JSON = "splat_report.json"
SPLAT_REPORT_HTML = "splat_report.html"


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
        "run",
        "--manifest-path",
        str(nerfstudio_dir / "pixi.toml"),
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


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def file_size_mb(path: Optional[Path]) -> Optional[float]:
    if path is None or not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 2)


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


def parse_training_downscale_factor(run_dir: Path) -> Optional[int]:
    log_path = run_dir / "gsplat" / "logs" / "train_splatfacto.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Auto image downscale factor of\s+(\d+)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"downscale_factor=([0-9]+)", text)
    if match:
        return int(match.group(1))
    return None


def manifest_camera_resolutions(run_dir: Path, downscale_factor: Optional[int]) -> List[Dict[str, Any]]:
    manifest = read_json(run_dir / "reports" / "image_manifest.json")
    groups: Dict[str, Dict[str, Any]] = {}
    factor = downscale_factor or 1
    for entry in manifest.get("images") or []:
        if not isinstance(entry, dict):
            continue
        group_id = entry.get("camera_group_id") or "{}_{}x{}".format(
            entry.get("camera_group"),
            entry.get("width"),
            entry.get("height"),
        )
        group = groups.setdefault(
            str(group_id),
            {
                "id": str(group_id),
                "camera_group": entry.get("camera_group"),
                "role": entry.get("role"),
                "source_width": entry.get("width"),
                "source_height": entry.get("height"),
                "training_width": None,
                "training_height": None,
                "image_count": 0,
                "locations": set(),
            },
        )
        group["image_count"] += 1
        if entry.get("location"):
            group["locations"].add(str(entry["location"]))

    output = []
    for group in groups.values():
        width = group.get("source_width")
        height = group.get("source_height")
        if isinstance(width, int) and isinstance(height, int):
            group["training_width"] = max(1, width // factor)
            group["training_height"] = max(1, height // factor)
        group["locations"] = sorted(group["locations"])
        output.append(group)
    return sorted(output, key=lambda item: item["id"])


def viewer_tier(splat_count: Optional[int]) -> Dict[str, Any]:
    if splat_count is None:
        return {"tier": "unknown", "detail": "PLY vertex count could not be read."}
    if splat_count <= 500_000:
        return {"tier": "web/mobile", "detail": "Within a common lightweight browser budget."}
    if splat_count <= 1_500_000:
        return {"tier": "desktop/browser", "detail": "Good desktop-quality range; mobile may need pruning."}
    if splat_count <= 3_000_000:
        return {"tier": "large/complex", "detail": "High-detail scene; expect heavier loading and rendering."}
    return {"tier": "archive/heavy", "detail": "Very heavy for browser delivery without pruning or compression."}


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


def camera_resolution_rows(groups: Sequence[Mapping[str, Any]]) -> str:
    if not groups:
        return '<tr><td colspan="7">No camera resolution manifest found.</td></tr>'
    rows = []
    for group in groups:
        source = "{}x{}".format(group.get("source_width"), group.get("source_height"))
        training = "{}x{}".format(group.get("training_width"), group.get("training_height"))
        rows.append(
            "<tr><td>{id}</td><td>{role}</td><td>{source}</td><td>{training}</td><td>{count}</td><td>{locations}</td></tr>".format(
                id=html.escape(str(group.get("id"))),
                role=html.escape(str(group.get("role"))),
                source=html.escape(source),
                training=html.escape(training),
                count=html.escape(str(group.get("image_count"))),
                locations=html.escape(", ".join(group.get("locations") or [])),
            )
        )
    return "\n".join(rows)


def write_splat_report_html(path: Path, report: Mapping[str, Any]) -> None:
    summary = report.get("summary") or {}
    training = report.get("training") or {}
    export = report.get("export") or {}
    viewer = report.get("viewer") or {}
    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Splat Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    main {{ max-width: 1100px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; width: 260px; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .note {{ color: #52606d; }}
  </style>
</head>
<body>
<main>
  <h1>Splat Report</h1>
  <p class="note">Generated {created_at}</p>

  <h2>Summary</h2>
  <table>{summary_rows}</table>

  <h2>Training Resolution</h2>
  <table>{training_rows}</table>
  <table>
    <tr><th>Camera Group</th><th>Role</th><th>Source Resolution</th><th>Training Resolution</th><th>Images</th><th>Locations</th></tr>
    {camera_rows}
  </table>

  <h2>Export</h2>
  <table>{export_rows}</table>

  <h2>Viewer Budget</h2>
  <table>{viewer_rows}</table>
</main>
</body>
</html>
""".format(
        created_at=html.escape(str(report.get("created_at"))),
        summary_rows=table_rows(summary),
        training_rows=table_rows(training),
        camera_rows=camera_resolution_rows(report.get("camera_resolutions") or []),
        export_rows=table_rows(export),
        viewer_rows=table_rows(viewer),
    )
    path.write_text(document, encoding="utf-8")


def build_splat_report(
    *,
    run_dir: Path,
    output_ply: Path,
    source_ply: Optional[Path],
    export_report: Mapping[str, Any],
    finished_at: str,
) -> Dict[str, Any]:
    training_report = read_json(run_dir / "reports" / "training_report.json")
    reconstruction_report = read_json(run_dir / "reports" / "reconstruction_report.json")
    downscale_factor = parse_training_downscale_factor(run_dir)
    splat_count = parse_ply_vertex_count(output_ply)
    tier = viewer_tier(splat_count)
    return {
        "schema_version": 1,
        "created_at": finished_at,
        "run": str(run_dir),
        "summary": {
            "status": export_report.get("status"),
            "splat_count": splat_count,
            "scene_ply_mb": file_size_mb(output_ply),
            "viewer_tier": tier["tier"],
            "training_downscale_factor": downscale_factor,
        },
        "training": {
            "method": (training_report.get("settings") or {}).get("method"),
            "max_steps": (training_report.get("settings") or {}).get("max_steps"),
            "num_downscales_generated": (training_report.get("settings") or {}).get("num_downscales"),
            "image_downscale_factor_used": downscale_factor,
            "selected_config": training_report.get("selected_config"),
            "registered_images": (reconstruction_report.get("reconstruction_metrics") or {}).get("registered_images"),
            "hero_registered": ((reconstruction_report.get("manifest_reconstruction") or {}).get("hero") or {}).get("registered"),
            "hero_dropped": ((reconstruction_report.get("manifest_reconstruction") or {}).get("hero") or {}).get("dropped"),
        },
        "camera_resolutions": manifest_camera_resolutions(run_dir, downscale_factor),
        "export": {
            "scene_ply": relative_to(output_ply, run_dir),
            "source_ply": relative_to(source_ply, run_dir),
            "scene_ply_mb": file_size_mb(output_ply),
            "source_ply_mb": file_size_mb(source_ply),
            "splat_count_from_ply_vertices": splat_count,
        },
        "viewer": tier,
    }


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
            "splat_report_json": relative_to(run_dir / "reports" / SPLAT_REPORT_JSON, run_dir),
            "splat_report_html": relative_to(run_dir / "reports" / SPLAT_REPORT_HTML, run_dir),
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
    splat_report = build_splat_report(
        run_dir=run_dir,
        output_ply=output_ply,
        source_ply=source_ply,
        export_report=report,
        finished_at=finished_at,
    )
    write_json(reports_dir / SPLAT_REPORT_JSON, splat_report)
    write_splat_report_html(reports_dir / SPLAT_REPORT_HTML, splat_report)
    print(f"\nExport complete. Canonical scene: {output_ply}")
    print(f"Report: {report_path}")
    print(f"Splat report: {reports_dir / SPLAT_REPORT_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
