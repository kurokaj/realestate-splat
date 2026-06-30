#!/usr/bin/env python3
"""Local SSH orchestration for the Verda reconstruction/training pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realestate_splat.cli import (  # noqa: E402
    CommandResult,
    command_results_to_json,
    relative_to,
    run_logged_command,
    utc_now,
    write_json,
)


DEFAULT_REMOTE_REPO = Path("/workspace/repo/realestate-splat")
DEFAULT_REMOTE_RUN_ROOT = Path("/workspace/runs")
DEFAULT_REMOTE_ENV = "/workspace/envs/splat-dev"
DEFAULT_REMOTE_CONFIG = "configs/training_splatfacto_dev.yaml"
FATAL_CAPTURE_WARNINGS = {
    "no_selected_frames",
    "target_max_below_coverage_minimum",
}
REVIEW_CAPTURE_WARNINGS = {
    "too_few_selected_frames",
    "too_few_selected_frames_total",
    "coverage_gaps_remain",
    "high_blur_rate",
    "low_texture_scene",
    "lighting_unstable",
}


@dataclass
class PreflightDecision:
    status: str
    fatal_reasons: List[str]
    review_reasons: List[str]
    warnings: List[str]
    selected_frame_count: int
    candidate_frame_count: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local preprocess, upload a zipped bundle, execute Verda COLMAP/training/export, and download artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path, help="Local source video directory to preprocess before upload.")
    source.add_argument("--run", type=Path, help="Existing local run directory containing frames_selected/.")
    parser.add_argument("--out", type=Path, help="Local run directory when --input-dir is used.")
    parser.add_argument("--profile", default="indoor_room", help="Preprocess profile when --input-dir is used.")
    parser.add_argument(
        "--preprocess-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument passed to scripts/preprocess_video.py. Repeat for multiple args, e.g. --preprocess-arg=--target-max=700.",
    )
    parser.add_argument("--host", required=True, help="SSH target, e.g. root@203.0.113.10 or a ~/.ssh/config alias.")
    parser.add_argument("--ssh-option", action="append", default=[], help="Extra SSH option, e.g. StrictHostKeyChecking=no.")
    parser.add_argument("--remote-repo", type=Path, default=DEFAULT_REMOTE_REPO, help="Repo path on Verda.")
    parser.add_argument("--remote-run-root", type=Path, default=DEFAULT_REMOTE_RUN_ROOT, help="Remote parent for run dirs.")
    parser.add_argument("--remote-run", type=Path, help="Remote run directory. Defaults to remote-run-root/local-run-name.")
    parser.add_argument("--remote-env", default=DEFAULT_REMOTE_ENV, help="Micromamba environment path on Verda.")
    parser.add_argument("--remote-config", default=DEFAULT_REMOTE_CONFIG, help="Config path on Verda, relative to remote repo unless absolute.")
    parser.add_argument(
        "--approval-mode",
        choices=["require_clean", "approve_warnings", "strict"],
        default="require_clean",
        help="Preflight gate before cloud spend. Fatal issues always stop. strict also stops on warnings; other modes ask the operator before upload.",
    )
    parser.add_argument("--colmap-mode", choices=["incremental", "global"], help="Override COLMAP mode on the remote run.")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential", "vocab_tree"], help="Override COLMAP matcher.")
    parser.add_argument("--max-steps", type=int, help="Override training max steps.")
    parser.add_argument("--num-downscales", type=int, help="Override Nerfstudio data downscale count.")
    parser.add_argument("--backend", choices=["splatfacto", "raw_gsplat"], help="Training backend.")
    parser.add_argument("--overwrite-remote", action="store_true", help="Replace remote generated outputs for this run.")
    parser.add_argument("--overwrite-local", action="store_true", help="Pass --overwrite to local preprocessing.")
    parser.add_argument("--skip-colmap", action="store_true", help="Skip remote COLMAP.")
    parser.add_argument("--skip-training", action="store_true", help="Skip remote training.")
    parser.add_argument("--skip-export", action="store_true", help="Skip remote export.")
    parser.add_argument("--skip-download", action="store_true", help="Do not download final artifacts/reports.")
    parser.add_argument("--yes-to-prompts", action="store_true", help="Run non-interactively after fatal preflight checks pass.")
    parser.add_argument("--dry-run", action="store_true", help="Print and validate planned steps without writing or connecting.")
    return parser.parse_args(argv)


def ssh_base(args: argparse.Namespace) -> List[str]:
    command = ["ssh"]
    for option in args.ssh_option:
        command.extend(["-o", option])
    command.append(args.host)
    return command


def rsync_base(args: argparse.Namespace) -> List[str]:
    command = ["rsync", "-az", "--info=progress2"]
    if args.ssh_option:
        ssh_parts = ["ssh"]
        for option in args.ssh_option:
            ssh_parts.extend(["-o", option])
        command.extend(["-e", shlex.join(ssh_parts)])
    return command


def remote_shell_command(script: str) -> str:
    return "bash -lc " + shlex.quote(script)


def remote_config_path(args: argparse.Namespace) -> Path:
    config = Path(args.remote_config)
    if config.is_absolute():
        return config
    return args.remote_repo / config


def remote_env_prefix(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(args.remote_repo))}",
            "source ~/.bashrc",
            f"micromamba activate {shlex.quote(args.remote_env)}",
        ]
    )


def build_remote_python_command(args: argparse.Namespace, script_name: str, remote_run: Path) -> List[str]:
    command = [
        "python",
        f"scripts/{script_name}",
        "--run",
        str(remote_run),
        "--config",
        str(remote_config_path(args)),
    ]
    if script_name == "run_colmap.py":
        if args.colmap_mode:
            command.extend(["--mode", args.colmap_mode])
        if args.matcher:
            command.extend(["--matcher", args.matcher])
        if args.overwrite_remote:
            command.append("--overwrite")
    elif script_name == "run_training.py":
        if args.backend:
            command.extend(["--backend", args.backend])
        if args.max_steps is not None:
            command.extend(["--max-steps", str(args.max_steps)])
        if args.num_downscales is not None:
            command.extend(["--num-downscales", str(args.num_downscales)])
    elif script_name == "export_scene.py":
        if args.backend:
            command.extend(["--backend", args.backend])
        if args.overwrite_remote:
            command.append("--overwrite")
    return command


def planned_remote_stage(args: argparse.Namespace, script_name: str, remote_run: Path) -> str:
    command = build_remote_python_command(args, script_name, remote_run)
    return remote_env_prefix(args) + "\n" + shlex.join(command)


def run_remote_stage(
    args: argparse.Namespace,
    name: str,
    script_name: str,
    remote_run: Path,
    logs_dir: Path,
) -> CommandResult:
    script = planned_remote_stage(args, script_name, remote_run)
    command = ssh_base(args) + [remote_shell_command(script)]
    return run_logged_command(name, command, logs_dir)


def preprocess_if_needed(args: argparse.Namespace, logs_dir: Path) -> Tuple[Path, Optional[CommandResult]]:
    if args.run is not None:
        return args.run.expanduser(), None

    assert args.input_dir is not None
    out_dir = args.out or (Path("runs") / args.input_dir.name)
    command = [
        sys.executable,
        "scripts/preprocess_video.py",
        "--input-dir",
        str(args.input_dir),
        "--out",
        str(out_dir),
        "--profile",
        args.profile,
    ]
    if args.overwrite_local:
        command.append("--overwrite")
    command.extend(args.preprocess_arg)
    result = run_logged_command("local_preprocess", command, logs_dir, Path.cwd())
    return out_dir, result


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Expected report does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse JSON report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON report root must be an object: {path}")
    return payload


def count_selected_frames(run_dir: Path) -> int:
    frames_dir = run_dir / "frames_selected"
    if not frames_dir.exists():
        return 0
    suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sum(1 for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def evaluate_preflight(run_dir: Path) -> PreflightDecision:
    report = load_json(run_dir / "reports" / "capture_report.json")
    warnings = sorted(str(item) for item in report.get("warnings") or [])
    summary = report.get("summary") or {}
    settings = report.get("settings") or {}
    selected = int(summary.get("selected_frame_count") or count_selected_frames(run_dir))
    candidates = int(summary.get("candidate_frame_count") or 0)
    target_min = int(settings.get("target_min") or 0)

    fatal = sorted(set(warnings).intersection(FATAL_CAPTURE_WARNINGS))
    review = sorted(set(warnings).intersection(REVIEW_CAPTURE_WARNINGS))

    if selected == 0 and "no_selected_frames" not in fatal:
        fatal.append("no_selected_frames")
    if target_min > 0 and selected < max(20, int(target_min * 0.25)):
        fatal.append("selected_frame_count_far_below_target_min")

    for video in report.get("videos") or []:
        coverage = video.get("coverage") or {}
        window_count = int(coverage.get("candidate_window_count") or 0)
        below_count = int(coverage.get("windows_below_minimum_count") or 0)
        if window_count > 0 and below_count / window_count >= 0.25:
            fatal.append(f"{video.get('source_id')}:coverage_windows_below_minimum_over_25_percent")
        elif below_count > 0:
            review.append(f"{video.get('source_id')}:coverage_gaps_remain")

    fatal = sorted(set(fatal))
    review = sorted(set(review))
    status = "clean"
    if fatal:
        status = "fatal"
    elif review or warnings:
        status = "review"

    return PreflightDecision(
        status=status,
        fatal_reasons=fatal,
        review_reasons=review,
        warnings=warnings,
        selected_frame_count=selected,
        candidate_frame_count=candidates,
    )


def enforce_preflight(decision: PreflightDecision, approval_mode: str) -> None:
    if approval_mode == "strict" and decision.warnings:
        raise SystemExit(
            "Capture preflight failed in strict mode because warnings are present:\n  "
            + "\n  ".join(decision.warnings)
        )
    if decision.fatal_reasons:
        raise SystemExit(
            "Capture preflight failed; do not start cloud processing for this capture:\n  "
            + "\n  ".join(decision.fatal_reasons)
        )


def prompt_to_continue(message: str, assume_yes: bool) -> None:
    if assume_yes:
        print(message)
        print("Continuing because --yes-to-prompts is set.")
        return

    print(message)
    answer = input("Continue? Type 'yes' to continue: ").strip().lower()
    if answer not in {"yes", "y"}:
        raise SystemExit("Operator stopped the pipeline.")


def add_to_zip(zip_file: zipfile.ZipFile, source: Path, arcname: Path) -> None:
    if source.is_dir():
        for child in sorted(source.rglob("*")):
            if child.is_file():
                zip_file.write(child, arcname / child.relative_to(source))
    elif source.exists():
        zip_file.write(source, arcname)


def build_upload_bundle(run_dir: Path, work_dir: Path) -> Path:
    bundle_path = work_dir / f"{run_dir.name}_upload_bundle.zip"
    if bundle_path.exists():
        bundle_path.unlink()
    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        add_to_zip(zip_file, run_dir / "frames_selected", Path("frames_selected"))
        add_to_zip(zip_file, run_dir / "reports" / "capture_report.json", Path("reports/capture_report.json"))
        add_to_zip(zip_file, run_dir / "reports" / "capture_report.html", Path("reports/capture_report.html"))
        add_to_zip(zip_file, run_dir / "reports" / "gpu_recommendation.json", Path("reports/gpu_recommendation.json"))
        add_to_zip(zip_file, run_dir / "run_config.json", Path("run_config.json"))
    return bundle_path


def remote_run_dir(args: argparse.Namespace, local_run: Path) -> Path:
    if args.remote_run is not None:
        return args.remote_run
    return args.remote_run_root / local_run.name


def make_remote_run(args: argparse.Namespace, remote_run: Path, logs_dir: Path) -> CommandResult:
    script = "set -euo pipefail\n" + f"mkdir -p {shlex.quote(str(remote_run))}"
    return run_logged_command("remote_prepare_run_dir", ssh_base(args) + [remote_shell_command(script)], logs_dir)


def upload_bundle(args: argparse.Namespace, bundle_path: Path, remote_run: Path, logs_dir: Path) -> CommandResult:
    destination = f"{args.host}:{shlex.quote(str(remote_run / bundle_path.name))}"
    return run_logged_command("upload_bundle", rsync_base(args) + [str(bundle_path), destination], logs_dir)


def unpack_remote_bundle(args: argparse.Namespace, bundle_path: Path, remote_run: Path, logs_dir: Path) -> CommandResult:
    remote_bundle = remote_run / bundle_path.name
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(remote_run))}",
            "rm -rf frames_selected",
            f"python3 -m zipfile -e {shlex.quote(str(remote_bundle))} {shlex.quote(str(remote_run))}",
            f"rm -f {shlex.quote(str(remote_bundle))}",
        ]
    )
    return run_logged_command("remote_unpack_bundle", ssh_base(args) + [remote_shell_command(script)], logs_dir)


def download_artifacts(args: argparse.Namespace, local_run: Path, remote_run: Path, logs_dir: Path) -> List[CommandResult]:
    destination = local_run / "cloud_artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    remote_bundle = remote_run / "cloud_artifacts.zip"
    local_bundle = destination / "cloud_artifacts.zip"
    zip_python = (
        "import pathlib, zipfile; "
        "roots = [pathlib.Path(p) for p in ['final', 'reports', 'colmap/logs', 'gsplat/logs']]; "
        "z = zipfile.ZipFile('cloud_artifacts.zip', 'w', zipfile.ZIP_DEFLATED); "
        "[z.write(path, path) for root in roots if root.exists() for path in root.rglob('*') if path.is_file()]; "
        "z.close()"
    )
    zip_script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(remote_run))}",
            "rm -f cloud_artifacts.zip",
            "python3 -c " + shlex.quote(zip_python),
        ]
    )
    results = [
        run_logged_command("remote_pack_artifacts", ssh_base(args) + [remote_shell_command(zip_script)], logs_dir),
        run_logged_command(
            "download_artifacts",
            rsync_base(args) + [f"{args.host}:{shlex.quote(str(remote_bundle))}", str(destination)],
            logs_dir,
        ),
        run_logged_command(
            "unpack_artifacts",
            [sys.executable, "-m", "zipfile", "-e", str(local_bundle), str(destination)],
            logs_dir,
        ),
    ]
    return results


def download_colmap_review(args: argparse.Namespace, local_run: Path, remote_run: Path, logs_dir: Path) -> List[CommandResult]:
    destination = local_run / "cloud_artifacts" / "colmap_review"
    destination.mkdir(parents=True, exist_ok=True)
    remote_bundle = remote_run / "colmap_review.zip"
    local_bundle = destination / "colmap_review.zip"
    zip_python = (
        "import pathlib, zipfile; "
        "roots = [pathlib.Path(p) for p in ['reports/reconstruction_report.json', "
        "'reports/reconstruction_report.html', 'colmap/logs/model_analyzer.log']]; "
        "z = zipfile.ZipFile('colmap_review.zip', 'w', zipfile.ZIP_DEFLATED); "
        "[z.write(path, path) for path in roots if path.exists() and path.is_file()]; "
        "z.close()"
    )
    zip_script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(remote_run))}",
            "rm -f colmap_review.zip",
            "python3 -c " + shlex.quote(zip_python),
        ]
    )
    return [
        run_logged_command("remote_pack_colmap_review", ssh_base(args) + [remote_shell_command(zip_script)], logs_dir),
        run_logged_command(
            "download_colmap_review",
            rsync_base(args) + [f"{args.host}:{shlex.quote(str(remote_bundle))}", str(destination)],
            logs_dir,
        ),
        run_logged_command(
            "unpack_colmap_review",
            [sys.executable, "-m", "zipfile", "-e", str(local_bundle), str(destination)],
            logs_dir,
        ),
    ]


def summarize_colmap_review(local_run: Path) -> str:
    report_path = local_run / "cloud_artifacts" / "colmap_review" / "reports" / "reconstruction_report.json"
    html_path = local_run / "cloud_artifacts" / "colmap_review" / "reports" / "reconstruction_report.html"
    if not report_path.exists():
        return f"COLMAP review report was not downloaded to {report_path}."

    report = load_json(report_path)
    metrics = report.get("reconstruction_metrics") or {}
    metric_lines = []
    for key in [
        "registered_images",
        "images",
        "points",
        "observations",
        "mean_track_length",
        "mean_observations_per_image",
        "mean_reprojection_error_px",
    ]:
        if key in metrics:
            metric_lines.append(f"  {key}: {metrics[key]}")
    metrics_text = "\n".join(metric_lines) if metric_lines else "  No model analyzer metrics found."
    return (
        "COLMAP finished and the review bundle has been downloaded locally.\n"
        f"HTML report: {html_path}\n"
        f"JSON report: {report_path}\n"
        "Key metrics:\n"
        f"{metrics_text}\n"
        "Review the HTML/metrics before starting training."
    )


def summarize_preflight(decision: PreflightDecision) -> str:
    lines = [
        f"Selected frames: {decision.selected_frame_count}, candidates: {decision.candidate_frame_count}",
        f"Preflight status: {decision.status}",
    ]
    if decision.review_reasons:
        lines.append("Review reasons:")
        lines.extend(f"  {reason}" for reason in decision.review_reasons)
    elif decision.warnings:
        lines.append("Warnings:")
        lines.extend(f"  {warning}" for warning in decision.warnings)
    else:
        lines.append("No warning flags.")
    return "\n".join(lines)


def print_dry_run(
    args: argparse.Namespace,
    local_run: Path,
    remote_run: Path,
    bundle_path: Path,
    decision: PreflightDecision,
) -> None:
    print("Dry run. No files will be created, uploaded, or executed.\n")
    print(f"# local run: {local_run}")
    print(f"# remote run: {remote_run}")
    print(f"# upload bundle: {bundle_path}")
    print(f"# preflight: {asdict(decision)}")
    print("# pause for operator approval after local capture report")
    prepare_script = "set -euo pipefail\nmkdir -p " + shlex.quote(str(remote_run))
    print(f"$ {shlex.join(ssh_base(args) + [remote_shell_command(prepare_script)])}")
    print(f"$ {shlex.join(rsync_base(args) + [str(bundle_path), f'{args.host}:{remote_run / bundle_path.name}'])}")
    for name, script_name, skip in [
        ("remote_colmap", "run_colmap.py", args.skip_colmap),
        ("remote_training", "run_training.py", args.skip_training),
        ("remote_export", "export_scene.py", args.skip_export),
    ]:
        if skip:
            print(f"# {name} skipped")
            continue
        print(f"$ {shlex.join(ssh_base(args) + [remote_shell_command(planned_remote_stage(args, script_name, remote_run))])}")
        if name == "remote_colmap" and not args.skip_training:
            print("# download reports/reconstruction_report.html, reconstruction_report.json, and model_analyzer.log")
            print("# pause for operator approval before remote_training")
    if not args.skip_download:
        print("# zip final/, reports/, and logs remotely, then rsync one cloud_artifacts.zip back")


def build_pipeline_report(
    *,
    local_run: Path,
    remote_run: Path,
    bundle_path: Path,
    decision: PreflightDecision,
    commands: Sequence[CommandResult],
    status: str,
    started_at: str,
    finished_at: str,
    error: Optional[str],
    dry_run: bool,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": finished_at,
        "status": status,
        "dry_run": dry_run,
        "started_at": started_at,
        "finished_at": finished_at,
        "local_run": str(local_run),
        "remote_run": str(remote_run),
        "upload_bundle": relative_to(bundle_path, local_run),
        "preflight": asdict(decision),
        "commands": command_results_to_json(commands),
        "error": error,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    initial_run = args.run.expanduser() if args.run is not None else (args.out or Path("runs") / args.input_dir.name)
    work_dir = initial_run / "cloud_pipeline"
    logs_dir = work_dir / "logs"
    commands: List[CommandResult] = []
    started_at = utc_now()
    status = "success"
    error: Optional[str] = None

    if not args.dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.dry_run:
            local_run = initial_run
        else:
            local_run, preprocess_result = preprocess_if_needed(args, logs_dir)
            if preprocess_result is not None:
                commands.append(preprocess_result)
        if args.dry_run and not (local_run / "reports" / "capture_report.json").exists():
            decision = PreflightDecision("unknown", [], [], [], 0, 0)
        else:
            decision = evaluate_preflight(local_run)
            enforce_preflight(decision, args.approval_mode)
            if not args.dry_run:
                prompt_to_continue(
                    "Local preprocessing is complete.\n"
                    f"Capture HTML report: {local_run / 'reports' / 'capture_report.html'}\n"
                    f"{summarize_preflight(decision)}\n"
                    "Review the capture report before uploading to Verda.",
                    args.yes_to_prompts,
                )
        remote_run = remote_run_dir(args, local_run)
        bundle_path = work_dir / f"{local_run.name}_upload_bundle.zip"

        if args.dry_run:
            print_dry_run(args, local_run, remote_run, bundle_path, decision)
            return 0

        bundle_path = build_upload_bundle(local_run, work_dir)
        commands.append(make_remote_run(args, remote_run, logs_dir))
        commands.append(upload_bundle(args, bundle_path, remote_run, logs_dir))
        commands.append(unpack_remote_bundle(args, bundle_path, remote_run, logs_dir))

        if not args.skip_colmap:
            commands.append(run_remote_stage(args, "remote_colmap", "run_colmap.py", remote_run, logs_dir))
            commands.extend(download_colmap_review(args, local_run, remote_run, logs_dir))
            if not args.skip_training:
                prompt_to_continue(summarize_colmap_review(local_run), args.yes_to_prompts)
        if not args.skip_training:
            commands.append(run_remote_stage(args, "remote_training", "run_training.py", remote_run, logs_dir))
        if not args.skip_export:
            commands.append(run_remote_stage(args, "remote_export", "export_scene.py", remote_run, logs_dir))
        if not args.skip_download:
            commands.extend(download_artifacts(args, local_run, remote_run, logs_dir))
    except Exception as exc:
        status = "failed"
        error = str(exc)
        finished_at = utc_now()
        if not args.dry_run:
            report = build_pipeline_report(
                local_run=initial_run,
                remote_run=remote_run_dir(args, initial_run),
                bundle_path=work_dir / f"{initial_run.name}_upload_bundle.zip",
                decision=evaluate_preflight(initial_run) if (initial_run / "reports" / "capture_report.json").exists() else PreflightDecision("unknown", [], [], [], 0, 0),
                commands=commands,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                error=error,
                dry_run=False,
            )
            write_json(work_dir / "cloud_pipeline_report.json", report)
        raise SystemExit(str(exc)) from exc

    finished_at = utc_now()
    report = build_pipeline_report(
        local_run=local_run,
        remote_run=remote_run,
        bundle_path=bundle_path,
        decision=decision,
        commands=commands,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        error=error,
        dry_run=False,
    )
    report_path = work_dir / "cloud_pipeline_report.json"
    write_json(report_path, report)
    print(f"\nCloud pipeline complete. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
