"""Terminal-run worker-controller for Milestone 8A."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import signal
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from psycopg.types.json import Jsonb

from controller_common.config import (
    controller_git_ref,
    controller_id,
    controller_repo_url,
    poll_interval_seconds,
    r2_endpoint,
    runpod_colmap_cloud_type,
    runpod_colmap_container_disk_gb,
    runpod_colmap_gpu_types,
    runpod_colmap_image,
    runpod_colmap_poll_seconds,
    runpod_colmap_timeout_seconds,
    stage_python_bin,
)
from controller_common.db import claim_next_queued_stage, complete_stage_run, connect, create_event, ensure_schema
from controller_common.fake_provider import FakeProvider
from controller_common.runpod_provider import RunpodClient


STOP_REQUESTED = False


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; stopping after current loop.", flush=True)


def run_once(*, worker_id: str) -> bool:
    with connect() as conn:
        with conn.transaction():
            stage_run = claim_next_queued_stage(conn, claimed_by=worker_id)
    if stage_run is None:
        return False

    stage_run_id = stage_run["id"]
    print(f"Claimed {stage_run['stage']} stage run {stage_run_id}", flush=True)
    try:
        if stage_run["provider"] == "local_preprocess" and stage_run["stage"] == "preprocess":
            summary, output_uri = run_local_preprocess(stage_run)
        elif stage_run["provider"] == "runpod_colmap" and stage_run["stage"] == "colmap":
            summary, output_uri = run_runpod_colmap(stage_run)
        else:
            summary, output_uri = run_fake_stage(stage_run)

        with connect() as conn:
            with conn.transaction():
                complete_stage_run(
                    conn,
                    stage_run_id=stage_run_id,
                    summary=summary,
                    output_uri=output_uri,
                )
        print(f"Completed stage run {stage_run_id}", flush=True)
        return True
    except Exception as exc:
        with connect() as conn:
            create_event(
                conn,
                stage_run_id=stage_run_id,
                kind="stage_failed",
                level="error",
                message=str(exc),
            )
            conn.execute(
                """
                UPDATE stage_runs
                SET status = 'failed',
                    error_message = %s,
                    finished_at = now(),
                    updated_at = now()
                WHERE id = %s
                """,
                (str(exc), stage_run_id),
            )
            conn.execute(
                """
                UPDATE projects
                SET status = 'failed',
                    updated_at = now()
                WHERE id = %s
                """,
                (stage_run["project_id"],),
            )
        print(f"Failed fake stage run {stage_run_id}: {exc}", flush=True)
        return True


def run_fake_stage(stage_run: dict[str, Any]) -> tuple[dict[str, Any], str]:
    provider = FakeProvider()
    stage_run_id = stage_run["id"]
    with connect() as conn:
        create_event(
            conn,
            stage_run_id=stage_run_id,
            kind="fake_provider_started",
            message="Fake provider started stage",
            payload={"provider": provider.name},
        )

    def progress(percent: int, message: str) -> None:
        record_progress(stage_run_id, percent, message, kind="fake_provider_progress")

    summary = provider.run_stage(stage_run, progress=progress)
    output_uri = stage_run["output_uri"] or fake_output_uri(stage_run)
    return summary, output_uri


def run_local_preprocess(stage_run: dict[str, Any]) -> tuple[dict[str, Any], str]:
    stage_run_id = stage_run["id"]
    inputs = stage_run["input_uri_json"] or {}
    raw_uri = inputs.get("raw_uri")
    output_base_uri = inputs.get("output_uri")
    if not raw_uri:
        raise ValueError("local_preprocess requires input_uri_json.raw_uri")
    if not output_base_uri:
        raise ValueError("local_preprocess requires input_uri_json.output_uri")
    require_r2_uri(raw_uri, "raw_uri")
    require_r2_uri(output_base_uri, "output_uri")

    command = build_preprocess_command(stage_run, inputs)
    with connect() as conn:
        create_event(
            conn,
            stage_run_id=stage_run_id,
            kind="local_preprocess_started",
            message="Started local preprocess stage command",
            payload={"command": command},
        )
        conn.execute(
            """
            UPDATE stage_runs
            SET command = %s,
                progress_json = jsonb_build_object('percent', 10, 'message', 'Local preprocess command starting'),
                updated_at = now()
            WHERE id = %s
            """,
            (" ".join(command), stage_run_id),
        )

    env = os.environ.copy()
    endpoint_url = inputs.get("endpoint_url") or r2_endpoint()
    if endpoint_url:
        env["R2_ENDPOINT"] = endpoint_url
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_line_count = 0
    assert process.stdout is not None
    for line in process.stdout:
        clean_line = line.rstrip()
        if clean_line:
            output_line_count += 1
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Local preprocess failed with exit code {return_code}")

    record_progress(stage_run_id, 90, "Local preprocess command finished", kind="local_preprocess_progress")
    current_uri = f"{output_base_uri.rstrip('/')}/current"
    summary = {} if inputs.get("dry_run") else load_preprocess_summary(output_base_uri)
    summary.update(
        {
            "provider": "local_preprocess",
            "stage": "preprocess",
            "return_code": return_code,
            "output_line_count": output_line_count,
            "stage_result_uri": f"{current_uri}/stage_result.json",
            "preprocess_summary_uri": f"{current_uri}/preprocess_summary.json",
        }
    )
    return summary, current_uri


def build_preprocess_command(stage_run: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    command = [
        inputs.get("stage_python_bin") or stage_python_bin(),
        "scripts/run_preprocess_stage.py",
        "--project-id",
        stage_run["project_id"],
        "--stage-run-id",
        stage_run["id"],
        "--raw-uri",
        inputs["raw_uri"],
        "--output-uri",
        inputs["output_uri"],
        "--profile",
        inputs.get("profile", "indoor_room"),
    ]
    endpoint_url = inputs.get("endpoint_url") or r2_endpoint()
    if endpoint_url:
        command.extend(["--endpoint-url", endpoint_url])
    python_bin = inputs.get("python_bin")
    if python_bin:
        command.extend(["--python-bin", python_bin])
    if inputs.get("dry_run"):
        command.append("--dry-run")
    for preprocess_arg in inputs.get("preprocess_args", []):
        append_argparse_value(command, "--preprocess-arg", preprocess_arg)
    return command


def append_argparse_value(command: list[str], option: str, value: Any) -> None:
    text = str(value)
    if text.startswith("-"):
        command.append(f"{option}={text}")
    else:
        command.extend([option, text])


def run_runpod_colmap(stage_run: dict[str, Any]) -> tuple[dict[str, Any], str]:
    stage_run_id = stage_run["id"]
    inputs = stage_run["input_uri_json"] or {}
    preprocess_uri = inputs.get("preprocess_uri") or inputs.get("input_uri")
    output_base_uri = inputs.get("output_uri")
    if not preprocess_uri:
        raise ValueError("runpod_colmap requires input_uri_json.preprocess_uri")
    if not output_base_uri:
        raise ValueError("runpod_colmap requires input_uri_json.output_uri")
    require_r2_uri(preprocess_uri, "preprocess_uri")
    require_r2_uri(output_base_uri, "output_uri")

    remote_command = build_colmap_stage_shell_command(stage_run, inputs)
    current_uri = f"{output_base_uri.rstrip('/')}/current"
    image = stage_run.get("image") or inputs.get("image") or runpod_colmap_image()
    with connect() as conn:
        create_event(
            conn,
            stage_run_id=stage_run_id,
            kind="runpod_colmap_prepared",
            message="Prepared RunPod COLMAP pod command",
            payload={
                "image": image,
                "gpu_type_ids": inputs.get("gpu_type_ids") or runpod_colmap_gpu_types(),
                "dry_run": bool(inputs.get("dry_run")),
            },
        )
        conn.execute(
            """
            UPDATE stage_runs
            SET command = %s,
                image = %s,
                progress_json = %s,
                updated_at = now()
            WHERE id = %s
            """,
            (
                remote_command,
                image,
                Jsonb({"percent": 10, "message": "RunPod COLMAP command prepared"}),
                stage_run_id,
            ),
        )

    if inputs.get("dry_run"):
        return (
            {
                "provider": "runpod_colmap",
                "stage": "colmap",
                "dry_run": True,
                "image": image,
                "mode": inputs.get("mode", "global"),
                "matcher": inputs.get("matcher", "exhaustive"),
                "stage_result_uri": f"{current_uri}/stage_result.json",
                "reconstruction_report_uri": f"{current_uri}/reconstruction_report.json",
            },
            current_uri,
        )

    pod_payload = build_runpod_colmap_pod_payload(stage_run, inputs, image=image, remote_command=remote_command)
    client = RunpodClient()
    pod = client.create_pod(pod_payload)
    with connect() as conn:
        create_event(
            conn,
            stage_run_id=stage_run_id,
            kind="runpod_pod_created",
            message="Created RunPod COLMAP pod",
            payload={"pod_id": pod.id, "image": image},
        )
        conn.execute(
            """
            UPDATE stage_runs
            SET provider_job_id = %s,
                provider_pod_id = %s,
                status = 'colmap_pod_starting',
                progress_json = %s,
                updated_at = now()
            WHERE id = %s
            """,
            (pod.id, pod.id, Jsonb({"percent": 20, "message": "RunPod pod created"}), stage_run_id),
        )

    keep_pod = bool(inputs.get("keep_pod"))
    try:
        stage_result = wait_for_colmap_stage_result(stage_run_id, client, pod.id, output_base_uri)
        reconstruction_report = load_optional_json_from_r2(f"{current_uri}/reconstruction_report.json")
        return compact_colmap_summary(stage_result, reconstruction_report, provider_job_id=pod.id), current_uri
    finally:
        if not keep_pod:
            delete_runpod_pod(client, stage_run_id=stage_run_id, pod_id=pod.id, reason="stage_finished")


def build_colmap_stage_shell_command(stage_run: dict[str, Any], inputs: dict[str, Any]) -> str:
    repo_url = inputs.get("repo_url") or controller_repo_url()
    if not repo_url:
        raise ValueError("CONTROLLER_REPO_URL or input_uri_json.repo_url is required for runpod_colmap")
    git_ref = inputs.get("git_ref") or controller_git_ref()
    output_uri = inputs["output_uri"].rstrip("/")
    command = [
        "python3",
        "scripts/run_colmap_stage.py",
        "--project-id",
        stage_run["project_id"],
        "--stage-run-id",
        stage_run["id"],
        "--input-uri",
        inputs.get("preprocess_uri") or inputs["input_uri"],
        "--output-uri",
        output_uri,
        "--mode",
        inputs.get("mode", "global"),
        "--matcher",
        inputs.get("matcher", "exhaustive"),
        "--camera-model",
        inputs.get("camera_model", "SIMPLE_RADIAL"),
        "--colmap-bin",
        inputs.get("colmap_bin", "/opt/colmap-cuda/bin/colmap"),
    ]
    for colmap_arg in inputs.get("colmap_args", []):
        append_argparse_value(command, "--colmap-arg", colmap_arg)

    return "\n".join(
        [
            "set -euo pipefail",
            "mkdir -p /workspace",
            "cd /workspace",
            f"if [ ! -d Buildvision3D/.git ]; then git clone --branch {shlex.quote(git_ref)} {shlex.quote(repo_url)} Buildvision3D; fi",
            "cd /workspace/Buildvision3D",
            f"git fetch origin {shlex.quote(git_ref)} || true",
            f"git checkout {shlex.quote(git_ref)}",
            "git pull --ff-only || true",
            " ".join(shlex.quote(part) for part in command),
        ]
    )


def build_runpod_colmap_pod_payload(
    stage_run: dict[str, Any],
    inputs: dict[str, Any],
    *,
    image: str,
    remote_command: str,
) -> dict[str, Any]:
    env = {
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "auto"),
        "R2_ENDPOINT": inputs.get("endpoint_url") or r2_endpoint() or "",
        "R2_BUCKET": os.environ.get("R2_BUCKET", ""),
    }
    missing_env = [key for key, value in env.items() if key != "AWS_DEFAULT_REGION" and not value]
    if missing_env:
        raise ValueError(f"Missing required RunPod/R2 environment variables: {', '.join(missing_env)}")

    gpu_type_ids = inputs.get("gpu_type_ids") or runpod_colmap_gpu_types()
    return {
        "name": f"buildvision3d-colmap-{stage_run['project_id']}-{stage_run['id']}"[:80],
        "imageName": image,
        "computeType": "GPU",
        "cloudType": inputs.get("cloud_type") or runpod_colmap_cloud_type(),
        "gpuCount": int(inputs.get("gpu_count") or 1),
        "gpuTypeIds": gpu_type_ids,
        "gpuTypePriority": inputs.get("gpu_type_priority") or "availability",
        "containerDiskInGb": int(inputs.get("container_disk_gb") or runpod_colmap_container_disk_gb()),
        "minVCPUPerGPU": int(inputs.get("min_vcpu_per_gpu") or 4),
        "minRAMPerGPU": int(inputs.get("min_ram_per_gpu") or 16),
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [remote_command],
        "env": env,
        "ports": [],
        "supportPublicIp": False,
    }


def wait_for_colmap_stage_result(
    stage_run_id: str,
    client: RunpodClient,
    pod_id: str,
    output_base_uri: str,
) -> dict[str, Any]:
    started = time.monotonic()
    poll_seconds = runpod_colmap_poll_seconds()
    timeout_seconds = runpod_colmap_timeout_seconds()
    result_uri = f"{output_base_uri.rstrip('/')}/current/stage_result.json"
    last_pod_status = None
    while True:
        if stage_was_cancelled(stage_run_id):
            raise RuntimeError("Stage was cancelled while RunPod COLMAP was running")
        stage_result = load_optional_json_from_r2(result_uri)
        if stage_result:
            status = stage_result.get("status")
            record_progress(stage_run_id, 95, f"Found COLMAP stage_result.json with status {status}", kind="runpod_colmap_stage_result")
            if status == "completed":
                return stage_result
            raise RuntimeError(stage_result.get("error_message") or f"COLMAP stage_result status was {status}")

        pod_status = provider_pod_status(client, pod_id)
        if pod_status != last_pod_status:
            last_pod_status = pod_status
            with connect() as conn:
                create_event(
                    conn,
                    stage_run_id=stage_run_id,
                    kind="runpod_pod_status",
                    message=f"RunPod pod status: {pod_status or 'unknown'}",
                    payload={"pod_id": pod_id, "status": pod_status},
                )
                conn.execute(
                    """
                    UPDATE stage_runs
                    SET status = 'colmap_running',
                        progress_json = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (Jsonb({"percent": 50, "message": f"Waiting for R2 stage_result.json ({pod_status or 'pod status unknown'})"}), stage_run_id),
                )

        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {result_uri}")
        time.sleep(poll_seconds)


def stage_was_cancelled(stage_run_id: str) -> bool:
    with connect() as conn:
        row = conn.execute("SELECT status FROM stage_runs WHERE id = %s", (stage_run_id,)).fetchone()
    return bool(row and row.get("status") == "cancelled")


def cleanup_runpod_colmap_pods(*, worker_id: str) -> None:
    try:
        client = RunpodClient()
    except Exception as exc:
        print(f"RunPod watchdog skipped: {exc}", flush=True)
        return

    timeout_seconds = runpod_colmap_timeout_seconds()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM stage_runs
            WHERE provider = 'runpod_colmap'
              AND provider_pod_id IS NOT NULL
              AND (
                    status = ANY(%s)
                    OR (
                        status = ANY(%s)
                        AND now() - COALESCE(updated_at, started_at, claimed_at, created_at) > (%s * interval '1 second')
                    )
              )
            ORDER BY updated_at ASC
            LIMIT 20
            """,
            (
                [
                    "approved",
                    "completed",
                    "failed",
                    "cancelled",
                    "awaiting_colmap_approval",
                    "colmap_rejected",
                ],
                ["colmap_pod_starting", "colmap_running"],
                timeout_seconds,
            ),
        ).fetchall()

    if not rows:
        return
    print(f"RunPod watchdog found {len(rows)} pod(s) to clean up.", flush=True)
    for row in rows:
        stage_run_id = row["id"]
        pod_id = row["provider_pod_id"]
        status = row["status"]
        timed_out = status in {"colmap_pod_starting", "colmap_running"}
        if timed_out:
            with connect() as conn:
                create_event(
                    conn,
                    stage_run_id=stage_run_id,
                    kind="runpod_watchdog_timeout",
                    level="warning",
                    message="RunPod COLMAP stage exceeded watchdog timeout",
                    payload={"pod_id": pod_id, "worker_id": worker_id, "timeout_seconds": timeout_seconds},
                )
                conn.execute(
                    """
                    UPDATE stage_runs
                    SET status = 'failed',
                        error_message = COALESCE(error_message, %s),
                        finished_at = COALESCE(finished_at, now()),
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (f"RunPod watchdog timed out after {timeout_seconds:.0f} seconds", stage_run_id),
                )
        delete_runpod_pod(client, stage_run_id=stage_run_id, pod_id=pod_id, reason="watchdog_cleanup")


def delete_runpod_pod(client: RunpodClient, *, stage_run_id: str, pod_id: str, reason: str) -> None:
    try:
        client.delete_pod(pod_id)
        with connect() as conn:
            create_event(
                conn,
                stage_run_id=stage_run_id,
                kind="runpod_pod_deleted",
                message="Deleted RunPod COLMAP pod",
                payload={"pod_id": pod_id, "reason": reason},
            )
            conn.execute(
                """
                UPDATE stage_runs
                SET provider_pod_id = NULL,
                    updated_at = now()
                WHERE id = %s
                """,
                (stage_run_id,),
            )
    except Exception as exc:
        message = str(exc)
        if "HTTP 404" in message:
            with connect() as conn:
                create_event(
                    conn,
                    stage_run_id=stage_run_id,
                    kind="runpod_pod_already_absent",
                    message="RunPod pod was already absent during cleanup",
                    payload={"pod_id": pod_id, "reason": reason},
                )
                conn.execute(
                    """
                    UPDATE stage_runs
                    SET provider_pod_id = NULL,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (stage_run_id,),
                )
            return
        with connect() as conn:
            create_event(
                conn,
                stage_run_id=stage_run_id,
                kind="runpod_pod_delete_failed",
                level="warning",
                message=message,
                payload={"pod_id": pod_id, "reason": reason},
            )


def provider_pod_status(client: RunpodClient, pod_id: str) -> Optional[str]:
    try:
        pod = client.get_pod(pod_id)
    except Exception:
        return None
    for key in ("desiredStatus", "status", "lastStatusChange"):
        value = pod.get(key)
        if value:
            return str(value)
    return None


def load_optional_json_from_r2(uri: str) -> dict[str, Any]:
    try:
        return load_json_from_r2(uri)
    except Exception:
        return {}


def compact_colmap_summary(
    stage_result: dict[str, Any],
    reconstruction_report: dict[str, Any],
    *,
    provider_job_id: str,
) -> dict[str, Any]:
    metadata = stage_result.get("metadata") if isinstance(stage_result.get("metadata"), dict) else {}
    wrapper_summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    metrics = reconstruction_report.get("reconstruction_metrics") if isinstance(reconstruction_report.get("reconstruction_metrics"), dict) else {}
    settings = reconstruction_report.get("settings") if isinstance(reconstruction_report.get("settings"), dict) else {}
    return {
        "provider": "runpod_colmap",
        "provider_job_id": provider_job_id,
        "stage": "colmap",
        "status": stage_result.get("status"),
        "mode": wrapper_summary.get("mode") or settings.get("mode"),
        "matcher": wrapper_summary.get("matcher") or settings.get("matcher"),
        "image_count": wrapper_summary.get("image_count") or ((reconstruction_report.get("input") or {}).get("image_count") if isinstance(reconstruction_report.get("input"), dict) else None),
        "selected_sparse_model": wrapper_summary.get("selected_sparse_model") or reconstruction_report.get("selected_sparse_model"),
        "registered_images": metrics.get("registered_images"),
        "registered_frames": metrics.get("registered_frames"),
        "point_count": metrics.get("points"),
        "mean_track_length": metrics.get("mean_track_length"),
        "mean_observations_per_image": metrics.get("mean_observations_per_image"),
        "mean_reprojection_error_px": metrics.get("mean_reprojection_error_px"),
        "stage_result_uri": f"{stage_result.get('output_uris', [''])[0].rstrip('/')}/stage_result.json" if stage_result.get("output_uris") else None,
        "reconstruction_report_uri": stage_result.get("metrics_uri"),
    }


def record_progress(stage_run_id: str, percent: int, message: str, *, kind: str) -> None:
    with connect() as conn:
        create_event(
            conn,
            stage_run_id=stage_run_id,
            kind=kind,
            message=message,
            payload={"percent": percent},
        )
        conn.execute(
            """
            UPDATE stage_runs
            SET progress_json = %s,
                updated_at = now()
            WHERE id = %s
            """,
            (Jsonb({"percent": percent, "message": message}), stage_run_id),
        )


def load_preprocess_summary(output_base_uri: str) -> dict[str, Any]:
    if output_base_uri.startswith("r2://"):
        return load_json_from_r2(f"{output_base_uri.rstrip('/')}/current/preprocess_summary.json")
    current_path = local_path_from_uri(output_base_uri)
    if current_path is None:
        return {}
    summary_path = current_path / "current" / "preprocess_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_json_from_r2(uri: str) -> dict[str, Any]:
    command = aws_cp_stdout_command(uri)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if not completed.stdout.strip():
        return {}
    return json.loads(completed.stdout)


def aws_cp_stdout_command(uri: str) -> list[str]:
    parsed = urlparse(uri)
    if parsed.scheme != "r2" or not parsed.netloc:
        raise ValueError(f"Expected r2:// URI: {uri}")
    command = ["aws"]
    endpoint_url = r2_endpoint()
    if endpoint_url:
        command.extend(["--endpoint-url", endpoint_url])
    command.extend(["s3", "cp", f"s3://{parsed.netloc}{parsed.path}", "-"])
    return command


def require_r2_uri(value: str, field_name: str) -> None:
    if not value.startswith("r2://"):
        raise ValueError(f"local_preprocess {field_name} must be an r2:// URI")


def local_path_from_uri(value: str) -> Optional[Path]:
    parsed = urlparse(value)
    if parsed.scheme in ("", "local"):
        return Path(parsed.path or parsed.netloc).expanduser()
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser()
    return None


def fake_output_uri(stage_run: dict[str, Any]) -> str:
    return f"fake://projects/{stage_run['project_id']}/{stage_run['stage']}/current"


def run_forever(*, worker_id: str, poll_seconds: float) -> None:
    print(f"Starting controller worker {worker_id}", flush=True)
    while not STOP_REQUESTED:
        did_work = run_once(worker_id=worker_id)
        if not did_work:
            time.sleep(poll_seconds)
    print("Controller worker stopped.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Buildvision3D controller worker.")
    parser.add_argument("--once", action="store_true", help="Claim and process one queued stage, then exit.")
    parser.add_argument("--controller-id", default=controller_id(), help="Identifier written into claimed_by.")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=poll_interval_seconds(),
        help="Seconds to sleep when no queued stage is available.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    ensure_schema()
    cleanup_runpod_colmap_pods(worker_id=args.controller_id)
    if args.once:
        did_work = run_once(worker_id=args.controller_id)
        if not did_work:
            print("No queued stages found.", flush=True)
        return

    run_forever(worker_id=args.controller_id, poll_seconds=max(0.1, args.poll_interval_seconds))
