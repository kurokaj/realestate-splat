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
    runpod_training_cloud_type,
    runpod_training_container_disk_gb,
    runpod_training_gpu_types,
    runpod_training_image,
    runpod_training_poll_seconds,
    runpod_training_timeout_seconds,
    stage_python_bin,
)
from controller_common.db import claim_next_queued_stage, complete_stage_run, connect, create_event, ensure_schema
from controller_common.fake_provider import FakeProvider
from controller_common.runpod_gpus import normalize_gpu_types
from controller_common.runpod_provider import RunpodClient


STOP_REQUESTED = False
ABSENT_POD_STATUS = "ABSENT"
TERMINAL_POD_STATUSES = {ABSENT_POD_STATUS, "DELETED", "EXITED", "FAILED", "STOPPED", "TERMINATED"}


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
        elif stage_run["provider"] == "runpod_training" and stage_run["stage"] == "training":
            summary, output_uri = run_runpod_training(stage_run)
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
        if stage_run["provider"] in {"runpod_colmap", "runpod_training"}:
            cleanup_stage_multipart_uploads(stage_run)
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
    output_tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        clean_line = line.rstrip()
        if clean_line:
            output_line_count += 1
            output_tail.append(clean_line)
            if len(output_tail) > 40:
                output_tail.pop(0)
    return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(output_tail)
        detail = f"\nLast preprocess output:\n{tail}" if tail else ""
        raise RuntimeError(f"Local preprocess failed with exit code {return_code}.{detail}")

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
    # Always use the manifest-driven group runner. An empty list means "use
    # profile defaults for every discovered group", which also keeps older
    # queued runs compatible with the new grouped raw layout.
    command.extend(["--group-config-json", json.dumps(inputs.get("group_configs", []), separators=(",", ":"))])
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
    inputs = dict(stage_run["input_uri_json"] or {})
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
                "feature_extractor": inputs.get("feature_extractor", "SIFT"),
                "matcher": inputs.get("matcher", "exhaustive"),
                "matching_type": inputs.get("matching_type", "SIFT_BRUTEFORCE"),
                "camera_model": inputs.get("camera_model", "SIMPLE_RADIAL"),
                "sequential_loop_detection": inputs.get("sequential_loop_detection", True),
                "vocab_tree": inputs.get("vocab_tree"),
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


def run_runpod_training(stage_run: dict[str, Any]) -> tuple[dict[str, Any], str]:
    stage_run_id = stage_run["id"]
    inputs = stage_run["input_uri_json"] or {}
    preprocess_uri = inputs.get("preprocess_uri")
    colmap_uri = inputs.get("colmap_uri")
    output_base_uri = inputs.get("output_uri")
    if not preprocess_uri:
        raise ValueError("runpod_training requires input_uri_json.preprocess_uri")
    if not colmap_uri:
        raise ValueError("runpod_training requires input_uri_json.colmap_uri")
    if not output_base_uri:
        raise ValueError("runpod_training requires input_uri_json.output_uri")
    require_r2_uri(preprocess_uri, "preprocess_uri")
    require_r2_uri(colmap_uri, "colmap_uri")
    require_r2_uri(output_base_uri, "output_uri")

    remote_command = build_training_stage_shell_command(stage_run, inputs)
    current_uri = f"{output_base_uri.rstrip('/')}/current"
    image = stage_run.get("image") or inputs.get("image") or runpod_training_image()
    with connect() as conn:
        create_event(
            conn,
            stage_run_id=stage_run_id,
            kind="runpod_training_prepared",
            message="Prepared RunPod training pod command",
            payload={
                "image": image,
                "gpu_type_ids": inputs.get("gpu_type_ids") or runpod_training_gpu_types(),
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
                Jsonb({"percent": 10, "message": "RunPod training command prepared"}),
                stage_run_id,
            ),
        )

    if inputs.get("dry_run"):
        return (
            {
                "provider": "runpod_training",
                "stage": "training",
                "dry_run": True,
                "image": image,
                "method": inputs.get("method", "splatfacto"),
                "max_steps": inputs.get("max_steps", 100),
                "stage_result_uri": f"{current_uri}/stage_result.json",
                "training_summary_uri": f"{current_uri}/training_summary.json",
            },
            current_uri,
        )

    pod_payload = build_runpod_training_pod_payload(stage_run, inputs, image=image, remote_command=remote_command)
    client = RunpodClient()
    pod = client.create_pod(pod_payload)
    with connect() as conn:
        create_event(
            conn,
            stage_run_id=stage_run_id,
            kind="runpod_pod_created",
            message="Created RunPod training pod",
            payload={"pod_id": pod.id, "image": image},
        )
        conn.execute(
            """
            UPDATE stage_runs
            SET provider_job_id = %s,
                provider_pod_id = %s,
                status = 'training_pod_starting',
                progress_json = %s,
                updated_at = now()
            WHERE id = %s
            """,
            (pod.id, pod.id, Jsonb({"percent": 20, "message": "RunPod training pod created"}), stage_run_id),
        )

    keep_pod = bool(inputs.get("keep_pod"))
    try:
        stage_result = wait_for_runpod_stage_result(
            stage_run_id,
            client,
            pod.id,
            output_base_uri,
            stage="training",
            poll_seconds=runpod_training_poll_seconds(),
            timeout_seconds=runpod_training_timeout_seconds(),
        )
        training_summary = load_optional_json_from_r2(f"{current_uri}/training_summary.json")
        return compact_training_summary(stage_result, training_summary, provider_job_id=pod.id), current_uri
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
        "--feature-extractor",
        inputs.get("feature_extractor", "SIFT"),
        "--matcher",
        inputs.get("matcher", "exhaustive"),
        "--matching-type",
        inputs.get("matching_type", "SIFT_BRUTEFORCE"),
        "--camera-model",
        inputs.get("camera_model", "SIMPLE_RADIAL"),
        "--max-image-size",
        str(inputs.get("max_image_size", 3200)),
        "--sequential-loop-detection" if inputs.get("sequential_loop_detection", True) else "--no-sequential-loop-detection",
        "--colmap-bin",
        inputs.get("colmap_bin", "/opt/colmap-cuda/bin/colmap"),
    ]
    if inputs.get("raw_uri"):
        command.extend(["--raw-uri", inputs["raw_uri"]])
    for group_output in inputs.get("preprocess_group_outputs", []):
        if isinstance(group_output, dict):
            command.extend(["--preprocess-group-output", json.dumps(group_output, separators=(",", ":"))])
    if inputs.get("vocab_tree"):
        command.extend(["--vocab-tree", inputs["vocab_tree"]])
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


def build_training_stage_shell_command(stage_run: dict[str, Any], inputs: dict[str, Any]) -> str:
    repo_url = inputs.get("repo_url") or controller_repo_url()
    if not repo_url:
        raise ValueError("CONTROLLER_REPO_URL or input_uri_json.repo_url is required for runpod_training")
    git_ref = inputs.get("git_ref") or controller_git_ref()
    output_uri = inputs["output_uri"].rstrip("/")
    command = [
        "python3",
        "scripts/run_training_stage.py",
        "--project-id",
        stage_run["project_id"],
        "--stage-run-id",
        stage_run["id"],
        "--preprocess-uri",
        inputs["preprocess_uri"],
        "--colmap-uri",
        inputs["colmap_uri"],
        "--output-uri",
        output_uri,
        "--method",
        inputs.get("method", "splatfacto"),
        "--max-steps",
        str(inputs.get("max_steps", 100)),
        "--save-every",
        str(inputs.get("save_every", 50)),
        "--eval-every",
        str(inputs.get("eval_every", 50)),
        "--num-downscales",
        str(inputs.get("num_downscales", 1)),
    ]
    if inputs.get("export", True):
        command.append("--export")
    else:
        command.append("--no-export")
    splatfacto_options = inputs.get("splatfacto_options")
    if isinstance(splatfacto_options, dict) and "use_scale_regularization" in splatfacto_options:
        command.append("--use-scale-regularization" if splatfacto_options.get("use_scale_regularization") else "--no-use-scale-regularization")
    for train_arg in inputs.get("train_options", []):
        append_argparse_value(command, "--train-option", train_arg)

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

    raw_gpu_type_ids = inputs.get("gpu_type_ids") or runpod_colmap_gpu_types()
    if isinstance(raw_gpu_type_ids, str):
        raw_gpu_type_ids = [raw_gpu_type_ids]
    gpu_type_ids = normalize_gpu_types(raw_gpu_type_ids)
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


def build_runpod_training_pod_payload(
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

    raw_gpu_type_ids = inputs.get("gpu_type_ids") or runpod_training_gpu_types()
    if isinstance(raw_gpu_type_ids, str):
        raw_gpu_type_ids = [raw_gpu_type_ids]
    gpu_type_ids = normalize_gpu_types(raw_gpu_type_ids)
    return {
        "name": f"buildvision3d-training-{stage_run['project_id']}-{stage_run['id']}"[:80],
        "imageName": image,
        "computeType": "GPU",
        "cloudType": inputs.get("cloud_type") or runpod_training_cloud_type(),
        "gpuCount": int(inputs.get("gpu_count") or 1),
        "gpuTypeIds": gpu_type_ids,
        "gpuTypePriority": inputs.get("gpu_type_priority") or "availability",
        "containerDiskInGb": int(inputs.get("container_disk_gb") or runpod_training_container_disk_gb()),
        "minVCPUPerGPU": int(inputs.get("min_vcpu_per_gpu") or 8),
        "minRAMPerGPU": int(inputs.get("min_ram_per_gpu") or 32),
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
    return wait_for_runpod_stage_result(
        stage_run_id,
        client,
        pod_id,
        output_base_uri,
        stage="colmap",
        poll_seconds=runpod_colmap_poll_seconds(),
        timeout_seconds=runpod_colmap_timeout_seconds(),
    )


def wait_for_runpod_stage_result(
    stage_run_id: str,
    client: RunpodClient,
    pod_id: str,
    output_base_uri: str,
    *,
    stage: str,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    result_uri = f"{output_base_uri.rstrip('/')}/current/stage_result.json"
    upload_complete_uri = f"{output_base_uri.rstrip('/')}/current/upload_complete.json"
    last_pod_status = None
    stale_result_run_id = None
    stale_upload_run_id = None
    running_status = f"{stage}_running"
    stage_label = stage.upper()
    while True:
        if stage_was_cancelled(stage_run_id):
            raise RuntimeError(f"Stage was cancelled while RunPod {stage_label} was running")
        stage_result = load_optional_json_from_r2(result_uri)
        if stage_result:
            result_stage_run_id = stage_result.get("stage_run_id")
            if result_stage_run_id and result_stage_run_id != stage_run_id:
                if stale_result_run_id != result_stage_run_id:
                    stale_result_run_id = str(result_stage_run_id)
                    record_progress(
                        stage_run_id,
                        25,
                        f"Ignoring stale {stage_label} stage_result.json from {result_stage_run_id}",
                        kind=f"runpod_{stage}_stale_stage_result",
                    )
                stage_result = {}
            else:
                stale_result_run_id = None
        if stage_result:
            status = stage_result.get("status")
            record_progress(stage_run_id, 95, f"Found {stage_label} stage_result.json with status {status}", kind=f"runpod_{stage}_stage_result")
            if status == "completed":
                upload_complete = load_optional_json_from_r2(upload_complete_uri)
                if upload_complete:
                    upload_run_id = upload_complete.get("stage_run_id")
                    if upload_run_id and upload_run_id != stage_run_id:
                        if stale_upload_run_id != upload_run_id:
                            stale_upload_run_id = str(upload_run_id)
                            record_progress(
                                stage_run_id,
                                96,
                                f"Ignoring stale {stage_label} upload_complete.json from {upload_run_id}",
                                kind=f"runpod_{stage}_stale_upload_complete",
                            )
                        upload_complete = {}
                    else:
                        stale_upload_run_id = None
                if upload_complete:
                    required_objects = upload_complete.get("required_objects") if isinstance(upload_complete.get("required_objects"), list) else []
                    missing_objects = missing_required_objects(output_base_uri, required_objects)
                    if missing_objects:
                        record_progress(
                            stage_run_id,
                            97,
                            f"Waiting for {len(missing_objects)} required {stage_label} object(s)",
                            kind=f"runpod_{stage}_waiting_required_objects",
                        )
                        time.sleep(poll_seconds)
                        continue
                    record_progress(stage_run_id, 100, f"Found {stage_label} upload_complete.json", kind=f"runpod_{stage}_upload_complete")
                    return stage_result
                record_progress(
                    stage_run_id,
                    96,
                    f"Waiting for {stage_label} upload_complete.json after completed stage_result",
                    kind=f"runpod_{stage}_waiting_upload_complete",
                )
                time.sleep(poll_seconds)
                continue
            raise RuntimeError(stage_result.get("error_message") or f"{stage_label} stage_result status was {status}")

        pod_status = provider_pod_status(client, pod_id)
        if pod_status != last_pod_status:
            last_pod_status = pod_status
            with connect() as conn:
                create_event(
                    conn,
                    stage_run_id=stage_run_id,
                    kind="runpod_pod_status",
                    message=f"RunPod pod status: {pod_status or 'unknown'}",
                    payload={"pod_id": pod_id, "status": pod_status, "stage": stage},
                )
                conn.execute(
                    """
                    UPDATE stage_runs
                    SET status = %s,
                        progress_json = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (
                        running_status,
                        Jsonb({"percent": 50, "message": f"Waiting for R2 stage_result.json ({pod_status or 'pod status unknown'})"}),
                        stage_run_id,
                    ),
                )
        if pod_status and pod_status.upper() in TERMINAL_POD_STATUSES:
            raise RuntimeError(f"RunPod {stage_label} pod ended before publishing stage_result.json: {pod_status}")

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

    timeout_seconds = max(runpod_colmap_timeout_seconds(), runpod_training_timeout_seconds())
    active_statuses = ["colmap_pod_starting", "colmap_running", "training_pod_starting", "training_running"]
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM stage_runs
            WHERE provider IN ('runpod_colmap', 'runpod_training')
              AND provider_pod_id IS NOT NULL
              AND (
                    status = ANY(%s)
                    OR status = ANY(%s)
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
                    "awaiting_training_approval",
                ],
                active_statuses,
            ),
        ).fetchall()

    if not rows:
        return
    print(f"RunPod watchdog found {len(rows)} pod(s) to clean up.", flush=True)
    for row in rows:
        stage_run_id = row["id"]
        pod_id = row["provider_pod_id"]
        status = row["status"]
        active = status in active_statuses
        reference_time = row.get("updated_at") or row.get("started_at") or row.get("claimed_at") or row.get("created_at")
        timed_out = bool(active and reference_time and (time.time() - reference_time.timestamp()) > timeout_seconds)
        pod_status = provider_pod_status(client, pod_id) if active and not timed_out else None
        pod_absent_or_terminal = bool(pod_status and pod_status.upper() in TERMINAL_POD_STATUSES)
        if active and not timed_out and not pod_absent_or_terminal:
            continue
        if timed_out:
            with connect() as conn:
                create_event(
                    conn,
                    stage_run_id=stage_run_id,
                    kind="runpod_watchdog_timeout",
                    level="warning",
                    message=f"RunPod {row['stage']} stage exceeded watchdog timeout",
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
            cleanup_stage_multipart_uploads(row)
        elif pod_absent_or_terminal:
            with connect() as conn:
                create_event(
                    conn,
                    stage_run_id=stage_run_id,
                    kind="runpod_watchdog_pod_terminal",
                    level="warning",
                    message=f"RunPod {row['stage']} pod is {pod_status} before stage completion",
                    payload={"pod_id": pod_id, "worker_id": worker_id, "pod_status": pod_status},
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
                    (f"RunPod pod ended before stage completion: {pod_status}", stage_run_id),
                )
            cleanup_stage_multipart_uploads(row)
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


def cleanup_stage_multipart_uploads(stage_run: dict[str, Any]) -> None:
    inputs = stage_run.get("input_uri_json") or {}
    output_base_uri = inputs.get("output_uri")
    if not output_base_uri or not str(output_base_uri).startswith("r2://"):
        return
    prefixes = [
        f"{output_base_uri.rstrip('/')}/current",
        f"{output_base_uri.rstrip('/')}/runs/{stage_run['id']}",
    ]
    for prefix in prefixes:
        try:
            aborted = abort_multipart_uploads_for_prefix(prefix)
            if aborted:
                with connect() as conn:
                    create_event(
                        conn,
                        stage_run_id=stage_run["id"],
                        kind="multipart_uploads_aborted",
                        level="warning",
                        message=f"Aborted {aborted} dangling multipart upload(s)",
                        payload={"prefix": prefix, "aborted": aborted},
                    )
        except Exception as exc:
            with connect() as conn:
                create_event(
                    conn,
                    stage_run_id=stage_run["id"],
                    kind="multipart_upload_cleanup_failed",
                    level="warning",
                    message=str(exc),
                    payload={"prefix": prefix},
                )


def provider_pod_status(client: RunpodClient, pod_id: str) -> Optional[str]:
    try:
        pod = client.get_pod(pod_id)
    except Exception as exc:
        if "HTTP 404" in str(exc):
            return ABSENT_POD_STATUS
        return None
    for key in ("desiredStatus", "status"):
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
        "sequential_loop_detection": wrapper_summary.get("sequential_loop_detection") if wrapper_summary.get("sequential_loop_detection") is not None else settings.get("sequential_loop_detection"),
        "vocab_tree": wrapper_summary.get("vocab_tree") or settings.get("vocab_tree"),
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


def compact_training_summary(
    stage_result: dict[str, Any],
    training_summary: dict[str, Any],
    *,
    provider_job_id: str,
) -> dict[str, Any]:
    metadata = stage_result.get("metadata") if isinstance(stage_result.get("metadata"), dict) else {}
    wrapper_summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    return {
        "provider": "runpod_training",
        "provider_job_id": provider_job_id,
        "stage": "training",
        "status": stage_result.get("status"),
        "method": wrapper_summary.get("method") or training_summary.get("method"),
        "selected_config": training_summary.get("selected_config"),
        "checkpoint_count": training_summary.get("checkpoint_count"),
        "latest_checkpoint": training_summary.get("latest_checkpoint"),
        "exported_ply": training_summary.get("exported_ply"),
        "training_diagnostics": compact_training_diagnostics(training_summary),
        "training_summary_uri": stage_result.get("metrics_uri"),
        "stage_result_uri": f"{stage_result.get('output_uris', [''])[0].rstrip('/')}/stage_result.json" if stage_result.get("output_uris") else None,
    }


def compact_training_diagnostics(training_summary: dict[str, Any]) -> dict[str, Any]:
    init_stats = training_summary.get("colmap_init_stats") if isinstance(training_summary.get("colmap_init_stats"), dict) else {}
    gaussian_stats = training_summary.get("gaussian_diagnostics") if isinstance(training_summary.get("gaussian_diagnostics"), dict) else {}
    return {
        "colmap_init_point_count": training_summary.get("colmap_init_point_count"),
        "colmap_init_xyz_min": init_stats.get("xyz_min"),
        "colmap_init_xyz_max": init_stats.get("xyz_max"),
        "colmap_init_error_median": init_stats.get("reprojection_error_median"),
        "checkpoint_count": training_summary.get("checkpoint_count"),
        "exported_ply_vertices": training_summary.get("exported_ply_vertices"),
        "oversized_gaussian_detected": gaussian_stats.get("oversized_gaussian_detected"),
        "oversized_gaussian_count": gaussian_stats.get("oversized_gaussian_count"),
        "oversized_gaussian_ratio_max": gaussian_stats.get("oversized_gaussian_ratio_max"),
        "gaussian_scale_p95": gaussian_stats.get("scale_exp_max_axis_p95"),
        "gaussian_scale_p99": gaussian_stats.get("scale_exp_max_axis_p99"),
        "gaussian_scale_max": gaussian_stats.get("scale_exp_max_axis_max"),
        "gaussian_anisotropy_p99": gaussian_stats.get("anisotropy_p99"),
        "gaussian_anisotropy_max": gaussian_stats.get("anisotropy_max"),
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


def abort_multipart_uploads_for_prefix(uri: str) -> int:
    parsed = urlparse(uri)
    if parsed.scheme != "r2" or not parsed.netloc:
        raise ValueError(f"Expected r2:// URI: {uri}")
    command = ["aws"]
    endpoint_url = r2_endpoint()
    if endpoint_url:
        command.extend(["--endpoint-url", endpoint_url])
    command.extend(
        [
            "s3api",
            "list-multipart-uploads",
            "--bucket",
            parsed.netloc,
            "--prefix",
            parsed.path.lstrip("/"),
        ]
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout or "{}")
    uploads = payload.get("Uploads") or []
    aborted = 0
    for upload in uploads:
        key = upload.get("Key")
        upload_id = upload.get("UploadId")
        if not key or not upload_id:
            continue
        abort_command = ["aws"]
        if endpoint_url:
            abort_command.extend(["--endpoint-url", endpoint_url])
        abort_command.extend(
            [
                "s3api",
                "abort-multipart-upload",
                "--bucket",
                parsed.netloc,
                "--key",
                str(key),
                "--upload-id",
                str(upload_id),
            ]
        )
        subprocess.run(abort_command, check=True)
        aborted += 1
    return aborted


def missing_required_objects(output_base_uri: str, required_objects: list[str]) -> list[str]:
    missing: list[str] = []
    for relative_path in required_objects:
        uri = f"{output_base_uri.rstrip('/')}/current/{relative_path.lstrip('/')}"
        if not object_exists_in_r2(uri):
            missing.append(relative_path)
    return missing


def object_exists_in_r2(uri: str) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme != "r2" or not parsed.netloc:
        raise ValueError(f"Expected r2:// URI: {uri}")
    command = ["aws"]
    endpoint_url = r2_endpoint()
    if endpoint_url:
        command.extend(["--endpoint-url", endpoint_url])
    command.extend(
        [
            "s3api",
            "head-object",
            "--bucket",
            parsed.netloc,
            "--key",
            parsed.path.lstrip("/"),
        ]
    )
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.returncode == 0


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
    last_watchdog_at = 0.0
    while not STOP_REQUESTED:
        did_work = run_once(worker_id=worker_id)
        if not did_work and time.monotonic() - last_watchdog_at > 30:
            cleanup_runpod_colmap_pods(worker_id=worker_id)
            last_watchdog_at = time.monotonic()
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
