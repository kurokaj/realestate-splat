"""Terminal-run worker-controller for Milestone 8A."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import signal
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from psycopg.types.json import Jsonb

from controller_common.config import controller_id, poll_interval_seconds, r2_endpoint, stage_python_bin
from controller_common.db import claim_next_queued_stage, complete_stage_run, connect, create_event, ensure_schema
from controller_common.fake_provider import FakeProvider


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
        command.extend(["--preprocess-arg", preprocess_arg])
    return command


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
    if args.once:
        did_work = run_once(worker_id=args.controller_id)
        if not did_work:
            print("No queued stages found.", flush=True)
        return

    run_forever(worker_id=args.controller_id, poll_seconds=max(0.1, args.poll_interval_seconds))
