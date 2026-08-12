"""Server-rendered internal controller UI."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from controller_common.config import default_colmap_provider, default_r2_bucket, default_training_provider
from controller_common.db import (
    cancel_stage_run,
    connect,
    create_stage_run,
    enqueue_next_stage_after_approval,
    reject_stage_run,
    retry_stage_run,
    row_to_json,
    rows_to_json,
)
from controller_common.runpod_gpus import (
    COLMAP_GPU_OPTIONS,
    DEFAULT_COLMAP_GPU,
    DEFAULT_TRAINING_GPU,
    TRAINING_GPU_OPTIONS,
    normalize_gpu_name,
)
from src.realestate_splat.storage import copy_file, parse_storage_uri
from scripts.preprocess_video import PROFILE_DEFAULTS


router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).with_name("templates")))


def json_pretty(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    return json.dumps(value, indent=2, sort_keys=True)


templates.env.filters["json_pretty"] = json_pretty


@router.get("/", response_class=HTMLResponse)
def ui_root() -> RedirectResponse:
    return RedirectResponse(url="/ui/projects", status_code=303)


@router.get("/projects", response_class=HTMLResponse)
def projects_index(request: Request) -> HTMLResponse:
    with connect() as conn:
        projects = rows_to_json(
            conn.execute(
                """
                SELECT project.*,
                    count(stage_run.id) AS stage_run_count,
                    max(stage_run.created_at) AS latest_stage_run_at
                FROM projects AS project
                LEFT JOIN stage_runs AS stage_run ON stage_run.project_id = project.id
                GROUP BY project.id
                ORDER BY project.updated_at DESC
                """
            ).fetchall()
        )
    return templates.TemplateResponse(
        request,
        "projects.html",
        {"projects": projects, "default_bucket": default_r2_bucket()},
    )


@router.post("/projects")
def ui_create_project(
    project_id: str = Form(...),
    name: str = Form(...),
    raw_uri: Optional[str] = Form(default=None),
) -> RedirectResponse:
    cleaned_project_id = project_id.strip()
    cleaned_name = name.strip()
    if not cleaned_project_id:
        raise HTTPException(status_code=400, detail="Project id is required")
    if not cleaned_name:
        raise HTTPException(status_code=400, detail="Project name is required")
    cleaned_raw_uri = empty_to_none(raw_uri)
    if cleaned_raw_uri:
        require_r2_uri(cleaned_raw_uri, "raw_uri")
    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO projects (id, name, status, raw_uri)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (cleaned_project_id, cleaned_name, "raw_uploaded" if cleaned_raw_uri else "created", cleaned_raw_uri),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create project")
    return RedirectResponse(url=f"/ui/projects/{cleaned_project_id}", status_code=303)


@router.get("/projects/{project_id}/colmap-viewer")
def project_colmap_viewer(project_id: str) -> JSONResponse:
    with connect() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    colmap_current_uri = project.get("colmap_current_uri")
    if not colmap_current_uri:
        raise HTTPException(status_code=404, detail="COLMAP current output is not available yet")
    payload = load_json_uri(f"{colmap_current_uri.rstrip('/')}/viewer/sparse_scene.json")
    if not payload:
        raise HTTPException(status_code=404, detail="COLMAP viewer artifact is not available yet")
    return JSONResponse(payload)


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_detail(request: Request, project_id: str) -> HTMLResponse:
    data = load_project_detail(project_id)
    if data["project"] is None:
        raise HTTPException(status_code=404, detail="Project not found")
    review = preprocess_review_context(data["project"], data["stage_runs"])
    colmap_review = colmap_review_context(data["project"], data["stage_runs"])
    training_review = training_review_context(data["project"], data["stage_runs"])
    return templates.TemplateResponse(
        request,
        "project_detail.html",
        {
            **data,
            **review,
            **colmap_review,
            **training_review,
            "default_bucket": default_r2_bucket(),
            "default_raw_uri": f"r2://{default_r2_bucket()}/projects/{project_id}/raw",
            "default_preprocess_uri": f"r2://{default_r2_bucket()}/projects/{project_id}/preprocess",
        },
    )


@router.post("/projects/{project_id}/preprocess")
def ui_queue_preprocess(
    project_id: str,
    raw_uri: Optional[str] = Form(default=None),
    output_uri: Optional[str] = Form(default=None),
    endpoint_url: Optional[str] = Form(default=None),
    profile: str = Form(default="indoor_room"),
    candidate_fps: Optional[float] = Form(default=None),
    target_min: Optional[int] = Form(default=None),
    target_max: Optional[int] = Form(default=None),
    min_blur: Optional[float] = Form(default=None),
    min_brightness: Optional[float] = Form(default=None),
    max_brightness: Optional[float] = Form(default=None),
    min_contrast: Optional[float] = Form(default=None),
    min_entropy: Optional[float] = Form(default=None),
    force_keep_interval: Optional[float] = Form(default=None),
    coverage_window_seconds: Optional[float] = Form(default=None),
    min_frames_per_window: Optional[int] = Form(default=None),
    coverage_hard_min_blur: Optional[float] = Form(default=None),
    coverage_hard_min_brightness: Optional[float] = Form(default=None),
    coverage_hard_max_brightness: Optional[float] = Form(default=None),
    coverage_hard_min_contrast: Optional[float] = Form(default=None),
    coverage_hard_min_entropy: Optional[float] = Form(default=None),
    dry_run: bool = Form(default=False),
) -> RedirectResponse:
    with connect() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        resolved_raw_uri = empty_to_none(raw_uri) or project.get("raw_uri")
        if not resolved_raw_uri:
            raise HTTPException(status_code=400, detail="Project raw_uri is required to queue preprocessing")
        resolved_output_uri = empty_to_none(output_uri) or f"r2://{default_r2_bucket()}/projects/{project_id}/preprocess"
        require_r2_uri(resolved_raw_uri, "raw_uri")
        require_r2_uri(resolved_output_uri, "output_uri")
        input_uri_json = {
            "raw_uri": resolved_raw_uri,
            "output_uri": resolved_output_uri,
            "endpoint_url": empty_to_none(endpoint_url),
            "profile": profile,
            "preprocess_args": preprocess_args_from_form(
                {
                    "candidate_fps": candidate_fps,
                    "target_min": target_min,
                    "target_max": target_max,
                    "min_blur": min_blur,
                    "min_brightness": min_brightness,
                    "max_brightness": max_brightness,
                    "min_contrast": min_contrast,
                    "min_entropy": min_entropy,
                    "force_keep_interval": force_keep_interval,
                    "coverage_window_seconds": coverage_window_seconds,
                    "min_frames_per_window": min_frames_per_window,
                    "coverage_hard_min_blur": coverage_hard_min_blur,
                    "coverage_hard_min_brightness": coverage_hard_min_brightness,
                    "coverage_hard_max_brightness": coverage_hard_max_brightness,
                    "coverage_hard_min_contrast": coverage_hard_min_contrast,
                    "coverage_hard_min_entropy": coverage_hard_min_entropy,
                }
            ),
            "dry_run": dry_run,
        }
        create_stage_run(
            conn,
            project_id=project_id,
            stage="preprocess",
            provider="local_preprocess",
            input_uri_json={key: value for key, value in input_uri_json.items() if value not in (None, [])},
            output_uri=f"{resolved_output_uri.rstrip('/')}/current",
        )
    return RedirectResponse(url=f"/ui/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/colmap")
def ui_queue_colmap(
    project_id: str,
    preprocess_uri: Optional[str] = Form(default=None),
    output_uri: Optional[str] = Form(default=None),
    endpoint_url: Optional[str] = Form(default=None),
    mode: str = Form(default="global"),
    matcher: str = Form(default="exhaustive"),
    camera_model: str = Form(default="SIMPLE_RADIAL"),
    provider: str = Form(default=default_colmap_provider()),
    image: Optional[str] = Form(default=None),
    repo_url: Optional[str] = Form(default=None),
    git_ref: Optional[str] = Form(default=None),
    gpu_type_id: str = Form(default=DEFAULT_COLMAP_GPU),
    dry_run: bool = Form(default=False),
) -> RedirectResponse:
    with connect() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        latest_preprocess = latest_stage_run(conn, project_id, "preprocess")
        if not latest_preprocess or latest_preprocess.get("status") != "approved":
            raise HTTPException(status_code=400, detail="Preprocess must be approved before COLMAP can be queued")
        resolved_preprocess_uri = empty_to_none(preprocess_uri) or project.get("preprocess_current_uri")
        if not resolved_preprocess_uri:
            raise HTTPException(status_code=400, detail="Project preprocess_current_uri is required to queue COLMAP")
        resolved_output_uri = empty_to_none(output_uri) or f"r2://{default_r2_bucket()}/projects/{project_id}/colmap"
        require_r2_uri(resolved_preprocess_uri, "preprocess_uri")
        require_r2_uri(resolved_output_uri, "output_uri")
        input_uri_json = {
            "preprocess_uri": resolved_preprocess_uri,
            "output_uri": resolved_output_uri,
            "endpoint_url": empty_to_none(endpoint_url),
            "mode": mode,
            "matcher": matcher,
            "camera_model": camera_model,
            "repo_url": empty_to_none(repo_url),
            "git_ref": empty_to_none(git_ref),
            "gpu_type_ids": [normalize_gpu_name(gpu_type_id)],
            "dry_run": dry_run,
        }
        create_stage_run(
            conn,
            project_id=project_id,
            stage="colmap",
            provider=provider,
            image=empty_to_none(image),
            input_uri_json={key: value for key, value in input_uri_json.items() if value not in (None, [])},
            output_uri=f"{resolved_output_uri.rstrip('/')}/current",
        )
    return RedirectResponse(url=f"/ui/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/training")
def ui_queue_training(
    project_id: str,
    preprocess_uri: Optional[str] = Form(default=None),
    colmap_uri: Optional[str] = Form(default=None),
    output_uri: Optional[str] = Form(default=None),
    endpoint_url: Optional[str] = Form(default=None),
    method: str = Form(default="splatfacto"),
    max_steps: int = Form(default=100),
    save_every: int = Form(default=50),
    eval_every: int = Form(default=50),
    num_downscales: int = Form(default=1),
    checkpoint_export_mode: str = Form(default="all"),
    provider: str = Form(default=default_training_provider()),
    image: Optional[str] = Form(default=None),
    repo_url: Optional[str] = Form(default=None),
    git_ref: Optional[str] = Form(default=None),
    gpu_type_id: str = Form(default=DEFAULT_TRAINING_GPU),
    export: bool = Form(default=False),
    dry_run: bool = Form(default=False),
) -> RedirectResponse:
    with connect() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        latest_colmap = latest_stage_run(conn, project_id, "colmap")
        if not latest_colmap or latest_colmap.get("status") != "approved":
            raise HTTPException(status_code=400, detail="COLMAP must be approved before training can be queued")
        resolved_preprocess_uri = empty_to_none(preprocess_uri) or project.get("preprocess_current_uri")
        resolved_colmap_uri = empty_to_none(colmap_uri) or project.get("colmap_current_uri")
        resolved_output_uri = empty_to_none(output_uri) or f"r2://{default_r2_bucket()}/projects/{project_id}/training"
        if not resolved_preprocess_uri:
            raise HTTPException(status_code=400, detail="Project preprocess_current_uri is required to queue training")
        if not resolved_colmap_uri:
            raise HTTPException(status_code=400, detail="Project colmap_current_uri is required to queue training")
        require_r2_uri(resolved_preprocess_uri, "preprocess_uri")
        require_r2_uri(resolved_colmap_uri, "colmap_uri")
        require_r2_uri(resolved_output_uri, "output_uri")
        input_uri_json = {
            "preprocess_uri": resolved_preprocess_uri,
            "colmap_uri": resolved_colmap_uri,
            "output_uri": resolved_output_uri,
            "endpoint_url": empty_to_none(endpoint_url),
            "method": method,
            "max_steps": max_steps,
            "save_every": save_every,
            "eval_every": eval_every,
            "num_downscales": num_downscales,
            "checkpoint_export_mode": checkpoint_export_mode,
            "repo_url": empty_to_none(repo_url),
            "git_ref": empty_to_none(git_ref),
            "gpu_type_ids": [normalize_gpu_name(gpu_type_id)],
            "export": export,
            "dry_run": dry_run,
        }
        create_stage_run(
            conn,
            project_id=project_id,
            stage="training",
            provider=provider,
            image=empty_to_none(image),
            input_uri_json={key: value for key, value in input_uri_json.items() if value not in (None, [])},
            output_uri=f"{resolved_output_uri.rstrip('/')}/current",
        )
    return RedirectResponse(url=f"/ui/projects/{project_id}", status_code=303)


@router.post("/stage-runs/{stage_run_id}/approve")
def ui_approve_stage_run(
    stage_run_id: str,
    notes: Optional[str] = Form(default=None),
) -> RedirectResponse:
    return run_stage_action(stage_run_id, notes, "approve")


@router.post("/stage-runs/{stage_run_id}/reject")
def ui_reject_stage_run(
    stage_run_id: str,
    notes: Optional[str] = Form(default=None),
) -> RedirectResponse:
    return run_stage_action(stage_run_id, notes, "reject")


@router.post("/stage-runs/{stage_run_id}/retry")
def ui_retry_stage_run(
    stage_run_id: str,
    notes: Optional[str] = Form(default=None),
) -> RedirectResponse:
    return run_stage_action(stage_run_id, notes, "retry")


@router.post("/stage-runs/{stage_run_id}/cancel")
def ui_cancel_stage_run(
    stage_run_id: str,
    notes: Optional[str] = Form(default=None),
) -> RedirectResponse:
    return run_stage_action(stage_run_id, notes, "cancel")


def run_stage_action(stage_run_id: str, notes: Optional[str], action: str) -> RedirectResponse:
    with connect() as conn:
        stage_run = conn.execute("SELECT project_id FROM stage_runs WHERE id = %s", (stage_run_id,)).fetchone()
        if stage_run is None:
            raise HTTPException(status_code=404, detail="Stage run not found")
        project_id = stage_run["project_id"]
        with conn.transaction():
            try:
                if action == "approve":
                    enqueue_next_stage_after_approval(conn, stage_run_id=stage_run_id, notes=empty_to_none(notes))
                elif action == "reject":
                    reject_stage_run(conn, stage_run_id=stage_run_id, notes=empty_to_none(notes))
                elif action == "retry":
                    retry_stage_run(conn, stage_run_id=stage_run_id, notes=empty_to_none(notes))
                elif action == "cancel":
                    cancel_stage_run(conn, stage_run_id=stage_run_id, notes=empty_to_none(notes))
                else:
                    raise ValueError(f"Unsupported action: {action}")
            except LookupError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url=f"/ui/projects/{project_id}", status_code=303)


def load_project_detail(project_id: str) -> dict[str, Any]:
    with connect() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
        if project is None:
            return {"project": None, "stage_runs": [], "events": [], "approvals": []}
        stage_runs = conn.execute(
            """
            SELECT *
            FROM stage_runs
            WHERE project_id = %s
            ORDER BY created_at DESC
            """,
            (project_id,),
        ).fetchall()
        events = conn.execute(
            """
            SELECT event.*, stage_run.stage, stage_run.status AS stage_status
            FROM events AS event
            JOIN stage_runs AS stage_run ON stage_run.id = event.stage_run_id
            WHERE stage_run.project_id = %s
            ORDER BY event.created_at DESC
            LIMIT 80
            """,
            (project_id,),
        ).fetchall()
        approvals = conn.execute(
            """
            SELECT *
            FROM approvals
            WHERE project_id = %s
            ORDER BY created_at DESC
            """,
            (project_id,),
        ).fetchall()
    stage_run_json = rows_to_json(stage_runs)
    approval_json = rows_to_json(approvals)
    raw_stage_signature = stage_signature(stage_run_json)
    apply_historical_approval_status(stage_run_json, approval_json)
    apply_stage_history_labels(stage_run_json)
    apply_stage_durations(stage_run_json)
    project_json = row_to_json(project)
    project_json["display_status"] = derive_project_display_status(project_json, stage_run_json)
    return {
        "project": project_json,
        "stage_runs": stage_run_json,
        "stage_signature": raw_stage_signature,
        "events": rows_to_json(events),
        "approvals": approval_json,
    }


def latest_stage_run(conn: Any, project_id: str, stage: str) -> Optional[dict[str, Any]]:
    return conn.execute(
        """
        SELECT *
        FROM stage_runs
        WHERE project_id = %s AND stage = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (project_id, stage),
    ).fetchone()


def derive_project_display_status(project: dict[str, Any], stage_runs: list[dict[str, Any]]) -> str:
    latest_by_stage: dict[str, dict[str, Any]] = {}
    for run in stage_runs:
        stage = run.get("stage")
        if stage and stage not in latest_by_stage:
            latest_by_stage[stage] = run

    priority = ("training", "colmap", "preprocess")
    active_prefixes = ("_running", "_queued")
    active_statuses = {
        "approved",
        "awaiting_preprocess_approval",
        "awaiting_colmap_approval",
    }
    for stage in priority:
        run = latest_by_stage.get(stage)
        if not run:
            continue
        status = str(run.get("status") or "")
        if status.endswith(active_prefixes) or status in active_statuses:
            return status

    if stage_runs:
        return str(stage_runs[0].get("status") or project.get("status") or "created")
    return str(project.get("status") or "created")


def apply_historical_approval_status(stage_runs: list[dict[str, Any]], approvals: list[dict[str, Any]]) -> None:
    approved_run_ids = {
        approval.get("stage_run_id")
        for approval in approvals
        if approval.get("decision") == "approved" and approval.get("stage_run_id")
    }
    awaiting_statuses = {"awaiting_preprocess_approval", "awaiting_colmap_approval"}
    for run in stage_runs:
        if run.get("id") in approved_run_ids and run.get("status") in awaiting_statuses:
            run["stored_status"] = run["status"]
            run["status"] = "approved"


def apply_stage_history_labels(stage_runs: list[dict[str, Any]]) -> None:
    latest_by_stage: dict[str, str] = {}
    for run in stage_runs:
        stage = run.get("stage")
        run_id = run.get("id")
        if stage and run_id and stage not in latest_by_stage:
            latest_by_stage[stage] = run_id
    for run in stage_runs:
        is_latest = latest_by_stage.get(run.get("stage")) == run.get("id")
        run["is_latest_stage_run"] = is_latest
        if not is_latest:
            run["status_label"] = "history run"
            run["status_css"] = "history"


def apply_stage_durations(stage_runs: list[dict[str, Any]]) -> None:
    now = datetime.now(timezone.utc)
    for run in stage_runs:
        started_at = parse_iso_datetime(run.get("started_at")) or parse_iso_datetime(run.get("claimed_at"))
        finished_at = parse_iso_datetime(run.get("finished_at"))
        if not started_at:
            run["duration_minutes_label"] = ""
            continue
        end_time = finished_at or now
        seconds = max(0.0, (end_time - started_at).total_seconds())
        run["duration_minutes"] = round(seconds / 60.0, 2)
        run["duration_minutes_label"] = format_duration_label(seconds)


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_duration_label(seconds: float) -> str:
    if seconds < 60:
        return f"{int(round(seconds))}s"
    minutes = seconds / 60.0
    if minutes < 120:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.1f}h"


def stage_signature(stage_runs: list[dict[str, Any]]) -> str:
    parts = []
    for run in stage_runs:
        progress = run.get("progress_json") if isinstance(run.get("progress_json"), dict) else {}
        parts.append(
            ":".join(
                [
                    str(run.get("id") or ""),
                    str(run.get("status") or ""),
                    str(run.get("updated_at") or ""),
                    str(progress.get("percent") or ""),
                    str(progress.get("message") or ""),
                ]
            )
        )
    return "|".join(parts)


def preprocess_args_from_form(values: dict[str, Any]) -> list[str]:
    flag_by_field = {
        "candidate_fps": "--candidate-fps",
        "target_min": "--target-min",
        "target_max": "--target-max",
        "min_blur": "--min-blur",
        "min_brightness": "--min-brightness",
        "max_brightness": "--max-brightness",
        "min_contrast": "--min-contrast",
        "min_entropy": "--min-entropy",
        "force_keep_interval": "--force-keep-interval",
        "coverage_window_seconds": "--coverage-window-seconds",
        "min_frames_per_window": "--min-frames-per-window",
        "coverage_hard_min_blur": "--coverage-hard-min-blur",
        "coverage_hard_min_brightness": "--coverage-hard-min-brightness",
        "coverage_hard_max_brightness": "--coverage-hard-max-brightness",
        "coverage_hard_min_contrast": "--coverage-hard-min-contrast",
        "coverage_hard_min_entropy": "--coverage-hard-min-entropy",
    }
    args: list[str] = []
    for field, value in values.items():
        if value is None:
            continue
        args.extend([flag_by_field[field], str(value)])
    return args


def preprocess_review_context(project: dict[str, Any], stage_runs: list[dict[str, Any]]) -> dict[str, Any]:
    preprocess_runs = [run for run in stage_runs if run.get("stage") == "preprocess"]
    latest_run = preprocess_runs[0] if preprocess_runs else None
    capture_report = load_capture_report(latest_run) if latest_run else {}
    summary = capture_report.get("summary", {}) if isinstance(capture_report, dict) else {}
    videos = capture_report.get("videos", []) if isinstance(capture_report, dict) else []
    if not capture_report and latest_run:
        compact_summary = latest_run.get("summary_json") or {}
        summary = compact_summary
        videos = compact_summary.get("videos", []) if isinstance(compact_summary, dict) else []
    return {
        "latest_preprocess_run": latest_run,
        "preprocess_capture_report": capture_report,
        "preprocess_summary": summary,
        "preprocess_settings": preprocess_settings(latest_run, capture_report),
        "preprocess_form_values": preprocess_form_values(project, latest_run, capture_report),
        "preprocess_quality_rows": quality_distribution_rows(summary),
        "preprocess_video_timeline_blocks": video_timeline_blocks(capture_report, latest_run),
        "preprocess_image_grid": coverage_image_grid(capture_report),
        "preprocess_video_rows": compact_video_rows(videos),
        "preprocess_run_rows": preprocess_run_rows(preprocess_runs),
        "preprocess_profile_defaults": ui_profile_defaults(),
        "raw_source_summary": raw_source_summary(project),
        "colmap_stats_rows": colmap_stats_rows(stage_runs),
    }


def colmap_review_context(project: dict[str, Any], stage_runs: list[dict[str, Any]]) -> dict[str, Any]:
    preprocess_runs = [run for run in stage_runs if run.get("stage") == "preprocess"]
    latest_preprocess = preprocess_runs[0] if preprocess_runs else None
    colmap_runs = [run for run in stage_runs if run.get("stage") == "colmap"]
    latest_run = colmap_runs[0] if colmap_runs else None
    input_json = latest_run.get("input_uri_json") if latest_run and isinstance(latest_run.get("input_uri_json"), dict) else {}
    selected_gpu = first_gpu_type(input_json.get("gpu_type_ids")) or DEFAULT_COLMAP_GPU
    colmap_output_base = project.get("colmap_current_uri", "").rsplit("/current", 1)[0] if project.get("colmap_current_uri") else ""
    return {
        "colmap_gate_open": bool(latest_preprocess and latest_preprocess.get("status") == "approved"),
        "latest_colmap_run": latest_run,
        "colmap_form_values": {
            "preprocess_uri": input_json.get("preprocess_uri") or project.get("preprocess_current_uri") or "",
            "output_uri": input_json.get("output_uri") or colmap_output_base,
            "endpoint_url": input_json.get("endpoint_url") or "",
            "mode": input_json.get("mode") or "global",
            "matcher": input_json.get("matcher") or "exhaustive",
            "camera_model": input_json.get("camera_model") or "SIMPLE_RADIAL",
            "provider": latest_run.get("provider") if latest_run else default_colmap_provider(),
            "image": latest_run.get("image") if latest_run else "",
            "repo_url": input_json.get("repo_url") or "",
            "git_ref": input_json.get("git_ref") or "",
            "gpu_type_id": selected_gpu,
        },
        "colmap_gpu_options": COLMAP_GPU_OPTIONS,
        "colmap_info_rows": stage_info_rows(latest_run, preferred_keys=["provider_job_id", "provider_pod_id", "registered_images", "point_count", "matcher", "mode"]),
    }


def training_review_context(project: dict[str, Any], stage_runs: list[dict[str, Any]]) -> dict[str, Any]:
    colmap_runs = [run for run in stage_runs if run.get("stage") == "colmap"]
    latest_colmap = colmap_runs[0] if colmap_runs else None
    training_runs = [run for run in stage_runs if run.get("stage") == "training"]
    latest_run = training_runs[0] if training_runs else None
    input_json = latest_run.get("input_uri_json") if latest_run and isinstance(latest_run.get("input_uri_json"), dict) else {}
    selected_gpu = first_gpu_type(input_json.get("gpu_type_ids")) or DEFAULT_TRAINING_GPU
    training_output_base = project.get("training_current_uri", "").rsplit("/current", 1)[0] if project.get("training_current_uri") else ""
    return {
        "training_gate_open": bool(latest_colmap and latest_colmap.get("status") == "approved"),
        "latest_training_run": latest_run,
        "training_form_values": {
            "preprocess_uri": input_json.get("preprocess_uri") or project.get("preprocess_current_uri") or "",
            "colmap_uri": input_json.get("colmap_uri") or project.get("colmap_current_uri") or "",
            "output_uri": input_json.get("output_uri") or training_output_base,
            "endpoint_url": input_json.get("endpoint_url") or "",
            "method": input_json.get("method") or "splatfacto",
            "max_steps": input_json.get("max_steps") or 100,
            "save_every": input_json.get("save_every") or 50,
            "eval_every": input_json.get("eval_every") or 50,
            "num_downscales": input_json.get("num_downscales") or 1,
            "checkpoint_export_mode": input_json.get("checkpoint_export_mode") or "all",
            "provider": latest_run.get("provider") if latest_run else default_training_provider(),
            "image": latest_run.get("image") if latest_run else "",
            "repo_url": input_json.get("repo_url") or "",
            "git_ref": input_json.get("git_ref") or "",
            "gpu_type_id": selected_gpu,
            "export": input_json.get("export", True),
        },
        "training_gpu_options": TRAINING_GPU_OPTIONS,
        "training_info_rows": stage_info_rows(latest_run, preferred_keys=["provider_job_id", "provider_pod_id", "method", "max_steps", "checkpoint_count", "exported_ply"]),
    }


def ui_profile_defaults() -> dict[str, dict[str, Any]]:
    visible_fields = {
        "candidate_fps",
        "target_min",
        "target_max",
        "min_blur",
        "min_brightness",
        "max_brightness",
        "min_contrast",
        "min_entropy",
        "force_keep_interval",
        "coverage_window_seconds",
        "min_frames_per_window",
        "coverage_hard_min_blur",
        "coverage_hard_min_brightness",
        "coverage_hard_max_brightness",
        "coverage_hard_min_contrast",
        "coverage_hard_min_entropy",
    }
    return {
        profile: {key: value for key, value in defaults.items() if key in visible_fields}
        for profile, defaults in sorted(PROFILE_DEFAULTS.items())
    }


def raw_source_summary(project: dict[str, Any]) -> dict[str, Any]:
    raw_uri = project.get("raw_uri") if project else None
    if not raw_uri:
        return {"loaded": False, "source_count": 0, "rows": [], "sources": []}
    try:
        manifest = load_json_uri(f"{raw_uri.rstrip('/')}/sources_manifest.json")
    except Exception as exc:
        return {"loaded": False, "source_count": 0, "rows": [], "sources": [], "error": str(exc)}
    sources = manifest.get("sources") if isinstance(manifest, dict) else []
    if not isinstance(sources, list):
        sources = []
    return {
        "loaded": True,
        "source_count": len(sources),
        "rows": raw_source_count_rows(sources),
        "sources": compact_raw_sources(sources),
    }


def raw_source_count_rows(sources: list[Any]) -> list[dict[str, Any]]:
    groups = [
        ("Role", "role"),
        ("Camera group", "camera_group"),
        ("Location", "location"),
        ("COLMAP", "colmap_policy"),
    ]
    rows = []
    for label, key in groups:
        counts = Counter(str(source.get(key) or "unset") for source in sources if isinstance(source, dict))
        for value, count in sorted(counts.items()):
            rows.append({"group": label, "value": value, "count": count})
    return rows


def compact_raw_sources(sources: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        rows.append(
            {
                "relative_path": source.get("relative_path"),
                "role": source.get("role"),
                "camera_group": source.get("camera_group"),
                "location": source.get("location"),
                "colmap_policy": source.get("colmap_policy"),
            }
        )
    return rows


def first_gpu_type(value: Any) -> Optional[str]:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str) and first.strip():
            return normalize_gpu_name(first)
    if isinstance(value, str) and value.strip():
        return normalize_gpu_name(value)
    return None


def colmap_stats_rows(stage_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    colmap_runs = [run for run in stage_runs if run.get("stage") == "colmap"]
    if not colmap_runs:
        return []
    latest = colmap_runs[0]
    summary = latest.get("summary_json") or {}
    rows = [
        {"label": "Run ID", "value": latest.get("id")},
        {"label": "Status", "value": latest.get("status")},
        {"label": "Provider", "value": latest.get("provider")},
        {"label": "Attempt", "value": latest.get("attempt")},
        {"label": "Output URI", "value": latest.get("output_uri")},
    ]
    if isinstance(summary, dict):
        preferred = [
            "registered_images",
            "total_images",
            "points3D",
            "point_count",
            "mean_reprojection_error",
            "median_reprojection_error",
            "track_length_mean",
            "camera_count",
            "sparse_model_count",
            "message",
            "provider",
        ]
        used = set()
        for key in preferred:
            if key in summary:
                rows.append({"label": humanize_key(key), "value": summary.get(key)})
                used.add(key)
        for key, value in summary.items():
            if key in used or isinstance(value, (dict, list)):
                continue
            rows.append({"label": humanize_key(key), "value": value})
    return [row for row in rows if row.get("value") not in (None, "", [], {})]


def stage_info_rows(run: Optional[dict[str, Any]], *, preferred_keys: list[str]) -> list[dict[str, Any]]:
    if not run:
        return []
    rows = [
        {"label": "Run ID", "value": run.get("id")},
        {"label": "Status", "value": run.get("status")},
        {"label": "Provider", "value": run.get("provider")},
        {"label": "Attempt", "value": run.get("attempt")},
    ]
    input_json = run.get("input_uri_json") if isinstance(run.get("input_uri_json"), dict) else {}
    summary = run.get("summary_json") if isinstance(run.get("summary_json"), dict) else {}
    merged = {**input_json, **summary}
    used = set()
    for key in preferred_keys:
        if key in merged and merged.get(key) not in (None, "", [], {}):
            rows.append({"label": humanize_key(key), "value": merged.get(key)})
            used.add(key)
    gpu_value = first_gpu_type(input_json.get("gpu_type_ids"))
    if gpu_value:
        rows.append({"label": "GPU", "value": gpu_value})
    if run.get("image"):
        rows.append({"label": "Image", "value": run.get("image")})
    if run.get("output_uri"):
        rows.append({"label": "Output URI", "value": run.get("output_uri")})
    return rows


def humanize_key(value: str) -> str:
    if value == "points3D":
        return "3D points"
    return value.replace("_", " ").strip().title()


def load_capture_report(run: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not run:
        return {}
    output_uri = run.get("output_uri")
    if not output_uri:
        return {}
    try:
        return load_json_uri(f"{output_uri.rstrip('/')}/capture_report.json")
    except Exception:
        return {}


def load_json_uri(uri: str) -> dict[str, Any]:
    storage_uri = parse_storage_uri(uri)
    if storage_uri.is_local:
        path = storage_uri.as_local_path()
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    suffix = Path(storage_uri.key).suffix or ".json"
    with tempfile.NamedTemporaryFile("w+", suffix=suffix) as handle:
        copy_file(uri, handle.name)
        payload = Path(handle.name).read_text(encoding="utf-8").strip()
    if not payload:
        return {}
    return json.loads(payload)


def preprocess_settings(run: Optional[dict[str, Any]], capture_report: dict[str, Any]) -> dict[str, Any]:
    if not run:
        return {}
    input_json = run.get("input_uri_json") if isinstance(run.get("input_uri_json"), dict) else {}
    profile = str(input_json.get("profile") or "indoor_room")
    settings = dict(PROFILE_DEFAULTS.get(profile, {}))
    settings["profile"] = profile
    settings.update(parse_preprocess_args(input_json.get("preprocess_args")))

    capture_settings = capture_report.get("settings", {}) if isinstance(capture_report, dict) else {}
    if isinstance(capture_settings, dict) and capture_settings:
        settings.update(capture_settings)

    summary = run.get("summary_json") or {}
    if isinstance(summary, dict):
        summary_settings = summary.get("settings") or {}
        if isinstance(summary_settings, dict) and summary_settings:
            settings.update(summary_settings)
    return settings


def preprocess_form_values(project: dict[str, Any], run: Optional[dict[str, Any]], capture_report: dict[str, Any]) -> dict[str, Any]:
    input_json = run.get("input_uri_json") if run and isinstance(run.get("input_uri_json"), dict) else {}
    settings = preprocess_settings(run, capture_report)
    preprocess_current_uri = project.get("preprocess_current_uri") or ""
    return {
        "raw_uri": input_json.get("raw_uri") or project.get("raw_uri") or "",
        "output_uri": input_json.get("output_uri") or preprocess_current_uri.rsplit("/current", 1)[0] or "",
        "endpoint_url": input_json.get("endpoint_url") or "",
        "profile": settings.get("profile") or input_json.get("profile") or "indoor_room",
        **settings,
    }


def parse_preprocess_args(values: Any) -> dict[str, Any]:
    if not isinstance(values, list):
        return {}
    field_by_flag = {
        "--candidate-fps": ("candidate_fps", float),
        "--target-min": ("target_min", int),
        "--target-max": ("target_max", int),
        "--min-blur": ("min_blur", float),
        "--min-brightness": ("min_brightness", float),
        "--max-brightness": ("max_brightness", float),
        "--min-contrast": ("min_contrast", float),
        "--min-entropy": ("min_entropy", float),
        "--force-keep-interval": ("force_keep_interval", float),
        "--coverage-window-seconds": ("coverage_window_seconds", float),
        "--min-frames-per-window": ("min_frames_per_window", int),
        "--coverage-hard-min-blur": ("coverage_hard_min_blur", float),
        "--coverage-hard-min-brightness": ("coverage_hard_min_brightness", float),
        "--coverage-hard-max-brightness": ("coverage_hard_max_brightness", float),
        "--coverage-hard-min-contrast": ("coverage_hard_min_contrast", float),
        "--coverage-hard-min-entropy": ("coverage_hard_min_entropy", float),
    }
    parsed: dict[str, Any] = {}
    index = 0
    while index + 1 < len(values):
        raw_flag = values[index]
        if not isinstance(raw_flag, str):
            index += 1
            continue
        field_spec = field_by_flag.get(raw_flag)
        raw_value = values[index + 1]
        if field_spec is None:
            index += 2
            continue
        field_name, caster = field_spec
        try:
            parsed[field_name] = caster(raw_value)
        except (TypeError, ValueError):
            pass
        index += 2
    return parsed


def preprocess_run_rows(stage_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in stage_runs:
        run_copy = dict(run)
        run_copy.update(preprocess_run_metrics(run_copy))
        rows.append(run_copy)
    return rows


def preprocess_run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary_json") if isinstance(run.get("summary_json"), dict) else {}
    videos = summary.get("videos") if isinstance(summary.get("videos"), list) else []
    rows = compact_video_rows(videos)
    if not rows:
        return {"selected_total": "", "fallback_total": "", "gap_total": "", "warning_summary": ""}
    warning_count = sum(len(row.get("warnings", [])) for row in rows)
    return {
        "selected_total": sum(int(row.get("selected_frame_count") or 0) for row in rows),
        "fallback_total": sum(int(row.get("fallback_count") or 0) for row in rows),
        "gap_total": sum(int(row.get("coverage_gaps") or 0) for row in rows),
        "warning_summary": f"{warning_count} warning{'s' if warning_count != 1 else ''}" if warning_count else "",
    }


def quality_distribution_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    source = summary.get("metric_distributions") if isinstance(summary, dict) else None
    if not isinstance(source, dict):
        source = {
            "blur_score": summary.get("blur_score_distribution") if isinstance(summary, dict) else None,
            "brightness": summary.get("brightness_distribution") if isinstance(summary, dict) else None,
            "contrast": summary.get("contrast_distribution") if isinstance(summary, dict) else None,
            "entropy": summary.get("entropy_distribution") if isinstance(summary, dict) else None,
        }
    labels = {
        "blur_score": "Blur",
        "brightness": "Brightness",
        "contrast": "Contrast",
        "entropy": "Entropy",
    }
    rows = []
    for key, label in labels.items():
        distribution = source.get(key) if isinstance(source, dict) else None
        if not isinstance(distribution, dict):
            continue
        rows.append({"key": key, "label": label, "distribution": distribution})
    return rows


def compact_video_rows(videos: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for video in videos:
        if not isinstance(video, dict):
            continue
        video_meta = video.get("video") if isinstance(video.get("video"), dict) else {}
        coverage = video.get("coverage") if isinstance(video.get("coverage"), dict) else {}
        rows.append(
            {
                "source_id": video.get("source_id"),
                "duration_seconds": video.get("duration_seconds") or video_meta.get("duration_seconds"),
                "candidate_frame_count": video.get("candidate_frame_count"),
                "selected_frame_count": video.get("selected_frame_count"),
                "fallback_count": video.get("coverage_fallback_frame_count") or coverage.get("coverage_fallback_frame_count"),
                "largest_gap_seconds": coverage.get("largest_selected_gap_seconds"),
                "coverage_gaps": coverage.get("windows_below_minimum_count"),
                "warnings": video.get("warnings", []),
            }
        )
    return rows


def video_timeline_blocks(capture_report: dict[str, Any], run: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(capture_report, dict) and capture_report.get("videos"):
        return full_timeline_blocks(capture_report)
    if run:
        return compact_selected_timeline_blocks(run.get("summary_json") or {})
    return []


def coverage_image_grid(capture_report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(capture_report, dict):
        return {"items": [], "counts": {}}
    frames = capture_report.get("frames", [])
    if not isinstance(frames, list):
        return {"items": [], "counts": {}}
    image_frames = [
        frame for frame in frames
        if isinstance(frame, dict) and str(frame.get("source_id") or "") == "coverage_images"
    ]
    items = []
    counts = Counter()
    for frame in image_frames:
        decision = image_decision(frame)
        counts[decision] += 1
        items.append(
            {
                "decision": decision,
                "title": image_hover_title(frame, decision),
            }
        )
    return {"items": items, "counts": dict(counts)}


def full_timeline_blocks(capture_report: dict[str, Any]) -> list[dict[str, Any]]:
    frames = capture_report.get("frames", []) if isinstance(capture_report, dict) else []
    frames_by_source: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        source_id = str(frame.get("source_id") or "unknown")
        frames_by_source.setdefault(source_id, []).append(frame)

    blocks = []
    for video in capture_report.get("videos", []):
        if not isinstance(video, dict):
            continue
        source_id = str(video.get("source_id") or "unknown")
        video_meta = video.get("video") if isinstance(video.get("video"), dict) else {}
        duration = float(video_meta.get("duration_seconds") or video.get("duration_seconds") or 0.0)
        source_frames = frames_by_source.get(source_id, [])
        max_time = max([duration, *[float(frame.get("timestamp_seconds") or 0.0) for frame in source_frames], 0.001])
        coverage = video.get("coverage") if isinstance(video.get("coverage"), dict) else {}
        selected_by = video.get("selected_by") if isinstance(video.get("selected_by"), dict) else {}
        blocks.append(
            {
                "source_id": source_id,
                "meta": {
                    "selected": video.get("selected_frame_count"),
                    "quality": selected_by.get("quality", 0),
                    "fallback": video.get("coverage_fallback_frame_count") or coverage.get("coverage_fallback_frame_count"),
                    "coverage_gaps": coverage.get("windows_below_minimum_count", 0),
                    "largest_gap_seconds": coverage.get("largest_selected_gap_seconds"),
                },
                "ticks": [timeline_tick(frame, max_time) for frame in source_frames],
                "gaps": [timeline_gap(gap, max_time) for gap in coverage.get("windows_below_minimum", []) if isinstance(gap, dict)],
                "markers": time_markers(max_time),
            }
        )
    return blocks


def compact_selected_timeline_blocks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = summary.get("selected_timeline", []) if isinstance(summary, dict) else []
    if not timeline:
        return []
    frames_by_source: dict[str, list[dict[str, Any]]] = {}
    for frame in timeline:
        if not isinstance(frame, dict):
            continue
        frames_by_source.setdefault(str(frame.get("source_id") or "unknown"), []).append(frame)
    blocks = []
    for source_id, frames in frames_by_source.items():
        max_time = max([float(frame.get("timestamp_seconds") or 0.0) for frame in frames] + [0.001])
        blocks.append(
            {
                "source_id": source_id,
                "meta": {"selected": len(frames), "quality": None, "fallback": None, "coverage_gaps": None, "largest_gap_seconds": None},
                "ticks": [timeline_tick(frame, max_time) for frame in frames],
                "gaps": [],
                "markers": time_markers(max_time),
                "compact": True,
            }
        )
    return blocks


def timeline_tick(frame: dict[str, Any], max_time: float) -> dict[str, Any]:
    timestamp = float(frame.get("timestamp_seconds") or 0.0)
    decision = str(frame.get("decision") or frame.get("selected_by") or "selected")
    if decision == "quality":
        decision = "selected"
    return {
        "left": percent(timestamp, max_time),
        "decision": decision,
        "title": video_hover_title(frame, decision),
    }


def timeline_gap(gap: dict[str, Any], max_time: float) -> dict[str, Any]:
    start = float(gap.get("start_seconds") or 0.0)
    end = float(gap.get("end_seconds") or start)
    left = percent(start, max_time)
    width = max(0.3, min(100.0 - left, percent(max(0.0, end - start), max_time)))
    return {
        "left": left,
        "width": width,
        "title": f"{gap.get('start_seconds')}s-{gap.get('end_seconds')}s | {gap.get('candidate_frames')} candidates | {gap.get('selected_frames')} selected",
    }


def video_hover_title(frame: dict[str, Any], decision: str) -> str:
    timestamp = frame.get("timestamp_seconds")
    blur_score = frame.get("blur_score")
    parts = [f"{timestamp}s", f"blur {blur_score}"]
    if decision not in {"selected", "coverage_fallback"}:
        parts.append(str(frame.get("reject_reason") or decision))
    return " | ".join(parts)


def image_hover_title(frame: dict[str, Any], decision: str) -> str:
    filename = frame.get("source_image") or frame.get("output_file") or frame.get("frame_index") or "image"
    blur_score = frame.get("blur_score")
    parts = [Path(str(filename)).name, f"blur {blur_score}"]
    if decision not in {"selected", "coverage_fallback"}:
        parts.append(str(frame.get("reject_reason") or decision))
    return " | ".join(parts)


def image_decision(frame: dict[str, Any]) -> str:
    decision = str(frame.get("decision") or frame.get("selected_by") or "selected")
    if decision in {"quality", "selected", "coverage_fallback"}:
        return "selected"
    return "rejected"


def time_markers(max_time: float) -> list[dict[str, Any]]:
    return [{"left": fraction * 100, "label": format_seconds(max_time * fraction)} for fraction in [0, 0.25, 0.5, 0.75, 1]]


def percent(value: float, max_value: float) -> float:
    return max(0.0, min(100.0, (value / max(max_value, 0.001)) * 100.0))


def format_seconds(value: float) -> str:
    if value >= 60:
        minutes = int(value // 60)
        seconds = int(round(value % 60))
        return f"{minutes}m {seconds}s"
    return f"{value:.1f}s"


def empty_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def require_r2_uri(value: str, field_name: str) -> None:
    if not value.startswith("r2://"):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be an r2:// URI for real preprocessing",
        )
