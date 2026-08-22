"""Server-rendered internal controller UI."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from controller_common.config import (
    default_colmap_provider,
    default_colmap_vocab_tree,
    default_r2_bucket,
    default_training_provider,
    runpod_colmap_container_disk_gb,
    runpod_training_container_disk_gb,
)
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
from controller_common.preprocess_assembly import assembled_project_preprocess_uri, preprocess_output_base_uri
from src.realestate_splat.storage import copy_file, parse_storage_uri
from scripts.preprocess_video import PROFILE_DEFAULTS
from controller_common.raw_upload import source_group_key
from controller_common.matching_plan import build_hybrid_matching_plan, build_single_matching_plan, build_source_groups, resolve_group_reference, validate_matching_plan
from controller_common.matching_plan import SUPPORTED_STRATEGIES


router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).with_name("templates")))

COLMAP_FEATURE_EXTRACTOR_OPTIONS = [
    {"value": "SIFT", "label": "SIFT"},
    {"value": "ALIKED_N16ROT", "label": "ALIKED N16ROT"},
    {"value": "ALIKED_N32", "label": "ALIKED N32"},
]

COLMAP_MATCHER_OPTIONS = [
    {"value": "exhaustive", "label": "Exhaustive"},
    {"value": "sequential", "label": "Sequential"},
    {"value": "vocab_tree", "label": "Vocabulary tree"},
]

COLMAP_FEATURE_MATCHER_OPTIONS = [
    {"value": "SIFT_BRUTEFORCE", "label": "SIFT bruteforce"},
    {"value": "SIFT_LIGHTGLUE", "label": "SIFT LightGlue"},
    {"value": "ALIKED_BRUTEFORCE", "label": "ALIKED bruteforce"},
    {"value": "ALIKED_LIGHTGLUE", "label": "ALIKED LightGlue"},
]

COLMAP_CAMERA_MODEL_OPTIONS = [
    {"value": "SIMPLE_RADIAL", "label": "SIMPLE_RADIAL"},
    {"value": "RADIAL", "label": "RADIAL"},
    {"value": "OPENCV", "label": "OPENCV"},
    {"value": "OPENCV_FISHEYE", "label": "OPENCV_FISHEYE"},
    {"value": "PINHOLE", "label": "PINHOLE"},
    {"value": "SIMPLE_PINHOLE", "label": "SIMPLE_PINHOLE"},
]

DEFAULT_PREPROCESS_PROFILE = "indoor_room"


def json_pretty(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    return json.dumps(value, indent=2, sort_keys=True)


templates.env.filters["json_pretty"] = json_pretty


def validate_colmap_feature_matcher(feature_extractor: str, matching_type: str) -> None:
    if feature_extractor == "SIFT" and not matching_type.startswith("SIFT_"):
        raise HTTPException(status_code=400, detail="SIFT features require a SIFT matching type")
    if feature_extractor.startswith("ALIKED") and not matching_type.startswith("ALIKED_"):
        raise HTTPException(status_code=400, detail="ALIKED features require an ALIKED matching type")


def default_colmap_max_image_size(feature_extractor: str) -> int:
    return 2048 if str(feature_extractor).startswith("ALIKED") else 3200


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
    blacklist = load_colmap_blacklist(row_to_json(project))
    excluded_names = {
        Path(str(entry.get("image_name") or "")).name
        for entry in blacklist.get("excluded_images", [])
        if isinstance(entry, dict) and entry.get("image_name")
    }
    if excluded_names and isinstance(payload.get("cameras"), list):
        original_cameras = payload["cameras"]
        payload = dict(payload)
        payload["cameras"] = [
            camera
            for camera in original_cameras
            if Path(str(camera.get("name") or camera.get("image_name") or "")).name not in excluded_names
        ]
        payload["blacklisted_camera_count"] = len(original_cameras) - len(payload["cameras"])
    return JSONResponse(payload)


@router.post("/projects/{project_id}/colmap-blacklist")
def blacklist_colmap_image(project_id: str, payload: dict[str, Any] = Body(default={})) -> JSONResponse:
    image_name = str(payload.get("image_name") or "").strip()
    if not image_name or "/" in image_name or "\\" in image_name:
        raise HTTPException(status_code=400, detail="A plain COLMAP image filename is required")
    with connect() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    project_json = row_to_json(project)
    blacklist_uri = colmap_blacklist_uri(project_json)
    blacklist = load_colmap_blacklist(project_json)
    entries = blacklist.get("excluded_images")
    if not isinstance(entries, list):
        entries = []
    if any(isinstance(entry, dict) and entry.get("image_name") == image_name for entry in entries):
        return JSONResponse({"status": "already_blacklisted", "image_name": image_name})
    entries.append(
        {
            "image_name": image_name,
            "role": payload.get("role"),
            "location": payload.get("location"),
            "camera_group": payload.get("camera_group"),
            "reason": str(payload.get("reason") or "Manually flagged in COLMAP viewer"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    document = {
        "schema_version": 1,
        "project_id": project_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "excluded_images": entries,
    }
    with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")
        handle.flush()
        copy_file(handle.name, blacklist_uri)
    return JSONResponse({"status": "blacklisted", "image_name": image_name, "count": len(entries)})


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_detail(request: Request, project_id: str) -> HTMLResponse:
    data = load_project_detail(project_id)
    if data["project"] is None:
        raise HTTPException(status_code=404, detail="Project not found")
    review = preprocess_review_context(data["project"], data["stage_runs"])
    colmap_review = colmap_review_context(data["project"], data["stage_runs"], review["raw_source_summary"])
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
    group_settings_json: Optional[str] = Form(default=None),
    profile: str = Form(default=DEFAULT_PREPROCESS_PROFILE),
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
        resolved_output_uri = preprocess_output_base_uri(
            empty_to_none(output_uri) or f"r2://{default_r2_bucket()}/projects/{project_id}/preprocess",
            project_id,
        )
        require_r2_uri(resolved_raw_uri, "raw_uri")
        require_r2_uri(resolved_output_uri, "output_uri")
        group_configs = parse_group_settings(group_settings_json)
        preprocess_scope = "group" if len(group_configs) == 1 else "project"
        if preprocess_scope == "group":
            resolved_output_uri = f"{resolved_output_uri.rstrip('/')}/groups/{ui_slug(str(group_configs[0].get('group_key')))}"
        input_uri_json = {
            "raw_uri": resolved_raw_uri,
            "output_uri": resolved_output_uri,
            "endpoint_url": empty_to_none(endpoint_url),
            "profile": profile,
            "group_configs": group_configs,
            "preprocess_scope": preprocess_scope,
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
    feature_extractor: str = Form(default="SIFT"),
    matcher: Optional[str] = Form(default=None),
    processing_strategy: Optional[str] = Form(default=None),
    matching_connections_json: Optional[str] = Form(default=None),
    matching_type: str = Form(default="SIFT_BRUTEFORCE"),
    camera_model: str = Form(default="SIMPLE_RADIAL"),
    max_image_size: int = Form(default=0),
    sequential_loop_detection: Optional[str] = Form(default=None),
    vocab_tree: Optional[str] = Form(default=None),
    provider: str = Form(default=default_colmap_provider()),
    image: Optional[str] = Form(default=None),
    repo_url: Optional[str] = Form(default=None),
    git_ref: Optional[str] = Form(default=None),
    gpu_type_id: str = Form(default=DEFAULT_COLMAP_GPU),
    container_disk_gb: int = Form(default=20),
    dry_run: bool = Form(default=False),
) -> RedirectResponse:
    with connect() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        preprocess_runs = rows_to_json(
            conn.execute(
                """
                SELECT *
                FROM stage_runs
                WHERE project_id = %s AND stage = 'preprocess'
                ORDER BY created_at DESC
                """,
                (project_id,),
            ).fetchall()
        )
        raw_summary = raw_source_summary(row_to_json(project))
        if not all_location_preprocess_runs_approved(raw_summary, preprocess_runs):
            raise HTTPException(status_code=400, detail="All source locations must have approved preprocess runs before COLMAP can be queued")
        project_json = row_to_json(project)
        assembled_preprocess_uri = assembled_project_preprocess_uri(project_json)
        preprocess_group_outputs = approved_preprocess_group_outputs(raw_summary, preprocess_runs)
        resolved_preprocess_uri = assembled_preprocess_uri or project.get("preprocess_current_uri")
        if not resolved_preprocess_uri:
            raise HTTPException(status_code=400, detail="Project preprocess_current_uri is required to queue COLMAP")
        resolved_output_uri = empty_to_none(output_uri) or f"r2://{default_r2_bucket()}/projects/{project_id}/colmap"
        require_r2_uri(resolved_preprocess_uri, "preprocess_uri")
        require_r2_uri(resolved_output_uri, "output_uri")
        validate_choice(feature_extractor, {option["value"] for option in COLMAP_FEATURE_EXTRACTOR_OPTIONS}, "feature_extractor")
        validate_choice(matching_type, {option["value"] for option in COLMAP_FEATURE_MATCHER_OPTIONS}, "matching_type")
        validate_choice(camera_model, {option["value"] for option in COLMAP_CAMERA_MODEL_OPTIONS}, "camera_model")
        validate_colmap_feature_matcher(feature_extractor, matching_type)
        resolved_max_image_size = max_image_size if max_image_size > 0 else default_colmap_max_image_size(feature_extractor)
        resolved_vocab_tree = empty_to_none(vocab_tree) or default_colmap_vocab_tree()
        source_manifest = approved_colmap_image_manifest(raw_summary, preprocess_runs)
        connections = parse_matching_connections(matching_connections_json)
        matching_plan = None
        try:
            saved_plan = load_json_uri(colmap_matching_plan_uri(project_json))
        except Exception:
            saved_plan = {}
        if saved_plan.get("strategy") not in SUPPORTED_STRATEGIES:
            raise HTTPException(status_code=400, detail="Select and save a matching strategy before queueing COLMAP")
        if processing_strategy is None and saved_plan.get("strategy") in SUPPORTED_STRATEGIES:
            matching_plan = saved_plan
            processing_strategy = str(saved_plan["strategy"])
        elif processing_strategy not in {None, "single"}:
            try:
                matching_plan = build_hybrid_matching_plan(
                    source_manifest,
                    {"processing_strategy": processing_strategy, "matching_type": matching_type},
                    connections,
                )
                validate_matching_plan(matching_plan)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        processing_strategy = processing_strategy or "single"
        if matching_plan is not None:
            connections = matching_plan.get("connections") or connections
        plan_matcher = next(
            (stage.get("matching_style") for stage in (matching_plan or {}).get("matching_stages", []) if stage.get("id") == "single_matcher"),
            None,
        )
        matcher = matcher or plan_matcher or "exhaustive"
        validate_choice(matcher, {option["value"] for option in COLMAP_MATCHER_OPTIONS}, "matcher")
        loop_detection = (
            saved_plan.get("sequential_loop_detection", True)
            if matching_plan is not None
            else str(sequential_loop_detection if sequential_loop_detection is not None else "true").lower() == "true"
        )
        matching_plan_uri = None
        if matching_plan is not None:
            matching_plan_uri = colmap_matching_plan_uri(project_json)
            with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
                handle.write(json.dumps(matching_plan, indent=2) + "\n")
                handle.flush()
                copy_file(handle.name, matching_plan_uri, endpoint_url=endpoint_url)
        input_uri_json = {
            "preprocess_uri": resolved_preprocess_uri,
            "preprocess_group_outputs": preprocess_group_outputs,
            "raw_uri": project.get("raw_uri"),
            "blacklist_uri": colmap_blacklist_uri(project_json),
            "output_uri": resolved_output_uri,
            "endpoint_url": empty_to_none(endpoint_url),
            "mode": mode,
            "feature_extractor": feature_extractor,
            "matcher": matcher,
            "processing_strategy": processing_strategy,
            "matching_connections": connections,
            "matching_plan_uri": matching_plan_uri,
            "matching_type": matching_type,
            "camera_model": camera_model,
            "max_image_size": resolved_max_image_size,
            "sequential_loop_detection": loop_detection,
            "vocab_tree": resolved_vocab_tree,
            "repo_url": empty_to_none(repo_url),
            "git_ref": empty_to_none(git_ref),
            "gpu_type_ids": [normalize_gpu_name(gpu_type_id)],
            "container_disk_gb": max(20, container_disk_gb),
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


@router.post("/projects/{project_id}/matching-strategy")
def ui_save_matching_strategy(
    project_id: str,
    processing_strategy: str = Form(...),
    single_matching_style: str = Form(default="exhaustive"),
    hero_matching_style: str = Form(default="exhaustive"),
    video_bridge_matching_style: str = Form(default="exhaustive"),
    sequential_loop_detection: Optional[str] = Form(default="true"),
    matching_connections_json: Optional[str] = Form(default=None),
) -> RedirectResponse:
    with connect() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
        preprocess_runs = rows_to_json(
            conn.execute(
                "SELECT * FROM stage_runs WHERE project_id = %s AND stage = 'preprocess' ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        )
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if processing_strategy not in {"single", "hybrid"}:
        raise HTTPException(status_code=400, detail=f"Unsupported processing strategy: {processing_strategy}")
    raw_summary = raw_source_summary(row_to_json(project))
    if not all_location_preprocess_runs_approved(raw_summary, preprocess_runs):
        raise HTTPException(status_code=400, detail="All source locations must have approved preprocess runs before selecting a matching strategy")
    manifest = approved_colmap_image_manifest(raw_summary, preprocess_runs)
    connections = parse_matching_connections(matching_connections_json)
    try:
        if processing_strategy == "single":
            if single_matching_style not in {"sequential", "exhaustive", "vocab_tree"}:
                raise ValueError(f"Unsupported single matching style: {single_matching_style}")
            plan = build_single_matching_plan(
                manifest,
                {"matcher": single_matching_style, "sequential_loop_detection": str(sequential_loop_detection).lower() == "true"},
            )
        else:
            coverage_count = sum(1 for item in build_source_groups(manifest) if item.get("kind") in {"video", "coverage_images"})
            internal_strategy = "video_plus_heroes" if coverage_count <= 1 else "multiple_videos_plus_heroes"
            plan = build_hybrid_matching_plan(
                manifest,
                {"processing_strategy": internal_strategy, "hero_matching_style": hero_matching_style, "video_bridge_matching_style": video_bridge_matching_style, "sequential_loop_detection": str(sequential_loop_detection).lower() == "true"},
                connections,
            )
        validate_matching_plan(plan)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
        handle.write(json.dumps(plan, indent=2) + "\n")
        handle.flush()
        copy_file(handle.name, colmap_matching_plan_uri(row_to_json(project)))
    return RedirectResponse(url=f"/ui/projects/{project_id}#matching", status_code=303)


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
    use_scale_regularization: str = Form(default="true"),
    provider: str = Form(default=default_training_provider()),
    image: Optional[str] = Form(default=None),
    repo_url: Optional[str] = Form(default=None),
    git_ref: Optional[str] = Form(default=None),
    gpu_type_id: str = Form(default=DEFAULT_TRAINING_GPU),
    container_disk_gb: int = Form(default=20),
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
        preprocess_runs = rows_to_json(
            conn.execute(
                """
                SELECT *
                FROM stage_runs
                WHERE project_id = %s AND stage = 'preprocess'
                ORDER BY created_at DESC
                """,
                (project_id,),
            ).fetchall()
        )
        project_json = row_to_json(project)
        preprocess_group_outputs = approved_preprocess_group_outputs(
            raw_source_summary(project_json),
            preprocess_runs,
        )
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
        splatfacto_options = {"use_scale_regularization": True}
        if use_scale_regularization in {"true", "false"}:
            splatfacto_options["use_scale_regularization"] = use_scale_regularization == "true"
        input_uri_json = {
            "preprocess_uri": resolved_preprocess_uri,
            "preprocess_group_outputs": preprocess_group_outputs,
            "colmap_uri": resolved_colmap_uri,
            "output_uri": resolved_output_uri,
            "endpoint_url": empty_to_none(endpoint_url),
            "method": method,
            "max_steps": max_steps,
            "save_every": save_every,
            "eval_every": eval_every,
            "num_downscales": num_downscales,
            "splatfacto_options": splatfacto_options or None,
            "repo_url": empty_to_none(repo_url),
            "git_ref": empty_to_none(git_ref),
            "gpu_type_ids": [normalize_gpu_name(gpu_type_id)],
            "container_disk_gb": max(20, container_disk_gb),
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
        stage_run = conn.execute("SELECT project_id, status FROM stage_runs WHERE id = %s", (stage_run_id,)).fetchone()
        if stage_run is None:
            raise HTTPException(status_code=404, detail="Stage run not found")
        project_id = stage_run["project_id"]
        if action == "approve" and stage_run["status"] == "approved":
            return RedirectResponse(url=f"/ui/projects/{project_id}", status_code=303)
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
                if action == "approve" and "not awaiting approval" in str(exc):
                    return RedirectResponse(url=f"/ui/projects/{project_id}", status_code=303)
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
    latest_by_scope: dict[str, str] = {}
    for run in stage_runs:
        scope_key = stage_run_history_scope(run)
        run_id = run.get("id")
        if scope_key and run_id and scope_key not in latest_by_scope:
            latest_by_scope[scope_key] = run_id
    for run in stage_runs:
        is_latest = latest_by_scope.get(stage_run_history_scope(run)) == run.get("id")
        run["is_latest_stage_run"] = is_latest
        if not is_latest:
            run["status_label"] = "history run"
            run["status_css"] = "history"


def stage_run_history_scope(run: dict[str, Any]) -> str:
    stage = str(run.get("stage") or "unknown")
    if stage != "preprocess":
        return stage
    group_keys = preprocess_run_group_keys(run)
    if group_keys:
        return f"{stage}:{','.join(group_keys)}"
    input_json = run.get("input_uri_json") if isinstance(run.get("input_uri_json"), dict) else {}
    if input_json.get("preprocess_scope") == "project":
        return f"{stage}:project"
    return f"{stage}:unknown"


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
        parts.append(
            ":".join(
                [
                    str(run.get("id") or ""),
                    str(run.get("status") or ""),
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


def parse_group_settings(raw: Optional[str]) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid per-group preprocess settings: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="Per-group preprocess settings must be a list")
    return [item for item in payload if isinstance(item, dict) and item.get("group_key")]


def ui_slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_") or "group"


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
    raw_summary = raw_source_summary(project)
    group_reports = latest_group_capture_reports(preprocess_runs, raw_summary)
    group_runs = latest_preprocess_runs_by_group(preprocess_runs)
    return {
        "latest_preprocess_run": latest_run,
        "preprocess_capture_report": capture_report,
        "preprocess_summary": summary,
        "preprocess_settings": preprocess_settings(latest_run, capture_report),
        "preprocess_form_values": preprocess_form_values(project, latest_run, capture_report),
        "preprocess_quality_rows": quality_distribution_rows(summary),
        "preprocess_video_timeline_blocks": video_timeline_blocks(capture_report, latest_run),
        "preprocess_image_grid": coverage_image_grid(capture_report),
        "preprocess_location_blocks": preprocess_location_blocks(raw_summary, capture_report, group_reports, group_runs),
        "preprocess_video_rows": compact_video_rows(videos),
        "preprocess_run_rows": preprocess_run_rows(preprocess_runs, raw_summary),
        "raw_source_summary": raw_summary,
        "colmap_stats_rows": colmap_stats_rows(stage_runs),
    }


def latest_group_capture_reports(preprocess_runs: list[dict[str, Any]], raw_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    current_group_keys = set(required_preprocess_group_keys(raw_summary))
    seen_groups: set[str] = set()
    for run in preprocess_runs:
        input_json = run.get("input_uri_json") if isinstance(run.get("input_uri_json"), dict) else {}
        if input_json.get("preprocess_scope") != "group":
            continue
        group_keys = preprocess_run_group_keys(run)
        if current_group_keys:
            group_keys = [group_key for group_key in group_keys if group_key in current_group_keys]
        if not group_keys:
            continue
        new_group_keys = [group_key for group_key in group_keys if group_key not in seen_groups]
        if not new_group_keys:
            continue
        seen_groups.update(new_group_keys)
        if preprocess_run_is_active(run):
            continue
        report = load_capture_report(run)
        if not report:
            continue
        for group_key in new_group_keys:
            reports[group_key] = report
    return reports


def latest_preprocess_runs_by_group(preprocess_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for run in preprocess_runs:
        for group_key in preprocess_run_group_keys(run):
            if group_key not in result:
                result[group_key] = run
    return result


def all_location_preprocess_runs_approved(raw_summary: dict[str, Any], preprocess_runs: list[dict[str, Any]]) -> bool:
    location_blocks = raw_summary.get("location_blocks") if isinstance(raw_summary, dict) else []
    if not isinstance(location_blocks, list) or not location_blocks:
        return False
    latest_by_group = latest_preprocess_runs_by_group(preprocess_runs)
    required_group_keys = [
        str(block.get("coverage_group_key"))
        for block in location_blocks
        if isinstance(block, dict)
        and not block.get("conflict")
        and block.get("coverage_kind") in {"video", "images"}
        and block.get("coverage_group_key")
    ]
    if not required_group_keys:
        return False
    for group_key in required_group_keys:
        run = latest_by_group.get(group_key)
        if not run or run.get("status") != "approved":
            return False
    return True


def approved_preprocess_group_outputs(raw_summary: dict[str, Any], preprocess_runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    latest_by_group = latest_preprocess_runs_by_group(preprocess_runs)
    outputs: list[dict[str, str]] = []
    for group_key in required_preprocess_group_keys(raw_summary):
        run = latest_by_group.get(group_key)
        if not run or run.get("status") != "approved" or not run.get("output_uri"):
            raise HTTPException(status_code=400, detail=f"Missing approved preprocess output for {group_key}")
        outputs.append({"group_key": group_key, "output_uri": str(run["output_uri"]).rstrip("/")})
    return outputs


def required_preprocess_group_keys(raw_summary: dict[str, Any]) -> list[str]:
    location_blocks = raw_summary.get("location_blocks") if isinstance(raw_summary, dict) else []
    keys = []
    for block in location_blocks if isinstance(location_blocks, list) else []:
        if (
            isinstance(block, dict)
            and not block.get("conflict")
            and block.get("coverage_kind") in {"video", "images"}
            and block.get("coverage_group_key")
        ):
            keys.append(str(block["coverage_group_key"]))
    return keys


def colmap_review_context(project: dict[str, Any], stage_runs: list[dict[str, Any]], raw_summary: dict[str, Any]) -> dict[str, Any]:
    preprocess_runs = [run for run in stage_runs if run.get("stage") == "preprocess"]
    preprocess_gate_open = all_location_preprocess_runs_approved(raw_summary, preprocess_runs)
    colmap_runs = [run for run in stage_runs if run.get("stage") == "colmap"]
    latest_run = colmap_runs[0] if colmap_runs else None
    input_json = latest_run.get("input_uri_json") if latest_run and isinstance(latest_run.get("input_uri_json"), dict) else {}
    selected_gpu = first_gpu_type(input_json.get("gpu_type_ids")) or DEFAULT_COLMAP_GPU
    colmap_output_base = project.get("colmap_current_uri", "").rsplit("/current", 1)[0] if project.get("colmap_current_uri") else ""
    feature_extractor = normalize_colmap_feature_extractor(input_json.get("feature_extractor") or "SIFT")
    assembled_preprocess_uri = assembled_project_preprocess_uri(project)
    source_manifest = source_manifest_from_raw_summary(raw_summary)
    try:
        saved_plan = load_json_uri(colmap_matching_plan_uri(project))
    except Exception:
        saved_plan = {}
    displayed_plan = saved_plan if saved_plan else {}
    matching_plan_selected = saved_plan.get("strategy") in SUPPORTED_STRATEGIES
    plan_connections = [
        item for item in displayed_plan.get("connections", [])
        if isinstance(item, dict) and item.get("kind") != "hero_location"
    ]
    source_groups = matching_source_group_cards(raw_summary, preprocess_runs)
    visible_group_ids = {str(group.get("id")): group for group in source_groups}
    visible_connections = []
    for connection in plan_connections:
        source = resolve_group_reference(str(connection.get("from") or ""), visible_group_ids)
        target = resolve_group_reference(str(connection.get("to") or ""), visible_group_ids)
        if source and target:
            visible_connections.append({**connection, "from": source, "to": target})
    has_heroes = any(group.get("kind") == "hero" for group in source_groups)
    coverage_group_count = sum(group.get("kind") in {"video", "coverage_images"} for group in source_groups)
    video_group_count = sum(group.get("kind") == "video" for group in source_groups)
    ui_strategy = "single" if displayed_plan.get("strategy") == "single" else ("hybrid" if displayed_plan else "single")
    single_matching_style = displayed_plan.get("single_matching_style") or next(
        (stage.get("matching_style") for stage in displayed_plan.get("matching_stages", []) if stage.get("id") == "single_matcher"),
        "exhaustive",
    )
    saved_hero_style = displayed_plan.get("hero_matching_style") or next(
        (
            stage.get("matching_style")
            for stage in displayed_plan.get("matching_stages", [])
            if stage.get("kind") == "bridge" and stage.get("matching_style") in {"exhaustive", "vocab_tree"}
        ),
        "exhaustive",
    )
    saved_bridge_style = displayed_plan.get("video_bridge_matching_style") or "exhaustive"
    return {
        "matching_gate_open": preprocess_gate_open,
        "colmap_gate_open": preprocess_gate_open and matching_plan_selected,
        "matching_plan_selected": matching_plan_selected,
        "matching_plan_status": (
            "Saved: " + (
                "Single + " + str(single_matching_style).replace("_", " ")
                if saved_plan.get("strategy") == "single"
                else "Hybrid + hero " + str(saved_hero_style).replace("_", " ") + " / video bridge " + str(saved_bridge_style).replace("_", " ")
            )
            if matching_plan_selected else "Not selected"
        ),
        "latest_colmap_run": latest_run,
        "colmap_form_values": {
            "preprocess_uri": assembled_preprocess_uri,
            "output_uri": input_json.get("output_uri") or colmap_output_base,
            "endpoint_url": input_json.get("endpoint_url") or "",
            "mode": input_json.get("mode") or "global",
            "feature_extractor": feature_extractor,
            "matcher": input_json.get("matcher") or "exhaustive",
            "processing_strategy": ui_strategy,
            "single_matching_style": single_matching_style,
            "sequential_loop_detection": displayed_plan.get("sequential_loop_detection", input_json.get("sequential_loop_detection", True)),
            "matching_connections_json": json.dumps(visible_connections or (input_json.get("matching_connections") or []), separators=(",", ":")),
            "hero_matching_style": saved_hero_style,
            "video_bridge_matching_style": saved_bridge_style,
            "matching_type": input_json.get("matching_type") or "SIFT_BRUTEFORCE",
            "camera_model": input_json.get("camera_model") or "SIMPLE_RADIAL",
            "max_image_size": input_json.get("max_image_size") or default_colmap_max_image_size(feature_extractor),
            "vocab_tree": input_json.get("vocab_tree") or default_colmap_vocab_tree() or "",
            "provider": latest_run.get("provider") if latest_run else default_colmap_provider(),
            "image": latest_run.get("image") if latest_run else "",
            "repo_url": input_json.get("repo_url") or "",
            "git_ref": input_json.get("git_ref") or "",
            "gpu_type_id": selected_gpu,
            "container_disk_gb": input_json.get("container_disk_gb") or runpod_colmap_container_disk_gb(),
        },
        "colmap_feature_extractor_options": COLMAP_FEATURE_EXTRACTOR_OPTIONS,
        "colmap_matcher_options": COLMAP_MATCHER_OPTIONS,
        "colmap_feature_matcher_options": COLMAP_FEATURE_MATCHER_OPTIONS,
        "colmap_camera_model_options": COLMAP_CAMERA_MODEL_OPTIONS,
        "colmap_gpu_options": COLMAP_GPU_OPTIONS,
        "colmap_source_groups": source_groups,
        "matching_strategy_options": [
            {"value": "single", "label": "Single matcher (fallback)", "enabled": True},
            {"value": "hybrid", "label": "Hybrid source matching", "enabled": video_group_count > 1 or has_heroes},
        ],
        "matching_has_heroes": has_heroes,
        "matching_has_multiple_coverages": video_group_count > 1,
        "colmap_info_rows": stage_info_rows(latest_run, preferred_keys=["provider_job_id", "provider_pod_id", "registered_images", "registered_by_location", "registered_by_group", "point_count", "feature_extractor", "matching_type", "matcher", "sequential_loop_detection", "vocab_tree", "camera_model", "max_image_size", "mode", "container_disk_gb"]),
        "colmap_blacklist": load_colmap_blacklist(project),
    }


def parse_matching_connections(raw: Optional[str]) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid matching connections JSON: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="Matching connections must be a list")
    return [item for item in payload if isinstance(item, dict)]


def approved_colmap_image_manifest(
    raw_summary: dict[str, Any],
    preprocess_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine approved group manifests for plan editing without creating R2 duplicates."""
    images: list[dict[str, Any]] = []
    try:
        outputs = approved_preprocess_group_outputs(raw_summary, preprocess_runs)
    except HTTPException:
        outputs = []
    for output in outputs:
        try:
            manifest = load_json_uri(f"{output['output_uri'].rstrip('/')}/image_manifest.json")
        except Exception:
            continue
        group_images = manifest.get("images") if isinstance(manifest, dict) else []
        if isinstance(group_images, list):
            images.extend(item for item in group_images if isinstance(item, dict))
    return {"schema_version": 1, "images": images}


def source_manifest_from_raw_summary(raw_summary: dict[str, Any]) -> dict[str, Any]:
    """Build lightweight editor groups without downloading R2 manifests on refresh."""
    images = []
    for source in raw_summary.get("sources", []) if isinstance(raw_summary, dict) else []:
        if not isinstance(source, dict):
            continue
        role = str(source.get("role") or "coverage_image")
        images.append(
            {
                "source_id": source.get("source_id") or source.get("camera_group") or "unassigned",
                "camera_group": source.get("camera_group") or "default",
                "role": "hero" if role == "hero_image" else role,
                "location": source.get("location"),
                "image_name": source.get("relative_path") or source.get("source_id"),
            }
        )
    return {"schema_version": 1, "images": images}


def matching_source_group_cards(
    raw_summary: dict[str, Any],
    preprocess_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = build_source_groups(source_manifest_from_raw_summary(raw_summary))
    hero_counts: Counter[str] = Counter(
        str(source.get("location") or "unassigned")
        for source in raw_summary.get("sources", [])
        if isinstance(source, dict) and source.get("role") == "hero_image"
    )
    reports = latest_group_capture_reports(preprocess_runs, raw_summary)
    for group in groups:
        locations = [str(location) for location in group.get("locations") or []]
        location = locations[0] if locations else "unassigned"
        report = reports.get(f"location:{location}") or {}
        videos = report.get("videos") if isinstance(report, dict) else []
        source_id = str(group.get("source_ids", [""])[0])
        selected_count = next(
            (
                int(video.get("selected_frame_count") or 0)
                for video in videos
                if isinstance(video, dict) and str(video.get("source_id") or "") == source_id
            ),
            0,
        )
        group["image_count"] = selected_count or int(group.get("image_count") or 0)
        group["hero_count"] = hero_counts.get(location, 0)
        group["locality"] = location
    return groups


def colmap_matching_plan_uri(project: dict[str, Any]) -> str:
    project_id = str(project.get("id") or "project")
    current_uri = str(project.get("colmap_current_uri") or "").rstrip("/")
    if current_uri.endswith("/current"):
        base_uri = current_uri[: -len("/current")].rstrip("/")
    else:
        base_uri = f"r2://{default_r2_bucket()}/projects/{project_id}/colmap"
    return f"{base_uri}/review/matching_plan.json"


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
            "use_scale_regularization": scale_regularization_form_value(input_json) or "true",
            "provider": latest_run.get("provider") if latest_run else default_training_provider(),
            "image": latest_run.get("image") if latest_run else "",
            "repo_url": input_json.get("repo_url") or "",
            "git_ref": input_json.get("git_ref") or "",
            "gpu_type_id": selected_gpu,
            "container_disk_gb": input_json.get("container_disk_gb") or runpod_training_container_disk_gb(),
            "export": input_json.get("export", True),
        },
        "training_gpu_options": TRAINING_GPU_OPTIONS,
        "training_summary_rows": training_summary_rows(latest_run),
    }


def raw_source_summary(project: dict[str, Any]) -> dict[str, Any]:
    raw_uri = project.get("raw_uri") if project else None
    if not raw_uri:
        return {"loaded": False, "source_count": 0, "rows": [], "sources": [], "location_blocks": [], "has_coverage_video": None}
    try:
        manifest = load_json_uri(f"{raw_uri.rstrip('/')}/sources_manifest.json")
    except Exception as exc:
        return {
            "loaded": False,
            "source_count": 0,
            "rows": [],
            "sources": [],
            "location_blocks": [],
            "has_coverage_video": None,
            "error": friendly_manifest_error(exc),
        }
    sources = manifest.get("sources") if isinstance(manifest, dict) else []
    if not isinstance(sources, list):
        sources = []
    stale_reasons = manifest.get("preprocess_stale_reasons") if isinstance(manifest.get("preprocess_stale_reasons"), dict) else {}
    stale_groups = {str(group_key) for group_key in stale_reasons}
    compact_sources = compact_raw_sources(sources, stale_groups)
    return {
        "loaded": True,
        "source_count": len(sources),
        "rows": raw_source_count_rows(sources),
        "sources": compact_sources,
        "group_rows": raw_source_group_rows(sources),
        "location_blocks": raw_source_location_blocks(compact_sources),
        "has_coverage_video": any(
            isinstance(source, dict) and source.get("role") == "coverage_video"
            for source in sources
        ),
    }


def raw_source_location_blocks(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for source in sources:
        location = str(source.get("location") or "unassigned")
        block = grouped.setdefault(
            location,
            {
                "location": location,
                "coverage_videos": [],
                "coverage_images": [],
                "hero_images": [],
                "sources": [],
                "stale": False,
                "conflict": False,
            },
        )
        block["sources"].append(source)
        block["stale"] = block["stale"] or bool(source.get("preprocess_stale"))
        role = source.get("role")
        if role == "coverage_video":
            block["coverage_videos"].append(source)
        elif role == "coverage_image":
            block["coverage_images"].append(source)
        elif role == "hero_image":
            block["hero_images"].append(source)

    blocks = []
    for location, block in sorted(grouped.items()):
        has_video = bool(block["coverage_videos"])
        has_images = bool(block["coverage_images"])
        block["conflict"] = (has_video and has_images) or len(block["coverage_videos"]) > 1
        if has_video:
            block["coverage_kind"] = "video"
            block["coverage_count"] = len(block["coverage_videos"])
            block["coverage_group_key"] = f"location:{location}"
        elif has_images:
            block["coverage_kind"] = "images"
            block["coverage_count"] = len(block["coverage_images"])
            block["coverage_group_key"] = f"location:{location}"
        else:
            block["coverage_kind"] = "none"
            block["coverage_count"] = 0
            block["coverage_group_key"] = f"location:{location}"
        block["hero_count"] = len(block["hero_images"])
        blocks.append(block)
    return blocks


def friendly_manifest_error(exc: Exception) -> str:
    message = str(exc)
    if "NoSuchKey" in message or "404" in message or "exit status 1" in message:
        return "No sources manifest found yet. Upload raw files to create one."
    return "Sources manifest could not be loaded."


def raw_source_group_rows(sources: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        role = str(source.get("role") or "coverage_image")
        location = str(source.get("location") or "unassigned")
        key = f"{role}:{location}"
        row = grouped.setdefault(key, {"group_key": key, "role": role, "location": location, "count": 0, "video_count": 0})
        row["count"] += 1
        if role == "coverage_video":
            row["video_count"] += 1
    return [grouped[key] for key in sorted(grouped)]


def raw_source_count_rows(sources: list[Any]) -> list[dict[str, Any]]:
    counts = Counter(
        (
            str(source.get("role") or "unknown"),
            str(source.get("location") or "unassigned"),
        )
        for source in sources
        if isinstance(source, dict)
    )
    return [
        {"role": role, "location": location, "count": count}
        for (role, location), count in sorted(counts.items())
    ]


def compact_raw_sources(sources: list[Any], stale_groups: set[str]) -> list[dict[str, Any]]:
    rows = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        width = source.get("width")
        height = source.get("height")
        resolution = f"{width}x{height}" if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0 else ""
        rows.append(
            {
                "source_id": source.get("source_id"),
                "relative_path": source.get("relative_path"),
                "role": source.get("role"),
                "camera_group": source.get("camera_group"),
                "location": source.get("location"),
                "colmap_policy": source.get("colmap_policy"),
                "resolution": resolution,
                "duration_seconds": source.get("duration_seconds"),
                "loaded_at": source.get("loaded_at"),
                "preprocess_stale": source_group_key(source) in stale_groups,
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


def validate_choice(value: str, allowed: set[str], name: str) -> None:
    if value not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported {name}: {value}")


def normalize_colmap_feature_extractor(value: Any) -> str:
    text = str(value or "SIFT").strip()
    if text.lower() == "sift":
        return "SIFT"
    return text


def scale_regularization_form_value(input_json: dict[str, Any]) -> str:
    options = input_json.get("splatfacto_options")
    if not isinstance(options, dict) or "use_scale_regularization" not in options:
        return ""
    return "true" if bool(options.get("use_scale_regularization")) else "false"


def training_summary_rows(run: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not run:
        return []
    summary = run.get("summary_json") if isinstance(run.get("summary_json"), dict) else {}
    inputs = run.get("input_uri_json") if isinstance(run.get("input_uri_json"), dict) else {}
    diagnostics = summary.get("training_diagnostics") if isinstance(summary.get("training_diagnostics"), dict) else {}
    rows: list[dict[str, Any]] = []

    def add(label: str, value: Any) -> None:
        if value not in (None, "", [], {}):
            rows.append({"label": label, "value": value})

    add("Status", summary.get("status") or run.get("status"))
    add("Provider", summary.get("provider") or run.get("provider"))
    add("Run ID", run.get("id"))
    add("Attempt", run.get("attempt"))
    add("Provider job ID", summary.get("provider_job_id"))
    add("Provider pod ID", summary.get("provider_pod_id"))
    add("GPU", first_gpu_type(inputs.get("gpu_type_ids")))
    add("Container disk GB", inputs.get("container_disk_gb"))
    add("Method", summary.get("method") or inputs.get("method"))
    add("Max steps", inputs.get("max_steps"))
    add("Save every", inputs.get("save_every"))
    add("Eval every", inputs.get("eval_every"))
    add("Scale regularization", scale_regularization_form_value(inputs) or "enabled")
    add("Image", run.get("image"))
    add("Output URI", run.get("output_uri"))
    add("Checkpoint count", summary.get("checkpoint_count") or diagnostics.get("checkpoint_count"))
    add("Latest checkpoint", summary.get("latest_checkpoint"))
    add("Exported PLY", summary.get("exported_ply"))
    add("Exported PLY vertices", diagnostics.get("exported_ply_vertices") or summary.get("exported_ply_vertices"))
    add("COLMAP init points", diagnostics.get("colmap_init_point_count") or summary.get("colmap_init_point_count"))
    add("COLMAP init XYZ min", diagnostics.get("colmap_init_xyz_min"))
    add("COLMAP init XYZ max", diagnostics.get("colmap_init_xyz_max"))
    add("COLMAP init error median", diagnostics.get("colmap_init_error_median"))
    add("Oversized Gaussian", diagnostics.get("oversized_gaussian_detected"))
    add("Oversized Gaussian count", diagnostics.get("oversized_gaussian_count"))
    add("Oversized Gaussian max scene ratio", diagnostics.get("oversized_gaussian_ratio_max"))
    add("Gaussian scale p95", diagnostics.get("gaussian_scale_p95"))
    add("Gaussian scale p99", diagnostics.get("gaussian_scale_p99"))
    add("Gaussian scale max", diagnostics.get("gaussian_scale_max"))
    add("Gaussian anisotropy p99", diagnostics.get("gaussian_anisotropy_p99"))
    add("Gaussian anisotropy max", diagnostics.get("gaussian_anisotropy_max"))
    return rows


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
            "registered_by_location",
            "registered_by_group",
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
    if preprocess_run_is_active(run):
        return {}
    output_uri = run.get("output_uri")
    if not output_uri:
        return {}
    try:
        return load_json_uri(f"{output_uri.rstrip('/')}/capture_report.json")
    except Exception:
        return {}


def preprocess_run_is_active(run: dict[str, Any]) -> bool:
    return str(run.get("status") or "") in {
        "preprocess_queued",
        "preprocess_running",
    }


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


def colmap_blacklist_uri(project: dict[str, Any]) -> str:
    project_id = str(project.get("id") or "project")
    current_uri = str(project.get("colmap_current_uri") or "").rstrip("/")
    if current_uri.endswith("/current"):
        base_uri = current_uri[: -len("/current")].rstrip("/")
    else:
        base_uri = f"r2://{default_r2_bucket()}/projects/{project_id}/colmap"
    return f"{base_uri}/review/colmap_blacklist.json"


def load_colmap_blacklist(project: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = load_json_uri(colmap_blacklist_uri(project))
    except Exception:
        payload = {}
    entries = payload.get("excluded_images") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        entries = []
    return {
        "schema_version": int(payload.get("schema_version") or 1) if isinstance(payload, dict) else 1,
        "project_id": str(project.get("id") or ""),
        "excluded_images": [entry for entry in entries if isinstance(entry, dict)],
    }


def preprocess_settings(run: Optional[dict[str, Any]], capture_report: dict[str, Any]) -> dict[str, Any]:
    if not run:
        return {}
    input_json = run.get("input_uri_json") if isinstance(run.get("input_uri_json"), dict) else {}
    profile = str(input_json.get("profile") or DEFAULT_PREPROCESS_PROFILE)
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
    output_uri = input_json.get("output_uri") or preprocess_current_uri.rsplit("/current", 1)[0] or f"r2://{default_r2_bucket()}/projects/{project.get('id')}/preprocess"
    return {
        "raw_uri": input_json.get("raw_uri") or project.get("raw_uri") or "",
        "output_uri": preprocess_output_base_uri(output_uri, str(project.get("id") or "")),
        "endpoint_url": input_json.get("endpoint_url") or "",
        "profile": settings.get("profile") or input_json.get("profile") or DEFAULT_PREPROCESS_PROFILE,
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


def preprocess_run_rows(stage_runs: list[dict[str, Any]], raw_summary: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    current_group_keys = set(required_preprocess_group_keys(raw_summary or {}))
    rows = []
    for run in stage_runs:
        run_copy = dict(run)
        run_copy["preprocess_location"] = preprocess_run_location_label(run_copy)
        group_keys = set(preprocess_run_group_keys(run_copy))
        if current_group_keys and group_keys and group_keys.isdisjoint(current_group_keys):
            run_copy["is_latest_stage_run"] = False
            run_copy["status_label"] = "history run"
            run_copy["status_css"] = "history"
        run_copy.update(preprocess_run_metrics(run_copy))
        rows.append(run_copy)
    return rows


def preprocess_run_group_keys(run: dict[str, Any]) -> list[str]:
    input_json = run.get("input_uri_json") if isinstance(run.get("input_uri_json"), dict) else {}
    group_configs = input_json.get("group_configs") if isinstance(input_json.get("group_configs"), list) else []
    keys = []
    for item in group_configs:
        if isinstance(item, dict) and item.get("group_key"):
            keys.append(str(item["group_key"]))
    return keys


def preprocess_run_location_label(run: dict[str, Any]) -> str:
    keys = preprocess_run_group_keys(run)
    if not keys:
        return "project"
    locations = []
    for key in keys:
        location = key.split(":", 1)[1] if ":" in key else key
        if location not in locations:
            locations.append(location)
    return ", ".join(locations)


def preprocess_run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("summary_json") if isinstance(run.get("summary_json"), dict) else {}
    videos = summary.get("videos") if isinstance(summary.get("videos"), list) else []
    rows = compact_video_rows(videos)
    if not rows:
        return {"selected_total": "", "fallback_total": "", "gap_total": "", "warning_summary": ""}
    warning_count = sum(len(row.get("warnings", [])) for row in rows)
    selected_by = summary.get("selected_by") if isinstance(summary.get("selected_by"), dict) else {}
    force_keep_total = int(selected_by.get("force_keep") or 0)
    return {
        "selected_total": sum(int(row.get("selected_frame_count") or 0) for row in rows),
        "fallback_total": force_keep_total,
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


def preprocess_location_blocks(
    raw_summary: dict[str, Any],
    capture_report: dict[str, Any],
    group_reports: Optional[dict[str, dict[str, Any]]] = None,
    group_runs: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    blocks = []
    for raw_block in raw_summary.get("location_blocks", []):
        if not isinstance(raw_block, dict):
            continue
        location = str(raw_block.get("location") or "unassigned")
        coverage_group_key = str(raw_block.get("coverage_group_key") or f"coverage_image:{location}")
        active_run = (group_runs or {}).get(coverage_group_key)
        hero_group_key = f"hero_image:{location}"
        if active_run is None:
            active_run = (group_runs or {}).get(hero_group_key)
        review_report = (group_reports or {}).get(coverage_group_key)
        if review_report is None and active_run is None:
            review_report = capture_report
        if review_report is None:
            review_report = {}
        summary = group_summary(review_report, coverage_group_key, location)
        field_values = preprocess_parameter_values(active_run)
        block = {
            **raw_block,
            "group_key": coverage_group_key,
            "hero_group_key": hero_group_key,
            "latest_run": active_run,
            "last_parameter_rows": preprocess_parameter_rows(active_run),
            "form_values": field_values,
            "timeline_blocks": [
                item for item in full_timeline_blocks(review_report)
                if item.get("group_key") == coverage_group_key or item.get("location") == location
            ],
            "coverage_image_grid": coverage_image_grid(review_report, group_key=coverage_group_key, location=location),
            "hero_image_grid": hero_image_grid(review_report, location=location),
            "quality_rows": quality_distribution_rows(summary),
            "result_summary": compact_group_result_summary(review_report, coverage_group_key, location),
        }
        blocks.append(block)
    return blocks


def preprocess_parameter_values(run: Optional[dict[str, Any]]) -> dict[str, Any]:
    values = {
        key: value
        for key, value in PROFILE_DEFAULTS.get(DEFAULT_PREPROCESS_PROFILE, {}).items()
        if key in ui_preprocess_field_names()
    }
    if run:
        input_json = run.get("input_uri_json") if isinstance(run.get("input_uri_json"), dict) else {}
        group_configs = input_json.get("group_configs") if isinstance(input_json.get("group_configs"), list) else []
        if group_configs and isinstance(group_configs[0], dict):
            values.update(parse_preprocess_args(group_configs[0].get("preprocess_args")))
        else:
            values.update(parse_preprocess_args(input_json.get("preprocess_args")))
    return values


def preprocess_parameter_rows(run: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    values = preprocess_parameter_values(run)
    fields = [
        ("Candidate FPS", "candidate_fps"),
        ("Target min", "target_min"),
        ("Target max", "target_max"),
        ("Min blur", "min_blur"),
        ("Brightness min", "min_brightness"),
        ("Brightness max", "max_brightness"),
        ("Min contrast", "min_contrast"),
        ("Min entropy", "min_entropy"),
        ("Force keep interval", "force_keep_interval"),
    ]
    return [
        {"label": label, "value": values.get(key)}
        for label, key in fields
        if values.get(key) not in (None, "", [], {})
    ]


def ui_preprocess_field_names() -> tuple[str, ...]:
    return (
        "candidate_fps",
        "target_min",
        "target_max",
        "min_blur",
        "min_brightness",
        "max_brightness",
        "min_contrast",
        "min_entropy",
        "force_keep_interval",
    )


def group_summary(capture_report: dict[str, Any], group_key: str, location: str) -> dict[str, Any]:
    frames = group_frames(capture_report, group_key, location)
    if not frames:
        return {}
    return {
        "metric_distributions": {
            metric: metric_distribution_from_records(frames, metric)
            for metric in ("blur_score", "brightness", "contrast", "entropy")
        }
    }


def compact_group_result_summary(capture_report: dict[str, Any], group_key: str, location: str) -> dict[str, Any]:
    if not capture_report:
        return {
            "selected": "",
            "force_keep": "",
            "gaps": "",
            "warnings": [],
            "coverage_selected": "",
            "hero_selected": "",
            "coverage_rejected": "",
            "hero_count": "",
        }
    frames = group_frames(capture_report, group_key, location)
    videos = [
        video for video in capture_report.get("videos", []) if isinstance(video, dict)
        and (video.get("group_key") == group_key or video.get("location") == location)
    ] if isinstance(capture_report, dict) else []
    selected = sum(1 for frame in frames if image_decision(frame) == "selected")
    force_keep = sum(1 for frame in frames if frame.get("selected_by") == "force_keep" or frame.get("decision") == "force_keep")
    gaps = 0
    warnings: list[str] = []
    for video in videos:
        coverage = video.get("coverage") if isinstance(video.get("coverage"), dict) else {}
        gaps += int(coverage.get("windows_below_minimum_count") or 0)
        warnings.extend(str(item) for item in video.get("warnings", []) if item)
    coverage_images = coverage_image_grid(capture_report, group_key=group_key, location=location)
    hero_images = hero_image_grid(capture_report, location=location)
    coverage_selected = sum(1 for frame in frames if image_decision(frame) == "selected")
    hero_selected = hero_images["counts"].get("selected", 0)
    return {
        "selected": selected,
        "force_keep": force_keep,
        "gaps": gaps,
        "warnings": sorted(set(warnings)),
        "coverage_selected": coverage_selected,
        "hero_selected": hero_selected,
        "coverage_rejected": coverage_images["counts"].get("rejected", 0),
        "hero_count": len(hero_images["items"]),
    }


def metric_distribution_from_records(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [record.get(metric) for record in records if isinstance(record.get(metric), (int, float))]
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)

    return {
        "min": round(ordered[0], 3),
        "p10": percentile(0.10),
        "p50": percentile(0.50),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p90": percentile(0.90),
        "max": round(ordered[-1], 3),
    }


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
    if isinstance(capture_report, dict):
        # A capture report is authoritative. Coverage stills belong to the image
        # grid and must never be promoted into a video timeline fallback.
        return full_timeline_blocks(capture_report)
    if run:
        summary = run.get("summary_json") or {}
        if isinstance(summary, dict) and isinstance(summary.get("videos"), list) and summary.get("videos"):
            return compact_selected_timeline_blocks(summary)
    return []


def coverage_image_grid(capture_report: dict[str, Any], group_key: Optional[str] = None, location: Optional[str] = None) -> dict[str, Any]:
    if not isinstance(capture_report, dict):
        return {"items": [], "counts": {}}
    image_frames = [
        frame for frame in group_frames(capture_report, group_key, location)
        if str(frame.get("source_id") or "") == "coverage_images"
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


def hero_image_grid(capture_report: dict[str, Any], location: Optional[str] = None) -> dict[str, Any]:
    if not isinstance(capture_report, dict):
        return {"items": [], "counts": {}}
    records = capture_report.get("hero_images", [])
    if not isinstance(records, list):
        return {"items": [], "counts": {}}
    items = []
    counts = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        if location and str(record.get("location") or "unassigned") != location:
            continue
        decision = "selected" if record.get("output_file") else "rejected"
        counts[decision] += 1
        items.append({"decision": decision, "title": image_hover_title(record, decision)})
    return {"items": items, "counts": dict(counts)}


def group_frames(capture_report: dict[str, Any], group_key: Optional[str], location: Optional[str]) -> list[dict[str, Any]]:
    frames = capture_report.get("frames", []) if isinstance(capture_report, dict) else []
    if not isinstance(frames, list):
        return []
    result = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        if not group_key and not location:
            result.append(frame)
            continue
        if group_key and frame.get("group_key") == group_key:
            result.append(frame)
            continue
        if location and str(frame.get("location") or "unassigned") == location:
            result.append(frame)
    return result


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
        force_keep_count = selected_by.get("force_keep", 0)
        blocks.append(
            {
                "source_id": source_id,
                "group_key": video.get("group_key"),
                "location": video.get("location"),
                "meta": {
                    "selected": video.get("selected_frame_count"),
                    "quality": selected_by.get("quality", 0),
                    "force_keep": force_keep_count,
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
                "meta": {"selected": len(frames), "quality": None, "force_keep": None, "coverage_gaps": None, "largest_gap_seconds": None},
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
    selected_by = str(frame.get("selected_by") or "")
    if selected_by in {"force_keep", "coverage_fallback"}:
        decision = selected_by
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
    if decision not in {"selected", "coverage_fallback", "force_keep"}:
        parts.append(str(frame.get("reject_reason") or decision))
    return " | ".join(parts)


def image_hover_title(frame: dict[str, Any], decision: str) -> str:
    filename = frame.get("source_image") or frame.get("output_file") or frame.get("frame_index") or "image"
    blur_score = frame.get("blur_score")
    parts = [Path(str(filename)).name, f"blur {blur_score}"]
    if decision not in {"selected", "coverage_fallback", "force_keep"}:
        parts.append(str(frame.get("reject_reason") or decision))
    return " | ".join(parts)


def image_decision(frame: dict[str, Any]) -> str:
    decision = str(frame.get("decision") or frame.get("selected_by") or "selected")
    if decision in {"quality", "selected", "coverage_fallback", "force_keep"}:
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
